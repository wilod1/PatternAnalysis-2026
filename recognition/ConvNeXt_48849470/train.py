"""
Training, validation, and (optionally) testing for the ADNI classifiers.
"""
import argparse
import copy
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg") # no display on the cluster
import matplotlib.pyplot as plt
import torch
from torch import nn

from dataset import SEED, get_dataloaders
from modules import ConvNeXt, SimpleCNN, SimpleResNet

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEPTHS = (2, 2, 4, 2)
DEFAULT_DIMS = (32, 64, 128, 256)


def set_seed(seed):
    """
    Seed torch (CPU and CUDA) for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name, args):
    """
    Return a fresh model by name, configured from the command-line arguments.
    """
    builders = {
        "cnn": lambda: SimpleCNN(),
        "resnet": lambda: SimpleResNet(),
        "convnext": lambda: ConvNeXt(
            drop_path_rate=args.drop_path,
            depths=tuple(getattr(args, "depths", DEFAULT_DEPTHS)),
            dims=tuple(getattr(args, "dims", DEFAULT_DIMS))),
    }
    if name not in builders:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(builders)}")
    return builders[name]()


def args_from_checkpoint(ckpt):
    """
    Rebuild the training arguments stored in a checkpoint, so a model can be
    reconstructed with the same architecture. Options missing from older
    checkpoints keep their defaults.
    """
    args = parse_args(["--model", ckpt["model"]])
    for key, value in ckpt["args"].items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def mixup_batch(x, y, alpha):
    """
    Mixup (Zhang et al., 2018): blend each image with another image from the
    batch using a weight drawn from Beta(alpha, alpha). Returns the mixed
    batch, both label sets and the weight on the first. alpha <= 0 is a no-op.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


class ModelEMA:
    """
    Exponential moving average of a model's weights. The averaged copy
    (self.module) is used for evaluation: it is smoother than the raw weights,
    which steadies validation accuracy between epochs. The decay ramps up over
    the first updates so the average does not start from the random init.
    """

    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for e, m in zip(self.module.parameters(), model.parameters()):
            e.mul_(d).add_(m.detach(), alpha=1 - d)
        for e, m in zip(self.module.buffers(), model.buffers()):
            e.copy_(m)


def build_scheduler(optimizer, schedule, epochs, steps_per_epoch, warmup_epochs):
    """
    Per-iteration learning-rate schedule. 'constant' returns None. 'cosine'
    warms up linearly for warmup_epochs, then decays to zero along a cosine.
    """
    if schedule == "constant":
        return None
    total = epochs * steps_per_epoch
    warm = warmup_epochs * steps_per_epoch

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None,
                    mixup_alpha=0.0, ema=None):
    """
    One pass over the training loader. Returns (mean loss, per-slice accuracy).
    The scheduler, if given, is stepped after every optimiser step. With
    mixup_alpha > 0 the batch is mixed and the accuracy counts a prediction as
    correct in proportion to the weight of each label. If ema is given, it is
    updated after every optimiser step.
    """
    model.train()
    total_loss, correct, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if lam >= 1.0:
            loss = criterion(logits, y_a)
        else:
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        pred = logits.argmax(1)
        total_loss += loss.item() * y.size(0)
        correct += (lam * (pred == y_a).sum().item()
                    + (1 - lam) * (pred == y_b).sum().item())
        n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate on a loader. Returns (mean loss, per-slice accuracy).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def plot_curves(history, path):
    """
    Save loss and accuracy curves (train vs validation) to one PNG.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="validation")
    ax2.set(xlabel="epoch", ylabel="per-slice accuracy", title="Accuracy")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fit(model, loaders, args, run, device, out_dir=HERE):
    """
    Train model for args.epochs, keeping the checkpoint with the best validation
    accuracy. Writes <out_dir>/results/<run>_history.json after every epoch,
    <out_dir>/assets/<run>_curves.png and <out_dir>/checkpoints/<run>_best.pt
    (*.pt is git-ignored). Training uses optional label smoothing and mixup;
    validation and test loss are always plain cross-entropy so runs stay
    comparable. With args.ema_decay > 0, validation, checkpoints and test use
    the exponential moving average of the weights. Returns the summary
    dictionary that is written to the history file.
    """
    assets = os.path.join(out_dir, "assets")
    results = os.path.join(out_dir, "results")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    for d in (assets, results, ckpt_dir):
        os.makedirs(d, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"run={run} device={device} params={n_params:,} "
          f"train/val slices={len(loaders['train'].dataset)}/"
          f"{len(loaders['val'].dataset)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.schedule, args.epochs,
                                len(loaders["train"]), args.warmup_epochs)
    mixup_alpha = getattr(args, "mixup", 0.0)
    ema_decay = getattr(args, "ema_decay", 0.0)
    ema = ModelEMA(model, ema_decay) if ema_decay > 0 else None
    eval_model = ema.module if ema is not None else model

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc",
                               "lr", "epoch_time")}
    best_acc, ckpt_path = -1.0, os.path.join(ckpt_dir, f"{run}_best.pt")
    history_path = os.path.join(results, f"{run}_history.json")
    summary = {"run": run, "args": vars(args), "params": n_params,
               "best_val_acc": best_acc, "history": history}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer,
                                          criterion, device, scheduler,
                                          mixup_alpha=mixup_alpha, ema=ema)
        va_loss, va_acc = evaluate(eval_model, loaders["val"], eval_criterion,
                                   device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time"].append(time.time() - t0)
        if va_acc > best_acc: # keep the checkpoint with best val accuracy
            best_acc = va_acc
            torch.save({"model": args.model, "state_dict": eval_model.state_dict(),
                        "args": vars(args), "epoch": epoch}, ckpt_path)
        print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"{history['epoch_time'][-1]:.0f}s")

        summary["best_val_acc"] = best_acc
        with open(history_path, "w") as f:
            json.dump(summary, f, indent=2)
        plot_curves(history, os.path.join(assets, f"{run}_curves.png"))

    if args.eval_test:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        te_loss, te_acc = evaluate(model, loaders["test"], eval_criterion, device)
        summary.update(test_loss=te_loss, test_acc=te_acc)
        print(f"TEST (best checkpoint): loss {te_loss:.4f} acc {te_acc:.4f}")
    with open(history_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="cnn")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subset", type=int, default=None,
                   help="slices per split, for quick smoke tests only")
    p.add_argument("--run-name", default=None)
    p.add_argument("--drop-path", type=float, default=0.1,
                   help="ConvNeXt stochastic-depth rate")
    p.add_argument("--depths", type=int, nargs=4, default=list(DEFAULT_DEPTHS),
                   help="ConvNeXt blocks per stage")
    p.add_argument("--dims", type=int, nargs=4, default=list(DEFAULT_DIMS),
                   help="ConvNeXt channels per stage")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--aug", choices=["mild", "strong"], default="mild",
                   help="training augmentation preset")
    p.add_argument("--mixup", type=float, default=0.0,
                   help="mixup Beta(alpha, alpha); 0 turns it off")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="EMA of weights used for evaluation; 0 turns it off")
    p.add_argument("--schedule", choices=["constant", "cosine"],
                   default="constant")
    p.add_argument("--warmup-epochs", type=int, default=2,
                   help="linear warmup length for the cosine schedule")
    p.add_argument("--eval-test", action="store_true",
                   help="evaluate the best checkpoint on the test split "
                        "(leave off while tuning)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    run = args.run_name or args.model
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dataloaders(batch_size=args.batch_size, img_size=args.img_size,
                              seed=args.seed, num_workers=args.num_workers,
                              subset=args.subset, aug=args.aug)
    model = build_model(args.model, args).to(device)
    fit(model, loaders, args, run, device)


if __name__ == "__main__":
    main()