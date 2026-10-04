"""
Training, validation, and (optionally) testing for the ADNI classifiers.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg") # no display on the cluster
import matplotlib.pyplot as plt
import torch
from torch import nn

from dataset import SEED, get_dataloaders
from modules import SimpleCNN

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
RESULTS = os.path.join(HERE, "results")
CKPT_DIR = os.path.join(HERE, "checkpoints") # *.pt is git-ignored


def set_seed(seed):
    """
    Seed torch (CPU and CUDA) for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name):
    """
    Return a fresh model by name.
    """
    models = {"cnn": SimpleCNN}
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(models)}")
    return models[name]()


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    One pass over the training loader. Returns (mean loss, per-slice accuracy).
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
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


def parse_args():
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
    p.add_argument("--eval-test", action="store_true",
                   help="evaluate the best checkpoint on the test split "
                        "(leave off while tuning)")
    return p.parse_args()


def main():
    args = parse_args()
    run = args.run_name or args.model
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in (ASSETS, RESULTS, CKPT_DIR):
        os.makedirs(d, exist_ok=True)

    loaders = get_dataloaders(batch_size=args.batch_size, img_size=args.img_size,
                              seed=args.seed, num_workers=args.num_workers,
                              subset=args.subset)

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"run={run} device={device} params={n_params:,} "
          f"train/val slices={len(loaders['train'].dataset)}/"
          f"{len(loaders['val'].dataset)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_acc, ckpt_path = -1.0, os.path.join(CKPT_DIR, f"{run}_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer,
                                          criterion, device)
        va_loss, va_acc = evaluate(model, loaders["val"], criterion, device)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)
        if va_acc > best_acc: # keep the checkpoint with best val accuracy
            best_acc = va_acc
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "args": vars(args), "epoch": epoch}, ckpt_path)
        print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.0f}s")

    plot_curves(history, os.path.join(ASSETS, f"{run}_curves.png"))
    summary = {"run": run, "args": vars(args), "params": n_params,
               "best_val_acc": best_acc, "history": history}
    if args.eval_test:
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        te_loss, te_acc = evaluate(model, loaders["test"], criterion, device)
        summary.update(test_loss=te_loss, test_acc=te_acc)
        print(f"TEST (best checkpoint): loss {te_loss:.4f} acc {te_acc:.4f}")
    with open(os.path.join(RESULTS, f"{run}_history.json"), "w") as f:
        json.dump(summary, f, indent = 2)


if __name__ == "__main__":
    main()