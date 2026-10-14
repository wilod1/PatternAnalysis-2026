"""
Train and evaluate a ConvNeXt model for ADNI MRI slice classification.

Classes:
    NC -> 0
    AD -> 1

The training and validation sets are used for model development.
The test set is not evaluated unless --eval-test is explicitly supplied.

Training choices aimed at a small, subject-limited dataset:
    - a size preset (--variant) instead of a fixed 28M-parameter model
    - exponential moving average (EMA) of the weights, used for validation
    - no weight decay on norm scales, biases or layer-scale parameters
    - a plain (unsmoothed) cross-entropy for validation and test loss
"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import RandomSampler
import torch.nn as nn

from dataset import AUG_CHOICES, get_dataloaders
from modules import CONVNEXT_VARIANTS, ConvNeXt

# Named bundles of defaults. Any flag given on the command line still wins.
RECIPES = {
    "default": {},
    # Resize-and-crop pipeline with a plain AdamW schedule: no EMA, no
    # gradient clipping, no dropout, weight decay on every parameter, no
    # early stopping, and the learning rate stepped once per epoch.
    "resizecrop": {
        "aug": "stretch_crop",
        "img_size": 224,
        "batch_size": 64,
        "epochs": 200,
        "patience": 1_000_000,
        "lr": 1e-3,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "warmup_epochs": 5.0,
        "label_smoothing": 0.05,
        "drop_path": 0.1,
        "dropout": 0.0,
        "max_grad_norm": 0.0,
        "ema_decay": 0.0,
        "decay_all": True,
        "lr_unit": "epoch",
    },
}


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed, deterministic=False):
    """Seed all generators. cuDNN autotuning is on unless deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# ---------------------------------------------------------------------
# Weight averaging
# ---------------------------------------------------------------------

class ModelEMA:
    """
    Exponential moving average of a model's weights.

    The decay ramps up from a small value so the average is useful early in
    training. The averaged copy is what gets validated and checkpointed.
    """

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()

        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))

        source = model.state_dict().values()
        target = self.module.state_dict().values()

        for ema_value, value in zip(target, source):
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)


# ---------------------------------------------------------------------
# Optimiser and learning-rate schedule
# ---------------------------------------------------------------------

def build_optimizer(model, lr, weight_decay, decay_all=False):
    """
    AdamW. By default weight decay skips norm scales, biases and
    layer-scale vectors; decay_all applies it to every parameter.
    """
    if decay_all:
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

    decay, no_decay = [], []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        # Norm scales, biases and layer-scale vectors are one-dimensional.
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def learning_rate_at_step(step, total_steps, warmup_steps, base_lr, min_lr):
    """
    Linear warm-up followed by cosine decay.

    The first training step starts at 10% of the base learning rate.
    """
    if warmup_steps > 0 and step < warmup_steps:
        warmup_fraction = step / warmup_steps
        return base_lr * (0.1 + 0.9 * warmup_fraction)

    decay_steps = max(total_steps - warmup_steps, 1)
    decay_step = max(step - warmup_steps, 0)
    progress = min(decay_step / decay_steps, 1.0)

    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))

    return min_lr + (base_lr - min_lr) * cosine


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def count_correct(logits, labels):
    """Number of correctly classified samples in a batch."""
    return (logits.argmax(dim=1) == labels).sum().item()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp_enabled,
    args,
    total_steps,
    warmup_steps,
    global_step,
    ema=None,
    fixed_lr=None,
):
    """
    One pass over the training loader. Returns (metrics, step, last lr).

    With fixed_lr set, that learning rate is used for the whole epoch;
    otherwise the schedule is evaluated at every optimiser step.
    """
    model.train()

    loss_sum = 0.0
    correct = 0
    samples = 0
    lr = args.lr

    for images, labels in loader:
        if fixed_lr is not None:
            lr = fixed_lr
        else:
            lr = learning_rate_at_step(
                step=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                base_lr=args.lr,
                min_lr=args.min_lr,
            )

        for group in optimizer.param_groups:
            group["lr"] = lr

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixup: blend each image with a shuffled partner from the same batch.
        # One mixing weight per batch; the loss blends both labels. The label
        # with the larger weight is used for the accuracy count only.
        mix_labels = None
        mix_lambda = 1.0
        if getattr(args, "mixup_alpha", 0.0) > 0:
            mix_lambda = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
            permutation = torch.randperm(images.size(0), device=images.device)
            images = mix_lambda * images + (1.0 - mix_lambda) * images[permutation]
            mix_labels = labels[permutation]

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            if mix_labels is None:
                loss = criterion(logits, labels)
            else:
                loss = (
                    mix_lambda * criterion(logits, labels)
                    + (1.0 - mix_lambda) * criterion(logits, mix_labels)
                )

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )

            optimizer.step()

        if ema is not None:
            ema.update(model)

        batch_size = labels.size(0)
        loss_sum += loss.detach().item() * batch_size
        dominant = labels if mix_labels is None or mix_lambda >= 0.5 else mix_labels
        correct += count_correct(logits.detach(), dominant)
        samples += batch_size
        global_step += 1

    metrics = {
        "loss": loss_sum / samples,
        "accuracy": correct / samples,
        "samples": samples,
    }
    return metrics, global_step, lr


SELECT_METRICS = {
    "slice_acc": "accuracy",
    "patient_acc": "patient_accuracy",
    "patient_bal_acc": "patient_balanced_accuracy",
    "patient_auc": "patient_auc",
}


def rank_auc(scores, labels):
    """Area under the ROC curve; ties count half. Returns 0.5 if one class."""
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    greater = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(greater + 0.5 * ties)


def patient_metrics(probs, labels, subject_ids):
    """
    Patient-level metrics from slice probabilities.

    A patient's score is the mean slice probability over all of their slices
    (every scan); the patient is called AD when the mean P(AD) exceeds 0.5,
    which is the same as argmax of the averaged probabilities.
    """
    _, inverse = np.unique(subject_ids, return_inverse=True)
    n_patients = inverse.max() + 1

    counts = np.bincount(inverse, minlength=n_patients).astype(float)
    mean_p_ad = np.bincount(
        inverse, weights=probs[:, 1], minlength=n_patients
    ) / counts
    patient_labels = np.zeros(n_patients, dtype=int)
    patient_labels[inverse] = labels

    predictions = (mean_p_ad > 0.5).astype(int)
    correct = predictions == patient_labels
    nc = patient_labels == 0
    ad = patient_labels == 1

    return {
        "patient_accuracy": float(correct.mean()),
        "patient_balanced_accuracy": float(
            0.5 * (correct[nc].mean() + correct[ad].mean())
        ),
        "patient_auc": rank_auc(mean_p_ad, patient_labels),
        "patients": int(n_patients),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled,
             with_patient=False):
    """
    Evaluate on a loader. Returns loss, accuracy and sample count.

    With with_patient=True the loader must be unshuffled so rows line up with
    loader.dataset.records, and patient-level metrics are added.
    """
    model.eval()

    loss_sum = 0.0
    correct = 0
    samples = 0
    all_probs = []
    all_labels = []

    if with_patient and isinstance(loader.sampler, RandomSampler):
        raise ValueError("Patient-level evaluation needs an unshuffled loader")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += count_correct(logits, labels)
        samples += batch_size

        if with_patient:
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu())
            all_labels.append(labels.cpu())

    if samples == 0:
        raise RuntimeError("The DataLoader produced no samples.")

    metrics = {
        "loss": loss_sum / samples,
        "accuracy": correct / samples,
        "samples": samples,
    }

    if with_patient:
        records = loader.dataset.records
        if len(records) != samples:
            raise RuntimeError("Record count does not match sample count")
        metrics.update(patient_metrics(
            torch.cat(all_probs).numpy(),
            torch.cat(all_labels).numpy(),
            np.asarray([record["subject_id"] for record in records]),
        ))

    return metrics


# ---------------------------------------------------------------------
# Checkpoint, history and plotting
# ---------------------------------------------------------------------

def config_dict(args):
    """Arguments as a plain dictionary that JSON and torch.save accept."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    path,
    model,
    epoch,
    best_val_accuracy,
    best_val_loss,
    model_config,
    args,
    optimizer=None,
):
    """
    Save weights plus the information needed to rebuild the model.

    `model` is the module that was validated, so with EMA enabled the stored
    weights are the averaged ones.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "model_name": "ConvNeXt",
        "model_config": model_config,
        "training_config": config_dict(args),
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, path)


def write_history(path, history, args):
    """Save the experiment configuration and epoch metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {"config": config_dict(args), "history": history},
            file,
            indent=2,
        )


def save_curves(history, path):
    """Plot train and validation loss and accuracy to one PNG."""
    epochs = [entry["epoch"] for entry in history]

    figure, (loss_axis, accuracy_axis) = plt.subplots(
        1, 2, figsize=(10, 3.6)
    )

    loss_axis.plot(epochs, [e["train_loss"] for e in history], label="train")
    loss_axis.plot(epochs, [e["val_loss"] for e in history], label="validation")
    loss_axis.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    loss_axis.legend()

    accuracy_axis.plot(
        epochs, [e["train_accuracy"] for e in history], label="train"
    )
    accuracy_axis.plot(
        epochs, [e["val_accuracy"] for e in history], label="validation"
    )
    if all("val_patient_accuracy" in e for e in history):
        accuracy_axis.plot(
            epochs, [e["val_patient_accuracy"] for e in history],
            label="validation (patient)", linestyle="--",
        )
    accuracy_axis.set(
        xlabel="epoch", ylabel="accuracy", title="Accuracy"
    )
    accuracy_axis.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args(argv=None):
    # First pass: find the recipe so its values can become the defaults.
    recipe_parser = argparse.ArgumentParser(add_help=False)
    recipe_parser.add_argument(
        "--recipe", choices=sorted(RECIPES), default="default"
    )
    known, _ = recipe_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(
        description="Train ConvNeXt on ADNI 2D MRI slices."
    )
    parser.add_argument(
        "--recipe", choices=sorted(RECIPES), default="default",
        help="Named bundle of defaults; explicit flags override it.",
    )

    # Data
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aug", choices=AUG_CHOICES, default="strong_blur")
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--subset", type=int, default=None)

    # Model
    parser.add_argument(
        "--variant", choices=sorted(CONVNEXT_VARIANTS), default="mini"
    )
    parser.add_argument("--depths", type=int, nargs=4, default=None)
    parser.add_argument("--dims", type=int, nargs=4, default=None)
    parser.add_argument("--drop-path", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Optimisation
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--select-metric", choices=sorted(SELECT_METRICS),
        default="slice_acc",
        help="Validation metric that decides the best checkpoint and early "
             "stopping. Ties are broken by lower validation loss.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--mixup-alpha", type=float, default=0.0,
        help="Beta(alpha, alpha) mixup on training batches; 0 disables it.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--ema-decay", type=float, default=0.999,
        help="EMA decay per optimiser step; 0 disables EMA.",
    )
    parser.add_argument(
        "--decay-all", action=argparse.BooleanOptionalAction, default=False,
        help="Apply weight decay to every parameter, including norms.",
    )
    parser.add_argument(
        "--lr-unit", choices=["step", "epoch"], default="step",
        help="Evaluate the learning-rate schedule per step or per epoch.",
    )

    # Runtime and output
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate the held-out test set once after training finishes.",
    )

    parser.set_defaults(**RECIPES[known.recipe])
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.patience < 1:
        parser.error("--patience must be positive")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.min_lr < 0 or args.min_lr > args.lr:
        parser.error("--min-lr must be between zero and --lr")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.warmup_epochs < 0:
        parser.error("--warmup-epochs cannot be negative")
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be in [0, 1)")
    if not 0 <= args.drop_path < 1:
        parser.error("--drop-path must be in [0, 1)")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if args.max_grad_norm < 0:
        parser.error("--max-grad-norm cannot be negative")
    if args.mixup_alpha < 0:
        parser.error("--mixup-alpha cannot be negative")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be in [0, 1)")
    if (args.depths is None) != (args.dims is None):
        parser.error("--depths and --dims must be given together")

    return args


def resolve_architecture(args):
    """Return (depths, dims, label) from a preset or explicit overrides."""
    if args.depths is not None:
        return tuple(args.depths), tuple(args.dims), "custom"

    preset = CONVNEXT_VARIANTS[args.variant]
    return preset["depths"], preset["dims"], args.variant


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp

    depths, dims, arch_label = resolve_architecture(args)
    run = args.run_name or arch_label

    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )

    print("=" * 65)
    print(f"ADNI MRI Classification - ConvNeXt ({arch_label})")
    print("=" * 65)
    print(f"Run name:        {run}   (recipe: {args.recipe})")
    print(f"Device:          {device}")
    print(f"Mixed precision: {amp_enabled}")
    print(f"CPUs available:  {cpus} (data workers: {args.workers})")
    print(f"Depths / dims:   {depths} / {dims}")
    print(f"Epochs:          {args.epochs} (patience {args.patience})")
    print(f"Batch size:      {args.batch_size}")
    print(f"Select metric:   {args.select_metric}")
    print(f"Mixup alpha:     {args.mixup_alpha}")
    print(f"Learning rate:   {args.lr}")
    print(f"Weight decay:    {args.weight_decay}")
    print(f"Drop path:       {args.drop_path}   Dropout: {args.dropout}")
    print(f"EMA decay:       {args.ema_decay}")
    print(f"Augmentation:    {args.aug}   Image size: {args.img_size}")
    print(f"LR unit: {args.lr_unit}   Decay all: {args.decay_all}   "
          f"Grad clip: {args.max_grad_norm}")
    print(f"Seed:            {args.seed}")
    print(f"Test evaluation: {args.eval_test}")
    print("=" * 65)

    # The dataset module creates the subject-level splits.
    loaders = get_dataloaders(
        batch_size=args.batch_size,
        img_size=args.img_size,
        seed=args.seed,
        num_workers=args.workers,
        horizontal_flip=args.horizontal_flip,
        aug=args.aug,
        subset=args.subset,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    print("\nDataLoader sizes")
    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches:       {len(test_loader)}")

    model_config = {
        "in_channels": 1,
        "num_classes": 2,
        "depths": depths,
        "dims": dims,
        "drop_path_rate": args.drop_path,
        "layer_scale_init": 1e-6,
        "dropout": args.dropout,
    }

    model = ConvNeXt(**model_config).to(device)

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {parameter_count:,}")

    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    # Weights that get validated, checkpointed and finally tested.
    eval_model = ema.module if ema is not None else model

    train_criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(
        model, args.lr, args.weight_decay, decay_all=args.decay_all
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(
        int(args.warmup_epochs * len(train_loader)),
        max(total_steps - 1, 0),
    )

    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    best_path = checkpoint_dir / f"{run}_best.pt"
    last_path = checkpoint_dir / f"{run}_last.pt"
    history_path = output_dir / f"{run}_history.json"
    curves_path = output_dir / f"{run}_curves.png"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_score = -1.0
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        epoch_lr = None

        if args.lr_unit == "epoch":
            epoch_lr = learning_rate_at_step(
                step=epoch - 1,
                total_steps=args.epochs,
                warmup_steps=int(args.warmup_epochs),
                base_lr=args.lr,
                min_lr=args.min_lr,
            )

        train_metrics, global_step, lr = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            args=args,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            global_step=global_step,
            ema=ema,
            fixed_lr=epoch_lr,
        )

        # Validation never updates weights.
        val_metrics = evaluate(
            eval_model, val_loader, eval_criterion, device, amp_enabled,
            with_patient=True,
        )
        score = val_metrics[SELECT_METRICS[args.select_metric]]

        epoch_seconds = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_patient_accuracy": val_metrics["patient_accuracy"],
            "val_patient_balanced_accuracy":
                val_metrics["patient_balanced_accuracy"],
            "val_patient_auc": val_metrics["patient_auc"],
            "epoch_seconds": epoch_seconds,
        })

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | LR {lr:.2e} | "
            f"Train loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['accuracy']:.4f} | "
            f"Val loss {val_metrics['loss']:.4f} "
            f"acc {val_metrics['accuracy']:.4f} | "
            f"Patient acc {val_metrics['patient_accuracy']:.4f} "
            f"auc {val_metrics['patient_auc']:.3f} | "
            f"{epoch_seconds:.1f}s",
            flush=True,
        )

        improved = (
            score > best_score
            or (score == best_score and val_metrics["loss"] < best_val_loss)
        )

        if improved:
            best_score = score
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                best_path, eval_model, epoch,
                best_val_accuracy, best_val_loss, model_config, args,
            )
            print(f"  Saved best checkpoint "
                  f"({args.select_metric} {best_score:.4f})")
        else:
            epochs_without_improvement += 1

        # Latest state, kept so an interrupted job can be inspected.
        save_checkpoint(
            last_path, eval_model, epoch,
            best_val_accuracy, best_val_loss, model_config, args,
            optimizer=optimizer,
        )

        write_history(history_path, history, args)
        save_curves(history, curves_path)

        if epochs_without_improvement >= args.patience:
            print(
                f"\nEarly stopping: no improvement for "
                f"{args.patience} consecutive epochs."
            )
            break

    # Restore the best validation weights into the evaluated module.
    if not best_path.is_file():
        raise RuntimeError("No best-model checkpoint was saved.")

    try:
        checkpoint = torch.load(
            best_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)

    eval_model.load_state_dict(checkpoint["model_state_dict"])

    elapsed_minutes = (time.time() - start_time) / 60.0

    print("\n" + "=" * 65)
    print("Training finished")
    print("=" * 65)
    print(f"Best epoch:           {best_epoch}")
    print(f"Selection metric:     {args.select_metric} = {best_score:.4f}")
    print(f"Best validation acc:  {best_val_accuracy:.4f} (per slice)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Elapsed time:         {elapsed_minutes:.1f} minutes")
    print(f"Best checkpoint:      {best_path}")
    print(f"History:              {history_path}")
    print(f"Curves:               {curves_path}")

    # The test set stays untouched unless explicitly requested.
    if args.eval_test:
        if args.subset is not None:
            raise ValueError(
                "Do not evaluate the test set with --subset. "
                "Rerun without --subset for a proper final evaluation."
            )

        test_metrics = evaluate(
            eval_model, test_loader, eval_criterion, device, amp_enabled
        )
        test_metrics["best_epoch"] = best_epoch

        print("\nFinal held-out test results (best validation checkpoint)")
        print(f"Test loss:     {test_metrics['loss']:.4f}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test samples:  {test_metrics['samples']}")

        test_path = output_dir / f"{run}_test_results.json"
        with test_path.open("w", encoding="utf-8") as file:
            json.dump(test_metrics, file, indent=2)

        print(f"Test results saved to: {test_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()