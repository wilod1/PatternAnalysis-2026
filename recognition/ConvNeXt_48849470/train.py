
"""
Train and evaluate a ConvNeXt model for ADNI MRI slice classification.

Classes:
    NC -> 0
    AD -> 1

The training and validation sets are used for model development.
The test set is not evaluated unless --eval-test is explicitly supplied.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from modules import ConvNeXt


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prefer reproducibility over potentially nondeterministic algorithms.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def calculate_accuracy(logits, labels):
    """Calculate the number of correctly classified samples."""
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    amp_enabled=False,
    max_grad_norm=1.0,
):
    """
    Run one training or evaluation epoch.

    Passing an optimizer enables training. Without an optimizer, gradients
    are disabled and the model is evaluated.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad if training else torch.no_grad

    with context():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )

                    optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.detach().item() * batch_size
            total_correct += calculate_accuracy(logits.detach(), labels)
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("The DataLoader produced no samples.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
    }


# ---------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------

def learning_rate_at_step(
    step,
    total_steps,
    warmup_steps,
    base_lr,
    min_lr,
):
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
# Checkpoint and history utilities
# ---------------------------------------------------------------------

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_accuracy,
    best_val_loss,
    args,
):
    """Save model weights and enough information to reproduce the run."""
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "model_name": "ConvNeXt",
            "model_config": {
                "in_channels": 1,
                "num_classes": 2,
                "depths": (3, 3, 9, 3),
                "dims": (96, 192, 384, 768),
                "drop_path_rate": args.drop_path,
                "layer_scale_init": 1e-6,
                "dropout": args.dropout,
            },
            "training_config": vars(args),
        },
        path,
    )


def write_history(path, history, args):
    """Save the experiment configuration and epoch metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config": vars(args),
                "history": history,
            },
            file,
            indent=2,
        )


# ---------------------------------------------------------------------
# Optional final test evaluation
# ---------------------------------------------------------------------

def evaluate_test_set(model, loader, criterion, device, amp_enabled):
    """Evaluate the held-out test set, intended for final evaluation only."""
    metrics = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )

    print("\nFinal held-out test results")
    print(f"Test loss:     {metrics['loss']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test samples:  {metrics['samples']}")

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt on ADNI 2D MRI slices."
    )

    # Data
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aug", choices=["mild", "strong"], default="mild")
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--subset", type=int, default=None)

    # Optimisation
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Runtime and output
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate the held-out test set once after training finishes.",
    )

    args = parser.parse_args()

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

    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device selection
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = device.type == "cuda" and not args.no_amp

    print("=" * 65)
    print("ADNI MRI Classification — ConvNeXt")
    print("=" * 65)
    print(f"Device:          {device}")
    print(f"Mixed precision: {amp_enabled}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print(f"Weight decay:     {args.weight_decay}")
    print(f"Augmentation:     {args.aug}")
    print(f"Seed:             {args.seed}")
    print(f"Test evaluation:  {args.eval_test}")
    print("=" * 65)

    # Load data. The dataset module creates subject-level splits.
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

    # Build the full ConvNeXt architecture.
    model = ConvNeXt(
        in_channels=1,
        num_classes=2,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=args.drop_path,
        layer_scale_init=1e-6,
        dropout=args.dropout,
    ).to(device)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nModel parameters: {parameter_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(
        int(args.warmup_epochs * len(train_loader)),
        max(total_steps - 1, 0),
    )

    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "convnext_best.pt"
    history_path = output_dir / "convnext_history.json"
    last_checkpoint_path = checkpoint_dir / "convnext_last.pt"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    start_time = time.time()

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
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

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.detach().item() * batch_size
            train_correct += calculate_accuracy(logits.detach(), labels)
            train_samples += batch_size
            global_step += 1

        train_metrics = {
            "loss": train_loss_sum / train_samples,
            "accuracy": train_correct / train_samples,
            "samples": train_samples,
        }

        # Validation: never update model weights using validation batches.
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        epoch_seconds = time.time() - epoch_start

        epoch_result = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"LR {lr:.2e} | "
            f"Train loss {train_metrics['loss']:.4f} | "
            f"Train acc {train_metrics['accuracy']:.4f} | "
            f"Val loss {val_metrics['loss']:.4f} | "
            f"Val acc {val_metrics['accuracy']:.4f} | "
            f"{epoch_seconds:.1f}s"
        )

        # Save the most recent state in case the job is interrupted.
        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_accuracy=best_val_accuracy,
            best_val_loss=best_val_loss,
            args=args,
        )

        # Select the best checkpoint using validation data only.
        improved = (
            val_metrics["accuracy"] > best_val_accuracy
            or (
                val_metrics["accuracy"] == best_val_accuracy
                and val_metrics["loss"] < best_val_loss
            )
        )

        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                best_val_loss=best_val_loss,
                args=args,
            )

            print(
                f"  Saved best checkpoint "
                f"(validation accuracy: {best_val_accuracy:.4f})"
            )
        else:
            epochs_without_improvement += 1

        write_history(history_path, history, args)

        if epochs_without_improvement >= args.patience:
            print(
                f"\nEarly stopping: validation performance did not "
                f"improve for {args.patience} consecutive epochs."
            )
            break

    # -------------------------------------------------------------
    # Restore the best validation checkpoint
    # -------------------------------------------------------------
    if not checkpoint_path.is_file():
        raise RuntimeError("No best-model checkpoint was saved.")

    # weights_only=False is required for the full checkpoint dictionary
    # on PyTorch versions where weights-only loading is the default.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    elapsed_minutes = (time.time() - start_time) / 60.0

    print("\n" + "=" * 65)
    print("Training finished")
    print("=" * 65)
    print(f"Best epoch:             {best_epoch}")
    print(f"Best validation acc:    {best_val_accuracy:.4f}")
    print(f"Best validation loss:   {best_val_loss:.4f}")
    print(f"Elapsed time:           {elapsed_minutes:.1f} minutes")
    print(f"Best checkpoint:        {checkpoint_path}")
    print(f"Last checkpoint:        {last_checkpoint_path}")
    print(f"Training history:       {history_path}")

    # The test set remains untouched unless explicitly requested.
    if args.eval_test:
        if args.subset is not None:
            raise ValueError(
                "Do not evaluate the test set with --subset. "
                "Rerun without --subset for a proper final evaluation."
            )

        test_metrics = evaluate_test_set(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        test_path = output_dir / "convnext_test_results.json"
        with test_path.open("w", encoding="utf-8") as file:
            json.dump(test_metrics, file, indent=2)

        print(f"Test results saved to: {test_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
