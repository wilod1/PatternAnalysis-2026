"""
Per-slice vs per-patient accuracy for a trained ConvNeXt checkpoint.

Usage (on Rangpur, inside the torch env, from the repo folder):
    python per_patient.py results/checkpoints/tiny_rc_best.pt
    python per_patient.py results/checkpoints/mini_a_best.pt --split val

Predictions are exported to results/preds/<run>.npz. Defaults to the VALIDATION
split; use --split test only once, after the metric choice is written in the README.
Patient score = mean of the slice probabilities over every slice of that subject.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import get_dataloaders
from utils import export_predictions


def patient_table(probs, labels, subjects):
    ids = np.unique(subjects)
    p = np.stack([probs[subjects == s].mean(0) for s in ids])
    y = np.array([labels[subjects == s][0] for s in ids])
    return ids, p, y


def bootstrap_ci(correct, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(correct)
    accs = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return np.percentile(accs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=None,
                    help="also report patient metrics when AD if mean P(AD) > t")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # new format stores training_config; old baseline checkpoints store args
    cfg = ckpt.get("training_config") or ckpt.get("args") or {}
    aug = cfg.get("aug", "mild")
    img_size = cfg.get("img_size", 256)
    print(f"checkpoint epoch {ckpt['epoch']}, aug={aug}, img_size={img_size}")

    loaders = get_dataloaders(
        batch_size=64, img_size=img_size, seed=cfg.get("seed", 42),
        num_workers=args.workers, aug=aug,
        split_mode=cfg.get("split_mode", "subject"),
    )
    run = args.checkpoint.stem.replace("_best", "")
    suffix = "" if args.split == "val" else f"_{args.split}"
    out_path = Path("results/preds") / f"{run}{suffix}.npz"
    out = export_predictions(str(args.checkpoint), loaders, str(out_path),
                             splits=(args.split,))

    s = args.split
    probs, labels = out[f"{s}_probs"], out[f"{s}_label"]
    subjects = out[f"{s}_subject_id"]

    slice_correct = (probs.argmax(1) == labels).astype(float)
    ids, pp, py = patient_table(probs, labels, subjects)
    pat_correct = (pp.argmax(1) == py).astype(float)

    lo, hi = bootstrap_ci(pat_correct)
    print(f"\n{run} [{s}]")
    print(f"per-slice   acc {slice_correct.mean():.4f}  (n={len(labels)} slices)")
    print(f"per-patient acc {pat_correct.mean():.4f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]  (n={len(ids)} subjects)")
    for c, name in ((0, "NC"), (1, "AD")):
        m = py == c
        print(f"  {name} patients: {pat_correct[m].mean():.3f} (n={m.sum()})")

    if args.threshold is not None:
        t = args.threshold
        pred = (pp[:, 1] > t).astype(int)
        tp = int(((pred == 1) & (py == 1)).sum())
        fn = int(((pred == 0) & (py == 1)).sum())
        tn = int(((pred == 0) & (py == 0)).sum())
        fp = int(((pred == 1) & (py == 0)).sum())
        correct = (pred == py).astype(float)
        lo, hi = bootstrap_ci(correct)
        print(f"\nat t={t:.2f} [{s}]")
        print(f"patient acc {correct.mean():.4f}  95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"sensitivity (AD recall) {tp / (tp + fn):.3f}   "
              f"specificity (NC recall) {tn / (tn + fp):.3f}")
        print(f"TP {tp}  FN {fn}  TN {tn}  FP {fp}")


if __name__ == "__main__":
    main()