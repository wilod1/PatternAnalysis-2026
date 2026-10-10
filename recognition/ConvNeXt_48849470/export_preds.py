"""
Export validation and test predictions for trained runs.

Usage: python export_preds.py cnn resnet_cos convnext_a
Reads checkpoints/<run>_best.pt, writes results/preds_<run>.npz.
"""
import argparse
import os

import torch

from dataset import get_dataloaders
from utils import export_predictions

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="run names, e.g. resnet_cos")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in a.runs:
        ckpt_path = os.path.join(HERE, "checkpoints", f"{run}_best.pt")
        train_args = torch.load(ckpt_path, map_location="cpu")["args"]
        # same seed and image size as training, so the split is identical
        loaders = get_dataloaders(batch_size=a.batch_size,
                                  img_size=train_args["img_size"],
                                  seed=train_args["seed"],
                                  num_workers=a.num_workers)
        out_path = os.path.join(HERE, "results", f"preds_{run}.npz")
        out = export_predictions(ckpt_path, loaders, out_path, device)
        # validation only: test numbers are not looked at until Phase 8
        val_acc = (out["val_preds"] == out["val_label"]).mean()
        print(f"{run}: epoch {int(out['epoch'])}, {len(out['val_preds'])} val "
              f"and {len(out['test_preds'])} test slices, val acc {val_acc:.4f}")


if __name__ == "__main__":
    main()