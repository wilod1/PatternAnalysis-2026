"""
Helper functions: visualisation, metrics, and profiling.
"""
import os
import random

import matplotlib
matplotlib.use("Agg") # no display on the cluster
import matplotlib.pyplot as plt
import torch

from dataset import (ADNISliceDataset, CLASS_TO_INDEX, SEED, build_transforms,
                     collect_samples, split_subjects)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def save_sample_grid(out_dir=ASSETS, n_cols=5, seed=SEED):
    """
    Save two figures: random slices per split/class, and augmentation draws.

    Images are shown after padding but before normalisation (mean 0, std 1
    makes the normalise step an identity), so brightness is readable.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    splits = split_subjects(collect_samples(), seed=seed)
    eval_tf = build_transforms(False, 0.0, 1.0)
    aug_tf = build_transforms(True, 0.0, 1.0)

    # Figure 1: random slices from train and test, per class.
    rows = [(s, c) for s in ("train", "test") for c in ("AD", "NC")]
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(2.2 * n_cols, 2.5 *len(rows)))
    for ax_row, (split, cls) in zip(axes, rows):
        pool = [r for r in splits[split] if r["label"] == CLASS_TO_INDEX[cls]]
        for ax, r in zip(ax_row, rng.sample(pool, n_cols)):
            img, _ = ADNISliceDataset([r], eval_tf)[0]
            ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{split} {cls}\nwindow {r['window'][0]}-{r['window'][1]}, "
                         f"slice {r['slice_index']}", fontsize=7)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_grid.png"), dpi=130)
    plt.close(fig)

    # Figure 2: one training slice, original then several augmentation draws.
    r = rng.choice([x for x in splits["train"] if x["label"] == CLASS_TO_INDEX["AD"]])
    original, _ = ADNISliceDataset([r], eval_tf)[0]
    aug_ds = ADNISliceDataset([r], aug_tf)
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
    axes[0].imshow(original[0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original", fontsize=8)
    for ax in axes[1:]:
        img, _ = aug_ds[0]  # a fresh random augmentation on every call
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("augmented", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "augmentation_examples.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    save_sample_grid()