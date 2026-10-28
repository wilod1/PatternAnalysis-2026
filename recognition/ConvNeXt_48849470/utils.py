"""
Helper functions: visualisation, prediction export, and the window baseline.

    predict_split        softmax probabilities for every slice in a data loader
    export_predictions   rebuild a model from a training checkpoint and save its
                         predictions (plus per-slice metadata) to an .npz file
    save_sample_grid     write the two example figures used in the report
    fit_window_baseline  slice-window-only baseline (uses no image content)
    window_baseline_probs  probabilities from that baseline for a list of records

    train.py and per_patient.py call export_predictions once per checkpoint.
    Every later analysis (benchmark, calibration, reject option, failure cases)
    reads the saved .npz files, so none of it needs a GPU or any retraining.

    python utils.py        regenerates assets/sample_grid.png and
                           assets/augmentation_examples.png
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg") # no display on the cluster
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import RandomSampler
from collections import Counter, defaultdict

from dataset import (ADNISliceDataset, CLASS_TO_INDEX, SEED, build_transforms,
                     collect_samples, split_subjects)

# Figures are written next to this file, in the assets/ folder of the project.
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# per-slice fields copied from dataset.records into every exported file
RECORD_FIELDS = ("label", "subject_id", "scan_id", "slice_index", "window", "path")

@torch.no_grad()  # inference only: no gradients, so less memory and faster
def predict_split(model, loader, device):
    """
    Softmax probabilities for every item in an unshuffled loader, shape (N, C).
    Rows follow the order of loader.dataset.records, so each row can be traced
    back to its subject, scan and window.
    """
    # A shuffled loader would return rows in random order and break the link
    # between a probability and the record (subject, scan, window) it belongs to.
    if isinstance(loader.sampler, RandomSampler):
        raise ValueError("predict_split needs an unshuffled loader so rows "
                         "line up with dataset.records")
    model.eval()  # switch off dropout and stochastic depth
    # One batch at a time: logits -> class probabilities, moved back to the CPU
    # so the GPU does not have to hold every prediction at once.
    probs = [torch.softmax(model(x.to(device)), dim=1).cpu() for x, _ in loader]
    return torch.cat(probs).numpy()


def export_predictions(ckpt_path, loaders, out_path, device=None,
                       splits=("val", "test")):
    """
    Rebuild the model stored in a training checkpoint and save its predictions
    for the given splits to a compressed .npz file.

    For each split the file holds <split>_probs (N, 2), <split>_preds (N,) and
    one array per field in RECORD_FIELDS (e.g. val_subject_id, test_window).
    Everything downstream (benchmark, calibration, reject option, failure
    analysis) reads these files, so no analysis needs a GPU or a retrain.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False because checkpoints also store the training settings
    # (paths, configuration dictionaries), not only tensors.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        # new-format checkpoints written by train.py (ConvNeXt or a baseline)
        name = ckpt.get("model_name", "ConvNeXt")
        if name == "ConvNeXt":
            # the checkpoint stores depths, widths and rates, so the exact
            # architecture can be rebuilt without knowing the variant name
            from modules import convnext_from_checkpoint
            model = convnext_from_checkpoint(ckpt).to(device)
        else:
            # baselines (SimpleCNN, SimpleResNet): look the class up by name
            import modules
            model = getattr(modules, name)(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # old-format checkpoints (baselines)
        from train import build_model
        model = build_model(ckpt["model"],
                            argparse.Namespace(**ckpt["args"])).to(device)
        model.load_state_dict(ckpt["state_dict"])

    # epoch of the saved (best-validation) checkpoint, kept for reference
    out = {"epoch": np.asarray(ckpt["epoch"])}
    for split in splits:
        loader = loaders[split]
        records = loader.dataset.records
        probs = predict_split(model, loader, device)
        # one probability row per record, otherwise metadata would be misaligned
        assert len(probs) == len(records), "prediction count != record count"
        out[f"{split}_probs"] = probs
        out[f"{split}_preds"] = probs.argmax(1)  # class with the highest probability
        # copy each record's metadata next to its prediction
        for field in RECORD_FIELDS:
            out[f"{split}_{field}"] = np.asarray([r[field] for r in records])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **out)
    return out


def save_sample_grid(out_dir=ASSETS, n_cols=5, seed=SEED):
    """
    Save two figures: random slices per split/class, and augmentation draws.

    Images are shown after padding but before normalisation (mean 0, std 1
    makes the normalise step an identity), so brightness is readable.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)  # also fixes the random augmentations in Figure 2
    # patient-disjoint split of the pooled data (same seed as every experiment)
    splits = split_subjects(collect_samples(), seed=seed)
    # build_transforms(training, mean, std): the evaluation pipeline pads only;
    # the training pipeline adds the default "mild" augmentation preset
    eval_tf = build_transforms(False, 0.0, 1.0)
    aug_tf = build_transforms(True, 0.0, 1.0)

    # Figure 1: random slices from train and test, per class.
    # One row per (split, class) pair, n_cols random slices in each row.
    rows = [(s, c) for s in ("train", "test") for c in ("AD", "NC")]
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(2.2 * n_cols, 2.5 *len(rows)))
    for ax_row, (split, cls) in zip(axes, rows):
        # all slices of this split that belong to the class being drawn
        pool = [r for r in splits[split] if r["label"] == CLASS_TO_INDEX[cls]]
        for ax, r in zip(ax_row, rng.sample(pool, n_cols)):
            # a one-item dataset gives the padded image; [0] drops the label
            img, _ = ADNISliceDataset([r], eval_tf)[0]
            ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
            # label each panel with its slice window and slice index
            ax.set_title(f"{split} {cls}\nwindow {r['window'][0]}-{r['window'][1]}, "
                         f"slice {r['slice_index']}", fontsize=7)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_grid.png"), dpi=130)
    plt.close(fig)  # free memory; no window is ever shown

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


def fit_window_baseline(train_records):
    """
    Slice-window-only baseline: for each slice window, the share of AD slices
    among the training records. Uses no image content, so it measures how
    much accuracy the slice position alone can give.
    """
    # counts[window][label] = slices with that label in that window;
    # overall[label] = the same over all windows (fallback for unseen windows)
    counts, overall = defaultdict(Counter), Counter()
    for r in train_records:
        counts[r["window"]][r["label"]] += 1
        overall[r["label"]] += 1
    # per window: number of slices n and the AD share p_ad (label 1 = AD)
    table = {w: {"n": sum(c.values()), "p_ad": c[1] / sum(c.values())}
             for w, c in sorted(counts.items())}
    return {"table": table, "default_p_ad": overall[1] / sum(overall.values())}


def window_baseline_probs(model, records):
    """
    (N, 2) probabilities [NC, AD] for each record from its window alone. A
    window never seen in training falls back to the overall training AD share.
    A 50/50 window predicts NC (argmax picks index 0 on ties).
    """
    # look up each record's window in the fitted table; use the overall AD
    # share when the window did not occur in training
    p_ad = np.array([model["table"].get(r["window"],
                                        {"p_ad": model["default_p_ad"]})["p_ad"]
                     for r in records])
    # columns are [NC, AD], so the NC probability is 1 - P(AD)
    return np.stack([1 - p_ad, p_ad], axis=1)


if __name__ == "__main__":
    save_sample_grid()