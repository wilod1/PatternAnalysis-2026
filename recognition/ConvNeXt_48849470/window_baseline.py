"""
Fit the slice-window-only baseline on the training split, evaluate it on the
validation and test splits, and save its predictions in the same format as the
neural models (results/preds_window_only.npz) plus a summary JSON.

Usage: python window_baseline.py   (reads metadata only, no images)
"""
import json
import os

import numpy as np

from dataset import SEED, collect_samples, split_subjects
from utils import RECORD_FIELDS, fit_window_baseline, window_baseline_probs

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_NAME = {0: "NC", 1: "AD"}

def main():
    splits = split_subjects(collect_samples(), seed=SEED)
    model = fit_window_baseline(splits["train"])
    train_labels = np.array([r["label"] for r in splits["train"]])
    majority = int(train_labels.mean() > 0.5)  # always-predict-this baseline

    arrays = {"epoch": np.asarray(0)}
    summary = {
        "majority_class": CLASS_NAME[majority],
        "windows": {str(w): {"n_train": v["n"],
                             "ad_share": round(v["p_ad"], 4),
                             "predicts": CLASS_NAME[int(v["p_ad"] > 0.5)]}
                    for w, v in model["table"].items()},
    }

    for split in ("val", "test"):
        records = splits[split]
        probs = window_baseline_probs(model, records)
        labels = np.array([r["label"] for r in records])
        arrays[f"{split}_probs"] = probs
        arrays[f"{split}_preds"] = probs.argmax(1)
        for field in RECORD_FIELDS:
            arrays[f"{split}_{field}"] = np.asarray([r[field] for r in records])
        summary[f"{split}_acc"] = round(float((probs.argmax(1) == labels).mean()), 4)
        summary[f"{split}_majority_acc"] = round(float((labels == majority).mean()), 4)
        summary[f"{split}_n_slices"] = len(records)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "results", "preds_window_only.npz"),
                        **arrays)
    with open(os.path.join(HERE, "results", "window_baseline.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "windows"}, indent=2))

if __name__ == "__main__":
    main()