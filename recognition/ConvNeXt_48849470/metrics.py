"""
Benchmark metrics computed from saved prediction files
(results/preds_<run>.npz, written by export_preds.py and window_baseline.py).

Everything is per slice. Confidence intervals resample whole SUBJECTS, because
slices from one subject are not independent.

Usage: python metrics.py --runs window_only cnn_cos resnet_cos --split val
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ("nc", "ad")


def load_predictions(path, split):
    """
    Load one split from a prediction file as plain arrays.
    """
    data = np.load(path)
    return {
        "probs": data[f"{split}_probs"],
        "preds": data[f"{split}_preds"],
        "labels": data[f"{split}_label"],
        "subjects": data[f"{split}_subject_id"],
        "windows": data[f"{split}_window"],
    }


def classification_metrics(labels, preds, p_ad):
    """
    Accuracy, per-class precision/recall/F1, macro-F1 and AUROC (AD is the
    positive class). A class with no predictions get precision 0.
    """
    out = {"accuracy": float((preds == labels).mean())}
    f1s = []
    for c, name in enumerate(CLASS_NAMES):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        out[f"precision_{name}"] = float(precision)
        out[f"recall_{name}"] = float(recall)
        out[f"f1_{name}"] = float(f1)
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    out["auroc"] = (float(roc_auc_score(labels, p_ad))
                    if len(np.unique(labels)) == 2 else float("nan"))
    return out


def subject_bootstrap_ci(data, n_boot=1000, seed=0, level=0.95):
    """
    Percentile bootstrap interval for every metric in classification_metrics.
    Each resample draws subjects with replacement and keeps all of a drawn
    subject's slices. Returns {metric: [low, high]}.
    """
    labels, preds, p_ad = data["labels"], data["preds"], data["probs"][:, 1]
    subjects, inverse = np.unique(data["subjects"], return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(len(subjects))]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        drawn = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in drawn])
        samples.append(classification_metrics(labels[idx], preds[idx], p_ad[idx]))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {m: [float(np.nanpercentile([s[m] for s in samples], lo)),
                float(np.nanpercentile([s[m] for s in samples], hi))]
            for m in samples[0]}


def accuracy_by_window(data):
    """
    Per slice window: number of slices, AD share, the majority-class rate in
    that window (the accuracy of always guessing the window's commoner class on
    this split) and the model's accuracy. A model that uses anatomy should beat
    the majority rate inside windows where both classes occur.
    """
    rows = {}
    for w in np.unique(data["windows"]):
        mask = data["windows"] == w
        labels = data["labels"][mask]
        ad_share = labels.mean()
        rows[str(w)] = {
            "n": int(mask.sum()),
            "ad_share": float(ad_share),
            "majority_rate": float(max(ad_share, 1 - ad_share)),
            "accuracy": float((data["preds"][mask] == labels).mean()),
        }
    return rows
