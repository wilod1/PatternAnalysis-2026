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

