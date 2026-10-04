"""
Data utilities for ADNI AD-vs-NC classification (2D brain MRI slices).

The privided train/test folders share patients, so all slices are pooled
here, and re-split by subject (please see split_subjects).
"""

import json
import os
import re
import random
from collections import Counter

ADNI_ROOT = "/home/groups/comp3710/ADNI"
JSON_NAME = "meta_data_with_label.json"
CLASS_TO_INDEX = {"NC": 0, "AD": 1} # model labels
JSON_LABEL_TO_CLASS = {0: "NC", 2: "AD"} # label 1 is not in the folders

# Subject IDs look like 068_S_0473 inside the JSON "raw" path
_SUBJECT_RE = re.compile(r"ADNI_(\d+_S_\d+)")


def load_scan_table(root=ADNI_ROOT):
    """
    Map scan ID -> {'subject_id': str, 'label': int} from the metadata JSON.
    """
    with open(os.path.join(root, JSON_NAME)) as f:
        meta = json.load(f)

    table = {}
    for scan_id, entry in meta.items():
        match = _SUBJECT_RE.search(entry["raw"])
        if match is None:
            raise ValueError(f"No subject ID found for scan {scan_id}")
        table[scan_id] = {"subject_id": match.group(1), "label": entry["label"]}

    return table


def collect_samples(root=ADNI_ROOT):
    """
    Pool the provided train/test folders into one record per slice.

    Each record holds: path, label (0=NC, 1=AD), scan_id, subject_id,
    slice_index and window (first, last slice index of the scan).
    """
    scans = load_scan_table(root)
    records = []
    for split in ("train", "test"):
        for cls, label in CLASS_TO_INDEX.items():
            folder = os.path.join(root, "AD_NC", split, cls)
            for name in sorted(os.listdir(folder)):
                scan_id, slice_idx = os.path.splitext(name)[0].split("_")
                info = scans[scan_id]

                # Sanity check -- folder class must agree with JSON label
                if JSON_LABEL_TO_CLASS[info["label"]] != cls:
                    raise ValueError(f"Label mismatch for scan {scan_id}")

                records.append({
                    "path": os.path.join(folder, name),
                    "label": label,
                    "scan_id": scan_id,
                    "subject_id": info["subject_id"],
                    "slice_index": int(slice_idx)
                })

    # Window = (first, last) sslice index of each scan
    spans = {}
    for r in records:
        lo, hi = spans.get(r["scan_id"], (10**9, -1))
        spans[r["scan_id"]] = (min(lo, r["slice_index"]), max(hi, r["slice_index"]))
    for r in records:
        r["window"] = spans[r["scan_id"]]
    return records

INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}
SEED = 42 # for reproducibility

def split_subjects(samples, seed=SEED, ratios=(0.70, 0.15, 0.15)):
    """
    Split slice records into train/val/test, stratified by class, by SUBJECT.

    Every slice of a subject (across all of their scans) goes to the same
    split, which prevents patient-level leakage. Each subject has a single
    class (checked below), so class is used as the stratum.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")

    # subject -> class label, verifying that a subject never has 2 classes
    subject_label = {}
    for r in samples:
        prev = subject_label.setdefault(r["subject_id"], r["label"])
        if prev != r["label"]:
            raise ValueError(f"Subject {r['subject_id']} has more than one class")

    rng = random.Random(seed)
    assignment = {}
    for label in sorted(set(subject_label.values())):
        # Sort before shuffling so result depends only on seed
        subjects = sorted(s for s, l in subject_label.items() if l == label)
        rng.shuffle(subjects)
        n_train = round(ratios[0] * len(subjects))
        n_val = round(ratios[1] * len(subjects))
        for i, s in enumerate(subjects):
            if i < n_train:
                assignment[s] = "train"
            elif i< n_train + n_val:
                assignment[s] = "val"
            else:
                assignment[s] = "test"

    splits = {"train": [], "val": [], "test": []}
    for r in samples:
        splits[assignment[r["subject_id"]]].append(r)
    return splits


def describe_split(splits):
    """
    Print subjects, scans, slices and window-by-class scan counts per split.
    """
    for name, recs in splits.items():
        subjects = {r["subject_id"]: r["label"] for r in recs}
        scans = {r["scan_id"]: (r["label"], r["window"]) for r in recs}
        n_ad = sum(l == CLASS_TO_INDEX["AD"] for l in subjects.values())
        print(f"{name}: {len(subjects)} subjects ({n_ad} AD / "
              f"{len(subjects) - n_ad} NC), {len(scans)} scans, {len(recs)} slices")
        windows = Counter((w, l) for l, w in scans.values())
        for (w, l), n in sorted(windows.items()):
            print(f"    window {w[0]}-{w[1]} {INDEX_TO_CLASS[l]}: {n} scans")
