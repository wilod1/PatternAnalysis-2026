"""
Data audit for the ADNI slices.

Covers counts, leakage in the provided split, subject statistics, the
slice-window/class association, our subject-level split, and a window-only
baseline. Reads file names and metadata only (no images).
Output: results/data_audit.json
"""
import json
import os
from collections import Counter, defaultdict

from dataset import INDEX_TO_CLASS, SEED, collect_samples, split_subjects

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "data_audit.json")


def provided_split(record):
    """Which provided folder ('train' or 'test') a slice came from."""
    return "train" if f"{os.sep}train{os.sep}" in record["path"] else "test"


def distinct_by_class(records, key):
    """Number of distinct `key` values (scan_id, subject_id) per class."""
    seen = {(r[key], r["label"]) for r in records}
    return {INDEX_TO_CLASS[l]: sum(1 for _, lab in seen if lab == l)
            for l in sorted(INDEX_TO_CLASS)}


def summary(records):
    """Subjects, scans and slices per class for a list of slice records."""
    return {"subjects": distinct_by_class(records, "subject_id"),
            "scans": distinct_by_class(records, "scan_id"),
            "slices": {INDEX_TO_CLASS[l]: sum(r["label"] == l for r in records)
                       for l in sorted(INDEX_TO_CLASS)}}


def window_baseline(train, evals):
    """Majority class per slice window, fitted on `train`, scored on each split.

    Windows unseen in training fall back to the overall training majority.
    """
    by_window = defaultdict(Counter)
    for r in train:
        by_window[r["window"]][r["label"]] += 1
    overall = Counter(r["label"] for r in train).most_common(1)[0][0]
    rule = {w: c.most_common(1)[0][0] for w, c in by_window.items()}
    acc = {name: sum(rule.get(r["window"], overall) == r["label"] for r in recs) / len(recs)
           for name, recs in evals.items()}
    majority = {name: sum(r["label"] == overall for r in recs) / len(recs)
                for name, recs in evals.items()}
    return {"rule": {f"{w[0]}-{w[1]}": INDEX_TO_CLASS[l] for w, l in sorted(rule.items())},
            "window_accuracy": acc, "always_train_majority": majority}


def main():
    samples = collect_samples()
    splits = split_subjects(samples, seed=SEED)
    audit = {"total": {"slices": len(samples),
                       "scans": len({r["scan_id"] for r in samples}),
                       "subjects": len({r["subject_id"] for r in samples}),
                       **summary(samples)}}

    # Leakage in the provided train/test folders.
    prov = {n: [r for r in samples if provided_split(r) == n] for n in ("train", "test")}
    subj = {n: {r["subject_id"] for r in recs} for n, recs in prov.items()}
    audit["provided_split"] = {
        **{n: summary(recs) for n, recs in prov.items()},
        "test_subjects_also_in_train": len(subj["train"] & subj["test"]),
        "test_subjects": len(subj["test"])}

    # Scans per subject, and slices per scan.
    per_subject = Counter(r["subject_id"] for r in {r["scan_id"]: r for r in samples}.values())
    audit["scans_per_subject"] = {str(k): v for k, v in sorted(Counter(per_subject.values()).items())}
    audit["slices_per_scan"] = {str(k): v for k, v in sorted(
        Counter(Counter(r["scan_id"] for r in samples).values()).items())}

    # Window-by-class table (scans).
    scans = {r["scan_id"]: r for r in samples}.values()
    table = defaultdict(Counter)
    for r in scans:
        table[f"{r['window'][0]}-{r['window'][1]}"][INDEX_TO_CLASS[r["label"]]] += 1
    audit["windows_scans_by_class"] = {w: dict(c) for w, c in sorted(table.items())}

    # Our subject-level split, and the window-only baseline.
    audit["our_split"] = {n: summary(recs) for n, recs in splits.items()}
    audit["window_baseline"] = window_baseline(splits["train"], splits)
    audit["window_baseline_in_sample_all_data"] = window_baseline(samples, {"all": samples})["window_accuracy"]["all"]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps({k: audit[k] for k in ("total", "provided_split", "window_baseline",
                                              "window_baseline_in_sample_all_data")}, indent=2))


if __name__ == "__main__":
    main()