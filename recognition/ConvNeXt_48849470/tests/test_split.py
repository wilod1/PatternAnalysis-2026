"""
Leakage and reproducibility tests for the subject-level split.
"""
import os
import sys
from functools import lru_cache

# Allow running from any directory -- make dataset.py importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import CLASS_TO_INDEX, SEED, collect_samples, split_subjects


@lru_cache(maxsize=1)
def _splits():
    """
    Build the split once and reuse it across tests.
    """
    return split_subjects(collect_samples(), seed=SEED)


def _subjects(name):
    """
    Set of subject IDs in one split.
    """
    return {r["subject_id"] for r in _splits()[name]}


def test_no_subject_in_two_splits():
    """
    No patient may appear in more than one split (leakage check).
    """
    train, val, test = (_subjects(n) for n in ("train", "val", "test"))
    assert not train & val
    assert not train & test
    assert not val & test


def test_both_classes_in_every_split():
    """
    Each split must contain both AD and NC slices.
    """
    for name, recs in _splits().items():
        assert {r["label"] for r in recs} == set(CLASS_TO_INDEX.values()), name


def test_fractions_and_reproducibility():
    """
    Subject fractions are near 70/15/15 and the same seed gives the same split.
    """
    splits = _splits()
    n_subjects = sum(len(_subjects(n)) for n in splits)
    for name, target in zip(("train", "val", "test"), (0.70, 0.15, 0.15)):
        frac = len(_subjects(name)) / n_subjects
        assert abs(frac - target) < 0.02, (name, frac)
    again = split_subjects(collect_samples(), seed=SEED)
    for name in splits:
        assert [r["path"] for r in splits[name]] == [r["path"] for r in again[name]]


def test_no_slice_lost_or_duplicated():
    """
    Every slice appears in exactly one split.
    """
    paths = [r["path"] for recs in _splits().values() for r in recs]
    assert len(paths) == len(set(paths)) == len(collect_samples())


if __name__ == "__main__":
    # Plain runner so no pytest is required.
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)