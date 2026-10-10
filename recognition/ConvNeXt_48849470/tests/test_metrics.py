"""
Tests for the benchmark metrics, on small hand-checkable examples.
"""
import numpy as np
import pytest

from metrics import (accuracy_by_window, classification_metrics,
                     load_predictions, subject_bootstrap_ci)


def test_perfect_predictions():
    labels = np.array([0, 0, 1, 1])
    m = classification_metrics(labels, labels, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["accuracy"] == m["macro_f1"] == m["auroc"] == 1.0


def test_hand_worked_example():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    m = classification_metrics(labels, preds, np.array([0.1, 0.6, 0.7, 0.9]))
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision_ad"] == pytest.approx(2 / 3)
    assert m["recall_ad"] == pytest.approx(1.0)
    assert m["precision_nc"] == pytest.approx(1.0)
    assert m["recall_nc"] == pytest.approx(0.5)
    assert m["f1_ad"] == pytest.approx(0.8)
    assert m["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert m["auroc"] == pytest.approx(1.0)  # every AD prob exceeds every NC prob


def make_data(n_subjects=40, per_subject=4, seed=0):
    """
    Synthetic predictions: each subject has one class, ~75% of slices correct.
    """
    rng = np.random.default_rng(seed)
    subj_labels = rng.integers(0, 2, n_subjects)
    labels = np.repeat(subj_labels, per_subject)
    subjects = np.repeat(np.arange(n_subjects).astype(str), per_subject)
    correct = rng.random(len(labels)) < 0.75
    preds = np.where(correct, labels, 1 - labels)
    p_ad = np.clip(0.5 + (preds - 0.5) * rng.uniform(0.2, 0.9, len(labels)), 0, 1)
    return {"labels": labels, "preds": preds, "subjects": subjects,
            "probs": np.stack([1 - p_ad, p_ad], axis=1),
            "windows": np.tile(np.arange(per_subject), n_subjects)}


def test_bootstrap_ci_is_ordered_and_brackets_the_estimate():
    data = make_data()
    point = classification_metrics(data["labels"], data["preds"],
                                   data["probs"][:, 1])
    ci = subject_bootstrap_ci(data, n_boot=200, seed=1)
    for name, (lo, hi) in ci.items():
        assert 0.0 <= lo <= hi <= 1.0, name
    assert ci["accuracy"][0] <= point["accuracy"] <= ci["accuracy"][1]


def test_bootstrap_is_reproducible_for_a_seed():
    data = make_data()
    assert (subject_bootstrap_ci(data, 50, seed=3)
            == subject_bootstrap_ci(data, 50, seed=3))


def test_accuracy_by_window():
    data = {"labels": np.array([0, 0, 1, 1, 1, 1]),
            "preds": np.array([0, 1, 1, 1, 0, 1]),
            "windows": np.array([0, 0, 0, 1, 1, 1])}
    rows = accuracy_by_window(data)
    assert rows["0"]["n"] == 3 and rows["0"]["accuracy"] == pytest.approx(2 / 3)
    assert rows["0"]["majority_rate"] == pytest.approx(2 / 3)  # 2 NC of 3
    assert rows["1"]["ad_share"] == 1.0 and rows["1"]["majority_rate"] == 1.0
    assert rows["1"]["accuracy"] == pytest.approx(2 / 3)


def test_load_predictions_reads_one_split(tmp_path):
    path = tmp_path / "preds_x.npz"
    np.savez(path, val_probs=np.array([[0.7, 0.3]]), val_preds=np.array([0]),
             val_label=np.array([1]), val_subject_id=np.array(["s1"]),
             val_window=np.array([4]))
    d = load_predictions(path, "val")
    assert d["labels"].tolist() == [1] and d["subjects"].tolist() == ["s1"]