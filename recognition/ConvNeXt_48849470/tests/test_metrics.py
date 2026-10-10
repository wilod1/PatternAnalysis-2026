"""
Tests for the benchmark metrics, on small hand-checkable examples.
"""
import numpy as np
import pytest

from metrics import (accuracy_by_window, calibration_summary,
                     classification_metrics, confidence_and_correct,
                     ece_bootstrap_ci, expected_calibration_error,
                     load_predictions, operating_point,
                     plot_confidence_histograms, plot_reliability_diagram,
                     plot_risk_coverage, reject_analysis, reliability_bins,
                     risk_coverage_curve, subject_bootstrap_ci)


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

def test_load_predictions_flattens_tuple_windows(tmp_path):
    path = tmp_path / "preds_x.npz"
    np.savez(path, val_probs=np.array([[0.7, 0.3], [0.2, 0.8]]),
             val_preds=np.array([0, 1]), val_label=np.array([0, 1]),
             val_subject_id=np.array(["s1", "s2"]),
             val_window=np.array([[60, 80], [80, 100]]))
    d = load_predictions(path, "val")
    assert d["windows"].tolist() == ["60-80", "80-100"]


def test_ece_is_zero_when_confidence_matches_accuracy():
    conf = np.full(10, 0.8)
    correct = np.array([True] * 8 + [False] * 2)  # 80% right at 80% confidence
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_of_an_overconfident_model():
    conf = np.full(10, 0.95)
    correct = np.array([True] * 5 + [False] * 5)  # 50% right at 95% confidence
    assert expected_calibration_error(conf, correct) == pytest.approx(0.45)


def test_ece_weights_bins_by_size():
    conf = np.array([0.6] * 5 + [0.9] * 5)
    correct = np.array([True] * 3 + [False] * 2 + [True] * 5)
    # bin 0.6: accuracy 0.6, gap 0; bin 0.9: accuracy 1.0, gap 0.1
    assert expected_calibration_error(conf, correct) == pytest.approx(0.05)


def test_reliability_bins_cover_every_prediction():
    conf = np.array([0.5, 0.73, 1.0])
    b = reliability_bins(conf, np.array([True, False, True]), n_bins=10)
    assert b["counts"].sum() == 3
    assert b["counts"][0] == 1 and b["counts"][-1] == 1  # 0.5 first, 1.0 last
    assert np.isnan(b["accuracy"][b["counts"] == 0]).all()


def test_calibration_summary_hand_example():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
    labels = np.array([0, 1, 0, 0])  # predictions 0, 1, 1, 0 -> third is wrong
    s = calibration_summary({"probs": probs, "labels": labels})
    assert s["accuracy"] == pytest.approx(0.75)
    assert s["mean_confidence"] == pytest.approx(0.75)
    assert s["mean_conf_correct"] == pytest.approx((0.9 + 0.8 + 0.6) / 3)
    assert s["mean_conf_wrong"] == pytest.approx(0.7)
    assert s["brier"] == pytest.approx((0.01 + 0.04 + 0.49 + 0.16) / 4)


def test_ece_ci_is_ordered_and_reproducible():
    data = make_data()
    lo, hi = ece_bootstrap_ci(data, n_boot=100, seed=2)
    assert 0.0 <= lo <= hi <= 1.0
    assert ece_bootstrap_ci(data, 100, seed=2) == [lo, hi]


def test_calibration_figures_are_written(tmp_path):
    runs = {"cnn_cos": make_data(seed=1), "resnet_cos": make_data(seed=2)}
    plot_reliability_diagram(runs, tmp_path / "rel.png")
    plot_confidence_histograms(runs, tmp_path / "hist.png")
    assert (tmp_path / "rel.png").stat().st_size > 1000
    assert (tmp_path / "hist.png").stat().st_size > 1000


def make_conf_data(conf, correct, labels=None):
    """
    Build prediction arrays with an exact confidence for every slice.
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    n = len(conf)
    labels = np.arange(n) % 2 if labels is None else np.asarray(labels)
    preds = np.where(correct, labels, 1 - labels)
    probs = np.zeros((n, 2))
    probs[np.arange(n), preds] = conf
    probs[np.arange(n), 1 - preds] = 1 - conf
    return {"probs": probs, "labels": labels, "preds": preds,
            "subjects": np.arange(n).astype(str), "windows": np.zeros(n, int)}


def test_risk_coverage_curve_handles_ties():
    conf = np.array([0.9, 0.9, 0.8, 0.6])
    correct = np.array([True, False, True, True])
    c = risk_coverage_curve(conf, correct)
    assert c["threshold"].tolist() == [0.9, 0.8, 0.6]
    assert np.allclose(c["coverage"], [0.5, 0.75, 1.0])
    assert np.allclose(c["accuracy"], [0.5, 2 / 3, 0.75])


def test_operating_point_counts():
    conf = np.array([0.9, 0.9, 0.8, 0.6])
    correct = np.array([True, False, True, True])
    p = operating_point(conf, correct, 0.8)
    assert p["coverage"] == 0.75 and p["n_accepted"] == 3 and p["n_review"] == 1
    assert p["accuracy_accepted"] == pytest.approx(2 / 3)
    assert p["errors_accepted"] == 1 and p["errors_sent_to_review"] == 0


def test_reject_analysis_reports_a_missed_target():
    val = make_conf_data([0.7] * 10, [True] * 5 + [False] * 5)
    r = reject_analysis(val, target_acc=0.90, min_coverage=0.5)
    assert not r["target_met_on_val"]
    assert r["threshold"] == 0.7 and r["val"]["accuracy_accepted"] == 0.5


def test_reject_analysis_picks_highest_coverage_meeting_the_target():
    # 6 right at 0.95, 4 wrong at 0.60: only the 0.95 threshold reaches 90%
    val = make_conf_data([0.95] * 6 + [0.60] * 4, [True] * 6 + [False] * 4)
    r = reject_analysis(val, target_acc=0.90, min_coverage=0.5)
    assert r["target_met_on_val"] and r["threshold"] == 0.95
    assert r["val"]["coverage"] == 0.6 and r["val"]["accuracy_accepted"] == 1.0


def test_reject_threshold_comes_from_val_and_is_applied_to_test():
    val = make_conf_data([0.95] * 6 + [0.60] * 4, [True] * 6 + [False] * 4)
    test = make_conf_data([0.95, 0.95, 0.6, 0.6], [True, False, True, True])
    r = reject_analysis(val, test, 0.90, 0.5)
    assert r["threshold"] == 0.95  # unchanged by the test data
    assert r["test"]["coverage"] == 0.5 and r["test"]["accuracy_accepted"] == 0.5
    assert r["test"]["errors_accepted"] == 1


def test_risk_coverage_figure_is_written(tmp_path):
    runs = {"cnn_cos": make_data(seed=1), "resnet_cos": make_data(seed=2)}
    plot_risk_coverage(runs, tmp_path / "rc.png", points={"cnn_cos": (0.6, 0.8)})
    assert (tmp_path / "rc.png").stat().st_size > 1000