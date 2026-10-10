"""
Tests for the slice-window-only baseline. No data needed.
"""
import numpy as np

from utils import fit_window_baseline, window_baseline_probs


def rec(window, label):
    return {"window": window, "label": label}


# window 0: 4 NC; window 1; 3AD + 1 NC; window 5: 1 AD + 1 NC (tie)
TRAIN = ([rec(0, 0)] * 4 + [rec(1, 1)] * 3 + [rec(1, 0)] + [rec(5, 1), rec(5, 0)])


def test_fit_uses_training_share_per_window():
    model = fit_window_baseline(TRAIN)
    assert model["table"][0]["p_ad"] == 0.0
    assert model["table"][1]["p_ad"] == 0.75
    assert model["default_p_ad"] == 0.4 # 4 AD of 10 train slices


def test_probs_sum_to_one_and_predict_the_majority():
    model = fit_window_baseline(TRAIN)
    records = [rec(0, 0), rec(0, 1), rec(1, 1), rec(1, 0), rec(9, 0)]  # 9 is unseen
    probs = window_baseline_probs(model, records)
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(1), 1.0)
    assert probs.argmax(1).tolist() == [0, 0, 1, 1, 0]
    labels = np.array([r["label"] for r in records])
    assert (probs.argmax(1) == labels).mean() == 0.6


def test_tie_window_predicts_nc():
    model = fit_window_baseline(TRAIN)
    assert window_baseline_probs(model, [rec(5, 1)]).argmax(1)[0] == 0