import numpy as np
import pytest
from PIL import Image

from failures import (error_summary, others_wrong_count, pick_cases,
                      plot_cases, select_failures)


def make_run(labels, preds, conf, paths=None):
    n = len(labels)
    return {"labels": np.array(labels), "preds": np.array(preds),
            "conf": np.array(conf),
            "subjects": np.array([f"s{i}" for i in range(n)]),
            "scans": np.array([f"scan{i}" for i in range(n)]),
            "slices": np.arange(n),
            "paths": np.array(paths if paths is not None else [""] * n)}


def test_select_failures_orders_by_confidence_one_per_subject():
    mask = np.ones(4, dtype=bool)
    conf = np.array([0.9, 0.8, 0.95, 0.6])
    subjects = np.array(["a", "a", "b", "c"])
    assert select_failures(mask, conf, subjects, 3) == [2, 0, 3]
    assert select_failures(mask, conf, subjects, 2) == [2, 0]


def test_select_failures_respects_mask():
    mask = np.array([True, False, True])
    conf = np.array([0.5, 0.99, 0.7])
    subjects = np.array(["a", "b", "c"])
    assert select_failures(mask, conf, subjects, 5) == [2, 0]


def test_others_wrong_count():
    labels = np.array([0, 1, 0, 1])
    others = [np.array([0, 0, 0, 1]), np.array([1, 0, 0, 1])]
    assert others_wrong_count(labels, others).tolist() == [1, 2, 0, 0]


def test_error_summary():
    run = make_run([0, 0, 1, 1], [1, 0, 0, 0], [0.9] * 4)
    s = error_summary(run)
    assert s["n_errors"] == 3
    assert s["by_true_class"][0] == {"errors": 1, "n": 2}
    assert s["by_true_class"][1] == {"errors": 2, "n": 2}


def test_pick_cases_splits_shared_and_specific():
    run = make_run([0, 0, 1, 1], [1, 1, 0, 1], [0.9, 0.8, 0.7, 0.99])
    a = make_run([0, 0, 1, 1], [1, 0, 1, 1], [0.9] * 4)  # wrong at 0
    b = make_run([0, 0, 1, 1], [1, 0, 0, 1], [0.9] * 4)  # wrong at 0, 2
    cases = pick_cases(run, [a, b], n_each=3)
    got = [(c["group"], c["true"], c["index"]) for c in cases]
    assert got == [("shared", 0, 0), ("specific", 0, 1)]
    assert cases[0]["others_wrong"] == 2 and cases[1]["others_wrong"] == 0


def test_pick_cases_needs_other_models():
    run = make_run([0, 1], [1, 0], [0.9, 0.9])
    with pytest.raises(ValueError):
        pick_cases(run, [], n_each=1)


def test_plot_cases_writes_figure(tmp_path):
    img = tmp_path / "x.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(img)
    run = make_run([0, 0, 1, 1], [1, 1, 0, 1], [0.9, 0.8, 0.7, 0.99],
                   paths=[str(img)] * 4)
    a = make_run([0, 0, 1, 1], [1, 0, 1, 1], [0.9] * 4)
    b = make_run([0, 0, 1, 1], [1, 0, 0, 1], [0.9] * 4)
    cases = pick_cases(run, [a, b], n_each=2)
    out = tmp_path / "f.png"
    plot_cases(cases, 2, str(out), "test")
    assert out.exists() and out.stat().st_size > 0