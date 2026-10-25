"""
Final benchmark, calibration, reject-option and failure-case analysis.

Reads the prediction files written by per_patient.py:
    results/preds/<run>.npz         validation split
    results/preds/<run>_test.npz    test split
No GPU is needed. Everything is per slice, and every confidence interval
resamples whole SUBJECTS, because the slices of one subject are not independent.

Writes results/final/final_results.json, results/final/summary.txt and figures
into assets/.

Usage:  python final_analysis.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

PRED_DIR = Path("results/preds")
OUT_DIR = Path("results/final")
ASSET_DIR = Path("assets")

# provided-split models (headline protocol)
RUNS = ["prov_rc", "prov_mix", "prov_tiny_a", "prov_mini_a", "prov_cnn", "prov_resnet"]
HEADLINE, BASELINE = "prov_rc", "prov_resnet"
# patient-disjoint models, used only for a reject-option check without leakage
CLEAN_RUNS = ["tiny_rc", "tiny_mix"]

N_BOOT = 1000
N_BINS = 10
TARGET_ACC = 0.90            # accuracy wanted on accepted slices (feasibility review)
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)

# one fixed colour per model, so a model keeps its colour in every figure
COLOURS = {
    "prov_rc": "#1baf7a", "prov_mix": "#7fd1b0", "prov_tiny_a": "#b5e3d0",
    "prov_mini_a": "#d9f0e6", "prov_cnn": "#2a78d6", "prov_resnet": "#eb6834",
    "window_only": "#6b6b66",
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"

LINES = []


def say(text=""):
    """Print a line and keep it for summary.txt."""
    print(text, flush=True)
    LINES.append(text)


# ---------------------------------------------------------------- loading

def load_run(run, split):
    """Prediction arrays for one run and split, or None if the file is missing."""
    name = f"{run}.npz" if split == "val" else f"{run}_test.npz"
    path = PRED_DIR / name
    if not path.is_file():
        return None
    d = np.load(path)
    windows = d[f"{split}_window"]
    if windows.ndim > 1:
        windows = np.array(["-".join(str(v) for v in row) for row in windows])
    return {
        "probs": d[f"{split}_probs"].astype(float),
        "labels": d[f"{split}_label"].astype(int),
        "subjects": d[f"{split}_subject_id"].astype(str),
        "windows": windows.astype(str),
        "paths": d[f"{split}_path"].astype(str),
        "slice_index": d[f"{split}_slice_index"].astype(int),
    }


def same_rows(a, b):
    """True if two prediction sets describe the same slices in the same order."""
    return (len(a["labels"]) == len(b["labels"])
            and np.array_equal(a["labels"], b["labels"])
            and np.array_equal(a["subjects"], b["subjects"]))


# ---------------------------------------------------------------- metrics

def per_class(y, pred):
    """Precision, recall and F1 for NC (0) and AD (1)."""
    out = {}
    for c, name in ((0, "nc"), (1, "ad")):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"precision_{name}"] = float(precision)
        out[f"recall_{name}"] = float(recall)
        out[f"f1_{name}"] = float(f1)
    return out


def macro_f1(y, pred):
    c = per_class(y, pred)
    return (c["f1_nc"] + c["f1_ad"]) / 2


def reliability_bins(conf, correct):
    """Equal-width confidence bins over [0.5, 1] (two classes, so conf >= 0.5)."""
    edges = np.linspace(0.5, 1.0, N_BINS + 1)
    idx = np.digitize(conf, edges[1:-1])
    counts = np.bincount(idx, minlength=N_BINS)
    sum_conf = np.bincount(idx, weights=conf, minlength=N_BINS)
    sum_correct = np.bincount(idx, weights=correct.astype(float), minlength=N_BINS)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, sum_conf / counts, np.nan)
        accuracy = np.where(counts > 0, sum_correct / counts, np.nan)
    return counts, mean_conf, accuracy


def ece(conf, correct):
    """Expected calibration error: bin-weighted gap between confidence and accuracy."""
    counts, mean_conf, accuracy = reliability_bins(conf, correct)
    full = counts > 0
    return float(np.sum(np.abs(accuracy[full] - mean_conf[full]) * counts[full]) / counts.sum())


def summarise(d):
    """All headline numbers for one prediction set (no intervals)."""
    y, p = d["labels"], d["probs"]
    pred = p.argmax(1)
    conf, correct = p.max(1), pred == y
    out = {"n_slices": int(len(y)), "accuracy": float(correct.mean())}
    out.update(per_class(y, pred))
    out["macro_f1"] = float(macro_f1(y, pred))
    out["auroc"] = float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else float("nan")
    out["ece"] = ece(conf, correct)
    out["brier"] = float(np.mean((p[:, 1] - y) ** 2))
    out["nll"] = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))
    out["mean_conf"] = float(conf.mean())
    out["mean_conf_correct"] = float(conf[correct].mean()) if correct.any() else float("nan")
    out["mean_conf_wrong"] = float(conf[~correct].mean()) if (~correct).any() else float("nan")
    # does confidence rank correct predictions above wrong ones?
    out["conf_auroc"] = (float(roc_auc_score(correct.astype(int), conf))
                         if correct.any() and (~correct).any() else float("nan"))
    return out


def bootstrap(datas, n_boot=N_BOOT, seed=0):
    """
    Subject-level bootstrap for several runs at once. Every run sees the same
    resamples, so differences between runs are paired. All runs must describe
    the same slices in the same order.
    """
    ref = next(iter(datas.values()))
    _, inverse = np.unique(ref["subjects"], return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)
    keys = ("accuracy", "macro_f1", "auroc", "ece")
    draws = {run: {k: [] for k in keys} for run in datas}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        for run, d in datas.items():
            y, p = d["labels"][idx], d["probs"][idx]
            pred = p.argmax(1)
            conf, correct = p.max(1), pred == y
            draws[run]["accuracy"].append(correct.mean())
            draws[run]["macro_f1"].append(macro_f1(y, pred))
            draws[run]["auroc"].append(
                roc_auc_score(y, p[:, 1]) if len(np.unique(y)) == 2 else np.nan)
            draws[run]["ece"].append(ece(conf, correct))
    return {run: {k: np.array(v) for k, v in dr.items()} for run, dr in draws.items()}


def interval(values):
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


# ---------------------------------------------------------------- reject option

def choose_tau(conf, correct, target):
    """
    Lowest confidence threshold whose accepted slices reach `target` accuracy,
    i.e. the largest coverage that still meets the target. None if no threshold
    does. Chosen on VALIDATION data only.
    """
    best = None
    for t in np.unique(np.round(conf, 4)):
        keep = conf >= t
        if not keep.any():
            continue
        acc, cov = correct[keep].mean(), keep.mean()
        if acc >= target and (best is None or cov > best[1]):
            best = (float(t), float(cov), float(acc))
    return best


def apply_tau(conf, correct, tau):
    keep = conf >= tau
    errors = int((~correct).sum())
    return {
        "tau": float(tau),
        "coverage": float(keep.mean()),
        "accepted_accuracy": float(correct[keep].mean()) if keep.any() else float("nan"),
        "residual_errors": int((~correct[keep]).sum()),
        "errors_caught": errors - int((~correct[keep]).sum()),
        "total_errors": errors,
    }


def accuracy_at_coverage(conf, correct):
    """Accuracy of the most confident `coverage` share of slices (label-free ranking)."""
    order = np.argsort(-conf, kind="stable")
    out = {}
    for cov in COVERAGES:
        k = max(int(round(cov * len(conf))), 1)
        out[f"{cov:.1f}"] = float(correct[order[:k]].mean())
    return out


def reject_analysis(runs_val, runs_test, title):
    """Choose tau on validation for each run and apply it once to test."""
    say(f"\n{title}")
    say(f"tau = lowest confidence whose accepted validation slices reach {TARGET_ACC:.2f} accuracy")
    say(f"{'run':14s} {'val cov':>8s} {'tau':>6s} {'test cov':>9s} {'test acc (accepted)':>20s} "
        f"{'errors caught':>14s}")
    results = {}
    for run in runs_test:
        if run not in runs_val:
            continue
        v, t = runs_val[run], runs_test[run]
        vconf, vcorrect = v["probs"].max(1), v["probs"].argmax(1) == v["labels"]
        tconf, tcorrect = t["probs"].max(1), t["probs"].argmax(1) == t["labels"]
        choice = choose_tau(vconf, vcorrect, TARGET_ACC)
        entry = {"accuracy_at_coverage_test": accuracy_at_coverage(tconf, tcorrect)}
        if choice is None:
            entry["rule"] = None
            say(f"{run:14s} {'none':>8s}  (no threshold reaches the target on validation)")
        else:
            tau, vcov, vacc = choice
            applied = apply_tau(tconf, tcorrect, tau)
            entry["rule"] = {"val_coverage": vcov, "val_accepted_accuracy": vacc, **applied}
            say(f"{run:14s} {vcov:8.3f} {tau:6.3f} {applied['coverage']:9.3f} "
                f"{applied['accepted_accuracy']:20.4f} "
                f"{applied['errors_caught']:>6d}/{applied['total_errors']:<6d}")
        results[run] = entry
    say("\ntest accuracy of the most confident share of slices (ranking by confidence only)")
    say(f"{'run':14s} " + " ".join(f"{c:>7.1f}" for c in COVERAGES))
    for run, entry in results.items():
        say(f"{run:14s} " + " ".join(f"{entry['accuracy_at_coverage_test'][f'{c:.1f}']:7.3f}"
                                     for c in COVERAGES))
    return results


# ---------------------------------------------------------------- figures

def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def plot_reliability(datas, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.8), facecolor=SURFACE)
    ax.plot([0.5, 1.0], [0.5, 1.0], "--", color="#9a9992", linewidth=1.2,
            label="perfect calibration")
    for run, d in datas.items():
        conf, correct = d["probs"].max(1), d["probs"].argmax(1) == d["labels"]
        counts, mean_conf, acc = reliability_bins(conf, correct)
        keep = counts >= 10
        ax.plot(mean_conf[keep], acc[keep], marker="o", markersize=5, linewidth=1.6,
                color=COLOURS.get(run, "#6b6b66"), markeredgecolor=SURFACE,
                label=f"{run} (ECE {ece(conf, correct):.3f})")
    ax.set(xlim=(0.5, 1.0), ylim=(0.0, 1.0), xlabel="mean confidence in bin",
           ylabel="accuracy in bin", title="Reliability diagram, test set")
    style(ax)
    legend = ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_hist(datas, path):
    n = len(datas)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 3.8), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    edges = np.linspace(0.5, 1.0, N_BINS + 1)
    centres, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    for i, (ax, (run, d)) in enumerate(zip(axes[0], datas.items())):
        conf, correct = d["probs"].max(1), d["probs"].argmax(1) == d["labels"]
        for sel, colour, name, shift in ((correct, "#2a78d6", "correct", -1),
                                         (~correct, "#eb6834", "wrong", 1)):
            share = np.histogram(conf[sel], bins=edges)[0] / max(int(sel.sum()), 1)
            ax.bar(centres + shift * width / 4, share, width=width / 2, color=colour,
                   edgecolor=SURFACE, linewidth=1.0, label=name)
        ax.set(xlim=(0.5, 1.0), xlabel="confidence",
               title=f"{run}\n{int((~correct).sum())} wrong of {len(correct)}")
        if i == 0:
            ax.set_ylabel("share of predictions in group")
        style(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_risk_coverage(datas, rules, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.6), facecolor=SURFACE)
    for run, d in datas.items():
        conf, correct = d["probs"].max(1), d["probs"].argmax(1) == d["labels"]
        order = np.argsort(-conf, kind="stable")
        acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
        cov = np.arange(1, len(order) + 1) / len(order)
        ax.plot(cov, acc, color=COLOURS.get(run, "#6b6b66"), linewidth=1.6, label=run)
        rule = (rules.get(run) or {}).get("rule")
        if rule:
            ax.plot(rule["coverage"], rule["accepted_accuracy"], "o", color=COLOURS.get(run),
                    markeredgecolor=INK, markersize=7)
    ax.axhline(TARGET_ACC, color="#9a9992", linestyle="--", linewidth=1.0)
    ax.set(xlim=(0.05, 1.0), xlabel="coverage (share of slices answered automatically)",
           ylabel="accuracy on answered slices",
           title="Accuracy against coverage, test set (dots: threshold chosen on validation)")
    style(ax)
    ax.title.set_fontsize(10)
    legend = ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- failure cases

def pick_failures(d):
    """Two false negatives and two false positives from distinct subjects, plus the
    most borderline error. Within each kind, the most confident error comes first."""
    y, p = d["labels"], d["probs"]
    conf, wrong = p.max(1), np.flatnonzero(p.argmax(1) != y)
    by_conf = wrong[np.argsort(-conf[wrong], kind="stable")]

    def top(true_label, n):
        picks, seen = [], set()
        for i in by_conf:
            if y[i] == true_label and d["subjects"][i] not in seen:
                picks.append(int(i))
                seen.add(d["subjects"][i])
                if len(picks) == n:
                    break
        return picks

    fn, fp = top(1, 2), top(0, 2)
    borderline = [int(i) for i in wrong[np.argsort(conf[wrong], kind="stable")]
                  if int(i) not in fn + fp][:1]
    return ([("false negative (AD called NC)", i) for i in fn]
            + [("false positive (NC called AD)", i) for i in fp]
            + [("borderline error", i) for i in borderline])


def failure_report(datas, train_subjects, path):
    d = datas[HEADLINE]
    base = datas.get(BASELINE)
    picks = pick_failures(d)
    say(f"\nfailure cases, {HEADLINE}, test set")
    cases = []
    fig, axes = plt.subplots(1, max(len(picks), 1), figsize=(3.2 * max(len(picks), 1), 4.0),
                             facecolor=SURFACE, squeeze=False)
    for ax, (kind, i) in zip(axes[0], picks):
        p_ad, label = float(d["probs"][i, 1]), int(d["labels"][i])
        subject = d["subjects"][i]
        patient_mean = float(d["probs"][d["subjects"] == subject][:, 1].mean())
        seen = (subject in train_subjects) if train_subjects is not None else None
        base_ok = None
        if base is not None and same_rows(base, d):
            base_ok = bool(base["probs"][i].argmax() == label)
        case = {
            "kind": kind, "file": Path(d["paths"][i]).name, "subject": subject,
            "true": "AD" if label else "NC", "p_ad": p_ad,
            "confidence": float(d["probs"][i].max()), "window": d["windows"][i],
            "slice_index": int(d["slice_index"][i]), "patient_mean_p_ad": patient_mean,
            "patient_seen_in_train": seen, "baseline_correct": base_ok,
        }
        cases.append(case)
        say(f"- {kind}: {case['file']} subject {subject}, true {case['true']}, "
            f"P(AD) {p_ad:.3f}, window {case['window']}, slice {case['slice_index']}, "
            f"patient mean P(AD) {patient_mean:.3f}, seen in train {seen}, "
            f"{BASELINE} correct {base_ok}")
        try:
            ax.imshow(Image.open(d["paths"][i]).convert("L"), cmap="gray")
        except OSError:
            ax.text(0.5, 0.5, "image not found", ha="center")
        ax.set_title(f"{kind}\ntrue {case['true']}, P(AD) {p_ad:.2f}\n"
                     f"window {case['window']}, slice {case['slice_index']}", fontsize=7,
                     color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return cases


# ---------------------------------------------------------------- window analysis

def by_window(d):
    rows = {}
    pred = d["probs"].argmax(1)
    for w in np.unique(d["windows"]):
        m = d["windows"] == w
        share = d["labels"][m].mean()
        rows[str(w)] = {"n": int(m.sum()), "ad_share": float(share),
                        "majority_rate": float(max(share, 1 - share)),
                        "accuracy": float((pred[m] == d["labels"][m]).mean())}
    return rows


# ---------------------------------------------------------------- main

def to_py(obj):
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    val = {r: load_run(r, "val") for r in RUNS}
    test = {r: load_run(r, "test") for r in RUNS}
    missing = [r for r in RUNS if val[r] is None or test[r] is None]
    for r in missing:
        say(f"WARNING: prediction files for {r} are missing; it is skipped")
    val = {r: d for r, d in val.items() if d is not None and r not in missing}
    test = {r: d for r, d in test.items() if d is not None and r not in missing}
    if HEADLINE not in test:
        raise SystemExit("headline run has no predictions; run per_patient.py first")

    # window-only baseline and leak status, rebuilt from the dataset module
    train_subjects = None
    try:
        from dataset import collect_samples, split_provided
        from utils import fit_window_baseline, window_baseline_probs

        records = collect_samples()
        train_subjects = {r["subject_id"] for r in records if "/AD_NC/train/" in r["path"]}
        splits = split_provided(records)
        model = fit_window_baseline(splits["train"])
        for name, store in (("val", val), ("test", test)):
            recs = splits[name]
            store["window_only"] = {
                "probs": window_baseline_probs(model, recs),
                "labels": np.array([r["label"] for r in recs]),
                "subjects": np.array([r["subject_id"] for r in recs]).astype(str),
                "windows": np.array(["-".join(str(v) for v in r["window"]) for r in recs]),
                "paths": np.array([r["path"] for r in recs]),
                "slice_index": np.array([r["slice_index"] for r in recs]),
            }
    except Exception as error:  # the rest of the analysis does not depend on it
        say(f"NOTE: window-only baseline and leak status unavailable ({error})")

    names = list(test)
    say("=" * 78)
    say("PROVIDED SPLIT, TEST SET: per-slice benchmark (95% intervals resample subjects)")
    say("=" * 78)

    # paired bootstrap over the runs whose rows match the headline run's
    ref = test[HEADLINE]
    paired = {r: d for r, d in test.items() if same_rows(d, ref)}
    draws = bootstrap(paired)
    for r, d in test.items():
        if r not in paired:
            draws.update(bootstrap({r: d}))

    summary = {r: summarise(d) for r, d in test.items()}
    for r in names:
        summary[r]["ci95"] = {k: interval(v) for k, v in draws[r].items()}
    results["test_summary"] = summary

    say(f"{'run':13s} {'accuracy [95% CI]':26s} {'macroF1':>8s} {'AUROC':>7s} {'ECE':>7s} "
        f"{'Brier':>7s} {'NLL':>7s}")
    for r in names:
        s, ci = summary[r], summary[r]["ci95"]
        say(f"{r:13s} {s['accuracy']:.4f} [{ci['accuracy'][0]:.3f}, {ci['accuracy'][1]:.3f}]   "
            f"{s['macro_f1']:8.3f} {s['auroc']:7.3f} {s['ece']:7.3f} {s['brier']:7.3f} "
            f"{s['nll']:7.3f}")

    say("\nper-class precision / recall / F1 (NC | AD)")
    for r in names:
        s = summary[r]
        say(f"{r:13s} NC {s['precision_nc']:.3f} / {s['recall_nc']:.3f} / {s['f1_nc']:.3f}   "
            f"AD {s['precision_ad']:.3f} / {s['recall_ad']:.3f} / {s['f1_ad']:.3f}")

    say("\nconfidence versus correctness (test)")
    say(f"{'run':13s} {'mean conf':>10s} {'when right':>11s} {'when wrong':>11s} "
        f"{'AUROC of confidence for correctness':>36s}")
    for r in names:
        s = summary[r]
        say(f"{r:13s} {s['mean_conf']:10.3f} {s['mean_conf_correct']:11.3f} "
            f"{s['mean_conf_wrong']:11.3f} {s['conf_auroc']:36.3f}")

    say(f"\npaired differences, {HEADLINE} minus each other model (95% interval)")
    paired_out = {}
    for r in paired:
        if r == HEADLINE or r not in paired:
            continue
        da = draws[HEADLINE]["accuracy"] - draws[r]["accuracy"]
        de = draws[HEADLINE]["ece"] - draws[r]["ece"]
        paired_out[r] = {
            "accuracy_diff": float(summary[HEADLINE]["accuracy"] - summary[r]["accuracy"]),
            "accuracy_diff_ci95": interval(da),
            "ece_diff": float(summary[HEADLINE]["ece"] - summary[r]["ece"]),
            "ece_diff_ci95": interval(de),
        }
        a, e = paired_out[r], paired_out[r]
        say(f"vs {r:13s} accuracy {a['accuracy_diff']:+.4f} "
            f"[{a['accuracy_diff_ci95'][0]:+.3f}, {a['accuracy_diff_ci95'][1]:+.3f}]   "
            f"ECE {e['ece_diff']:+.4f} [{e['ece_diff_ci95'][0]:+.3f}, {e['ece_diff_ci95'][1]:+.3f}]")
    results["paired_differences"] = paired_out

    # leakage breakdown: accuracy on test patients seen vs not seen in the train folder
    if train_subjects is not None:
        say("\naccuracy on test patients seen vs not seen in the provided train folder")
        leak = {}
        for r in names:
            d = test[r]
            seen = np.array([s in train_subjects for s in d["subjects"]])
            ok = d["probs"].argmax(1) == d["labels"]
            leak[r] = {"seen_patients": int(len(set(d["subjects"][seen]))),
                       "seen_slices": int(seen.sum()), "seen_accuracy": float(ok[seen].mean()),
                       "unseen_patients": int(len(set(d["subjects"][~seen]))),
                       "unseen_slices": int((~seen).sum()),
                       "unseen_accuracy": float(ok[~seen].mean())}
            say(f"{r:13s} seen {leak[r]['seen_accuracy']:.4f} "
                f"({leak[r]['seen_patients']} patients, {leak[r]['seen_slices']} slices)   "
                f"unseen {leak[r]['unseen_accuracy']:.4f} "
                f"({leak[r]['unseen_patients']} patients, {leak[r]['unseen_slices']} slices)")
        results["leakage_breakdown"] = leak

    # validation accuracy of the same models, for the record (leaked under this protocol)
    say("\nvalidation accuracy (provided protocol; leaked, so not used to choose between models)")
    for r in val:
        if r in names:
            say(f"{r:13s} {summarise(val[r])['accuracy']:.4f}")

    # reject option on the provided split
    model_names = [r for r in names if r != "window_only"]
    say("\n" + "=" * 78)
    say("REJECT OPTION")
    say("=" * 78)
    reject = reject_analysis(val, test, "provided split (validation is leaked, so tau tends to be too permissive)")
    results["reject_provided"] = reject

    clean_val = {r: load_run(r, "val") for r in CLEAN_RUNS}
    clean_test = {r: load_run(r, "test") for r in CLEAN_RUNS}
    clean_val = {r: d for r, d in clean_val.items() if d is not None}
    clean_test = {r: d for r, d in clean_test.items() if d is not None and r in clean_val}
    if clean_test:
        results["reject_patient_disjoint"] = reject_analysis(
            clean_val, clean_test, "patient-disjoint split (validation has no leakage)")

    # window analysis
    say("\n" + "=" * 78)
    say("ACCURACY PER SLICE WINDOW, test set (majority rate = always guess the window's commoner class)")
    say("=" * 78)
    results["by_window"] = {}
    for r in (HEADLINE, BASELINE, "window_only"):
        if r not in test:
            continue
        rows = by_window(test[r])
        results["by_window"][r] = rows
        say(f"\n{r}")
        say(f"{'window':>10s} {'n':>6s} {'AD share':>9s} {'majority':>9s} {'accuracy':>9s}")
        for w, v in rows.items():
            say(f"{w:>10s} {v['n']:6d} {v['ad_share']:9.3f} {v['majority_rate']:9.3f} "
                f"{v['accuracy']:9.3f}")

    # figures
    plot_reliability({r: test[r] for r in model_names}, ASSET_DIR / "final_reliability_test.png")
    hist_runs = {r: test[r] for r in (HEADLINE, BASELINE, "prov_cnn") if r in test}
    plot_confidence_hist(hist_runs, ASSET_DIR / "final_confidence_hist_test.png")
    plot_risk_coverage({r: test[r] for r in model_names}, reject,
                       ASSET_DIR / "final_risk_coverage_test.png")
    say("\nfigures: assets/final_reliability_test.png, final_confidence_hist_test.png, "
        "final_risk_coverage_test.png")

    # failure cases
    results["failure_cases"] = failure_report(test, train_subjects,
                                              ASSET_DIR / "final_failure_cases.png")
    say("figure: assets/final_failure_cases.png")

    with open(OUT_DIR / "final_results.json", "w") as f:
        json.dump(to_py(results), f, indent=2)
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("\n".join(LINES) + "\n")
    say("\nsaved results/final/final_results.json and results/final/summary.txt")


if __name__ == "__main__":
    main()