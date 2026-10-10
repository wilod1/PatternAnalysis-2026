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

import matplotlib
matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ("nc", "ad")
# figure styling: light surface, recessive grid, one fixed colour per model family
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
FAMILY_COLOURS = {"window": "#6b6b66", "cnn": "#2a78d6",
                  "resnet": "#eb6834", "convnext": "#1baf7a"}
CORRECT_COLOUR, WRONG_COLOUR = "#2a78d6", "#eb6834"


def load_predictions(path, split):
    """
    Load one split from a prediction file as plain arrays. A window stored as a
    tuple such as (start, end) is turned into a single label like "60-80".
    """
    data = np.load(path)
    windows = data[f"{split}_window"]
    if windows.ndim > 1:
        windows = np.array(["-".join(str(v) for v in row) for row in windows])
    return {
        "probs": data[f"{split}_probs"],
        "preds": data[f"{split}_preds"],
        "labels": data[f"{split}_label"],
        "subjects": data[f"{split}_subject_id"],
        "windows": windows,
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


def benchmark(pred_path, split, n_boot=1000, seed=0):
    """
    Full benchmark for one prediction file and split.
    """
    data = load_predictions(pred_path, split)
    return {
        "split": split,
        "n_slices": int(len(data["labels"])),
        "n_subjects": int(len(np.unique(data["subjects"]))),
        "metrics": classification_metrics(data["labels"], data["preds"],
                                          data["probs"][:, 1]),
        "ci95": subject_bootstrap_ci(data, n_boot, seed),
        "by_window": accuracy_by_window(data),
    }


def confidence_and_correct(probs, labels):
    """
    Confidence (probability of the predicted class) and whether it was right.
    """
    return probs.max(axis=1), probs.argmax(axis=1) == labels


def reliability_bins(conf, correct, n_bins=10):
    """
    Group predictions into equal-width confidence bins over [0.5, 1] (with two
    classes the top probability is never below 0.5). Returns the bin edges,
    counts, mean confidence and accuracy per bin (NaN for empty bins).
    """
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.digitize(conf, edges[1:-1])  # 0 .. n_bins-1; conf == 1.0 -> last bin
    counts = np.bincount(idx, minlength=n_bins)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    sum_correct = np.bincount(idx, weights=correct.astype(float), minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, sum_conf / counts, np.nan)
        accuracy = np.where(counts > 0, sum_correct / counts, np.nan)
    return {"edges": edges, "counts": counts,
            "mean_conf": mean_conf, "accuracy": accuracy}


def expected_calibration_error(conf, correct, n_bins=10):
    """
    ECE: the average gap between confidence and accuracy over the bins,
    weighted by how many predictions fall in each bin.
    """
    b = reliability_bins(conf, correct, n_bins)
    full = b["counts"] > 0
    gap = np.abs(b["accuracy"][full] - b["mean_conf"][full])
    return float(np.sum(gap * b["counts"][full]) / b["counts"].sum())


def calibration_summary(data, n_bins=10):
    """
    ECE, Brier score and how confident the model is when right vs when wrong.
    A calibrated model is clearly less confident on its mistakes.
    """
    conf, correct = confidence_and_correct(data["probs"], data["labels"])
    brier = np.mean((data["probs"][:, 1] - data["labels"]) ** 2)
    return {
        "ece": expected_calibration_error(conf, correct, n_bins),
        "brier": float(brier),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "mean_conf_correct": (float(conf[correct].mean())
                              if correct.any() else float("nan")),
        "mean_conf_wrong": (float(conf[~correct].mean())
                            if (~correct).any() else float("nan")),
        "n_bins": n_bins,
    }


def ece_bootstrap_ci(data, n_boot=1000, seed=0, n_bins=10, level=0.95):
    """
    Percentile interval for ECE, resampling whole subjects like the other CIs.
    """
    conf, correct = confidence_and_correct(data["probs"], data["labels"])
    subjects, inverse = np.unique(data["subjects"], return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(len(subjects))]
    rng = np.random.default_rng(seed)
    eces = []
    for _ in range(n_boot):
        drawn = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in drawn])
        eces.append(expected_calibration_error(conf[idx], correct[idx], n_bins))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return [float(np.percentile(eces, lo)), float(np.percentile(eces, hi))]


def run_colour(run):
    """
    One fixed colour per model family, so a model keeps its colour in every figure.
    """
    family = "window" if run.startswith("window") else run.split("_")[0]
    return FAMILY_COLOURS.get(family, "#6b6b66")


def _style_axes(ax):
    """
    Light surface, recessive grid and axes, no top or right spine.
    """
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


def plot_reliability_diagram(runs_data, path, n_bins=10, min_count=10):
    """
    One reliability diagram for all runs: accuracy against mean confidence per
    bin. Points on the dashed diagonal are perfectly calibrated; below it means
    overconfident. Bins with fewer than min_count predictions are left out.
    """
    fig, ax = plt.subplots(figsize=(5.4, 5.6), facecolor=SURFACE)
    ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="#9a9992",
            linewidth=1.2, label="perfect calibration")
    for run, data in runs_data.items():
        conf, correct = confidence_and_correct(data["probs"], data["labels"])
        b = reliability_bins(conf, correct, n_bins)
        keep = b["counts"] >= min_count
        ece = expected_calibration_error(conf, correct, n_bins)
        ax.plot(b["mean_conf"][keep], b["accuracy"][keep], marker="o",
                markersize=6, linewidth=1.8, color=run_colour(run),
                markeredgecolor=SURFACE, label=f"{run} (ECE {ece:.3f})")
    ax.set(xlim=(0.5, 1.0), ylim=(0.0, 1.0), xlabel="mean confidence in bin",
           ylabel="accuracy in bin", title="Reliability diagram")
    _style_axes(ax)
    legend = ax.legend(frameon=False, fontsize=8, loc="upper center",
                       bbox_to_anchor=(0.5, -0.16))  # below the axes, off the data
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_histograms(runs_data, path, n_bins=10):
    """
    One panel per run: the distribution of confidence for correct and for wrong
    predictions, each as a share of its own group so the shapes are comparable.
    A useful model puts its mistakes at lower confidence than its hits.
    """
    n = len(runs_data)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 3.8), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    centres, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    for i, (ax, (run, data)) in enumerate(zip(axes[0], runs_data.items())):
        conf, correct = confidence_and_correct(data["probs"], data["labels"])
        for sel, colour, name, shift in ((correct, CORRECT_COLOUR, "correct", -1),
                                         (~correct, WRONG_COLOUR, "wrong", 1)):
            share = np.histogram(conf[sel], bins=edges)[0] / max(int(sel.sum()), 1)
            ax.bar(centres + shift * width / 4, share, width=width / 2,
                   color=colour, edgecolor=SURFACE, linewidth=1.0, label=name)
        ax.set(xlim=(0.5, 1.0), xlabel="confidence",
               title=f"{run}\n{int((~correct).sum())} wrong of {len(correct)}")
        if i == 0:
            ax.set_ylabel("share of predictions in group")
        _style_axes(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", ncol=2,
                        frameon=False, fontsize=9)  # one legend above all panels
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def risk_coverage_curve(conf, correct):
    """
    Accuracy of the accepted predictions as the confidence threshold is lowered.
    Coverage is the fraction of slices decided automatically (confidence >=
    threshold). Predictions tied at one confidence are accepted together, so a
    threshold never splits a tie. Arrays run from the highest threshold (low
    coverage) to the lowest (coverage 1.0).
    """
    order = np.argsort(-conf, kind="stable")
    c = conf[order]
    hits = np.cumsum(correct[order])
    n = np.arange(1, len(c) + 1)
    last = np.r_[c[1:] != c[:-1], True] # last item of each run of equal confidence
    return {"threshold": c[last], "coverage": n[last] / len(c),
            "accuracy": hits[last] / n[last]}


def operating_point(conf, correct, threshold):
    """
    What a reject rule does on this data: accept predictions with confidence >=
    threshold, send the rest to human review. Counts the errors on each side.
    """
    accept = conf >= threshold
    n_accept = int(accept.sum())
    return {
        "threshold": float(threshold),
        "coverage": float(n_accept / len(conf)),
        "accuracy_accepted": (float(correct[accept].mean())
                              if n_accept else float("nan")),
        "n_accepted": n_accept,
        "n_review": int(len(conf) - n_accept),
        "errors_accepted": int((~correct[accept]).sum()),
        "errors_sent_to_review": int((~correct[~accept]).sum()),
    }


def reject_analysis(val_data, test_data=None, target_acc=0.90, min_coverage=0.5):
    """
    Choose a confidence threshold on VALIDATION data only, then report what it
    does there and, if test_data is given, on test data with the same threshold.

    Rule: the lowest threshold (highest coverage) whose accepted predictions
    reach target_acc while coverage stays at least min_coverage. If no threshold
    does, fall back to the best accuracy at coverage >= min_coverage and flag
    target_met_on_val as False so the shortfall is reported, not hidden.
    """
    conf, correct = confidence_and_correct(val_data["probs"], val_data["labels"])
    curve = risk_coverage_curve(conf, correct)
    ok = (curve["accuracy"] >= target_acc) & (curve["coverage"] >= min_coverage)
    if ok.any():
        threshold, met = float(curve["threshold"][ok][-1]), True
    else:
        pool = curve["coverage"] >= min_coverage
        best = int(np.argmax(np.where(pool, curve["accuracy"], -1.0)))
        threshold, met = float(curve["threshold"][best]), False
    result = {"target_acc": target_acc, "min_coverage": min_coverage,
              "target_met_on_val": met, "threshold": threshold,
              "val": operating_point(conf, correct, threshold)}
    if test_data is not None:
        t_conf, t_correct = confidence_and_correct(test_data["probs"],
                                                   test_data["labels"])
        result["test"] = operating_point(t_conf, t_correct, threshold)
    return result


def plot_risk_coverage(runs_data, path, target_acc=0.90, min_coverage=0.5,
                       points=None):
    """
    Accuracy of accepted slices against coverage, one line per run. The dashed
    lines mark the target accuracy and the minimum coverage, so the acceptable
    region is the top-right corner. points maps run -> (coverage, accuracy) of
    its chosen threshold, drawn as a marker on that run's line.
    """
    fig, ax = plt.subplots(figsize=(5.4, 5.6), facecolor=SURFACE)
    ax.axhline(target_acc, linestyle="--", color="#9a9992", linewidth=1.2,
               label=f"target accuracy {target_acc:.2f}")
    ax.axvline(min_coverage, linestyle=":", color="#9a9992", linewidth=1.2,
               label=f"minimum coverage {min_coverage:.2f}")
    for run, data in runs_data.items():
        conf, correct = confidence_and_correct(data["probs"], data["labels"])
        curve = risk_coverage_curve(conf, correct)
        ax.plot(curve["coverage"], curve["accuracy"], linewidth=1.8,
                color=run_colour(run), label=run)
        if points and run in points:
            ax.plot(*points[run], marker="o", markersize=8, color=run_colour(run),
                    markeredgecolor=SURFACE, linestyle="none")
    ax.set(xlim=(0.0, 1.0), ylim=(0.4, 1.0), xlabel="coverage (share decided automatically)",
           ylabel="accuracy of accepted slices", title="Risk-coverage")
    _style_axes(ax)
    legend = ax.legend(frameon=False, fontsize=8, loc="upper center",
                       bbox_to_anchor=(0.5, -0.16))
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run_reject_analysis(a, loaded):
    """
    Reject-option analysis for every run, written to results/reject_option_<split>.json
    and assets/risk_coverage_<split>.png. The threshold always comes from the
    validation split; with --split test it is then applied to test.
    """
    out, points = {}, {}
    for run, data in loaded.items():
        if a.split == "val":
            val, test = data, None
        else:
            val = load_predictions(
                os.path.join(HERE, "results", f"preds_{run}.npz"), "val")
            test = data
        out[run] = reject_analysis(val, test, a.target_acc, a.min_coverage)
        r = out[run]
        shown = r["val"] if a.split == "val" else r["test"]
        points[run] = (shown["coverage"], shown["accuracy_accepted"])
        print(f"{run:14s} threshold {r['threshold']:.3f} "
              f"({'target met' if r['target_met_on_val'] else 'TARGET NOT MET'} on val)  "
              f"{a.split}: coverage {shown['coverage']:.3f}, accuracy accepted "
              f"{shown['accuracy_accepted']:.3f}, {shown['n_review']} to review, "
              f"{shown['errors_accepted']} errors accepted")
    os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"reject_option_{a.split}.json"), "w") as f:
        json.dump(out, f, indent=2)
    plot_risk_coverage(loaded, os.path.join(HERE, "assets", f"risk_coverage_{a.split}.png"),
                       a.target_acc, a.min_coverage, points)
    print("saved reject-option json and figure")


def agreement_analysis(runs_data, reference=None, top_subjects=10):
    """
    Do the models fail on the same slices, and do errors cluster in a few
    subjects? All runs must be predictions for the same split (same slices in
    the same order). The reference run (default: the last one listed) is the
    model whose errors are examined most closely, e.g. the ConvNeXt.
    """
    runs = list(runs_data)
    first = runs_data[runs[0]]
    for r in runs[1:]:
        for key in ("labels", "subjects", "windows"):
            if not np.array_equal(runs_data[r][key], first[key]):
                raise ValueError(f"{r} and {runs[0]} disagree on '{key}': "
                                 "prediction files must be from the same split")
        reference = reference or runs[-1]
        labels = first["labels"]
        wrong = {r: runs_data[r]["preds"] != labels for r in runs}
    n_wrong = np.sum([wrong[r] for r in runs], axis=0)  # models wrong per slice

    pairs = {}
    for i, a in enumerate(runs):
        for b in runs[i + 1]:
            both = int((wrong[a] & wrong[b]).sum())
            union = int((wrong[a] | wrong[b]).sum())
            pairs[f"{a} & {b}"] = {
                "both_wrong": both,
                "only_first": int((wrong[a] & ~wrong[b]).sum()),
                "only_second": int((wrong[b] & ~wrong[a]).sum()),
                "jaccard": both / union if union else float("nan"),
            }

    ref_wrong = wrong[reference]
    n_ref = int(ref_wrong.sum())
    shared = {r: (float((ref_wrong & wrong[r]).sum() / n_ref)
                  if n_ref else float("nan"))
              for r in runs if r != reference}

    by_window = {}
    for w in np.unique(first["windows"]):
        m = first["windows"] == w
        by_window[str(w)] = {
            "n": int(m.sum()),
            "error_rate": {r: float(wrong[r][m].mean()) for r in runs},
            "wrong_by_all": float((n_wrong[m] == len(runs)).mean()),
        }

    subjects, inverse = np.unique(first["subjects"], return_inverse=True)
    n_per = np.bincount(inverse)
    rates = {r: np.bincount(inverse, weights=wrong[r].astype(float)) / n_per
             for r in runs}
    mean_rate = np.mean([rates[r] for r in runs], axis=0)
    hard = []
    for k in np.argsort(-mean_rate, kind="stable")[:top_subjects]:
        label = int(labels[inverse == k][0])  # a subject has a single class
        hard.append({"subject": str(subjects[k]),
                     "class": CLASS_NAMES[label].upper(),
                     "n": int(n_per[k]),
                     "mean_error_rate": float(mean_rate[k]),
                     "error_rate": {r: float(rates[r][k]) for r in runs}})

    ref_errors = np.bincount(inverse, weights=ref_wrong.astype(float))
    top = np.sort(ref_errors)[::-1][:top_subjects]
    concentration = {
        "top_subjects": top_subjects,
        "share_of_errors": (float(top.sum() / ref_errors.sum())
                            if ref_errors.sum() else float("nan")),
        "subjects_over_half_wrong": int((rates[reference] > 0.5).sum()),
        "n_subjects": int(len(subjects)),
    }
    return {
        "runs": runs,
        "reference": reference,
        "n_slices": int(len(labels)),
        "n_slices_wrong_by_k": {str(k): int((n_wrong == k).sum())
                                for k in range(len(runs) + 1)},
        "pairs": pairs,
        "reference_errors_also_made_by": shared,
        "by_window": by_window,
        "hard_subjects": hard,
        "reference_error_concentration": concentration,
    }


def run_agreement(a, loaded):
    """
    Agreement analysis for the runs given on the command line; prints a summary
    and writes results/agreement_<split>.json.
    """
    if len(loaded) < 2:
        raise SystemExit("--agreement needs at least two runs")
    out = agreement_analysis(loaded, a.reference)
    ref = out["reference"]
    print(f"agreement on {a.split}: {out['n_slices']} slices, reference = {ref}")
    print("slices wrong by k models:", out["n_slices_wrong_by_k"])
    for pair, v in out["pairs"].items():
        print(f"  {pair:32s} both wrong {v['both_wrong']:5d}  "
              f"jaccard {v['jaccard']:.3f}")
    for run, share in out["reference_errors_also_made_by"].items():
        print(f"  {share:.3f} of {ref}'s errors are also made by {run}")
    c = out["reference_error_concentration"]
    print(f"  worst {c['top_subjects']} subjects hold {c['share_of_errors']:.3f} "
          f"of {ref}'s errors; {c['subjects_over_half_wrong']} of "
          f"{c['n_subjects']} subjects have over half their slices wrong")
    with open(os.path.join(HERE, "results", f"agreement_{a.split}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved agreement json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True,
                   help="run names; window_only for the window baseline")
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="use test only for the final models (Phase 8)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--calibration", action="store_true",
                   help="also write calibration metrics and figures")
    p.add_argument("--reject", action="store_true",
                   help="also run the reject-option (risk-coverage) analysis")
    p.add_argument("--target-acc", type=float, default=0.90,
                   help="accuracy wanted on accepted slices")
    p.add_argument("--min-coverage", type=float, default=0.5,
                   help="smallest acceptable share decided automatically")
    p.add_argument("--agreement", action="store_true",
                   help="also compare which slices the runs get wrong")
    p.add_argument("--reference", default=None,
                   help="run whose errors are examined (default: last in --runs)")
    a = p.parse_args()

    results, loaded = {}, {}
    for run in a.runs:
        path = os.path.join(HERE, "results", f"preds_{run}.npz")
        results[run] = benchmark(path, a.split, a.n_boot)
        loaded[run] = load_predictions(path, a.split)
        m, ci = results[run]["metrics"], results[run]["ci95"]
        print(f"{run:14s} acc {m['accuracy']:.3f} "
              f"[{ci['accuracy'][0]:.3f}, {ci['accuracy'][1]:.3f}]  "
              f"macro-F1 {m['macro_f1']:.3f}  AUROC {m['auroc']:.3f}")

    out_path = os.path.join(HERE, "results", f"benchmark_{a.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_path}")

    if a.calibration:
        calibration = {}
        for run, data in loaded.items():
            calibration[run] = calibration_summary(data)
            calibration[run]["ece_ci95"] = ece_bootstrap_ci(data, a.n_boot)
            c = calibration[run]
            print(f"{run:14s} ECE {c['ece']:.3f} "
                  f"[{c['ece_ci95'][0]:.3f}, {c['ece_ci95'][1]:.3f}]  "
                  f"conf when right {c['mean_conf_correct']:.3f}, "
                  f"when wrong {c['mean_conf_wrong']:.3f}")
        os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
        with open(os.path.join(HERE, "results", f"calibration_{a.split}.json"), "w") as f:
            json.dump(calibration, f, indent=2)
        plot_reliability_diagram(
            loaded, os.path.join(HERE, "assets", f"reliability_{a.split}.png"))
        plot_confidence_histograms(
            loaded, os.path.join(HERE, "assets", f"confidence_hist_{a.split}.png"))
        print("saved calibration json and figures")

    if a.reject:
        run_reject_analysis(a, loaded)


if __name__ == "__main__":
    main()