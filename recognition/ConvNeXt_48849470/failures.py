"""
Failure cases for one model. Reads exported predictions, picks the most
confidently wrong slices (one per subject), split by true class and by whether
the other models also get the slice wrong, then saves a panel figure and a JSON
list of the chosen cases.
"""
import argparse
import json
import os

import numpy as np

from metrics import CLASS_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("shared", "specific")


def load_run(run, split):
    """
    Predictions plus the record fields needed to find each image again.
    """
    z = np.load(os.path.join(HERE, "results", f"preds_{run}.npz"))
    p = z[f"{split}_probs"]
    conf = np.maximum(p, 1 - p) if p.ndim == 1 else p.max(axis=1)
    return {
        "preds": z[f"{split}_preds"],
        "labels": z[f"{split}_label"],
        "conf": conf,
        "subjects": z[f"{split}_subject_id"],
        "scans": z[f"{split}_scan_id"],
        "slices": z[f"{split}_slice_index"],
        "paths": z[f"{split}_path"],
    }


def others_wrong_count(labels, others_preds):
    """
    For each slice, how many of the other models predict the wrong class.
    """
    if not others_preds:
        return np.zeros(len(labels), dtype=int)
    return np.sum([p != labels for p in others_preds], axis=0)


def select_failures(mask, conf, subjects, n):
    """
    Up to n indices where mask is true: highest confidence first, at most
    one slice per subject.
    """
    idx = np.flatnonzero(mask)
    order = idx[np.argsort(-conf[idx], kind="stable")]
    chosen, seen = [], set()
    for i in order:
        if subjects[i] in seen:
            continue
        chosen.append(int(i))
        seen.add(subjects[i])
        if len(chosen) == n:
            break
    return chosen


def error_summary(run):
    """
    Error count overall and by true class.
    """
    labels = run["labels"]
    wrong = run["preds"] != labels
    return {
        "n_errors": int(wrong.sum()),
        "by_true_class": {
            int(c): {"errors": int((wrong & (labels == c)).sum()),
                     "n": int((labels == c).sum())}
            for c in (0, 1)
        },
    }


def pick_cases(run, others, n_each):
    """
    Candidate failure cases for run: n_each per (kind, true class) group.
    """
    if not others:
        raise ValueError("need at least one other model to split the failures")
    labels = run["labels"]
    wrong = run["preds"] != labels
    ow = others_wrong_count(labels, [o["preds"] for o in others])
    masks = {"shared": ow == len(others), "specific": ow == 0}
    cases = []
    for kind in KINDS:
        for cls in (0, 1):
            group = wrong & masks[kind] & (labels == cls)
            for i in select_failures(group, run["conf"], run["subjects"], n_each):
                cases.append({
                    "group": kind,
                    "true": int(labels[i]),
                    "pred": int(run["preds"][i]),
                    "conf": float(run["conf"][i]),
                    "subject": str(run["subjects"][i]),
                    "scan": str(run["scans"][i]),
                    "slice": int(run["slices"][i]),
                    "others_wrong": int(ow[i]),
                    "path": str(run["paths"][i]),
                    "index": int(i),
                })
    return cases


def plot_cases(cases, n_each, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = [(k, c) for k in KINDS for c in (0, 1)]
    labels = {"shared": "all models wrong", "specific": "only this model wrong"}
    fig, axes = plt.subplots(len(rows), n_each, squeeze=False,
                             figsize=(2.3 * n_each, 2.7 * len(rows)))
    fig.patch.set_facecolor("#fcfcfb")
    for r, (kind, cls) in enumerate(rows):
        group = [c for c in cases if c["group"] == kind and c["true"] == cls]
        for j in range(n_each):
            ax = axes[r][j]
            ax.axis("off")
            if j == 0:
                ax.text(-0.06, 0.5,
                        f"{labels[kind]}\ntrue {str(CLASS_NAMES[cls]).upper()}",
                        transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=8, color="#52514e")
            if j >= len(group):
                continue
            c = group[j]
            ax.imshow(Image.open(c["path"]).convert("L"), cmap="gray")
            ax.set_title(
                f"{str(CLASS_NAMES[c['true']]).upper()} as "
                f"{str(CLASS_NAMES[c['pred']]).upper()}, conf {c['conf']:.2f}\n"
                f"{c['subject']} slice {c['slice']}", fontsize=7, color="#0b0b0b")
    fig.suptitle(title, fontsize=10, color="#0b0b0b")
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--others", nargs="+",
                   default=["cnn_cos", "resnet_cos", "window_only"])
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--n-each", type=int, default=3)
    a = p.parse_args()

    run = load_run(a.run, a.split)
    others = [load_run(o, a.split) for o in a.others]
    for o in others:
        if not np.array_equal(o["labels"], run["labels"]):
            raise SystemExit("prediction files are not from the same split")
    summary = error_summary(run)
    cases = pick_cases(run, others, a.n_each)

    print(f"{a.run} on {a.split}: {summary['n_errors']} errors")
    for c, v in summary["by_true_class"].items():
        print(f"  true {str(CLASS_NAMES[c]).upper()}: {v['errors']} of {v['n']} wrong")
    print(f"{len(cases)} candidate cases:")
    for c in cases:
        print(f"  {c['group']:8s} {str(CLASS_NAMES[c['true']]).upper()} as "
              f"{str(CLASS_NAMES[c['pred']]).upper()}  conf {c['conf']:.3f}  "
              f"{c['subject']}  slice {c['slice']}  "
              f"others wrong {c['others_wrong']}/{len(others)}")

    out = {"run": a.run, "split": a.split, "others": a.others,
           "summary": summary, "cases": cases}
    with open(os.path.join(HERE, "results",
                           f"failures_{a.run}_{a.split}.json"), "w") as f:
        json.dump(out, f, indent=2)
    plot_cases(cases, a.n_each,
               os.path.join(HERE, "assets", f"failures_{a.run}_{a.split}.png"),
               f"{a.run}: confident failures ({a.split})")
    print("saved failures json and figure")


if __name__ == "__main__":
    main()