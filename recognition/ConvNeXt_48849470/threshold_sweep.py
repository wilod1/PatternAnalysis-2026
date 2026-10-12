"""
Patient-level threshold sweep on saved predictions (no model, no GPU).

Usage:  python threshold_sweep.py results/preds/tiny_rc.npz
        python threshold_sweep.py results/preds/mini_a.npz

Uses the VAL split only. A patient is called AD when the mean AD probability over
their slices exceeds t. The threshold chosen here must be frozen before the test
set is used; the val accuracy at the chosen t is optimistic because t was tuned on
the same patients.
"""
import sys

import numpy as np


def auc(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return greater + 0.5 * ties


def main(path, split="val"):
    d = np.load(path, allow_pickle=True)
    probs = d[f"{split}_probs"]
    labels = d[f"{split}_label"]
    subjects = d[f"{split}_subject_id"]

    ids = np.unique(subjects)
    p_ad = np.array([probs[subjects == s][:, 1].mean() for s in ids])
    y = np.array([labels[subjects == s][0] for s in ids])

    print(f"{path} [{split}]  n={len(ids)} patients ({(y == 0).sum()} NC / {(y == 1).sum()} AD)")
    print(f"patient-level AUC: {auc(p_ad, y):.3f}\n")
    print("   t    acc   bal_acc   NC_rec   AD_rec")
    best = None
    for t in np.round(np.arange(0.20, 0.71, 0.05), 2):
        pred = (p_ad > t).astype(int)
        acc = (pred == y).mean()
        nc, ad = (pred[y == 0] == 0).mean(), (pred[y == 1] == 1).mean()
        bal = (nc + ad) / 2
        mark = ""
        if best is None or bal > best[1]:
            best = (t, bal)
        print(f" {t:.2f}  {acc:.3f}   {bal:.3f}    {nc:.3f}    {ad:.3f}{mark}")
    print(f"\nbest balanced accuracy at t={best[0]:.2f} ({best[1]:.3f})")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))