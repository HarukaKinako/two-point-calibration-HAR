"""
Which calibrated features does the forest actually use?

Reconnaissance for selective calibration. The two-point view doubles the input
(raw ++ calibrated); this script asks, per fold, how much of the forest's total
impurity reduction goes to the calibrated half, and which individual features
those are.

If only a small subset of the calibrated features carries the weight, then
calibrating everything is mostly adding noise, and a selective version should
help — especially on the configurations where calibration currently does
nothing (wrist) or hurts (binary task).

Protocol matches calib_holdout_run.py: calibration statistics come from the
reserved half of each subject's walk / sit windows, and those windows are
excluded from training and testing.

Outputs (Results/walk_calibration/importance/):
  importance_<config>_<task>.csv   per-feature mean importance, raw vs calibrated
  importance_summary.txt           share of importance held by the calibrated half
  fig_importance_share.png         that share, per config and task
"""

import os, sys, time
import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pca_final_comparison as M
from calib_holdout_run import split_calibration, WALK_REF, STAT_REF, EPS

OUT_DIR = os.path.join(M.PARENT, "Results", "walk_calibration", "importance")
os.makedirs(OUT_DIR, exist_ok=True)
ATTEMPTS = 5          # importances are stable; 5 seeds x 45 folds is plenty
NON_REF = ['bicycle', 'downstairs', 'kick', 'run', 'skate', 'upstairs']


def two_point(X, y, subj, is_cal):
    out = np.zeros_like(X)
    for s in sorted(set(subj)):
        m = subj == s
        mu_w = X[m & is_cal & np.isin(y, WALK_REF)].mean(0)
        mu_s = X[m & is_cal & np.isin(y, STAT_REF)].mean(0)
        span = mu_w - mu_s
        span = np.where(np.abs(span) < EPS, 1.0, span)
        out[m] = (X[m] - mu_s) / span
    return out


def fold_importance(Xv, y, subj, keep, mask_task, combo):
    in_tr = np.array([s in combo for s in subj])
    tr = in_tr & keep & mask_task
    imps = []
    for a in range(ATTEMPTS):
        rf = RandomForestClassifier(n_estimators=M.N_TREES, class_weight="balanced",
                                    n_jobs=1, random_state=a).fit(Xv[tr], y[tr])
        imps.append(rf.feature_importances_)
    return np.mean(imps, axis=0)


def main():
    lines = ["=" * 78,
             " Share of Random-Forest importance held by the CALIBRATED half",
             " (input = raw ++ two-point calibrated; 50% would mean 'used equally')",
             "=" * 78,
             f"{'config':11s} {'task':12s} {'calib share':>12s} {'top-20 calib':>13s}"]
    shares = {}

    for name, _sd, _ch, feat_glob in M.CONFIGS:
        X, y, subj, fdim = M.load_features(feat_glob)
        cols = pd.read_csv(sorted(__import__("glob").glob(os.path.join(
            M.PARENT, f"FeatureDataFrames_{M.WINSIZE}", feat_glob)), reverse=True)[0],
            index_col=0, nrows=1).columns[:fdim].tolist()
        is_cal = split_calibration(y, subj)
        keep = ~is_cal
        Xv = np.hstack([X, two_point(X, y, subj, is_cal)])
        names = [f"raw::{c}" for c in cols] + [f"cal::{c}" for c in cols]
        combos = list(combinations(sorted(set(subj)), M.N_TRAIN))

        for task, mask in [("multiclass6", np.isin(y, NON_REF)),
                           ("pushpush", np.ones(len(y), bool))]:
            yy = y if task == "multiclass6" else np.where(
                np.isin(y, ["kick", "skate"]), "pushpush", "other")
            t0 = time.time()
            imps = Parallel(n_jobs=M.N_JOBS)(
                delayed(fold_importance)(Xv, yy, subj, keep, mask, c) for c in combos)
            imp = np.mean(imps, axis=0)
            share = imp[fdim:].sum()
            top20 = np.argsort(imp)[::-1][:20]
            n_cal_top = int((top20 >= fdim).sum())
            shares[(name, task)] = (share, n_cal_top)

            pd.DataFrame({"feature": names, "importance": imp,
                          "half": ["raw"] * fdim + ["calibrated"] * fdim}) \
              .sort_values("importance", ascending=False) \
              .to_csv(os.path.join(OUT_DIR, f"importance_{name}_{task}.csv"), index=False)

            lines.append(f"{name:11s} {task:12s} {share*100:11.1f}% {n_cal_top:9d}/20")
            print(f"  [{name}] {task:12s} calibrated share {share*100:.1f}%  "
                  f"top-20: {n_cal_top}/20 calibrated  ({time.time()-t0:.0f}s)", flush=True)
            top = np.argsort(imp)[::-1][:8]
            print("     top-8: " + ", ".join(names[i].replace("::", " ") for i in top),
                  flush=True)

    # figure
    cfgs = [c[0] for c in M.CONFIGS]
    x = np.arange(len(cfgs)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    for off, task, col in [(-w/2, "multiclass6", "#2a78d6"), (w/2, "pushpush", "#e34948")]:
        vals = [shares[(c, task)][0] * 100 for c in cfgs]
        ax.bar(x + off, vals, w, label=task, color=col)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.6, f"{v:.0f}%", ha="center", fontsize=8)
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.text(len(cfgs) - 0.4, 51, "equal use (50%)", fontsize=7.5, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(cfgs, fontsize=9)
    ax.set_ylabel("Share of total importance held by\nthe calibrated features (%)", fontsize=9)
    ax.set_title("How much does the forest rely on the calibrated half?", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_importance_share.png"), dpi=190)

    txt = "\n".join(lines)
    open(os.path.join(OUT_DIR, "importance_summary.txt"), "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
