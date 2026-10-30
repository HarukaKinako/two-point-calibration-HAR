"""
Do we need the explicit calibration, or is the reference INFORMATION enough?

Two things are tested here.

(1) A natural objection: "the gain is obvious, the model was handed walk/sit".
    Answer: report a 6-class problem in which the reference activities
    (sit / stand / walk) are removed from BOTH training and testing. Only the
    six activities that actually have to be recognised remain
    (bicycle, downstairs, kick, run, skate, upstairs). Nothing about that
    metric can be inflated by the references being easy.

(2) Instead of computing the calibration ourselves, simply append
    each subject's reference STATISTICS (mean / sd / min of their own
    walk and sit windows) to the raw features and let the Random Forest work
    out the comparison by itself.

    Why this is a real question: an RF splits one feature at a time on an
    axis-parallel threshold, so it cannot form x - mu_walk. What it CAN do is
    branch on mu_walk first and then apply a different threshold to x in each
    branch — i.e. learn person-adaptive thresholds. So the experiment separates
    "the information in the references" from "the normalising operation".

Views:
  raw            211 features                                  (baseline)
  raw_stats      raw ++ per-subject reference statistics                 
  raw_2pt        raw ++ (x - mu_sit)/(mu_walk - mu_sit)        (explicit)
  raw_2pt_stats  raw ++ two-point ++ reference statistics      (both)

Protocol: identical to calib_holdout_run.py — half of each subject's walk and
sit/stand windows are reserved for computing the statistics and dropped from
training and testing. C(10,8)=45 folds, RF 100 trees.

Tasks: multiclass9 (all classes), multiclass6 (non-reference only), pushpush.

Output: Results/walk_calibration/refstats_results.csv
"""

import os, sys, time
import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pca_final_comparison as M
from calib_holdout_run import split_calibration, WALK_REF, STAT_REF, EPS

OUT_CSV = os.path.join(M.PARENT, "Results", "walk_calibration", "refstats_results.csv")
ATTEMPTS = 10                      # exploratory; winner can be re-run at 100
NON_REF = ['bicycle', 'downstairs', 'kick', 'run', 'skate', 'upstairs']
VIEWS = ["raw", "raw_stats", "raw_2pt", "raw_2pt_stats"]


def build_views(X, y, subj, is_cal):
    """Reference statistics and the explicit two-point calibration."""
    n, d = X.shape
    stats  = np.zeros((n, 6 * d), dtype=X.dtype)   # mean/sd/min for walk and sit
    two_pt = np.zeros_like(X)
    for s in sorted(set(subj)):
        m  = subj == s
        cw = m & is_cal & np.isin(y, WALK_REF)
        cs = m & is_cal & np.isin(y, STAT_REF)
        Xw, Xs = X[cw], X[cs]
        mu_w, sd_w, mn_w = Xw.mean(0), Xw.std(0), Xw.min(0)
        mu_s, sd_s, mn_s = Xs.mean(0), Xs.std(0), Xs.min(0)
        # the person's own reference profile, repeated on every one of their rows
        stats[m] = np.hstack([mu_w, sd_w, mn_w, mu_s, sd_s, mn_s])
        span = mu_w - mu_s
        span = np.where(np.abs(span) < EPS, 1.0, span)
        two_pt[m] = (X[m] - mu_s) / span
    return {"raw":           X,
            "raw_stats":     np.hstack([X, stats]),
            "raw_2pt":       np.hstack([X, two_pt]),
            "raw_2pt_stats": np.hstack([X, two_pt, stats])}


def rf_acc(Xtr, ytr, Xte, yte):
    accs = []
    for a in range(ATTEMPTS):
        rf = RandomForestClassifier(n_estimators=M.N_TREES, class_weight="balanced",
                                    n_jobs=1, random_state=a).fit(Xtr, ytr)
        accs.append(balanced_accuracy_score(yte, rf.predict(Xte)))
    return float(np.mean(accs))


def eval_fold(views, y9, y_bin, subj, keep, non_ref_mask, combo):
    in_train = np.array([s in combo for s in subj])
    tr, te = in_train & keep, (~in_train) & keep
    tr6, te6 = tr & non_ref_mask, te & non_ref_mask      # 6-class problem
    held = ",".join(str(s) for s in sorted(set(subj) - set(combo)))
    rows = []
    for meth, Xv in views.items():
        rows.append((held, meth, "multiclass9", rf_acc(Xv[tr], y9[tr], Xv[te], y9[te])))
        rows.append((held, meth, "multiclass6", rf_acc(Xv[tr6], y9[tr6], Xv[te6], y9[te6])))
        rows.append((held, meth, "pushpush",    rf_acc(Xv[tr], y_bin[tr], Xv[te], y_bin[te])))
    return rows


def main():
    rows = []
    if os.path.exists(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
    done = {r["config"] for r in rows}

    for name, _sd, _ch, feat_glob in M.CONFIGS:
        if name in done:
            print(f"=== {name}: already done ===", flush=True); continue
        X, y9, subj, fdim = M.load_features(feat_glob)
        y_bin = np.where(np.isin(y9, ["kick", "skate"]), "pushpush", "other")
        is_cal = split_calibration(y9, subj)
        keep = ~is_cal
        non_ref = np.isin(y9, NON_REF)
        views = build_views(X, y9, subj, is_cal)
        combos = list(combinations(sorted(set(subj)), M.N_TRAIN))
        print(f"=== {name} === dims {[v.shape[1] for v in views.values()]}  "
              f"6-class windows: {(non_ref & keep).sum()}", flush=True)
        t0 = time.time()
        folds = Parallel(n_jobs=M.N_JOBS)(
            delayed(eval_fold)(views, y9, y_bin, subj, keep, non_ref, c) for c in combos)
        for fr in folds:
            for held, meth, task, acc in fr:
                rows.append({"config": name, "method": meth, "task": task,
                             "fold_heldout": held, "bal_acc": round(acc, 4)})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        df = pd.DataFrame(rows); base = {}
        for meth in VIEWS:
            s = df[(df.config == name) & (df.method == meth)]
            v = {t: s[s.task == t].bal_acc.mean() * 100 for t in
                 ("multiclass9", "multiclass6", "pushpush")}
            if meth == "raw":
                base = dict(v); d = ""
            else:
                d = ("   Δ " + "  ".join(f"{t[-1] if t!='pushpush' else 'B'}"
                     f"{v[t]-base[t]:+5.2f}" for t in v))
            print(f"  [{name}] {meth:14s} 9cls={v['multiclass9']:5.1f}  "
                  f"6cls={v['multiclass6']:5.1f}  bin={v['pushpush']:5.1f}{d}", flush=True)
        print(f"  [{name}] done in {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("  REFERENCE STATISTICS vs EXPLICIT CALIBRATION")
    print("  multiclass6 = sit/stand/walk removed from train AND test (the fair metric)")
    print("=" * 100)
    for task in ("multiclass6", "multiclass9", "pushpush"):
        print(f"\n  --- {task} ---")
        print(f"  {'config':11s}" + "".join(f"{m:>16s}" for m in VIEWS))
        for cfg in [c[0] for c in M.CONFIGS]:
            b = df[(df.config==cfg)&(df.method=="raw")&(df.task==task)].bal_acc.mean()
            line = f"  {cfg:11s}"
            for meth in VIEWS:
                v = df[(df.config==cfg)&(df.method==meth)&(df.task==task)].bal_acc
                if not len(v): line += f"{'-':>16s}"; continue
                v = v.mean()
                line += (f"{v*100:10.1f}      " if meth == "raw"
                         else f"{v*100:10.1f}({(v-b)*100:+4.1f})")
            print(line)
    print(f"\n  Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
