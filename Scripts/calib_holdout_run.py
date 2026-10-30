"""
Personal calibration with HELD-OUT calibration windows (strict version).

Why this run exists
-------------------
In multi_reference_calibration.py the per-subject statistics (mu, sd of walk /
sit-stand) were computed from ALL of that subject's reference windows, and those
same windows were then also classified. A walk window therefore contributed to
the mu_walk used to normalise itself — a mild circularity that can inflate the
recall of the reference activities.

Dropping sit/stand/walk from the metric (as a quick check) is the opposite
error: at deployment time the system does NOT know which incoming window is a
walk, so classifying those activities is genuine work and excluding them
under-states the method.

This script removes the circularity instead of removing the classes:

    for each subject:
        split that subject's walk windows      -> CAL half / EVAL half
        split that subject's sit+stand windows -> CAL half / EVAL half
        calibration statistics come ONLY from the CAL half
        the CAL windows are then dropped from both train and test sets
    -> every remaining window (all 9 classes) is scored fairly

This mirrors deployment: a short enrolment recording is used to calibrate, and
everything afterwards is classified. CAL_FRACTION also lets us ask how much
enrolment data is actually needed.

Protocol otherwise unchanged: C(10,8)=45 subject-wise folds, RF 100 trees,
ATTEMPTS seeds, balanced accuracy over all 9 classes (and the binary task).

Output: Results/walk_calibration/calib_holdout_results.csv
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

OUT_DIR = os.path.join(M.PARENT, "Results", "walk_calibration")
OUT_CSV = os.path.join(OUT_DIR, "calib_holdout_results.csv")
ATTEMPTS = 20
EPS = 1e-8
CAL_FRACTION = 0.5          # half of the reference windows are used for calibration
CAL_SEED = 0
WALK_REF = ["walk"]
STAT_REF = ["sit", "stand"]
VIEWS = ["M1_raw", "raw_zwalk_zsit", "raw_2pt", "raw_all"]


def split_calibration(y, subj, frac=CAL_FRACTION, seed=CAL_SEED):
    """Mark, per subject, which reference windows are reserved for calibration."""
    rng = np.random.RandomState(seed)
    is_cal = np.zeros(len(y), dtype=bool)
    for s in sorted(set(subj)):
        for ref in (WALK_REF, STAT_REF):
            idx = np.where((subj == s) & np.isin(y, ref))[0]
            n = max(5, int(round(len(idx) * frac)))     # keep at least 5 windows
            n = min(n, len(idx) - 5)                    # and leave >=5 for evaluation
            is_cal[rng.permutation(idx)[:n]] = True
    return is_cal


def build_views(X, y, subj, is_cal):
    """Calibrated views; statistics come only from the reserved CAL windows."""
    z_walk = np.zeros_like(X)
    z_sit  = np.zeros_like(X)
    two_pt = np.zeros_like(X)
    for s in sorted(set(subj)):
        m  = subj == s
        cw = m & is_cal & np.isin(y, WALK_REF)
        cs = m & is_cal & np.isin(y, STAT_REF)
        mu_w, sd_w = X[cw].mean(axis=0), X[cw].std(axis=0)
        mu_s, sd_s = X[cs].mean(axis=0), X[cs].std(axis=0)
        sd_w = np.where(sd_w < EPS, 1.0, sd_w)
        sd_s = np.where(sd_s < EPS, 1.0, sd_s)
        span = mu_w - mu_s
        span = np.where(np.abs(span) < EPS, 1.0, span)
        z_walk[m] = (X[m] - mu_w) / sd_w
        z_sit[m]  = (X[m] - mu_s) / sd_s
        two_pt[m] = (X[m] - mu_s) / span
    return {"M1_raw": X,
            "raw_zwalk_zsit": np.hstack([X, z_walk, z_sit]),
            "raw_2pt":        np.hstack([X, two_pt]),
            "raw_all":        np.hstack([X, z_walk, z_sit, two_pt])}


def rf_acc(Xtr, ytr, Xte, yte):
    accs = []
    for a in range(ATTEMPTS):
        rf = RandomForestClassifier(n_estimators=M.N_TREES, class_weight="balanced",
                                    n_jobs=1, random_state=a).fit(Xtr, ytr)
        accs.append(balanced_accuracy_score(yte, rf.predict(Xte)))
    return float(np.mean(accs))


def eval_fold(views, y_multi, y_bin, subj, keep, combo):
    """keep = windows NOT reserved for calibration; only these are trained/scored."""
    tr = np.array([s in combo for s in subj]) & keep
    te = (~np.array([s in combo for s in subj])) & keep
    held = ",".join(str(s) for s in sorted(set(subj) - set(combo)))
    rows = []
    for meth, Xv in views.items():
        for task, y in [("multiclass", y_multi), ("pushpush", y_bin)]:
            rows.append((held, meth, task, rf_acc(Xv[tr], y[tr], Xv[te], y[te])))
    return rows


def main():
    rows = []
    if os.path.exists(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
    done = {r["config"] for r in rows}

    for name, _sd, _ch, feat_glob in M.CONFIGS:
        if name in done:
            print(f"=== {name}: already done ===", flush=True); continue
        X, y_multi, subj, fdim = M.load_features(feat_glob)
        y_bin = np.where(np.isin(y_multi, ["kick", "skate"]), "pushpush", "other")
        is_cal = split_calibration(y_multi, subj)
        keep = ~is_cal
        views = build_views(X, y_multi, subj, is_cal)
        combos = list(combinations(sorted(set(subj)), M.N_TRAIN))
        print(f"=== {name} ===  calibration windows: {is_cal.sum()} "
              f"({is_cal.sum()/len(X)*100:.0f}%), evaluated: {keep.sum()}", flush=True)
        t0 = time.time()
        folds = Parallel(n_jobs=M.N_JOBS)(
            delayed(eval_fold)(views, y_multi, y_bin, subj, keep, c) for c in combos)
        for fr in folds:
            for held, meth, task, acc in fr:
                rows.append({"config": name, "method": meth, "task": task,
                             "fold_heldout": held, "bal_acc": round(acc, 4)})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        df = pd.DataFrame(rows); base = {}
        for meth in VIEWS:
            s = df[(df.config == name) & (df.method == meth)]
            mc = s[s.task == "multiclass"].bal_acc; pp = s[s.task == "pushpush"].bal_acc
            if meth == "M1_raw":
                base = {"mc": mc.mean(), "pp": pp.mean()}; d = ""
            else:
                d = (f"   Δ MC {(mc.mean()-base['mc'])*100:+5.1f}"
                     f"  PP {(pp.mean()-base['pp'])*100:+5.1f}")
            print(f"  [{name}] {meth:16s} MC={mc.mean()*100:5.1f}({mc.std()*100:.0f})  "
                  f"PP={pp.mean()*100:5.1f}({pp.std()*100:.0f}){d}", flush=True)
        print(f"  [{name}] done in {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print("  STRICT CALIBRATION (calibration windows held out) — balanced accuracy %")
    print("  All 9 classes scored; calibration windows excluded from train and test.")
    print("=" * 92)
    print(f"  {'config':11s} {'task':11s}" + "".join(f"{m:>17s}" for m in VIEWS))
    for cfg in [c[0] for c in M.CONFIGS]:
        for task in ("multiclass", "pushpush"):
            b = df[(df.config==cfg)&(df.method=="M1_raw")&(df.task==task)].bal_acc.mean()
            line = f"  {cfg:11s} {task:11s}"
            for meth in VIEWS:
                v = df[(df.config==cfg)&(df.method==meth)&(df.task==task)].bal_acc
                if not len(v):
                    line += f"{'-':>17s}"; continue
                v = v.mean()
                line += (f"{v*100:11.1f}      " if meth == "M1_raw"
                         else f"{v*100:11.1f}({(v-b)*100:+4.1f})")
            print(line)
    print(f"\n  Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
