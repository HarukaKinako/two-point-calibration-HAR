"""
How much enrolment data does the calibration actually need?

The two-point calibration assumes a user contributes some labelled walking and
sitting windows. That is the price of the method, so it should be quoted: this
script varies the number of reference windows per subject and measures what the
accuracy gain does.

    n_cal in {2, 5, 10, 15, 20, all-but-5} windows per reference activity

Windows used for calibration are still removed from training and testing, so
the comparison is honest at every point on the curve. The baseline is the same
model without calibration, evaluated on the SAME surviving windows, so the two
curves are directly comparable at each n_cal.

At 20 s per window, n_cal = 5 is roughly 100 s of walking — the practical
question is whether the gain survives that far down.

Protocol: C(10,8)=45 subject-wise folds, RF 100 trees, 6-class metric
(reference activities removed from train and test) plus the binary task.

Output: Results/walk_calibration/calib_amount_results.csv
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
from calib_holdout_run import WALK_REF, STAT_REF, EPS

OUT_CSV = os.path.join(M.PARENT, "Results", "walk_calibration",
                       "calib_amount_results.csv")
ATTEMPTS = 10
N_CAL_LIST = [2, 5, 10, 15, 20, -1]        # -1 = all but 5 windows
MIN_EVAL = 5                                # always leave this many for scoring
NON_REF = ['bicycle', 'downstairs', 'kick', 'run', 'skate', 'upstairs']
SEED = 0


def split_n(y, subj, n_cal):
    """Reserve n_cal windows per reference activity per subject (-1 = all but 5)."""
    rng = np.random.RandomState(SEED)
    is_cal = np.zeros(len(y), dtype=bool)
    for s in sorted(set(subj)):
        for ref in (WALK_REF, STAT_REF):
            idx = np.where((subj == s) & np.isin(y, ref))[0]
            take = len(idx) - MIN_EVAL if n_cal < 0 else min(n_cal, len(idx) - MIN_EVAL)
            is_cal[rng.permutation(idx)[:take]] = True
    return is_cal


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


def rf_acc(Xtr, ytr, Xte, yte):
    accs = []
    for a in range(ATTEMPTS):
        rf = RandomForestClassifier(n_estimators=M.N_TREES, class_weight="balanced",
                                    n_jobs=1, random_state=a).fit(Xtr, ytr)
        accs.append(balanced_accuracy_score(yte, rf.predict(Xte)))
    return float(np.mean(accs))


def eval_fold(X, Xc, y, y_bin, subj, keep, non_ref, combo):
    in_tr = np.array([s in combo for s in subj])
    tr, te = in_tr & keep, (~in_tr) & keep
    tr6, te6 = tr & non_ref, te & non_ref
    held = ",".join(str(s) for s in sorted(set(subj) - set(combo)))
    rows = []
    for meth, Xv in [("raw", X), ("raw_2pt", Xc)]:
        rows.append((held, meth, "multiclass6",
                     rf_acc(Xv[tr6], y[tr6], Xv[te6], y[te6])))
        rows.append((held, meth, "pushpush",
                     rf_acc(Xv[tr], y_bin[tr], Xv[te], y_bin[te])))
    return rows


def main():
    rows = []
    if os.path.exists(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
    done = {(r["config"], r["n_cal"]) for r in rows}

    for name, _sd, _ch, feat_glob in M.CONFIGS:
        X, y, subj, fdim = M.load_features(feat_glob)
        y_bin = np.where(np.isin(y, ["kick", "skate"]), "pushpush", "other")
        non_ref = np.isin(y, NON_REF)
        combos = list(combinations(sorted(set(subj)), M.N_TRAIN))
        for n_cal in N_CAL_LIST:
            if (name, n_cal) in done:
                continue
            is_cal = split_n(y, subj, n_cal)
            keep = ~is_cal
            Xc = np.hstack([X, two_point(X, y, subj, is_cal)])
            per_subj = int(np.mean([  # windows reserved per subject, both refs
                (is_cal & (subj == s)).sum() for s in sorted(set(subj))]))
            t0 = time.time()
            folds = Parallel(n_jobs=M.N_JOBS)(
                delayed(eval_fold)(X, Xc, y, y_bin, subj, keep, non_ref, c)
                for c in combos)
            for fr in folds:
                for held, meth, task, acc in fr:
                    rows.append({"config": name, "n_cal": n_cal,
                                 "cal_windows_per_subject": per_subj,
                                 "method": meth, "task": task,
                                 "fold_heldout": held, "bal_acc": round(acc, 4)})
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            df = pd.DataFrame(rows)
            s = df[(df.config == name) & (df.n_cal == n_cal)]
            msg = []
            for task in ("multiclass6", "pushpush"):
                b = s[(s.method == "raw") & (s.task == task)].bal_acc.mean() * 100
                c = s[(s.method == "raw_2pt") & (s.task == task)].bal_acc.mean() * 100
                msg.append(f"{task[:6]} {b:.1f}->{c:.1f} ({c-b:+.2f})")
            label = "all" if n_cal < 0 else str(n_cal)
            print(f"  [{name}] n_cal={label:>3s} ({per_subj:3d} win/subj)  "
                  + "   ".join(msg) + f"   ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print("  CALIBRATION AMOUNT vs GAIN  (windows per reference activity; 20 s each)")
    print("=" * 88)
    for task in ("multiclass6", "pushpush"):
        print(f"\n  --- {task} ---")
        print(f"  {'config':11s}" + "".join(
            f"{('all' if n < 0 else n):>10}" for n in N_CAL_LIST))
        for cfg in [c[0] for c in M.CONFIGS]:
            line = f"  {cfg:11s}"
            for n in N_CAL_LIST:
                s = df[(df.config == cfg) & (df.n_cal == n) & (df.task == task)]
                if not len(s):
                    line += f"{'-':>10s}"; continue
                b = s[s.method == "raw"].bal_acc.mean()
                c = s[s.method == "raw_2pt"].bal_acc.mean()
                line += f"{(c-b)*100:+10.2f}"
            print(line)
    print(f"\n  Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
