"""
Per-fold results for paired statistical tests.

Runs, under the IDENTICAL protocol (subject-wise 8:2, C(10,8)=45 folds,
100 RF runs per fold, leak-free in-fold scaler+PCA):

  M1_feat_RF   : 211 features -> RF          (paper baseline, re-run locally;
                 no scaling — matches the paper, and RF is scale-invariant)
  M3_PCA_k     : 211 features -> scaler -> PCA(k) -> RF   for k in 25/50/100/200
                 (one exact PCA fit at k_max per fold, columns sliced — nested)

Saves ONE row per (config, method, task, fold) so that Wilcoxon signed-rank /
paired tests can be run across the 45 shared folds (perfold_stats.py).

Output : Results/PCA_final_comparison/perfold_results.csv   (checkpointed,
         resumable per config)
"""

import os, sys, time
import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pca_final_comparison as M   # thread caps, CONFIGS, PAPER, loaders

K_LIST  = [25, 50, 100, 200]
PERFOLD = os.path.join(M.OUT_DIR, "perfold_results.csv")


def rf_mean_acc(Xtr, ytr, Xte, yte, attempts):
    accs = []
    for a in range(attempts):
        rf = RandomForestClassifier(n_estimators=M.N_TREES, class_weight="balanced",
                                    n_jobs=1, random_state=a)
        rf.fit(Xtr, ytr)
        accs.append(balanced_accuracy_score(yte, rf.predict(Xte)))
    return float(np.mean(accs))


def eval_fold(X, y_multi, y_bin, subj, combo, ks, attempts):
    tr = np.array([s in combo for s in subj]); te = ~tr
    held = ",".join(str(s) for s in sorted(set(subj) - set(combo)))
    tasks = [("multiclass", y_multi), ("pushpush", y_bin)]
    rows = []

    # Method 1: features -> RF (baseline re-run)
    for task, y in tasks:
        rows.append((held, "M1_feat_RF", task,
                     rf_mean_acc(X[tr], y[tr], X[te], y[te], attempts)))

    # Method 3 at each k, nested exact PCA
    sc = StandardScaler().fit(X[tr])
    Xtr_s, Xte_s = sc.transform(X[tr]), sc.transform(X[te])
    kmax = min(max(ks), Xtr_s.shape[1]) #t-SNE　x 
    pca  = PCA(n_components=kmax, svd_solver="full").fit(Xtr_s)
    Ztr, Zte = pca.transform(Xtr_s), pca.transform(Xte_s)
    for k in ks:
        kk = min(k, kmax)
        for task, y in tasks:
            rows.append((held, f"M3_PCA_k{k}", task,
                         rf_mean_acc(Ztr[:, :kk], y[tr], Zte[:, :kk], y[te], attempts)))
    return rows


def run_config(name, feat_glob, all_rows, attempts):
    X, y_multi, subj, fdim = M.load_features(feat_glob)
    y_bin  = np.where(np.isin(y_multi, ["kick", "skate"]), "pushpush", "other")
    combos = list(combinations(sorted(set(subj)), M.N_TRAIN))
    t0 = time.time()
    folds = Parallel(n_jobs=M.N_JOBS)(
        delayed(eval_fold)(X, y_multi, y_bin, subj, c, K_LIST, attempts) for c in combos)
    for fold_rows in folds:
        for held, method, task, acc in fold_rows:
            all_rows.append({"config": name, "method": method, "task": task,
                             "fold_heldout": held, "bal_acc": round(acc, 4)})
    df = pd.DataFrame(all_rows)
    df.to_csv(PERFOLD, index=False)
    # quick aggregate printout
    sub = df[df.config == name]
    for method in ["M1_feat_RF"] + [f"M3_PCA_k{k}" for k in K_LIST]:
        mm = sub[(sub.method == method)]
        mc = mm[mm.task == "multiclass"].bal_acc
        pp = mm[mm.task == "pushpush"].bal_acc
        print(f"  [{name}] {method:11s} MC={mc.mean()*100:.1f}({mc.std()*100:.0f})  "
              f"PP={pp.mean()*100:.1f}({pp.std()*100:.0f})", flush=True)
    print(f"  [{name}] done in {time.time()-t0:.0f}s", flush=True)


def main(attempts=None):
    attempts = attempts or M.ATTEMPTS
    print(f"per-fold run | M1 + M3@{K_LIST} | attempts={attempts} | folds=45 | trees={M.N_TREES}\n",
          flush=True)
    all_rows, done = [], set()
    if os.path.exists(PERFOLD):   # resume completed configs
        prev = pd.read_csv(PERFOLD)
        need = {"M1_feat_RF"} | {f"M3_PCA_k{k}" for k in K_LIST}
        for cfg, grp in prev.groupby("config"):
            if need.issubset(set(grp.method)) and len(grp) == len(need) * 2 * 45:
                all_rows += grp.to_dict("records")
                done.add(cfg)
        if done:
            print(f"resuming — already complete: {sorted(done)}", flush=True)
    for name, _sdir, _ch, feat_glob in M.CONFIGS:
        if name in done:
            continue
        print(f"=== CONFIG: {name} ===", flush=True)
        run_config(name, feat_glob, all_rows, attempts)

    # final summary vs paper values
    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 70)
    print("  M1 re-run vs paper Table 2   (balanced accuracy %)")
    print("=" * 70)
    for name, *_ in M.CONFIGS:
        for task, pkey in [("multiclass", "multiclass"), ("pushpush", "pushpush")]:
            ours = df[(df.config == name) & (df.method == "M1_feat_RF") &
                      (df.task == task)].bal_acc
            print(f"  {name:11s} {task:11s} ours={ours.mean()*100:5.1f}({ours.std()*100:.0f})   "
                  f"paper={M.PAPER[name][pkey][0]}({M.PAPER[name][pkey][1]})")
    print(f"\nSaved per-fold data -> {PERFOLD}")


if __name__ == "__main__":
    main()
