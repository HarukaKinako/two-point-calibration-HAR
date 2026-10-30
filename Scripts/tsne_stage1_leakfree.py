"""
Stage 1 — LEAK-FREE non-linear embeddings -> RF.

Stage 0 showed that even a leaky, all-subject t-SNE map loses to PCA in 7 of 8
cases; the single exception was hip / binary (97.3%). Stage 1 asks two things
under the usual honest protocol (embedding fitted on the 8 training subjects,
held-out subjects only transformed; C(10,8)=45 folds; RF 100 trees):

  A) Is the hip / binary result real, or an artifact of the leak?
     -> openTSNE, which can embed unseen points into an existing map.

  B) Does supervision fix the "islands are subjects" problem?
     -> UMAP (unsupervised) vs supervised UMAP (fit with training labels),
        which have transform() by construction.

Methods evaluated (all leak-free):
  tsneOF_d{2}     openTSNE fit on train, .transform() on test
  umap_d{2,10}    UMAP unsupervised
  sumap_d{2,10}   UMAP supervised (y passed to fit)

Scope:
  --scope hip   : hip_Acc only, binary + multiclass          (task A, fast)
  --scope all   : all four configs, both tasks               (task B)

Output: Results/tsne/stage1_leakfree.csv  (one row per config/method/task/fold)
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pca_final_comparison as M

OUT_CSV  = os.path.join(M.PARENT, "Results", "tsne", "stage1_leakfree.csv")
ATTEMPTS = 20
UMAP_DIMS = [2, 10]


def embed_opentsne(Xtr, Xte, dim=2):
    from openTSNE import TSNE as OTSNE
    emb = OTSNE(n_components=dim, perplexity=30, initialization="pca",
                random_state=42, n_jobs=1, verbose=False).fit(Xtr)
    return np.asarray(emb), np.asarray(emb.transform(Xte))


def embed_umap(Xtr, Xte, dim, ytr=None):
    import umap
    red = umap.UMAP(n_components=dim, n_neighbors=15, min_dist=0.1,
                    random_state=42, verbose=False)
    if ytr is not None:
        # UMAP needs numeric labels; map the string classes to integer codes.
        # Codes come from the TRAINING labels only — no test information.
        classes = sorted(set(ytr))
        ytr = np.array([classes.index(v) for v in ytr], dtype=np.int64)
    Ztr = red.fit_transform(Xtr, y=ytr)      # y=None -> unsupervised
    return np.asarray(Ztr), np.asarray(red.transform(Xte))


def rf_acc(Ztr, ytr, Zte, yte):
    accs = []
    for a in range(ATTEMPTS):
        rf = RandomForestClassifier(n_estimators=M.N_TREES, class_weight="balanced",
                                    n_jobs=1, random_state=a).fit(Ztr, ytr)
        accs.append(balanced_accuracy_score(yte, rf.predict(Zte)))
    return float(np.mean(accs))


def eval_fold(X, y_multi, y_bin, subj, combo, methods):
    """One subject split. Embeddings are fitted on TRAIN ONLY."""
    tr = np.array([s in combo for s in subj]); te = ~tr
    held = ",".join(str(s) for s in sorted(set(subj) - set(combo)))
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    tasks = [("multiclass", y_multi), ("pushpush", y_bin)]
    rows = []

    for meth in methods:
        if meth.startswith("tsneOF"):
            dim = int(meth.split("_d")[1])
            Ztr, Zte = embed_opentsne(Xtr, Xte, dim)
            for task, y in tasks:
                rows.append((held, meth, task, rf_acc(Ztr, y[tr], Zte, y[te])))
        elif meth.startswith("umap"):
            dim = int(meth.split("_d")[1])
            Ztr, Zte = embed_umap(Xtr, Xte, dim, ytr=None)
            for task, y in tasks:
                rows.append((held, meth, task, rf_acc(Ztr, y[tr], Zte, y[te])))
        elif meth.startswith("sumap"):
            # supervised: the embedding itself depends on the labels, so it must
            # be refitted per task (still train-only -> no leakage)
            dim = int(meth.split("_d")[1])
            for task, y in tasks:
                Ztr, Zte = embed_umap(Xtr, Xte, dim, ytr=y[tr])
                rows.append((held, meth, task, rf_acc(Ztr, y[tr], Zte, y[te])))
    return rows


def run(configs, methods):
    rows = []
    if os.path.exists(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
    done = {(r["config"], r["method"]) for r in rows}

    for name, _sd, _ch, feat_glob in configs:
        X, y_multi, subj, fdim = M.load_features(feat_glob)
        y_bin = np.where(np.isin(y_multi, ["kick", "skate"]), "pushpush", "other")
        combos = list(combinations(sorted(set(subj)), M.N_TRAIN))
        todo = [m for m in methods if (name, m) not in done]
        if not todo:
            print(f"=== {name}: already complete ===", flush=True); continue
        print(f"=== {name} === methods: {todo}", flush=True)
        t0 = time.time()
        folds = Parallel(n_jobs=M.N_JOBS)(
            delayed(eval_fold)(X, y_multi, y_bin, subj, c, todo) for c in combos)
        for fr in folds:
            for held, meth, task, acc in fr:
                rows.append({"config": name, "method": meth, "task": task,
                             "fold_heldout": held, "bal_acc": round(acc, 4)})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        df = pd.DataFrame(rows)
        for meth in todo:
            s = df[(df.config == name) & (df.method == meth)]
            mc = s[s.task == "multiclass"].bal_acc; pp = s[s.task == "pushpush"].bal_acc
            print(f"  [{name}] {meth:12s} MC={mc.mean()*100:5.1f}({mc.std()*100:.0f})  "
                  f"PP={pp.mean()*100:5.1f}({pp.std()*100:.0f})", flush=True)
        print(f"  [{name}] done in {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def summary(df):
    ref = pd.read_csv(os.path.join(M.OUT_DIR, "perfold_results.csv"))
    leaky_p = os.path.join(M.PARENT, "Results", "tsne", "stage0_upperbound.csv")
    leaky = pd.read_csv(leaky_p) if os.path.exists(leaky_p) else None
    print("\n" + "=" * 92)
    print("  STAGE 1 — LEAK-FREE embeddings vs references")
    print("=" * 92)
    hdr = f"  {'config':10s} {'task':11s}"
    meths = sorted(df.method.unique())
    for m in meths:
        hdr += f" {m:>11s}"
    hdr += f" {'PCAbest':>8s} {'M1':>7s} {'tSNE2D*':>8s}"
    print(hdr)
    for cfg in df.config.unique():
        for task in ("multiclass", "pushpush"):
            line = f"  {cfg:10s} {task:11s}"
            for m in meths:
                v = df[(df.config == cfg) & (df.method == m) & (df.task == task)].bal_acc
                line += f" {v.mean()*100:10.1f}%" if len(v) else f" {'-':>11s}"
            r = ref[(ref.config == cfg) & (ref.task == task)]
            pbest = (r[r.method.str.startswith("M3_PCA_k")]
                     .groupby("method").bal_acc.mean().max() * 100)
            m1 = r[r.method == "M1_feat_RF"].bal_acc.mean() * 100
            line += f" {pbest:7.1f}% {m1:6.1f}%"
            if leaky is not None:
                lv = leaky[(leaky.config == cfg) & (leaky.method == "tsne2d_LEAKY")
                           & (leaky.task == task)].bal_acc
                line += f" {lv.mean()*100:7.1f}%" if len(lv) else f" {'-':>8s}"
            print(line)
    print("\n  * = leaky upper bound from Stage 0")
    print(f"  Saved -> {OUT_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["hip", "all"], default="hip")
    a = ap.parse_args()
    if a.scope == "hip":
        cfgs = [M.CONFIGS[0]]                       # hip_Acc
        meths = ["tsneOF_d2"] + [f"umap_d{d}" for d in UMAP_DIMS] \
                              + [f"sumap_d{d}" for d in UMAP_DIMS]
    else:
        cfgs = M.CONFIGS
        meths = [f"umap_d{d}" for d in UMAP_DIMS] + [f"sumap_d{d}" for d in UMAP_DIMS]
    summary(run(cfgs, meths))
