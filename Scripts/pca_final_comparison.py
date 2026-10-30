"""
Final 3-method comparison for skateboard/kickboard IMU recognition.

Method 1 : 211 features -> RF                (NOT run here; quoted from the paper, Table 2)
Method 2 : raw waveform -> flatten -> PCA -> RF
Method 3 : 211 features -> PCA -> RF

Design notes (compared with an earlier version of this analysis):
  * PCA (and StandardScaler) are fit INSIDE each CV fold on the training subjects
    only  -> no data leakage  (original fit PCA on all 10 subjects at once).
  * Standardization added before PCA (raw acc ~±2 g vs gyro ~±500 deg/s differ ~100x;
    without scaling PCA is dominated by gyro).  Measured to help ~+5 pts.
  * Raw waveform generated cleanly for ALL sensor configs here (the original
    save_data(acc_only=True) crashed on raw data -> only hip was ever produced).

Protocol matches the paper:  subject-wise 8:2 split, C(10,8)=45 folds,
100 independent RF runs averaged per fold.  Balanced accuracy reported.
"""

import os, sys, glob, time
# Limit BLAS/OpenMP threads to 1 so joblib's 10 process-workers don't oversubscribe
# the 10 cores (each PCA SVD would otherwise spawn its own thread pool -> ~100 threads).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
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

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from data_preparation import data_prep2, get_data_files, get_label, get_subject_label  # noqa

# ── Config ────────────────────────────────────────────────────────────────────
WINSIZE      = 20
TIMESCALE    = 100
N_COMPONENTS = 50
N_TREES      = 100
ATTEMPTS     = 100          # paper: 100 independent runs per split
N_TRAIN      = 8            # 8:2 subject split -> C(10,8)=45 folds
N_JOBS       = 10
CACHE_DIR    = os.path.join(HERE, ".raw_cache")
OUT_DIR      = os.path.join(PARENT, "Results", "PCA_final_comparison")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ACC   = ['acceleration_x', 'acceleration_y', 'acceleration_z']
GYRO  = ['gyro_x', 'gyro_y', 'gyro_z']

SUBJECT_DATA = pd.DataFrame([
    ["001",'woman',12,"kid","expert","regular",38],["002",'woman',26,"young adult","beginner","goofy",42],
    ["003",'man',40,"young adult","expert","regular",83],["004",'woman',27,"young adult","expert","regular",45],
    ["005",'man',22,"young adult","beginner","regular",54],["006",'man',22,"young adult","expert","goofy",65],
    ["007",'man',13,"kid","expert","regular",50],["008",'man',25,"young adult","expert","goofy",80],
    ["009",'man',22,"young adult","expert","regular",65],["010",'man',55,"older adult","beginner","regular",65],
], columns=['subject_ID','sex','age','age_group','level','stance','weight'])
SUBJECTS = list(SUBJECT_DATA["subject_ID"])

# Paper configs: (name, raw sensor dir, raw channels, feature-file glob pattern)
CONFIGS = [
    ("hip_Acc",   "sensor_hip_CUT_data",    ACC,        "*_DataFeatures_sensor_hip_CUT_data_Acc.csv"),
    ("pocket",    "sensor_pocket_CUT_data", ACC + GYRO, "*_DataFeatures_sensor_pocket_CUT_data.csv"),
    ("wrist_Acc", "sensor_wrist_CUT_data",  ACC,        "*_DataFeatures_sensor_wrist_CUT_data_Acc.csv"),
    ("wrist",     "sensor_wrist_CUT_data",  ACC + GYRO, "*_DataFeatures_sensor_wrist_CUT_data.csv"),
]

# Paper Table 2 (20-s window) balanced accuracy % mean(SD) — Method 1 reference
PAPER = {
    "hip_Acc":   {"multiclass": (87, 5), "pushpush": (93, 7)},
    "pocket":    {"multiclass": (87, 5), "pushpush": (95, 4)},
    "wrist_Acc": {"multiclass": (85, 4), "pushpush": (91, 6)},
    "wrist":     {"multiclass": (88, 3), "pushpush": (92, 8)},
}


# ── Raw window extraction (Method 2 input), cached ────────────────────────────
def extract_raw_windows(sensor_dir, channels, W=WINSIZE, ts=TIMESCALE):
    files = get_data_files(PARENT, sensor_dir, SUBJECTS)
    RAW, LAB, SUB = [], [], []
    for f in files:
        timeS = data_prep2(pd.read_csv(f))
        first = round(timeS.iloc[0]["ElapsedTime"], 3)
        starts, ends, wc = [0], [], 0
        for row, t in enumerate(timeS["ElapsedTime"]):
            if round(t, 3) == round(first + W * (wc + 1), 3) and row < len(timeS):
                ends.append(row); starts.append(row); wc += 1
        starts.pop()
        for a, b in zip(starts, ends):
            if b - a != W * ts:
                continue
            RAW.append(timeS.iloc[a:b][channels].values.flatten())
            LAB.append(get_label(f))
            SUB.append(int(get_subject_label(f, SUBJECT_DATA)[0]))
    return np.array(RAW, dtype=np.float32), np.array(LAB), np.array(SUB, dtype=int)


def get_raw_cached(name, sensor_dir, channels):
    cache = os.path.join(CACHE_DIR, f"raw_{name}_{WINSIZE}s.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], d["subj"]
    print(f"  [raw] extracting {name} ({len(channels)}ch) ...", flush=True)
    X, y, subj = extract_raw_windows(sensor_dir, channels)
    np.savez_compressed(cache, X=X, y=y, subj=subj)
    print(f"  [raw] {name}: {X.shape}", flush=True)
    return X, y, subj


def load_features(pattern):
    files = sorted(glob.glob(os.path.join(PARENT, f"FeatureDataFrames_{WINSIZE}", pattern)), reverse=True)
    # skip any raw-waveform file (12000-dim); keep the engineered-feature file
    for f in files:
        cols = pd.read_csv(f, index_col=0, nrows=1).columns
        idx = np.where(cols == "true_label")[0][0]
        if idx < 1000:
            df = pd.read_csv(f, index_col=0)
            X = df.iloc[:, :idx].values.astype(np.float32)
            return X, df["true_label"].values, df["subject"].astype(int).values, idx
    raise FileNotFoundError(pattern)


# ── One CV fold: fit scaler+PCA on train, average 100 RF runs, both tasks ─────
def eval_fold(X, y_multi, y_bin, subj, combo):
    tr = np.array([s in combo for s in subj]); te = ~tr
    sc  = StandardScaler().fit(X[tr])
    pca = PCA(n_components=min(N_COMPONENTS, X[tr].shape[1]),
              svd_solver="randomized", random_state=0).fit(sc.transform(X[tr]))
    Xtr = pca.transform(sc.transform(X[tr]))
    Xte = pca.transform(sc.transform(X[te]))
    evr = float(pca.explained_variance_ratio_.sum())
    out = {}
    for task, y in [("multiclass", y_multi), ("pushpush", y_bin)]:
        accs = []
        for a in range(ATTEMPTS):
            rf = RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced",
                                        n_jobs=1, random_state=a)
            rf.fit(Xtr, y[tr])
            accs.append(balanced_accuracy_score(y[te], rf.predict(Xte)))
        out[task] = float(np.mean(accs))
    out["evr"] = evr
    return out


def run_method(name, X, y_multi, subj, method_tag):
    y_bin  = np.where(np.isin(y_multi, ["kick", "skate"]), "pushpush", "other")
    combos = list(combinations(sorted(set(subj)), N_TRAIN))
    t0 = time.time()
    folds = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(eval_fold)(X, y_multi, y_bin, subj, c) for c in combos)
    mc  = np.array([f["multiclass"] for f in folds])
    pp  = np.array([f["pushpush"]   for f in folds])
    evr = np.mean([f["evr"] for f in folds])
    print(f"  [{method_tag}] {name:10s} dim={X.shape[1]:6d} "
          f"MC={mc.mean()*100:.1f}({mc.std()*100:.0f})  "
          f"PP={pp.mean()*100:.1f}({pp.std()*100:.0f})  "
          f"EVR={evr*100:.0f}%  ({time.time()-t0:.0f}s)", flush=True)
    return {"config": name, "method": method_tag, "input_dim": X.shape[1],
            "pca_evr": round(evr, 4),
            "multiclass_mean": round(mc.mean(), 4), "multiclass_sd": round(mc.std(), 4),
            "pushpush_mean":   round(pp.mean(), 4), "pushpush_sd":   round(pp.std(), 4)}


def main():
    print(f"Protocol: winsize={WINSIZE}s, PCA={N_COMPONENTS}, trees={N_TREES}, "
          f"attempts={ATTEMPTS}, folds=C(10,8)=45\n", flush=True)
    rows = []
    ckpt = os.path.join(OUT_DIR, "checkpoint.csv")
    for name, sensor_dir, channels, feat_glob in CONFIGS:
        print(f"=== CONFIG: {name} ===", flush=True)
        # Method 2: raw -> PCA -> RF
        Xr, yr, sr = get_raw_cached(name, sensor_dir, channels)
        rows.append(run_method(name, Xr, yr, sr, "M2_raw+PCA"))
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        # Method 3: features -> PCA -> RF
        Xf, yf, sf, fdim = load_features(feat_glob)
        rows.append(run_method(name, Xf, yf, sf, "M3_feat+PCA"))
        pd.DataFrame(rows).to_csv(ckpt, index=False)

    df = pd.DataFrame(rows)
    # attach paper Method-1 reference
    df["paper_M1_multiclass"] = df["config"].map(lambda c: PAPER[c]["multiclass"][0] / 100)
    df["paper_M1_pushpush"]   = df["config"].map(lambda c: PAPER[c]["pushpush"][0] / 100)
    stamp = time.strftime("%Y-%m-%dT%H_%M_%S")
    final = os.path.join(OUT_DIR, f"{stamp}_final_comparison.csv")
    df.to_csv(final, index=False)

    print("\n" + "=" * 72)
    print("  FINAL 3-METHOD COMPARISON  (balanced accuracy %)")
    print("=" * 72)
    print(f"{'config':11s} {'task':11s} {'M1 paper':>9s} {'M2 raw+PCA':>11s} {'M3 feat+PCA':>12s}")
    for c in [x[0] for x in CONFIGS]:
        for task, key in [("multiclass", "multiclass"), ("pushpush", "pushpush")]:
            m1 = PAPER[c][task][0]
            m2 = df[(df.config == c) & (df.method == "M2_raw+PCA")][f"{key}_mean"].values[0] * 100
            m3 = df[(df.config == c) & (df.method == "M3_feat+PCA")][f"{key}_mean"].values[0] * 100
            print(f"{c:11s} {task:11s} {m1:8.0f}% {m2:10.1f}% {m3:11.1f}%")
    print(f"\nSaved -> {final}")


if __name__ == "__main__":
    main()
