"""Numbers reported in the DOCSAI 2026 paper, recomputed from the per-fold result files.

Reads (all produced by the experiment scripts in this folder):
  Results/walk_calibration/refstats_results.csv     refstats_vs_calibration.py  (Tables 2-3, Sec. 5.5)
  Results/walk_calibration/calib_amount_results.csv calib_amount_curve.py       (Sec. 5.4)
  Results/walk_calibration/importance/*.csv         calib_importance.py         (Sec. 5.3)
  Results/PCA_final_comparison/perfold_results.csv  perfold_baseline_run.py     (Sec. 6.3, baseline)
  Results/tsne/stage1_leakfree.csv                  tsne_stage1_leakfree.py     (Sec. 6.3)

Every method is evaluated on the same 45 subject-wise folds, so the comparisons are paired:
Wilcoxon signed-rank test over folds, Holm correction across the four sensor configurations.

Run from the Scripts directory:  python3 paper_stats.py
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CAL = os.path.join(ROOT, "Results", "walk_calibration")
CONFIGS = ["hip_Acc", "pocket", "wrist_Acc", "wrist"]
NAME = {"hip_Acc": "hip", "pocket": "pocket", "wrist_Acc": "wrist (without gyro)", "wrist": "wrist (with gyro)"}


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def paired(df, cfg, task, method, base="raw"):
    """Per-fold accuracies (%) of `base` and `method`, aligned on the held-out pair."""
    piv = df[(df.config == cfg) & (df.task == task)].pivot_table(
        index="fold_heldout", columns="method", values="bal_acc") * 100
    return piv[base].values, piv[method].values


def compare(df, task, method, base="raw"):
    rows = []
    for cfg in CONFIGS:
        b, m = paired(df, cfg, task, method, base)
        rows.append({"config": NAME[cfg], "base": b.mean(), "method": m.mean(), "gain": (m - b).mean(),
                     "wins": f"{int((m > b).sum())}/{len(b)}", "p": stats.wilcoxon(m - b).pvalue})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p)
    return out


def show(title, table):
    print(f"\n{title}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}" if v < 1 else f"{v:.1f}"))


def main():
    ref = pd.read_csv(os.path.join(CAL, "refstats_results.csv"))

    show("Table 2 - six-activity evaluation, no calibration vs two-point calibration",
         compare(ref, "multiclass6", "raw_2pt"))

    print("\nTable 3 - gain over no calibration (points, six-activity evaluation)")
    gains = {}
    for method, label in [("raw_stats", "statistics"), ("raw_2pt", "two-point"), ("raw_2pt_stats", "both")]:
        t = compare(ref, "multiclass6", method)
        gains[label] = [f"{g:+.1f}" + ("*" if p < 0.05 else "") for g, p in zip(t.gain, t.p_holm)]
    print(pd.DataFrame(gains, index=[NAME[c] for c in CONFIGS]).to_string())
    print("  * Holm-corrected p < 0.05")

    show("Section 5.5 - nine-activity evaluation (reference activities included)",
         compare(ref, "multiclass9", "raw_2pt"))
    show("Section 5.5 - push-push-glide vs the rest", compare(ref, "pushpush", "raw_2pt"))

    amount = pd.read_csv(os.path.join(CAL, "calib_amount_results.csv"))
    print("\nSection 5.4 - calibration windows per reference activity (six-activity evaluation)")
    for cfg in CONFIGS:
        cells = []
        for n in sorted(amount.n_cal.unique(), key=lambda v: (v < 0, v)):
            b, m = paired(amount[amount.n_cal == n], cfg, "multiclass6", "raw_2pt")
            label = "all" if n < 0 else str(n)
            cells.append(f"{label}: {(m - b).mean():+.2f} (p={stats.wilcoxon(m - b).pvalue:.3f})")
        print(f"  {NAME[cfg]:22s} " + "  ".join(cells))

    print("\nSection 5.3 - share of total importance held by the calibrated features")
    for cfg in CONFIGS:
        imp = pd.read_csv(os.path.join(CAL, "importance", f"importance_{cfg}_multiclass6.csv"))
        print(f"  {NAME[cfg]:22s} {imp[imp.half == 'calibrated'].importance.sum() * 100:.1f}%")

    base = pd.read_csv(os.path.join(ROOT, "Results", "PCA_final_comparison", "perfold_results.csv"))
    tsne = pd.read_csv(os.path.join(ROOT, "Results", "tsne", "stage1_leakfree.csv"))
    pick = lambda d, method: d[(d.config == "hip_Acc") & (d.method == method) & (d.task == "pushpush")].bal_acc.mean()
    print("\nSection 6.3 - hip sensor, push-push-glide vs the rest")
    print(f"  original features          {pick(base, 'M1_feat_RF') * 100:.1f}%")
    print(f"  t-SNE fitted on training   {pick(tsne, 'tsneOF_d2') * 100:.1f}%")


if __name__ == "__main__":
    main()
