"""Figures 1-5 of the DOCSAI 2026 paper, drawn from the result files in Results/.

fig1_tsne_activity_person.png  t-SNE map of the hip features, coloured by activity and by person
                               (embedding cached by tsne_plots.py).
fig2_calibration_illustration  two-point calibration explained with two made-up people.
fig3_results_6activities.png   six-activity results, no calibration vs two-point, every split shown.
fig4_importance_share.png      share of the forest's importance held by the calibrated features.
fig5_calibration_amount.png    accuracy gain against the number of calibration windows.

Run from the Scripts directory:  python3 paper_figures.py
Output: Results/paper_figures/
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "Results")
CAL = os.path.join(RES, "walk_calibration")
OUT = os.path.join(RES, "paper_figures")

CONFIGS = ["hip_Acc", "pocket", "wrist_Acc", "wrist"]
NAME = {"hip_Acc": "Hip", "pocket": "Pocket", "wrist_Acc": "Wrist\n(without gyro)", "wrist": "Wrist\n(with gyro)"}
GRAY, ORANGE, TEAL, INK, MUTED, LINE, BAND = ("#9AA5B1", "#F26B38", "#2A9D8F", "#1F2A36", "#6B7580",
                                              "#D5DADF", "#FDECE4")
STYLE = {"font.family": "Arial", "font.size": 13, "axes.edgecolor": "#C5CCD3", "axes.labelcolor": INK,
         "xtick.color": INK, "ytick.color": INK}


def fig1_tsne():
    d = np.load(os.path.join(RES, "tsne", "tsne_hip_Acc.npz"), allow_pickle=True)
    emb, y, subj = d["emb"], d["y"], d["subj"].astype(int)
    act = {"kick": "#E34948", "skate": "#2A78D6", "bicycle": "#1BAF7A", "run": "#EDA100", "walk": "#7F77DD",
           "upstairs": "#D55181", "downstairs": "#995A1D", "sit": "#B4B2A9", "stand": "#5F5E5A"}
    label = {"kick": "kickboard", "skate": "skateboard"}
    per = ["#E34948", "#2A78D6", "#1BAF7A", "#EDA100", "#7F77DD", "#D55181", "#00929C", "#995A1D", "#1F2A36",
           "#C86BD6"]
    subs = sorted(set(subj))
    pmap = {s: per[i] for i, s in enumerate(subs)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, cols, title in [(axes[0], [act[v] for v in y], "Coloured by activity"),
                            (axes[1], [pmap[s] for s in subj], "Coloured by person")]:
        ax.scatter(emb[:, 0], emb[:, 1], c=cols, s=6, alpha=.75, linewidths=0)
        ax.set_title(title, fontsize=15, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(LINE)
    h1 = [Line2D([0], [0], marker="o", ls="", ms=8, mfc=c, mew=0, label=label.get(a, a)) for a, c in act.items()]
    h2 = [Line2D([0], [0], marker="o", ls="", ms=8, mfc=pmap[s], mew=0, label=f"P{i + 1}")
          for i, s in enumerate(subs)]
    for ax, h in [(axes[0], h1), (axes[1], h2)]:
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(.5, -.02), ncol=5, frameon=False, fontsize=11.5,
                  handletextpad=.2, columnspacing=.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_tsne_activity_person.png"), dpi=200)
    plt.close(fig)


def fig2_illustration():
    colors = {"resting": "#8C96A0", "walking": TEAL, "skateboarding": ORANGE}
    # Made-up values: person A moves more vigorously than person B.
    people = {"Person A": {"resting": 0.10, "walking": 0.50, "skateboarding": 0.70},
              "Person B": {"resting": 0.05, "walking": 0.30, "skateboarding": 0.42}}
    two_point = lambda v: {k: (x - v["resting"]) / (v["walking"] - v["resting"]) for k, x in v.items()}

    def number_lines(ax, values, xlim, title, xlabel):
        rows = list(values)
        for r, person in enumerate(rows):
            yy = len(rows) - 1 - r
            ax.plot(xlim, [yy, yy], color=LINE, lw=2, zorder=1)
            for a, x in values[person].items():
                ax.scatter(x, yy, s=190, color=colors[a], zorder=3, edgecolor="white", linewidth=1.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows[::-1], fontsize=14, color=INK)
        ax.set_title(title, fontsize=15, color=INK, pad=12)
        ax.set_xlabel(xlabel, fontsize=12.5, color=MUTED)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=MUTED, labelsize=11)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"wspace": 0.28})
    number_lines(a, people, (0, 0.8), "Before: original feature value", "e.g. movement intensity (arbitrary units)")
    a.axvspan(0.40, 0.52, color=BAND, zorder=0)
    a.text(0.46, 1.45, "B's skateboarding ≈ A's walking", ha="center", fontsize=11.5, color="#C4501F")
    calibrated = {p: two_point(v) for p, v in people.items()}
    number_lines(b, calibrated, (-0.25, 1.85), "After: two-point calibration", "calibrated value")
    for x, lab in [(0, "resting = 0"), (1, "walking = 1")]:
        b.axvline(x, color=MUTED, lw=1, ls=":", zorder=0)
        b.text(x, 1.45, lab, ha="center", fontsize=11.5, color=MUTED, zorder=2,
               bbox=dict(facecolor="white", edgecolor="none", pad=2))
    b.axvspan(1.40, 1.58, color=BAND, zorder=0)
    for r, person in enumerate(calibrated):
        b.text(calibrated[person]["skateboarding"], 1 - r - 0.36, f"{calibrated[person]['skateboarding']:.2f}",
               ha="center", fontsize=11, color="#C4501F")
    handles = [Line2D([0], [0], marker="o", ls="", ms=11, mfc=c, mec="white", label=k) for k, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=12.5,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.6)
    fig.text(0.99, 0.01, "Illustration with made-up numbers", ha="right", fontsize=10.5, color=MUTED, style="italic")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.30)
    fig.savefig(os.path.join(OUT, "fig2_calibration_illustration.png"), dpi=200)
    plt.close(fig)


def fig3_results():
    df = pd.read_csv(os.path.join(CAL, "refstats_results.csv"))
    d6 = df[df.task == "multiclass6"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    rng = np.random.RandomState(0)
    for i, cfg in enumerate(CONFIGS):
        piv = d6[d6.config == cfg].pivot_table(index="fold_heldout", columns="method", values="bal_acc") * 100
        before, after = piv["raw"].values, piv["raw_2pt"].values
        for offset, values, color in [(-0.21, before, GRAY), (0.21, after, ORANGE)]:
            bp = ax.boxplot([values], positions=[i + offset], widths=0.34, patch_artist=True, showfliers=False,
                            medianprops=dict(color=INK, lw=1.8), whiskerprops=dict(color="#7A8591"),
                            capprops=dict(color="#7A8591"), boxprops=dict(edgecolor="#7A8591"))
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.30)
            ax.scatter(i + offset + rng.uniform(-0.09, 0.09, len(values)), values, s=15, color=color,
                       alpha=0.9, linewidths=0, zorder=3)
        p = stats.wilcoxon(after - before).pvalue
        label = "p < 0.01" if p < 0.01 else "p < 0.05" if p < 0.05 else "n.s."
        ax.text(i, max(before.max(), after.max()) + 1.0, label, ha="center", va="bottom", fontsize=12.5,
                color=MUTED)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([NAME[c] for c in CONFIGS])
    ax.set_ylim(70, 106)
    ax.set_yticks(range(70, 101, 5))
    ax.set_ylabel("Balanced accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(facecolor=GRAY, alpha=.55, label="No calibration"),
                       Patch(facecolor=ORANGE, alpha=.75, label="Two-point calibration")],
              loc="upper left", bbox_to_anchor=(0, 1.13), ncol=2, frameon=False, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_results_6activities.png"), dpi=200)
    plt.close(fig)


def fig4_importance():
    share = [pd.read_csv(os.path.join(CAL, "importance", f"importance_{c}_multiclass6.csv"))
             .query("half == 'calibrated'").importance.sum() * 100 for c in CONFIGS]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(range(4), share, width=.55, color=[ORANGE, ORANGE, GRAY, GRAY])
    for b, v in zip(bars, share):
        ax.text(b.get_x() + b.get_width() / 2, v - 2, f"{v:.0f}%", ha="center", va="top", fontsize=17,
                color="white", fontweight="bold")
    ax.axhline(50, color=MUTED, ls="--", lw=1.2)
    ax.text(3.45, 51, "equal use (50%)", ha="right", fontsize=11.5, color=MUTED)
    ax.set_xticks(range(4))
    ax.set_xticklabels([NAME[c] for c in CONFIGS])
    ax.set_ylim(0, 58)
    ax.set_ylabel("Importance held by\ncalibrated features (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_importance_share.png"), dpi=200)
    plt.close(fig)


def fig5_amount():
    da = pd.read_csv(os.path.join(CAL, "calib_amount_results.csv"))
    n_list = [2, 5, 10, 15, 20, -1]
    xl = ["2\n(40 s)", "5\n(100 s)", "10\n(200 s)", "15\n(300 s)", "20\n(400 s)", "all"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    x = np.arange(len(n_list))
    for ax, cfg, color in zip(axes, ["hip_Acc", "pocket"], [ORANGE, TEAL]):
        rows = []
        for n in n_list:
            s = da[(da.config == cfg) & (da.n_cal == n) & (da.task == "multiclass6")]
            piv = s.pivot_table(index="fold_heldout", columns="method", values="bal_acc") * 100
            d = piv["raw_2pt"] - piv["raw"]
            rows.append((d.mean(), d.quantile(.25), d.quantile(.75)))
        mean, lo, hi = map(np.array, zip(*rows))
        ax.fill_between(x, lo, hi, color=color, alpha=.18, lw=0)
        ax.plot(x, mean, marker="o", color=color, lw=2.8, ms=7)
        ax.axhline(0, color="#7A8591", lw=1)
        ax.axvline(1, color=INK, lw=1, ls=":")
        ax.set_title(NAME[cfg], fontsize=14, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(xl, fontsize=11.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy gain (pt)")
    axes[1].text(.98, .97, "line: mean of 45 splits\nband: middle 50% of splits", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=11, color=MUTED)
    fig.supxlabel("Calibration windows per reference activity (20 s each)", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_calibration_amount.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with plt.rc_context(STYLE):
        fig1_tsne()
        fig2_illustration()
        fig3_results()
        fig4_importance()
        fig5_amount()
    print("saved figures to", OUT)
