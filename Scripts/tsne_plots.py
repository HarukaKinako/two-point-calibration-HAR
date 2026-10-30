"""
t-SNE visualization of the 211-feature space (NOT PCA — direct t-SNE to 2-D).

For each sensor config:
  standardize -> t-SNE(2D, perplexity=30, init='pca', random_state=42)
Embeddings are cached (.npz) so recoloring doesn't recompute.

Outputs (Results/tsne/):
  tsne_multiclass.png : 2x2 grid, one panel per config, 9 activity colors
  tsne_binary.png     : same embeddings, push-push-glide vs other (2 colors)
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pca_final_comparison as M

OUT_DIR = os.path.join(M.PARENT, "Results", "tsne")
os.makedirs(OUT_DIR, exist_ok=True)

NAMES = {"hip_Acc": "Hip (acc)", "pocket": "Pocket (acc+gyro)",
         "wrist_Acc": "Wrist (acc)", "wrist": "Wrist (acc+gyro)"}

ACT_COLORS = {
    "kick":       "#e34948",   # red   — target
    "skate":      "#2a78d6",   # blue  — target
    "bicycle":    "#1baf7a",
    "run":        "#eda100",
    "walk":       "#7f77dd",
    "upstairs":   "#d55181",
    "downstairs": "#995a1d",
    "sit":        "#b4b2a9",   # sedentary in grays
    "stand":      "#5f5e5a",
}
BIN_COLORS = {"pushpush": "#d4537e", "other": "#b4b2a9"}


def get_embedding(name, feat_glob):
    cache = os.path.join(OUT_DIR, f"tsne_{name}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["emb"], d["y"], d["subj"]
    X, y, subj, fdim = M.load_features(feat_glob)
    Xs = StandardScaler().fit_transform(X)
    t0 = time.time()
    emb = TSNE(n_components=2, perplexity=30, init="pca",
               learning_rate="auto", random_state=42).fit_transform(Xs)
    print(f"  [{name}] t-SNE done ({fdim} dims, n={len(X)}) in {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(cache, emb=emb, y=y, subj=subj)
    return emb, y, subj


def scatter_panel(ax, emb, colors_per_point, title):
    ax.scatter(emb[:, 0], emb[:, 1], c=colors_per_point, s=5, alpha=0.65, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")


def main():
    data = {}
    for name, _sdir, _ch, feat_glob in M.CONFIGS:
        print(f"=== {name} ===", flush=True)
        data[name] = get_embedding(name, feat_glob)

    # ---- multiclass figure ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, name in zip(axes.flat, NAMES):
        emb, y, _ = data[name]
        scatter_panel(ax, emb, [ACT_COLORS[v] for v in y], NAMES[name])
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                      markerfacecolor=c, markeredgewidth=0, label=a)
               for a, c in ACT_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("t-SNE of the 211-feature space — colored by activity (9 classes)", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    p1 = os.path.join(OUT_DIR, "tsne_multiclass.png")
    fig.savefig(p1, dpi=170); plt.close(fig)

    # ---- binary figure ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, name in zip(axes.flat, NAMES):
        emb, y, _ = data[name]
        ybin = np.where(np.isin(y, ["kick", "skate"]), "pushpush", "other")
        order = np.argsort(ybin == "pushpush")   # draw pushpush on top
        scatter_panel(ax, emb[order], [BIN_COLORS[v] for v in ybin[order]], NAMES[name])
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                      markerfacecolor=c, markeredgewidth=0, label=l)
               for l, c in [("push-push-glide (kick+skate)", BIN_COLORS["pushpush"]),
                            ("other activities", BIN_COLORS["other"])]]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle("t-SNE of the 211-feature space — push-push-glide vs other", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    p2 = os.path.join(OUT_DIR, "tsne_binary.png")
    fig.savefig(p2, dpi=170); plt.close(fig)

    print(f"\nSaved:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
