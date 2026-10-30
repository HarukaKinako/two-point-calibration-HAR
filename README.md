# Two-point personal calibration for activity recognition of unseen users

Code for the paper:

> 平田 陽香, Julien Tripette, Nathanael Aubert-Kato.
> 安静時・歩行時データを基準とした2点個人較正による未知ユーザーの活動認識
> (Two-Point Personal Calibration Using Resting and Walking Data for Activity Recognition of Unseen Users).
> 人工知能学会 研究会 SIG-DOCSAI, 2026.

Each feature computed from a sensor window is rescaled with the user's own mean values while resting
(sitting or standing) and walking, so that resting maps to 0 and walking maps to 1:

```
x_cal = (x - mu_rest) / (mu_walk - mu_rest)
```

The calibrated features are appended to the original ones and classified with a random forest.
On the six activities not used for calibration, balanced accuracy for unseen users improved from
83.2% to 85.1% (hip sensor) and from 85.5% to 86.8% (pocket sensor).

## Data

The data come from Aubert-Kato et al. (2025): ten participants performing nine activities, including
skateboarding and kickboarding, with sensors on the wrist, the hip, and in the back pocket.
The data are **not** included in this repository. Download them from Zenodo:

- Data and code: <https://zenodo.org/records/14769798>
- Paper: Aubert-Kato, N., Hatori, H., Orihara, A., Nakagata, T., Ohta, Y., Tripette, J.:
  Recognizing Skateboard and Kickboard Commuting Behaviors Using Activity Trackers: Feasibility Study
  Using Machine Learning Approaches, *JMIR Formative Research*, 9, e71969 (2025).
  <https://doi.org/10.2196/71969>

Place the raw sensor files so that the folder layout is:

```
two-point-calibration-HAR/
├── DataSample/sensor_CUT_data/sensor_hip_CUT_data/001/*.csv   (and pocket, wrist; subjects 001-010)
├── Scripts/
└── Results/
```

## Setup

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## What is in this repository

| Script | What it does | Paper |
|---|---|---|
| `data_preparation.py` | Splits the signals into 20-s windows and computes the features (from Aubert-Kato et al.) | Sec. 3.1 |
| `pca_final_comparison.py` | Shared settings and data loaders used by the other scripts; also the PCA baseline | Sec. 6.3 |
| `calib_holdout_run.py` | Holds out half of each person's resting and walking windows for calibration only | Sec. 3.2 |
| `refstats_vs_calibration.py` | No calibration vs two-point calibration vs giving the reference statistics directly (6 and 9 activities, push-push-glide vs rest) | Tables 2-3, Sec. 5.5 |
| `calib_amount_curve.py` | Accuracy gain against the number of calibration windows | Fig. 5, Sec. 5.4 |
| `calib_importance.py` | Share of the forest's feature importance held by the calibrated features | Fig. 4, Sec. 5.3 |
| `tsne_plots.py` | t-SNE map of the features | Fig. 1 |
| `neighbour_purity.py` | Share of nearest neighbours that come from the same person | Sec. 4.1 |
| `perfold_baseline_run.py` | Per-split baseline without calibration | Sec. 3.2, 6.3 |
| `tsne_stage1_leakfree.py` | t-SNE / UMAP fitted on the training participants only, used as features | Sec. 6.3 |
| `paper_stats.py` | Recomputes every number in the paper from `Results/` (Wilcoxon signed-rank test, Holm correction) | Tables 2-3, Sec. 5 |
| `paper_figures.py` | Draws Figures 1-5 from `Results/` | Figs. 1-5 |

`Results/` contains the per-split results behind the paper, so `paper_stats.py` and `paper_figures.py`
run without the raw data:

```bash
cd Scripts
python3 paper_stats.py
python3 paper_figures.py      # writes Results/paper_figures/
```

## Re-running the experiments

All commands are run from `Scripts/`.

```bash
python3 data_preparation.py          # writes FeatureDataFrames_20/ (takes a while)
python3 refstats_vs_calibration.py   # Tables 2-3
python3 calib_amount_curve.py        # Fig. 5
python3 calib_importance.py          # Fig. 4
python3 tsne_plots.py                # Fig. 1 (embedding cached in Results/tsne/)
python3 neighbour_purity.py          # Sec. 4.1
python3 perfold_baseline_run.py      # baseline
python3 tsne_stage1_leakfree.py      # Sec. 6.3 (needs openTSNE and umap-learn)
```

Evaluation protocol: train on 8 participants and test on the remaining 2, for all 45 combinations.
Random forests use 100 trees with balanced class weights; each split is averaged over several random
seeds. The number of parallel jobs is set by `N_JOBS` in `pca_final_comparison.py`.

## License

MIT License (see `LICENSE`). `data_preparation.py` is from the code released with Aubert-Kato et al. (2025)
by OchaUni-Physical-Activity-Measurement under the MIT License.

## Use of generative AI

Parts of the analysis code were written with the help of a generative AI assistant (Claude).
All code and results were checked by the authors.
