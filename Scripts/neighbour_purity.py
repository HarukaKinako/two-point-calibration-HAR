"""How strongly do windows cluster by person within each activity? (Sec. 4.1 of the DOCSAI 2026 paper)

For every window, take the 10 nearest windows of the SAME activity in the standardized hip feature space
and count how many come from the same person. Chance level is about 10% with ten participants.

Needs the feature files in FeatureDataFrames_20/ (see README).
Run from the Scripts directory:  python3 neighbour_purity.py
"""
import os
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pca_final_comparison as M

K = 10
ACTIVITIES = ["sit", "stand", "walk", "run", "bicycle", "upstairs", "downstairs", "kick", "skate"]


def same_person_share(Z, s, k=K):
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    return (s[idx] == s[:, None]).mean() * 100


if __name__ == "__main__":
    X, y, subj, _ = M.load_features(M.CONFIGS[0][3])      # hip, accelerometer only
    Z = StandardScaler().fit_transform(X)
    print(f"same-person share among the {K} nearest windows of the same activity (hip sensor)")
    for a in ACTIVITIES:
        m = y == a
        s = subj[m]
        chance = np.mean([((s == v).sum() - 1) / (len(s) - 1) for v in s]) * 100
        print(f"  {a:10s} n={m.sum():4d}  same person {same_person_share(Z[m], s):5.1f}%  (chance {chance:4.1f}%)")
