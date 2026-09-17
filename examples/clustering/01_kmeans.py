"""
1. K-Means Clustering
============================================================
Track: Clustering (Review 2)

WHAT IT IS (in simple words)
----------------------------
Splits customers into k groups by repeatedly moving k "centre points" until
each customer belongs to whichever centre is nearest. You must tell it k in
advance - it cannot work out how many groups there should be.

HOW IT WORKS (step by step)
---------------------------
  1. Pick k starting centre points (k-means++ spreads them out sensibly rather
     than choosing at random).
  2. Assign every customer to the nearest centre. That creates k groups.
  3. Move each centre to the average position of the customers now in its group.
  4. Reassign everyone to the nearest of the NEW centres.
  5. Repeat steps 3-4 until nobody changes group. That is convergence.

  n_init=10 means the whole procedure runs 10 times from 10 different starts,
  keeping the best. K-Means can land in a poor solution from a bad start, so
  this matters.

SETTINGS YOU CAN TUNE
---------------------
n_clusters (k) is the big one, chosen with the elbow curve. n_init trades
compute for reliability. random_state=42 makes the result reproducible.

WHAT TO SAY IN THE VIVA
-----------------------
K-Means assumes clusters are round and roughly equal in size, because it
assigns by straight-line distance to a centre. If the real groups were long
and thin, it would cut them in half. It also needs k up front, and our elbow
(k=5) disagrees with our silhouette maximum (k=2) - a disagreement worth
owning rather than hiding.

OUR RESULT ON THIS DATASET
--------------------------
At k=5: Silhouette 0.1975, Davies-Bouldin 1.4304, Calinski-Harabasz 2373.4.
It beat Agglomerative on all three, and was far more stable under resampling
(ARI 0.87 against 0.31). The silhouette of ~0.20 is weak in absolute terms.

RUN IT
------
  python examples/clustering/01_kmeans.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from src import clustering_models as cu


# ---------------------------------------------------------------------------
# 1. LOAD  -  7 behaviour features. There is NO target: clustering is
#    unsupervised. Attrition_Flag is deliberately held back and is not used to
#    pick features, scale, choose k, or fit anything.
# ---------------------------------------------------------------------------
df = dl.prepare("clustering")                 # 10,127 rows x 7 columns

# ---------------------------------------------------------------------------
# 2. SCALE  -  mandatory here. Credit_Limit runs to ~35,000 while
#    Avg_Utilization_Ratio sits between 0 and 1. Without standardising, the
#    distance between two customers would be decided almost entirely by credit
#    limit and the other six features would barely count.
# ---------------------------------------------------------------------------
X = pp.make_cluster_preprocessor(list(df.columns)).fit_transform(df)
print(f"scaled data: {X.shape}  (all 7 features, not PCA components)")


# ---------------------------------------------------------------------------
# 3. CHOOSE k  -  the elbow method. Fit K-Means for every k from 2 to 10 and
#    record the inertia (total squared distance from each point to its own
#    cluster centre). Inertia always falls as k rises, so we look for the
#    "elbow": the point after which it stops falling steeply.
# ---------------------------------------------------------------------------
table, models = cu.kmeans_sweep(X, range(2, 11))
k_elbow = cu.elbow_knee(table["k"], table["inertia"])
k_sil = int(table.loc[table["Silhouette"].idxmax(), "k"])
print(f"\nelbow suggests k = {k_elbow} | best silhouette at k = {k_sil}")
print("NOTE: these disagree. The team decides - see docs/team_analysis_prompts.md Q-CL2.")

# ---------------------------------------------------------------------------
# 4. FIT at the chosen k
# ---------------------------------------------------------------------------
km = models[k_elbow]
labels = km.labels_

# ---------------------------------------------------------------------------
# 5. EVALUATE
#    Silhouette        : -1 to +1, HIGHER is better. How much closer a point is
#                        to its own cluster than to the next nearest one.
#    Davies-Bouldin    : 0 upward,  LOWER is better. Average similarity between
#                        each cluster and the one it most resembles.
#    Calinski-Harabasz : HIGHER is better. Between-cluster spread divided by
#                        within-cluster spread.
# ---------------------------------------------------------------------------
m = ev.clustering_metrics(X, labels)
print("\nK-Means at k =", k_elbow)
print("=" * 60)
print(f"  Silhouette        : {m['Silhouette']:.4f}   (higher better)")
print(f"  Davies-Bouldin    : {m['Davies_Bouldin']:.4f}   (LOWER better)")
print(f"  Calinski-Harabasz : {m['Calinski_Harabasz']:.1f}   (higher better)")
print(f"  exact silhouette  : {m['silhouette_exact']}  (all 10,127 rows, not sampled)")

print("\n  Cluster sizes:")
for c in sorted(set(labels)):
    n = int((labels == c).sum())
    print(f"    cluster {c}: {n:5d} customers ({100*n/len(labels):5.2f}%)")

# ---------------------------------------------------------------------------
# 6. PROFILE  -  the mean of each ORIGINAL (unscaled) feature per cluster.
#    This table is what you interpret. Naming the clusters is TEAM work (Q-CL3).
# ---------------------------------------------------------------------------
print("\n  Cluster profiles (original units):")
print(cu.cluster_profile(df, labels).to_string())

