"""
2. Agglomerative Hierarchical Clustering
============================================================
Track: Clustering (Review 2)

WHAT IT IS (in simple words)
----------------------------
Starts with every customer as their own cluster and repeatedly merges the two
closest clusters, building a tree of merges. You cut that tree wherever you
want to get however many clusters you want.

HOW IT WORKS (step by step)
---------------------------
  1. Begin with 10,127 clusters - one per customer.
  2. Compute the distance between every pair of clusters, using the LINKAGE rule.
  3. Merge the closest pair. Now there are 10,126 clusters.
  4. Recompute distances involving the newly merged cluster and merge again.
  5. Repeat until the desired number of clusters remains (or until one is left,
     which gives the full tree the dendrogram draws).

SETTINGS YOU CAN TUNE
---------------------
n_clusters - where to cut the tree.
linkage - the rule for group-to-group distance: ward, complete, average, single.
Ward is the usual default and pairs naturally with Euclidean distance.

WHAT TO SAY IN THE VIVA
-----------------------
Unlike K-Means it does not need k before fitting - you see the whole tree and
cut afterwards. But it cannot assign a NEW customer without refitting
everything, which is exactly why ordinary cross-validation does not apply to
it (see docs/instructor_clarifications.md IC-3). It is also O(n^2) in memory:
we checked the cost was 0.76 GB against 2.99 GB available before running it on
all 10,127 rows.

OUR RESULT ON THIS DATASET
--------------------------
At k=5 with ward: Silhouette 0.1612, Davies-Bouldin 1.6437,
Calinski-Harabasz 1944.1 - worse than K-Means on all three. Under resampling
it was markedly less stable (ARI 0.31 against K-Means' 0.87), so its cluster
descriptions deserve less confidence.

RUN IT
------
  python examples/clustering/02_agglomerative.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from src import clustering_models as cu
from sklearn.cluster import AgglomerativeClustering

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
# 3. COMPARE LINKAGE STRATEGIES  -  "linkage" is the rule for measuring the
#    distance between two GROUPS of customers, not two customers.
#      ward     : merge the pair that increases total within-cluster variance least
#      complete : distance = the FARTHEST pair between the two groups
#      average  : distance = the average over all pairs
#      single   : distance = the CLOSEST pair  (prone to "chaining")
# ---------------------------------------------------------------------------
K = 5
print(f"\nLinkage comparison at k = {K}:")
print(cu.linkage_comparison(X, K).to_string(index=False))
print("\nWatch single linkage: its Calinski-Harabasz collapses to ~3.7 against")
print("ward's ~1944. That is the signature of one giant cluster plus slivers.")

# ---------------------------------------------------------------------------
# 4. FIT with ward linkage
#    How agglomerative clustering works:
#      1. Start with 10,127 clusters - every customer is their own cluster.
#      2. Find the two closest clusters (by the linkage rule) and merge them.
#      3. Repeat. Each step reduces the cluster count by one.
#      4. After 10,122 merges only 5 clusters remain - stop there.
#    It is called "agglomerative" because it builds UP by merging, in contrast
#    to "divisive" methods that split down from one big cluster.
# ---------------------------------------------------------------------------
agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
labels = agg.fit_predict(X)

m = ev.clustering_metrics(X, labels)
print(f"\nAgglomerative (ward) at k = {K}")
print("=" * 60)
print(f"  Silhouette        : {m['Silhouette']:.4f}   (higher better)")
print(f"  Davies-Bouldin    : {m['Davies_Bouldin']:.4f}   (LOWER better)")
print(f"  Calinski-Harabasz : {m['Calinski_Harabasz']:.1f}   (higher better)")

print("\n  Cluster sizes:")
for c in sorted(set(labels)):
    n = int((labels == c).sum())
    print(f"    cluster {c}: {n:5d} customers ({100*n/len(labels):5.2f}%)")

# ---------------------------------------------------------------------------
# 5. DENDROGRAM  -  the picture of the merge history. The height of each join
#    is the distance at which those two groups merged. Cutting the tree
#    horizontally at a chosen height gives you a chosen number of clusters.
#    We truncate the DISPLAY to the last 30 merges; the hierarchy itself is
#    complete over all 10,127 customers.
# ---------------------------------------------------------------------------
from src import plotting as pl
p = pl.dendrogram_plot(X, "ward", "example_dendrogram", "clustering", truncate_p=30)
print(f"\n  Dendrogram saved to {p}")

print("\n  Cluster profiles (original units):")
print(cu.cluster_profile(df, labels).to_string())

