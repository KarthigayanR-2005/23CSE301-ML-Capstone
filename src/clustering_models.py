"""Clustering track: K-Means and Agglomerative Hierarchical clustering.

Rules enforced here:
  * fitting uses ONLY the selected input features; the withheld Attrition_Flag
    never enters feature selection, scaling, k selection or fitting
  * clustering is fitted on the FULL scaled feature space, not on two PCA
    components; PCA and t-SNE are used for VISUALISATION only
  * anything approximated or sampled is labelled as such in its return value
"""
from __future__ import annotations

import resource

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from . import config
from .evaluation import clustering_metrics

RS = config.RANDOM_STATE
LINKAGES = ("ward", "complete", "average", "single")


# --------------------------------------------------------------------------
# Memory feasibility
# --------------------------------------------------------------------------
def memory_probe(n_rows: int) -> dict:
    """Estimate the cost of the O(n^2) steps before running them.

    AgglomerativeClustering and the exact silhouette both materialise an
    n x n float64 distance structure: 8 * n^2 bytes.
    """
    pair_bytes = 8 * n_rows ** 2
    try:
        import os
        avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        avail = None
    est_gb = pair_bytes / 1024 ** 3
    return {
        "n_rows": n_rows,
        "pairwise_matrix_GB": round(est_gb, 3),
        "available_RAM_GB": round(avail / 1024 ** 3, 2) if avail else "unknown",
        "feasible": (avail is None) or (pair_bytes * 3 < avail),
        "note": "scipy/sklearn need roughly 2-3x the bare matrix during linkage",
    }


# --------------------------------------------------------------------------
# K-Means
# --------------------------------------------------------------------------
def kmeans_sweep(X, k_range=range(2, 11), silhouette_sample: int | None = None
                 ) -> tuple[pd.DataFrame, dict]:
    """Elbow + metric sweep. Returns one row per k and the fitted models."""
    rows, models = [], {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RS, n_init=10)
        labels = km.fit_predict(X)
        m = clustering_metrics(X, labels, sample_size=silhouette_sample)
        m.update({"k": k, "algorithm": "KMeans", "inertia": float(km.inertia_)})
        rows.append(m)
        models[k] = km
        print(f"  k={k:<3} inertia={km.inertia_:>12.1f} "
              f"sil={m['Silhouette']:.4f} DB={m['Davies_Bouldin']:.4f} "
              f"CH={m['Calinski_Harabasz']:.1f}")
    cols = ["algorithm", "k", "inertia", "Silhouette", "Davies_Bouldin",
            "Calinski_Harabasz", "silhouette_exact", "silhouette_sample_size"]
    return pd.DataFrame(rows)[cols], models


def elbow_knee(ks, inertias) -> int:
    """Largest-perpendicular-distance knee ('elbow') of the inertia curve.

    Adapted from the standard 'Kneedle' line-distance idea
    (Satopaa et al., 2011, 'Finding a Kneedle in a Haystack'). This is a
    numerical aid for the team's own choice of k, not a substitute for it.
    """
    ks = np.asarray(list(ks), dtype=float)
    y = np.asarray(list(inertias), dtype=float)
    p1, p2 = np.array([ks[0], y[0]]), np.array([ks[-1], y[-1]])
    line = p2 - p1
    line = line / np.linalg.norm(line)
    d = []
    for xi, yi in zip(ks, y):
        v = np.array([xi, yi]) - p1
        d.append(np.linalg.norm(v - np.dot(v, line) * line))
    return int(ks[int(np.argmax(d))])


# --------------------------------------------------------------------------
# Agglomerative
# --------------------------------------------------------------------------
def linkage_comparison(X, k: int, linkages=LINKAGES,
                       silhouette_sample: int | None = None) -> pd.DataFrame:
    """Compare linkage strategies at a fixed k (PDF note for algorithm 2)."""
    rows = []
    for link in linkages:
        metric = "euclidean"
        agg = AgglomerativeClustering(n_clusters=k, linkage=link, metric=metric)
        labels = agg.fit_predict(X)
        m = clustering_metrics(X, labels, sample_size=silhouette_sample)
        m.update({"algorithm": "Agglomerative", "linkage": link, "k": k,
                  "metric": metric})
        rows.append(m)
        print(f"  linkage={link:<9} sil={m['Silhouette']:.4f} "
              f"DB={m['Davies_Bouldin']:.4f} CH={m['Calinski_Harabasz']:.1f}")
    cols = ["algorithm", "linkage", "k", "Silhouette", "Davies_Bouldin",
            "Calinski_Harabasz", "silhouette_exact", "silhouette_sample_size"]
    return pd.DataFrame(rows)[cols]


def agglomerative_sweep(X, k_range=range(2, 11), linkage: str = "ward",
                        silhouette_sample: int | None = None
                        ) -> tuple[pd.DataFrame, dict]:
    rows, models = [], {}
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = agg.fit_predict(X)
        m = clustering_metrics(X, labels, sample_size=silhouette_sample)
        m.update({"k": k, "algorithm": "Agglomerative", "linkage": linkage})
        rows.append(m)
        models[k] = (agg, labels)
        print(f"  k={k:<3} sil={m['Silhouette']:.4f} DB={m['Davies_Bouldin']:.4f} "
              f"CH={m['Calinski_Harabasz']:.1f}")
    cols = ["algorithm", "linkage", "k", "Silhouette", "Davies_Bouldin",
            "Calinski_Harabasz", "silhouette_exact", "silhouette_sample_size"]
    return pd.DataFrame(rows)[cols], models


# --------------------------------------------------------------------------
# Supplementary validation
# --------------------------------------------------------------------------
def stability_assessment(X, k: int, algorithm: str = "kmeans",
                         n_repeats: int = 20, subsample_frac: float = 0.8,
                         linkage: str = "ward",
                         random_state: int = RS) -> dict:
    """Resampling-based cluster stability.

    !! THIS IS NOT SUPERVISED CROSS-VALIDATION.
    The PDF asks for cross-validation in every track, but k-fold CV has no
    direct meaning for AgglomerativeClustering, which has no `predict` for
    unseen points. This routine is offered as SUPPLEMENTARY evidence only:
    it repeatedly draws overlapping subsamples, clusters each, and measures
    label agreement (Adjusted Rand Index) on the overlap. A high ARI means the
    partition is reproducible under resampling; it does NOT mean the rubric's
    cross-validation requirement has been satisfied. See
    docs/instructor_clarifications.md item IC-3.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    n = len(X)
    size = int(subsample_frac * n)
    aris = []
    for i in range(n_repeats):
        idx_a = rng.choice(n, size=size, replace=False)
        idx_b = rng.choice(n, size=size, replace=False)
        overlap = np.intersect1d(idx_a, idx_b)
        if len(overlap) < k * 2:
            continue
        if algorithm == "kmeans":
            la = KMeans(n_clusters=k, random_state=random_state + i,
                        n_init=10).fit_predict(X[idx_a])
            lb = KMeans(n_clusters=k, random_state=random_state + 100 + i,
                        n_init=10).fit_predict(X[idx_b])
        else:
            la = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X[idx_a])
            lb = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X[idx_b])
        pos_a = {v: p for p, v in enumerate(idx_a)}
        pos_b = {v: p for p, v in enumerate(idx_b)}
        aris.append(adjusted_rand_score([la[pos_a[o]] for o in overlap],
                                        [lb[pos_b[o]] for o in overlap]))
    return {
        "algorithm": algorithm, "k": k, "n_repeats_completed": len(aris),
        "subsample_frac": subsample_frac,
        "mean_ARI": float(np.mean(aris)) if aris else float("nan"),
        "std_ARI": float(np.std(aris)) if aris else float("nan"),
        "min_ARI": float(np.min(aris)) if aris else float("nan"),
        "interpretation_status": "SUPPLEMENTARY - not supervised cross-validation",
    }


# --------------------------------------------------------------------------
# Visualisation embeddings (never used for fitting)
# --------------------------------------------------------------------------
def pca_embedding(X, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=RS)
    coords = pca.fit_transform(X)
    print(f"  [PCA] explained variance ratio: "
          f"{np.round(pca.explained_variance_ratio_, 4)} "
          f"(cumulative {pca.explained_variance_ratio_.sum():.4f})")
    return coords, pca


def tsne_embedding(X, labels=None, sample_size: int | None = 3000,
                   perplexity: float = 30.0, random_state: int = RS):
    """t-SNE for visualisation. Returns (coords, labels_subset, note).

    t-SNE is O(n^2)-ish and slow on 10k rows, so a labelled subsample is used
    by default. The returned note states the sample size and is printed on the
    figure title by the caller.
    """
    X = np.asarray(X)
    idx = np.arange(len(X))
    note = "full dataset"
    if sample_size and sample_size < len(X):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        note = f"SAMPLED: {sample_size} of {len(X)} rows (visualisation only)"
    ts = TSNE(n_components=2, perplexity=perplexity, init="pca",
              random_state=random_state, max_iter=1000)
    coords = ts.fit_transform(X[idx])
    lab = None if labels is None else np.asarray(labels)[idx]
    return coords, lab, note


# --------------------------------------------------------------------------
# Post-hoc validation (AFTER clusters are chosen)
# --------------------------------------------------------------------------
def posthoc_label_crosstab(labels, holdout: pd.Series) -> pd.DataFrame:
    """Cross-tabulate final clusters against the withheld Attrition_Flag.

    Run ONLY after k and the final model are fixed. Clusters are not expected
    or required to align with churn.
    """
    ct = pd.crosstab(pd.Series(labels, name="cluster"),
                     pd.Series(np.asarray(holdout), name="Attrition_Flag"))
    ct["row_pct_attrited"] = (ct.get("Attrited Customer", 0) / ct.sum(axis=1) * 100).round(2)
    return ct


def cluster_profile(df_features: pd.DataFrame, labels) -> pd.DataFrame:
    """Mean of each ORIGINAL (unscaled) feature per cluster, plus cluster size."""
    prof = df_features.copy()
    prof["cluster"] = labels
    out = prof.groupby("cluster").mean(numeric_only=True)
    out.insert(0, "n_members", prof.groupby("cluster").size())
    out.insert(1, "pct_of_data", (out["n_members"] / len(prof) * 100).round(2))
    return out.round(3)
