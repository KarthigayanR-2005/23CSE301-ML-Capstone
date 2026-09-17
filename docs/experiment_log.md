# Experiment Log

Chronological record of what was actually run. Machine-readable equivalents:
`results/tables/run_log_*.json`.

Environment: Python 3.12.3, single CPU core, scikit-learn 1.8.0, pandas 3.0.2,
numpy 2.4.4. Seed `random_state=42` throughout.

---

## Session 1 — September 17, 2026

### Data acquisition
| Step | Outcome |
|---|---|
| Searched workspace for CSVs | none present |
| Attempted Kaggle download | **blocked** — sandbox network denied `kaggle.com` (`host_not_allowed`) |
| Team supplied the three CSVs manually | ✅ all three verified |

Shape verification — **all three matched the contract exactly**:
`Student_Performance.csv` 10,000×6 · `predictive_maintenance.csv` 10,000×10 ·
`BankChurners.csv` 10,127×23. Checksums in
`results/tables/dataset_provenance.json`.

Findings at audit: 0 missing cells in all three; **127 exact duplicate rows** in
the regression data; classification target 9,661 / 339 (**3.39 % positive**).

### Feasibility probes (run before committing to the full runs)
| Probe | Result |
|---|---|
| SVC fit, 8,000 rows | 0.84 s |
| SVR fit, 8,000 rows | 1.99 s |
| Agglomerative ward, full 10,127 rows | 2.3 s |
| Exact silhouette, full 10,127 rows | 1.2 s |
| Pairwise memory estimate | 0.76 GB vs 2.99 GB available → **feasible** |

→ Decision: run hierarchical clustering and silhouette on the **full** dataset.
No downsampling. t-SNE subsampled (3,000 rows) and labelled on the figure.

### Development runs (`--mode dev`) — NOT reported figures
| Track | Time | Purpose |
|---|---|---|
| regression | 0.6 min | shake out errors |
| classification | 4.6 min | surfaced an MLP `ConvergenceWarning` |
| clustering | 1.0 min | verify sweeps and embeddings |

**Fix applied:** MLP `max_iter` 600 → 1000 after the convergence warning. The
warning was acted on, not suppressed.

### Full runs (`--mode full`) — these are the reported figures
| Track | Time | Key outcome |
|---|---|---|
| regression | 0.6 min | best test R² 0.98899 (Polynomial deg 2); all 10 within 1.2 % R² |
| classification | 4.5 min | best weighted F1 0.9859 (Bagging); failure recall spans 0.10–0.75 |
| clustering | 2.4 min | k=5 by elbow (silhouette prefers k=2); K-Means sil 0.1975 vs Agglomerative 0.1612 |

### Notable honest results
- **Regression tuning moved CV R² in the 5th decimal.** Ridge `alpha=0.1`,
  Lasso `alpha=0.001`. Recorded as-is.
- **MLP tuning produced zero improvement** — GridSearchCV returned the default
  `(64,32)/relu`. Recorded as-is; no alternative grid was substituted to
  manufacture a gain.
- **Lasso zeroed 0 of 5 coefficients** — no sparsity to report.
- **Polynomial degree 3 was worse than degree 2** on test R² (0.98896 vs
  0.98899) despite 55 terms vs 20.
- **Single-linkage Calinski-Harabasz = 3.7** vs ward's 1944 — classic
  one-giant-cluster degeneracy, visible in the metrics.
- **Agglomerative stability ARI 0.31** vs K-Means 0.87 over 20 resamples.

### Verification
| Check | Result |
|---|---|
| `pytest tests/` | 15/15 passed |
| Notebook execution | all three ran top-to-bottom, no cell errors |
| Stored R² reproduces from scratch | exact to 1e-9 |
| Saved pipelines round-trip | identical predictions from raw-shaped input |

### Not done
- Public deployment (no URL claimed)
- All team-authored analysis (deliberately)
