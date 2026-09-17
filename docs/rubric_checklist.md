# Rubric Checklist — Requirement → Artifact → Status

**Legend:** ✅ done & executed · ⚠️ done, team confirmation needed · ❌ requires
team-authored work (AI must not supply it) · ➖ not applicable yet

Live version: `python scripts/validate_project.py` → `results/tables/validation_report.csv`.

---

# Review 1 — 25 marks

## Section A — Dataset & EDA (4)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| A1 | Shape, dtypes, missing counts, target distribution | 1 | `regression.ipynb` §1 · `classification.ipynb` §1 · `results/tables/*_audit_raw.csv` | ✅ |
| A2 | Distribution plot per feature | 2 | `reg_feature_distributions.png`, `clf_feature_distributions.png` | ✅ |
| A2 | Correlation heatmap | ↑ | `reg_correlation_heatmap.png`, `clf_correlation_heatmap.png` | ✅ |
| A2 | Target distribution | ↑ | `reg_target_distribution.png`, `clf_target_distribution.png` | ✅ |
| A2 | ≥2 feature–target scatter plots | ↑ | `reg_feature_target_scatter.png` (2 panels) · `clf_feature_target_scatter.png` (2, jittered) + `clf_feature_by_class.png` companion | ✅ |
| A3 | **Written observation after each major visualisation** | 1 | ✍️ cells in both notebooks | ❌ **team** |

## Section B — Preprocessing & Feature Engineering (3)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| B1 | Missing values, justified strategy | 1 | `preprocessing.make_preprocessor` (median / most-frequent, inside pipelines). **0 missing cells in all three datasets** | ✅ |
| B1 | Duplicates checked and treated | ↑ | 127 found in regression; `results/tables/regression_duplicate_policy.json` | ⚠️ **Q-RG1** |
| B1 | Outliers checked, not blanket-deleted | ↑ | distribution grids + `grouped_box`; nothing auto-deleted | ⚠️ team to comment |
| B2 | Encoding for categoricals | 1 | `OneHotEncoder(handle_unknown='ignore', drop='if_binary')` in every pipeline | ✅ |
| B2 | Scaler fitted on train only | ↑ | `StandardScaler` inside Pipeline → refit per CV fold. Test: `test_scaler_is_fitted_per_fold_not_on_full_data` | ✅ |
| B2 | Stratified split | ↑ | classification ✅ stratified; regression **not** stratified (continuous target) | ⚠️ **IC-2** |
| B3 | **≥1 engineered feature + written justification** | 1 | `src/feature_engineering.py` — **EMPTY BY DESIGN** | ❌ **team — Q-RG3 / Q-CF1** |

## Section C — Regression Track (9)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| C1 | All 10 algorithms trained, no errors | 4 | `regression_models.build_models` → 10 pipelines; `regression.ipynb` §4 | ✅ |
| C2 | Single table: R², RMSE, MAE, ranked by R² | 2 | `results/tables/regression_comparison.csv` — 10 rows, monotonic descending | ✅ |
| C3 | GridSearchCV on ≥2 models, best params + improvement | 2 | `results/tables/regression_tuning.csv` (Ridge, Lasso) + `results/tuning/*.csv` | ✅ *(gains are 5th-decimal — reported honestly)* |
| C4 | Residual plot + predicted-vs-actual for best model | 1 | `reg_residuals.png`, `reg_pred_vs_actual.png` | ✅ |
| C4 | Feature-importance plot, ≥1 tree model | ↑ | `reg_feature_importance.png` (Random Forest) | ✅ |
| — | 5-fold CV R² for the two best models | mandatory | `regression_cv_top2.csv` — leaders picked on **training** CV | ✅ |
| — | Coefficients (#1) | PDF note | `regression_linear_coefficients.csv` + plot | ✅ |
| — | Lasso sparsity (#3) | PDF note | `regression_lasso_sparsity.json` — **0 of 5 zeroed** | ✅ |
| — | Polynomial degree comparison (#5) | PDF note | `regression_polynomial_degrees.csv` — degrees 1, 2, 3 | ✅ |

## Section D — Classification Part A (3)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| D1 | All 5 Part-A algorithms, no errors | 2 | `build_part_a` — LogReg, KNN, **Gaussian**NB, DT, SVC | ✅ |
| D2 | Accuracy, weighted F1, confusion matrix per algorithm | 1 | `classification_partA_comparison.csv`, `clf_confusion_partA.png` | ✅ |
| D2 | Preliminary comparison table | ↑ | same | ✅ |

## Section E — Presentation (1) + Viva (5)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| E1 | Clear narrative; every member can explain their code | 1 | `docs/review1_outline.md` | ❌ **team** |
| — | Viva | 5 | `docs/team_analysis_prompts.md` ⚪ items | ❌ **team** |

---

# Review 2 — 25 marks (+ up to 2 bonus)

## Section A — Classification Part B (6)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| A1 | All 5 Part-B algorithms, same dataset/split as Part A | 3 | `build_part_b` — RF, AdaBoost, GB, Bagging(DT base), MLP | ✅ |
| A2 | **Single table, all 10, Accuracy/Precision/Recall/F1/ROC-AUC** | 1 | `classification_all10_comparison.csv` — 10 rows, all columns, AUC from `predict_proba` | ✅ |
| A3 | Best model identified with clear justification | 2 | table ranks Bagging first | ❌ **justification: team — Q-CF2** |
| A3 | Tuning applied, ≥1 metric improved, documented | ↑ | `classification_tuning.csv`: Bagging recall 0.7206 → 0.7500 ✅; **MLP: no improvement, reported** | ✅ |
| — | Majority-class reference | requested | `classification_majority_baseline.json` — acc 0.9660, F1w 0.9493 | ✅ |
| — | Odds/coefficients (#1) | PDF note | `classification_logreg_odds.csv` + plot | ✅ |
| — | Distance metrics (#2) | PDF note | `classification_knn_distance_metrics.csv` | ✅ |
| — | Tree visualisation (#4) | PDF note | `clf_decision_tree.png` (depth-3 display) | ✅ |
| — | Feature importance (#6) | PDF note | `clf_feature_importance.png` | ✅ |
| — | CV for the two leading models | §7.2 | `classification_cv_top2.csv` | ✅ |

## Section B — Clustering Track (8)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| B1 | Both algorithms fitted, labels obtained, no errors | 4 | `clustering.ipynb` §4–5; fitted on all 7 standardised features | ✅ |
| B2 | Silhouette, Davies-Bouldin, Calinski-Harabasz for both | 2 | `clustering_final_metrics.csv` — **exact**, full 10,127 rows | ✅ |
| B2 | Elbow curve (K-Means) | ↑ | `clu_kmeans_elbow.png` (inertia + silhouette overlay) | ✅ |
| B2 | Dendrogram (Hierarchical) | ↑ | `clu_dendrogram_ward.png` — real fitted hierarchy, display truncated to 30 merges | ✅ |
| B3 | PCA 2D scatter, colour-coded, **for every algorithm** | 2 | `clu_pca_kmeans.png`, `clu_pca_agglomerative.png` | ✅ |
| B3 | t-SNE for ≥1 algorithm | ↑ | `clu_tsne_kmeans.png` — 3,000-row subsample, stated on the figure | ✅ |
| — | Linkage comparison | PDF note | `clustering_linkage_comparison.csv` — ward/complete/average/single | ✅ |
| — | Labels not used during fitting | §3.3 | `Attrition_Flag` withheld; test `test_attrition_flag_never_reaches_the_clustering_frame` | ✅ |
| — | Cluster interpretation | viva | profiles computed; **naming and meaning** | ❌ **team — Q-CL3** |
| — | Feature subset justification | defensibility | subset is **PROVISIONAL** | ❌ **team — Q-CL1** |
| — | Choice of k | B2 | elbow k=5 used; silhouette prefers k=2 | ⚠️ **team — Q-CL2** |
| — | "CV in every track" | §7.2 | stability resampling, labelled NOT CV | ⚠️ **IC-3** |

## Section C — Pipeline Integration & Quality (3)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| C1 | Notebooks clean, run top-to-bottom, modular, commented | 1 | all three executed with outputs; logic lives in `src/` | ✅ |
| C2 | README: dataset description, problem statement, results table, setup, how-to-run | 1 | `README.md` §1–5 | ⚠️ problem statements are **drafts** — Q-X1 |
| C3 | Meaningful commit history + requirements.txt | 1 | milestone commits (see `git log`); `requirements.txt` pinned | ✅ |

## Section D — Presentation & Viva (8)

| # | Criterion | Max | Where | Status |
|---|---|---|---|---|
| D1 | Story arc, clear visuals, organised | 3 | `docs/review2_outline.md` | ❌ **team** |
| D2 | All members demonstrate understanding | 5 | `docs/team_analysis_prompts.md` | ❌ **team** |

## Bonus (up to +2)

| # | Criterion | Where | Status |
|---|---|---|---|
| +1 | Working GUI accepting inputs, returning predictions | `app/streamlit_app.py` — loads saved pipelines | ✅ built, local |
| +1 | Public deployment at a live URL | — | ❌ **not done, not claimed** |

See **IC-4** on the contradictory "capped at 20" wording.

---

## Deliverables

| # | Deliverable | R1 | R2 | Status |
|---|---|---|---|---|
| D1 | Notebooks, fully run, outputs visible | ✓ | ✓ | ✅ |
| D2 | GitHub repo, meaningful commits | ✓ | ✓ | ✅ local; **not pushed** (needs your instruction) |
| D3 | Comparative results table | ✓ | ✓ | ✅ |
| D4 | All required visualisations | ✓ | ✓ | ✅ |
| D5 | README | — | ✓ | ⚠️ drafts flagged |
| D6 | Consolidated 10-algorithm table | — | ✓ | ✅ |
| D7 | Elbow + dendrogram + PCA | — | ✓ | ✅ |
| D8 | GUI / deployment (bonus) | — | opt | ✅ GUI · ❌ deployment |

---

## Summary

**Every mechanical requirement is implemented and executed.** What remains is
the marks that depend on your own words: A3 (1), B3 (1), E1 (1) and viva (5) in
Review 1; A3 justification (2), D1 (3) and D2 (5) in Review 2 — plus the
clustering decisions that make B2/B3 defensible under questioning.
