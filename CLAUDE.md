# CLAUDE.md — Project Rules and Progress

Read this first in any later session. It records the **rules** this project runs
under and the **actual** state of the work.

---

## 1. Hard rules (do not violate)

### Academic integrity (guidelines §7.5)
AI may generate code scaffolding. AI must **NOT** generate:
- EDA observations or plot interpretations
- Feature-engineering decisions or their justifications
- Model-selection arguments
- Cluster naming or business interpretation
- Conclusions

`src/feature_engineering.py` is **empty on purpose**. Do not populate it with an
invented feature. Do not answer anything in `docs/team_analysis_prompts.md`.
Do not fill any `✍️ TEAM TO COMPLETE` notebook cell.

### Honesty
- Never fabricate a result, improvement, execution status, or commit.
- If tuning does not help, report that it did not. (It already happened: MLP
  tuning returned the default config with zero gain — that stands in the README.)
- Distinguish **implemented** / **executed** / **awaiting team input**.
- Never mark the project submission-ready while team sections are outstanding.
- Label anything sampled or approximated (t-SNE subsample, any sampled metric).

### Data
- `data/raw/` is immutable. No script writes there.
- Never substitute a dataset or manufacture rows to work around a problem.
- Never commit credentials. Kaggle tokens live in `~/.kaggle/kaggle.json`.

### Methodology
- `random_state=42` everywhere.
- Split before any data-driven decision.
- All transformers inside Pipelines → refitted per CV fold.
- Select models on **training** CV; the held-out set is scored once.
- Never scale the regression target.
- Clustering: `Attrition_Flag` is withheld from every fitting decision.

---

## 2. Architecture

One implementation, in `src/`. Notebooks and `scripts/run_all.py` both import it,
so they cannot drift apart. Do not copy pipeline logic into a notebook.

```
src/config.py                paths, seed, dataset contracts (edit specs here)
src/data_loading.py          load + audit + provenance/checksums
src/preprocessing.py         splits, ColumnTransformers, leakage checks
src/feature_engineering.py   EMPTY HOOK - team input required
src/evaluation.py            metrics + comparison tables + persistence
src/plotting.py              every figure (styling rules enforced here)
src/{regression,classification,clustering}_models.py   models + tuning grids
```

Adding a model = add it to the right `build_*` function and, if tunable, to
`PARAM_GRIDS`. Nothing else changes; the tables and plots pick it up.

---

## 3. Commands

```bash
python scripts/download_data.py        # verify/acquire data
python scripts/run_all.py --mode full  # the reported run
python scripts/run_all.py --mode dev   # fast pass; outputs suffixed _dev
python scripts/build_notebooks.py      # regenerate notebooks from source
python scripts/execute_notebooks.py    # run them top-to-bottom
python -m pytest tests/ -v
python scripts/validate_project.py     # live rubric + integrity status
streamlit run app/streamlit_app.py
```

**`--mode dev` results are never the reported figures.** They carry a `_dev`
suffix and the run prints a DEVELOPMENT RUN banner.

---

## 4. Progress log

### Completed and executed
- [x] Dataset acquisition verified; all three shapes match the contract exactly
- [x] Provenance + SHA-256 checksums recorded
- [x] Regression: 10 models, CV, 2× GridSearchCV, all required figures
- [x] Classification: Part A (5) + Part B (5), consolidated 10-model table,
      CV, 2× GridSearchCV, ROC curves, tree viz, odds, importances
- [x] Clustering: K-Means k=2..10, Agglomerative (4 linkages), exact silhouette
      on the full 10,127 rows, elbow, dendrogram, PCA ×2, t-SNE, stability
- [x] All three notebooks executed top-to-bottom with outputs
- [x] 15 focused tests passing
- [x] Streamlit app (local only — NOT deployed)
- [x] README with computed numbers + AI disclosure

### Outstanding — team only
- [ ] **Feature engineering, both supervised tracks** (rubric B3) — Q-RG3, Q-CF1
- [ ] All EDA observations (rubric A3) — ✍️ cells
- [ ] Clustering feature justification (Q-CL1) — subset is PROVISIONAL
- [ ] Choice of *k* (Q-CL2) — currently the numerical elbow, k=5, as a placeholder
- [ ] Cluster interpretation (Q-CL3)
- [ ] Final classification model justification (Q-CF2)
- [ ] Duplicate-row policy (Q-RG1)
- [ ] Track conclusions
- [ ] Raise IC-1 … IC-5 with the instructor

---

## 5. Known issues and decisions already taken

| Item | Decision | Revisit? |
|---|---|---|
| 127 duplicate regression rows | **retained**; dedup would give 9,873 rows | Q-RG1 |
| ~12.9 % train/test feature-row overlap | reported, not "fixed" | Q-RG2 |
| Regression stratification | not stratified; binned option available | IC-2 |
| Clustering "cross-validation" | stability resampling, labelled NOT CV | IC-3 |
| t-SNE | 3,000-row subsample, stated on the figure | IC-5 |
| k = 5 | numerical elbow; silhouette prefers k=2 | Q-CL2 |
| MLP tuning | no improvement — reported as such | — |
| Public deployment | **not done, not claimed** | — |

---

## 6. Environment

Python 3.12.3, single CPU core. scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4.
Note `sklearn.metrics.root_mean_squared_error` is used (the `squared=False`
argument was removed in newer scikit-learn). Full run ≈ 8 minutes.
