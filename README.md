# 23CSE301 Machine Learning — Capstone Project

**B.Tech. Computer Science and Engineering, III Year · Academic Year 2026–27**
Three-member team · Three tracks: Regression, Classification, Clustering

---

## ⚠️ Project status — read this first

| | |
|---|---|
| **Implemented** | All 22 algorithms (10 regression, 10 classification, 2 clustering), full pipelines, all required visualisations, tuning, validation tooling, Streamlit GUI |
| **Executed** | All three tracks run end-to-end on the real data; all three notebooks run top-to-bottom with outputs visible; 15/15 tests pass |
| **Awaiting team input** | Feature engineering (rubric B3) · every EDA observation · every interpretation · model-selection arguments · conclusions · clustering feature justification · choice of *k* |

> **This project is NOT submission-ready.** Guidelines §7.5 requires that
> analysis, interpretation and feature-engineering decisions are the team's own
> work. Those sections are deliberately empty. Run
> `python scripts/validate_project.py` for the live checklist, and work through
> `docs/team_analysis_prompts.md`.

---

## 1. Overview

An end-to-end ML pipeline across three problem tracks: data loading, EDA,
cleaning, model training and comparison, hyperparameter tuning, and result
visualisation. Review 1 covers the full regression track plus classification
Part A; Review 2 covers classification Part B plus the full clustering track.

### Problem statements — **DRAFTS for team review** *(rewrite these yourselves)*

**Regression.** Given a student's study hours, previous scores, sleep hours,
practice-paper count and extracurricular participation, predict their
Performance Index. The practical question is which controllable habits carry
predictive weight once prior attainment is accounted for.

**Classification.** Given a machine's product-quality class, air and process
temperature, rotational speed, torque and tool wear, predict whether the machine
will fail. The operational question is whether failures can be caught early
enough to act on, given that failures are rare.

**Clustering.** Given credit-card customers' demographic and transaction
behaviour, discover natural customer segments without using any label, then
describe those segments in terms a bank could act on.

*(These are drafts written as scaffolding. Q-X1 in `docs/team_analysis_prompts.md`
asks you to replace them with your own wording.)*

---

## 2. Datasets

| Track | Dataset | Raw shape | Prepared shape | Target | Synthetic? |
|---|---|---|---|---|---|
| Regression | [Student Performance](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression) | 10,000 × 6 | 10,000 × 6 | `Performance Index` (continuous) | **Yes** |
| Classification | [Machine Predictive Maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) | 10,000 × 10 | 10,000 × 7 | `Target` (binary, 3.39 % positive) | **Yes** |
| Clustering | [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) | 10,127 × 23 | 10,127 × 7 | none (unsupervised) | No |

### 🔬 Synthetic-data disclosure
**Both supervised datasets are synthetic.** No real student and no real machine
was measured. Nothing here supports a claim about real students or real
equipment; the datasets are teaching instruments with clean, well-behaved
generating processes — which is itself why nearly every regression model lands
within 1 % R² of every other.

### Exclusions and why

| Dataset | Excluded | Reason |
|---|---|---|
| Predictive Maintenance | `Failure Type` | **Target leakage** — a second recording of the same event as `Target` |
| Predictive Maintenance | `UDI`, `Product ID` | row identifiers |
| BankChurners | `Attrition_Flag` | **withheld** — post-hoc validation only, never used to select features, scale, choose *k*, or fit |
| BankChurners | `CLIENTNUM` | customer identifier |
| BankChurners | `Naive_Bayes_Classifier_...` ×2 | the dataset author instructs users to delete them |

Full provenance, checksums and licence pointers: **`docs/dataset_sources.md`**.

---

## 3. Setup

Python 3.12 (3.10+ works). Roughly 8 minutes for the full run on one CPU core.

**macOS / Linux**
```bash
git clone <your-repo-url> && cd 23CSE301-ML-Capstone
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py      # verifies data/raw/, prints instructions if absent
```

**Windows (PowerShell)**
```powershell
git clone <your-repo-url>; cd 23CSE301-ML-Capstone
py -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_data.py
```

The three CSVs must sit in `data/raw/`. They are **not** committed — `.gitignore`
excludes them. `scripts/download_data.py` tries the Kaggle API first and prints
exact manual steps otherwise. **Never put credentials in source code**; Kaggle
tokens belong in `~/.kaggle/kaggle.json`, which is git-ignored.

### Running everything

```bash
python scripts/run_all.py --mode full          # all three tracks
python scripts/run_all.py --track regression   # one track
python scripts/run_all.py --mode dev           # fast development pass ('_dev' outputs)
python scripts/build_notebooks.py              # regenerate notebooks
python scripts/execute_notebooks.py            # run them top-to-bottom
python -m pytest tests/ -v                     # 15 focused tests
python scripts/validate_project.py             # rubric + integrity report
streamlit run app/streamlit_app.py             # GUI (bonus)
```

`--mode dev` exists so a quick pass is never mistaken for the reported figures:
its outputs carry a `_dev` suffix and the run prints a `DEVELOPMENT RUN` banner.
**Every number below comes from `--mode full`.**

---

## 4. Methodology — how leakage is prevented

* Held-out 80:20 split established **before** any data-driven modelling decision.
* Every model is a `Pipeline(preprocess → estimator)`, so imputers, encoders and
  scalers are refitted on the training part of **every** CV fold — never on the
  full dataset (§7.1: fitting on full data is a mark deduction).
* Model selection uses **training-side** cross-validation. The held-out set is
  scored once, at the end, and never tuned against.
* The regression target is never scaled; R², RMSE and MAE are in Performance-Index points.
* `random_state=42` throughout.
* Classification uses a stratified split (3.39 % positive class). Regression does
  not — see `docs/instructor_clarifications.md` **IC-2**.

---

## 5. Results

### 5.1 Regression — all ten models, same held-out split, ranked by test R²

| Rank | Model | R² | RMSE | MAE | Train R² |
|---:|---|---:|---:|---:|---:|
| 1 | Polynomial (deg 2) + Linear | 0.988989 | 2.0201 | 1.6115 | 0.988705 |
| 2 | Linear Regression | 0.988983 | 2.0206 | 1.6111 | 0.988690 |
| 3 | Ridge Regression | 0.988982 | 2.0207 | 1.6112 | 0.988690 |
| 4 | Lasso Regression | 0.988969 | 2.0218 | 1.6117 | 0.988688 |
| 5 | ElasticNet Regression | 0.988886 | 2.0295 | 1.6174 | 0.988661 |
| 6 | Gradient Boosting Regressor | 0.988411 | 2.0723 | 1.6432 | 0.989104 |
| 7 | Random Forest Regressor | 0.986175 | 2.2635 | 1.8097 | 0.997498 |
| 8 | Support Vector Regressor | 0.985924 | 2.2839 | 1.7928 | 0.986171 |
| 9 | Decision Tree Regressor | 0.983901 | 2.4426 | 1.9514 | 0.985967 |
| 10 | K-Nearest Neighbors Regressor | 0.976841 | 2.9296 | 2.3615 | 0.984027 |

**5-fold CV R² (leaders nominated on training CV):** Linear Regression
0.98866 ± 0.00028 · Ridge Regression 0.98866 ± 0.00028.

**Tuning (GridSearchCV, 2 models).** Both improved — but in the **fifth decimal
place**, which is the honest result:

| Model | Best params | CV R² before → after | Held-out R² before → after |
|---|---|---|---|
| Ridge | `alpha=0.1` | 0.988660 → 0.988660 | 0.98898 → 0.98898 |
| Lasso | `alpha=0.001` | 0.988660 → 0.988660 | 0.98897 → 0.98898 |

**Polynomial degrees:** deg 1 → test R² 0.98898 · deg 2 → 0.98899 (20 terms) ·
deg 3 → 0.98896 (55 terms). **Lasso sparsity:** 0 of 5 coefficients driven to zero.

### 5.2 Classification — consolidated 10-algorithm table (Review 2 deliverable D6)

Precision/Recall are **binary with failure (`Target == 1`) as the positive
class**; F1 is **weighted**; ROC-AUC uses `predict_proba` for all ten models.
**Majority-class reference: accuracy 0.9660, weighted F1 0.9493.**

| Rank | Model | Accuracy | Precision (fail) | Recall (fail) | F1 (weighted) | ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|
| 1 | B9. Bagging (Decision Tree base) | 0.9865 | 0.8596 | 0.7206 | 0.9859 | 0.9598 |
| 2 | B8. Gradient Boosting Classifier | 0.9860 | 0.8448 | 0.7206 | 0.9855 | 0.9673 |
| 3 | B10. MLP Classifier | 0.9810 | 0.7344 | 0.6912 | 0.9807 | 0.9789 |
| 4 | B6. Random Forest Classifier | 0.9815 | 0.8974 | 0.5147 | 0.9791 | 0.9709 |
| 5 | A4. Decision Tree Classifier | 0.9755 | 0.7879 | 0.3824 | 0.9714 | 0.9219 |
| 6 | A2. K-Nearest Neighbors | 0.9740 | 0.8333 | 0.2941 | 0.9679 | 0.8291 |
| 7 | B7. AdaBoost Classifier | 0.9725 | 0.7407 | 0.2941 | 0.9667 | 0.9492 |
| 8 | A5. Support Vector Classifier | 0.9720 | 0.8750 | 0.2059 | 0.9635 | 0.9468 |
| 9 | A1. Logistic Regression | 0.9675 | 0.6364 | 0.1029 | 0.9560 | 0.8994 |
| 10 | A3. Gaussian Naive Bayes | 0.9580 | 0.2500 | 0.1176 | 0.9506 | 0.8468 |

Rows 5, 6, 8, 9, 10 are the Part A models (Review 1); the rest are Part B.

**Tuning (GridSearchCV, 2 models selected on training CV):**

| Model | Best params | CV F1w before → after | Held-out recall (fail) before → after | Improved on CV? |
|---|---|---|---|---|
| B9. Bagging | `max_samples=0.8, n_estimators=100` | 0.98512 → 0.98526 | 0.7206 → 0.7500 | Yes |
| B10. MLP | `activation=relu, hidden=(64,32)` | 0.98301 → 0.98301 | 0.6912 → 0.6912 | **No — grid search returned the default configuration** |

> Reported as measured. Tuning the MLP produced no improvement, and that is
> stated rather than hidden.

### 5.3 Clustering — both algorithms at k = 5

*k* = 5 comes from the numerical elbow of the inertia curve. The silhouette
maximum is at k = 2. **They disagree, and the team must decide** (Q-CL2).

| Algorithm | k | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---:|---:|---:|---:|
| K-Means | 5 | 0.1975 | 1.4304 | 2373.4 |
| Agglomerative (ward) | 5 | 0.1612 | 1.6437 | 1944.1 |

Silhouette is **exact** (full 10,127 rows — memory was checked first, not assumed).

**Linkage comparison at k = 5:** ward 0.1612 · complete 0.1377 · average 0.1619
· single 0.1379. Single linkage's Calinski-Harabasz collapses to **3.7**, the
signature of one giant cluster plus slivers — check sizes before trusting a
linkage's silhouette.

**Supplementary stability (Adjusted Rand Index over 20 resamples — NOT
cross-validation, see IC-3):** K-Means 0.867 ± 0.130 · Agglomerative 0.307 ± 0.059.

**Post-hoc `Attrition_Flag` check** (run only after k and the models were fixed):
cluster attrition rates range from 4.5 % to 24.0 %. This was **not** used to
choose anything.

PCA components used for plotting explain 29.7 % + 25.3 % = **55.0 %** of variance.
Both algorithms were fitted on all **seven** standardised features, never on the
two components.

---

## 6. Repository structure

```
├── README.md                  ← you are here
├── requirements.txt           pinned to the versions actually tested
├── CLAUDE.md                  project rules + progress for later sessions
├── .gitignore                 credentials, venvs, caches, raw data
├── data/
│   ├── README.md              how to obtain the data
│   ├── raw/                   immutable source CSVs (git-ignored)
│   └── processed/             derived artifacts (regenerable)
├── notebooks/
│   ├── regression.ipynb       Review 1 — full track
│   ├── classification.ipynb   Part A (R1) + Part B (R2)
│   └── clustering.ipynb       Review 2 — full track
├── src/                       the single shared implementation
│   ├── config.py              paths, seed, dataset contracts
│   ├── data_loading.py        loading, audit, provenance, checksums
│   ├── preprocessing.py       splits, column transformers, leakage checks
│   ├── feature_engineering.py ⚠️ EMPTY HOOK — team input required
│   ├── evaluation.py          metrics + comparison tables
│   ├── plotting.py            every figure
│   ├── regression_models.py   the 10 regressors + tuning grids
│   ├── classification_models.py  the 10 classifiers + tuning grids
│   └── clustering_models.py   K-Means, Agglomerative, stability, embeddings
├── scripts/
│   ├── download_data.py       acquisition + verification
│   ├── run_all.py             the execution engine
│   ├── build_notebooks.py     generates the notebooks
│   ├── execute_notebooks.py   runs them top-to-bottom
│   └── validate_project.py    rubric + integrity checks
├── results/{tables,figures,tuning}/   every computed artifact
├── models/                    saved pipelines (preprocessing included)
├── app/streamlit_app.py       GUI (bonus)
├── docs/                      rubric checklist, sources, clarifications,
│                              team prompts, review outlines, experiment log
└── tests/                     15 focused tests
```

---

## 7. Review scope

| | Review 1 | Review 2 |
|---|---|---|
| **Scope** | Full regression track + classification Part A | Classification Part B + full clustering |
| **Notebooks** | `regression.ipynb`, `classification.ipynb` §1–4 | `classification.ipynb` §5–9, `clustering.ipynb` |
| **Status** | Code ✅ executed · Analysis ❌ unwritten | Code ✅ executed · Analysis ❌ unwritten |

Per-criterion mapping: **`docs/rubric_checklist.md`**.

---

## 8. 🤖 AI-Assistance Disclosure

*Required by guidelines §7.5: "Generative AI tools may be used for code
scaffolding but not for analysis or interpretation. If AI assistance is used,
cite it in the README."*

**Tool used:** Claude (Anthropic), via the Claude chat interface, September 2026.

**What AI generated:**
* All Python module, script and notebook **scaffolding** in `src/`, `scripts/`,
  `tests/` and `app/`
* Plot-generation code and figure styling
* Structural documentation: repository layout, this README's structure, the
  rubric checklist, review outlines, and the guiding questions in
  `docs/team_analysis_prompts.md`
* Descriptions of **what the code does**
* The **draft** problem statements in §1, flagged as drafts for team rewriting

**What AI did NOT generate, and must not:**
* Any EDA observation or interpretation of any plot
* Any feature-engineering decision or its justification —
  `src/feature_engineering.py` ships **empty on purpose**
* Any model-selection argument or conclusion
* Any cluster naming or business interpretation
* Any answer in `docs/team_analysis_prompts.md`

The notebooks mark these locations with **✍️ TEAM TO COMPLETE** cells containing
guiding questions and no answers.

**Metrics:** every number in §5 was computed by executing the code on the real
data. None was estimated, predicted or transcribed from an AI's expectation.
They can be reproduced with `python scripts/run_all.py --mode full` and
independently re-derived by `python scripts/validate_project.py`.

### Other source citations
* **scikit-learn** (BSD-3-Clause) — estimators, metrics, splitters, transformers
* **SciPy** (BSD-3-Clause) — `scipy.cluster.hierarchy` linkage and dendrogram
* **Knee detection** in `clustering_models.elbow_knee` implements the
  perpendicular-distance idea described in Satopaa et al. (2011), *"Finding a
  Kneedle in a Haystack: Detecting Knee Points in System Behavior"*, ICDCS
  Workshops. Written from the description; no code copied.
* No StackOverflow or blog code was pasted into this repository.

---

## 9. Outstanding team-authored sections

| # | Item | Rubric impact | Where |
|---|---|---|---|
| 1 | **Engineered feature — regression** | Review 1 **B3 (1 mark)** | Q-RG3 |
| 2 | **Engineered feature — classification** | Review 1 **B3** | Q-CF1 |
| 3 | EDA observations (all three notebooks) | Review 1 **A3 (1 mark)** | ✍️ cells |
| 4 | Clustering feature justification | Review 2 B1 defensibility | Q-CL1 |
| 5 | Choice of *k* | Review 2 **B2** | Q-CL2 |
| 6 | Cluster naming and interpretation | Review 2 **B3**, viva | Q-CL3 |
| 7 | Final classification model justification | Review 2 **A3 (2 marks)** | Q-CF2 |
| 8 | Duplicate-row policy | Review 1 **B1** | Q-RG1 |
| 9 | Problem statements (final wording) | Review 2 **C2** | Q-X1 |
| 10 | Track conclusions | Review 2 **D1** | ✍️ cells |
| 11 | Raise IC-1 … IC-5 with the instructor | possibly all of it (IC-1) | — |

---

## 10. Limitations requiring team discussion

1. **Both supervised datasets are synthetic.** Clean generating processes are
   why ten very different regression algorithms agree to within 1 % R². Do not
   read that as "the problem is solved."
2. **Regression: 127 duplicate rows, ~12.9 % feature-row overlap between train
   and test.** With five low-cardinality inputs this is arithmetically expected,
   not a pipeline bug — but decide it yourselves (Q-RG1, Q-RG2).
3. **Classification: 3.39 % positive class.** Weighted F1 is flattered by the
   majority class; the majority-class baseline already scores 0.9493. Failure
   recall is the metric that discriminates, and it ranges from 0.10 to 0.75.
4. **Clustering: silhouette ≈ 0.20.** Weak separation. Honest reading: this may
   be a continuum with a partition imposed on it, not five natural groups.
5. **Agglomerative clustering is unstable under resampling** (ARI 0.31 vs
   K-Means' 0.87). Cluster descriptions from it deserve less confidence.
6. **PCA shows only 55 % of variance.** Clusters that look overlapping in the
   plot may be separated in the dimensions not shown.
7. **No instructor confirmation on IC-1** (were datasets assigned?). This is the
   one open question that could invalidate everything.

---

## 11. Bonus status

| Bonus | Status |
|---|---|
| +1 Interactive GUI | ✅ **Built** — `app/streamlit_app.py`, loads saved pipelines, runs locally |
| +1 Public deployment | ❌ **Not claimed.** The app has not been deployed and there is no public URL. Claiming this without a working verified URL would be a false claim. |

To deploy later: push to GitHub, then Streamlit Community Cloud → "New app" →
point at `app/streamlit_app.py`. Note that `data/raw/` is git-ignored, so the
deployment needs `models/*.joblib` committed (they are) — the app loads saved
pipelines and does not need the raw CSVs.

See `docs/instructor_clarifications.md` **IC-4** on the contradictory bonus cap.
