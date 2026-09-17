"""build_notebooks.py - generate notebooks/{regression,classification,clustering}.ipynb

Structure: EVERY algorithm gets its own numbered section inside the notebook -

    Markdown : what it is (plain words), how it works (numbered steps),
               settings you can tune, what to say in the viva
    Code     : that ONE model built, trained on the data, and evaluated,
               with its own visible output
    Extras   : the algorithm-specific display the PDF asks for (coefficients,
               Lasso sparsity, degree comparison, tree plot, odds ratios, ...)

Every model trains on the SAME split created once in the preprocessing section,
so the consolidated comparison tables required by rubric C2 / A2 stay fair.

The algorithm write-ups are imported from build_examples.py, so there is one
source of truth for them.

    python scripts/build_notebooks.py          # write the .ipynb files
    python scripts/execute_notebooks.py        # run them top-to-bottom
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import nbformat as nbf

from src import config
from build_examples import CLASSIFICATION, REGRESSION

NB_DIR = config.NOTEBOOKS_DIR


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


def team(heading: str, *questions: str):
    qs = "\n".join(f"> {i}. {q}" for i, q in enumerate(questions, 1))
    return md(f"""
> ### TEAM TO COMPLETE - {heading}
>
> *Left unwritten on purpose. Guidelines section 7.5 requires analysis and
> interpretation to be the team's own work. Replace the line at the bottom with
> your own observation.*
>
> **Guiding questions**
>
{qs}
>
> *(the numbers you need are already extracted in `docs/evidence_for_team_cells.md`)*
>
> ---
>
> *(your written observation here)*
""")


def show(var: str = "p") -> str:
    return f"from IPython.display import Image, display\ndisplay(Image(filename=str({var})))"


def algo_markdown(m: dict, section: str) -> str:
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(m["how"], 1))
    return f"""
### {section} {m["title"]}

**What it is**

{m["what"]}

**How it works**

{steps}

**Settings you can tune**

{m["settings"]}

**Viva note** - {m["viva"]}
"""


def strip_pipeline_import(imports: str) -> str:
    """Pipeline is imported once in the setup cell."""
    return "\n".join(l for l in imports.splitlines()
                     if "sklearn.pipeline" not in l).strip()


HEADER = """
# 23CSE301 Machine Learning - Capstone Project
## {title}

**Track owner:** _(name - see `docs/team_contributions.md`)_
**Review:** {review}
**Dataset:** {dataset}
**Source:** {url}

---

### How to read this notebook

Each algorithm has its **own section**: first a short explanation of what it is
and how it works, then the code that builds and trains *that* model on our data,
then its result. Every model uses the same train/test split created in section 3,
which is what makes the comparison tables fair.

Pipeline steps come from `src/`, the same code `scripts/run_all.py` runs, so the
notebook and the scripts cannot drift apart.

**AI assistance:** code scaffolding was AI-generated (see `README.md`,
AI-Assistance Disclosure). Every **TEAM TO COMPLETE** cell must be written by the
team and is empty on purpose.
"""

SETUP = """
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))          # project root on the path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from src import config, data_loading as dl, preprocessing as pp
from src import feature_engineering as fe, evaluation as ev, plotting as pl

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
warnings.filterwarnings("default")                  # warnings stay visible

print("random_state =", config.RANDOM_STATE,
      "| test_size =", config.TEST_SIZE,
      "| cv folds =", config.CV_FOLDS)
"""


# ==========================================================================
def regression_nb():
    spec = config.REGRESSION
    cells = [
        md(HEADER.format(title="Regression Track - Student Performance",
                         review="Review 1 (full track)",
                         dataset=f"{spec['name']} - `{spec['filename']}`",
                         url=spec["url"])),
        md("## 0. Setup"),
        code(SETUP),
        code(f"""
TARGET = {spec["target"]!r}
NUM = {spec["numeric_features"]!r}
CAT = {spec["categorical_features"]!r}
print("target          :", TARGET)
print("numeric inputs  :", NUM)
print("categorical inpt:", CAT)
"""),

        md("""
## 1. Data loading and audit  *(Rubric A1)*

The raw CSV is read from `data/raw/` and never modified. We report shape,
dtypes, missing-value counts and the target distribution.
"""),
        code("""
raw = dl.load_raw("regression")
summary = dl.audit_summary(raw, "regression")
for k, v in summary.items():
    print(f"  {k}: {v}")
"""),
        code("""dl.audit(raw, "regression", "raw")"""),
        code("""raw.head()"""),
        code("""raw.describe(include="all").T"""),

        md("""
### 1.1 Duplicate rows - a decision for the team

127 exact duplicate rows exist. All five inputs are low-cardinality integers or
a binary flag, so identical rows arise by chance in a 10,000-row synthetic
sample. Removing them leaves 9,873 rows, **below the 10,000-row working
preference**. The reported run **retains** them; both options are measured below.
"""),
        code("""
df = dl.prepare("regression", raw)
dedup = pp.drop_exact_duplicates(df)
print(f"\\nretained : {len(df)} rows  (the reported configuration)")
print(f"if dropped: {len(dedup)} rows  -> below the 10,000 preference: {len(dedup) < 10000}")

combos = 1
for c in NUM + CAT:
    combos *= df[c].nunique()
print(f"\\ndistinct input combinations possible: {combos:,}")
print(f"rows drawn from those combinations   : {len(df):,}")
"""),
        team("Duplicate-handling decision (Q-RG1)",
             "Given that many distinct input combinations, how many identical rows would you EXPECT by chance in 10,000 draws?",
             "Coincidence, or a data-collection artefact?",
             "Which option do you choose, and why?"),

        md("""
## 2. Exploratory data analysis  *(Rubric A2, A3)*

Distribution plot per input feature, correlation heatmap, target distribution,
and two feature-target scatter plots.
"""),
        code("""
p = pl.distribution_grid(df[NUM + CAT],
                         "Student Performance - input feature distributions",
                         "nb_reg_feature_distributions", "regression")
"""),
        code(show()),
        team("Feature distributions",
             "Which features are roughly uniform and which are skewed?",
             "Does any feature show an implausible value or a suspicious spike?",
             "What does each shape imply for scaling, or for models that assume normality?"),

        code("""
p = pl.correlation_heatmap(df, "Student Performance - correlation heatmap (numeric)",
                           "nb_reg_correlation_heatmap", "regression")
"""),
        code(show()),
        code("""
corr_t = df.select_dtypes("number").corr()[TARGET].drop(TARGET)
print("correlation of each input with the target:")
print(corr_t.round(4).sort_values(ascending=False).to_string())
"""),
        team("Correlation structure",
             "Which input correlates most strongly with Performance Index, and by how much does it lead the others?",
             "Is any correlation BETWEEN inputs high enough to worry about multicollinearity in the linear models?",
             "Does the ranking match what you would expect from the domain?"),

        code("""
p = pl.target_distribution(df[TARGET], "Target distribution - Performance Index",
                           "nb_reg_target_distribution", "regression")
"""),
        code(show()),
        code("""
top2 = corr_t.abs().sort_values(ascending=False).index[:2].tolist()
print("two strongest relationships with the target:", top2)
p = pl.feature_target_scatter(df, top2, TARGET,
                              "nb_reg_feature_target_scatter", "regression")
"""),
        code(show()),
        team("Feature-target relationships",
             "Is each relationship linear, curved, or absent?",
             "How much vertical spread is there at a fixed x - how much variance can that feature alone NOT explain?",
             "Does this support a linear model as the baseline?"),

        md("""
## 3. Preprocessing and the held-out split  *(Rubric B1, B2)*

The split happens **before** any modelling decision, so the test set stays
genuinely unseen.

**Why preprocessing goes inside a Pipeline.** If we scaled the whole dataset
first, the scaler would have computed its mean and standard deviation from the
test rows too - that is data leakage, and section 7.1 makes it a mark deduction.
Inside a `Pipeline`, `.fit()` fits the scaler on training data only, and during
cross-validation it is refitted from scratch on every fold.

**Stratification note.** Rubric B2 asks for a stratified split, but
stratification needs discrete classes and our target is continuous. We use an
unstratified random split and log the ambiguity as **IC-2**.
`pp.split_supervised(..., stratify_bins=k)` implements quantile-binned
stratification if the instructor confirms it is wanted.
"""),
        code("""
X_train, X_test, y_train, y_test = pp.split_supervised(df, TARGET, stratify=False)
print(f"\\ntrain: {X_train.shape}   test: {X_test.shape}")
print("overlap check:", pp.check_no_row_leakage(X_train, X_test))
"""),
        code("""
# What the preprocessor actually does, shown once.
prep = pp.make_preprocessor(NUM, CAT, scale=True)
prep.fit(X_train)                                   # TRAINING data only
print("output feature names:", list(prep.get_feature_names_out()))
print("\\nnumeric block   : median impute -> StandardScaler")
print("categorical block: most-frequent impute -> OneHotEncoder(drop='if_binary')")
Xt = prep.transform(X_train)
print("\\ntraining data after transform - mean ~0, std ~1:")
print("  mean:", np.round(Xt[:, :len(NUM)].mean(axis=0), 6))
print("  std :", np.round(Xt[:, :len(NUM)].std(axis=0), 6))
"""),
        team("Train/test overlap (Q-RG2)",
             "Why is the overlap percentage this large here, and is it 'leakage' in the harmful sense?",
             "Would deduplicating before splitting change your answer?"),

        md("""
### 3.1 Feature engineering  *(Rubric B3)* - NOT YET SATISFIED

`src/feature_engineering.py` ships **empty on purpose**. Until the team registers
a feature *and its justification*, the baseline pipeline runs and this rubric
item reports INCOMPLETE.
"""),
        code("""fe.status("regression")"""),
        team("Engineered feature (Q-RG3) - required for rubric B3",
             "Which new feature will you create, from which existing columns?",
             "Why do you expect it to help? Write this BEFORE you measure it.",
             "After registering it in src/feature_engineering.py and re-running, did it help? Report honestly either way."),

        md("""
## 4. The ten regression algorithms  *(Rubric C1)*

Each gets its own section below: what it is, how it works, the code that trains
it on our data, and its result. All ten use the same `X_train` / `X_test` from
section 3.

The helper below records each result so section 5 can assemble the comparison
table.
"""),
        code("""
results = {}     # model name -> metrics dict
fitted  = {}     # model name -> fitted Pipeline

def record(name, model):
    \"\"\"Evaluate a fitted pipeline on the held-out test set and store it.\"\"\"
    m = ev.regression_metrics(y_test, model.predict(X_test))
    m["Train_R2"] = ev.regression_metrics(y_train, model.predict(X_train))["R2"]
    results[name] = m
    fitted[name] = model
    print(f"{name}")
    print(f"  R2   : {m['R2']:.6f}")
    print(f"  RMSE : {m['RMSE']:.4f}   (Performance Index points)")
    print(f"  MAE  : {m['MAE']:.4f}   (Performance Index points)")
    print(f"  train R2 (overfitting check): {m['Train_R2']:.6f}")
    return m
"""),
    ]

    for i, m in enumerate(REGRESSION, start=1):
        cells.append(md(algo_markdown(m, f"4.{i}")))
        cells.append(code(f"""{strip_pipeline_import(m["imports"])}

model = Pipeline([
    ("prep", pp.make_preprocessor(NUM, CAT, scale={m['scale']})),
{m['poly']}    ("model", {m['estimator']}),
])

model.fit(X_train, y_train)
record("{m['title']}", model)
"""))
        if m["extra"].strip():
            cells.append(code(m["extra"].strip()))

    cells += [
        team("Coefficients, sparsity and polynomial degree",
             "State in one sentence, in plain language, what the largest Linear Regression coefficient means for a student.",
             "Do the signs of all coefficients match your domain expectation?",
             "How many coefficients did Lasso drive to zero, and what does that imply about redundant features?",
             "Compare train R2 and test R2 as the polynomial degree rises. At which degree does overfitting begin, and how do you see it?"),

        md("""
## 5. Comparative evaluation  *(Rubric C2)*

One table, all ten models, same test split, ranked by R2.
"""),
        code("""
table = (pd.DataFrame(results).T
         .reset_index().rename(columns={"index": "Model"})
         .sort_values("R2", ascending=False).reset_index(drop=True))
table.index = table.index + 1
table.index.name = "Rank"
table = table[["Model", "R2", "RMSE", "MAE", "Train_R2"]]
print(f"{len(table)} models compared on the same held-out split\\n")
table
"""),
        code("""
p = pl.model_comparison_bar(table, "R2", "Regression models ranked by test R2",
                            "nb_reg_model_comparison", "regression")
"""),
        code(show()),
        team("Comparative results",
             "Which model ranks first, and by how much does it beat the plain linear baseline?",
             "Is that margin large enough to matter, given RMSE is in Performance-Index points?",
             "What does it tell you that regularised linear models and tree ensembles land so close together?",
             "Which model shows the largest gap between train R2 and test R2, and what is that gap called?"),

        md("""
## 6. Cross-validation  *(mandatory: 5-fold CV R2 for the two best models)*

Leading models are nominated using **training-side** cross-validation. The
held-out test set plays no part in selection.

**How 5-fold CV works:** split the training data into 5 equal parts; train on 4
and score on the 1 left out; repeat 5 times so each part is held out once;
average the 5 scores. Because the whole Pipeline is passed in, the scaler is
refitted inside every fold.
"""),
        code("""
cv_all = ev.cv_scores(fitted, X_train, y_train, scoring="r2")
cv_all.drop(columns="fold_scores").sort_values("CV_mean", ascending=False)
"""),
        code("""
leaders = cv_all.sort_values("CV_mean", ascending=False)["Model"].head(2).tolist()
print("two leading models, selected on TRAINING cross-validation:")
for l in leaders:
    print("   ", l)
cv_all[cv_all["Model"].isin(leaders)][["Model", "CV_mean", "CV_std", "fold_scores"]]
"""),

        md("""
## 7. Hyperparameter tuning  *(Rubric C3)*

`GridSearchCV` on at least two models. It tries every combination in the grid,
scoring each by cross-validation on the training data only, and keeps the best.

**If tuning does not improve a model, that is what gets reported.**
"""),
        code("""
from src import regression_models as rm

tunable = [m for m in leaders if m in rm.PARAM_GRIDS]
for cand in rm.PARAM_GRIDS:                       # ensure at least two
    if len(tunable) >= 2:
        break
    if cand not in tunable:
        tunable.append(cand)
print("tuning:", tunable)
for t in tunable:
    print(f"  grid for {t}: {rm.PARAM_GRIDS[t]}")
"""),
        code("""
rows, tuned = {}, {}
for name in tunable:
    before_cv   = float(cv_all.loc[cv_all["Model"] == name, "CV_mean"].iloc[0])
    before_test = float(table.loc[table["Model"] == name, "R2"].iloc[0])
    gs = rm.tune(name, fitted[name], X_train, y_train)
    after = ev.regression_metrics(y_test, gs.best_estimator_.predict(X_test))
    rows[name] = {
        "best_params": gs.best_params_,
        "CV_R2_before": round(before_cv, 6),
        "CV_R2_after": round(gs.best_score_, 6),
        "HeldOut_R2_before": round(before_test, 6),
        "HeldOut_R2_after": round(after["R2"], 6),
        "improved_on_CV": bool(gs.best_score_ > before_cv),
    }
    tuned[name] = gs.best_estimator_
pd.DataFrame(rows).T
"""),
        team("Tuning outcome",
             "Did tuning change the score meaningfully, or only in the fourth or fifth decimal place?",
             "What does the size of the change suggest about whether these models were under-regularised to begin with?",
             "Would you ship the tuned or the default configuration, and why?"),

        md("""
## 8. Required visualisations  *(Rubric C4)*

**How to read a residual plot.** Residual = actual - predicted. Points should
scatter randomly around the zero line with roughly constant spread. A funnel
shape means the error grows with the prediction; a curve means a non-linear term
is missing.
"""),
        code("""
best_name = table.sort_values("R2", ascending=False).iloc[0]["Model"]
best_model = tuned.get(best_name, fitted[best_name])
y_pred = best_model.predict(X_test)
print("best model on the held-out set:", best_name)

p1 = pl.residual_plot(y_test, y_pred, best_name, "nb_reg_residuals", "regression")
p2 = pl.predicted_vs_actual(y_test, y_pred, best_name, "nb_reg_pred_vs_actual", "regression")
"""),
        code(show("p1")),
        code(show("p2")),
        team("Residual diagnostics",
             "Are residuals centred on zero with roughly constant spread, or do they fan out?",
             "Does the residual histogram look approximately normal?",
             "What would a visible curve here tell you about a missing feature or a missing non-linear term?"),
        code("""
imp = rm.tree_importances(fitted["7. Random Forest Regressor"])
p = pl.importance_plot(imp.values, imp.index,
                       "Random Forest Regressor - feature importance",
                       "nb_reg_feature_importance", "regression",
                       xlabel="Gini importance")
imp.round(4)
"""),
        code(show()),
        team("Feature importance vs linear coefficients",
             "Do the Random Forest importances rank the features the same way the linear coefficients did?",
             "Where they disagree, which do you trust for this dataset, and why?"),

        md("## 9. Review 1 - regression conclusion"),
        team("Regression track conclusion",
             "Which model do you nominate as final, and on what evidence (test metric, CV stability, simplicity, training cost)?",
             "What is the practical meaning of the RMSE in Performance-Index points?",
             "What is the main limitation of this dataset, given that it is synthetic?",
             "What would you do next if you had another week?"),
    ]
    return cells


# ==========================================================================
def classification_nb():
    spec = config.CLASSIFICATION
    cells = [
        md(HEADER.format(title="Classification Track - Machine Predictive Maintenance",
                         review="Part A -> Review 1 | Part B -> Review 2",
                         dataset=f"{spec['name']} - `{spec['filename']}`",
                         url=spec["url"])),
        md("## 0. Setup"),
        code(SETUP),
        code(f"""
TARGET = "Target"
CLS = ["No Failure", "Failure"]
NUM = {spec["numeric_features"]!r}
CAT = {spec["categorical_features"]!r}
print("target:", TARGET, "| classes:", CLS)
"""),

        md("""
## 1. Data loading, audit and leakage exclusions  *(Rubric A1)*

**Target leakage, handled explicitly.** `Failure Type` is **not** a predictor: it
records the same event as `Target`. A model given it would score near-perfectly
and learn nothing. `UDI` and `Product ID` are row identifiers. All three are
dropped.
"""),
        code("""
raw = dl.load_raw("classification")
summary = dl.audit_summary(raw, "classification")
for k, v in summary.items():
    print(f"  {k}: {v}")
"""),
        code("""
# The evidence for excluding Failure Type: it is determined by Target.
pd.crosstab(raw["Failure Type"], raw["Target"])
"""),
        code("""dl.audit(raw, "classification", "raw")"""),
        code("""
df = dl.prepare("classification", raw)
print("\\nprepared columns:", list(df.columns))
df.head()
"""),

        md("""
## 2. Exploratory data analysis  *(Rubric A2, A3)*

The target is binary and heavily imbalanced, so a plain feature-vs-target
scatter would collapse onto two horizontal lines. We add jitter and a companion
box plot.
"""),
        code("""
p = pl.distribution_grid(df[NUM + CAT],
                         "Predictive Maintenance - input feature distributions",
                         "nb_clf_feature_distributions", "classification")
"""),
        code(show()),
        team("Feature distributions",
             "Which features look approximately normal, and which are skewed or bounded?",
             "Air and process temperature: what does their shape suggest about how this synthetic data was generated?",
             "Does the Type category split the machines evenly?"),
        code("""
p = pl.correlation_heatmap(df, "Predictive Maintenance - correlation heatmap (numeric)",
                           "nb_clf_correlation_heatmap", "classification")
"""),
        code(show()),
        code("""df.select_dtypes("number").corr().round(3)"""),
        team("Correlation structure",
             "Two pairs of features are strongly related. Which pairs, and why physically?",
             "Does any single feature correlate strongly with Target on its own? What does that imply about whether failure is a single-variable phenomenon?"),
        code("""
p = pl.target_distribution(df[TARGET], "Target distribution - machine failure",
                           "nb_clf_target_distribution", "classification",
                           discrete=True, class_names={0: CLS[0], 1: CLS[1]})
"""),
        code(show()),
        team("Class imbalance",
             "What fraction of rows are failures?",
             "What accuracy would a model get by predicting 'No Failure' every single time?",
             "Which metric should therefore lead your comparison, and why not accuracy?"),
        code("""
p = pl.feature_target_scatter(df, ["Torque [Nm]", "Rotational speed [rpm]"], TARGET,
                              "nb_clf_feature_target_scatter", "classification",
                              jitter=0.08,
                              title="Feature-target relationships (target jittered for readability)")
"""),
        code(show()),
        code("""
p = pl.grouped_box(df, ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]"],
                   TARGET, "nb_clf_feature_by_class", "classification",
                   class_names={0: CLS[0], 1: CLS[1]})
"""),
        code(show()),
        code("""
print("mean of each feature, by class:")
df.groupby(TARGET).mean(numeric_only=True).round(3).T
"""),
        team("Feature-target relationships",
             "In which region of torque and rotational speed do failures concentrate?",
             "Do the box plots separate the two classes on any single feature, or is failure an interaction between features?",
             "What does that suggest about linear versus tree-based models here?"),

        md("""
## 3. Split, baseline and feature engineering  *(Rubric B1, B2, B3)*

**Stratified** 80:20. With only 3.4% positives, an unstratified split could give
train and test noticeably different failure rates and make the comparison unfair.

As in the regression notebook, all preprocessing lives inside each Pipeline, so
scalers and encoders are fitted on training data only and refitted per CV fold.
"""),
        code("""
X_train, X_test, y_train, y_test = pp.split_supervised(df, TARGET, stratify=True)
print(f"\\ntrain: {X_train.shape}   test: {X_test.shape}")
print("train failure rate:", round(y_train.mean() * 100, 3), "%")
print("test  failure rate:", round(y_test.mean() * 100, 3), "%")
"""),
        code("""
baseline = ev.majority_class_baseline(y_train, y_test)
baseline
"""),
        md("""
> **Read the baseline before any model score.** Every model below must be judged
> against this number, not against zero.
"""),
        code("""fe.status("classification")"""),
        team("Engineered feature (Q-CF1) - required for rubric B3",
             "Which physically meaningful quantity could you derive from the existing columns?",
             "State your justification BEFORE measuring it.",
             "After registering it and re-running, did failure recall improve? Report honestly."),

        md("""
## 4. Part A - the five Review 1 algorithms  *(Rubric D1, D2)*

**Metric conventions, stated once and used everywhere:**

* Precision and Recall are **binary**, with **failure (`Target == 1`) as the positive class**
* F1 is **weighted**, as the rubric mandates
* ROC-AUC uses `predict_proba` / `decision_function` - **never** hard predictions

**What each metric means here:**

| Metric | Plain meaning |
|---|---|
| Accuracy | share of all 2,000 test machines classified correctly |
| Precision (failure) | of the machines we called failures, how many really failed |
| Recall (failure) | of the machines that really failed, how many we caught |
| F1 (weighted) | harmonic mean of precision and recall, averaged over classes by size |
| ROC-AUC | probability the model scores a random failure above a random healthy machine |
"""),
        code("""
from sklearn.metrics import confusion_matrix

results = {}     # model name -> metrics
fitted  = {}     # model name -> fitted Pipeline
cms     = {}     # model name -> confusion matrix

def record(name, model):
    \"\"\"Evaluate on the held-out test set, print a readable report, store it.\"\"\"
    y_pred = model.predict(X_test)
    y_score, how = ev._scores_for_auc(model, X_test)
    m = ev.classification_metrics(y_test, y_pred, y_score)
    m["AUC_source"] = how
    results[name] = m
    fitted[name] = model
    cm = confusion_matrix(y_test, y_pred)
    cms[name] = cm
    tn, fp, fn, tp = cm.ravel()
    print(f"{name}")
    print(f"  Accuracy          : {m['Accuracy']:.4f}   (baseline {baseline['baseline_accuracy']:.4f})")
    print(f"  Precision(failure): {m['Precision_failure']:.4f}")
    print(f"  Recall(failure)   : {m['Recall_failure']:.4f}")
    print(f"  F1 (weighted)     : {m['F1_weighted']:.4f}")
    print(f"  ROC-AUC           : {m['ROC_AUC']:.4f}   (from {how})")
    print(f"  confusion: correct-safe={tn}  false-alarms={fp}  MISSED={fn}  caught={tp}")
    return m
"""),
    ]

    for i, m in enumerate(CLASSIFICATION[:5], start=1):
        cells.append(md(algo_markdown(m, f"4.{i}")))
        cells.append(code(f"""{strip_pipeline_import(m["imports"])}

model = Pipeline([
    ("prep", pp.make_preprocessor(NUM, CAT, scale={m['scale']})),
    ("model", {m['estimator']}),
])

model.fit(X_train, y_train)
record("{m['title'].split('  [')[0]}", model)
"""))
        if m["extra"].strip():
            cells.append(code(m["extra"].strip()))
            if "decision_tree_figure" in m["extra"]:
                cells.append(code(show()))

    cells += [
        md("### 4.6 Part A comparison table  *(Review 1 deliverable)*"),
        code("""
cols = ["Model", "Accuracy", "Precision_failure", "Recall_failure",
        "F1_weighted", "ROC_AUC", "AUC_source"]
tableA = (pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
          .sort_values("F1_weighted", ascending=False).reset_index(drop=True))[cols]
tableA.index = tableA.index + 1
tableA.index.name = "Rank"
tableA
"""),
        code("""
p = pl.confusion_grid(cms, CLS, "Part A confusion matrices (held-out test set)",
                      "nb_clf_confusion_partA", "classification")
"""),
        code(show()),
        team("Part A comparison (Review 1 deliverable)",
             "Rank the five models on weighted F1, then on failure recall. Does the ranking change?",
             "Look at the confusion matrices: which models achieve high accuracy mainly by rarely predicting failure at all?",
             "In a maintenance setting, which error costs more - a missed failure or a false alarm? Which model does your answer favour?"),

        md("""
---
# REVIEW 2 STARTS HERE
---

## 5. Part B - the five Review 2 algorithms  *(Review 2, Rubric A1)*

Same dataset, same split, same preprocessing as Part A, so the consolidated
table in section 6 is a fair comparison.
"""),
    ]

    for i, m in enumerate(CLASSIFICATION[5:], start=1):
        cells.append(md(algo_markdown(m, f"5.{i}")))
        cells.append(code(f"""{strip_pipeline_import(m["imports"])}

model = Pipeline([
    ("prep", pp.make_preprocessor(NUM, CAT, scale={m['scale']})),
    ("model", {m['estimator']}),
])

model.fit(X_train, y_train)
record("{m['title'].split('  [')[0]}", model)
"""))
        if m["extra"].strip():
            cells.append(code(m["extra"].strip()))

    cells += [
        md("""
## 6. Consolidated 10-algorithm table  *(Review 2, Rubric A2 - deliverable D6)*

One table, all ten algorithms, same dataset and same held-out split.
"""),
        code("""
table = (pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
         .sort_values("F1_weighted", ascending=False).reset_index(drop=True))[cols]
table.index = table.index + 1
table.index.name = "Rank"
print(f"{len(table)} models | majority-class baseline: "
      f"accuracy {baseline['baseline_accuracy']:.4f}, "
      f"weighted F1 {baseline['baseline_f1_weighted']:.4f}\\n")
table
"""),
        code("""
p = pl.model_comparison_bar(table, "F1_weighted",
                            "Classifiers ranked by weighted F1",
                            "nb_clf_model_comparison", "classification")
"""),
        code(show()),
        code("""
p = pl.confusion_grid(cms, CLS,
                      "All 10 classifiers - confusion matrices (held-out test set)",
                      "nb_clf_confusion_all10", "classification", ncols=4)
"""),
        code(show()),
        code("""
p = pl.roc_curves(fitted, X_test, y_test,
                  "ROC curves (probability / decision scores, not hard labels)",
                  "nb_clf_roc_curves", "classification")
"""),
        code(show()),
        team("Consolidated comparison",
             "Which models beat the majority-class baseline on weighted F1 by a margin you consider meaningful?",
             "Do the ROC-AUC ranking and the weighted-F1 ranking agree? Where they differ, explain why (AUC is threshold-free).",
             "Which family - linear, kernel, tree ensemble, neural - suits this problem best, and what about the data explains that?",
             "Accuracy spans a narrow range while failure recall spans a wide one. What does that tell you?"),

        md("## 7. Cross-validation and tuning  *(Review 2, Rubric A3)*"),
        code("""
cv_all = ev.cv_scores(fitted, X_train, y_train, scoring="f1_weighted")
cv_all.drop(columns="fold_scores").sort_values("CV_mean", ascending=False)
"""),
        code("""
from src import classification_models as cm

leaders = cv_all.sort_values("CV_mean", ascending=False)["Model"].head(2).tolist()
print("leaders selected on TRAINING cross-validation:", leaders)

rows, tuned = {}, {}
for name in leaders:
    if name not in cm.PARAM_GRIDS:
        continue
    before_cv = float(cv_all.loc[cv_all["Model"] == name, "CV_mean"].iloc[0])
    before = table[table["Model"] == name].iloc[0]
    gs = cm.tune(name, fitted[name], X_train, y_train)
    score, _ = ev._scores_for_auc(gs.best_estimator_, X_test)
    after = ev.classification_metrics(y_test, gs.best_estimator_.predict(X_test), score)
    rows[name] = {
        "best_params": gs.best_params_,
        "CV_F1w_before": round(before_cv, 5),
        "CV_F1w_after": round(gs.best_score_, 5),
        "HeldOut_F1w_before": round(float(before["F1_weighted"]), 5),
        "HeldOut_F1w_after": round(after["F1_weighted"], 5),
        "Recall_fail_before": round(float(before["Recall_failure"]), 5),
        "Recall_fail_after": round(after["Recall_failure"], 5),
        "improved_on_CV": bool(gs.best_score_ > before_cv),
    }
    tuned[name] = gs.best_estimator_
pd.DataFrame(rows).T
"""),
        team("Final model selection (Review 2, Rubric A3)",
             "Which model do you select as final, and which metric drove the decision?",
             "Did tuning improve at least one metric? Quote the before/after numbers - and if it did not, say so.",
             "Does your chosen model trade failure recall for accuracy? Is that trade acceptable for the use case?"),

        md("## 8. Classification track conclusion"),
        team("Classification conclusion",
             "Summarise the whole track: problem -> data -> method -> result -> conclusion.",
             "What is the single most important caveat a reader should keep in mind (imbalance? synthetic data? threshold choice?)",
             "If this were deployed on a real production line, what would you monitor?"),
    ]
    return cells


# ==========================================================================
def clustering_nb():
    spec = config.CLUSTERING
    cells = [
        md(HEADER.format(title="Clustering Track - Credit Card Customers (BankChurners)",
                         review="Review 2 (full track)",
                         dataset=f"{spec['name']} - `{spec['filename']}`",
                         url=spec["url"])),
        md("""
> ### The rule that governs this whole notebook
> `Attrition_Flag` is **withheld**. It does not influence feature selection,
> scaling, the choice of *k*, or any fitting step. It is loaded once, set aside,
> and cross-tabulated only at the very end - **after** the models are fixed.
> Clusters are not expected, and not required, to line up with churn.
"""),
        md("## 0. Setup"),
        code(SETUP),
        code(f"""
from src import clustering_models as cu
FEATURES = {spec["cluster_features"]!r}
print("clustering features:", FEATURES)
"""),

        md("""
## 1. Load, audit and feature selection

**The seven-feature subset is PROVISIONAL.** It was a starting suggestion; the
team must record its own justification (Q-CL1). `CLIENTNUM` (identifier) and the
two `Naive_Bayes_Classifier_...` columns (which the dataset author tells users to
delete) are excluded.
"""),
        code("""
raw = dl.load_raw("clustering")
summary = dl.audit_summary(raw, "clustering")
print("raw shape:", summary["shape"])
print("duplicates:", summary["n_duplicate_rows"],
      "| missing cells:", summary["total_missing_cells"])
print("\\nfeature subset status:", config.CLUSTERING["feature_subset_status"])
"""),
        code("""
df = dl.prepare("clustering", raw)
holdout = dl.holdout_labels(raw)          # set aside; NOT used until section 8
assert "Attrition_Flag" not in df.columns and "CLIENTNUM" not in df.columns
df.describe().T.round(3)
"""),
        team("Feature subset justification (Q-CL1) - required",
             "Why these seven variables? What customer behaviour is each meant to capture?",
             "Which candidate columns did you reject, and why (redundancy? identifier? derived from the label?)",
             "Credit_Limit, Avg_Open_To_Buy and Total_Revolving_Bal are algebraically related - how did that affect your choice?"),

        md("## 2. Exploratory data analysis"),
        code("""
p = pl.distribution_grid(df, "BankChurners - clustering feature distributions",
                         "nb_clu_feature_distributions", "clustering")
"""),
        code(show()),
        code("""
print("skewness of each feature (0 = symmetric):")
print(df.skew(numeric_only=True).round(3).sort_values(ascending=False).to_string())
"""),
        code("""
p = pl.correlation_heatmap(df, "BankChurners - correlation heatmap (clustering features)",
                           "nb_clu_correlation_heatmap", "clustering")
"""),
        code(show()),
        team("Feature distributions and correlations",
             "Which features are heavily skewed, and what does skew do to a Euclidean distance?",
             "Which pair is most correlated, and does that effectively double-count one behaviour in the distance calculation?",
             "Would a log transform of any feature be justified? Say why or why not."),

        md("""
## 3. Scaling - and why it is not optional here

Distance-based algorithms compare customers by straight-line distance across all
seven features. If one feature has a far larger numeric range, it dominates that
distance and the rest barely count.
"""),
        code("""
print("raw feature ranges BEFORE scaling:")
print(df.agg(["min", "max", "mean", "std"]).round(2).T.to_string())
ratio = df["Credit_Limit"].std() / df["Avg_Utilization_Ratio"].std()
print(f"\\nCredit_Limit std is about {ratio:,.0f}x Avg_Utilization_Ratio's.")
print("Unscaled, the distance between two customers would be almost entirely")
print("their credit-limit difference.")
"""),
        code("""
scaler = pp.make_cluster_preprocessor(list(df.columns))
X = scaler.fit_transform(df)
print("scaled:", X.shape, "| mean ~", np.round(X.mean(), 10), "| std ~", np.round(X.std(), 6))
"""),
        md("""
### 3.1 Memory feasibility

Hierarchical clustering and the exact silhouette both build an *n x n* distance
structure. At 10,127 rows that is 8 x 10,127^2 bytes. We check the cost before
running, rather than assuming it will fit.
"""),
        code("""
probe = cu.memory_probe(len(X))
sil_sample = None if probe["feasible"] else 5000
print(probe)
print("\\nexact silhouette on the full dataset:", sil_sample is None)
"""),

        md("""
## 4. K-Means Clustering  *(Rubric B1, B2)*

**What it is**

Splits customers into *k* groups by repeatedly moving *k* "centre points" until
every customer belongs to whichever centre is nearest. You must supply *k* - the
algorithm cannot work out how many groups there should be.

**How it works**

1. Pick *k* starting centres (`k-means++` spreads them out sensibly rather than at random).
2. Assign every customer to the nearest centre. That creates *k* groups.
3. Move each centre to the average position of the customers now in its group.
4. Reassign everyone to the nearest of the **new** centres.
5. Repeat 3-4 until nobody changes group - that is convergence.

`n_init=10` runs the whole procedure from 10 different starts and keeps the best,
because a bad start can land K-Means in a poor solution.

**Settings** - `n_clusters` (chosen with the elbow curve below), `n_init`, and
`random_state=42` for reproducibility.

**Viva note** - K-Means assumes clusters are round and roughly equal in size,
because it assigns by straight-line distance to a centre. Long, thin groups would
get cut in half.
"""),
        md("""
### 4.1 The elbow curve - choosing k

Inertia is the total squared distance from each customer to their own cluster
centre. It **always** falls as *k* rises, so the lowest inertia is not the answer.
We look for the "elbow": the point after which it stops falling steeply.
"""),
        code("""
km_table, km_models = cu.kmeans_sweep(X, range(2, 11), silhouette_sample=sil_sample)
km_table
"""),
        code("""
p = pl.elbow_plot(km_table["k"], km_table["inertia"], km_table["Silhouette"],
                  "nb_clu_kmeans_elbow", "clustering")
k_elbow = cu.elbow_knee(km_table["k"], km_table["inertia"])
k_sil   = int(km_table.loc[km_table["Silhouette"].idxmax(), "k"])
k_ch    = int(km_table.loc[km_table["Calinski_Harabasz"].idxmax(), "k"])
k_db    = int(km_table.loc[km_table["Davies_Bouldin"].idxmin(), "k"])
print(f"\\nelbow            -> k = {k_elbow}")
print(f"best silhouette  -> k = {k_sil}")
print(f"best Calinski-H  -> k = {k_ch}")
print(f"best Davies-B    -> k = {k_db}")
print("\\nThese criteria DISAGREE. The team decides - Q-CL2.")
"""),
        code(show()),
        team("Choosing k (Q-CL2) - required",
             "Where do YOU read the elbow? Is it sharp or gradual?",
             "Four criteria give four different answers. Which do you follow, and why?",
             "Does a larger k give clusters you can actually describe in business terms, or just smaller ones?",
             "State your chosen k and your reason. The cell below uses the numerical elbow as a placeholder."),
        code("""
k_final = k_elbow          # TEAM: replace with your chosen k once decided
print("k used from here on:", k_final)

km = km_models[k_final]
km_labels = km.labels_

m = ev.clustering_metrics(X, km_labels, sample_size=sil_sample)
print(f"\\nK-Means at k={k_final}")
print(f"  Silhouette        : {m['Silhouette']:.4f}   (higher better)")
print(f"  Davies-Bouldin    : {m['Davies_Bouldin']:.4f}   (LOWER better)")
print(f"  Calinski-Harabasz : {m['Calinski_Harabasz']:.1f}   (higher better)")
print("\\n  cluster sizes:")
for c in sorted(set(km_labels)):
    n = int((km_labels == c).sum())
    print(f"    cluster {c}: {n:5d} ({100*n/len(km_labels):5.2f}%)")
"""),

        md("""
## 5. Agglomerative Hierarchical Clustering  *(Rubric B1, B2)*

**What it is**

Starts with every customer as their own cluster and repeatedly merges the two
closest clusters, building a tree of merges. You cut that tree wherever you like
to get however many clusters you want.

**How it works**

1. Begin with 10,127 clusters - one per customer.
2. Compute the distance between every pair of clusters, using the **linkage** rule.
3. Merge the closest pair. Now there are 10,126.
4. Recompute distances involving the newly merged cluster, and merge again.
5. Repeat until the desired number of clusters remains.

**Linkage** is the rule for the distance between two *groups*, not two customers:

| Linkage | Distance between two groups |
|---|---|
| `ward` | the merge that increases total within-cluster variance least |
| `complete` | the **farthest** pair between them |
| `average` | the average over all pairs |
| `single` | the **closest** pair - prone to "chaining" |

**Viva note** - unlike K-Means it does not need *k* before fitting; you see the
whole tree and cut afterwards. But it cannot assign a **new** customer without
refitting everything, which is precisely why ordinary cross-validation does not
apply to it (IC-3). It is also O(n^2) in memory - hence the check in 3.1.
"""),
        md("### 5.1 Comparing linkage strategies"),
        code("""
link_cmp = cu.linkage_comparison(X, k_final, silhouette_sample=sil_sample)
link_cmp
"""),
        md("""
> Look at single linkage's Calinski-Harabasz against ward's. A near-zero value is
> the signature of one giant cluster plus a few slivers - always check cluster
> sizes before trusting a linkage's silhouette.
"""),
        team("Linkage comparison",
             "Which linkage scores best on silhouette, and which on Davies-Bouldin? Do they agree?",
             "Check the cluster sizes for single linkage. Did the one-giant-cluster problem happen, and why does that make its metric score misleading?",
             "Which linkage do you adopt, and why?"),
        code("""
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=k_final, linkage="ward")
agg_labels = agg.fit_predict(X)

m = ev.clustering_metrics(X, agg_labels, sample_size=sil_sample)
print(f"Agglomerative (ward) at k={k_final}")
print(f"  Silhouette        : {m['Silhouette']:.4f}")
print(f"  Davies-Bouldin    : {m['Davies_Bouldin']:.4f}")
print(f"  Calinski-Harabasz : {m['Calinski_Harabasz']:.1f}")
print("\\n  cluster sizes:")
for c in sorted(set(agg_labels)):
    n = int((agg_labels == c).sum())
    print(f"    cluster {c}: {n:5d} ({100*n/len(agg_labels):5.2f}%)")
"""),
        code("""
agg_table, _ = cu.agglomerative_sweep(X, range(2, 11), linkage="ward",
                                      silhouette_sample=sil_sample)
agg_table
"""),
        md("""
### 5.2 Dendrogram

The height of each join is the distance at which those two groups merged.
Cutting horizontally at a chosen height gives a chosen number of clusters. The
**display** is truncated to the last 30 merges for legibility; the hierarchy
itself covers all 10,127 customers.
"""),
        code("""
p = pl.dendrogram_plot(X, "ward", "nb_clu_dendrogram_ward", "clustering", truncate_p=30)
"""),
        code(show()),
        team("Dendrogram",
             "At what merge distance would a horizontal cut give you your chosen k?",
             "Do the branch heights suggest a natural number of clusters, or a smooth continuum?",
             "Are the cluster sizes at your cut balanced?"),

        md("""
## 6. Metrics for both algorithms  *(Rubric B2)*

| Metric | Range | Direction | Meaning |
|---|---|---|---|
| Silhouette | -1 to +1 | **higher** better | how much closer a point is to its own cluster than to the next nearest |
| Davies-Bouldin | 0 upward | **LOWER** better | average similarity between each cluster and the one it most resembles |
| Calinski-Harabasz | 0 upward | **higher** better | between-cluster spread divided by within-cluster spread |
"""),
        code("""
rows = []
for algo, labels in (("K-Means", km_labels), ("Agglomerative (ward)", agg_labels)):
    m = ev.clustering_metrics(X, labels, sample_size=sil_sample)
    m["algorithm"] = algo
    m["k"] = k_final
    rows.append(m)
final = pd.DataFrame(rows)[["algorithm", "k", "n_clusters", "Silhouette",
                            "Davies_Bouldin", "Calinski_Harabasz", "silhouette_exact"]]
final
"""),
        team("Metric comparison",
             "Remember LOWER Davies-Bouldin is better. Do all three metrics agree on which algorithm wins?",
             "The absolute silhouette values are low. Does this customer base have genuinely separated groups, or a continuum?",
             "Would you describe the result as 'clusters found' or 'a partition imposed'? Defend your answer."),

        md("""
## 7. Cluster visualisation  *(Rubric B3)*

**Important:** both algorithms were fitted on the full 7-dimensional standardised
space. PCA and t-SNE below are **projections for viewing only** - no model was
fitted on two components. Fitting on two components and then plotting those same
two components would be circular.

**PCA** finds the directions of greatest variance and projects onto the top two.
**t-SNE** instead tries to preserve which points are near neighbours, which makes
groups look more separated - sometimes more separated than they really are.
"""),
        code("""
coords, pca = cu.pca_embedding(X, 2)
evr = pca.explained_variance_ratio_
note = (f"PCA is for VISUALISATION only; clustering was fitted on all 7 standardised "
        f"features. PC1+PC2 explain {evr.sum()*100:.1f}% of variance.")
print(note)
p1 = pl.cluster_scatter_2d(coords, km_labels, f"K-Means (k={k_final}) - PCA projection",
                           "nb_clu_pca_kmeans", "clustering",
                           xlabel=f"PC1 ({evr[0]*100:.1f}% var)",
                           ylabel=f"PC2 ({evr[1]*100:.1f}% var)", note=note)
p2 = pl.cluster_scatter_2d(coords, agg_labels,
                           f"Agglomerative ward (k={k_final}) - PCA projection",
                           "nb_clu_pca_agglomerative", "clustering",
                           xlabel=f"PC1 ({evr[0]*100:.1f}% var)",
                           ylabel=f"PC2 ({evr[1]*100:.1f}% var)", note=note)
"""),
        code(show("p1")),
        code(show("p2")),
        code("""
tc, tl, tnote = cu.tsne_embedding(X, km_labels, sample_size=3000)
print(tnote)
p = pl.cluster_scatter_2d(tc, tl, f"K-Means (k={k_final}) - t-SNE projection",
                          "nb_clu_tsne_kmeans", "clustering",
                          xlabel="t-SNE 1", ylabel="t-SNE 2", note=tnote)
"""),
        code(show()),
        team("Reading the projections",
             "Do the clusters look separated in the PCA view, or do they tile a single cloud? Remember PC1+PC2 capture only part of the variance.",
             "Does t-SNE show more separation than PCA? Why can t-SNE exaggerate separation, and how much should you trust it?",
             "Do K-Means and Agglomerative carve the space in a similar way?"),

        md("""
## 8. Supplementary validation and the post-hoc label check

### 8.1 Resampling stability - **not** cross-validation

The PDF asks for cross-validation in every track. Ordinary k-fold CV has no
direct meaning for `AgglomerativeClustering`, which cannot assign unseen points.
What follows is **supplementary evidence only** and does not by itself satisfy
that rubric wording - logged as **IC-3**.

Overlapping subsamples are clustered independently and the label agreement on
their overlap is measured with the Adjusted Rand Index: 1.0 means identical
partitions, 0.0 means chance agreement.
"""),
        code("""
stab = [cu.stability_assessment(X, k_final, "kmeans", n_repeats=20),
        cu.stability_assessment(X, k_final, "agglomerative", n_repeats=20)]
pd.DataFrame(stab)[["algorithm", "k", "mean_ARI", "std_ARI", "min_ARI",
                    "interpretation_status"]]
"""),
        team("Stability",
             "How reproducible is each algorithm's partition under resampling?",
             "If one algorithm is markedly less stable, what does that say about trusting its cluster descriptions?"),

        md("""
### 8.2 Cluster profiles - the basis for your interpretation

The mean of each **original, unscaled** feature per cluster. This table is what
you interpret; naming the clusters is team work (Q-CL3).
"""),
        code("""
print("dataset means for reference:")
print(df.mean().round(2).to_string())
"""),
        code("""
from IPython.display import display
for algo, labels in (("K-Means", km_labels), ("Agglomerative", agg_labels)):
    print(f"\\n=== {algo} cluster profiles (original units) ===")
    display(cu.cluster_profile(df, labels))
"""),
        team("Cluster interpretation (Q-CL3) - required, the heart of this track",
             "Give each K-Means cluster a short descriptive NAME based on its profile row.",
             "For each cluster, name the two or three features that most distinguish it from the overall average.",
             "Which cluster would a bank care about most, and what action would you recommend?",
             "Do any two clusters differ so little that you would merge them?"),

        md("""
### 8.3 Post-hoc check against the withheld label

Run **only now**, after *k* and the models were fixed. This is validation, not
optimisation - and the clusters were never meant to reproduce churn.
"""),
        code("""
overall = (holdout == "Attrited Customer").mean() * 100
print(f"overall attrition rate: {overall:.2f}%\\n")
ct = cu.posthoc_label_crosstab(km_labels, holdout)
ct
"""),
        team("Post-hoc validation (Q-CL4)",
             "Does any cluster show a notably higher attrition rate than the overall rate?",
             "If clusters do NOT align with churn, is the clustering therefore useless? Argue your position.",
             "Why would it have been methodologically wrong to use this table to choose k?"),

        md("## 9. Clustering track conclusion"),
        team("Clustering conclusion",
             "Summarise: which algorithm, which k, which metrics, and what the clusters mean in business terms.",
             "What is the strongest argument AGAINST your own conclusion?",
             "What extra data would make this segmentation more convincing?"),
    ]
    return cells


# ==========================================================================
def write(name: str, cells: list) -> Path:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    NB_DIR.mkdir(parents=True, exist_ok=True)
    path = NB_DIR / f"{name}.ipynb"
    nbf.write(nb, path)
    n_code = sum(1 for c in cells if c.cell_type == "code")
    n_team = sum(1 for c in cells
                 if c.cell_type == "markdown" and "TEAM TO COMPLETE" in c.source)
    print(f"[notebook] {path.relative_to(config.PROJECT_ROOT)} "
          f"({len(cells)} cells: {n_code} code, {n_team} team cells)")
    return path


def main() -> int:
    write("regression", regression_nb())
    write("classification", classification_nb())
    write("clustering", clustering_nb())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
