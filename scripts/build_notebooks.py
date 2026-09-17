"""build_notebooks.py - generate notebooks/{regression,classification,clustering}.ipynb

The notebooks import the SAME modules in src/ that scripts/run_all.py uses, so
there is exactly one implementation of every pipeline step. The notebook adds
the narrative Markdown, the inline tables, and the 'TEAM TO COMPLETE' cells.

    python scripts/build_notebooks.py          # write the .ipynb files
    python scripts/execute_notebooks.py        # run them top-to-bottom
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf

from src import config

NB_DIR = config.NOTEBOOKS_DIR


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


def team(heading: str, *questions: str):
    """A clearly marked cell that AI must not fill in (PDF section 7.5)."""
    qs = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    return md(f"""
> ### ✍️ TEAM TO COMPLETE — {heading}
>
> *This cell is intentionally left unwritten. Section 7.5 of the guidelines
> requires that analysis and interpretation are the team's own work. Replace
> everything below the line with your own observations.*
>
> **Guiding questions**
>
{chr(10).join('> ' + line for line in qs.split(chr(10)))}
>
> ---
>
> *(your written observation here)*
""")


HEADER = """
# 23CSE301 Machine Learning — Capstone Project
## {title}

**Track owner:** _(team member name — see `docs/team_contributions.md`)_
**Review:** {review}
**Dataset:** {dataset}
**Source:** {url}

---

### How this notebook is organised
Every pipeline step calls the shared implementation in `src/`, which is the same
code that `scripts/run_all.py` executes. That keeps the notebook and the scripts
from drifting apart and means no result shown here is computed twice in two
different ways.

**On AI assistance:** code scaffolding in this project was generated with an AI
assistant (see `README.md` § AI-Assistance Disclosure). Every cell marked
**✍️ TEAM TO COMPLETE** must be written by the team. Those cells are empty on
purpose.
"""

SETUP = """
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))   # project root on the path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config, data_loading as dl, preprocessing as pp
from src import feature_engineering as fe, evaluation as ev, plotting as pl

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
warnings.filterwarnings("default")      # warnings stay visible on purpose

print("random_state =", config.RANDOM_STATE, "| test_size =", config.TEST_SIZE,
      "| cv folds =", config.CV_FOLDS)
"""


def show_fig(path_expr: str) -> str:
    return f"""
from IPython.display import Image, display
display(Image(filename=str({path_expr})))
"""


# ==========================================================================
def regression_nb():
    spec = config.REGRESSION
    cells = [
        md(HEADER.format(title="Regression Track — Student Performance",
                         review="Review 1 (full track)",
                         dataset=f"{spec['name']} — `{spec['filename']}`",
                         url=spec["url"])),
        md("## 0. Environment and shared configuration"),
        code(SETUP),

        md("""
## 1. Data loading and audit  *(Rubric A1)*
Raw file is read from `data/raw/` and left untouched. Shape, dtypes,
missing-value counts and the target distribution are all reported below.
"""),
        code("""
raw = dl.load_raw("regression")
print("\\nAudit summary:")
summary = dl.audit_summary(raw, "regression")
for k, v in summary.items():
    print(f"  {k}: {v}")
"""),
        code("""
dl.audit(raw, "regression", "raw")
"""),
        code("""
raw.head()
"""),
        code("""
raw.describe(include="all").T
"""),

        md("""
### 1.1 Duplicate records — a decision the team must confirm
127 exact duplicate rows exist. All five inputs are low-cardinality integers or
a binary flag, so identical rows arise by coincidence in a synthetic 10,000-row
sample. Removing them would drop the prepared set to 9,873 rows, **below the
10,000-row working preference**. The reported run RETAINS them; the alternative
is measured below so the team can decide on evidence.
"""),
        code("""
df = dl.prepare("regression", raw)
dedup = pp.drop_exact_duplicates(df)
print(f"\\nretained : {len(df)} rows  (the reported configuration)")
print(f"if dropped: {len(dedup)} rows  -> below the 10,000 preference: {len(dedup) < 10000}")
"""),
        team("Duplicate-handling decision (Q-RG1)",
             "Are these duplicated rows plausible coincidences given the feature cardinality, or do they look like a data-collection artefact?",
             "Which option do you choose, and what is your justification?",
             "What effect, if any, could retaining them have on the train/test comparison?"),

        md("""
## 2. Exploratory data analysis  *(Rubric A2, A3)*
Distribution plot for every input feature, correlation heatmap, target
distribution, and two feature–target scatter plots.
"""),
        code("""
feats = spec_num + spec_cat
p = pl.distribution_grid(df[feats], "Student Performance - input feature distributions",
                         "nb_reg_feature_distributions", "regression")
""".replace("spec_num", repr(spec["numeric_features"]))
             .replace("spec_cat", repr(spec["categorical_features"]))),
        code(show_fig('p')),
        team("Feature distributions",
             "Which features are roughly uniform, and which are skewed?",
             "Does any feature show an implausible value or a suspicious spike?",
             "What does the shape of each distribution imply for scaling or for the models that assume normality?"),

        code("""
p = pl.correlation_heatmap(df, "Student Performance - correlation heatmap (numeric)",
                           "nb_reg_correlation_heatmap", "regression")
"""),
        code(show_fig('p')),
        team("Correlation structure",
             "Which input correlates most strongly with Performance Index?",
             "Is there any correlation between inputs high enough to raise a multicollinearity concern for the linear models?",
             "Does the correlation ranking match what you would expect from the domain?"),

        code("""
p = pl.target_distribution(df[TARGET], "Target distribution - Performance Index",
                           "nb_reg_target_distribution", "regression")
""".replace("TARGET", repr(spec["target"]))),
        code(show_fig('p')),

        code("""
corr_t = df.select_dtypes("number").corr()[TARGET].drop(TARGET)
top2 = corr_t.abs().sort_values(ascending=False).index[:2].tolist()
print("two strongest linear relationships with the target:", top2)
p = pl.feature_target_scatter(df, top2, TARGET, "nb_reg_feature_target_scatter", "regression")
""".replace("TARGET", repr(spec["target"]))),
        code(show_fig('p')),
        team("Feature–target relationships",
             "Is each relationship linear, curved, or absent?",
             "How much vertical spread is there at a fixed x — i.e. how much variance can this feature alone not explain?",
             "Does this support or undermine a linear model as the baseline?"),

        md("""
## 3. Preprocessing and the held-out split  *(Rubric B1, B2)*
The split is made **before** any data-driven modelling decision. Imputation,
one-hot encoding and scaling all live inside pipelines, so each is refitted on
the training part of every CV fold — never on the full dataset.

**Stratification note.** Rubric B2 asks for a stratified split. Stratification
is defined for discrete classes; the target here is continuous. The project
therefore uses an unstratified random split and records the ambiguity in
`docs/instructor_clarifications.md` (IC-2). `pp.split_supervised(...,
stratify_bins=k)` implements quantile-binned stratification if the instructor
confirms it is wanted.
"""),
        code("""
X_train, X_test, y_train, y_test = pp.split_supervised(df, TARGET, stratify=False)
print(pp.check_no_row_leakage(X_train, X_test))
""".replace("TARGET", repr(spec["target"]))),
        team("Train/test overlap (Q-RG2)",
             "The overlap report counts test rows whose feature values also appear in training. Why does this number look large here, and is it 'leakage' in the harmful sense?",
             "Would deduplicating before splitting change your answer?"),

        md("""
### 3.1 Feature engineering  *(Rubric B3)* — ⚠️ NOT YET SATISFIED
`src/feature_engineering.py` ships an empty hook. Until the team registers a
feature **and its justification**, the baseline pipeline runs and this rubric
item is reported INCOMPLETE. The AI assistant deliberately did not invent one.
"""),
        code("""
fe.status("regression")
"""),
        team("Engineered feature (Q-RG3) — required for rubric B3",
             "Which new feature will you create, from which existing columns?",
             "What is your reasoning for why it should help — before you look at whether it does?",
             "After registering it in src/feature_engineering.py and re-running, did it help? Report the result honestly either way."),

        md("""
## 4. All ten regression algorithms  *(Rubric C1, C2)*
Every model is a `Pipeline(preprocess → estimator)` trained on the same training
split and scored on the same held-out test split.
"""),
        code("""
from src import regression_models as rm
models = rm.build_models(NUM, CAT)
print(f"{len(models)} models:")
for k in models: print("  ", k)
""".replace("NUM", repr(spec["numeric_features"])).replace("CAT", repr(spec["categorical_features"]))),
        code("""
table, fitted = ev.evaluate_regressors(models, X_train, y_train, X_test, y_test)
table
"""),
        code("""
p = pl.model_comparison_bar(table, "R2", "Regression models ranked by test R2",
                            "nb_reg_model_comparison", "regression")
"""),
        code(show_fig('p')),
        team("Comparative results",
             "Which model ranks first on test R², and by how much does it beat the linear baseline?",
             "Is that margin large enough to matter, given the RMSE is in Performance-Index points?",
             "What does it tell you that the regularised linear models and the ensembles land so close together?"),

        md("""
## 5. Cross-validation  *(Rubric: 5-fold CV R² for the two best models)*
Leading models are nominated using **training-side** cross-validation. The
held-out test set plays no part in selection.
"""),
        code("""
cv_all = ev.cv_scores(models, X_train, y_train, scoring="r2")
cv_all.drop(columns="fold_scores")
"""),
        code("""
leaders = cv_all.sort_values("CV_mean", ascending=False)["Model"].head(2).tolist()
print("leaders selected on training CV:", leaders)
cv_all[cv_all["Model"].isin(leaders)][["Model","CV_mean","CV_std","fold_scores"]]
"""),

        md("""
## 6. Hyperparameter tuning  *(Rubric C3)*
`GridSearchCV` on at least two models. Before/after scores are reported on both
the CV score used for selection and the held-out set. **If tuning does not
improve a model, that is what gets reported.**
"""),
        code("""
import json
tunable = [m for m in leaders if m in rm.PARAM_GRIDS]
for cand in rm.PARAM_GRIDS:
    if len(tunable) >= 2: break
    if cand not in tunable: tunable.append(cand)
print("tuning:", tunable)

rows = {}
tuned = {}
for name in tunable:
    base_cv = float(cv_all.loc[cv_all["Model"] == name, "CV_mean"].iloc[0])
    base_test = float(table.loc[table["Model"] == name, "R2"].iloc[0])
    gs = rm.tune(name, models[name], X_train, y_train)
    after = ev.regression_metrics(y_test, gs.best_estimator_.predict(X_test))
    rows[name] = {"best_params": gs.best_params_,
                  "CV_R2_before": round(base_cv, 5), "CV_R2_after": round(gs.best_score_, 5),
                  "HeldOut_R2_before": round(base_test, 5),
                  "HeldOut_R2_after": round(after["R2"], 5),
                  "improved_on_CV": bool(gs.best_score_ > base_cv)}
    tuned[name] = gs.best_estimator_
pd.DataFrame(rows).T
"""),
        team("Tuning outcome",
             "Did tuning change the score meaningfully, or only in the fourth decimal place?",
             "What does the size of the improvement suggest about whether these models were under-regularised to begin with?",
             "Would you ship the tuned or the default configuration, and why?"),

        md("""
## 7. Algorithm-specific requirements from the PDF
Coefficient interpretation (#1), Lasso sparsity (#3), polynomial degree
comparison (#5), tree feature importance (#6/#7).
"""),
        code("""
coefs = rm.coefficient_table(fitted["1. Linear Regression"])
print("intercept:", round(coefs.attrs["intercept"], 4))
print("NOTE: numeric inputs are standardised, so each coefficient is the change "
      "in Performance Index per 1 standard deviation of that feature.")
coefs
"""),
        team("Coefficient interpretation",
             "State in one sentence, in plain language, what the largest coefficient means for a student.",
             "Do the signs of all coefficients match your domain expectation?"),
        code("""
rm.sparsity_report(fitted["3. Lasso Regression"])
"""),
        code("""
poly = rm.polynomial_degree_comparison(NUM, CAT, X_train, y_train, X_test, y_test,
                                       degrees=(1, 2, 3))
poly
""".replace("NUM", repr(spec["numeric_features"])).replace("CAT", repr(spec["categorical_features"]))),
        team("Lasso sparsity and polynomial degree",
             "How many coefficients did Lasso drive to zero, and what does that imply about redundant features?",
             "Compare train R² and test R² as the polynomial degree rises. At which degree does overfitting begin, and how do you see it?"),

        md("## 8. Required visualisations  *(Rubric C4)*"),
        code("""
best_name = table.sort_values("R2", ascending=False).iloc[0]["Model"]
best_model = tuned.get(best_name, fitted[best_name])
y_pred = best_model.predict(X_test)
print("best model on held-out R2:", best_name)
p1 = pl.residual_plot(y_test, y_pred, best_name, "nb_reg_residuals", "regression")
p2 = pl.predicted_vs_actual(y_test, y_pred, best_name, "nb_reg_pred_vs_actual", "regression")
"""),
        code(show_fig('p1')),
        code(show_fig('p2')),
        team("Residual diagnostics",
             "Are residuals centred on zero and roughly constant across the predicted range, or do they fan out?",
             "Does the residual histogram look approximately normal?",
             "What would a visible pattern here tell you about a missing feature or a missing non-linear term?"),
        code("""
imp = rm.tree_importances(fitted["7. Random Forest Regressor"])
p = pl.importance_plot(imp.values, imp.index, "Random Forest Regressor - feature importance",
                       "nb_reg_feature_importance", "regression", xlabel="Gini importance")
imp
"""),
        code(show_fig('p')),
        team("Feature importance vs linear coefficients",
             "Do the Random Forest importances rank the features the same way the linear coefficients do?",
             "Where they disagree, which do you trust for this dataset, and why?"),

        md("""
## 9. Review 1 summary

Fill in the conclusion yourself — this is explicitly excluded from AI assistance.
"""),
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
    NUM, CAT = repr(spec["numeric_features"]), repr(spec["categorical_features"])
    cells = [
        md(HEADER.format(title="Classification Track — Machine Predictive Maintenance",
                         review="Part A → Review 1 · Part B → Review 2",
                         dataset=f"{spec['name']} — `{spec['filename']}`",
                         url=spec["url"])),
        md("## 0. Environment"),
        code(SETUP),
        code("""
from src import classification_models as cm
TARGET = "Target"
CLS = ["No Failure", "Failure"]
NUM = %s
CAT = %s
""" % (NUM, CAT)),

        md("""
## 1. Data loading, audit and leakage exclusions  *(Rubric A1)*

**Target leakage, explicitly handled.** `Failure Type` is *not* a predictor: it
is a second recording of the same event as `Target`. A model given it would
score near-perfectly and learn nothing. `UDI` and `Product ID` are row
identifiers. All three are dropped.
"""),
        code("""
raw = dl.load_raw("classification")
summary = dl.audit_summary(raw, "classification")
for k, v in summary.items(): print(f"  {k}: {v}")
"""),
        code("""
pd.crosstab(raw["Failure Type"], raw["Target"])
"""),
        md("""
The crosstab above is the evidence for the exclusion: `Failure Type` is
deterministic given `Target` apart from a small number of ambiguous rows.
"""),
        code("""
dl.audit(raw, "classification", "raw")
"""),
        code("""
df = dl.prepare("classification", raw)
print("\\nprepared columns:", list(df.columns))
df.head()
"""),

        md("""
## 2. EDA  *(Rubric A2, A3)*
The target is binary and heavily imbalanced, so a raw feature-vs-target scatter
would collapse onto two horizontal lines. Jitter plus a companion box plot is
used instead.
"""),
        code("""
p = pl.distribution_grid(df[NUM + CAT], "Predictive Maintenance - input feature distributions",
                         "nb_clf_feature_distributions", "classification")
"""),
        code(show_fig('p')),
        team("Feature distributions",
             "Which features look approximately normal and which are skewed or bounded?",
             "Air and process temperature: what does their shape suggest about how this synthetic data was generated?",
             "Does the Type category split the machines evenly?"),
        code("""
p = pl.correlation_heatmap(df, "Predictive Maintenance - correlation heatmap (numeric)",
                           "nb_clf_correlation_heatmap", "classification")
"""),
        code(show_fig('p')),
        team("Correlation structure",
             "Two pairs of features are strongly related. Which, and why physically?",
             "Does any single feature correlate strongly with Target on its own? What does that imply about whether failure is a single-variable phenomenon?"),
        code("""
p = pl.target_distribution(df[TARGET], "Target distribution - machine failure",
                           "nb_clf_target_distribution", "classification",
                           discrete=True, class_names={0: CLS[0], 1: CLS[1]})
"""),
        code(show_fig('p')),
        team("Class imbalance",
             "What fraction of rows are failures?",
             "What accuracy would a model get by predicting 'No Failure' every single time?",
             "Which metric should therefore lead your model comparison, and why not accuracy?"),
        code("""
p = pl.feature_target_scatter(df, ["Torque [Nm]", "Rotational speed [rpm]"], TARGET,
                              "nb_clf_feature_target_scatter", "classification", jitter=0.08,
                              title="Feature-target relationships (target jittered for readability)")
"""),
        code(show_fig('p')),
        code("""
p = pl.grouped_box(df, ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]"], TARGET,
                   "nb_clf_feature_by_class", "classification",
                   class_names={0: CLS[0], 1: CLS[1]})
"""),
        code(show_fig('p')),
        team("Feature–target relationships",
             "In which region of torque and rotational speed do failures concentrate?",
             "Do the box plots separate the two classes for any single feature, or is failure an interaction between features?",
             "What does that suggest about linear vs tree-based models here?"),

        md("""
## 3. Split, baseline and feature engineering  *(Rubric B1, B2, B3)*
Stratified 80:20 — with a 3.4 % positive class, an unstratified split could give
the folds very different failure rates.
"""),
        code("""
X_train, X_test, y_train, y_test = pp.split_supervised(df, TARGET, stratify=True)
print(pp.check_no_row_leakage(X_train, X_test))
base = ev.majority_class_baseline(y_train, y_test)
base
"""),
        md("""
> **Read the baseline before reading any model score.** Any model below must be
> compared against this number, not against zero.
"""),
        code("""
fe.status("classification")
"""),
        team("Engineered feature (Q-CF1) — required for rubric B3",
             "Which physically meaningful quantity could you derive from the existing columns?",
             "State your justification BEFORE measuring it.",
             "After registering it and re-running, did recall on the failure class improve? Report honestly."),

        md("""
## 4. Part A — the five Review 1 algorithms  *(Rubric D1, D2)*

**Metric conventions, stated once and used everywhere:**
* Precision and Recall are **binary**, positive class = **failure (`Target == 1`)**
* F1 is **weighted**, as the rubric mandates
* ROC-AUC uses `predict_proba` / `decision_function` — never hard labels
"""),
        code("""
part_a = cm.build_part_a(NUM, CAT)
ta, fa, cma = ev.evaluate_classifiers(part_a, X_train, y_train, X_test, y_test)
ta
"""),
        code("""
p = pl.confusion_grid(cma, CLS, "Part A confusion matrices (held-out test set)",
                      "nb_clf_confusion_partA", "classification")
"""),
        code(show_fig('p')),
        team("Part A comparison (Review 1 deliverable)",
             "Rank the five models on weighted F1, then on failure recall. Does the ranking change?",
             "Look at the confusion matrices: which models achieve high accuracy mainly by rarely predicting failure?",
             "In a maintenance setting, which error is more expensive — a missed failure or a false alarm? Which model does your answer favour?"),

        md("""
## 5. Part B — the five Review 2 algorithms  *(Review 2, Rubric A1)*
Same dataset, same split, same preprocessing as Part A.
"""),
        code("""
part_b = cm.build_part_b(NUM, CAT)
tb, fb, cmb = ev.evaluate_classifiers(part_b, X_train, y_train, X_test, y_test)
tb
"""),
        code("""
p = pl.confusion_grid(cmb, CLS, "Part B confusion matrices (held-out test set)",
                      "nb_clf_confusion_partB", "classification")
"""),
        code(show_fig('p')),

        md("""
## 6. Consolidated 10-algorithm table  *(Review 2, Rubric A2 — deliverable D6)*
"""),
        code("""
all_models = {**part_a, **part_b}
fitted = {**fa, **fb}
cms = {**cma, **cmb}
table = (pd.concat([ta.reset_index(drop=True), tb.reset_index(drop=True)])
         .sort_values("F1_weighted", ascending=False).reset_index(drop=True))
table.index = table.index + 1
table.index.name = "Rank"
table
"""),
        code("""
print("majority-class reference -> accuracy %.4f, weighted F1 %.4f"
      % (base["baseline_accuracy"], base["baseline_f1_weighted"]))
p = pl.model_comparison_bar(table, "F1_weighted", "Classifiers ranked by weighted F1",
                            "nb_clf_model_comparison", "classification")
"""),
        code(show_fig('p')),
        code("""
p = pl.roc_curves(fitted, X_test, y_test,
                  "ROC curves (probability / decision scores, not hard labels)",
                  "nb_clf_roc_curves", "classification")
"""),
        code(show_fig('p')),
        team("Consolidated comparison",
             "Which models beat the majority-class baseline on weighted F1 by a margin you consider meaningful?",
             "Do the ROC-AUC ranking and the weighted-F1 ranking agree? Where they differ, explain why (hint: AUC is threshold-free).",
             "Which family — linear, kernel, tree ensemble, neural — suits this problem best, and what about the data explains that?"),

        md("""
## 7. Cross-validation and tuning  *(Review 2, Rubric A3)*
"""),
        code("""
cv_all = ev.cv_scores(all_models, X_train, y_train, scoring="f1_weighted")
cv_all.drop(columns="fold_scores")
"""),
        code("""
leaders = cv_all.sort_values("CV_mean", ascending=False)["Model"].head(2).tolist()
print("leaders (training CV):", leaders)
rows, tuned = {}, {}
for name in leaders:
    if name not in cm.PARAM_GRIDS: continue
    base_cv = float(cv_all.loc[cv_all["Model"] == name, "CV_mean"].iloc[0])
    before = table[table["Model"] == name].iloc[0]
    gs = cm.tune(name, all_models[name], X_train, y_train)
    score, _ = ev._scores_for_auc(gs.best_estimator_, X_test)
    after = ev.classification_metrics(y_test, gs.best_estimator_.predict(X_test), score)
    rows[name] = {"best_params": gs.best_params_,
                  "CV_F1w_before": round(base_cv, 5), "CV_F1w_after": round(gs.best_score_, 5),
                  "HeldOut_F1w_before": round(float(before["F1_weighted"]), 5),
                  "HeldOut_F1w_after": round(after["F1_weighted"], 5),
                  "Recall_fail_before": round(float(before["Recall_failure"]), 5),
                  "Recall_fail_after": round(after["Recall_failure"], 5)}
    tuned[name] = gs.best_estimator_
pd.DataFrame(rows).T
"""),
        team("Final model selection (Review 2, Rubric A3)",
             "Which model do you select as final, and which metric drove the decision?",
             "Did tuning improve at least one metric? Quote the before/after numbers — and if it did not, say so.",
             "Does your chosen model trade failure recall for accuracy? Is that trade acceptable for the use case?"),

        md("""
## 8. Algorithm-specific requirements from the PDF
Odds/coefficients (#1), distance metrics (#2), conditional independence (#3),
tree visualisation (#4), feature importance (#6).
"""),
        code("""
odds = cm.odds_table(fitted["A1. Logistic Regression"])
print("intercept:", round(odds.attrs["intercept"], 4), "|", odds.attrs["scale_note"])
odds
"""),
        code("""
cm.knn_distance_metric_comparison(NUM, CAT, X_train, y_train, X_test, y_test)
"""),
        team("Odds ratios, distance metrics, and the Naive Bayes assumption",
             "Pick the largest odds ratio and state what a one-standard-deviation increase does to the odds of failure.",
             "Did the KNN distance metric change performance materially? Why might that be, given the feature scales?",
             "Gaussian NB assumes features are conditionally independent given the class. Using your correlation heatmap, name a pair that violates this. How did GaussianNB's score reflect that?"),
        code("""
dt = fitted["A4. Decision Tree Classifier"]
p = pl.decision_tree_figure(dt.named_steps["model"],
                            dt.named_steps["prep"].get_feature_names_out(),
                            CLS, "nb_clf_decision_tree", "classification", max_depth=3)
"""),
        code(show_fig('p')),
        code("""
imp = cm.tree_importances(fitted["B6. Random Forest Classifier"])
p = pl.importance_plot(imp.values, imp.index, "Random Forest Classifier - feature importance",
                       "nb_clf_feature_importance", "classification", xlabel="Gini importance")
imp
"""),
        code(show_fig('p')),
        team("Tree structure and importance",
             "Read the top two splits of the tree. What physical rule has it learned?",
             "Do the Random Forest importances agree with those splits and with the logistic coefficients?"),

        md("## 9. Classification track conclusion"),
        team("Classification conclusion",
             "Summarise the whole track: problem → data → method → result → conclusion.",
             "What is the single most important caveat a reader should keep in mind (imbalance? synthetic data? threshold choice?).",
             "If this were deployed on a real production line, what would you monitor?"),
    ]
    return cells


# ==========================================================================
def clustering_nb():
    spec = config.CLUSTERING
    cells = [
        md(HEADER.format(title="Clustering Track — Credit Card Customers (BankChurners)",
                         review="Review 2 (full track)",
                         dataset=f"{spec['name']} — `{spec['filename']}`",
                         url=spec["url"])),
        md("""
> ### ⚠️ The one rule that governs this whole notebook
> `Attrition_Flag` is **withheld**. It does not influence feature selection,
> scaling, the choice of *k*, or any fitting step. It is loaded once, kept aside,
> and cross-tabulated only in the final section — **after** the models are fixed.
> Clusters are not expected, and not required, to line up with churn.
"""),
        md("## 0. Environment"),
        code(SETUP),
        code("""
from src import clustering_models as cu
FEATURES = %s
""" % repr(spec["cluster_features"])),

        md("""
## 1. Load, audit and feature selection

**The seven-feature subset is PROVISIONAL.** It was suggested as a starting
point; the team must record its own justification before it is final
(see `docs/team_analysis_prompts.md` Q-CL1). `CLIENTNUM` (an identifier) and the
two `Naive_Bayes_Classifier_...` columns (which the dataset author instructs
users to delete) are excluded.
"""),
        code("""
raw = dl.load_raw("clustering")
summary = dl.audit_summary(raw, "clustering")
print("raw shape:", summary["shape"])
print("duplicates:", summary["n_duplicate_rows"], "| missing cells:", summary["total_missing_cells"])
print("\\nfeature subset status:", config.CLUSTERING["feature_subset_status"])
"""),
        code("""
df = dl.prepare("clustering", raw)
holdout = dl.holdout_labels(raw)     # set aside, NOT used until section 8
assert "Attrition_Flag" not in df.columns and "CLIENTNUM" not in df.columns
df.describe().T
"""),
        team("Feature subset justification (Q-CL1) — required",
             "Why these seven variables? What customer behaviour is each meant to capture?",
             "Which candidate columns did you reject, and why (redundancy? identifier? derived from the label?)",
             "Note that Credit_Limit, Avg_Open_To_Buy and Total_Revolving_Bal are algebraically related — how did that affect your choice?"),

        md("## 2. EDA"),
        code("""
p = pl.distribution_grid(df, "BankChurners - clustering feature distributions",
                         "nb_clu_feature_distributions", "clustering")
"""),
        code(show_fig('p')),
        code("""
p = pl.correlation_heatmap(df, "BankChurners - correlation heatmap (clustering features)",
                           "nb_clu_correlation_heatmap", "clustering")
"""),
        code(show_fig('p')),
        team("Feature distributions and correlations",
             "Which features are heavily skewed, and what does that do to a Euclidean distance?",
             "Which pair is most correlated, and does that effectively double-count one behaviour in the distance calculation?",
             "Would a log transform of any feature be justified? Say why or why not."),

        md("""
## 3. Scaling and memory feasibility
Standardisation is mandatory here: `Credit_Limit` runs to five figures while
`Avg_Utilization_Ratio` lies in [0, 1], so without scaling the distance metric
would be decided almost entirely by credit limit.

Hierarchical clustering and the exact silhouette both build an *n × n*
structure, so the cost is checked before running them.
"""),
        code("""
scaler = pp.make_cluster_preprocessor(list(df.columns))
X = scaler.fit_transform(df)
probe = cu.memory_probe(len(X))
probe
"""),
        code("""
sil_sample = None if probe["feasible"] else 5000
print("scaled:", X.shape, "| exact silhouette:", sil_sample is None)
"""),

        md("""
## 4. K-Means — elbow curve and metric sweep  *(Rubric B1, B2)*
"""),
        code("""
km_table, km_models = cu.kmeans_sweep(X, range(2, 11), silhouette_sample=sil_sample)
km_table
"""),
        code("""
p = pl.elbow_plot(km_table["k"], km_table["inertia"], km_table["Silhouette"],
                  "nb_clu_kmeans_elbow", "clustering")
k_elbow = cu.elbow_knee(km_table["k"], km_table["inertia"])
k_sil = int(km_table.loc[km_table["Silhouette"].idxmax(), "k"])
print("elbow suggests k =", k_elbow, "| best silhouette at k =", k_sil)
"""),
        code(show_fig('p')),
        team("Choosing k (Q-CL2) — required",
             "Where do YOU read the elbow? Is it sharp or gradual?",
             "The elbow and the silhouette maximum may disagree. Which do you follow, and why?",
             "Does a larger k give you clusters that are actually describable in business terms, or just smaller ones?",
             "State your chosen k and your reason. The code below uses the numerical elbow as a placeholder."),
        code("""
k_final = k_elbow          # TEAM: replace with your chosen k once decided
print("k used below:", k_final)
km_final = km_models[k_final]
km_labels = km_final.labels_
"""),

        md("""
## 5. Agglomerative hierarchical clustering  *(Rubric B1, B2)*
Linkage strategies are compared, and the dendrogram is drawn from the **actual
fitted hierarchy** (display truncated to the last 30 merges so the leaves stay
legible — the hierarchy itself is complete).
"""),
        code("""
link_cmp = cu.linkage_comparison(X, k_final, silhouette_sample=sil_sample)
link_cmp
"""),
        team("Linkage comparison",
             "Which linkage scores best on silhouette, and which on Davies-Bouldin? Do they agree?",
             "Single linkage often produces one huge cluster and several tiny ones. Check the cluster sizes — did that happen here, and why is its metric score misleading if so?",
             "Which linkage do you adopt, and why?"),
        code("""
agg_table, agg_models = cu.agglomerative_sweep(X, range(2, 11), linkage="ward",
                                               silhouette_sample=sil_sample)
agg_model, agg_labels = agg_models[k_final]
agg_table
"""),
        code("""
p = pl.dendrogram_plot(X, "ward", "nb_clu_dendrogram_ward", "clustering", truncate_p=30)
"""),
        code(show_fig('p')),
        team("Dendrogram",
             "At what merge distance would a horizontal cut give you your chosen k?",
             "Do the branch heights suggest a natural number of clusters, or a smooth continuum?",
             "Are the cluster sizes at the cut balanced?"),

        md("""
## 6. Headline metrics for both algorithms  *(Rubric B2)*
Silhouette, Davies-Bouldin and Calinski-Harabasz for K-Means and Agglomerative
at the same k.
"""),
        code("""
rows = []
for algo, labels in (("K-Means", km_labels), ("Agglomerative (ward)", agg_labels)):
    m = ev.clustering_metrics(X, labels, sample_size=sil_sample)
    m["algorithm"] = algo; m["k"] = k_final
    rows.append(m)
final = pd.DataFrame(rows)[["algorithm","k","n_clusters","Silhouette",
                            "Davies_Bouldin","Calinski_Harabasz","silhouette_exact"]]
final
"""),
        team("Metric comparison",
             "Higher silhouette and Calinski-Harabasz are better; LOWER Davies-Bouldin is better. Do all three agree on which algorithm wins?",
             "The absolute silhouette values here are low. What does that say about whether this customer base has genuinely separated groups or a continuum?",
             "Would you describe the result as 'clusters found' or 'a partition imposed'? Defend your answer."),

        md("""
## 7. Cluster visualisation  *(Rubric B3)*

**Important:** both algorithms were fitted on the full 7-dimensional standardised
space. PCA and t-SNE below are **projections for viewing only** — no model was
fitted on two components.
"""),
        code("""
coords, pca = cu.pca_embedding(X, 2)
evr = pca.explained_variance_ratio_
note = (f"PCA is for VISUALISATION only; clustering was fitted on all 7 standardised "
        f"features. PC1+PC2 explain {evr.sum()*100:.1f}% of variance.")
p1 = pl.cluster_scatter_2d(coords, km_labels, f"K-Means (k={k_final}) - PCA projection",
                           "nb_clu_pca_kmeans", "clustering",
                           xlabel=f"PC1 ({evr[0]*100:.1f}% var)",
                           ylabel=f"PC2 ({evr[1]*100:.1f}% var)", note=note)
p2 = pl.cluster_scatter_2d(coords, agg_labels, f"Agglomerative ward (k={k_final}) - PCA projection",
                           "nb_clu_pca_agglomerative", "clustering",
                           xlabel=f"PC1 ({evr[0]*100:.1f}% var)",
                           ylabel=f"PC2 ({evr[1]*100:.1f}% var)", note=note)
"""),
        code(show_fig('p1')),
        code(show_fig('p2')),
        code("""
tc, tl, tnote = cu.tsne_embedding(X, km_labels, sample_size=3000)
print(tnote)
p = pl.cluster_scatter_2d(tc, tl, f"K-Means (k={k_final}) - t-SNE projection",
                          "nb_clu_tsne_kmeans", "clustering",
                          xlabel="t-SNE 1", ylabel="t-SNE 2", note=tnote)
"""),
        code(show_fig('p')),
        team("Reading the projections",
             "Do the clusters look separated in the PCA view, or do they mostly tile a single cloud? Remember PC1+PC2 capture only part of the variance.",
             "Does t-SNE show more separation than PCA? Why can t-SNE exaggerate separation, and what does that mean for how much you should trust it?",
             "Do K-Means and Agglomerative carve the space in a similar way?"),

        md("""
## 8. Supplementary validation and post-hoc label check

### 8.1 Resampling stability — **not** cross-validation
The PDF asks for cross-validation in every track. Ordinary k-fold CV has no
direct meaning for `AgglomerativeClustering`, which cannot assign unseen points.
What follows is **supplementary evidence only** and does not itself satisfy that
rubric wording — logged as `IC-3` in `docs/instructor_clarifications.md`.
Overlapping subsamples are clustered independently and the label agreement on
the overlap is measured with the Adjusted Rand Index.
"""),
        code("""
stab = [cu.stability_assessment(X, k_final, "kmeans", n_repeats=20),
        cu.stability_assessment(X, k_final, "agglomerative", n_repeats=20)]
pd.DataFrame(stab)
"""),
        team("Stability",
             "ARI of 1.0 means identical partitions, 0.0 means chance agreement. How reproducible is each algorithm's partition?",
             "If one algorithm is markedly less stable, what does that say about trusting its cluster descriptions?"),

        md("""
### 8.2 Cluster profiles — the basis for your interpretation
Mean of each **original, unscaled** feature per cluster.
"""),
        code("""
for algo, labels in (("K-Means", km_labels), ("Agglomerative", agg_labels)):
    print(f"\\n=== {algo} ===")
    display(cu.cluster_profile(df, labels))
"""),
        team("Cluster interpretation (Q-CL3) — required, the heart of this track",
             "Give each K-Means cluster a short descriptive NAME based on its profile row.",
             "For each cluster, name the two or three features that most distinguish it from the overall average.",
             "Which cluster would a bank care about most, and what action would you recommend for it?",
             "Do any two clusters differ so little that you would merge them?"),

        md("""
### 8.3 Post-hoc check against the withheld label
Run **only now**, after k and the models were fixed. This is validation, not
optimisation — and the clusters were never meant to reproduce churn.
"""),
        code("""
ct = cu.posthoc_label_crosstab(km_labels, holdout)
ct
"""),
        team("Post-hoc validation (Q-CL4)",
             "Does any cluster show a notably higher attrition rate than the overall rate?",
             "If clusters do NOT align with churn, is the clustering therefore useless? Argue your position.",
             "Why would it have been methodologically wrong to use this table to pick k?"),

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
    print(f"[notebook] {path.relative_to(config.PROJECT_ROOT)} "
          f"({len(cells)} cells)")
    return path


def main() -> int:
    write("regression", regression_nb())
    write("classification", classification_nb())
    write("clustering", clustering_nb())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
