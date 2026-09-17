"""build_examples.py - generate one standalone, runnable script per algorithm.

    python scripts/build_examples.py

Creates examples/{regression,classification,clustering}/NN_<name>.py — 22 files.

Each script is SELF-CONTAINED for one algorithm: it loads the data, builds that
one model, fits it, evaluates it, and prints the result. Its docstring explains
what the algorithm is, how it works step by step, and what the settings mean.

Purpose: viva preparation. The guidelines (§7.4) say an owner must be able to
explain *every line* of their track. A 10-model notebook cell is hard to defend
line by line; a 40-line script for one model is not.

These are STUDY AIDS. The figures reported in the README come from
scripts/run_all.py, which trains all models on one shared split so the
comparison is fair. A single example script re-creates the same split, so its
numbers match — but run_all.py remains the source of record.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

EX = config.PROJECT_ROOT / "examples"

# ---------------------------------------------------------------- templates
REG_BODY = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
{imports}

# ---------------------------------------------------------------------------
# 1. LOAD  -  read the raw CSV and keep only the columns we model with.
# ---------------------------------------------------------------------------
spec = config.REGRESSION
df = dl.prepare("regression")                 # 10,000 rows x 6 columns
TARGET = spec["target"]                       # "Performance Index"
NUM = spec["numeric_features"]                # 4 numeric inputs
CAT = spec["categorical_features"]            # 1 categorical input

# ---------------------------------------------------------------------------
# 2. SPLIT  -  80% train / 20% test, fixed seed so it is identical every run.
#    The split happens BEFORE anything is fitted, so the test set stays unseen.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = pp.split_supervised(df, TARGET)

# ---------------------------------------------------------------------------
# 3. BUILD  -  preprocessing + model in ONE Pipeline object.
#    Why a Pipeline? Because .fit() then fits the scaler on the TRAINING data
#    only. If we scaled the whole dataset first, the scaler would have seen the
#    test set - that is data leakage and costs marks (guidelines section 7.1).
# ---------------------------------------------------------------------------
preprocessor = pp.make_preprocessor(NUM, CAT, scale={scale})
model = Pipeline([
    ("prep", preprocessor),
{poly}    ("model", {estimator}),
])

# ---------------------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------------------
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 5. PREDICT AND EVALUATE  -  on the held-out test set.
#    R2   : fraction of the variation in the target the model explains (1 = perfect)
#    RMSE : typical error size, in Performance Index points, punishing big misses
#    MAE  : average error size, in Performance Index points
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
metrics = ev.regression_metrics(y_test, y_pred)

print("\\n{title}")
print("=" * 60)
print(f"  R2   : {{metrics['R2']:.6f}}")
print(f"  RMSE : {{metrics['RMSE']:.4f}}  (Performance Index points)")
print(f"  MAE  : {{metrics['MAE']:.4f}}  (Performance Index points)")
{extra}'''

CLF_BODY = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import confusion_matrix
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
{imports}

# ---------------------------------------------------------------------------
# 1. LOAD  -  6 inputs + 1 target. Failure Type is EXCLUDED: it records the same
#    event as Target, so using it would be target leakage (a fake perfect score).
# ---------------------------------------------------------------------------
spec = config.CLASSIFICATION
df = dl.prepare("classification")             # 10,000 rows x 7 columns
TARGET = spec["target"]                       # "Target": 0 = no failure, 1 = failure
NUM = spec["numeric_features"]
CAT = spec["categorical_features"]

# ---------------------------------------------------------------------------
# 2. SPLIT  -  stratified, because only 3.39% of rows are failures. Stratifying
#    keeps that same 3.39% in both train and test; a plain random split could
#    give them very different failure rates.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = pp.split_supervised(df, TARGET, stratify=True)

# ---------------------------------------------------------------------------
# 3. BUILD  -  preprocessing + model in one Pipeline (see the leakage note in
#    any regression example).
# ---------------------------------------------------------------------------
model = Pipeline([
    ("prep", pp.make_preprocessor(NUM, CAT, scale={scale})),
    ("model", {estimator}),
])

# ---------------------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------------------
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 5. EVALUATE
#    Accuracy is MISLEADING here: predicting "no failure" every time already
#    scores 96.6%. Watch Recall on the failure class - the share of real
#    failures the model actually caught.
#    ROC-AUC is computed from PROBABILITIES, never from the 0/1 predictions.
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_score, how = ev._scores_for_auc(model, X_test)
m = ev.classification_metrics(y_test, y_pred, y_score)

print("\\n{title}")
print("=" * 60)
print(f"  Accuracy          : {{m['Accuracy']:.4f}}   (majority-class baseline = 0.9660)")
print(f"  Precision(failure): {{m['Precision_failure']:.4f}}   of predicted failures, how many were real")
print(f"  Recall(failure)   : {{m['Recall_failure']:.4f}}   of real failures, how many we caught")
print(f"  F1 (weighted)     : {{m['F1_weighted']:.4f}}")
print(f"  ROC-AUC           : {{m['ROC_AUC']:.4f}}   (from {{how}})")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("\\n  Confusion matrix on 2,000 test machines:")
print(f"    correctly called safe     : {{tn:5d}}")
print(f"    false alarms              : {{fp:5d}}  (said fail, was fine)")
print(f"    MISSED FAILURES           : {{fn:5d}}  (said fine, actually failed)")
print(f"    failures caught           : {{tp:5d}}")
{extra}'''

CLU_BODY = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from src import clustering_models as cu
{imports}

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
print(f"scaled data: {{X.shape}}  (all 7 features, not PCA components)")

{body}'''


# ---------------------------------------------------------------- the content
REGRESSION = [
    dict(n=1, slug="linear_regression", title="1. Linear Regression",
         imports="from sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import Pipeline",
         estimator="LinearRegression()", scale="True", poly="",
         what="Draws the single best straight line (in 5 dimensions, a flat plane) through the data. It assumes the target is just a weighted sum of the inputs plus a constant.",
         how=["Give every input feature a weight, starting from anything.",
              "For each student, predict: weight1 x hours + weight2 x previous score + ... + intercept.",
              "Measure how wrong the predictions are, using the sum of squared errors.",
              "Solve, in one step of algebra (the 'normal equation'), for the weights that make that total error as small as it can be.",
              "Those weights are the trained model. There is nothing to iterate - the answer is exact."],
         settings="None worth tuning. That is the point of a baseline: it has no knobs to hide behind.",
         viva="A coefficient says: if this feature goes up by one standard deviation and everything else stays fixed, the predicted Performance Index changes by this much. Our inputs are standardised, so the coefficients are directly comparable to each other.",
         result="Test R2 = 0.988983, RMSE = 2.02 points. Ranked 2nd of 10 - a plain straight line beat eight more complicated models.",
         extra='''
# --- ALGORITHM-SPECIFIC: read the coefficients (PDF note: "interpret coefficients")
from src.regression_models import coefficient_table
coefs = coefficient_table(model)
print("\\n  Coefficients (change in Performance Index per 1 standard deviation):")
print(coefs.to_string(index=False))
print(f"  intercept: {coefs.attrs['intercept']:.4f}")'''),

    dict(n=2, slug="ridge_regression", title="2. Ridge Regression",
         imports="from sklearn.linear_model import Ridge\nfrom sklearn.pipeline import Pipeline",
         estimator="Ridge(alpha=1.0, random_state=42)", scale="True", poly="",
         what="Linear Regression with a leash. It still fits a straight line, but it is penalised for using large weights, which stops any single feature from dominating.",
         how=["Set up the same squared-error total as Linear Regression.",
              "Add a penalty term: alpha x (sum of all the weights squared).",
              "Minimise error + penalty together, not error alone.",
              "Because big weights now cost something, the solution shrinks every weight toward zero - but never exactly to zero.",
              "alpha controls the leash length: alpha = 0 is plain Linear Regression, huge alpha forces all weights near zero."],
         settings="alpha - the strength of the penalty. We searched 0.01, 0.1, 1.0, 10.0, 100.0 and the best was 0.1, i.e. almost no penalty needed.",
         viva="This is L2 regularisation. It helps when features are correlated or when you have more features than data. Here neither is true, which is exactly why it scored the same as plain Linear Regression.",
         result="Test R2 = 0.988982. Tuning alpha moved the score in the 5th decimal place.",
         extra=""),

    dict(n=3, slug="lasso_regression", title="3. Lasso Regression",
         imports="from sklearn.linear_model import Lasso\nfrom sklearn.pipeline import Pipeline",
         estimator="Lasso(alpha=0.01, random_state=42, max_iter=10000)", scale="True", poly="",
         what="Like Ridge, but its penalty can switch features off completely by setting their weight to exactly zero. It does feature selection while it fits.",
         how=["Same squared-error total as before.",
              "Add a penalty of alpha x (sum of the ABSOLUTE values of the weights).",
              "Absolute value has a sharp corner at zero, and that corner makes it possible for a weight to land exactly on zero rather than merely near it.",
              "Solve iteratively (coordinate descent) - unlike Ridge there is no one-step formula.",
              "Any feature whose weight ends at zero has been dropped from the model."],
         settings="alpha - larger alpha zeroes out more features. We searched 0.001, 0.01, 0.1, 1.0; best was 0.001.",
         viva="L1 versus L2 is the classic question. L1 (Lasso) gives sparsity - it deletes features. L2 (Ridge) only shrinks them. On our data Lasso zeroed NOTHING: 0 of 5 coefficients. That means every one of our five inputs is carrying real signal.",
         result="Test R2 = 0.988969. Zero features dropped - reported in results/tables/regression_lasso_sparsity.json.",
         extra='''
# --- ALGORITHM-SPECIFIC: feature sparsity (PDF note: "observe feature sparsity")
from src.regression_models import sparsity_report
print("\\n  Sparsity:", sparsity_report(model))'''),

    dict(n=4, slug="elasticnet_regression", title="4. ElasticNet Regression",
         imports="from sklearn.linear_model import ElasticNet\nfrom sklearn.pipeline import Pipeline",
         estimator="ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)",
         scale="True", poly="",
         what="Ridge and Lasso mixed together. You choose how much of each penalty to apply.",
         how=["Same squared-error total.",
              "Add BOTH penalties: l1_ratio x (Lasso penalty) + (1 - l1_ratio) x (Ridge penalty), all scaled by alpha.",
              "l1_ratio = 1 makes it pure Lasso; l1_ratio = 0 makes it pure Ridge; 0.5 is an even blend.",
              "Solve iteratively, as with Lasso.",
              "You get some feature dropping from the L1 part and some smooth shrinkage from the L2 part."],
         settings="alpha (overall penalty strength) and l1_ratio (the mix). We searched alpha in 0.001-1.0 and l1_ratio in 0.1/0.5/0.9.",
         viva="Why exist at all, if Ridge and Lasso already do? Because when features come in correlated groups, pure Lasso arbitrarily keeps one and drops the rest, whereas ElasticNet tends to keep or drop the group together.",
         result="Test R2 = 0.988886 - the weakest of the four linear models here, by a hair.",
         extra=""),

    dict(n=5, slug="polynomial_regression", title="5. Polynomial Features + Linear Regression",
         imports="from sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import PolynomialFeatures",
         estimator="LinearRegression()", scale="True",
         poly='    ("poly", PolynomialFeatures(degree=2, include_bias=False)),\n',
         what="Linear Regression given extra, invented columns: every feature squared, and every pair of features multiplied together. The model is still linear in its weights, but it can now bend.",
         how=["Start with the 5 preprocessed input columns.",
              "PolynomialFeatures(degree=2) manufactures new columns: hours^2, sleep^2, hours x sleep, and so on. 5 columns become 20.",
              "Hand all 20 columns to an ordinary Linear Regression.",
              "It fits weights for the new columns exactly as it did for the originals.",
              "The result is a curved surface in the original feature space, even though the fitting maths never changed."],
         settings="degree - how far to expand. Higher degree means far more columns and a strong risk of overfitting.",
         viva="Degree 1 gave 5 terms and test R2 0.98898. Degree 2 gave 20 terms and 0.98899. Degree 3 gave 55 terms and 0.98896 - WORSE despite 11x the columns. That is overfitting made visible: more flexibility, worse performance on unseen data.",
         result="Test R2 = 0.988989. Ranked 1st of 10 - but it beat plain Linear Regression only in the 5th decimal, using 4x the terms.",
         extra='''
# --- ALGORITHM-SPECIFIC: compare degrees (PDF note: "compare degrees")
from src.regression_models import polynomial_degree_comparison
print("\\n  Degree comparison:")
print(polynomial_degree_comparison(NUM, CAT, X_train, y_train, X_test, y_test,
                                   degrees=(1, 2, 3)).to_string(index=False))'''),

    dict(n=6, slug="decision_tree_regressor", title="6. Decision Tree Regressor",
         imports="from sklearn.tree import DecisionTreeRegressor\nfrom sklearn.pipeline import Pipeline",
         estimator="DecisionTreeRegressor(max_depth=8, random_state=42)", scale="False", poly="",
         what="A flowchart of yes/no questions. Follow the answers down to a leaf, and the prediction is the average target value of the training students who ended up in that leaf.",
         how=["Start with all 8,000 training rows in one group.",
              "Try every feature and every possible cut point (e.g. 'Previous Scores < 62?').",
              "Pick the single cut that most reduces the variance of the target inside the two resulting groups.",
              "Split, then repeat the whole procedure independently on each group.",
              "Stop at max_depth, or when a group gets too small. Each final group is a leaf, and its prediction is the mean target of its members."],
         settings="max_depth - how many questions deep the tree may go. Deeper means it can memorise the training data. We searched 3, 5, 8, 12, None plus min_samples_leaf.",
         viva="Note scale=False. Trees do not care about feature scale at all, because a cut at 'hours < 5' means the same thing whatever units hours are in. Leaving them unscaled also keeps the tree readable in original units.",
         result="Test R2 = 0.983901, ranked 9th. A single tree predicts in steps, but the true relationship here is smooth - so steps cost accuracy.",
         extra='''
# --- ALGORITHM-SPECIFIC: feature importance (PDF note: "show feature importance")
from src.regression_models import tree_importances
print("\\n  Feature importance:")
print(tree_importances(model).round(4).to_string())'''),

    dict(n=7, slug="random_forest_regressor", title="7. Random Forest Regressor",
         imports="from sklearn.ensemble import RandomForestRegressor\nfrom sklearn.pipeline import Pipeline",
         estimator="RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)",
         scale="False", poly="",
         what="300 different decision trees, each deliberately trained on a slightly different view of the data, and their predictions averaged.",
         how=["Draw a random sample of the training rows WITH replacement (bootstrap) - this tree sees a different dataset.",
              "Grow a decision tree on it, but at every split consider only a random subset of the features, not all of them.",
              "That double randomness makes each tree wrong in a different direction.",
              "Repeat 300 times, independently. The trees never see each other.",
              "To predict, ask all 300 trees and average their answers. Individual errors cancel out; shared signal survives."],
         settings="n_estimators (how many trees - more is steadier but slower) and max_depth. We searched 100/300/500 trees.",
         viva="Look at the overfitting gap: train R2 = 0.9975 but test R2 = 0.9862. The forest has partly memorised the training set. Averaging 300 trees controls that, but does not remove it.",
         result="Test R2 = 0.986175, ranked 7th. Slowest model in the track at 3.4 seconds, and beaten by a one-line linear fit.",
         extra='''
# --- ALGORITHM-SPECIFIC: feature importance
from src.regression_models import tree_importances
print("\\n  Feature importance (averaged over 300 trees):")
print(tree_importances(model).round(4).to_string())'''),

    dict(n=8, slug="gradient_boosting_regressor", title="8. Gradient Boosting Regressor",
         imports="from sklearn.ensemble import GradientBoostingRegressor\nfrom sklearn.pipeline import Pipeline",
         estimator="GradientBoostingRegressor(learning_rate=0.1, n_estimators=200, random_state=42)",
         scale="False", poly="",
         what="Trees built one after another, where each new tree's job is to fix the mistakes left by all the trees before it.",
         how=["Start with a single flat guess: the mean Performance Index of the training set.",
              "Compute the residual for every row - how far off that guess was.",
              "Train a SMALL tree to predict those residuals, i.e. to predict the error itself.",
              "Add a fraction of that tree's output (the learning_rate) to the running prediction.",
              "Recompute the new, smaller residuals and repeat 200 times. Each tree corrects what remains."],
         settings="learning_rate (how much of each correction to accept - small is safer, needs more trees) and n_estimators. We searched 0.03/0.1/0.2 and 100/200/400.",
         viva="The difference from Random Forest is the key question. Random Forest builds trees in PARALLEL and averages; Gradient Boosting builds them in SEQUENCE, each fixing the last. Boosting usually wins, and here it did - 0.9884 versus 0.9862.",
         result="Test R2 = 0.988411, ranked 6th, and the best non-linear model.",
         extra=""),

    dict(n=9, slug="support_vector_regressor", title="9. Support Vector Regressor (SVR)",
         imports="from sklearn.svm import SVR\nfrom sklearn.pipeline import Pipeline",
         estimator="SVR(C=1.0, kernel='rbf')", scale="True", poly="",
         what="Fits a tube around the data and tries to get as many points as possible inside it. Errors smaller than the tube's width are treated as no error at all.",
         how=["Choose a tube width, epsilon. Any prediction within epsilon of the truth counts as correct.",
              "Find the flattest possible function that keeps most points inside the tube.",
              "Points that fall outside are penalised, with C controlling how harshly.",
              "The 'kernel trick' lets it fit a curved function without ever computing the curved coordinates - it only needs distances between points.",
              "Only the points on or outside the tube boundary (the 'support vectors') shape the final model. The rest are ignored entirely."],
         settings="C (penalty for points outside the tube - high C means fit hard, risk overfitting) and kernel ('rbf' curved, 'linear' straight). We searched C in 0.1/1/10 and both kernels.",
         viva="Why scale=True and why it matters most here: the RBF kernel is built on distances between points. An unscaled feature with a big range would swamp the distance calculation and the model would effectively ignore everything else.",
         result="Test R2 = 0.985924, ranked 8th. Took 2 seconds, versus 0.014 for Linear Regression, for a worse score.",
         extra=""),

    dict(n=10, slug="knn_regressor", title="10. K-Nearest Neighbors Regressor",
         imports="from sklearn.neighbors import KNeighborsRegressor\nfrom sklearn.pipeline import Pipeline",
         estimator="KNeighborsRegressor(n_neighbors=5)", scale="True", poly="",
         what="No training at all. To predict for a new student, find the 5 most similar students in the training data and average their Performance Index.",
         how=["'Fitting' just stores the training rows. Nothing is learned.",
              "When a prediction is requested, compute the distance from the new row to every stored training row.",
              "Sort those distances and keep the k smallest.",
              "Average the target values of those k neighbours - that is the prediction.",
              "With weights='distance', closer neighbours count for more than farther ones."],
         settings="n_neighbors (k) - small k is jumpy and noise-sensitive, large k over-smooths. We searched 3/5/9/15/25 and both weighting schemes.",
         viva="Scaling is not optional here, it is the whole ballgame. Distance is the model. If Previous Scores (range 40-99) were left unscaled against Sleep Hours (range 4-9), 'similar student' would mean 'similar previous score' and nothing else.",
         result="Test R2 = 0.976841 - LAST of 10. It is the only model that cannot extrapolate: it can never predict a value outside the range of its training neighbours.",
         extra=""),
]

CLASSIFICATION = [
    dict(n=1, slug="logistic_regression", title="A1. Logistic Regression  [Part A, Review 1]",
         imports="from sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import Pipeline",
         estimator="LogisticRegression(max_iter=2000, random_state=42)", scale="True",
         what="Despite the name, this is a classifier. It computes a weighted sum of the inputs, then squashes that number into a probability between 0 and 1 with the sigmoid function.",
         how=["Compute z = weight1 x torque + weight2 x speed + ... + intercept.",
              "Squash it: probability = 1 / (1 + e^-z). Large positive z gives a probability near 1; large negative gives near 0.",
              "Find the weights that make the training labels most likely (maximum likelihood), by iterative optimisation.",
              "To predict a class, compare the probability to a threshold - 0.5 by default.",
              "The threshold is a CHOICE, not part of the model. Lowering it catches more failures at the cost of more false alarms."],
         settings="C - the inverse of regularisation strength; class_weight='balanced' makes rare failures count more. We searched C in 0.01-10 and both weightings.",
         viva="Exponentiating a coefficient gives an ODDS RATIO: how much the odds of failure multiply per one standard deviation of that feature. See results/tables/classification_logreg_odds.csv.",
         result="F1(weighted) = 0.9560, ROC-AUC = 0.8994, but Recall on failures = 0.1029. It caught 7 of 68 real failures while scoring 96.75% accuracy - the single best illustration of why accuracy is the wrong metric here.",
         extra='''
# --- ALGORITHM-SPECIFIC: odds ratios (PDF note: "interpret coefficients/odds")
from src.classification_models import odds_table
o = odds_table(model)
print("\\n  Coefficients and odds ratios (per 1 standard deviation):")
print(o.to_string(index=False))'''),

    dict(n=2, slug="knn_classifier", title="A2. K-Nearest Neighbors  [Part A, Review 1]",
         imports="from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.pipeline import Pipeline",
         estimator="KNeighborsClassifier(n_neighbors=5)", scale="True",
         what="Find the 5 most similar machines in the training data and take a majority vote on whether they failed.",
         how=["Store the training rows. No model is fitted.",
              "For a new machine, measure the distance to every training machine.",
              "Keep the k closest.",
              "Count their labels. The majority label wins.",
              "The predicted probability is simply the fraction of those k neighbours that failed - which is why with k=5 the only possible probabilities are 0, 0.2, 0.4, 0.6, 0.8, 1.0."],
         settings="n_neighbors (k) and the distance metric. We compared euclidean, manhattan and chebyshev.",
         viva="Coarse probabilities are why its ROC-AUC (0.8291) is the second worst of all ten models even though its accuracy looks fine. AUC needs finely graded scores to rank cases; KNN with k=5 offers only six possible values.",
         result="F1(weighted) = 0.9679, Recall(failure) = 0.2941, ROC-AUC = 0.8291.",
         extra='''
# --- ALGORITHM-SPECIFIC: distance metrics (PDF note: "discuss distance metrics")
from src.classification_models import knn_distance_metric_comparison
print("\\n  Distance metric comparison:")
print(knn_distance_metric_comparison(NUM, CAT, X_train, y_train,
                                     X_test, y_test).to_string(index=False))'''),

    dict(n=3, slug="gaussian_naive_bayes", title="A3. Gaussian Naive Bayes  [Part A, Review 1]",
         imports="from sklearn.naive_bayes import GaussianNB\nfrom sklearn.pipeline import Pipeline",
         estimator="GaussianNB()", scale="True",
         what="Uses Bayes' theorem to flip the question around. Instead of asking 'given these readings, will it fail?', it asks 'if a machine were failing, how likely are these readings?' - and combines that with how common failure is overall.",
         how=["For each class separately, compute the mean and variance of every feature from the training data.",
              "That gives a bell curve per feature per class.",
              "For a new machine, read off how likely each of its readings is under each class's bell curve.",
              "MULTIPLY those likelihoods together - this is the 'naive' step, and it assumes the features are independent given the class.",
              "Multiply by the class's base rate (3.39% for failure) and pick whichever class scores higher."],
         settings="Effectively none. GaussianNB is parameter-free; it just measures means and variances.",
         viva="The independence assumption is the exam question. Air temperature and process temperature are strongly correlated in our data - physically, the process is heated by its surroundings. Naive Bayes multiplies their evidence as if they were unrelated, so it double-counts one fact. Its F1 of 0.9506 is the lowest of all ten, and that is why.",
         result="F1(weighted) = 0.9506 - last of 10. Precision on failures 0.25: three quarters of its failure alarms were false. Do NOT substitute MultinomialNB or BernoulliNB; the PDF names Gaussian specifically.",
         extra=""),

    dict(n=4, slug="decision_tree_classifier", title="A4. Decision Tree Classifier  [Part A, Review 1]",
         imports="from sklearn.tree import DecisionTreeClassifier\nfrom sklearn.pipeline import Pipeline",
         estimator="DecisionTreeClassifier(max_depth=6, random_state=42)", scale="False",
         what="A flowchart of yes/no questions about the sensor readings, ending in a verdict of fail or no-fail.",
         how=["Begin with all 8,000 training machines in one group.",
              "Try every feature and cut point, and score each by how much it purifies the groups - measured by Gini impurity.",
              "A perfect split puts all failures on one side and all healthy machines on the other.",
              "Take the best split, then repeat on each resulting branch.",
              "Stop at max_depth. Each leaf predicts the majority class of its training members, and the probability is the class proportion in that leaf."],
         settings="max_depth, min_samples_leaf, class_weight. We searched depth 3/5/8/12/None.",
         viva="This is the only model you can literally read. results/figures/classification/clf_decision_tree.png shows the top 3 levels - point at the first split and state the physical rule it found.",
         result="F1(weighted) = 0.9714, Recall(failure) = 0.3824 - the best Part A model on failure recall, nearly 4x Logistic Regression.",
         extra='''
# --- ALGORITHM-SPECIFIC: draw the tree (PDF note: "visualise the tree")
from src import plotting as pl
p = pl.decision_tree_figure(model.named_steps["model"],
                            model.named_steps["prep"].get_feature_names_out(),
                            ["No Failure", "Failure"],
                            "example_decision_tree", "classification", max_depth=3)
print(f"\\n  Tree diagram saved to {p}")'''),

    dict(n=5, slug="support_vector_classifier", title="A5. Support Vector Classifier (SVC)  [Part A, Review 1]",
         imports="from sklearn.svm import SVC\nfrom sklearn.pipeline import Pipeline",
         estimator="SVC(C=1.0, kernel='rbf', probability=True, random_state=42)", scale="True",
         what="Draws the boundary between failing and healthy machines that leaves the widest possible empty margin on both sides.",
         how=["Look for a surface separating the two classes.",
              "Among all surfaces that separate them, prefer the one with the largest gap to the nearest point on each side.",
              "Allow some points to sit inside the margin or on the wrong side, penalised by C.",
              "The RBF kernel measures similarity by distance, letting the boundary curve without computing curved coordinates explicitly.",
              "Only the points nearest the boundary - the support vectors - define it. Moving a far-away point changes nothing."],
         settings="C (low C means a wide, forgiving margin; high C means fit the training data hard) and kernel. probability=True is needed for ROC-AUC and costs extra fitting time.",
         viva="Precision 0.875 with Recall 0.2059 - it is very careful. When it cries failure it is almost always right, but it stays quiet about 54 of the 68 real failures. Wide-margin fitting on an imbalanced dataset pushes the boundary toward the majority class.",
         result="F1(weighted) = 0.9635, ROC-AUC = 0.9468.",
         extra=""),

    dict(n=6, slug="random_forest_classifier", title="B6. Random Forest Classifier  [Part B, Review 2]",
         imports="from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.pipeline import Pipeline",
         estimator="RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)", scale="False",
         what="300 decision trees, each on a different bootstrap sample with a random subset of features, voting on the outcome.",
         how=["Bootstrap-sample the training rows for tree 1.",
              "Grow a tree, considering only a random subset of features at each split.",
              "Repeat 300 times independently.",
              "For a new machine, collect all 300 votes.",
              "The predicted probability is the fraction of trees voting 'failure' - which gives 301 possible values and therefore a much better ROC-AUC than KNN's six."],
         settings="n_estimators, max_depth, class_weight. We searched 200/400 trees, depth None/10/20, and balanced weighting.",
         viva="Precision 0.8974 is the highest of all ten models, but Recall 0.5147 means it misses half the failures. Majority voting is conservative by construction: for a rare event, a failure prediction needs over 150 of 300 trees to agree.",
         result="F1(weighted) = 0.9791, ROC-AUC = 0.9709, Recall(failure) = 0.5147.",
         extra='''
# --- ALGORITHM-SPECIFIC: feature importance (PDF note: "feature importance")
from src.classification_models import tree_importances
print("\\n  Feature importance:")
print(tree_importances(model).round(4).to_string())'''),

    dict(n=7, slug="adaboost_classifier", title="B7. AdaBoost Classifier  [Part B, Review 2]",
         imports="from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.pipeline import Pipeline",
         estimator="AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42)",
         scale="False",
         what="Builds very shallow trees in sequence, and after each one it increases the weight of the machines that were misclassified, so the next tree is forced to focus on them.",
         how=["Give all 8,000 training rows equal weight.",
              "Train a very shallow tree (a 'stump', one split deep) on the weighted data.",
              "Measure its weighted error, and give the stump a say proportional to how good it was.",
              "INCREASE the weight of every row it got wrong and decrease the weight of every row it got right.",
              "Train the next stump on the reweighted data, and repeat 200 times. Predict by weighted vote of all stumps."],
         settings="n_estimators and learning_rate (how aggressively to reweight). We searched 100/200/400 and 0.1/0.5/1.0.",
         viva="The contrast with Gradient Boosting is worth knowing: AdaBoost reweights the DATA POINTS; Gradient Boosting fits the RESIDUALS. Here that mattered a lot - 0.9667 versus 0.9855 F1.",
         result="F1(weighted) = 0.9667, Recall(failure) = 0.2941. The weakest Part B model, beaten by a single Decision Tree.",
         extra=""),

    dict(n=8, slug="gradient_boosting_classifier", title="B8. Gradient Boosting Classifier  [Part B, Review 2]",
         imports="from sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.pipeline import Pipeline",
         estimator="GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, random_state=42)",
         scale="False",
         what="Trees built in sequence, each one trained to correct the remaining error of everything built so far.",
         how=["Start with a constant prediction based on the overall failure rate.",
              "Compute the gradient of the loss for every row - roughly, how wrong and in which direction.",
              "Fit a small tree to those gradients.",
              "Add a learning_rate-sized fraction of it to the running prediction.",
              "Repeat 200 times. Convert the accumulated score to a probability with the sigmoid."],
         settings="learning_rate and n_estimators, which trade off against each other: halve the rate, roughly double the trees.",
         viva="It reached Recall 0.7206 - tied best - AND the second-best ROC-AUC at 0.9673. Sequential error-correction gets at the rare class in a way that parallel voting does not.",
         result="F1(weighted) = 0.9855, ranked 2nd of 10.",
         extra=""),

    dict(n=9, slug="bagging_classifier", title="B9. Bagging Classifier (Decision Tree base)  [Part B, Review 2]",
         imports="from sklearn.ensemble import BaggingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.pipeline import Pipeline",
         estimator="BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),\n                      n_estimators=100, random_state=42, n_jobs=-1)",
         scale="False",
         what="Bootstrap AGGregatING. Train 100 full decision trees on 100 different random resamples of the data and average their votes.",
         how=["Draw a bootstrap sample: 8,000 rows sampled WITH replacement, so about 63% of rows appear and some appear several times.",
              "Train a full, unpruned decision tree on it.",
              "Repeat 100 times, each on a fresh bootstrap sample.",
              "Each tree overfits its own sample, but each overfits differently.",
              "Average the votes. The random errors cancel; the real pattern survives."],
         settings="n_estimators and max_samples (what fraction of rows each tree sees). Tuning found max_samples=0.8, n_estimators=100.",
         viva="How is this different from Random Forest? Bagging uses ALL features at every split; Random Forest also randomises the FEATURES. So Random Forest is Bagging plus one extra source of randomness. Here the extra randomness hurt - Bagging beat it on recall, 0.72 versus 0.51.",
         result="F1(weighted) = 0.9859 - RANKED 1st of 10. After tuning, recall rose from 0.7206 to 0.7500.",
         extra=""),

    dict(n=10, slug="mlp_classifier", title="B10. MLP Classifier (Neural Network)  [Part B, Review 2]",
         imports="from sklearn.neural_network import MLPClassifier\nfrom sklearn.pipeline import Pipeline",
         estimator="MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',\n                  max_iter=1000, random_state=42)",
         scale="True",
         what="A small neural network: the 6 inputs feed a layer of 64 neurons, which feeds a layer of 32, which produces one failure probability.",
         how=["Each neuron computes a weighted sum of everything in the previous layer, then applies ReLU (negatives become zero, positives pass through).",
              "Stacking these non-linear layers lets the network represent shapes a straight line cannot.",
              "Forward pass: push a batch of training rows through and read the predicted probabilities.",
              "Compute the loss, then BACKPROPAGATE: use the chain rule to find how each weight contributed to the error.",
              "Nudge every weight against its gradient. Repeat for up to 1000 iterations."],
         settings="hidden_layer_sizes (shape of the network), activation, max_iter. We searched (32,)/(64,32)/(128,64) and relu/tanh.",
         viva="Two honest points. First, it needs scaling - unscaled inputs make gradients explode or vanish. Second, we originally set max_iter=600, got a ConvergenceWarning, and raised it to 1000; we did not suppress the warning. And tuning produced ZERO improvement - GridSearchCV returned the defaults.",
         result="F1(weighted) = 0.9807, but the BEST ROC-AUC of all ten at 0.9789 - it ranks cases better than anything else, even though its 0.5-threshold decisions are not the best.",
         extra=""),
]


def build_regression():
    for m in REGRESSION:
        doc = make_doc(m, "Regression", "python examples/regression/%02d_%s.py" % (m["n"], m["slug"]))
        body = REG_BODY.format(imports=m["imports"], estimator=m["estimator"],
                               scale=m["scale"], poly=m["poly"], title=m["title"],
                               extra=m["extra"])
        (EX / "regression" / f"{m['n']:02d}_{m['slug']}.py").write_text(doc + body + "\n")


def build_classification():
    for m in CLASSIFICATION:
        tag = "A" if m["n"] <= 5 else "B"
        doc = make_doc(m, "Classification (Part %s)" % tag,
                       "python examples/classification/%02d_%s.py" % (m["n"], m["slug"]))
        body = CLF_BODY.format(imports=m["imports"], estimator=m["estimator"],
                               scale=m["scale"], title=m["title"], extra=m["extra"])
        (EX / "classification" / f"{m['n']:02d}_{m['slug']}.py").write_text(doc + body + "\n")


def make_doc(m: dict, track: str, runcmd: str) -> str:
    steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(m["how"], 1))
    return f'''"""
{m["title"]}
{"=" * max(len(m["title"]), 60)}
Track: {track}

WHAT IT IS (in simple words)
----------------------------
{wrap(m["what"])}

HOW IT WORKS (step by step)
---------------------------
{steps}

SETTINGS YOU CAN TUNE
---------------------
{wrap(m["settings"])}

WHAT TO SAY IN THE VIVA
-----------------------
{wrap(m["viva"])}

OUR RESULT ON THIS DATASET
--------------------------
{wrap(m["result"])}

RUN IT
------
  {runcmd}
"""
'''


def wrap(text: str, width: int = 76) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


# ------------------------------------------------------------------ clustering
KMEANS_BODY = '''
# ---------------------------------------------------------------------------
# 3. CHOOSE k  -  the elbow method. Fit K-Means for every k from 2 to 10 and
#    record the inertia (total squared distance from each point to its own
#    cluster centre). Inertia always falls as k rises, so we look for the
#    "elbow": the point after which it stops falling steeply.
# ---------------------------------------------------------------------------
table, models = cu.kmeans_sweep(X, range(2, 11))
k_elbow = cu.elbow_knee(table["k"], table["inertia"])
k_sil = int(table.loc[table["Silhouette"].idxmax(), "k"])
print(f"\\nelbow suggests k = {k_elbow} | best silhouette at k = {k_sil}")
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
print("\\nK-Means at k =", k_elbow)
print("=" * 60)
print(f"  Silhouette        : {m['Silhouette']:.4f}   (higher better)")
print(f"  Davies-Bouldin    : {m['Davies_Bouldin']:.4f}   (LOWER better)")
print(f"  Calinski-Harabasz : {m['Calinski_Harabasz']:.1f}   (higher better)")
print(f"  exact silhouette  : {m['silhouette_exact']}  (all 10,127 rows, not sampled)")

print("\\n  Cluster sizes:")
for c in sorted(set(labels)):
    n = int((labels == c).sum())
    print(f"    cluster {c}: {n:5d} customers ({100*n/len(labels):5.2f}%)")

# ---------------------------------------------------------------------------
# 6. PROFILE  -  the mean of each ORIGINAL (unscaled) feature per cluster.
#    This table is what you interpret. Naming the clusters is TEAM work (Q-CL3).
# ---------------------------------------------------------------------------
print("\\n  Cluster profiles (original units):")
print(cu.cluster_profile(df, labels).to_string())
'''

AGG_BODY = '''
# ---------------------------------------------------------------------------
# 3. COMPARE LINKAGE STRATEGIES  -  "linkage" is the rule for measuring the
#    distance between two GROUPS of customers, not two customers.
#      ward     : merge the pair that increases total within-cluster variance least
#      complete : distance = the FARTHEST pair between the two groups
#      average  : distance = the average over all pairs
#      single   : distance = the CLOSEST pair  (prone to "chaining")
# ---------------------------------------------------------------------------
K = 5
print(f"\\nLinkage comparison at k = {K}:")
print(cu.linkage_comparison(X, K).to_string(index=False))
print("\\nWatch single linkage: its Calinski-Harabasz collapses to ~3.7 against")
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
print(f"\\nAgglomerative (ward) at k = {K}")
print("=" * 60)
print(f"  Silhouette        : {m['Silhouette']:.4f}   (higher better)")
print(f"  Davies-Bouldin    : {m['Davies_Bouldin']:.4f}   (LOWER better)")
print(f"  Calinski-Harabasz : {m['Calinski_Harabasz']:.1f}   (higher better)")

print("\\n  Cluster sizes:")
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
print(f"\\n  Dendrogram saved to {p}")

print("\\n  Cluster profiles (original units):")
print(cu.cluster_profile(df, labels).to_string())
'''


def build_clustering():
    km_doc = '''"""
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
'''
    (EX / "clustering" / "01_kmeans.py").write_text(
        km_doc + CLU_BODY.format(imports="", body=KMEANS_BODY) + "\n")

    agg_doc = '''"""
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
'''
    (EX / "clustering" / "02_agglomerative.py").write_text(
        agg_doc + CLU_BODY.format(
            imports="from sklearn.cluster import AgglomerativeClustering",
            body=AGG_BODY) + "\n")


def main() -> int:
    for d in ("regression", "classification", "clustering"):
        (EX / d).mkdir(parents=True, exist_ok=True)
    build_regression()
    build_classification()
    build_clustering()
    files = sorted(EX.rglob("*.py"))
    for f in files:
        print(f"  {f.relative_to(config.PROJECT_ROOT)}")
    print(f"\n{len(files)} standalone example scripts written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
