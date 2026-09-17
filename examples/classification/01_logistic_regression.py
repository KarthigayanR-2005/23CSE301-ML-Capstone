"""
A1. Logistic Regression  [Part A, Review 1]
============================================================
Track: Classification (Part A)

WHAT IT IS (in simple words)
----------------------------
Despite the name, this is a classifier. It computes a weighted sum of the
inputs, then squashes that number into a probability between 0 and 1 with
the sigmoid function.

HOW IT WORKS (step by step)
---------------------------
  1. Compute z = weight1 x torque + weight2 x speed + ... + intercept.
  2. Squash it: probability = 1 / (1 + e^-z). Large positive z gives a probability near 1; large negative gives near 0.
  3. Find the weights that make the training labels most likely (maximum likelihood), by iterative optimisation.
  4. To predict a class, compare the probability to a threshold - 0.5 by default.
  5. The threshold is a CHOICE, not part of the model. Lowering it catches more failures at the cost of more false alarms.

SETTINGS YOU CAN TUNE
---------------------
C - the inverse of regularisation strength; class_weight='balanced' makes
rare failures count more. We searched C in 0.01-10 and both weightings.

WHAT TO SAY IN THE VIVA
-----------------------
Exponentiating a coefficient gives an ODDS RATIO: how much the odds of
failure multiply per one standard deviation of that feature. See
results/tables/classification_logreg_odds.csv.

OUR RESULT ON THIS DATASET
--------------------------
F1(weighted) = 0.9560, ROC-AUC = 0.8994, but Recall on failures = 0.1029. It
caught 7 of 68 real failures while scoring 96.75% accuracy - the single best
illustration of why accuracy is the wrong metric here.

RUN IT
------
  python examples/classification/01_logistic_regression.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import confusion_matrix
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
    ("prep", pp.make_preprocessor(NUM, CAT, scale=True)),
    ("model", LogisticRegression(max_iter=2000, random_state=42)),
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

print("\nA1. Logistic Regression  [Part A, Review 1]")
print("=" * 60)
print(f"  Accuracy          : {m['Accuracy']:.4f}   (majority-class baseline = 0.9660)")
print(f"  Precision(failure): {m['Precision_failure']:.4f}   of predicted failures, how many were real")
print(f"  Recall(failure)   : {m['Recall_failure']:.4f}   of real failures, how many we caught")
print(f"  F1 (weighted)     : {m['F1_weighted']:.4f}")
print(f"  ROC-AUC           : {m['ROC_AUC']:.4f}   (from {how})")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("\n  Confusion matrix on 2,000 test machines:")
print(f"    correctly called safe     : {tn:5d}")
print(f"    false alarms              : {fp:5d}  (said fail, was fine)")
print(f"    MISSED FAILURES           : {fn:5d}  (said fine, actually failed)")
print(f"    failures caught           : {tp:5d}")

# --- ALGORITHM-SPECIFIC: odds ratios (PDF note: "interpret coefficients/odds")
from src.classification_models import odds_table
o = odds_table(model)
print("\n  Coefficients and odds ratios (per 1 standard deviation):")
print(o.to_string(index=False))
