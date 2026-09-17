"""
B8. Gradient Boosting Classifier  [Part B, Review 2]
============================================================
Track: Classification (Part B)

WHAT IT IS (in simple words)
----------------------------
Trees built in sequence, each one trained to correct the remaining error of
everything built so far.

HOW IT WORKS (step by step)
---------------------------
  1. Start with a constant prediction based on the overall failure rate.
  2. Compute the gradient of the loss for every row - roughly, how wrong and in which direction.
  3. Fit a small tree to those gradients.
  4. Add a learning_rate-sized fraction of it to the running prediction.
  5. Repeat 200 times. Convert the accumulated score to a probability with the sigmoid.

SETTINGS YOU CAN TUNE
---------------------
learning_rate and n_estimators, which trade off against each other: halve
the rate, roughly double the trees.

WHAT TO SAY IN THE VIVA
-----------------------
It reached Recall 0.7206 - tied best - AND the second-best ROC-AUC at
0.9673. Sequential error-correction gets at the rare class in a way that
parallel voting does not.

OUR RESULT ON THIS DATASET
--------------------------
F1(weighted) = 0.9855, ranked 2nd of 10.

RUN IT
------
  python examples/classification/08_gradient_boosting_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import confusion_matrix
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.ensemble import GradientBoostingClassifier
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
    ("prep", pp.make_preprocessor(NUM, CAT, scale=False)),
    ("model", GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, random_state=42)),
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

print("\nB8. Gradient Boosting Classifier  [Part B, Review 2]")
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

