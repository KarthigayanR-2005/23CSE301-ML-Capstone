"""
B6. Random Forest Classifier  [Part B, Review 2]
============================================================
Track: Classification (Part B)

WHAT IT IS (in simple words)
----------------------------
300 decision trees, each on a different bootstrap sample with a random
subset of features, voting on the outcome.

HOW IT WORKS (step by step)
---------------------------
  1. Bootstrap-sample the training rows for tree 1.
  2. Grow a tree, considering only a random subset of features at each split.
  3. Repeat 300 times independently.
  4. For a new machine, collect all 300 votes.
  5. The predicted probability is the fraction of trees voting 'failure' - which gives 301 possible values and therefore a much better ROC-AUC than KNN's six.

SETTINGS YOU CAN TUNE
---------------------
n_estimators, max_depth, class_weight. We searched 200/400 trees, depth
None/10/20, and balanced weighting.

WHAT TO SAY IN THE VIVA
-----------------------
Precision 0.8974 is the highest of all ten models, but Recall 0.5147 means
it misses half the failures. Majority voting is conservative by
construction: for a rare event, a failure prediction needs over 150 of 300
trees to agree.

OUR RESULT ON THIS DATASET
--------------------------
F1(weighted) = 0.9791, ROC-AUC = 0.9709, Recall(failure) = 0.5147.

RUN IT
------
  python examples/classification/06_random_forest_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import confusion_matrix
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.ensemble import RandomForestClassifier
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
    ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
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

print("\nB6. Random Forest Classifier  [Part B, Review 2]")
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

# --- ALGORITHM-SPECIFIC: feature importance (PDF note: "feature importance")
from src.classification_models import tree_importances
print("\n  Feature importance:")
print(tree_importances(model).round(4).to_string())
