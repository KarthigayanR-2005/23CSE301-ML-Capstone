"""
A5. Support Vector Classifier (SVC)  [Part A, Review 1]
============================================================
Track: Classification (Part A)

WHAT IT IS (in simple words)
----------------------------
Draws the boundary between failing and healthy machines that leaves the
widest possible empty margin on both sides.

HOW IT WORKS (step by step)
---------------------------
  1. Look for a surface separating the two classes.
  2. Among all surfaces that separate them, prefer the one with the largest gap to the nearest point on each side.
  3. Allow some points to sit inside the margin or on the wrong side, penalised by C.
  4. The RBF kernel measures similarity by distance, letting the boundary curve without computing curved coordinates explicitly.
  5. Only the points nearest the boundary - the support vectors - define it. Moving a far-away point changes nothing.

SETTINGS YOU CAN TUNE
---------------------
C (low C means a wide, forgiving margin; high C means fit the training data
hard) and kernel. probability=True is needed for ROC-AUC and costs extra
fitting time.

WHAT TO SAY IN THE VIVA
-----------------------
Precision 0.875 with Recall 0.2059 - it is very careful. When it cries
failure it is almost always right, but it stays quiet about 54 of the 68
real failures. Wide-margin fitting on an imbalanced dataset pushes the
boundary toward the majority class.

OUR RESULT ON THIS DATASET
--------------------------
F1(weighted) = 0.9635, ROC-AUC = 0.9468.

RUN IT
------
  python examples/classification/05_support_vector_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import confusion_matrix
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.svm import SVC
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
    ("model", SVC(C=1.0, kernel='rbf', probability=True, random_state=42)),
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

print("\nA5. Support Vector Classifier (SVC)  [Part A, Review 1]")
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

