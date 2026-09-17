"""
A2. K-Nearest Neighbors  [Part A, Review 1]
============================================================
Track: Classification (Part A)

WHAT IT IS (in simple words)
----------------------------
Find the 5 most similar machines in the training data and take a majority
vote on whether they failed.

HOW IT WORKS (step by step)
---------------------------
  1. Store the training rows. No model is fitted.
  2. For a new machine, measure the distance to every training machine.
  3. Keep the k closest.
  4. Count their labels. The majority label wins.
  5. The predicted probability is simply the fraction of those k neighbours that failed - which is why with k=5 the only possible probabilities are 0, 0.2, 0.4, 0.6, 0.8, 1.0.

SETTINGS YOU CAN TUNE
---------------------
n_neighbors (k) and the distance metric. We compared euclidean, manhattan
and chebyshev.

WHAT TO SAY IN THE VIVA
-----------------------
Coarse probabilities are why its ROC-AUC (0.8291) is the second worst of all
ten models even though its accuracy looks fine. AUC needs finely graded
scores to rank cases; KNN with k=5 offers only six possible values.

OUR RESULT ON THIS DATASET
--------------------------
F1(weighted) = 0.9679, Recall(failure) = 0.2941, ROC-AUC = 0.8291.

RUN IT
------
  python examples/classification/02_knn_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import confusion_matrix
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.neighbors import KNeighborsClassifier
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
    ("model", KNeighborsClassifier(n_neighbors=5)),
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

print("\nA2. K-Nearest Neighbors  [Part A, Review 1]")
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

# --- ALGORITHM-SPECIFIC: distance metrics (PDF note: "discuss distance metrics")
from src.classification_models import knn_distance_metric_comparison
print("\n  Distance metric comparison:")
print(knn_distance_metric_comparison(NUM, CAT, X_train, y_train,
                                     X_test, y_test).to_string(index=False))
