"""
B9. Bagging Classifier (Decision Tree base)  [Part B, Review 2]
===============================================================
Track: Classification (Part B)

WHAT IT IS (in simple words)
----------------------------
Bootstrap AGGregatING. Train 100 full decision trees on 100 different random
resamples of the data and average their votes.

HOW IT WORKS (step by step)
---------------------------
  1. Draw a bootstrap sample: 8,000 rows sampled WITH replacement, so about 63% of rows appear and some appear several times.
  2. Train a full, unpruned decision tree on it.
  3. Repeat 100 times, each on a fresh bootstrap sample.
  4. Each tree overfits its own sample, but each overfits differently.
  5. Average the votes. The random errors cancel; the real pattern survives.

SETTINGS YOU CAN TUNE
---------------------
n_estimators and max_samples (what fraction of rows each tree sees). Tuning
found max_samples=0.8, n_estimators=100.

WHAT TO SAY IN THE VIVA
-----------------------
How is this different from Random Forest? Bagging uses ALL features at every
split; Random Forest also randomises the FEATURES. So Random Forest is
Bagging plus one extra source of randomness. Here the extra randomness hurt
- Bagging beat it on recall, 0.72 versus 0.51.

OUR RESULT ON THIS DATASET
--------------------------
F1(weighted) = 0.9859 - RANKED 1st of 10. After tuning, recall rose from
0.7206 to 0.7500.

RUN IT
------
  python examples/classification/09_bagging_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import confusion_matrix
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
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
    ("model", BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),
                      n_estimators=100, random_state=42, n_jobs=-1)),
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

print("\nB9. Bagging Classifier (Decision Tree base)  [Part B, Review 2]")
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

