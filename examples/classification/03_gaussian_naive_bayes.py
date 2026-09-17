"""
A3. Gaussian Naive Bayes  [Part A, Review 1]
============================================================
Track: Classification (Part A)

WHAT IT IS (in simple words)
----------------------------
Uses Bayes' theorem to flip the question around. Instead of asking 'given
these readings, will it fail?', it asks 'if a machine were failing, how
likely are these readings?' - and combines that with how common failure is
overall.

HOW IT WORKS (step by step)
---------------------------
  1. For each class separately, compute the mean and variance of every feature from the training data.
  2. That gives a bell curve per feature per class.
  3. For a new machine, read off how likely each of its readings is under each class's bell curve.
  4. MULTIPLY those likelihoods together - this is the 'naive' step, and it assumes the features are independent given the class.
  5. Multiply by the class's base rate (3.39% for failure) and pick whichever class scores higher.

SETTINGS YOU CAN TUNE
---------------------
Effectively none. GaussianNB is parameter-free; it just measures means and
variances.

WHAT TO SAY IN THE VIVA
-----------------------
The independence assumption is the exam question. Air temperature and
process temperature are strongly correlated in our data - physically, the
process is heated by its surroundings. Naive Bayes multiplies their evidence
as if they were unrelated, so it double-counts one fact. Its F1 of 0.9506 is
the lowest of all ten, and that is why.

OUR RESULT ON THIS DATASET
--------------------------
F1(weighted) = 0.9506 - last of 10. Precision on failures 0.25: three
quarters of its failure alarms were false. Do NOT substitute MultinomialNB
or BernoulliNB; the PDF names Gaussian specifically.

RUN IT
------
  python examples/classification/03_gaussian_naive_bayes.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import confusion_matrix
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.naive_bayes import GaussianNB
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
    ("model", GaussianNB()),
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

print("\nA3. Gaussian Naive Bayes  [Part A, Review 1]")
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

