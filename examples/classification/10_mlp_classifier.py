"""
B10. MLP Classifier (Neural Network)  [Part B, Review 2]
============================================================
Track: Classification (Part B)

WHAT IT IS (in simple words)
----------------------------
A small neural network: the 6 inputs feed a layer of 64 neurons, which feeds
a layer of 32, which produces one failure probability.

HOW IT WORKS (step by step)
---------------------------
  1. Each neuron computes a weighted sum of everything in the previous layer, then applies ReLU (negatives become zero, positives pass through).
  2. Stacking these non-linear layers lets the network represent shapes a straight line cannot.
  3. Forward pass: push a batch of training rows through and read the predicted probabilities.
  4. Compute the loss, then BACKPROPAGATE: use the chain rule to find how each weight contributed to the error.
  5. Nudge every weight against its gradient. Repeat for up to 1000 iterations.

SETTINGS YOU CAN TUNE
---------------------
hidden_layer_sizes (shape of the network), activation, max_iter. We searched
(32,)/(64,32)/(128,64) and relu/tanh.

WHAT TO SAY IN THE VIVA
-----------------------
Two honest points. First, it needs scaling - unscaled inputs make gradients
explode or vanish. Second, we originally set max_iter=600, got a
ConvergenceWarning, and raised it to 1000; we did not suppress the warning.
And tuning produced ZERO improvement - GridSearchCV returned the defaults.

OUR RESULT ON THIS DATASET
--------------------------
F1(weighted) = 0.9807, but the BEST ROC-AUC of all ten at 0.9789 - it ranks
cases better than anything else, even though its 0.5-threshold decisions are
not the best.

RUN IT
------
  python examples/classification/10_mlp_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import confusion_matrix
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.neural_network import MLPClassifier
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
    ("model", MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                  max_iter=1000, random_state=42)),
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

print("\nB10. MLP Classifier (Neural Network)  [Part B, Review 2]")
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

