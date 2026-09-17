"""
A4. Decision Tree Classifier  [Part A, Review 1]
============================================================
Track: Classification (Part A)

WHAT IT IS (in simple words)
----------------------------
A flowchart of yes/no questions about the sensor readings, ending in a
verdict of fail or no-fail.

HOW IT WORKS (step by step)
---------------------------
  1. Begin with all 8,000 training machines in one group.
  2. Try every feature and cut point, and score each by how much it purifies the groups - measured by Gini impurity.
  3. A perfect split puts all failures on one side and all healthy machines on the other.
  4. Take the best split, then repeat on each resulting branch.
  5. Stop at max_depth. Each leaf predicts the majority class of its training members, and the probability is the class proportion in that leaf.

SETTINGS YOU CAN TUNE
---------------------
max_depth, min_samples_leaf, class_weight. We searched depth 3/5/8/12/None.

WHAT TO SAY IN THE VIVA
-----------------------
This is the only model you can literally read.
results/figures/classification/clf_decision_tree.png shows the top 3 levels
- point at the first split and state the physical rule it found.

OUR RESULT ON THIS DATASET
--------------------------
F1(weighted) = 0.9714, Recall(failure) = 0.3824 - the best Part A model on
failure recall, nearly 4x Logistic Regression.

RUN IT
------
  python examples/classification/04_decision_tree_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import confusion_matrix
from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
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
    ("model", DecisionTreeClassifier(max_depth=6, random_state=42)),
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

print("\nA4. Decision Tree Classifier  [Part A, Review 1]")
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

# --- ALGORITHM-SPECIFIC: draw the tree (PDF note: "visualise the tree")
from src import plotting as pl
p = pl.decision_tree_figure(model.named_steps["model"],
                            model.named_steps["prep"].get_feature_names_out(),
                            ["No Failure", "Failure"],
                            "example_decision_tree", "classification", max_depth=3)
print(f"\n  Tree diagram saved to {p}")
