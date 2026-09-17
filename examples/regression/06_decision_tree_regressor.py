"""
6. Decision Tree Regressor
============================================================
Track: Regression

WHAT IT IS (in simple words)
----------------------------
A flowchart of yes/no questions. Follow the answers down to a leaf, and the
prediction is the average target value of the training students who ended up
in that leaf.

HOW IT WORKS (step by step)
---------------------------
  1. Start with all 8,000 training rows in one group.
  2. Try every feature and every possible cut point (e.g. 'Previous Scores < 62?').
  3. Pick the single cut that most reduces the variance of the target inside the two resulting groups.
  4. Split, then repeat the whole procedure independently on each group.
  5. Stop at max_depth, or when a group gets too small. Each final group is a leaf, and its prediction is the mean target of its members.

SETTINGS YOU CAN TUNE
---------------------
max_depth - how many questions deep the tree may go. Deeper means it can
memorise the training data. We searched 3, 5, 8, 12, None plus
min_samples_leaf.

WHAT TO SAY IN THE VIVA
-----------------------
Note scale=False. Trees do not care about feature scale at all, because a
cut at 'hours < 5' means the same thing whatever units hours are in. Leaving
them unscaled also keeps the tree readable in original units.

OUR RESULT ON THIS DATASET
--------------------------
Test R2 = 0.983901, ranked 9th. A single tree predicts in steps, but the
true relationship here is smooth - so steps cost accuracy.

RUN IT
------
  python examples/regression/06_decision_tree_regressor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

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
preprocessor = pp.make_preprocessor(NUM, CAT, scale=False)
model = Pipeline([
    ("prep", preprocessor),
    ("model", DecisionTreeRegressor(max_depth=8, random_state=42)),
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

print("\n6. Decision Tree Regressor")
print("=" * 60)
print(f"  R2   : {metrics['R2']:.6f}")
print(f"  RMSE : {metrics['RMSE']:.4f}  (Performance Index points)")
print(f"  MAE  : {metrics['MAE']:.4f}  (Performance Index points)")

# --- ALGORITHM-SPECIFIC: feature importance (PDF note: "show feature importance")
from src.regression_models import tree_importances
print("\n  Feature importance:")
print(tree_importances(model).round(4).to_string())
