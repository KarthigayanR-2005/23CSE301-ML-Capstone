"""
7. Random Forest Regressor
============================================================
Track: Regression

WHAT IT IS (in simple words)
----------------------------
300 different decision trees, each deliberately trained on a slightly
different view of the data, and their predictions averaged.

HOW IT WORKS (step by step)
---------------------------
  1. Draw a random sample of the training rows WITH replacement (bootstrap) - this tree sees a different dataset.
  2. Grow a decision tree on it, but at every split consider only a random subset of the features, not all of them.
  3. That double randomness makes each tree wrong in a different direction.
  4. Repeat 300 times, independently. The trees never see each other.
  5. To predict, ask all 300 trees and average their answers. Individual errors cancel out; shared signal survives.

SETTINGS YOU CAN TUNE
---------------------
n_estimators (how many trees - more is steadier but slower) and max_depth.
We searched 100/300/500 trees.

WHAT TO SAY IN THE VIVA
-----------------------
Look at the overfitting gap: train R2 = 0.9975 but test R2 = 0.9862. The
forest has partly memorised the training set. Averaging 300 trees controls
that, but does not remove it.

OUR RESULT ON THIS DATASET
--------------------------
Test R2 = 0.986175, ranked 7th. Slowest model in the track at 3.4 seconds,
and beaten by a one-line linear fit.

RUN IT
------
  python examples/regression/07_random_forest_regressor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.ensemble import RandomForestRegressor
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
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
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

print("\n7. Random Forest Regressor")
print("=" * 60)
print(f"  R2   : {metrics['R2']:.6f}")
print(f"  RMSE : {metrics['RMSE']:.4f}  (Performance Index points)")
print(f"  MAE  : {metrics['MAE']:.4f}  (Performance Index points)")

# --- ALGORITHM-SPECIFIC: feature importance
from src.regression_models import tree_importances
print("\n  Feature importance (averaged over 300 trees):")
print(tree_importances(model).round(4).to_string())
