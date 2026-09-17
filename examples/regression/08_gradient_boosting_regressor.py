"""
8. Gradient Boosting Regressor
============================================================
Track: Regression

WHAT IT IS (in simple words)
----------------------------
Trees built one after another, where each new tree's job is to fix the
mistakes left by all the trees before it.

HOW IT WORKS (step by step)
---------------------------
  1. Start with a single flat guess: the mean Performance Index of the training set.
  2. Compute the residual for every row - how far off that guess was.
  3. Train a SMALL tree to predict those residuals, i.e. to predict the error itself.
  4. Add a fraction of that tree's output (the learning_rate) to the running prediction.
  5. Recompute the new, smaller residuals and repeat 200 times. Each tree corrects what remains.

SETTINGS YOU CAN TUNE
---------------------
learning_rate (how much of each correction to accept - small is safer, needs
more trees) and n_estimators. We searched 0.03/0.1/0.2 and 100/200/400.

WHAT TO SAY IN THE VIVA
-----------------------
The difference from Random Forest is the key question. Random Forest builds
trees in PARALLEL and averages; Gradient Boosting builds them in SEQUENCE,
each fixing the last. Boosting usually wins, and here it did - 0.9884 versus
0.9862.

OUR RESULT ON THIS DATASET
--------------------------
Test R2 = 0.988411, ranked 6th, and the best non-linear model.

RUN IT
------
  python examples/regression/08_gradient_boosting_regressor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.ensemble import GradientBoostingRegressor
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
    ("model", GradientBoostingRegressor(learning_rate=0.1, n_estimators=200, random_state=42)),
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

print("\n8. Gradient Boosting Regressor")
print("=" * 60)
print(f"  R2   : {metrics['R2']:.6f}")
print(f"  RMSE : {metrics['RMSE']:.4f}  (Performance Index points)")
print(f"  MAE  : {metrics['MAE']:.4f}  (Performance Index points)")

