"""
1. Linear Regression
============================================================
Track: Regression

WHAT IT IS (in simple words)
----------------------------
Draws the single best straight line (in 5 dimensions, a flat plane) through
the data. It assumes the target is just a weighted sum of the inputs plus a
constant.

HOW IT WORKS (step by step)
---------------------------
  1. Give every input feature a weight, starting from anything.
  2. For each student, predict: weight1 x hours + weight2 x previous score + ... + intercept.
  3. Measure how wrong the predictions are, using the sum of squared errors.
  4. Solve, in one step of algebra (the 'normal equation'), for the weights that make that total error as small as it can be.
  5. Those weights are the trained model. There is nothing to iterate - the answer is exact.

SETTINGS YOU CAN TUNE
---------------------
None worth tuning. That is the point of a baseline: it has no knobs to hide
behind.

WHAT TO SAY IN THE VIVA
-----------------------
A coefficient says: if this feature goes up by one standard deviation and
everything else stays fixed, the predicted Performance Index changes by this
much. Our inputs are standardised, so the coefficients are directly
comparable to each other.

OUR RESULT ON THIS DATASET
--------------------------
Test R2 = 0.988983, RMSE = 2.02 points. Ranked 2nd of 10 - a plain straight
line beat eight more complicated models.

RUN IT
------
  python examples/regression/01_linear_regression.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.linear_model import LinearRegression
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
preprocessor = pp.make_preprocessor(NUM, CAT, scale=True)
model = Pipeline([
    ("prep", preprocessor),
    ("model", LinearRegression()),
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

print("\n1. Linear Regression")
print("=" * 60)
print(f"  R2   : {metrics['R2']:.6f}")
print(f"  RMSE : {metrics['RMSE']:.4f}  (Performance Index points)")
print(f"  MAE  : {metrics['MAE']:.4f}  (Performance Index points)")

# --- ALGORITHM-SPECIFIC: read the coefficients (PDF note: "interpret coefficients")
from src.regression_models import coefficient_table
coefs = coefficient_table(model)
print("\n  Coefficients (change in Performance Index per 1 standard deviation):")
print(coefs.to_string(index=False))
print(f"  intercept: {coefs.attrs['intercept']:.4f}")
