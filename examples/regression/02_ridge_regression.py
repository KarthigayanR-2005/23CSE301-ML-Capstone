"""
2. Ridge Regression
============================================================
Track: Regression

WHAT IT IS (in simple words)
----------------------------
Linear Regression with a leash. It still fits a straight line, but it is
penalised for using large weights, which stops any single feature from
dominating.

HOW IT WORKS (step by step)
---------------------------
  1. Set up the same squared-error total as Linear Regression.
  2. Add a penalty term: alpha x (sum of all the weights squared).
  3. Minimise error + penalty together, not error alone.
  4. Because big weights now cost something, the solution shrinks every weight toward zero - but never exactly to zero.
  5. alpha controls the leash length: alpha = 0 is plain Linear Regression, huge alpha forces all weights near zero.

SETTINGS YOU CAN TUNE
---------------------
alpha - the strength of the penalty. We searched 0.01, 0.1, 1.0, 10.0, 100.0
and the best was 0.1, i.e. almost no penalty needed.

WHAT TO SAY IN THE VIVA
-----------------------
This is L2 regularisation. It helps when features are correlated or when you
have more features than data. Here neither is true, which is exactly why it
scored the same as plain Linear Regression.

OUR RESULT ON THIS DATASET
--------------------------
Test R2 = 0.988982. Tuning alpha moved the score in the 5th decimal place.

RUN IT
------
  python examples/regression/02_ridge_regression.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.linear_model import Ridge
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
    ("model", Ridge(alpha=1.0, random_state=42)),
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

print("\n2. Ridge Regression")
print("=" * 60)
print(f"  R2   : {metrics['R2']:.6f}")
print(f"  RMSE : {metrics['RMSE']:.4f}  (Performance Index points)")
print(f"  MAE  : {metrics['MAE']:.4f}  (Performance Index points)")

