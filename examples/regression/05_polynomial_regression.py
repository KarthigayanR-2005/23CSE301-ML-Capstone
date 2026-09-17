"""
5. Polynomial Features + Linear Regression
============================================================
Track: Regression

WHAT IT IS (in simple words)
----------------------------
Linear Regression given extra, invented columns: every feature squared, and
every pair of features multiplied together. The model is still linear in its
weights, but it can now bend.

HOW IT WORKS (step by step)
---------------------------
  1. Start with the 5 preprocessed input columns.
  2. PolynomialFeatures(degree=2) manufactures new columns: hours^2, sleep^2, hours x sleep, and so on. 5 columns become 20.
  3. Hand all 20 columns to an ordinary Linear Regression.
  4. It fits weights for the new columns exactly as it did for the originals.
  5. The result is a curved surface in the original feature space, even though the fitting maths never changed.

SETTINGS YOU CAN TUNE
---------------------
degree - how far to expand. Higher degree means far more columns and a
strong risk of overfitting.

WHAT TO SAY IN THE VIVA
-----------------------
Degree 1 gave 5 terms and test R2 0.98898. Degree 2 gave 20 terms and
0.98899. Degree 3 gave 55 terms and 0.98896 - WORSE despite 11x the columns.
That is overfitting made visible: more flexibility, worse performance on
unseen data.

OUR RESULT ON THIS DATASET
--------------------------
Test R2 = 0.988989. Ranked 1st of 10 - but it beat plain Linear Regression
only in the 5th decimal, using 4x the terms.

RUN IT
------
  python examples/regression/05_polynomial_regression.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

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
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
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

print("\n5. Polynomial Features + Linear Regression")
print("=" * 60)
print(f"  R2   : {metrics['R2']:.6f}")
print(f"  RMSE : {metrics['RMSE']:.4f}  (Performance Index points)")
print(f"  MAE  : {metrics['MAE']:.4f}  (Performance Index points)")

# --- ALGORITHM-SPECIFIC: compare degrees (PDF note: "compare degrees")
from src.regression_models import polynomial_degree_comparison
print("\n  Degree comparison:")
print(polynomial_degree_comparison(NUM, CAT, X_train, y_train, X_test, y_test,
                                   degrees=(1, 2, 3)).to_string(index=False))
