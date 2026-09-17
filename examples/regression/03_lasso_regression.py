"""
3. Lasso Regression
============================================================
Track: Regression

WHAT IT IS (in simple words)
----------------------------
Like Ridge, but its penalty can switch features off completely by setting
their weight to exactly zero. It does feature selection while it fits.

HOW IT WORKS (step by step)
---------------------------
  1. Same squared-error total as before.
  2. Add a penalty of alpha x (sum of the ABSOLUTE values of the weights).
  3. Absolute value has a sharp corner at zero, and that corner makes it possible for a weight to land exactly on zero rather than merely near it.
  4. Solve iteratively (coordinate descent) - unlike Ridge there is no one-step formula.
  5. Any feature whose weight ends at zero has been dropped from the model.

SETTINGS YOU CAN TUNE
---------------------
alpha - larger alpha zeroes out more features. We searched 0.001, 0.01, 0.1,
1.0; best was 0.001.

WHAT TO SAY IN THE VIVA
-----------------------
L1 versus L2 is the classic question. L1 (Lasso) gives sparsity - it deletes
features. L2 (Ridge) only shrinks them. On our data Lasso zeroed NOTHING: 0
of 5 coefficients. That means every one of our five inputs is carrying real
signal.

OUR RESULT ON THIS DATASET
--------------------------
Test R2 = 0.988969. Zero features dropped - reported in
results/tables/regression_lasso_sparsity.json.

RUN IT
------
  python examples/regression/03_lasso_regression.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.linear_model import Lasso
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
    ("model", Lasso(alpha=0.01, random_state=42, max_iter=10000)),
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

print("\n3. Lasso Regression")
print("=" * 60)
print(f"  R2   : {metrics['R2']:.6f}")
print(f"  RMSE : {metrics['RMSE']:.4f}  (Performance Index points)")
print(f"  MAE  : {metrics['MAE']:.4f}  (Performance Index points)")

# --- ALGORITHM-SPECIFIC: feature sparsity (PDF note: "observe feature sparsity")
from src.regression_models import sparsity_report
print("\n  Sparsity:", sparsity_report(model))
