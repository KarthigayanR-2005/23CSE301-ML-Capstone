"""
4. ElasticNet Regression
============================================================
Track: Regression

WHAT IT IS (in simple words)
----------------------------
Ridge and Lasso mixed together. You choose how much of each penalty to
apply.

HOW IT WORKS (step by step)
---------------------------
  1. Same squared-error total.
  2. Add BOTH penalties: l1_ratio x (Lasso penalty) + (1 - l1_ratio) x (Ridge penalty), all scaled by alpha.
  3. l1_ratio = 1 makes it pure Lasso; l1_ratio = 0 makes it pure Ridge; 0.5 is an even blend.
  4. Solve iteratively, as with Lasso.
  5. You get some feature dropping from the L1 part and some smooth shrinkage from the L2 part.

SETTINGS YOU CAN TUNE
---------------------
alpha (overall penalty strength) and l1_ratio (the mix). We searched alpha
in 0.001-1.0 and l1_ratio in 0.1/0.5/0.9.

WHAT TO SAY IN THE VIVA
-----------------------
Why exist at all, if Ridge and Lasso already do? Because when features come
in correlated groups, pure Lasso arbitrarily keeps one and drops the rest,
whereas ElasticNet tends to keep or drop the group together.

OUR RESULT ON THIS DATASET
--------------------------
Test R2 = 0.988886 - the weakest of the four linear models here, by a hair.

RUN IT
------
  python examples/regression/04_elasticnet_regression.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.linear_model import ElasticNet
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
    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)),
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

print("\n4. ElasticNet Regression")
print("=" * 60)
print(f"  R2   : {metrics['R2']:.6f}")
print(f"  RMSE : {metrics['RMSE']:.4f}  (Performance Index points)")
print(f"  MAE  : {metrics['MAE']:.4f}  (Performance Index points)")

