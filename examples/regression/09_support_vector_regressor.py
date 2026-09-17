"""
9. Support Vector Regressor (SVR)
============================================================
Track: Regression

WHAT IT IS (in simple words)
----------------------------
Fits a tube around the data and tries to get as many points as possible
inside it. Errors smaller than the tube's width are treated as no error at
all.

HOW IT WORKS (step by step)
---------------------------
  1. Choose a tube width, epsilon. Any prediction within epsilon of the truth counts as correct.
  2. Find the flattest possible function that keeps most points inside the tube.
  3. Points that fall outside are penalised, with C controlling how harshly.
  4. The 'kernel trick' lets it fit a curved function without ever computing the curved coordinates - it only needs distances between points.
  5. Only the points on or outside the tube boundary (the 'support vectors') shape the final model. The rest are ignored entirely.

SETTINGS YOU CAN TUNE
---------------------
C (penalty for points outside the tube - high C means fit hard, risk
overfitting) and kernel ('rbf' curved, 'linear' straight). We searched C in
0.1/1/10 and both kernels.

WHAT TO SAY IN THE VIVA
-----------------------
Why scale=True and why it matters most here: the RBF kernel is built on
distances between points. An unscaled feature with a big range would swamp
the distance calculation and the model would effectively ignore everything
else.

OUR RESULT ON THIS DATASET
--------------------------
Test R2 = 0.985924, ranked 8th. Took 2 seconds, versus 0.014 for Linear
Regression, for a worse score.

RUN IT
------
  python examples/regression/09_support_vector_regressor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.svm import SVR
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
    ("model", SVR(C=1.0, kernel='rbf')),
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

print("\n9. Support Vector Regressor (SVR)")
print("=" * 60)
print(f"  R2   : {metrics['R2']:.6f}")
print(f"  RMSE : {metrics['RMSE']:.4f}  (Performance Index points)")
print(f"  MAE  : {metrics['MAE']:.4f}  (Performance Index points)")

