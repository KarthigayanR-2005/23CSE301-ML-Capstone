"""
10. K-Nearest Neighbors Regressor
============================================================
Track: Regression

WHAT IT IS (in simple words)
----------------------------
No training at all. To predict for a new student, find the 5 most similar
students in the training data and average their Performance Index.

HOW IT WORKS (step by step)
---------------------------
  1. 'Fitting' just stores the training rows. Nothing is learned.
  2. When a prediction is requested, compute the distance from the new row to every stored training row.
  3. Sort those distances and keep the k smallest.
  4. Average the target values of those k neighbours - that is the prediction.
  5. With weights='distance', closer neighbours count for more than farther ones.

SETTINGS YOU CAN TUNE
---------------------
n_neighbors (k) - small k is jumpy and noise-sensitive, large k over-
smooths. We searched 3/5/9/15/25 and both weighting schemes.

WHAT TO SAY IN THE VIVA
-----------------------
Scaling is not optional here, it is the whole ballgame. Distance is the
model. If Previous Scores (range 40-99) were left unscaled against Sleep
Hours (range 4-9), 'similar student' would mean 'similar previous score' and
nothing else.

OUR RESULT ON THIS DATASET
--------------------------
Test R2 = 0.976841 - LAST of 10. It is the only model that cannot
extrapolate: it can never predict a value outside the range of its training
neighbours.

RUN IT
------
  python examples/regression/10_knn_regressor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, data_loading as dl, preprocessing as pp, evaluation as ev
from sklearn.neighbors import KNeighborsRegressor
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
    ("model", KNeighborsRegressor(n_neighbors=5)),
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

print("\n10. K-Nearest Neighbors Regressor")
print("=" * 60)
print(f"  R2   : {metrics['R2']:.6f}")
print(f"  RMSE : {metrics['RMSE']:.4f}  (Performance Index points)")
print(f"  MAE  : {metrics['MAE']:.4f}  (Performance Index points)")

