# Standalone Algorithm Examples

One runnable script per algorithm — 22 in total. Built for **viva preparation**.

Guidelines §7.4: *"Ownership means they can explain every line of code in that
section during the viva."* A notebook cell that loops over ten models is hard to
defend line by line. A 40-line script for one model is not.

## What each script contains

1. **Docstring** — what the algorithm is in plain words, how it works step by
   step, which settings matter, what to say in the viva, and our actual result.
2. **Numbered code sections** — load → split → build → train → evaluate, each
   with a comment explaining *why*, not just *what*.
3. **Algorithm-specific extras** where the PDF asks for them — coefficients,
   Lasso sparsity, polynomial degree comparison, tree importance, odds ratios,
   distance-metric comparison, tree diagram.

## Running them

```bash
python examples/regression/01_linear_regression.py
python examples/classification/09_bagging_classifier.py
python examples/clustering/01_kmeans.py
```

Each is self-contained and takes seconds (the SVC and MLP take a little longer).
All 22 were executed and verified.

## Index

### Regression
| # | File | Algorithm | Our test R² |
|---|---|---|---:|
| 1 | `regression/01_linear_regression.py` | Linear Regression | 0.988983 |
| 2 | `regression/02_ridge_regression.py` | Ridge (L2) | 0.988982 |
| 3 | `regression/03_lasso_regression.py` | Lasso (L1) | 0.988969 |
| 4 | `regression/04_elasticnet_regression.py` | ElasticNet (L1+L2) | 0.988886 |
| 5 | `regression/05_polynomial_regression.py` | Polynomial + Linear | **0.988989** |
| 6 | `regression/06_decision_tree_regressor.py` | Decision Tree | 0.983901 |
| 7 | `regression/07_random_forest_regressor.py` | Random Forest | 0.986175 |
| 8 | `regression/08_gradient_boosting_regressor.py` | Gradient Boosting | 0.988411 |
| 9 | `regression/09_support_vector_regressor.py` | SVR | 0.985924 |
| 10 | `regression/10_knn_regressor.py` | KNN | 0.976841 |

### Classification
| # | File | Algorithm | Part | F1 (w) | Recall (fail) |
|---|---|---|---|---:|---:|
| 1 | `classification/01_logistic_regression.py` | Logistic Regression | A | 0.9560 | 0.1029 |
| 2 | `classification/02_knn_classifier.py` | KNN | A | 0.9679 | 0.2941 |
| 3 | `classification/03_gaussian_naive_bayes.py` | Gaussian NB | A | 0.9506 | 0.1176 |
| 4 | `classification/04_decision_tree_classifier.py` | Decision Tree | A | 0.9714 | 0.3824 |
| 5 | `classification/05_support_vector_classifier.py` | SVC | A | 0.9635 | 0.2059 |
| 6 | `classification/06_random_forest_classifier.py` | Random Forest | B | 0.9791 | 0.5147 |
| 7 | `classification/07_adaboost_classifier.py` | AdaBoost | B | 0.9667 | 0.2941 |
| 8 | `classification/08_gradient_boosting_classifier.py` | Gradient Boosting | B | 0.9855 | 0.7206 |
| 9 | `classification/09_bagging_classifier.py` | Bagging (DT base) | B | **0.9859** | 0.7206 |
| 10 | `classification/10_mlp_classifier.py` | MLP Neural Network | B | 0.9807 | 0.6912 |

### Clustering
| # | File | Algorithm | Silhouette | Davies-Bouldin |
|---|---|---|---:|---:|
| 1 | `clustering/01_kmeans.py` | K-Means | **0.1975** | 1.4304 |
| 2 | `clustering/02_agglomerative.py` | Agglomerative (ward) | 0.1612 | 1.6437 |

## Important

These are **study aids**. The figures reported in `README.md` come from
`scripts/run_all.py`, which trains every model on one shared split so the
comparison is fair. The examples re-create the same split with the same seed, so
their numbers match — but `run_all.py` is the source of record.

Regenerate them with `python scripts/build_examples.py`.
