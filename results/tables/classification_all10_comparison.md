|   Rank | Model                            |   Accuracy |   Precision_failure |   Recall_failure |   F1_weighted |   F1_failure |   ROC_AUC | AUC_source    |   Fit_seconds |
|-------:|:---------------------------------|-----------:|--------------------:|-----------------:|--------------:|-------------:|----------:|:--------------|--------------:|
|      1 | B9. Bagging (Decision Tree base) |     0.9865 |            0.859649 |         0.720588 |      0.985925 |     0.784    |  0.959818 | predict_proba |         1.699 |
|      2 | B8. Gradient Boosting Classifier |     0.986  |            0.844828 |         0.720588 |      0.985463 |     0.777778 |  0.967292 | predict_proba |         1.613 |
|      3 | B10. MLP Classifier              |     0.981  |            0.734375 |         0.691176 |      0.980722 |     0.712121 |  0.9789   | predict_proba |         2.996 |
|      4 | B6. Random Forest Classifier     |     0.9815 |            0.897436 |         0.514706 |      0.979062 |     0.654206 |  0.970866 | predict_proba |         1.689 |
|      5 | A4. Decision Tree Classifier     |     0.9755 |            0.787879 |         0.382353 |      0.971365 |     0.514851 |  0.921854 | predict_proba |         0.026 |
|      6 | A2. K-Nearest Neighbors          |     0.974  |            0.833333 |         0.294118 |      0.967929 |     0.434783 |  0.829086 | predict_proba |         0.018 |
|      7 | B7. AdaBoost Classifier          |     0.9725 |            0.740741 |         0.294118 |      0.96671  |     0.421053 |  0.949169 | predict_proba |         0.884 |
|      8 | A5. Support Vector Classifier    |     0.972  |            0.875    |         0.205882 |      0.963519 |     0.333333 |  0.946817 | predict_proba |         0.762 |
|      9 | A1. Logistic Regression          |     0.9675 |            0.636364 |         0.102941 |      0.956012 |     0.177215 |  0.899388 | predict_proba |         0.021 |
|     10 | A3. Gaussian Naive Bayes         |     0.958  |            0.25     |         0.117647 |      0.950634 |     0.16     |  0.846814 | predict_proba |         0.014 |