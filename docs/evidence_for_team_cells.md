# Evidence Pack — the numbers for every ✍️ cell

**What this is:** every figure you need, already extracted, so you are not
digging through 47 tables while writing.

**What this is not:** the answers. Guidelines §7.5 requires the analysis to be
your own, and the viva is worth 10 of your 50 marks — answers you didn't write
are answers you can't defend when the examiner picks you and asks "why?".

**How to use it:** each block gives you the FACTS and then the ONE judgment
you have to supply. Most cells need two or three sentences, not an essay.

---

# REGRESSION

## Q-RG1 · Duplicate rows — the decision

**Facts:**
- 127 exact duplicate rows in 10,000.
- Input cardinality: Hours Studied **9** values, Previous Scores **60**, Sleep Hours **6**, Papers **10**, Extracurricular **2**.
- Distinct input combinations: 9 × 60 × 6 × 10 × 2 = **64,800**.
- Draw 10,000 rows from 64,800 possible input patterns and repeats are arithmetically unavoidable (birthday-problem territory).
- Target takes 91 distinct values, so full-row combinations ≈ 5.9 million.
- Nothing else is wrong: **0 missing cells**, no impossible values.

**You decide:** coincidence or data-collection artefact? Keep or drop? One
sentence of reasoning. (Dropping gives 9,873 rows — under the 10,000 preference.)

## Q-RG2 · Train/test overlap

**Facts:**
- 258 of 2,000 test rows (**12.9%**) have feature values that also appear in training.
- With only 64,800 possible input patterns and 8,000 training rows, roughly 12% of any new sample will match a training pattern by chance.
- These are *feature* matches. The target still varies.

**You decide:** is this harmful leakage, or unavoidable given the cardinality?
Does it inflate R²?

## Q-RG3 · Engineered feature ← **required, rubric B3, 1 mark**

**Facts to work from:**
- Correlation with target: Previous Scores **0.915**, Hours Studied **0.374**, Sleep Hours **0.048**, Papers **0.043**.
- Max correlation *between* inputs: **0.018** — the inputs are essentially independent of each other.
- Lasso zeroed **0 of 5** coefficients — every input carries signal.
- Current best test R² = **0.98899**. Ceiling is close.

**You decide:** which feature, built from which columns, and **why you expect it
to help — written before you measure**. Then register it in
`src/feature_engineering.py`, re-run, and report the honest result.

## Q-RG4 · Why all ten models land within 1.2% R²

**Facts:**
- Spread: 0.9768 (KNN) to 0.9890 (Polynomial). Range = 0.0122.
- Linear models occupy ranks 1–5; every ensemble and kernel method ranks below them.
- Random Forest: train R² **0.9975** vs test **0.9862** — a 0.011 gap.
- Polynomial degree 3 (55 terms) scored **worse** than degree 2 (20 terms).
- Dataset is **synthetic**.

**You decide:** what does this pattern say about how the data was generated, and
why flexibility doesn't pay here?

## Q-RG5 · Model selection

**Facts:**
- Rank 1 Polynomial deg-2: R² 0.988989, 20 terms, 0.016 s.
- Rank 2 Linear: R² 0.988983, 5 terms, 0.014 s. **Difference: 0.000006.**
- Rank 7 Random Forest: R² 0.986175, 300 trees, 3.426 s.
- Tuning moved CV R² in the 5th decimal.

**You decide:** which do you ship, and on what grounds — accuracy, simplicity,
interpretability, cost?

---

# CLASSIFICATION

## Q-CF1 · Engineered feature ← **required, rubric B3**

**Facts:**
- Mechanical power = torque × angular velocity. You have both: Torque [Nm] and Rotational speed [rpm].
- Correlation with Target: Torque **0.191**, Tool wear **0.105**, Air temp **0.083**, Process temp **0.036**, Speed **−0.044**. No single feature is strong.
- Failure-vs-healthy means — Torque **50.2 vs 39.6**, Tool wear **143.8 vs 106.7**, Speed **1496 vs 1540**, Air temp **300.9 vs 300.0**.
- Temperature *difference* between process and air averages ~10 K.

**You decide:** which derived quantity, and **your justification, written first**.
Then measure the effect on failure recall and report it honestly either way.

## Q-CF2 · Final model selection ← **required, 2 marks**

**Facts:**

| Model | F1w | Recall(fail) | Precision(fail) | ROC-AUC |
|---|---|---|---|---|
| B9 Bagging | 0.9859 | 0.7206 | 0.8596 | 0.9598 |
| B8 Gradient Boosting | 0.9855 | 0.7206 | 0.8448 | 0.9673 |
| B10 MLP | 0.9807 | 0.6912 | 0.7344 | **0.9789** |
| B6 Random Forest | 0.9791 | 0.5147 | **0.8974** | 0.9709 |

- Baseline: accuracy 0.9660, F1w 0.9493.
- Tuning: Bagging recall **0.7206 → 0.7500**. MLP: **no change at all**.
- Different metrics crown different winners: F1w → Bagging, AUC → MLP, Precision → Random Forest.

**You decide:** which model, which metric drove it, and is the recall/precision
trade acceptable for a maintenance setting?

## Q-CF3 · Which metric should lead

**Facts:**
- 339 failures in 10,000 rows = **3.39%**.
- Predicting "no failure" always → accuracy **0.9660**, F1w **0.9493**.
- Accuracy across all ten models spans only 0.958–0.987.
- Recall on failures spans **0.103–0.721** — a 7× spread.
- Logistic Regression: accuracy 0.9675 (above baseline) while catching **7 of 68** failures.

**You decide:** which metric leads, and would you move the 0.5 threshold?

## Q-CF5 · The Naive Bayes assumption

**Facts:**
- Air temp ↔ Process temp: **r = 0.876**.
- Rotational speed ↔ Torque: **r = −0.875**.
- GaussianNB: F1w **0.9506** — last of ten; precision on failures **0.25**.
- It multiplies per-feature likelihoods as if independent.

**You decide:** name the pair, the physical reason, and connect it to the score.

## Q-CF6 · Tree structure

**Facts:** `results/figures/classification/clf_decision_tree.png`, depth-3 view.
Logistic odds ratios: Torque **15.2×**, Speed **7.5×**, Air temp **4.2×**,
Tool wear **2.2×** per standard deviation.

**You decide:** read the top two splits and state the physical rule in words.

---

# CLUSTERING

## Q-CL1 · Feature subset justification ← **required**

**Facts:**
- Current seven: Customer_Age, Months_on_book, Total_Relationship_Count, Credit_Limit, Total_Trans_Amt, Total_Trans_Ct, Avg_Utilization_Ratio.
- Correlations inside the subset: Total_Trans_Amt ↔ Total_Trans_Ct **0.807**; Customer_Age ↔ Months_on_book **0.789**; Credit_Limit ↔ Avg_Utilization_Ratio **−0.483**.
- Excluded but available: Avg_Open_To_Buy, Total_Revolving_Bal — algebraically **Credit_Limit = Total_Revolving_Bal + Avg_Open_To_Buy**, so including all three would triple-count one fact.
- Also excluded: Gender, Education_Level, Marital_Status, Income_Category, Card_Category, Dependent_count, Months_Inactive_12_mon, Contacts_Count_12_mon, and the two ratio-change columns.

**You decide:** why these seven? What behaviour does each capture? Two pairs are
correlated above 0.78 — does that double-count in the distance metric, and do
you keep both?

## Q-CL2 · Choice of k ← **required**

**Facts:**

| k | Inertia | Silhouette | Davies-Bouldin ↓ | Calinski-Harabasz |
|---:|---:|---:|---:|---:|
| 2 | 57331 | **0.2954** | 1.4020 | 2394 |
| 3 | 47331 | 0.1875 | 1.6769 | **2520** |
| 4 | 40987 | 0.1895 | 1.5139 | 2462 |
| **5** | 36580 | 0.1975 | 1.4304 | 2373 |
| 6 | 33624 | 0.1872 | 1.4711 | 2243 |
| 8 | 29339 | 0.1844 | **1.3347** | 2047 |
| 10 | 26389 | 0.1864 | 1.3779 | 1896 |

- Elbow detector → **k = 5**. Silhouette max → **k = 2**. Calinski-Harabasz max → **k = 3**. Davies-Bouldin min → **k = 8**.
- **All four criteria disagree.**
- At k=5 the clusters are 12.0% / 26.7% / 26.9% / 10.9% / 23.5% — reasonably balanced.

**You decide:** your k and your reason. Does k=2 give you two describable
customer types, or one big blob split down the middle?

## Q-CL3 · Cluster interpretation ← **required, the heart of the track**

**Facts — K-Means profiles at k=5** (dataset means: Age 46.3, Months 35.9,
Relationships 3.81, Credit_Limit 8,632, Trans_Amt 4,404, Trans_Ct 64.9,
Utilization 0.27):

| Cluster | n | % | Age | Months | Rel. | Credit_Limit | Trans_Amt | Trans_Ct | Utilization |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1219 | 12.0 | 46.3 | 35.9 | 3.97 | **26,920** | 3,515 | 58.6 | **0.044** |
| 1 | 2701 | 26.7 | **38.5** | **28.7** | 4.25 | 5,789 | 3,220 | 58.8 | 0.212 |
| 2 | 2726 | 26.9 | **53.4** | **42.6** | 3.98 | 6,324 | 3,003 | 54.2 | 0.163 |
| 3 | 1102 | 10.9 | 45.4 | 35.2 | **2.13** | 13,889 | **12,570** | **103.9** | 0.166 |
| 4 | 2379 | 23.5 | 47.5 | 36.9 | 3.82 | **2,699** | 4,027 | 69.1 | **0.643** |

Each cluster has an obvious standout (bolded). **You decide:** a short NAME for
each, the 2–3 features that distinguish it, which one the bank should care about
most, and what action follows.

## Q-CL4 · Post-hoc churn check

**Facts:** overall attrition **16.07%**.

| Cluster | Attrited | Existing | % attrited |
|---:|---:|---:|---:|
| 0 | 216 | 1003 | 17.72 |
| 1 | 514 | 2187 | 19.03 |
| 2 | 653 | 2073 | **23.95** |
| 3 | 50 | 1052 | **4.54** |
| 4 | 194 | 2185 | 8.15 |

- Spread: 4.54% to 23.95% against a 16.07% base. Cluster 3 (the heavy transactors) churns least.
- `Attrition_Flag` was withheld from feature selection, scaling, k selection and fitting. This table was computed **after** everything was fixed.

**You decide:** is this meaningful separation? If clusters didn't track churn at
all, would the segmentation be worthless? Why would using this table to pick k
have been cheating?

## Q-CL5 · Are these clusters real?

**Facts:**
- Best silhouette at k=5 is **0.1975**. Rough convention: >0.7 strong, 0.5–0.7 reasonable, 0.25–0.5 weak, <0.25 very weak.
- Stability over 20 resamples: K-Means ARI **0.867**, Agglomerative **0.307**.
- PCA PC1+PC2 explain only **55.0%** of variance.
- Skew: Total_Trans_Amt **2.04**, Credit_Limit **1.67**.

**You decide:** "clusters discovered" or "a partition imposed on a continuum"?
Defend it. And what follows from Agglomerative's instability?

## Q-CL6 · Why scaling is mandatory

**Facts — raw ranges:**

| Feature | Min | Max | Std |
|---|---:|---:|---:|
| Credit_Limit | 1,438 | **34,516** | **9,089** |
| Total_Trans_Amt | 510 | 18,484 | 3,397 |
| Total_Trans_Ct | 10 | 139 | 23.5 |
| Customer_Age | 26 | 73 | 8.0 |
| Avg_Utilization_Ratio | 0.0 | **1.0** | **0.28** |

Credit_Limit's standard deviation is **32,000× larger** than
Avg_Utilization_Ratio's. Unscaled Euclidean distance would be Credit_Limit and
essentially nothing else.

**You decide:** state it in your own words for the viva.

---

# CROSS-CUTTING

## Q-X2 · Both supervised datasets are synthetic

**Facts:** regression — 10 algorithms within 1.2% R², max inter-feature
correlation 0.018, no missing values. Classification — exactly 3.39% positives,
two feature pairs at |r| ≈ 0.88. Real-world data rarely looks this tidy.

**You decide:** what can you legitimately conclude, and what would you need
before trusting either in production?
