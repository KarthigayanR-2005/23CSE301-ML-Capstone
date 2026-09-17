# Team-Authored Analysis — Questions To Answer

Guidelines §7.5: *"Each team's analysis, interpretation, and feature engineering
decisions must be original."* Generative AI may scaffold code but **not** the
analysis. Everything below is therefore deliberately unanswered.

Work through these **as a team**. Every member should be able to defend every
answer in the viva, not only the ones for their own track.

**Legend:** 🔴 blocks a rubric mark · 🟡 strengthens the review · ⚪ viva preparation

---

## Regression track

### 🔴 Q-RG1 — Duplicate-row policy
127 exact duplicate rows exist. Retaining them keeps 10,000 rows; removing them
leaves 9,873 (below the working preference).
- Given that the five inputs are low-cardinality integers and one binary flag,
  how many identical rows would you *expect* by chance in 10,000 draws?
- Coincidence, or a data-generation artefact?
- **Your decision and justification:**

### 🔴 Q-RG2 — Train/test overlap
`check_no_row_leakage` reports ~12.9 % of test rows whose feature values also
appear in training.
- Is this "leakage" in the harmful sense, or an inevitable consequence of low
  feature cardinality?
- Does it inflate the R² figures, and if so by how much would you guess?
- **Your position:**

### 🔴 Q-RG3 — Engineered feature *(rubric B3 — currently INCOMPLETE)*
`src/feature_engineering.py` is empty by design.
- Which feature will you create, from which columns?
- **State your justification before you measure it.**
- Register it, re-run `python scripts/run_all.py --track regression --mode full`,
  and report the honest before/after — including if it made things worse.
- **Feature / justification / result:**

### 🟡 Q-RG4 — Why do all ten models land within ~1 % R²?
Every model scores between about 0.977 and 0.989 test R².
- What does that say about the relationship between inputs and target?
- What does it say about how this synthetic data was generated?
- Why do the ensembles fail to beat plain linear regression here?

### 🟡 Q-RG5 — Model selection
- Which model do you nominate, and on what grounds — accuracy, stability,
  simplicity, interpretability, training cost?
- Tuning moved CV R² only in the fourth decimal. Does that change your pick?

### ⚪ Q-RG6 — Viva readiness
- What does R² = 0.989 actually mean in words?
- Why is RMSE ≈ 2.0 index points more useful to a reader than R²?
- What would a residual plot look like if the model were missing a non-linear term?

---

## Classification track

### 🔴 Q-CF1 — Engineered feature *(rubric B3 — currently INCOMPLETE)*
- Which physical quantity could you derive from the six inputs?
- **Justify before measuring.**
- Did it improve *failure recall*, and did it hurt anything else?
- **Feature / justification / result:**

### 🔴 Q-CF2 — Final model selection *(Review 2 rubric A3)*
- Which of the ten do you select, and on which metric?
- Quote the before/after tuning numbers. If tuning did not help, say so plainly.
- Does your model trade failure recall for accuracy? Acceptable?

### 🟡 Q-CF3 — Which metric should lead, and why not accuracy?
Only 3.39 % of rows are failures; predicting "no failure" always scores ~96.6 %.
- Which metric leads your comparison and why?
- Would you lower the decision threshold below 0.5? What would that cost?

### 🟡 Q-CF4 — Why is failure recall low across most models?
Several models catch well under half the failures.
- Is that the models' fault, the imbalance, or the feature set?
- If you applied class weights or resampling, it must be inside training folds
  only. Did you? What happened?

### 🟡 Q-CF5 — The Naive Bayes assumption
Gaussian NB assumes features are conditionally independent given the class. The
correlation heatmap shows at least one strongly correlated pair.
- Name the pair and the physical reason.
- How did the violation show up in GaussianNB's score?

### ⚪ Q-CF6 — Viva readiness
- Why is ROC-AUC computed from probabilities and not from predicted labels?
- What does weighted F1 average over, and why is it flattering here?
- Read the top two splits of the decision tree: what rule has it learned?

---

## Clustering track

### 🔴 Q-CL1 — Feature subset justification *(currently PROVISIONAL)*
The seven features are a starting suggestion and **not yet your decision**.
- Why these seven? What behaviour does each capture?
- What did you reject and why?
- Note `Credit_Limit`, `Avg_Open_To_Buy` and `Total_Revolving_Bal` are
  algebraically related — how did that affect your choice?
- **Your subset and justification:**

### 🔴 Q-CL2 — Choice of k
The elbow detector and the silhouette maximum do not agree.
- Where do *you* read the elbow?
- Which criterion do you follow and why?
- Does a larger k give describable groups or just smaller ones?
- **Your k and reasoning:**

### 🔴 Q-CL3 — Cluster interpretation
Using the profile tables in `results/tables/clustering_profile_*.csv`:
- Give each cluster a short descriptive **name**.
- Name the two or three features that most distinguish each from the average.
- Which cluster matters most to a bank, and what action follows?
- Would you merge any two?

### 🟡 Q-CL4 — Post-hoc churn check
- Does any cluster show an attrition rate well above the overall rate?
- If clusters do **not** track churn, is the segmentation worthless? Argue it.
- Why would using this table to pick k have been methodologically wrong?

### 🟡 Q-CL5 — Are these clusters real?
Silhouette scores here are low in absolute terms.
- "Clusters discovered" or "a partition imposed on a continuum"? Defend it.
- K-Means resamples far more stably than Agglomerative. What follows from that?

### ⚪ Q-CL6 — Viva readiness
- Why must features be scaled before clustering here? Name the feature that
  would otherwise dominate.
- Why is fitting on two PCA components and then plotting them circular reasoning?
- Why can t-SNE exaggerate separation?

---

## Cross-cutting

### 🟡 Q-X1 — Problem statements
`README.md` carries three **draft** problem statements. Rewrite them in your own
words — they are drafts, not final text.

### 🟡 Q-X2 — Both supervised datasets are synthetic
- What can you legitimately conclude from a model trained on synthetic data?
- What would you need before trusting either in the real world?

### ⚪ Q-X3 — Presentation story arc *(Review 2 rubric D1)*
problem → data → method → results → conclusion. Who presents which track, and
what is the single sentence each track must land?
