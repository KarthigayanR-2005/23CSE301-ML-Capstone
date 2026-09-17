# Review 2 — Presentation Outline

**Scope:** Classification Part B + full Clustering track · **25 marks (+bonus)**
Rubric D1 explicitly wants a story arc: **problem → data → method → results →
conclusion**.

> Structure only. The narrative is yours to write.

---

## Suggested timing (~15 min + questions)

| # | Slide | Min | Owner |
|---|---|---:|---|
| 1 | Recap: what Review 1 established | 1.0 | — |
| 2 | Part B: the five new algorithms | 1.5 | classification owner |
| 3 | **Consolidated 10-algorithm table** (D6) | 2.0 | classification owner |
| 4 | Final model selection + tuning evidence | 2.0 | classification owner |
| 5 | Clustering: problem + feature choice | 1.5 | clustering owner |
| 6 | Choosing k: elbow vs silhouette | 1.5 | clustering owner |
| 7 | Both algorithms + metric table | 1.5 | clustering owner |
| 8 | Dendrogram + linkage comparison | 1.0 | clustering owner |
| 9 | PCA + t-SNE visualisations | 1.5 | clustering owner |
| 10 | Cluster interpretation — the payoff | 1.5 | clustering owner |
| 11 | Pipeline quality, repo, README | 0.5 | third member |
| 12 | GUI demo (bonus) | 1.0 | third member |
| 13 | Conclusion + honest limitations | 0.5 | all |

---

## Slide notes

**3 — The consolidated table.** Deliverable D6 and rubric A2. One table, ten
rows, same split. Sort by weighted F1 but **talk about failure recall** — the
spread is 0.10 to 0.75 while accuracy barely moves from 0.958 to 0.987. That gap
is the story.

**4 — Selection (2 marks).** Rubric wants a *justification*, not a winner. Name
the metric that drove it and why. Show the tuning evidence: Bagging recall
0.7206 → 0.7500. Then say plainly that **MLP tuning gave no improvement** —
volunteering a negative result reads as competence, and hiding it is the thing
that gets caught.

**6 — Choosing k (Q-CL2).** The elbow says 5, the silhouette says 2. Do not
pretend they agree. Show both, state your choice, defend it. An examiner who
sees you acknowledge the conflict will not need to find it.

**7 — Metrics.** Remember **lower Davies-Bouldin is better**; the other two are
higher-is-better. State that the silhouette is exact on all 10,127 rows, not
sampled.

**8 — Linkage.** The single-linkage result is a gift: Calinski-Harabasz collapses
to **3.7** against ward's 1944 — textbook chaining. Show it and explain why you
rejected it.

**9 — Visualisations.** Say explicitly: *both algorithms were fitted on all seven
standardised features; PCA and t-SNE are projections for viewing only.* Note
PC1+PC2 hold **55 %** of variance, so apparent overlap may not be real overlap.
Note the t-SNE subsample.

**10 — Interpretation.** The marks and the viva both live here. Named clusters
with 2–3 distinguishing features each and one business recommendation. If you
discuss the churn crosstab, say clearly that it was computed **after** the models
were fixed.

**12 — GUI.** Demo it live from `streamlit run app/streamlit_app.py`. State that
it is **local only** and you are not claiming the deployment bonus. Do not show a
URL you do not have.

**13 — Limitations.** Silhouette ≈ 0.20 is weak. Agglomerative resamples at ARI
0.31 versus K-Means' 0.87. Saying this yourself is far better than being told it.

---

## Pre-review checklist

- [ ] All ✍️ cells in `classification.ipynb` §5–9 and `clustering.ipynb` written
- [ ] Clustering feature subset justified (Q-CL1) — no longer PROVISIONAL
- [ ] k chosen and defended (Q-CL2)
- [ ] Clusters named and interpreted (Q-CL3)
- [ ] Final classification model justified (Q-CF2)
- [ ] Problem statements rewritten in your own words (Q-X1)
- [ ] Notebooks re-executed after all of the above
- [ ] `python scripts/validate_project.py` — no FAIL rows
- [ ] `git log --oneline` shows meaningful milestones (rubric C3: a single bulk
      commit scores **zero**)
- [ ] GUI runs on the presenting laptop
- [ ] IC-3 and IC-4 raised with the instructor

## Questions to expect

1. Why does accuracy barely change while failure recall varies 7×?
2. Why is ROC-AUC computed from probabilities, not labels?
3. Bagging vs Random Forest — what is actually different?
4. Why can't you cross-validate agglomerative clustering?
5. You used PCA to plot. Why not cluster on the components?
6. Your silhouette is 0.20. Are these real clusters?
7. Why is scaling mandatory here? Which feature would dominate without it?
8. How do you know `Attrition_Flag` did not influence the clustering?
9. Why does t-SNE look more separated than PCA?
10. Cluster 3 has 4.5 % attrition, cluster 2 has 24 %. What do you do with that?
