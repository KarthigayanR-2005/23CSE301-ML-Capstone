# Instructor Clarifications Required

Ambiguities found while mapping the PDF to an implementation. Each is
**implemented one way and documented**, never silently resolved. Bring these to
the instructor before Review 1 where marked.

---

## IC-1 — Dataset assignment: "assigned by instructor" vs team-selected
**PDF says (§1, §Overview table):** "using the dataset and problem statement
**assigned by the instructor**" and "Datasets: Assigned by instructor (one
dataset per track per team)".

**What we did:** the three datasets in this repository were **selected by the
team**, not issued by the instructor.

**Risk:** if datasets were in fact assigned and we missed it, every result here
is on the wrong data and both reviews are affected.

**Ask:** *Were datasets assigned? If so, which three?* **Resolve before Review 1.**

---

## IC-2 — "Stratified train/test split" applied to a continuous regression target
**PDF says (§5, rubric B2):** "…correct scaler applied (fitted on train set
only); **stratified** train/test split used." This criterion is shared across
tracks, but the regression target (`Performance Index`) is continuous.
Stratification is defined for discrete strata.

**What we did:** an unstratified random 80:20 split (`random_state=42`) for
regression, and a genuinely stratified split for classification.
`pp.split_supervised(..., stratify_bins=k)` implements quantile-binned
stratification and is ready to switch on.

**Why:** binning a continuous target to force stratification is a defensible
technique but it is a *choice*, not a requirement, and it changes the split. We
did not want to silently alter the split to satisfy a wording ambiguity.

**Ask:** *Does B2's "stratified" apply to the regression track? If yes, how many
quantile bins?* A one-line answer flips a flag.

---

## IC-3 — "Cross-validation for the two best models in each track" vs clustering
**PDF says (§7.2):** "Cross-validation must be used for at least the two
best-performing models **in each track**." §3.3 lists only K-Means and
Agglomerative Hierarchical Clustering.

**The problem:** ordinary k-fold CV needs a supervised score and a `predict` for
held-out points. `AgglomerativeClustering` has no `predict`; it cannot assign an
unseen point without refitting the whole hierarchy. There is no standard
"cross-validated silhouette".

**What we did:** implemented a clearly labelled **resampling-based stability
assessment** (`clu_m.stability_assessment`): cluster overlapping subsamples
independently, measure label agreement on the overlap with the Adjusted Rand
Index. Every output carries
`interpretation_status: "SUPPLEMENTARY - not supervised cross-validation"`.

**We explicitly do NOT claim this satisfies §7.2.**

**Ask:** *Is stability resampling the intended reading, or is something else
expected — e.g. CV only on the supervised tracks?*

---

## IC-4 — Contradictory bonus grading cap
**PDF says (§6):** bonus is "up to +2 marks in Review 2", Review 2 total is
**25 marks**, and then: "Bonus marks are added to the Review 2 score but the
total displayed is **capped at 20** for official grading purposes unless
otherwise communicated."

**The contradiction:** a 25-mark review capped at a displayed 20 means the
bonus is unreachable — indeed 5 ordinary marks are also unreachable. Either
"20" is a typo for "25" (bonus absorbed within the existing total), or the cap
is something else, or Review 2 is rescaled to 20.

**What we did:** built the GUI anyway (it is genuinely useful for the viva) and
did **not** deploy publicly, so only the +1 GUI bonus is in scope. See IC-5.

**Ask:** *What is the actual Review 2 cap, and is the +2 bonus reachable?*

---

## IC-5 — t-SNE: "encouraged" vs required
**PDF says:** §3.3 — "t-SNE visualisation is **encouraged**." But the Review 2
rubric B3 full-marks descriptor says: "PCA-reduced 2D scatter plot with cluster
colour-coding for every algorithm; **t-SNE plot for at least one algorithm**."

**What we did:** treated t-SNE as **required** for full rubric coverage, since
B3 is where the marks are. Implemented for K-Means on a labelled 3,000-row
subsample (t-SNE is O(n²)-ish; the sample size is printed on the figure).

**Ask:** none strictly needed — but confirm the subsample is acceptable, or we
can run the full 10,127 rows at greater cost.

---

## IC-6 — Duplicate rows vs the 10,000-row working preference
Not a PDF ambiguity, but a conflict worth raising. The regression dataset has
**127 exact duplicate rows**. Removing them gives 9,873 rows — below the
10,000-row working preference. See `docs/team_analysis_prompts.md` Q-RG1.
Current run **retains** them and reports both figures.
