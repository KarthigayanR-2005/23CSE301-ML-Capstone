# Team Contributions

**Names are intentionally blank** — fill them in yourselves.

Guidelines §7.4: divide by **track**, not by step. Do **not** split it as "one
person does EDA, one does models, one does slides." Each owner must be able to
explain *every line* of their track — and the instructor may ask **any member
about any part** during the viva.

---

## Primary ownership

| Track | Primary owner | Cross-reviewer | Second reviewer |
|---|---|---|---|
| Regression (Review 1) | _______________ | _______________ | _______________ |
| Classification Part A (Review 1) | _______________ | _______________ | _______________ |
| Classification Part B (Review 2) | _______________ | _______________ | _______________ |
| Clustering (Review 2) | _______________ | _______________ | _______________ |

Suggested rotation so every member reviews work they do not own:

| Member | Owns | Reviews |
|---|---|---|
| Member 1: ______ | Regression | Clustering |
| Member 2: ______ | Classification (A + B) | Regression |
| Member 3: ______ | Clustering | Classification |

## Shared responsibilities

| Item | Owner | Status |
|---|---|---|
| README problem statements (final wording) | _______ | ⬜ draft only |
| Feature engineering — regression (Q-RG3) | _______ | 🔴 not started |
| Feature engineering — classification (Q-CF1) | _______ | 🔴 not started |
| Clustering feature justification (Q-CL1) | _______ | 🔴 provisional |
| Choice of k (Q-CL2) | _______ | 🔴 placeholder in use |
| All notebook "TEAM TO COMPLETE" cells | all three | 🔴 unwritten |
| Instructor clarifications (IC-1 … IC-5) | _______ | ⬜ to raise |
| Review 1 slides | _______ | ⬜ |
| Review 2 slides | _______ | ⬜ |
| Streamlit app demo | _______ | ✅ built, local only |

## What "ownership" means for the viva

Rubric D2 (Review 2) is worth **5 of 25 marks** and reads: *"All team members
demonstrate understanding; questions on algorithm choice, metric interpretation,
and trade-offs are answered."*

Before each review, every member should be able to, for **all three** tracks:

1. Explain the dataset, its target, and why columns were excluded.
2. Explain why *that* algorithm suits *that* problem.
3. Interpret every metric in the tables — in plain words, not formulas.
4. Justify preprocessing, and explain what data leakage is and how the pipelines prevent it.
5. Explain the trade-offs behind the final model choice.
6. Say honestly what the project does **not** establish.

**Rehearsal method:** each member is quizzed on the track they do **not** own.

## Commit hygiene

Rubric C3 (Review 2): *"Repository has a meaningful commit history (not a single
bulk upload)."* A single bulk commit scores **zero** on that criterion. Commit at
each real milestone, with a message saying what changed.
