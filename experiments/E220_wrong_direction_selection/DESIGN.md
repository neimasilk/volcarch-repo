# E220 — Design (pre-registration)

**Written 2026-07-27, before running.** Requested by the co-author review (Go Frendi) in
`papers/P2_settlement_model/review_package_20260727/05_REVIEW_COAUTHOR_GO_FRENDI.md` §8. Two purposes:
sharpen the novelty claim (R1's "not entirely novel"), and close the two residual holes the co-author
probe found in our own refutation machinery.

---

## Part 1 — Selection on the reported metric walks backwards (the headline)

E218b showed two curves moving in opposite directions. Reviewers will ask the procedural question: **so
what would each selection rule have picked, and what did it cost?** This part turns the dose-response into
a model-selection experiment — the form the "wrong gradient" claim takes when it has to survive review.

**Setup.** hard_frac ∈ {0.0, 0.1, …, 1.0} × **20 seeds** × 3 algorithms (MaxEnt, XGBoost, RandomForest).
Per configuration: `auc_own` (scored on the design's own background — "the number a paper would report"),
`auc_common`, `tss_common`, Boyce (all scored on a fixed uniform evaluation background). One CV pass per
configuration: the model is fit once per fold and scored on both test sets on **identical folds** (a
control improvement over E218b, which used two passes with marginally different fold partitions).

**Selection rules simulated (what a practitioner does):**
- **R-own:** pick argmax `auc_own` — the submitted manuscript's rule.
- **R-common:** pick argmax `auc_common` — the honest rule. Evaluated **cross-fitted**: select hard_frac on
  one half of the seeds (mean metric), score its `auc_common` on the other half, swap, average. In-sample
  argmax is also recorded but labelled optimistic and never headline-quoted.
- **R-boyce:** pick argmax Boyce — tests whether the presence-only metric is an honest selector.

**Pre-registered predictions:**

| # | Prediction | Reading if it fails |
|---|---|---|
| P1 | R-own picks hard_frac ≥ 0.7 in ≥ 60% of the 60 seed×algorithm cases | The reported-metric curve is not monotone enough to drive selection; downgrade "walks backwards" to "walks nowhere" |
| P2 | R-own's pick is worst-or-within-0.01-of-worst on `auc_common` in ≥ 50% of cases | The cost is diffuse, not concentrated at the dial's top; report the shape actually found |
| P3 | Cross-fitted cost = `auc_common`(R-common pick) − `auc_common`(R-own pick) ≥ **+0.05** | The selection cost is real but small; headline becomes the mechanism, not the magnitude |
| P4 (fork) | If R-boyce picks low hard_frac like R-common → Boyce validated as honest selector. If not → report Boyce is noisier than hoped; recommend common-background AUC as the primary selection metric | Both outcomes are publishable; neither is hidden |

Secondary, free of charge: the 20-seed sweep re-estimates E218b's dose-response (Spearman inflation vs
dissimilarity) at the paper's own seed standard, and the E013 sub-grid {0.0, 0.15, 0.30} is compared
against the full dial to quantify what the restricted sweep hid.

## Part 2 — Buffered evaluation background (closes co-author probe #3)

E217b/E218 drew the common evaluation background from the **unbuffered** frame while training backgrounds
came from the buffered frame. Symmetric across designs, but "symmetric" is an assertion until run.

**Setup.** 3 designs × 3 algorithms × 20 seeds; common evaluation background = uniform draw from the
**buffered** frame (> 2 km from any presence). Everything else identical to E218 Stage A.

**Prediction:** hybrid − random ≤ 0 for all three algorithms, within ±0.01 of the Stage A uniform column.
If the ranking changes materially, the artefact conclusion is contingent on eval-background construction
and the manuscript must say so.

## Part 3 — Boyce window sensitivity (closes co-author probe #4)

Our Boyce uses width = range/10, 101 windows. **Setup:** 3 designs × 3 algorithms × 5 seeds, uniform eval;
Boyce recomputed on the same predictions at widths {range/5, range/10, range/20} × windows {51, 101, 201}.

**Prediction:** sign of (hybrid − random) Boyce stable across ≥ 8/9 configurations per algorithm. If not,
Boyce drops from "artefact-immune metric" to "another knob", and §4.2 of the revision argument gets
rewritten accordingly.

## Part 4 — Wilcoxon signed-rank on every headline paired contrast

No new fits. Contrasts: E218 Stage A hybrid−random per algorithm × eval background (20 seeds); E220 sweep
`auc_common` at hard_frac 1.0 vs 0.0 paired per seed; E220 selection cost; E217b feature gain (60 pairs)
and evaluation inflation (15 pairs). Report statistic + p; keep sign counts alongside (they are the more
honest effect summary).

## Sequencing

Parts 1–3 in one script, ~35 min total. Part 4 pure analysis. Results → `results/`, verdicts →
`results/e220_outcome.json`.
