# E220 — Selection on the Reported Metric Walks Backwards

**Status:** SUCCESS (P1–P3 confirmed beyond thresholds; P4 resolved to its pre-registered "noisy Boyce"
branch) | **Date:** 2026-07-27 | **Pre-registration:** `DESIGN.md` (written before running)
**Commissioned by:** co-author review `papers/P2_settlement_model/review_package_20260727/05_REVIEW_COAUTHOR_GO_FRENDI.md` §8
**Script:** `01_selection_and_robustness.py` (~25 min). Reuses the E217 base and E218 machinery.

## Hypothesis

E218b showed the reported metric and generalisation move in opposite directions along the `hard_frac`
dial. The reviewer-facing form of that claim is procedural: *the selection rule practitioners actually use
(pick the configuration with the best reported score) systematically chooses the worst-generalising
configurations.* Tested directly, plus two robustness closures on our own refutation machinery.

## Method

- **Part 1:** `hard_frac` ∈ {0.0,…,1.0} × 20 seeds × 3 algorithms. One CV pass per configuration: fit
  once per spatial-block fold, score both the design's own background and a fixed uniform evaluation
  background on **identical folds** (control improvement over E218b's two-pass design). Selection rules
  simulated per (seed, algorithm): R-own = argmax own-background AUC (the submitted manuscript's rule);
  R-common = argmax common-background AUC (honest rule, evaluated cross-fitted across seed halves);
  R-boyce = argmax continuous Boyce.
- **Part 2:** common evaluation background drawn from the **buffered** frame (closes the probe that
  E217b/E218 evaluation negatives came from the unbuffered frame).
- **Part 3:** Boyce recomputed at 3 window widths × 3 window counts on identical predictions.
- **Part 4:** Wilcoxon signed-rank on every headline paired contrast (no new fits).

## Results

**Dose-response re-confirmed at 20 seeds** (E218b used 5): inflation Spearman **+0.967**, common-AUC
Spearman **−0.689**. Own-background AUC 0.707→0.842; common 0.695→0.601.

**The selection verdicts (pre-registered):**

| # | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | R-own picks hard_frac ≥ 0.7 in ≥ 60% of 60 cases | **100%** (56× picks 1.0, 4× picks 0.9) | SUPPORTED, maximal |
| P2 | R-own's pick (near-)worst on common AUC in ≥ 50% | **93%** | SUPPORTED |
| P3 | Cross-fitted cost ≥ +0.05 AUC | **+0.094** (MaxEnt +0.111, XGB +0.098, RF +0.072) | SUPPORTED |
| P4 | Fork: does Boyce track the honest rule? | **No** — median pick 0.4 vs 0.15 | "Noisy Boyce" branch |

In absolute terms: a model tuned by the reported metric generalises at **0.55–0.63**, one tuned on the
fixed background at **0.66–0.72**. E013's restricted grid {0, 0.15, 0.30} showed +0.023 own-AUC gain; the
full dial would have shown +0.135 — the restricted sweep hid the slope it was climbing.

**P4 in detail (it is a finding, not a footnote).** Boyce rises 0.53 → ~0.58 across hard_frac 0.0→0.2–0.6
(per-algorithm optima: MaxEnt 0.2–0.4, XGB 0.5–0.6, RF flat), then collapses to 0.17 at 1.0. So:
(a) Boyce cannot serve as the sole selection metric either; (b) there is a narrow regime where moderate
hard negatives slightly improve presence-vs-availability ranking — the *only* partial rehabilitation of
the original intuition available, and it is small (Δ ≈ +0.05 Boyce) and does not rescue the AUC ladder;
(c) the corrected protocol must **declare its evaluation availability and selection rule explicitly**
rather than trust any single metric.

**Part 2 (buffered eval):** hybrid − random = −0.029 / −0.004 / −0.008 (MaxEnt/XGB/RF) — ranking
unchanged vs Stage A. Probe closed.
**Part 3 (Boyce windows):** sign of hybrid − random stable across all 9 configurations, all algorithms.
Probe closed.
**Part 4 (Wilcoxon):** all headline contrasts significant — e.g. selection cost +0.098 (p = 1.6e-11),
common-AUC drop hard_frac 1.0 vs 0.0 = −0.07…−0.11 (p = 1.9e-6), E013's pick (0.3 vs 0.0) = −0.002…−0.006
(p = 0.07–0.29, i.e. nothing real gained), feature gain +0.042 (p = 1.6e-11), inflation +0.046
(p = 6.1e-5).

## Conclusion

The manuscript's selection rule does not merely fail to find improvement — it is a **maximally wrong
selector** in this design space: it chooses the extreme of the dial every time, at a cross-fitted
generalisation cost of ≈ 0.09 AUC. This is the concrete, quantified form of the novelty claim answering
Reviewer 1's "not entirely novel": not "AUC is not comparable across backgrounds" (Lobo 2008) but
"optimising the reported comparison selects measurably worse models, and here is the dose-response and
the selection cost". The companion protocol message: no single metric is trustworthy — declare the
evaluation availability, fix it, and state the selection rule.

## Files

- `results/e220_sweep.csv` — 20 seeds × 11 settings × 3 algorithms × (auc_own, auc_common, tss, boyce)
- `results/e220_selection_by_seed.csv` — per-seed picks of the three rules
- `results/e220_selection_crossfitted.csv` — honest cost estimates
- `results/e220_buffered_eval.csv`, `results/e220_boyce_windows.csv`, `results/e220_wilcoxon.csv`
- `results/e220_outcome.json` — pre-registered verdicts
