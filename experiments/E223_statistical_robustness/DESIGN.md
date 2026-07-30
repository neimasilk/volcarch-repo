# E223 — Design (pre-registration)

**Written 2026-07-27, before running.** Commissioned by the harsh co-author review (round 2).
Purpose: close the statistical-rigour holes a Q1 methods reviewer will poke in the E217–E221 package.
Four independent components, one script.

## Part A — Equivalence, not just nulls

"Hybrid shows no reliable benefit" is currently supported by failing to find a benefit. A Q1 reviewer
wants the published effect *rejected*: the 95% CI on (hybrid − random, common evaluation, 20 seeds, per
algorithm) must exclude the submitted manuscript's ladder gain (+0.092 AUC, 0.659→0.751). Pure analysis
of `e218_stageA_raw.csv`. Report: mean, 95% CI (t), and the statement "any benefit larger than U is
rejected at 95%".

## Part B — Block bootstrap: seeds are Monte Carlo, not data

Current inference treats 20 seeds as the replication unit — they measure pipeline noise, not sampling
error of the archaeological record itself. Fix: **spatial block bootstrap**. Resample the presence blocks
(0.45°) with replacement, B = 30 replicates × 3 designs × 3 algorithms; fit on in-bag sites + fresh
background draw; score once on out-of-bag sites + a fixed uniform evaluation background drawn per
replicate (single held-out evaluation, no nested CV). Report the bootstrap distribution and 95%
percentile CI of (hybrid − random), and whether it excludes +0.092.

**Prediction:** bootstrap CI upper bound < +0.02 — the published ladder is rejected under resampling of
the data itself, not only of the RNG.

## Part C — MaxEnt regularisation sensitivity

The benchmark used one MaxEnt configuration (hinge+linear+product, beta_multiplier 1.5). Sweep
beta_multiplier ∈ {0.5, 1.0, 1.5, 2.5, 4.0} × 3 designs × 10 seeds on the common evaluation background.

**Prediction:** hybrid − random ≤ 0 at every beta — the artefact conclusion is not a MaxEnt-tuning
artefact.

## Part D — k* threshold sensitivity (analysis only)

E221 recommended ≥ 7 seeds from the J ≥ 0.9 criterion. Recompute k* at J ≥ 0.85 and J ≥ 0.95 from
`e221_stabilisation_curve.csv` so the recommendation can be stated as a range with its sensitivity,
not a single arbitrary threshold.

## Sequencing

A+D pure analysis; B ≈ 10 min (270 fits); C ≈ 15 min (150 fits). Results → `results/e223_*.csv`,
verdicts → `results/e223_outcome.json`.
