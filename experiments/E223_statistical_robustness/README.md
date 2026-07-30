# E223 — Statistical Robustness Package

**Status:** SUCCESS | **Date:** 2026-07-27 | **Pre-registration:** `DESIGN.md`
**Commissioned by:** harsh co-author review round 2 (`papers/P2_settlement_model/review_package_20260727/07_REVIEW_KERAS_Q1_GO_FRENDI.md`)
**Script:** `01_robustness.py` (~25 min)

## Hypothesis

Four statistical-rigour holes a Q1 methods reviewer would poke in the E217–E221 package:
(A) "no benefit" was asserted from failed significance, not shown by equivalence; (B) inference treats
Monte Carlo seeds as the replication unit; (C) one MaxEnt configuration; (D) k* = 7 rests on one
arbitrary Jaccard threshold.

## Method & Results

### A — Equivalence, not just nulls
95% t-CI on paired (hybrid − random) AUC per algorithm × evaluation background (20 seeds,
`e218_stageA_raw.csv`), vs the submitted manuscript's ladder gain (+0.092).

**All 12 cells exclude +0.092.** Uniform column: MaxEnt −0.033 (CI −0.039…−0.028), XGB −0.009
(−0.013…−0.004), RF −0.009 (−0.013…−0.005). The MaxEnt CI also excludes 0 — hybrid is significantly
*worse* there. The published ladder is **positively rejected**, not merely unsupported.

### B — Spatial block bootstrap (seeds are Monte Carlo, not data)
30 replicates: resample presence blocks (0.45°) with replacement (multiplicity kept), fit on in-bag
sites + fresh background, score out-of-bag sites + fixed uniform evaluation background. (Replicate 4
skipped, OOB < 50; n = 29.)

| algorithm | mean (hybrid−random) | 95% percentile CI | excludes +0.092 |
|---|---|---|---|
| MaxEnt | −0.019 | −0.059…+0.008 | yes |
| XGBoost | −0.005 | −0.024…+0.025 | yes |
| RandomForest | −0.001 | −0.017…+0.026 | yes |

The conclusion survives resampling of the archaeological record itself. **Honest power statement (must
travel with the claim):** at n = 378 sites the bootstrap CIs are ±0.02–0.06 wide — effects smaller than
~+0.03 cannot be excluded; what is excluded decisively is the published +0.092.

### C — MaxEnt regularisation sensitivity
beta_multiplier ∈ {0.5, 1.0, 1.5, 2.5, 4.0} × 3 designs × 10 seeds, common evaluation background.
hybrid − random = **−0.020…−0.022 at every beta** (1/10 seeds positive throughout). The artefact
conclusion is not a MaxEnt-tuning artefact.

### D — k* threshold sensitivity
From E221's stored curve: k* = **2–5** seeds at J ≥ 0.85, **4–7** at J ≥ 0.90, **7–9** at J ≥ 0.95.
The recommendation is now a range with its sensitivity stated, not a single number.

## Conclusion

Every statistical-rigour attack pre-registered against the package is closed: the published ladder is
rejected under both Monte Carlo and data-level resampling; MaxEnt tuning does not move the conclusion;
the seed-ensemble recommendation carries its threshold sensitivity. Remaining honesty item for the
manuscript: label every p-value with its replication unit (seeds = pipeline noise; bootstrap = data).

## Files

- `results/e223a_equivalence_ci.csv` — 12 cells, CI vs +0.092
- `results/e223b_block_bootstrap.csv`, `e223b_bootstrap_summary.csv`
- `results/e223c_maxent_beta.csv`, `e223c_beta_summary.csv`
- `results/e223d_kstar_thresholds.csv`
- `results/e223_outcome.json`
