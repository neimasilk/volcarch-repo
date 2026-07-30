# E222 — Synthetic Ground-Truth Validation

**Status:** SUCCESS (dengan satu koreksi klaim yang wajib dibaca — lihat §Conclusion) | **Date:** 2026-07-27
**Pre-registration:** `DESIGN.md` | **Commissioned by:** harsh review round 2, M1/M5/M8
(`papers/P2_settlement_model/review_package_20260727/07_REVIEW_KERAS_Q1_GO_FRENDI.md`)
**Scripts:** `01_synthetic_truth.py` (worlds A/B, ~35 min), `02_world_c_regional_bias.py` (~15 min),
`03_world_d_balanced_truth.py` (~15 min)

## Hypothesis

Every "generalisation" number in E217–E220 is scored against the same 378 survey-biased presences — a
ruler that cannot see the bias target-group backgrounds are meant to correct. The harsh-review demand:
demonstrate the selection pathology against **ground truth**, or it is local to a broken-ruler
comparison. No ground truth exists for the real record, so truth is built: synthetic worlds on the real
East Java lattice, known intensity, deliberately applied survey bias, identical pipeline code.

## Method

- **Worlds:** A — intensity fully observed (elevation/slope/river/TWI); B — misspecified (A + clay, a
  real raster withheld from features); C — A + regionally uneven survey effort [1.0, 0.4, 0.15, 0.05];
  D — region-BALANCED truth + regional survey effort (the quota's friendliest regime). Survey bias
  (road-decay, TGB-shaped) applied in all; 10 worlds each; observed n ≈ 300–500 (real dataset's power).
- **Configs:** random, tgb, hybrid at hard_frac ∈ {0.0, 0.3, 0.7, 1.0}; algorithms MaxEnt/XGBoost/RF
  (E217 hyperparameters); E217 draw functions verbatim.
- **Truth scoring:** `auc_true` = AUC against an independent unbiased presence sample + fixed uniform
  availability; map recovery = top-decile Jaccard and Spearman vs the intensity surface; plus `auc_own`
  (the number a paper reports) and Boyce.

## Results

### The selection pathology replicates against ground truth (worlds A/B)

| | auc_own ("reported") | auc_true | inflation |
|---|---|---|---|
| random | 0.847 | **0.828** | +0.019 |
| tgb | 0.840 | **0.827** | +0.012 |
| hybrid(0.0) | 0.737 | 0.541 | +0.196 |
| hybrid(1.0) | **0.890** | 0.617 | +0.273 |

R-own (argmax auc_own) picks hybrid(1.0) in **60/60** selection cases; truth says random is better by
**median +0.194 AUC** (100% of cases positive; map-recovery cost 0.35–0.53 Jaccard). Misspecification
(surface B) changes nothing qualitatively.

### What had to be corrected in our own claim (pre-registered P1 failed as written)

- Pooled Spearman(hard_frac, inflation) = +0.44 < the pre-registered 0.5. Per-run it is unanimous
  (median +1.000, 100% > 0.5; paired inflation(1.0)−inflation(0.0) = +0.077, p = 1.6e-11) — the pooled
  statistic was diluted by between-world level shifts on a 4-point dial.
- Deeper: **synthetic auc_true RISES with hard_frac** (0.54→0.62) while real-data auc_common FALLS
  (0.699→0.602). The truth slope's SIGN is regime-contingent. What is structural in every world: the
  reported number is always inflated, and the dial moves it ~10× faster than truth, in either direction.

### Mechanistic insight: quota contamination

In concentrated worlds the hybrid's regional quota draws negatives INTO the presence cluster, injecting
false negatives — explaining why hybrid truth-scores are far lower synthetically than in the real
(diffuse-record) data, and naming the danger regime for the real case (concentrated record + quota).

### P3 — TGB does not beat random on truth (fork resolved: NO)

Even when TGB is the correctly specified bias model: Δmap-Jaccard −0.010 (47% positive), Δauc_true
≈ 0.000. No algorithm shows a convincing recovery benefit. **Scope limit:** one bias shape, n ≈ 500 —
not a universal refutation of Phillips et al. (2009); possibly power-limited.

### P4 — Boyce against truth (fork resolved: qualified YES, demoted anyway)

Boyce ranks config families correctly (random/tgb above hybrids) and punishes the dial's extreme, but
its optimum is not truth-calibrated (per-run agreement median +0.50/+0.54, borderline). Role in the
protocol: directional sanity check, not a selector.

### World C — the quota fails even under regional survey bias

quota(0.0) − random: auc_true **−0.246 (0/30 positive)**, map-Jaccard **−0.469 (0/30)**. Region-0 share
of observed presences 43% (survey-driven) — and the quota fails anyway, via the same contamination
mechanism (truth is concentrated, so quota draws false negatives into it).

### World D — region-balanced truth, the quota's friendliest regime (fork resolved: NO)

quota(0.0) − random: auc_true **−0.203 (0/30 positive)**, map-Jaccard **−0.283 (0/30)**. With truth
rebalanced so every quadrant carries equal intensity, the observed record concentrates in region 0
(63%) through survey effort alone — exactly the bias the quota was designed to correct — and it still
loses decisively. The mechanism is the same in every world: matching the background to the observed
record's distribution concentrates negatives where presences concentrate; whenever the record
clusters (survey or truth), that injects false negatives exactly where the model most needs to learn.
**Across four synthetic regimes, no background design ever beats uniform on truth — while the reported
AUC always prefers the most extreme design.**

## Conclusion

The manuscript's strongest claim survives ground truth in a sharpened form: **own-background evaluation
is always inflated; the design dial moves the reported number an order of magnitude faster than it moves
truth (in either direction); therefore model selection on the reported number is unsound in principle —
not merely unlucky in our dataset.** The earlier "hard negatives degrade generalisation" phrasing must
be narrowed: that slope's sign is regime-dependent; the inflation and the selection failure are not.

## Files

- `results/e222_runs.csv`, `e222_selection.csv` — worlds A/B (2×10 worlds × 6 configs × 3 algos)
- `results/e222c_runs.csv`, `e222c_outcome.json` — world C
- `results/e222d_runs.csv`, `e222d_outcome.json` — world D
- `results/e222_outcome.json` — pre-registered verdicts (worlds A/B)
