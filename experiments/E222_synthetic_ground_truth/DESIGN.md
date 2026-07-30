# E222 — Design (pre-registration)

**Written 2026-07-27, before running.** Commissioned by the harsh co-author review (round 2,
`papers/P2_settlement_model/review_package_20260727/07_REVIEW_KERAS_Q1_GO_FRENDI.md`). One purpose:
close the deepest hole in the revision's evidence chain.

## The hole

Every "generalisation" number in E217–E220 is scored against the same 378 survey-biased presences.
A hostile reviewer says: *your honest metric is blind to exactly what target-group background is for.
You proved the reported metric is broken using a ruler that cannot see bias-correction. Show me the
pathology against ground truth, or the claim is local to your broken-ruler comparison.*

There is no ground truth for the real East Java record. So the truth must be built: a synthetic world
where the intensity surface is known, the survey bias is applied deliberately, and the same pipeline
(same design code, same CV, same algorithms) is run end-to-end. If the wrong-direction selection
replicates when judged against truth, the claim is mechanism, not dataset anecdote. If it does not,
the manuscript's strongest new claim is narrowed before a reviewer narrows it for us.

## Synthetic worlds

- **Frame & covariates:** the real East Java lattice (588,535 cells, 6 features + road_dist), so
  environmental structure and spatial autocorrelation are real; only the archaeology is synthetic.
- **Intensity surface A (fully observed):** log λ = −1.0·z(elev) − 0.8·z(slope) − 1.2·z(river_dist)
  + 0.6·z(twi). Lowlands, gentle, near-river, wet — plausible settlement logic.
- **Intensity surface B (misspecified):** A + 0.8·z(clay), where clay (jatim_clay.tif) is a real raster
  that is NOT in the feature set. Tests whether conclusions survive realistic model misspecification.
- **True presences:** N cells sampled without replacement with probability ∝ λ (N tuned so that after
  survey thinning the observed count lands in 250–800, matching the real dataset's power).
- **Survey bias:** observed = true presences thinned with acceptance = clip(exp(−road/12 km), 0.03, 1) —
  the same functional form the TGB design assumes. TGB is thus the *correctly specified* bias model:
  the rationale gets its best possible shot.
- **Held-out truth sample:** an independent N=400 draw from λ (no survey thinning), used only for
  truth-anchored evaluation. 10 independent worlds per surface.

## Pipeline (identical code to the real experiments)

Configs: random, tgb, hybrid at hard_frac ∈ {0.0, 0.3, 0.7, 1.0} — background draws use the E217 base
functions verbatim. Algorithms: MaxEnt, XGBoost, RandomForest (same hyperparameters as E217–E221).
Per (world × surface × config × algo):

- **auc_own** — spatial block CV on observed presences + config background (the number a paper reports);
- **auc_true** — same folds, scored against the held-out truth sample + fixed uniform availability;
- **map recovery** — full-data fit, full-frame prediction: top-decile Jaccard vs λ's top decile,
  full-frame Spearman vs λ, and Spearman restricted to the road-remotest quintile;
- **Boyce** — full-fit predictions at observed presences vs uniform availability.

## Pre-registered predictions

| # | Prediction | If it fails |
|---|---|---|
| P1 | inflation (auc_own − auc_true) rises with hard_frac (Spearman > 0.5 across all runs); R-own (argmax auc_own) picks hybrid(1.0) in ≥ 60% of 180 selection cases | The artefact is contingent on the real dataset's idiosyncrasies; headline narrows to "demonstrated in one archaeological case" |
| P2 | truth-anchored selection cost: median auc_true(oracle) − auc_true(R-own) ≥ 0.02, positive in ≥ 70% of cases | The selection pathology is real but weak under known truth; magnitude language is downgraded |
| P3 (fork) | tgb beats random on map recovery (top-decile Jaccard, road-remote Spearman) in ≥ 60% of cases | If YES: "background design helps the map, not the score" is quantified in truth terms — the paper's balanced message. If NO: the TGB rationale fails even when correctly specified — reported plainly |
| P4 (fork) | Boyce tracks truth: median Spearman(Boyce, map-recovery) across configs ≥ 0.5 | If YES: Boyce earns a principled role ("map quality"), complementing fixed-background AUC ("discrimination"). If NO: Boyce is another broken instrument; removed from the recommendations |
| P5 | Misspecification (surface B) amplifies or shrinks the effects — no committed direction | Descriptive either way |

## Sequencing

One script, ~90 min (2 surfaces × 10 worlds × 6 configs × 3 algos × 6 fits + 360 full-frame predictions).
Maps are scored and discarded, not stored. Results → `results/e222_*.csv`, verdicts →
`results/e222_outcome.json`.
