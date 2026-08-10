# SIG G1 - blind re-derivation of the P2 v0.2 headline numbers

**Run:** 20260810 · **Script:** `revision_ammo/verify_headline_numbers.py` · **Checks:** 64 · **Mismatches:** 4

Every value in the *derived* column was recomputed from the per-run result files
(`*_runs.csv`, `*_raw*.csv`, `*_sweep.csv`, per-cell CSVs). The `*_outcome.json`
summaries written by the experiment scripts were **not** read, so this is an
independent check of them and not a restatement.

| # | Claim | Source | Claimed | Re-derived | Verdict |
|---|---|---|---|---|---|
| 1 | K3 inflation is systematic, not universal | `e222_runs.csv (auc_own - auc_true)` | 343/360 = 95.3% | 343/360 = 95.3% | OK |
| 2 | K3 minimum inflation | `e222_runs.csv` | -0.031 | -0.0312 | OK |
| 3 | K3 median inflation | `e222_runs.csv` | +0.187 | +0.1874 | OK |
| 4 | No design exceeds uniform on truth by a meaningful margin (aggregate) | `e222_runs.csv` | -0.0004 AUC; TGB>random 27/60 | -0.0004 AUC; TGB>random 27/60 | OK |
| 5 | K2 dial 0.0->1.0, reported AUC | `e222_runs.csv` | +0.1538 | +0.1538 | OK |
| 6 | K2 dial 0.0->1.0, true AUC | `e222_runs.csv` | +0.0764 | +0.0764 | OK |
| 7 | K2 ratio (synthetic, pooled) | `derived` | 2.01x | 2.01x | OK |
| 8 | K2 per-run change, reported AUC (median) | `e222_runs.csv paired` | +0.1535 (doc 09, estimator unstated) | endpoint diff +0.1543 \| OLS slope +0.1535 | OK |
| 9 | K2 per-run change, true AUC (median) | `e222_runs.csv paired` | +0.0726 (doc 09, estimator unstated) | endpoint diff +0.0768 \| OLS slope +0.0726 | OK |
| 10 | K2 per-run ratio | `derived` | 2.12x | endpoint 2.01x \| OLS 2.12x \| median of per-run ratios 2.00x | OK |
| 11 | K1 full grid: rule picks hybrid(1.0) | `e222_runs.csv re-selection` | 60/60 | 60/60 | OK |
| 12 | K1 full grid: median truth cost | `derived` | +0.1937 | +0.1937 | OK |
| 13 | K1 full grid: fraction of positive cost | `derived` | 100% | 100.0% | OK |
| 14 | K5 'the rule picks the WORST configuration in 100% of cases' (doc 08 sec 3) | `e222_runs.csv re-selection` | 60/60 | 0/60 - the truth-worst design is {'hybrid(0.0)': 50, 'hybrid(0.3)': 8, 'hybrid(0.7)': 2}, while the rule picks hybrid(1.0) | **MISMATCH** |
| 15 | K1 paper grid (<=0.30): what the rule picks | `re-selection` | random 50, tgb 10 | {'random': 50, 'tgb': 10} | OK |
| 16 | K1 paper grid: what the truth prefers | `re-selection` | random 33, tgb 27 | {'random': 33, 'tgb': 27} | OK |
| 17 | K1 paper grid: median truth cost | `derived` | +0.0000 | +0.0000 | OK |
| 18 | K1 paper grid: 'wrong selection 0/60' | `derived` | 0/60 wrong (doc 09 wording) | pick != truth-best in 29/60; picked the worst config in 0/60; mean cost +0.0012, max +0.0088 | OK |
| 19 | No interior optimum: reported AUC rises to the end of the dial | `e222_runs.csv` | monotone increasing in hard_frac | 0.7367 -> 0.7820 -> 0.8412 -> 0.8904 | OK |
| 20 | P1 inflation vs hard_frac (pooled Spearman) | `e222_runs.csv` | 0.4395 (P1 FAILED) | 0.4395 | OK |
| 21 | P3 TGB - random map Jaccard (mean) | `e222_runs.csv` | -0.010 | -0.0100 | OK |
| 22 | P3 fraction positive | `derived` | 46.67% | 46.67% | OK |
| 23 | m-b world A: mean map Jaccard tgb vs random | `e222_runs.csv` | 0.6898 vs 0.7145 | 0.6898 vs 0.7145 | OK |
| 24 | m-b world B: mean map Jaccard tgb vs random | `e222_runs.csv` | 0.4504 vs 0.4458 | 0.4504 vs 0.4458 | OK |
| 25 | World C: quota vs random, true AUC | `e222c_runs.csv` | -0.2457 | -0.2457 | OK |
| 26 | World C: quota vs random, map Jaccard | `e222c_runs.csv` | -0.4688 | -0.4688 | OK |
| 27 | World C: quota beats random in | `derived` | 0/30 | 0/30 (AUC), 0/30 (Jaccard) | OK |
| 28 | m-d World C: TGB vs random, true AUC | `e222c_runs.csv` | -0.0010 | -0.0010 (56.7% positive) | OK |
| 29 | World D: quota vs random, true AUC | `e222d_runs.csv` | -0.2027 | -0.2027 | OK |
| 30 | World D: quota vs random, map Jaccard | `e222d_runs.csv` | -0.2826 | -0.2826 | OK |
| 31 | World D: quota beats random in | `derived` | 0/30 | 0/30 (AUC), 0/30 (Jaccard) | OK |
| 32 | m-d World D: TGB vs random, true AUC | `e222d_runs.csv` | +0.0022 | +0.0022 (73.3% positive) | OK |
| 33 | K2 real data, dial 0.0->1.0, reported AUC | `e218b_hardfrac_sweep.csv` | +0.1227 | +0.1227 | OK |
| 34 | K2 real data, dial 0.0->1.0, common-background AUC | `e218b_hardfrac_sweep.csv` | -0.0973 | -0.0973 | OK |
| 35 | K2 real data ratio \|reported\| / \|truth\| | `derived` | 1.26x | 1.26x | OK |
| 36 | K1 real data, paper grid (<=0.30): selected hard_frac | `e218b sweep re-selection` | 0.3 | 0.3 | OK |
| 37 | K1 real data, paper grid: cost in common-background AUC | `derived` | +0.0044 | +0.0044 | OK |
| 38 | K1 real data, full dial: selected hard_frac | `derived` | 1.0 | 1.0 | OK |
| 39 | K1 real data, full dial: cost in common-background AUC | `derived` | +0.0973 | +0.0973 | OK |
| 40 | K6 'the reported criterion rises monotonically to the end of the dial' (real data) | `e218b sweep` | monotone increasing | 0.7208 -> 0.7137 -> 0.7253 -> 0.7384 -> 0.7457 -> 0.7602 -> 0.7627 -> 0.7832 -> 0.8057 -> 0.8271 -> 0.8435 \| dips at hard_frac [np.float64(0.0)] (-0.0071) | **MISMATCH** |
| 41 | Hybrid beats random ONLY when evaluated on hybrid's own background | `e218_stageA_raw.csv` | {uniform:0, tgb:0, hybrid:3, stratified:0} | {'hybrid': 3, 'stratified': 0, 'tgb': 0, 'uniform': 0} | OK |
| 42 | Real-data inflation of the hybrid design (per seed x algorithm) | `e218_stageA_raw.csv` | +0.041 ... +0.051, 15/15 positive | +0.0046 ... +0.0838, 60/60 positive (mean +0.0370) | OK |
| 43 | E013 hybrid design on common (uniform) background (XGBoost) | `e218_stageA_raw.csv` | 0.706 | 0.706 | OK |
| 44 | TGB home-court inflation (own tgb minus uniform) | `e218_stageA_raw.csv` | -0.0054 mean, 22/60 positive | -0.0054 mean, 22/60 positive | OK |
| 45 | Background redesign on a COMMON evaluation background [terrain] | `e217b_raw_results.csv` | (not the headline cell) | +0.0054 | OK |
| 46 | Background redesign on a COMMON evaluation background [terrain_river] | `e217b_raw_results.csv` | -0.0142 (mean) | -0.0142 | OK |
| 47 | The same redesign scored on its OWN background [terrain_river] | `e217b_raw_results.csv` | +0.0145 ... +0.0431 by algorithm | +0.0316 (mean); maxent +0.0145, xgboost +0.0431, randomforest +0.0373 | OK |
| 48 | Adding the river feature (terrain -> terrain_river), common background | `e217b_raw_results.csv` | +0.0424 | +0.0424 (12/12 positive) | OK |
| 49 | E223-A: every cell excludes the published +0.092 ladder | `e223a_equivalence_ci.csv` | 12/12 | 12/12 | OK |
| 50 | E223-A: cells whose CI excludes ZERO from above (positive) | `e223a_equivalence_ci.csv` | 3 cells, +0.007...+0.016 | 3 cells: maxent/hybrid +0.0066, randomforest/hybrid +0.0098, xgboost/hybrid +0.0155 | OK |
| 51 | E223-A: MaxEnt on the uniform evaluation background | `e223a_equivalence_ci.csv` | -0.0389 ... -0.0279 | -0.0389 ... -0.0279 | OK |
| 52 | E223-B: block bootstrap, n replicates | `e223b_bootstrap_summary.csv` | 29 | [np.int64(29)] | OK |
| 53 | E223-B: upper bounds of the bootstrap CIs | `e223b_bootstrap_summary.csv` | +0.0082 / +0.0253 / +0.0256 | +0.0082 / +0.0253 / +0.0256 | OK |
| 54 | E223-C: MaxEnt regularisation beta 0.5-4.0 changes nothing | `e223c_beta_summary.csv` | -0.0198 ... -0.0217, 1/10 positive | -0.0198 ... -0.0217, frac positive [np.float64(0.1)] | OK |
| 55 | E223-D: seeds needed for Jaccard >= 0.85 | `e223d_kstar_thresholds.csv` | 2-5 | 2-5 | OK |
| 56 | E223-D: seeds needed for Jaccard >= 0.9 | `e223d_kstar_thresholds.csv` | 4-7 | 4-7 | OK |
| 57 | E223-D: seeds needed for Jaccard >= 0.95 | `e223d_kstar_thresholds.csv` | 7-9 | 7-9 | OK |
| 58 | Top-decile turnover from seed alone (1 - Jaccard) | `e221_turnover_pairs.csv` | 28-47% | 28.1%-47.4% | OK |
| 59 | Field product: site density, robust core vs contingent fringe | `e221_priority_sets.csv` | 2-5.6x (doc 08 sec 3) | randomforest 1.93x, xgboost 4.34x, maxent 5.62x | **MISMATCH** |
| 60 | Field product: absolute densities (sites per 1000 km2) | `e221_priority_sets.csv` | 40.8/9.4, 30.7/15.9, 31.7/5.7 | maxent: robust 31.7 vs contingent 5.7; randomforest: robust 30.7 vs contingent 15.9; xgboost: robust 40.8 vs contingent 9.4 | OK |
| 61 | INT-1: Test 1 volcano-distance correlation, legacy 7-volcano inventory | `e219_outcome.json (no per-run file exists)` | -0.163 (published, submission_jcaa_v0.1.tex l.319) | -0.2435 | **MISMATCH** |
| 62 | INT-1: same correlation on the canonical inventory | `e219_outcome.json` | -0.281, 13 centres in bounds | -0.2811, 13 centres | OK |
| 63 | R2-F matched control: mean suitability, volcanic vs non-volcanic uplands | `e219_terrain_matched.csv` | 0.2249 vs 0.1702 (+0.055) | 0.2249 vs 0.1702 (+0.0547), 90 strata | OK |
| 64 | R2-F matched control: site density (sites per km2) | `e219_terrain_matched.csv` | 0.01377 vs 0.00048 | 0.01377 vs 0.00048; sites 145 vs 2 | OK |

## Notes

- **No design exceeds uniform on truth by a meaningful margin (aggregate)** - G9: holds only as an aggregate mean; per-run TGB wins ~45%
- **K2 per-run ratio** - no estimator in the 2.0-2.1 band reaches 2.12x on the endpoint definition; state the estimator in the manuscript and quote ~2x, not 2.12x
- **K5 'the rule picks the WORST configuration in 100% of cases' (doc 08 sec 3)** - FALSE AS WORDED. The rule picks the design that costs +0.194 against the BEST design; hybrid(1.0) is not the worst - hybrid(0.0) is. Say 'costs +0.194 against the best available design', never 'picks the worst'.
- **K1 paper grid: 'wrong selection 0/60'** - '0/60 wrong' is only true under 'picked the WORST config'. The rule still misses the truth-best design in most runs; the cost of doing so is ~0.
- **K2 real data ratio |reported| / |truth|** - the two move in OPPOSITE directions, which is the stronger statement
- **K6 'the reported criterion rises monotonically to the end of the dial' (real data)** - NOT strictly monotone on the real data: one dip between 0.0 and 0.1. It IS monotone from 0.1 upward, and the maximum is at the end of the dial in both worlds. Say 'the criterion has no interior optimum: its maximum lies at the edge of whatever grid is swept' - which is the claim that matters and is true in both.
- **Real-data inflation of the hybrid design (per seed x algorithm)** - range quoted in doc 08 was per-algorithm means, not per-run
- **E013 hybrid design on common (uniform) background (XGBoost)** - level claim survives but the margin vs DKNS shrinks from +0.105 to ~+0.06; +0.105 = 0.751 seed-avg - 0.646, NOT +0.122 which used the 0.768 best run
- **TGB home-court inflation (own tgb minus uniform)** - G9: the 60/60 inflation is hybrid-specific; TGB has no home-court gain
- **Background redesign on a COMMON evaluation background [terrain]** - reported for completeness; the manuscript's model is terrain_river
- **The same redesign scored on its OWN background [terrain_river]** - the sign reversal between these two rows IS the paper's finding
- **E223-A: cells whose CI excludes ZERO from above (positive)** - m-c: all three are the hybrid evaluation column - the artefact's signature
- **E223-B: upper bounds of the bootstrap CIs** - detectable effect floor is ~+0.03 at n=378 - a declared limit, not a result
- **Field product: site density, robust core vs contingent fringe** - the low end is 1.93x (randomforest), not 2x. Quote '1.9-5.6x' or give the three values; rounding 1.93 up to 2 is the same class of error as K3.
- **INT-1: Test 1 volcano-distance correlation, legacy 7-volcano inventory** - The E219 re-run does NOT reproduce the published -0.163 even on the same 7-volcano inventory: it gives -0.2435 (5-seed mean). The published value came from a single model instance. This is the seed-instability of D1/D2 showing up inside the manuscript's own tautology diagnostic - disclose it, and quote the ensemble value.
- **INT-1: same correlation on the canonical inventory** - verdict unchanged: |rho| < 0.5, so Test 1 still passes - the number moved, the conclusion did not
- **R2-F matched control: site density (sites per km2)** - the non-volcanic arm rests on n=2 sites - state it as consistency, never validation
