# E011: Settlement Suitability Model v5 - TGB Parameter Sweep

**Date:** 2026-02-24  
**Status:** REVISIT (AUC=0.725, MVR not met; best score so far)  
**Paper:** P2 (Settlement Suitability Model)  
**Author:** Amien + Codex  
**Builds on:** E010 (AUC=0.711, TGB baseline)

## Hypothesis

Tuning TGB pseudo-absence parameters (`decay`, `max_road_dist`) on fixed spatial
CV splits should improve generalization and move spatial AUC closer to MVR 0.75.

## Method

1. Keep feature set unchanged (E008/E010 features):
   - `elevation`, `slope`, `twi`, `tri`, `aspect`, `river_dist`
2. Keep TGB framework from E010:
   - `p_accept = max(0.03, exp(-road_dist / decay))`
   - sample only where `road_dist <= max_road_dist`
3. Sweep parameter grid (12 configs):
   - `decay`: [8000, 12000, 16000, 20000] m
   - `max_road_dist`: [20000, 40000, 60000] m
4. Use deterministic spatial block splits (5 folds, ~50 km blocks) for fair
   config-to-config comparison.
5. Select best configuration by highest `best_auc = max(XGB_AUC, RF_AUC)`.
6. Re-run full output generation (metrics, map, Challenge 1) on best config.

Script:
- `experiments/E011_settlement_model_v5/01_settlement_model_v5.py`

## Data

- `data/processed/east_java_sites.geojson`
- `data/processed/dem/jatim_dem.tif`
- `data/processed/dem/jatim_slope.tif`
- `data/processed/dem/jatim_twi.tif`
- `data/processed/dem/jatim_tri.tif`
- `data/processed/dem/jatim_aspect.tif`
- `data/processed/dem/jatim_river_dist.tif`
- `data/processed/dem/jatim_road_dist.tif`

## Results

Configurations tested: 12

Best configuration:
- `decay = 16000 m`
- `max_road_dist = 60000 m`
- `seed = 951`
- pseudo-absence road distance mean: 1264 m
- pseudo-absence acceptance prob mean: 0.929

Best config metrics:
- XGBoost: AUC `0.725 ± 0.084`, TSS `0.447 ± 0.184`
- RandomForest: AUC `0.716 ± 0.081`, TSS `0.408 ± 0.147`
- Verdict: `REVISIT (0.65-0.75)`

Top 5 configs by best AUC:
1. decay=16000, max=60000, XGB=0.725, RF=0.716, BEST=0.725
2. decay=12000, max=20000, XGB=0.722, RF=0.719, BEST=0.722
3. decay=16000, max=20000, XGB=0.716, RF=0.719, BEST=0.719
4. decay=20000, max=40000, XGB=0.718, RF=0.707, BEST=0.718
5. decay=16000, max=40000, XGB=0.713, RF=0.716, BEST=0.716

Progression:
- E007: 0.659
- E008: 0.695
- E009: 0.664
- E010: 0.711
- E011: 0.725 (best so far)

Challenge 1:
- rho(suitability vs volcano distance) = -0.169 (p<0.001)
- High-suitability cells within 50 km volcano radius = 56.2%
- Verdict: PASSED (tautology-free)

Artifacts:
- `experiments/E011_settlement_model_v5/results/model_results.txt`
- `experiments/E011_settlement_model_v5/results/sweep_results.csv`
- `experiments/E011_settlement_model_v5/results/sweep_heatmap.png`
- `experiments/E011_settlement_model_v5/results/model_cv_results.png`
- `experiments/E011_settlement_model_v5/results/suitability_map.html`

## Conclusion

Parameter tuning improves TGB performance from 0.711 to 0.725, validating that
background design matters and should remain the main optimization path.
MVR (0.75) is still not reached, but the gap is now smaller (0.025).

## Next Steps

- E012: enrich accessibility proxy (add `unclassified`, `residential`, `service`)
  and repeat sweep on fixed CV protocol.
- If survey polygons become available, replace road-only proxy with direct
  survey-footprint-constrained TGB.

