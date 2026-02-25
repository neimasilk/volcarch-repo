# E012: Settlement Suitability Model v6 - TGB Proxy Enrichment

**Date:** 2026-02-24  
**Status:** REVISIT (AUC=0.730, MVR not met; best score so far)  
**Paper:** P2 (Settlement Suitability Model)  
**Author:** Amien + Codex  
**Builds on:** E011 (AUC=0.725)

## Hypothesis

Expanding survey-accessibility proxy from major roads only to broader local-road
coverage (`unclassified`, `residential`, `service`) should improve TGB background
sampling realism and increase spatial AUC.

## Method

1. Built enriched road-distance proxy raster:
   - Output: `data/processed/dem/jatim_road_dist_expanded.tif`
   - OSM classes included:
     `motorway`, `trunk`, `primary`, `secondary`, `tertiary`,
     `unclassified`, `residential`, `service`
2. Kept model feature set unchanged for fair comparison:
   - `elevation`, `slope`, `twi`, `tri`, `aspect`, `river_dist`
3. Reused fixed-split TGB sweep protocol from E011:
   - `decay` in {8km, 12km, 16km, 20km}
   - `max_road_dist` in {20km, 40km, 60km}
   - `min_accept_prob=0.03`
   - 5-fold deterministic spatial block CV
4. Selected best config by highest `best_auc=max(XGB_AUC, RF_AUC)`.
5. Ran full output generation (metrics, map, Challenge 1) for best config.

Scripts:
- `experiments/E012_settlement_model_v6/00_prepare_road_proxy_expanded.py`
- `experiments/E012_settlement_model_v6/01_settlement_model_v6.py`

## Data

- `data/processed/east_java_sites.geojson`
- `data/processed/dem/jatim_dem.tif`
- `data/processed/dem/jatim_slope.tif`
- `data/processed/dem/jatim_twi.tif`
- `data/processed/dem/jatim_tri.tif`
- `data/processed/dem/jatim_aspect.tif`
- `data/processed/dem/jatim_river_dist.tif`
- `data/processed/dem/jatim_road_dist_expanded.tif` (new)

## Results

Configurations tested: 12

Best configuration:
- `decay = 12000 m`
- `max_road_dist = 20000 m`
- `seed = 446`
- pseudo-absence road distance mean: 509 m
- pseudo-absence acceptance prob mean: 0.962

Best config metrics:
- XGBoost: AUC `0.730 ± 0.085`, TSS `0.420 ± 0.170`
- RandomForest: AUC `0.724 ± 0.081`, TSS `0.413 ± 0.152`
- Verdict: `REVISIT (0.65-0.75)`

Top 5 configs by best AUC:
1. decay=12000, max=20000, XGB=0.730, RF=0.724, BEST=0.730
2. decay=12000, max=60000, XGB=0.723, RF=0.716, BEST=0.723
3. decay=16000, max=40000, XGB=0.719, RF=0.710, BEST=0.719
4. decay=8000, max=40000, XGB=0.717, RF=0.713, BEST=0.717
5. decay=16000, max=60000, XGB=0.715, RF=0.712, BEST=0.715

Progression:
- E007: 0.659
- E008: 0.695
- E009: 0.664
- E010: 0.711
- E011: 0.725
- E012: 0.730

Challenge 1:
- rho(suitability vs volcano distance) = -0.160 (p<0.001)
- High-suitability cells within 50 km volcano radius = 55.3%
- Verdict: PASSED (tautology-free)

Artifacts:
- `experiments/E012_settlement_model_v6/results/model_results.txt`
- `experiments/E012_settlement_model_v6/results/sweep_results.csv`
- `experiments/E012_settlement_model_v6/results/sweep_heatmap.png`
- `experiments/E012_settlement_model_v6/results/model_cv_results.png`
- `experiments/E012_settlement_model_v6/results/suitability_map.html`

## Conclusion

Expanded accessibility proxy improves performance slightly over E011
(`+0.005` AUC), and establishes the best model so far (`0.730`), but still
below MVR 0.75.

## Next Steps

- E013: combine expanded-road proxy with stronger survey-bias correction:
  - spatially stratified TGB quotas by subregion, or
  - survey-footprint polygons (if available), or
  - hybrid negative sampling (TGB + environmental dissimilarity constraints).
- Consider calibrating pseudo-absence ratio (e.g., 3x, 4x, 5x, 6x) under fixed CV.

