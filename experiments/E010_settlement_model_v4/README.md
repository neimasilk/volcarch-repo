# E010: Settlement Suitability Model v4 - Target-Group Background

**Date:** 2026-02-24  
**Status:** REVISIT (AUC=0.711, MVR not met; promising improvement)  
**Paper:** P2 (Settlement Suitability Model)  
**Author:** Amien + Codex  
**Builds on:** E008 (AUC=0.695) and E009 (AUC=0.664)

## Hypothesis

If pseudo-absences are sampled from survey-accessible background (Target-Group
Background, TGB) instead of uniform random background, spatial CV performance should
improve by reducing survey-bias contamination.

## Method

1. Created survey-accessibility proxy raster from OSM major roads:
   - Source: Overpass API (`highway=motorway|trunk|primary|secondary|tertiary`)
   - Output: `data/processed/dem/jatim_road_dist.tif`
2. Kept E008 feature set unchanged to isolate pseudo-absence strategy effect:
   - `elevation`, `slope`, `twi`, `tri`, `aspect`, `river_dist`
3. Replaced random pseudo-absence sampling with TGB sampling:
   - Exclude points within 2 km of known sites
   - Restrict candidate background to `road_dist <= 40 km`
   - Acceptance probability: `p = max(0.03, exp(-road_dist / 12000))`
4. Validation and models unchanged:
   - Spatial block CV: 5 folds, ~50 km blocks
   - Models: XGBoost + RandomForest
5. Challenge 1 tautology test retained.

## Data

- `data/processed/east_java_sites.geojson`
- `data/processed/dem/jatim_dem.tif`
- `data/processed/dem/jatim_slope.tif`
- `data/processed/dem/jatim_twi.tif`
- `data/processed/dem/jatim_tri.tif`
- `data/processed/dem/jatim_aspect.tif`
- `data/processed/dem/jatim_river_dist.tif`
- `data/processed/dem/jatim_road_dist.tif` (new for TGB)

## Results

Sample sizes:
- Presences: 378
- Pseudo-absences (TGB): 1,890

TGB diagnostics:
- Site road distance: mean=796 m, median=210 m
- TGB pseudo-absence road distance: mean=1,198 m, median=674 m

| Model | Spatial AUC | TSS |
|-------|------------|-----|
| XGBoost | 0.711 ± 0.085 | 0.384 ± 0.150 |
| RandomForest | 0.699 ± 0.081 | 0.380 ± 0.130 |

Fold-level AUCs:
- XGBoost: 0.769, 0.779, 0.602, 0.613, 0.792
- RandomForest: 0.787, 0.732, 0.572, 0.640, 0.766

AUC progression:
- E007 (random PA): 0.659
- E008 (random PA): 0.695
- E009 (random PA + soil): 0.664
- E010 (TGB PA): 0.711

Delta:
- vs E007: +0.052
- vs E008: +0.016
- vs E009: +0.047

Challenge 1:
- Spearman rho(suitability, volcano distance) = -0.142 (p<0.001)
- High-suitability cells within 50 km volcano radius = 54.7%
- Verdict: PASSED (tautology-free)

Artifacts:
- `experiments/E010_settlement_model_v4/results/model_results.txt`
- `experiments/E010_settlement_model_v4/results/model_cv_results.png`
- `experiments/E010_settlement_model_v4/results/suitability_map.html`

## Conclusion

TGB pseudo-absence sampling improves spatial AUC over E008 and confirms survey-bias
correction is directionally useful. However, MVR (AUC > 0.75) is still not met.
Weak folds remain in the eastern/coastal transfer zones.

## Next Steps

- Tune TGB weighting and accessibility proxy:
  - Add more road classes (`unclassified`, `residential`, `service`) and compare.
  - Sweep `decay` and `max_road_dist` parameters.
  - Optionally combine road proxy with known survey footprint polygons if available.
- Run as next experiment (E011) with parameter grid + fixed CV split for fair comparison.

