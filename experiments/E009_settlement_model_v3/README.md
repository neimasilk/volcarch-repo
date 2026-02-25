# E009: Settlement Suitability Model v3 — SoilGrids Clay + Silt

**Date:** 2026-02-24
**Status:** REVISIT (AUC=0.664, MVR not met; proceed to Target-Group Background)
**Paper:** P2 (Settlement Suitability Model)
**Author:** Amien + Codex
**Builds on:** E008 (best AUC=0.695, BELOW MVR)

## Hypothesis

Adding topsoil composition features from SoilGrids (`clay`, `silt`) should improve settlement
suitability prediction beyond E008 and help push spatial AUC above the MVR threshold (0.75).

## Method

- Downloaded SoilGrids VRT layers:
  - `clay_0-5cm_mean`
  - `silt_0-5cm_mean`
- Reprojected and resampled both layers to match `jatim_dem.tif` grid (EPSG:32749, ~30.66 m).
- Saved aligned rasters:
  - `data/processed/dem/jatim_clay.tif`
  - `data/processed/dem/jatim_silt.tif`
- Trained E009 model with same protocol as E008:
  - Features: elevation, slope, TWI, TRI, aspect, river_dist, clay, silt
  - Pseudo-absences: 5x ratio, 2 km exclusion buffer
  - Validation: Spatial block CV, 5 folds, ~50 km blocks
  - Algorithms: XGBoost + RandomForest
  - Challenge 1 tautology test (suitability vs volcano distance)

Scripts:
- `experiments/E009_settlement_model_v3/00_prepare_soilgrids.py`
- `experiments/E009_settlement_model_v3/01_settlement_model_v3.py`

## Data

- `data/processed/east_java_sites.geojson`
- `data/processed/dem/jatim_dem.tif`
- `data/processed/dem/jatim_slope.tif`
- `data/processed/dem/jatim_twi.tif`
- `data/processed/dem/jatim_tri.tif`
- `data/processed/dem/jatim_aspect.tif`
- `data/processed/dem/jatim_river_dist.tif`
- `data/processed/dem/jatim_clay.tif`
- `data/processed/dem/jatim_silt.tif`

## Results

Sample size after valid-feature filtering:
- Presences: 259
- Pseudo-absences: 1,295

| Model | Spatial AUC | TSS |
|-------|------------|-----|
| XGBoost | 0.664 ± 0.049 | 0.337 ± 0.083 |
| Random Forest | 0.643 ± 0.054 | 0.312 ± 0.072 |

Fold-level AUCs:
- XGBoost: 0.701, 0.657, 0.579, 0.662, 0.722
- Random Forest: 0.704, 0.643, 0.603, 0.566, 0.700

AUC progression:
- E007: 0.659
- E008: 0.695
- E009: 0.664
- Delta vs E007: +0.005
- Delta vs E008: -0.031

Feature importances (XGBoost):
- elevation: 0.165
- silt: 0.156
- river_dist: 0.123
- clay: 0.121
- tri: 0.119
- slope: 0.119
- twi: 0.106
- aspect: 0.092

Challenge 1 (Tautology Test):
- Spearman rho (suitability vs volcano distance): -0.266 (p<0.001)
- High-suitability cells within 50 km of volcano: 57.8%
- Verdict: PASSED (tautology-free)

Artifacts:
- `experiments/E009_settlement_model_v3/results/model_results.txt`
- `experiments/E009_settlement_model_v3/results/model_cv_results.png`
- `experiments/E009_settlement_model_v3/results/suitability_map.html`

## Conclusion

Path A (SoilGrids feature addition) did not meet MVR and reduced performance relative to E008.
The model remains tautology-free, but predictive generalization is still limited under current
random background pseudo-absence strategy.

## Next Steps

Move to Path B:
- Implement Target-Group Background (TGB) pseudo-absences as E010/E009b.
- Use survey-accessibility proxy (road density and/or known survey footprint) for background
  sampling to reduce survey-bias contamination.

