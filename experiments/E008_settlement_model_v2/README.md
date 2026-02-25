# E008: Settlement Suitability Model v2 — River Distance Feature

**Date:** 2026-02-24
**Status:** REVISIT (AUC=0.695, MVR not met; positive trend from E007)
**Paper:** P2 (Settlement Suitability Model)
**Author:** Amien + Claude
**Builds on:** E007 (baseline AUC = 0.659, BELOW MVR)

## Hypothesis

E007 showed AUC=0.659 using only terrain features (elevation, slope, TWI, TRI, aspect).
The dominant missing signal is river proximity — ancient settlements were invariably located
near rivers for water, agriculture, transport, and defense. Adding a direct river distance
raster (from OSM waterway lines) should push AUC above the MVR of 0.75.

## Method

All settings identical to E007 except:
- **Added feature:** `river_dist` — Euclidean distance in metres to nearest named river or canal
  (computed from OSM Overpass API waterway lines via `tools/compute_river_distance.py`)
- Features: elevation, slope, TWI, TRI, aspect, **river_dist** (6 total vs 5 in E007)
- Same: XGBoost (primary) + Random Forest (secondary)
- Same: Spatial block CV (5 folds, ~50km blocks)
- Same: Pseudo-absences (5x ratio, 2km exclusion buffer)

## MVR

- Spatial AUC > 0.75 → Paper 2 GO
- Spatial AUC > 0.65 but < 0.75 → add soil data, retry as E009
- Spatial AUC < 0.65 consistently → kill signal (H3 may be falsified)

## Data

- `data/processed/east_java_sites.geojson` — positive samples (n=391 geocoded)
- `data/processed/dem/jatim_*.tif` — DEM derivatives
- `data/processed/dem/jatim_river_dist.tif` — river proximity raster (computed by this experiment)

## Results

| Model | Spatial AUC | TSS | Delta vs E007 |
|-------|------------|-----|--------------|
| XGBoost | 0.685 ± 0.074 | 0.345 ± 0.135 | +0.026 |
| Random Forest | 0.695 ± 0.107 | 0.379 ± 0.200 | +0.039 |

MVR (AUC > 0.75): NOT MET

Fold-level AUCs (XGBoost): 0.718, 0.620, 0.596, 0.804, 0.686
Fold-level AUCs (RF):      0.701, 0.631, 0.565, 0.885, 0.695

Feature importances (XGBoost):
- elevation: 0.212
- tri: 0.185
- river_dist: 0.168 ← new, ranked 3rd
- slope: 0.159
- twi: 0.152
- aspect: 0.124

Challenge 1 (Tautology Test): PASSED — rho=-0.153 (tautology-free)
High-suitability cells within 50km of volcano: 55.2%

River distance at known sites: mean=2,531m, median=1,355m, max=15,706m

## Conclusion

River distance adds real signal (+0.036 over E007 baseline). River_dist ranks 3rd in importance.
But MVR of 0.75 not reached. Fold 4 is very strong (RF AUC=0.885) — the Malang/Brantas basin
area is well-predicted when trained on similar data. Folds 2–3 remain weak (AUC < 0.65),
suggesting spatial domain shift in coastal / East-tip areas.

Trend: 0.659 (E007) → 0.695 (E008). Positive, not a kill signal.
Next: E009 with soil data (SoilGrids clay content) OR bias-corrected pseudo-absences.
