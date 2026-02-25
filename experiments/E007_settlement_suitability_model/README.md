# E007: Settlement Suitability Model — Paper 2 MVP

**Date:** 2026-02-24
**Status:** REVISIT (AUC=0.659, MVR not met)
**Paper:** P2 (Settlement Suitability Model)
**Author:** Amien + Claude

## Hypothesis

Ancient settlements in East Java were located in terrain that can be predicted from
environmental features (slope, elevation, water access, terrain stability). A model
trained ONLY on these features — with no volcanic proximity information — should achieve
spatial AUC > 0.75.

Challenge 1 (Tautology Test): If the model predicts high suitability in high-burial zones
(where few sites are currently known), that is evidence that settlement suitability and
volcanic burial are spatially correlated but independent — supporting H1.

## Method

1. Positive samples: 378 geocoded sites from east_java_sites.geojson (after feature filtering)
2. Pseudo-absences: 1890 (5x sites), spatially stratified (not in same grid cell as known sites)
3. Features: slope, elevation, TWI, TRI, aspect (from E003 DEM — all environmental, NO volcanic proximity)
4. Algorithm: XGBoost classifier (primary) + Random Forest (secondary)
5. Validation: Spatial block cross-validation (5 folds, ~50km blocks)
6. Metrics: Spatial AUC-ROC, TSS (per EVAL.md)
7. Output: Probability map + Challenge 1 tautology test

## MVR

- Spatial AUC > 0.75 → Paper 2 GO
- Spatial AUC > 0.65 but < 0.75 → tune features, retry
- Spatial AUC < 0.65 consistently → kill signal (H3 may be falsified)

## Data

- `data/processed/east_java_sites.geojson` — positive samples (n=378 after filtering)
- `data/processed/dem/jatim_*.tif` — DEM derivatives (slope, TWI, TRI, aspect, elevation)

## Results

| Model | Spatial AUC | TSS |
|-------|------------|-----|
| XGBoost | 0.659 ± 0.077 | 0.318 ± 0.126 |
| RandomForest | 0.656 ± 0.090 | 0.314 ± 0.133 |

Top features: elevation (0.238), TWI (0.217), TRI (0.206), slope (0.176), aspect (0.164).

Challenge 1: rho=-0.095 (p<0.001), TAUTOLOGY-FREE.

## Conclusion

Terrain-only baseline achieves AUC=0.659 — within REVISIT range (0.65-0.75). Not a kill signal,
but MVR not met. Model captures weak terrain signal; needs richer features or better
pseudo-absence design. Proceeded to E008 (add river distance).
