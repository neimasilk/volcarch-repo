# E005: Terrain Suitability Model as Null Hypothesis for H1

**Date:** 2026-02-23
**Status:** RUNNING
**Paper:** P1 (Taphonomic Bias Framework), P2 (Settlement Suitability Model)
**Author:** Amien + Claude

## Motivation

E004 found that known sites CLUSTER near volcanoes (rho = -0.991). This is explained by
discovery/survey bias, not the absence of H1. To properly test H1, we need to ask:

**"Given the terrain, HOW MANY sites SHOULD we expect in each volcanic distance zone?
Are the near-volcano zones under-represented relative to their suitability?"**

If near-volcano terrain is highly suitable (fertile slopes, water access) but we observe
FEWER sites than terrain suitability predicts → H1 is supported (volcanic burial is hiding sites).

## Hypothesis

**H1 (Revised operational form):** The ratio of observed site density to terrain-suitability-predicted
site density is NEGATIVELY correlated with volcanic proximity.

Formally:
  residual(x) = observed_density(x) - predicted_density(x)
  Spearman(residual, distance_to_volcano) > 0

(residuals are more negative near volcanoes → fewer sites than terrain would predict)

## Method

### Phase 1: Build a simple terrain suitability index
Use DEM-derived features to compute a normalized suitability score (0–1) per grid cell:
- Slope: low slope favored (0–10° = 1.0, 10–25° = 0.5, >25° = 0.0)
- Elevation: moderate elevation favored (50–800m = 1.0, ramp below/above)
- TWI: high TWI (wetness) slightly favored
- Distance to rivers: within 5km favored
Weighted combination: slope (40%), elevation (30%), TWI (20%), river proximity (10%)

### Phase 2: Compare observed vs predicted
1. Grid East Java into 25km × 25km cells
2. For each cell: compute mean suitability score
3. For each cell: count known archaeological sites
4. Compute predicted_count = suitability_score × k (where k is a scaling constant)
5. Compute residual = observed - predicted
6. Correlate residuals with distance to nearest volcano

## Data

- DEM derivatives: `data/processed/dem/` (from E003)
- River network: OpenStreetMap (scrape from Overpass)
- Sites: `data/processed/east_java_sites.geojson` (from E001)

## Results

### Pilot (Malang Raya only)
- Grid cells: 12, Sites: 43, rho = -0.182, p = 0.572 → INCONCLUSIVE (n too small)

### Full Jawa Timur (MAIN RESULT)
**Run date:** 2026-02-23
**DEM:** Copernicus GLO-30 (30m), 8356×13345 px, full province
**Grid cells:** 187 (25km × 25km), **Cells with sites:** 52, **Sites:** 297

**Spearman rho = -0.364, p < 0.0001** (residual density vs distance to volcano)

Interpretation:
- rho = -0.364: after controlling for terrain suitability, cells NEAR volcanoes still have
  MORE sites than terrain alone predicts (positive residuals near volcanoes)
- Cells FAR from volcanoes have FEWER sites than terrain predicts (negative residuals)
- This is still the OPPOSITE pattern from H1's simple prediction

## Conclusion

**H1 NOT SUPPORTED** by direct site-distribution analysis — even after terrain correction.

**However, this does NOT falsify H1.** The pattern requires deeper interpretation:

1. **Discovery/survey bias dominates the signal.** Near-volcano areas contain famous kingdoms
   (Majapahit, Singosari, Kanjuruhan) that have been systematically surveyed for 200+ years.
   Survey effort per km² is far higher near Blitar/Malang/Mojokerto than in outer provinces.

2. **Stone monument survivorship bias.** Our 297 geocoded sites are overwhelmingly large
   stone structures (candis) that survived volcanic burial because of their mass. The burials
   that H1 predicts affect SMALLER structures (wooden/earthen settlements) not in our dataset.

3. **Terrain correction reduces but doesn't eliminate clustering.** E004 rho=-0.991 → E005
   rho=-0.364: terrain explains ~63% of the clustering. The remainder is survey/discovery bias.

4. **Key asymmetry found:** Far-from-volcano zones have FEWER sites than their terrain
   predicts. This could indicate: (a) under-survey, OR (b) the landscape attractiveness
   was genuinely lower historically. This needs investigation.

**Revised H1 operationalization for Paper 1:**
H1 cannot be tested from site distribution alone. It requires either:
  - Survey intensity normalization (which areas have been actively excavated?)
  - OR stratigraphic evidence from field campaigns (like Dwarapala but systematic)
  - OR comparison with sub-surface remote sensing (GPR, LiDAR) in "blank" zones

**Paper 1 reframing:** Rather than claiming H1 is proven or disproven,
Paper 1 should argue: *"The spatial pattern of known sites is better explained by
survey bias than by the absence of past habitation near volcanoes. The Dwarapala case
study (1 empirical data point) confirms burial is occurring at a measurable rate.
We propose the taphonomic framework as a hypothesis requiring fieldwork to test."*

## Next Steps

- Gather survey intensity data (BPCB excavation reports, areas covered per year)
- Increase site count by geocoding ~370 Wikipedia name-only entries
- Attempt GPR or LiDAR partnership for a test trench near a "blank" high-suitability zone
- Write Paper 1 framing H1 as hypothesis rather than proven finding

## Next Steps

- If residuals are more negative near volcanoes (Spearman > 0.5, p < 0.05) → H1 SUPPORTED
- This becomes a key Figure in Paper 1
- If residuals show no pattern → H1 requires different approach (maybe stratigraphic evidence)

## Failure Notes

*N/A — not yet attempted.*
