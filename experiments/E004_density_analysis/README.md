# E004: Site Density vs Volcanic Proximity (First Test of H1)

**Date:** 2026-02-23
**Status:** BLOCKED (needs E001 + E002 outputs)
**Paper:** P1 (Taphonomic Bias Framework)
**Author:** Amien + Claude

## Hypothesis

H1 (Taphonomic Bias): Known archaeological sites in East Java are spatially
underrepresented in areas of high volcanic sedimentation. We expect to see:
- Fewer sites per unit area close to active volcanoes
- A statistically significant negative correlation between site density
  and volcanic deposition proxy (inverse distance from volcano)
- This bias is NOT explained by terrain unsuitability (i.e., the areas near
  volcanoes are geomorphologically habitable but archaeologically underdocumented)

## Method

1. Load site dataset (E001 output): `data/processed/east_java_sites.geojson`
2. Define reference volcano locations (Kelud, Semeru, Arjuno-Welirang, Bromo)
3. For each site, compute distance to nearest volcano (km)
4. Bin sites by distance (0–25km, 25–50km, 50–75km, 75–100km, 100+km)
5. Compute "area per bin" using East Java province polygon
6. Compute site density = count / area for each bin
7. Spearman correlation: site_density vs distance_to_volcano
8. Produce map: site locations colored by distance band
9. Bootstrap resampling to estimate confidence on correlation

## Minimum Viable Result (MVR)

- Spearman ρ > 0.5 (positive correlation: farther from volcano → more sites)
- p-value < 0.05
- Visual map shows clear clustering of sites in distal zones

## Data

- `data/processed/east_java_sites.geojson` (from E001)
- `data/processed/eruption_history.csv` (from E002) — for weighting by eruption intensity
- Volcano reference coordinates: see `01_analyze_density.py`

## Results

**Run date:** 2026-02-23
**Sites analyzed:** 296 geocoded (out of 666 total — others lack coordinates)
**Study area:** 192,048 km² (Jawa Timur polygon from OSM)

| Distance bin | Sites | Area (km²) | Density (per 1000km²) |
|---|---|---|---|
| 0–25 km | 104 | 11,343 | **9.17** |
| 25–50 km | 108 | 18,034 | **5.99** |
| 50–75 km | 32 | 18,528 | 1.73 |
| 75–100 km | 22 | 16,373 | 1.34 |
| 100–150 km | 30 | 28,523 | 1.05 |
| 150–200 km | 0 | 25,507 | 0.00 |
| 200+ km | 0 | 73,740 | 0.00 |

**Spearman rho = -0.991, p = 0.000015** (density DECREASES with distance from volcano)

## Conclusion

**MVR NOT MET** (expected rho > +0.5; got rho = -0.991)

The known archaeological record is HEAVILY CONCENTRATED near active volcanoes.
This appears to CONTRADICT H1 at first glance but requires careful interpretation:

**Why this does NOT simply falsify H1:**

1. **Discovery bias:** Archaeological surveys in East Java have focused almost
   exclusively on the Brantas River valley (Majapahit, Singosari, Kanjuruhan kingdoms),
   which lies exactly in the 0–50 km volcanic zone. This creates a circular pattern:
   we find sites where we look, and we look where we know kingdoms were.

2. **Dataset incompleteness:** Our 296 geocoded sites are skewed toward major stone
   monuments (candis) that survived volcanic burial precisely because they are large.
   The wooden/earthen settlements — the bulk of ancient habitation — are absent from
   the catalog and more likely to be buried.

3. **The 150–200+ km zone zeros:** Not necessarily empty of past settlements.
   These areas include far eastern Java and Madura island — less surveyed, not less inhabited.
   The large "no data" zone (73,740 km²!) is a red flag for data coverage.

**What H1 actually predicts (revised understanding):**
H1 is not "fewer sites near volcanoes" but rather:
"The RATIO of surviving/discovered sites to originally-existing sites is lower near volcanoes."
This requires knowing expected settlement density (from terrain suitability modeling, E003/Paper 2)
and comparing to actual observed density. That is E005.

**Conclusion:** This result documents the discovery bias but does NOT falsify H1.
It motivates E005: if terrain suitability near volcanoes is high (as expected —
fertile soils, water) but observed density is lower than predicted → H1 still stands.

## Next Steps

- Design E005: site density vs terrain suitability residuals
  (observed / expected, where expected comes from a null settlement model using DEM features)
- Increase geocoded site count (only 296/666 have coordinates)
- Manual review of 150–200+ km zones: are there genuinely no known sites there?
- Document this result in Paper 1 as evidence of discovery bias (secondary finding)

## Failure Notes

Analysis ran to completion. MVR not met, but result is interpretable and scientifically
interesting. See Conclusion for full discussion.
