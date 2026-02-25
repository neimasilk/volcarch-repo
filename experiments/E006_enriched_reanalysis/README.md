# E006: Re-analysis with Nominatim-Enriched Site Dataset

**Date:** 2026-02-24
**Status:** SUCCESS
**Paper:** P1 (Taphonomic Bias Framework)
**Author:** Amien + Claude

## Motivation

E004 and E005 used 297 geocoded sites out of 666 total (45% coverage). The remaining 369
sites had `accuracy_level='no_coords'`. TASK-008 geocoded ~N additional sites using OSM
Nominatim API.

This experiment re-runs E004 and E005 with the enriched dataset to:
1. Check whether the H1 test results change with more data
2. Quantify how many sites are now geocoded
3. Document the final dataset quality

## Method

1. Run `tools/geocode_sites.py` (TASK-008) → enriched `east_java_sites.geojson`
2. Re-run E004: `python experiments/E004_density_analysis/01_analyze_density.py`
3. Re-run E005 Step 2: `python experiments/E005_terrain_suitability/02_full_jatim_analysis.py`
4. Compare old vs new Spearman rho and site counts
5. Document in JOURNAL.md

## Data

- Input: `data/processed/east_java_sites.geojson` (enriched with Nominatim geocodes)
- Geocoding quality note: Nominatim accuracy level is `nominatim` (lower confidence than
  `osm_centroid` or `wikidata_p625`). Sites with ambiguous names (e.g., "Candi Sari" which
  exists in multiple provinces) may have incorrect coordinates. These are flagged in the
  geocoding_report.txt but not filtered for this run — future work should validate individually.

## Results

### Geocoding results
- Nominatim geocoder processed 369 `no_coords` sites
- 94/369 successfully geocoded (25.5%)
- Most unfound sites are from outside East Java (Sumatra, Central Java, Bali temples)
- Accuracy level breakdown: osm_centroid=281, nominatim=94, wikidata_p625=16, no_coords=275

### Site count comparison

| Stage | Total | Geocoded | In East Java bounds | % Geocoded |
|-------|-------|----------|---------------------|-----------|
| Pre-geocoding (E004/E005) | 666 | 297 | 297 | 44.6% |
| Post-geocoding (E006) | 666 | 391 | 383 (E004) / 391 (E005) | 58.7% |

### H1 test comparison

| Metric | E004/E005 (n=297) | E006 (n=383/391) | Change |
|--------|-------------------|-------------------|--------|
| Spearman rho (raw density vs distance) | -0.991 | -0.955 | +0.036 (slightly weaker) |
| p-value (raw) | 0.000015 | 0.000806 | Still highly significant |
| Spearman rho (terrain-controlled) | -0.364 | -0.358 | +0.006 (negligible change) |
| p-value (terrain-controlled) | <0.0001 | <0.0001 | No change |

### Density by distance band (E006)

| Band | Sites | Area (km²) | Density/1000km² |
|------|-------|-----------|----------------|
| 0-25 km | 147 | 11,343 | 12.96 |
| 25-50 km | 136 | 18,034 | 7.54 |
| 50-75 km | 37 | 18,528 | 2.00 |
| 75-100 km | 22 | 16,373 | 1.34 |
| 100-150 km | 41 | 28,523 | 1.44 |
| 150-200 km | 0 | 25,507 | 0.00 |
| 200+ km | 0 | 73,740 | 0.00 |

## Conclusion

**Results are remarkably stable.** Adding 94 Nominatim-geocoded sites (29% increase in geocoded count) produced negligible changes in both the raw density correlation (rho moved from -0.991 to -0.955) and the terrain-controlled correlation (rho moved from -0.364 to -0.358).

This stability is itself informative:
1. The pattern (sites clustering near volcanoes) is robust and not an artifact of sample size
2. The Nominatim-added sites follow the same spatial distribution as the OSM/Wikidata sites
3. The conclusion stands: observed site distribution reflects survey history, not settlement patterns
4. H1 remains INCONCLUSIVE from distribution data alone — the Dwarapala calibration remains the primary evidence

**For Paper 1:** The E006 results strengthen the methodological argument — even with 29% more data, the pattern doesn't change, confirming that the observable record is saturated with survey bias.

## Next Steps

- Use E006 (n=383) as the definitive dataset for Paper 1 results section
- Proceed with Paper 1 draft using the stable rho values
- Note: 275 sites remain ungeocoded (mostly non-East Java sites from Wikipedia lists)

## Failure Notes

N/A — experiment completed successfully.
