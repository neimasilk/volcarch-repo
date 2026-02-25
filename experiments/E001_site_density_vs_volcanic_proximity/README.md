# E001: Archaeological Site Geocoding & Density Analysis

**Date:** 2026-02-23
**Status:** RUNNING
**Paper:** P1 (Taphonomic Bias Framework)
**Author:** Amien + Claude

## Hypothesis

Known archaeological sites in East Java are spatially biased away from high-volcanic-deposition zones.
Sites near active volcanoes (Kelud, Semeru, Arjuno-Welirang) will be underrepresented relative to
expected settlement density given terrain suitability.

## Method

1. Collect all known archaeological sites in East Java from:
   - OpenStreetMap Overpass API (`historic=archaeological_site` + `historic=ruins` in Jawa Timur)
   - Wikipedia list of candis in East Java (manual + scripted)
   - BPCB Jawa Timur portal (if parseable)
2. Geocode sites to lat/lon with accuracy flags
3. Compute distance from each site to nearest active volcano (Kelud, Semeru, Arjuno, Welirang, Bromo)
4. Compare site density in bins of volcanic proximity
5. Run correlation test (Spearman) between site density and volcanic deposition proxy (distance-based for now)

## Data

- Input: none (collecting from web sources)
- Output: `data/processed/east_java_sites.geojson`
- Schema: `name`, `type`, `period`, `lat`, `lon`, `source`, `discovery_year`, `accuracy_level`, `notes`, `osm_id`

## Results

*Pending — experiment not yet run.*

## Conclusion

*Pending.*

## Next Steps

- E004 depends on this: site density vs volcanic proximity statistical test
- Wikipedia and BPCB data should supplement OSM (OSM likely undercounts)

## Failure Notes (if applicable)

*N/A — not yet attempted.*
