# E167: Priority Fieldwork Map — The VOLCARCH Treasure Map

**Status:** SUCCESS
**Date:** 2026-03-31
**Type:** [M] Methodology / GIS
**Papers:** P1, P2, P17
**Output:** `results/priority_score.tif` (GeoTIFF, 30m) + `results/priority_fieldwork_map.png`

## Method

Integrated 3 spatial layers into a single priority score for every 30m pixel in East Java:

1. **Settlement suitability** — Proxy from slope, river distance, elevation (calibrated to E013 key features)
2. **Burial feasibility** — From E166 burial depth map. Optimal: 1-3m (GPR range). Penalized: <1m (already visible) or >6m (too deep)
3. **Novelty** — Areas with NO known archaeological sites within 5km score highest

**Priority = Suitability × Feasibility × Novelty** (normalized 0-100)

## Key Results

- **Top 1% priority area: 994 km2** of East Java
- Top targets cluster around **Lawu western flank** (1.0-1.2m burial depth)
- **89.3% of East Java has no known sites within 5km** — vast untested area
- Priority score GeoTIFF available for GIS overlay with any fieldwork planning tool

## For Fieldwork Partners

This map answers: **"If you had one week and one GPR unit, where would you go?"**

The top-priority zones are where:
- Ancient settlement was probable (good slope, near rivers, moderate elevation)
- Burial depth is in GPR range (1-3m, detectable)
- No sites are currently known (genuine terra incognita)

Load `priority_score.tif` into QGIS or ArcGIS to explore specific locations.
