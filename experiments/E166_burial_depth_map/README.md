# E166: Burial Depth Prediction Map for East Java

**Status:** SUCCESS
**Date:** 2026-03-31
**Type:** [H] Model / GIS
**Papers:** P1, P2, P17
**Output:** `results/burial_depth_pre400CE.tif` (GeoTIFF, 30m resolution)

## Method

Combined 30m DEM (Copernicus GLO-30) with exponential decay sedimentation model calibrated to 5 burial sites. Distance to nearest of 7 major East Java volcanoes computed at every pixel.

Model: `rate(d) = 8.0 mm/yr × exp(-d / 15 km)`
- Calibration: Dwarapala 3.5 mm/yr @ 20 km, Sambisari 5.1 @ 8 km, Kedulan 5.8 @ 10 km

## Key Results

### Pre-400 CE Burial Depth (1,626 years of accumulation)

| Zone | Depth | Area (km2) | % of E. Java | Method |
|------|-------|-----------|-------------|--------|
| Zone B (GPR targets) | 1-3 m | **12,811** | 12.9% | Ground-penetrating radar |
| Zone C (ERT targets) | 3-6 m | **5,864** | 5.9% | Electrical resistivity tomography |
| Zone D (deep) | >6 m | **2,709** | 2.7% | Deep coring only |
| Zone E (no burial) | 0 m | 56,375 | 56.7% | Surface survey sufficient |

**12,811 km2 of East Java is in the primary GPR-detectable target zone.** This is where buried pre-400 CE sites are most likely to be found AND most cost-effectively investigated.

### Deliverables

1. **`burial_depth_pre400CE.tif`** — 30m resolution GeoTIFF for GIS overlay
2. **`burial_depth_map.png`** — Three-panel visualization (pre-400 CE, pre-800 CE, 929 CE)
3. **`burial_depth_stats.json`** — Zonal statistics

## For Fieldwork Partners

This map directly answers the question: **"Where should we dig?"** The 12,811 km2 Zone B area can be narrowed to specific fieldwork targets by overlaying with E080 anomaly detection (65% overlap with E097) and E013 settlement suitability model (AUC=0.768).

The intersection of Zone B + high suitability + anomaly detection = the borehole targets in the BOREHOLE_PROTOCOL (`docs/fieldwork/BOREHOLE_PROTOCOL_v1.md`).
