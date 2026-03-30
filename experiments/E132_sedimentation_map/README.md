# E132: Sedimentation Rate Prediction Map

**Date:** 2026-03-30
**Status:** PARTIAL (model too simple, predictions useful as first approximation)
**Paper:** P1 (supporting), P22 (JavaTephroChron precursor)
**Layer:** L1

---

## Hypothesis

Using calibration data from E083 + E128, we can build a spatial model predicting sedimentation rate and burial depth at any point in East Java.

## Method

Exponential decay model: rate = background + sum(volcano_rate * exp(-distance/15km)). Validated against 7 calibration points. Applied to 2,700 grid cells across East Java.

## Results

### Validation: POOR

| Metric | Value |
|--------|:---:|
| RMSE | 2.54 mm/yr |
| Correlation | 0.162 |
| Mean predicted | 2.43 mm/yr |
| Mean observed | 4.59 mm/yr |

**Model systematically underpredicts.** The exponential decay is too simple — doesn't account for topography, wind direction, eruption history. Merapi (Sambisari/Kedulan) has anomalously high rates that a generic model misses.

### Key Location Predictions (use with caution)

| Location | Rate (mm/yr) | Depth 400 CE | Depth 200 BCE |
|----------|:---:|:---:|:---:|
| E080 Target #1 (Kelud) | 3.1 | 5.1 m | 7.0 m |
| E080 Target #7 (Arjuno) | 4.7 | 7.6 m | 10.4 m |
| Trowulan (Majapahit) | 1.9 | 3.1 m | 4.3 m |
| Sangiran (H. erectus) | 0.9 | 1.5 m | 2.0 m |

### Grid Statistics

- 10% of East Java has rate >2 mm/yr (volcanic zone)
- 4% has predicted burial >5m since 400 CE (deep burial zone)

## Conclusion

**PARTIAL.** The model concept works (exponential decay from volcanoes) but implementation is too simple for publication. Needs per-volcano calibration, topographic correction, and wind-field modeling — exactly what P22 (JavaTephroChron) proposes. The 2,700-cell grid is a useful framework for future refinement.

**Honest note:** RMSE of 2.54 mm/yr against 4.59 mm/yr mean = 55% error. This model is a SKETCH, not a prediction. P22 would improve by 10-50x using FALL3D per-eruption modeling.

## Scripts

- `sedimentation_map.py` — Exponential decay model + grid generation
