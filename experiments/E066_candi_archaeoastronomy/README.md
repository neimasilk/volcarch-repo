# E066: Candi Archaeoastronomy — Entrance Orientation vs Solar Azimuths

**Status:** SUCCESS
**Date:** 2026-03-12
**Channel:** Ch9 (Archaeoastronomy)
**Papers served:** P11 (volcanic informedness — siting vs orientation contrast)

## Hypothesis

Candi entrance orientations follow equinoctial (E/W) directions prescribed by Hindu canonical architecture, not volcanic azimuths. This would confirm that WHERE to build is volcanically informed while HOW to orient is astronomically/religiously determined.

## Method

- 20 candi with documented entrance orientations from E031 dataset
- Computed solar azimuths (equinox sunrise/sunset, solstice sunrise/sunset) at Java latitude (7.5°S)
- Binomial tests comparing alignment rates vs random expectation
- McNemar paired test: equinox alignment vs volcanic alignment for same 20 candi
- Regional comparison: East Java vs Central Java patterns

## Key Results

| Metric | Observed | Expected (random) | p-value |
|--------|----------|-------------------|---------|
| Equinox-aligned (E or W) | 85.0% (17/20) | 11.1% | 4.9×10⁻¹⁴ |
| Cardinal-aligned (N/E/S/W) | 100% (20/20) | 22.2% | 8.6×10⁻¹⁴ |
| Faces volcano (±45°) | 35.0% (7/20) | 25.0% | 0.94 (null) |

**McNemar test (equinox vs volcano):** χ²=10.00, p=0.0016
- 10 candi face equinox but NOT volcano
- 0 candi face volcano but NOT equinox
- All 7 "volcano-facing" candi face west, where the volcano coincidentally is

### Entrance Distribution
- East (90° = equinox sunrise): 5 (25%)
- West (270° = equinox sunset): 12 (60%)
- North (0°): 2 (10%)
- South (180°): 1 (5%)

### Solar Azimuths at Java Latitude (7.5°S)
- Equinox sunrise: 90.0° / sunset: 270.0°
- June solstice sunrise: 66.3° / sunset: 293.7°
- December solstice sunrise: 113.7° / sunset: 246.3°

### Regional Pattern
- East Java: 7/10 face west (70%) — Majapahit convention
- Central Java: 5/10 face west (50%)
- Fisher exact: p=0.65 (no significant regional difference)

## Conclusion

**SUCCESS.** 85% of candi entrances align with equinox directions (due East or West), far exceeding the 11% random expectation (p=4.9×10⁻¹⁴). All 20 candi are on exact cardinal axes (100%). In contrast, only 35% face their nearest volcano, and all of these coincidentally face west where the volcano happens to be — zero candi face a volcano without also facing an equinox direction.

This confirms the P11 siting-vs-orientation contrast: candi builders used VOLCANIC knowledge to choose WHERE to build (western flanks, Zone A) but followed ASTRONOMICAL/CANONICAL conventions for HOW to orient the entrance. The two knowledge systems coexist without conflict.

## Files
- `analyze.py` — Analysis script
- `results/e066_results.json` — Machine-readable results
- `results/candi_archaeoastronomy.png/.pdf` — Compass rose comparison
- `results/alignment_comparison.png/.pdf` — Alignment rate bar chart
- `results/regional_pattern.png` — East vs Central Java
