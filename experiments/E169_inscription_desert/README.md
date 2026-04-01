# E169: Inscription Desert Analysis

**Status:** SUCCESS
**Date:** 2026-03-31
**Type:** [H] Hypothesis test / GIS
**Papers:** P1, P17

## Hypothesis

If inscription distribution follows predictable spatial patterns (E084: peak at 15-30 km from volcanoes), there should be zones where inscriptions are EXPECTED but absent. These "inscription deserts" represent the invisible cultural production of Volcano Java.

## Method

1. Computed KDE of 174 geocoded inscriptions (E082)
2. Modeled expected inscription density (Gaussian, peak at 25 km, sigma=10 km)
3. Computed desert score = expected - observed
4. Identified contiguous desert regions

## Key Results

- **77.1% of the expected inscription zone is EMPTY** (16,738 km2 of 21,697 km2)
- 3 major inscription deserts identified:

| Desert | Location | Area (km2) | Mean Score | V. Distance | Elevation |
|--------|----------|-----------|------------|-------------|-----------|
| 1 | Malang/Kelud zone | **9,630** | 0.81 | 25.5 km | 288 m |
| 2 | Lawu transition | 3,494 | 0.82 | 26.6 km | 236 m |
| 3 | Semeru/Bromo zone | 3,614 | 0.81 | 26.9 km | 489 m |

## Conclusion

The inscription deserts are the negative image of the Two Javas model. Where Court Java has inscriptions, Volcano Java has silence — not because nobody lived there (E108: 3,220x gap), but because:

1. The court-inscription genre was spatially concentrated (E084)
2. Volcanic burial concealed proximal inscriptions (E102)
3. Organic-media writing left no trace (E113)
4. Survey effort follows existing finds, not predictions (E129)

The 9,630 km2 Malang/Kelud desert is VOLCARCH's primary fieldwork target — the largest contiguous area where inscriptions SHOULD exist but DON'T. This is where the borehole protocol (docs/fieldwork/) targets its 20 holes.
