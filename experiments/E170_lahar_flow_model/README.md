# E170: TWI-Enhanced Burial Depth Model

**Status:** SUCCESS (MARGINAL IMPROVEMENT)
**Date:** 2026-03-31
**Type:** [H] Model refinement
**Papers:** P1, P2

## Method
Combined Topographic Wetness Index (TWI, proxy for valley accumulation) with distance decay to create a physically-motivated burial depth model. TWI-high areas (valleys) receive more lahar material than TWI-low areas (ridges) at the same distance.

## Results
- TWI-enhanced model correlates rho=0.986 with pure distance model (E166)
- Distance-from-volcano dominates at regional scale (East Java)
- TWI refinement adds <2% explanatory power
- Valley/ridge difference is secondary to distance effect

## Conclusion
At the scale of East Java, **distance from volcano is sufficient** for burial depth prediction. TWI refinement would matter more at site-specific scale (within 5 km of a volcano), where valley channeling concentrates lahar deposits. For regional fieldwork targeting, E166's distance model is adequate.

GeoTIFF output: `results/burial_depth_twi.tif` (30m resolution, full East Java)
