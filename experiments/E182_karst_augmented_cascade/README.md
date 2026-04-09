# E182: Karst-Augmented Cascade Model

**Date:** 2026-04-09
**Paper:** P1, P17 (cascade revision)
**Status:** SUCCESS (PARTIAL) — Karst improves rank prediction (rho 0.321→0.500) but magnitude calibration poor. Directional insight confirmed.
**Type:** [H] Hypothesis test

## Hypothesis

Adding a karst bypass term to the E110 cascade model improves cross-regional prediction. Cave sites bypass all 5 cascade factors, creating an additive escape route from the multiplicative extinction.

## Method

Augmented model: P(visible) = [F1 x F2 x F3 x F4 x F5] + [karst_fraction x P(cave_preserved)]

Tested across 7 SE Asian regions from E178. P(cave_preserved) = 0.10 baseline, sensitivity 0.01-0.50.

## Results

- Rank correlation improved: rho 0.321 → 0.500 (with karst)
- Best P(cave) for rank = 0.05 (rho=0.607)
- Log RMSE worsened (magnitude calibration poor — model overpredicts everywhere)
- Key: Java volcanic predicted correctly as lowest-visibility region with karst model

## Key Insight

Karst is directionally important but the additive model is too simple. Cave preservation interacts with survey intensity (caves are EASY to survey → F3 bypass) and recognition (cave stratigraphy understood → F4 bypass). A more sophisticated model would make karst bypass ALL factors, not just add independently.

## Caveats

1. P(cave_preserved) = 0.10 is a guess; needs calibration from actual cave survey data
2. Expected site density per 1000km2 varies by region — this creates magnitude errors
3. N=7 regions is small for model comparison
4. Factor values for non-Java regions are estimated, not empirical
