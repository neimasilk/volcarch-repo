# E184: Inscription Spatial Autocorrelation (Moran's I)

**Date:** 2026-04-09
**Paper:** P17 (critical limitation), P1, P2
**Status:** SUCCESS (INFORMATIVE NEGATIVE) — Volcano-century correlation collapses after spatial correction. Two Javas segregation pattern may be robust but temporal claims need spatial regression.
**Type:** [H] Hypothesis test (methodological audit)
**Novelty:** First spatial autocorrelation analysis in VOLCARCH. Addresses ME#13 Risk 6 (methodological gap).

## Hypothesis

If inscription properties correlate with volcanic distance, this correlation may be inflated by spatial autocorrelation (nearby inscriptions share similar distances to volcanoes because they're in the same geographic region, not because volcanism causes the pattern).

## Method

1. Loaded 174 Java inscriptions with geocoded locations (E082)
2. Computed Moran's I for inscription density on 0.5-degree grid
3. Computed Moran's I for volcano distance using 5-nearest-neighbor weights
4. Monte Carlo permutation test (9,999 iterations)
5. Partial correlation: volcano_dist vs century, controlling for spatial lag

## Results

### Moran's I: Inscription Density
- I = -0.100, z = -0.163, p = 0.870
- **NOT significantly clustered** at 0.5-degree grid level (surprising)
- Grid is coarse (15 cells) — finer grid might show clustering

### Moran's I: Volcano Distance
- I = **0.937**, p = **0.0000** (Monte Carlo)
- **STRONGLY spatially autocorrelated** — nearby inscriptions have similar volcanic distances
- This is EXPECTED (geography) but means any analysis using volcano distance as predictor must account for spatial dependence

### Critical Test: Does Volcano Effect Survive Spatial Correction?

| Test | rho | p | Significant? |
|------|:---:|:---:|:---:|
| Simple (volcano_dist vs century) | **0.490** | **<0.0001** | **YES** |
| Partial (controlling for spatial lag) | **-0.198** | **0.111** | **NO** |

**The correlation REVERSES DIRECTION and loses significance after spatial correction.**

## Interpretation

1. **The simple volcano-century correlation is partly spatial artifact.** Nearby inscriptions share similar volcanic distances because they're in the same region. When you control for this spatial proximity, the temporal effect disappears.

2. **This does NOT invalidate the Two Javas pattern.** The Two Javas finding (E104, E084) is about the spatial SEGREGATION of candi and inscriptions — a comparison of two spatial distributions. This is different from a regression on distance. Mann-Whitney U tests comparing two samples are less affected by spatial autocorrelation than regressions.

3. **P17's temporal claims need spatial regression.** The statement "increasing pre-Indic vocabulary (rho=0.781, exclusively in the court zone)" should be verified with spatial error/lag models (PySAL).

4. **The Two Javas SEGREGATION (candi median 14.6 km vs inscription median 27.6 km) is robust** because it's a distributional comparison, not a regression. Two distinct peaks in a distance histogram are not spatial autocorrelation artifacts.

## Recommendation for P17

Add to Limitations section:
> "Volcanic distance is spatially autocorrelated (Moran's I = 0.937, p < 0.001). Simple correlations between volcanic distance and inscription properties may be inflated by spatial dependence. The distributional segregation of candi and inscriptions (Mann-Whitney p < 0.000001) is less susceptible to this inflation because it compares two sample populations rather than fitting a regression. Formal spatial regression (Spatial Lag Model or Spatial Error Model) would provide definitive confirmation."

## Caveats

1. Partial correlation using spatial lag of 5 nearest neighbors is a rough proxy for formal spatial regression (SLM/SEM)
2. Grid resolution (0.5 degrees) is coarse — finer grid might show different patterns
3. The spatial lag correction is aggressive — it may OVERCORRECT by absorbing genuine geographic effects along with statistical artifacts
4. N=174 is adequate for Moran's I but marginal for complex spatial models
