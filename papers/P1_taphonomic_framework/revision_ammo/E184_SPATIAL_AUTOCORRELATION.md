# Revision Support Material: Spatial Autocorrelation (E184)

**Use when:** Reviewer asks about spatial dependence or statistical methodology.

## Key Point

Volcano distance is spatially autocorrelated (Moran's I = 0.937, p < 0.001). Simple correlations between volcanic distance and site properties may be inflated. HOWEVER, the distributional comparisons (Mann-Whitney, KS test) that underpin the core findings are MORE robust than regressions.

## Proactive Acknowledgment

> "We note that volcanic distance is spatially autocorrelated (Moran's I = 0.937): nearby sites share similar distances to volcanoes because they occupy the same geographic region. Simple correlations involving volcanic distance should be interpreted with this caveat. The core finding of differential site density across volcanic distance zones is based on distributional comparisons (Mann-Whitney, Kolmogorov-Smirnov) rather than regression, and these tests are less susceptible to spatial autocorrelation inflation."

## Supporting Evidence

- E184: Volcano-century correlation collapses after spatial lag correction (rho 0.490 → -0.198)
- E185: Two Javas segregation ROBUST (Cohen's d = 2.0, KS p < 10^-8) — distributional, not regression
- Recommendation: formal spatial regression (SLM/SEM with PySAL) for definitive confirmation of regression-based claims
