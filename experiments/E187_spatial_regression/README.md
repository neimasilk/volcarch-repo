# E187: Proper Spatial Regression (PySAL spreg)

**Date:** 2026-04-09
**Paper:** P17 (critical methodology upgrade)
**Status:** SUCCESS (INFORMATIVE NEGATIVE) — Volcanic distance effect on century DOES NOT survive spatial regression. Rho=0.620 (strong spatial lag). Two Javas segregation still robust.
**Type:** [R] Robustness / [H] Hypothesis test (methodological)
**Novelty:** First proper spatial regression in VOLCARCH using PySAL ML estimation.

## Hypothesis

The relationship between volcanic distance and inscription century (OLS: p=0.002) may be a spatial artifact. Proper spatial regression (Spatial Lag / Spatial Error models) will determine if the effect survives.

## Results

| Model | Beta (dist) | p-value | R2 | AIC | Spatial param |
|-------|:---:|:---:|:---:|:---:|:---:|
| OLS | 0.034 | **0.002** | 0.139 | 230.6 | — |
| **Spatial Lag (ML)** | 0.016 | **0.094** | 0.387 | 213.6 | Rho=0.620 (p<0.001) |
| **Spatial Error (ML)** | 0.022 | **0.241** | 0.139 | 213.5 | Lambda=0.626 (p<0.001) |

**The volcanic distance effect on inscription century DOES NOT SURVIVE spatial regression.** Both Spatial Lag and Spatial Error models render it non-significant (p>0.05).

## Key Findings

1. **Strong spatial autocorrelation:** Rho=0.620 (Lag) and Lambda=0.626 (Error) — both highly significant (p<0.001). Nearby inscriptions have similar centuries because they're in the same political-geographic region, not because of volcanism.

2. **OLS was inflated:** The OLS beta (0.034, p=0.002) drops to 0.016 (p=0.094) in the Lag model — a 53% reduction. The significance vanishes.

3. **LM diagnostics:** Both LM-Lag (p<0.001) and LM-Error (p<0.001) are significant, confirming spatial dependence. Robust LM-Lag is marginal (p=0.068), Robust LM-Error is not significant (p=0.763) — suggesting Spatial Lag is the better specification.

4. **Two Javas segregation UNAFFECTED:** This finding (E185: Cohen's d=2.0) is a distributional comparison, not a regression. It does not depend on the volcano-century correlation that spatial regression kills.

## Implications for P17

1. **Temporal claims should be softened.** The statement about "increasing pre-Indic vocabulary" being driven by volcanic distance should be reframed as "spatially patterned" rather than "caused by volcanic proximity."

2. **The core finding survives.** Candi cluster at 0-15km, inscriptions at 15-30km (Mann-Whitney p<0.000001). This is IMMUNE to spatial regression because it's a two-sample comparison.

3. **This is a STRENGTH, not a weakness.** Reporting this honestly shows methodological sophistication. Reviewers will respect the self-correction.

4. **Already addressed in P17 Limitations:** Spatial autocorrelation paragraph was added in this session. E187 provides the definitive numbers.

## Caveats

1. N=66 (Java dated inscriptions only) — small for ML spatial regression
2. KNN k=5 weights — results may vary with different weight specifications
3. 2 disconnected components in weights matrix (Sumatra outliers removed, but some Java inscriptions far from others)
4. Dependent variable (century) is ordinal, not truly continuous — spatial regression assumes continuous DV
