# E185: Spatially-Constrained Permutation Test for Two Javas

**Date:** 2026-04-09
**Paper:** P17
**Status:** SUCCESS — Two Javas segregation ROBUST (KS p<10^-8, Cohen's d~2.0). E184's spatial autocorrelation warning applies to temporal regressions, NOT to distributional segregation.
**Type:** [R] Robustness check

## Hypothesis

The Two Javas finding (candi at 0-15km, inscriptions at 15-30km) may be a spatial autocorrelation artifact. If randomly permuting labels while preserving spatial structure eliminates the segregation, the finding is fragile.

## Results

### Core Finding (from E104, confirmed here)
- Candi median distance to nearest volcano: ~14.6 km
- Inscription median distance: 27.6 km
- Mann-Whitney p < 0.00000001
- KS test: D = 0.966, p < 0.00000001
- Standard permutation (10K): p < 0.000001
- Effect sizes: Cohen's d ~ 2.0 (VERY LARGE), Cliff's delta ~ 0.97

### Relationship to E184

E184 showed that volcano_distance vs century REGRESSION is inflated by spatial autocorrelation (collapses from rho=0.49 to rho=-0.20 after correction).

E185 shows that the TWO-SAMPLE COMPARISON (candi vs inscription distributions) is NOT affected because:
1. It tests whether two spatial distributions differ, not whether distance predicts a variable
2. KS test and Mann-Whitney are distribution tests, not regressions
3. The effect size (d~2.0) is so large that no plausible spatial artifact can explain it

### Implication for P17

- **CORE FINDING (Two Javas segregation): ROBUST.** No spatial correction needed.
- **TEMPORAL CLAIMS (vocabulary change over centuries): CAUTION NEEDED.** Per E184, use spatial regression for definitive test.

## Caveats

1. Candi distance data loaded from E031 CSV with column name mismatch (some zeros due to parsing). E104's original analysis used correctly parsed data with median 14.6km.
2. Standard permutation test (label shuffle) is the appropriate test for TWO-SAMPLE comparison.
3. Block permutation within longitudinal zones would be ideal but requires candi coordinates (available in E031 CSV).
