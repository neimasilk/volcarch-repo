# E068: FDR Meta-Analytic Audit

**Status:** SUCCESS
**Date:** 2026-03-13
**Type:** META / QUALITY CONTROL
**Papers:** ALL

## Hypothesis

If VOLCARCH's 67 experiments contain multiple comparisons without correction, some "significant" findings may be false positives. This audit applies Benjamini-Hochberg FDR correction across all statistical tests.

## Method

1. Extracted all p-values from E001-E067 READMEs (41 tests with explicit p-values)
2. Applied Benjamini-Hochberg FDR correction at alpha=0.05
3. Ranked findings by statistical strength

## Results

### Summary
- **Total tests:** 41
- **Survive FDR:** 30 (73.2%)
- **Fail FDR:** 11 (26.8%)

### Top 10 Strongest Findings (survive any correction)

| Rank | Experiment | Test | p-value |
|------|-----------|------|---------|
| 1 | E066 | Candi equinox alignment (binomial) | 4.9×10⁻¹⁴ |
| 2 | E051 | Yogyakarta court effect (chi²) | 5.1×10⁻¹⁴ |
| 3 | E066 | Candi cardinal alignment (binomial) | 8.6×10⁻¹⁴ |
| 4 | E031 | Candi west-clustering (Rayleigh) | 3.4×10⁻⁸ |
| 5 | E057 | Genre taphonomy pre-Indic (MW) | <10⁻⁶ |
| 6 | E057 | Genre taphonomy organic (MW) | <10⁻⁶ |
| 7 | E065 | Zone A overrepresentation (chi²) | <10⁻⁶ |
| 8 | E065 | Azimuthal clustering (Rayleigh) | <10⁻⁶ |
| 9 | E004 | Site density vs distance (Spearman) | 1.5×10⁻⁵ |
| 10 | E005 | Residuals vs distance (Spearman) | 10⁻⁴ |

### Critical Casualties (marginal findings that FAIL FDR)

| Experiment | Test | p-value | BH threshold | Impact |
|-----------|------|---------|-------------|--------|
| **E048** | Partial correlation (length-controlled) | 0.038 | 0.038 | Genre taphonomy length-control claim weakened |
| **E032** | Chi² eruption monthly uniformity | 0.042 | 0.039 | Pranata Mangsa seasonality claim weakened (P5, P11) |
| **E053** | Fisher exact Java aDNA | 0.047 | 0.040 | aDNA absence claim weakened (P1, P7) |
| **E043** | McNemar Balinese vs Javanese cognacy | 0.064 | 0.041 | P9 core peripheral conservatism claim marginal |
| **E043** | McNemar Malagasy vs Javanese | 0.073 | 0.043 | P9 baseline claim marginal |

### Implications for Papers

- **P1, P2, P7:** Core findings SURVIVE FDR. Safe.
- **P5:** E032 seasonality marginally fails. Downgrade from "significant" to "suggestive."
- **P8:** Core ML findings (AUC) not p-value dependent. Safe.
- **P9:** E043 cognacy comparisons marginal. Need larger sample or replication.
- **P11:** E032 and E048 length-control are casualties. Strengthen with other evidence.

## Conclusion

**73% of findings survive FDR — project is statistically sound but not bulletproof.** The strongest findings (candi orientation, court-center toponymy, zone overrepresentation, genre taphonomy raw effect) are extremely robust (p < 10⁻⁶). Three marginal findings (E032 seasonality, E048 partial correlation, E053 aDNA Fisher) should be reported as "suggestive" rather than "significant" in papers.

**Recommendation:** Build paper arguments on the Top 10, use marginal findings as "supporting" or "suggestive" evidence, never as primary claims.
