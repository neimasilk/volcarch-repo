# E154: Comprehensive FDR Re-Audit at 153 Experiments

**Status:** SUCCESS
**Date:** 2026-03-31
**Type:** [R] META / QUALITY CONTROL
**Papers:** ALL
**Supersedes:** E068 (partial — E068 remains valid for its original scope)

## Hypothesis

E068 audited 41 statistical tests at 90 experiments. With 153 experiments and 83 identifiable statistical tests, the Benjamini-Hochberg correction thresholds have shifted. Some previously marginal findings may change status.

## Method

1. Combined E068's 42 original tests with 41 new tests from E069-E153
2. Applied Benjamini-Hochberg FDR correction at alpha=0.05 across all 83 tests
3. Compared survival rates and identified status changes

## Results

### Summary

| Metric | E068 | E154 | Change |
|--------|------|------|--------|
| Total tests | 42 | 83 | +41 |
| Survive BH | 30 (73.2%) | 65 (78.3%) | **+5.1pp** |
| Cathedral (p<10⁻⁴) | 10 | 13 | +3 |
| Solid (10⁻⁴<p<0.01) | — | 42 | — |
| Marginal (0.01<p<0.05) | — | 10 | — |
| FDR casualties | 3 | 2 | **-1** |
| Not significant | — | 16 | — |

### Key Changes from E068

1. **E048 RESCUED** — Partial correlation (p=0.038) now survives BH at threshold 0.0392. Previously failed at BH threshold 0.038. The addition of strong new tests raised the BH thresholds enough to rescue this borderline finding.

2. **E032 and E053 STILL FAIL** — Pranata Mangsa seasonality (p=0.042) and Java aDNA Fisher (p=0.047) remain FDR casualties. Continue to report as "suggestive."

3. **New cathedral findings:** E152a (post-929 longitude shift, p=3.89×10⁻¹²), E084 (inscription-volcano MW, p=5.2×10⁻⁸), E085 (substrate noise permutation, p<10⁻⁵) — all in the unassailable tier.

4. **E149d borderline failure** — Volcano distance vs inscriptions (p=0.052) narrowly misses BH threshold (0.041). This is a new marginal casualty. E149's other three tests survive.

### New Cathedral Findings (p < 10⁻⁴, survive BH)

| Rank | Experiment | Test | p-value |
|------|-----------|------|---------|
| 4 | E152a | Post-929 longitude shift | 3.89×10⁻¹² |
| 6 | E084 | Inscription-volcano divergence | 5.2×10⁻⁸ |
| 11 | E085 | Substrate noise permutation | <10⁻⁵ |

### FDR Casualties (only 2 — improved from 3)

| Experiment | Test | p-value | BH threshold | Recommendation |
|-----------|------|---------|-------------|----------------|
| E032 | Pranata Mangsa seasonality chi² | 0.042 | 0.040 | Report as "suggestive" |
| E053 | Java aDNA Fisher exact | 0.047 | 0.040 | Report as "suggestive" |

## Conclusion

**The project's statistical foundation is STRONGER at 153 experiments than at 90.** The new experiments (E069-E153) have higher average statistical power than the originals, raising the overall survival rate. The addition of many strong results (E152, E084, E085, E145, E147) with p-values well below 10⁻⁴ has widened the BH thresholds enough to rescue E048 from FDR casualty status.

**Reporting rules (updated):**
- p < 10⁻⁴ after BH: "highly significant" (13 tests)
- 10⁻⁴ < p < 0.01 after BH: "significant" (42 tests)
- 0.01 < p < 0.05 after BH: "marginally significant" (10 tests)
- p < 0.05 uncorrected but fails BH: "suggestive" with FDR note (2 tests: E032, E053)
- p > 0.05: not significant (16 tests, including informative negatives)

**Net assessment:** 78.3% survival is excellent for a 153-experiment project spanning 4 analytical domains. The core claims (volcanic burial, cosmological overwrite, genre taphonomy, court-center model, post-929 shift) all rest on cathedral findings that survive any reasonable correction.
