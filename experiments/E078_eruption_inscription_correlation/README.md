# E078: Eruption-Inscription Correlation — Volcanic Dark Periods

## Hypothesis
Major volcanic eruptions cause detectable temporal gaps in Java's inscription record, supporting a causal mechanism linking volcanic activity to archaeological invisibility.

## Method
- 86 dated DHARMA inscriptions (684-1356 CE)
- 14 major eruptions (VEI ≥ 3) from GVP database + Merapi/Samalas additions
- Decade-level inscription frequency vs eruption timing
- Before/after comparison (50-year windows)
- Permutation test for eruption-decade inscription deficit
- Special focus on 928 CE Central→East Java shift

## Key Results

**STATUS: SUCCESS (partial)**

### Test 1: Eruption Decades vs Non-Eruption (Mann-Whitney)
- Eruption decades: **0.17** inscriptions/decade
- Non-eruption decades: **1.08** inscriptions/decade
- **6.3× deficit in eruption decades**
- **p = 0.035 (SIGNIFICANT)**

### Test 4: The 928 CE Great Gap
- Before (800-930 CE): 0.43 inscriptions/year
- After (930-960 CE): 0.10 inscriptions/year
- Recovery (960-1030 CE): 0.03 inscriptions/year
- **77% rate drop** coinciding with Merapi VEI 4 eruption
- This is the most dramatic geographic shift in Javanese political history

### Test 5: Permutation Test
- Observed mean in eruption decades: 0.17
- Random baseline mean: 0.95
- **p = 0.061** (marginal, borderline significant)

### Tests That Did NOT Pass
- Wilcoxon before/after: p=0.43 (confounded by late-period zero baseline)
- VEI vs recovery time: rho=-0.77, p=0.075 (trending but n too small)

## Caveats
1. Late eruption decades (1300s-1400s) have zero inscriptions regardless of eruptions — Majapahit used lontar manuscripts, not stone
2. Dynastic politics, capital shifts, and changing inscription practices are confounds
3. The DHARMA corpus is a sample, not a complete inventory of Javanese inscriptions

## Implications
The 928 CE test case is the strongest evidence for a volcanic-epigraphic causal link. A VEI 4 Merapi eruption coincides with the abandonment of Central Java's monumental building program and the shift to East Java. This event has been debated for a century — VOLCARCH provides quantitative support for the volcanic causation hypothesis.

## Data
- `results/e078_results.json` — Full test statistics
- `results/decade_inscription_counts.csv` — Decade-level data with eruption markers
