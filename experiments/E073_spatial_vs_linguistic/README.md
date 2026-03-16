# E073: Spatial vs Linguistic Evidence Meta-Test

## Hypothesis
Volcanic informedness in pre-modern Java is **behavioral/spatial, not lexical**. Evidence of volcanic awareness should be detectable in architectural siting patterns but absent from linguistic markers (vocabulary, toponyms, phonological substrates).

## Method
Meta-analysis combining 9 tests from 6 experiments across two evidence domains:

**Spatial domain (5 tests):**
- E065: Candi volcanic zone overrepresentation (chi-squared)
- E065: Candi western quadrant clustering (Rayleigh)
- E066: Candi equinox alignment (binomial)
- E066: Candi NOT volcano-facing (McNemar)
- ADV-3: Volcanic proximity after survey control (quasi-Poisson LR)

**Linguistic domain (4 tests):**
- E029: Substrate cross-linguistic cognacy (permutation)
- E038: Volcanic vocabulary diversity by proximity (t-test)
- E067: Volcanic toponym proximity correlation (Spearman)
- E067: Volcanic toponym close vs far (Mann-Whitney)

**Statistical approach:**
1. Fisher's combined probability test within each domain
2. Stouffer's Z-method within each domain
3. Mann-Whitney U test for domain asymmetry
4. Vote counting with Fisher's exact test

## Results

| Metric | Spatial | Linguistic |
|--------|---------|------------|
| Tests significant (α=0.05) | 5/5 (100%) | 0/4 (0%) |
| Fisher's combined p | < 1e-30 | 0.606 |
| Stouffer's Z | 10.86 | -1.39 |
| Median -log10(p) | 6.00 | 0.21 |

**Domain asymmetry:**
- Mann-Whitney U = 0.0, p = 0.008 (one-tailed)
- Rank-biserial correlation = 1.0 (perfect separation)
- Fisher's exact OR = ∞, p = 0.008

## Conclusion
**STATUS: SUCCESS**

The asymmetry is striking and statistically significant:
- ALL spatial tests detect volcanic informedness
- NO linguistic test detects volcanic informedness
- The two domains are perfectly separated (r = 1.0)

This supports the thesis that volcanic knowledge in pre-modern Java was **embodied** (expressed through practice — where to build, how to orient) rather than **lexicalized** (expressed through vocabulary or naming). This is consistent with oral tradition, ritual practice, and tacit knowledge transmission.

**Implication for fieldwork:** Archaeological survey (spatial methods) will recover evidence of volcanic awareness; linguistic analysis alone will not.

**Implication for P11:** This directly supports the "candi as proxy" methodology — architectural patterns encode information that linguistic patterns do not.

## Data
- `results/e073_results.json` — Full results with all test statistics
- `results/evidence_table.csv` — Evidence compilation table
