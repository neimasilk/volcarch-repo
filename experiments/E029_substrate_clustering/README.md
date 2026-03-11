# E029: Phonological Clustering of Consensus Substrate Candidates

**Status: COMPLETE (MIXED RESULTS)**
**Date: 2026-03-10**

## Hypothesis

If the 266 consensus substrate candidates (identified by both rule-based E022 and ML E027 methods) represent a coherent pre-Austronesian language layer, they should cluster phonologically into word families. Specifically:
1. Substrate forms for the SAME concept should be phonologically similar across different languages (cross-linguistic cognates)
2. Substrates within the same semantic domain should be phonologically more similar than across domains

## Method

### Data
- 266 consensus substrates from E028 across 6 Sulawesi languages (Tolaki: 121, Makassar: 48, Bugis: 36, Wolio: 36, Toraja-Sadan: 15, Muna: 10)
- Original ABVD forms (210 concepts across 6 languages) for null distribution

### Phonological Distance
- Normalized Levenshtein edit distance on orthographic forms (0 = identical, 1 = maximally different)
- Manual DP implementation (no external dependency)
- Limitation: orthographic, not IPA-based. Captures gross phonological similarity only.

### Clustering
- **Hierarchical agglomerative** (Ward's method) with silhouette-optimized k
- **DBSCAN** with grid search over eps=[0.3-0.5], min_samples=[3-5]

### Cross-Linguistic Cognate Detection
- For 20 concepts appearing as consensus substrates in 3+ languages: compute mean pairwise Levenshtein distance across languages
- Compare to null distribution: 1000 random concept forms across the same languages
- If substrate forms are more similar than random, this suggests shared inheritance

### Semantic-Phonological Correlation
- Within-domain vs between-domain mean distances
- 1000-permutation test

## Results

### 1. Distance Matrix Statistics
| Metric | Value |
|--------|-------|
| Mean distance | 0.8636 |
| Median distance | 0.8750 |
| Std distance | 0.1263 |
| Min distance | 0.0000 |
| Max distance | 1.0000 |

Overall high distances (mean 0.86) indicate most substrate forms are phonologically quite different from each other, as expected for words from different languages and semantic domains.

### 2. Clustering
- **Ward's method**: Optimal k=30, silhouette=0.114 (weak clustering)
- **DBSCAN**: Best config eps=0.3, min_samples=3, silhouette=0.535 but only 4 tiny clusters (94.7% noise)

**Interpretation:** The low silhouette scores indicate no strong phonological clustering among the 266 substrates as a whole. This is expected -- these are words from 6 different languages spanning many semantic domains. The interesting question is whether forms for the SAME concept cluster across languages (see below).

### 3. Cross-Linguistic Cognate Detection (KEY RESULT)

**20 concepts appear as consensus substrates in 3+ languages.**

| Metric | Substrate | Null (random) |
|--------|-----------|---------------|
| Mean cross-ling distance | 0.7693 | 0.6772 +/- 0.226 |
| p-value | 0.569 | -- |

**Overall, substrate forms are NOT more similar across languages than random concept forms.** The substrate mean (0.77) is actually higher than the null mean (0.68), meaning substrates are on average MORE different across languages than regular Austronesian vocabulary.

**However, two numeral concepts are strong outliers:**

| Concept | n_langs | Distance | z-score | Forms |
|---------|---------|----------|---------|-------|
| Fifty | 4 | 0.204 | -2.09 | Mak: limampulo, Tol: limambulo, Tor: limangpulo, Wol: lima pulu |
| Twenty | 4 | 0.292 | -1.70 | Mak: ruampulo, Tol: ruambulo, Tor: duangpulo, Wol: rua pulu |
| One Thousand | 3 | 0.619 | -0.26 | Mak: sisabu, Tol: aso-sou, Tor: sangsabu |

**Fifty** and **Twenty** are clearly cognate across all 4 languages -- they are compound numerals built from inherited Austronesian roots (*lima 'five', *puluq 'ten', *duSa 'two'). These are NOT pre-Austronesian substrate; they are Austronesian innovations that both detection methods flagged as "substrate" because compound numerals have unusual phonological properties (long forms, nasal clusters at morpheme boundaries).

This is an important **methodological insight**: compound numerals are a systematic false positive for substrate detection methods.

### 4. Semantic-Phonological Correlation

| Metric | Value |
|--------|-------|
| Within-domain mean distance | 0.8525 |
| Between-domain mean distance | 0.8673 |
| Difference | 0.0148 |
| Permutation p-value | 0.000 |

**Statistically significant** but the effect size is tiny (0.015). Words within the same semantic domain are slightly more phonologically similar, likely due to shared morphological patterns (e.g., action verbs sharing prefixes like mo'- in Tolaki).

## Conclusions

1. **No evidence for a coherent pre-Austronesian substrate phonology.** The 266 consensus substrates do not cluster into word families that would suggest a unified pre-AN language layer.

2. **Numeral compounds are systematic false positives.** "Fifty" (lima+pulo) and "Twenty" (rua+pulo) are clearly inherited Austronesian compounds, not substrate. Both rule-based and ML methods flag them because their phonological properties (length, nasal clusters) match substrate heuristics. Recommendation: exclude compound numerals from substrate candidate lists.

3. **Weak semantic-phonological clustering** reflects language-specific morphology (Tolaki mo'- prefix on verbs, Makassar an-/am- prefixes) rather than substrate inheritance.

4. **The consensus substrates likely represent independent lexical innovations** in each language rather than a shared pre-Austronesian vocabulary. This is consistent with the "lexical gap" interpretation: these are words where each language independently innovated away from the proto-Austronesian form, rather than inheriting from a common substrate.

5. **Informative negative result for P8.** The linguistic fossils paper should frame substrates as language-specific innovations with possible areal/contact influences, not as evidence for a single pre-Austronesian language.

## Output Files

- `results/clustering_summary.json` -- Full statistics
- `results/cross_linguistic_cognates.csv` -- Per-concept cross-linguistic distances
- `results/dendrogram.png` -- Ward clustering dendrogram (truncated to 50 leaves)
- `results/cluster_heatmap.png` -- 266x266 distance matrix with cluster boundaries
- `results/cross_ling_distance_histogram.png` -- Substrate vs null distribution

## Dependencies

- Python 3.10+
- numpy, scipy, scikit-learn, matplotlib (all standard Anaconda)
- No external Levenshtein library required (manual DP implementation)
