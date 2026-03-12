# E062 — Temporal Synthesis: Multi-Dimensional Visibility Curve

**Status:** CONDITIONAL SUCCESS

## Hypothesis

All evidence dimensions from previous experiments (ritual keywords, botanical references, organic material culture) co-vary with century, creating a single "visibility curve" that captures how indigenous content becomes increasingly visible in later Old Javanese inscriptions. The curve peaks in C10-C11 (height of the sima grant tradition), not C14.

## Method

1. Joined 4 datasets on `filename`:
   - E023 (ritual screening): 268 rows — pre_indic_ratio, has_hyang, has_manhuri, has_wuku
   - E030 (dated inscriptions): 166 rows — century, year_ce (used as temporal base)
   - E035 (botanical): 249 rows — n_plants, has_ritual_context
   - E040 (material culture): 268 rows — n_organic, n_lithic, n_metal

2. For 166 dated inscriptions, computed per-century averages for all dimensions.

3. Tested three hypotheses:
   - H1: Indigenous markers co-vary (Spearman correlation matrix)
   - H2: A single PCA component explains >50% variance
   - H3: The visibility curve peaks C10-C11

## Data

- Input: 166 dated Old Javanese inscriptions (C6-C14) from DHARMA corpus
- Century distribution: C8 (n=55), C10 (n=47), C9 (n=30), C11 (n=11), C13 (n=10), C14 (n=6), C7 (n=4), C12 (n=2), C6 (n=1)

## Results

### H1: Co-variation — SUPPORTED

7/10 indigenous marker pairs show significant positive Spearman correlation (p < 0.05). Key correlations:
- pre_indic_ratio x has_hyang: rho = +0.902 (p < 0.0001) — near-perfect
- n_organic x word_count: rho = +0.788 (p < 0.0001)
- n_plants x word_count: rho = +0.706 (p < 0.0001)

Exception: has_wuku does not correlate with most other markers (rare feature, only 7 inscriptions).

### H2: Single visibility component — SUPPORTED

PCA PC1 explains 51.3% of total variance (threshold: >50%). All loadings positive:
- Word Count: +0.483
- N Organic: +0.479
- Has Hyang: +0.472
- N Plants: +0.468
- Pre-Indic Ratio: +0.305
- Has Wuku: +0.052

PC2 (19.5%) separates has_wuku from the main cluster. PC1+PC2 = 70.8% cumulative.

### H3: Peak C10-C11 — PARTIAL (peak at C13)

All six dimensions show significant positive temporal trend (rho = +0.25 to +0.67, all p < 0.01). The composite visibility score (PC1 mean) by century:

| Century | n   | Visibility Score |
|---------|-----|-----------------|
| C6      | 1   | -1.517          |
| C7      | 4   | -0.078          |
| C8      | 55  | -1.485          |
| C9      | 30  | +0.001          |
| C10     | 47  | +1.084          |
| C11     | 11  | +1.390          |
| C12     | 2   | +0.826          |
| C13     | 10  | +1.476          |
| C14     | 6   | +0.136          |

Peak is at C13 (score = +1.476), narrowly above C11 (+1.390). The C13 peak is driven by high pre_indic_ratio (0.369) despite moderate word counts. C14 drops sharply — consistent with the Indianization wave decline from E033.

### Supplementary: Organic/Lithic Ratio

Organic ratio increases C6-C10 (0.50 to 0.86), plateaus, then drops at C14 (0.63). Not significant over time (rho = +0.054, p = 0.595) — organic mentions are ubiquitous across all centuries.

## Key Insight

The visibility curve is substantially a **genre effect**: the shift from short Sanskrit-style dedications (C8 avg 17 words) to long Old Javanese sima grants (C10-C11 avg 644-830 words) mechanically increases all markers. Word count loads highest on PC1 (+0.483). This does not invalidate the finding — the genre shift IS the indigenous visibility phenomenon — but it means "visibility" conflates two things: (1) genuine increase in indigenous content proportion, and (2) longer inscriptions simply having more room for everything.

The pre_indic_ratio, which controls for inscription length, shows the cleanest signal: monotonic rise from C8 (0.005) to C13 (0.369), confirming that indigenous content genuinely increases as a proportion, not just in absolute count.

## Figures

1. `fig1_temporal_multipanel.png` — Six-panel temporal profile (all dimensions by century)
2. `fig2_correlation_heatmap.png` — Spearman correlation heatmap (7 dimensions)
3. `fig3_pca_biplot.png` — PCA biplot colored by century
4. `fig4_visibility_curve.png` — The composite visibility curve (PC1 by century)

## Conclusion

There IS a single "visibility axis" in Old Javanese inscriptions: PC1 captures 51.3% of variance across ritual, botanical, and material dimensions. Indigenous content systematically increases from C8 to C11-C13, then drops at C14. The peak is C13 rather than the hypothesized C10-C11, driven by high pre_indic_ratio in late Majapahit-era inscriptions. This is consistent with E033's finding that Indianization is a wave that recedes — the highest indigenous visibility coincides with the declining phase of Sanskrit influence.

The dominant driver is inscription length (word_count loading = +0.483), meaning the visibility curve largely reflects the shift from terse Sanskrit dedications to expansive Old Javanese sima charters. This is a taphonomic finding itself: **what we can see of indigenous Indonesia depends on how much room the inscription format allows**.

## Files

- `analyze.py` — Analysis script
- `results/joined_dated_inscriptions.csv` — 166-row joined dataset
- `results/century_averages.csv` — Per-century summary statistics
- `results/fig1_temporal_multipanel.png` — Multi-panel temporal profile
- `results/fig2_correlation_heatmap.png` — Correlation heatmap
- `results/fig3_pca_biplot.png` — PCA biplot
- `results/fig4_visibility_curve.png` — Visibility curve
