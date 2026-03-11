# E038 — Volcanic Vocabulary Semantic Drift Across Austronesian

**Status:** INFORMATIVE NEGATIVE (with important confound)
**Date:** 2026-03-10
**Idea ID:** I-007

## Hypothesis

Languages in volcanic regions show different cognacy patterns for volcanic-domain vocabulary (fire, ash, smoke, stone, earth) compared to non-volcanic regions:
- **H1:** MORE lexical diversity (more unique terms for frequently-encountered phenomena)
- **H2:** LOWER cognacy rates (local innovation for important concepts)
- **H3:** Retention of pre-Austronesian substrate terms

## Method

- 1,330 Austronesian languages from ABVD with coordinates
- Classified as VOLCANIC (< 200km from active volcano, n=490) or NON-VOLCANIC (n=840)
- 30 concepts across 3 domains: volcanic (11), control/body (11), environment (8)
- Metrics: cognate set diversity (unique cognate sets / n languages), form length, unique form ratio
- Distance-based analysis: Spearman correlation between volcano distance and cognate conservatism

## Key Results

### H1+H2: NO significant diversity difference

| Domain | Mean diversity diff | t | p |
|--------|-------------------|---|---|
| Volcanic concepts | -0.004 | -0.420 | 0.68 |
| Control (body) | +0.007 | 1.141 | 0.32 |
| Environment | -0.001 | -0.158 | 0.88 |

Mann-Whitney U (volcanic vs control): p=0.38

**No domain shows significantly different cognate diversity between volcanic and non-volcanic regions.** The hypothesis that volcanic proximity drives lexical innovation for fire/ash/smoke is NOT supported.

### Distance correlation: SIGNIFICANT but CONFOUNDED

| Metric | Value |
|--------|-------|
| Languages with 3+ volcanic concepts | 1,277 |
| Spearman rho (distance vs conservatism) | -0.301 |
| p-value | < 0.0001 |

| Distance band | n | Mean conservatism |
|--------------|---|-------------------|
| < 100 km | 326 | 0.495 |
| 100-200 km | 159 | 0.386 |
| 200-500 km | 391 | 0.398 |
| 500-1000 km | 326 | 0.311 |
| > 1000 km | 71 | 0.371 |

**Languages CLOSER to volcanoes are MORE conservative** (retain dominant cognate sets more). But this is almost certainly a **phylogenetic confound**: volcanic regions (Indonesia, Philippines, Melanesia) are the Austronesian heartland where Western Malayo-Polynesian languages — the most conservative major branch — dominate. Distant languages (Polynesia, Micronesia, Madagascar) have undergone more change due to migration/isolation, not volcanic absence.

### Deep Dive: Core volcanic vocabulary is EXTREMELY conservative

| Concept | PAn reconstruction | Dominant form | Distribution |
|---------|-------------------|---------------|-------------|
| fire | *hapuy | api / apuy / afi | Universal, 5000+ years stable |
| ashes | *qabu | abu / awu | Universal |
| stone | *batu | batu / watu / vatu | Universal |
| earth | *taneq | tano / tana / tanah | Universal |
| smoke | *qasu | asu / ahu / asap | Universal |

These PAn reconstructions are retained across virtually all Austronesian languages. The vocabulary is so basic and universal that it CANNOT diverge — it's core vocabulary (Swadesh-level stability).

### H3: No substrate evidence

No clear substrate signal in volcanic vocabulary. The forms "api", "abu", "batu", "tana" are PAn inheritances across the board. If pre-Austronesian substrate languages contributed volcanic vocabulary, it is not detectable at this level of analysis.

## Interpretation

**Informative negative.** Volcanic vocabulary is too STABLE to show drift — these are core concepts (fire, ash, stone, earth) retained since Proto-Austronesian (~3000 BCE). Unlike specialized vocabulary that might evolve under environmental pressure, these basic terms are learned early, used universally, and resistant to replacement.

The distance correlation is an artifact of Austronesian phylogeography, not volcanic influence. A proper test would require phylogenetic comparative methods (PGLS/BayesTraits) controlling for genetic relatedness.

## Limitations

1. **Phylogenetic confound** — volcanic proximity correlates with being in the Indonesian/Philippine heartland, which is also the center of Austronesian conservatism
2. **No "mountain" concept in ABVD** — a critical gap for volcanic vocabulary analysis
3. **200km threshold arbitrary** — results may change with different thresholds
4. **Core vocabulary only** — ABVD's 210-concept list is biased toward stable core vocabulary. Specialized volcanic terms (lahar, kawah, magma-equivalents) are not in the database
5. **No phylogenetic control** — the significant distance correlation cannot be interpreted without controlling for genetic relatedness

## What Would Improve This

1. Phylogenetic comparative methods (PGLS) controlling for language family branch
2. Specialized volcanic vocabulary beyond ABVD core list
3. Ethnolinguistic fieldwork on volcanic terminology in Java, Bali, Vanuatu, Hawaii
4. Compare within single language family (e.g., Javanese dialects near vs far from volcanoes)

## Files

- `00_volcanic_vocabulary_drift.py` — Analysis script
- `results/vocabulary_drift_summary.json` — Structured metrics
- `results/concept_metrics.csv` — Per-concept diversity metrics
- `results/volcanic_vocab_4panel.png` — Visualization

## Cross-Paper Implications

- **P8 (Linguistic Fossils):** Core Austronesian vocabulary (fire/stone/earth) is extremely stable — substrate detection needs to focus on NON-core vocabulary (as E027 already does).
- **P11 (VCS):** Volcanic vocabulary does not diverge by volcanic proximity at the Austronesian-wide scale. VCS effects, if they exist, must be culturally mediated (ritual, practice), not lexically mediated.
- **Future:** A within-Java comparison (highland vs lowland Javanese dialects) could detect local volcanic vocabulary innovation not visible at pan-Austronesian scale.

## Conclusion

**INFORMATIVE NEGATIVE.** Volcanic vocabulary (fire, ash, smoke, stone, earth) shows no significant cognacy difference between volcanic and non-volcanic Austronesian regions. Core vocabulary is too stable for environmental pressure to create detectable drift at this scale. The significant distance-conservatism correlation (rho=-0.301) is a phylogenetic confound, not a volcanic effect. Substrate detection must focus on non-core vocabulary (as P8 already does).
