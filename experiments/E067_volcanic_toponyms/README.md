# E067: Volcanic Toponyms — Do Volcanic Place Names Cluster Near Volcanoes?

**Status:** INFORMATIVE NEGATIVE
**Date:** 2026-03-12
**Channel:** Ch1 (Geology), Ch9 (Archaeoastronomy/landscape)
**Papers served:** P11 (volcanic informedness — constrains scope of the claim)

## Hypothesis

If volcanic informedness extends to place naming, kabupaten closer to active volcanoes should have higher proportions of volcanic morphemes (gunung, kawah, lahar, watu, etc.) in their village names.

## Method

- 25,244 Java village names from E051 dataset (BPS/cahyadsn/wilayah)
- 22 volcanic morphemes classified into 3 tiers:
  - Tier 1 (directly volcanic): kawah, lahar, gumuk
  - Tier 2 (volcanic landscape): gunung, watu, batu, pasir, segara, tlogo, sendang
  - Tier 3 (potentially volcanic): api, panas, belerang, gede, agung, volcano names
- Aggregated to 110 kabupaten, correlated with distance to nearest volcano
- Spearman correlation, Mann-Whitney U, Kruskal-Wallis zone analysis

## Key Results

| Metric | Value | p-value |
|--------|-------|---------|
| Villages with ANY volcanic morpheme | 1,073/25,244 (4.3%) | — |
| Spearman (all morphemes vs distance) | rho = +0.140 | 0.146 |
| Spearman (Tier 2 only vs distance) | rho = +0.095 | 0.324 |
| Mann-Whitney (close vs far) | close 3.9% vs far 4.0% | 0.734 |
| Kruskal-Wallis (3 zones) | H = 7.75 | 0.021 |

Zone analysis (non-monotonic):
- Near (<25km): 4.5%
- Mid (25-50km): 3.3%
- Far (>50km): 4.7%

The KW test is significant but the pattern is **non-monotonic** — not consistent with a proximity effect. The mid-zone is lowest, while both near and far zones are similar.

### Top morphemes (false positive risk)
- **agung** (253 villages, 1.0%) — means "great/noble" in Javanese, NOT necessarily volcanic
- **gunung** (210, 0.8%) — means "mountain," most common genuinely volcanic term
- **pasir** (150, 0.6%) — means "sand," not specifically volcanic

## Conclusion

**INFORMATIVE NEGATIVE.** Volcanic morphemes in Java village names are NOT concentrated near active volcanoes (Spearman rho=+0.140, p=0.146). Several factors explain this:

1. **Lexical ubiquity:** Java's volcanic landscape is so pervasive that volcanic terms (gunung, batu, watu) are part of the general Javanese lexicon regardless of proximity
2. **Semantic broadening:** Terms like "agung" (great), "gede" (big), "pasir" (sand) have generalized beyond volcanic meaning
3. **Tier 1 scarcity:** Only 19/25,244 villages (0.08%) have unambiguously volcanic terms (kawah, lahar, gumuk)

**For P11:** This constrains the volcanic informedness claim further — it operates through **spatial practices** (architecture siting, calendar timing) not through **linguistic marking** of the landscape. Volcanic informedness is behavioral, not lexical. This is consistent with the paper's overall argument that VI is encoded in practice, not in explicit knowledge.

## Files
- `analyze.py` — Analysis script
- `results/e067_results.json` — Machine-readable results
- `results/volcanic_toponyms.png/.pdf` — Scatterplot + zone boxplot
- `results/morpheme_frequency.png` — Bar chart of morpheme counts
