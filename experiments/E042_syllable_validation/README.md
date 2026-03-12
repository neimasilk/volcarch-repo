# E042: Syllable Count Validation

**Date:** 2026-03-11
**Status:** SUCCESS — Model robust to syllable vs character length metric
**For:** P8 (Linguistic Fossils) — addresses "character count ≠ phonological metric"

---

## Hypothesis

If Model B relies on character count (an orthographic artifact), then replacing `form_length` (character count) with syllable count (vowel nuclei) should significantly change performance.

## Method

Syllable count = number of vowel nuclei (maximal vowel sequences). Standard approximation in computational phonology.

Four model variants tested:
1. **char_length** — original character count (baseline)
2. **syllable_count** — vowel nuclei count (replacement)
3. **both** — character + syllable count together
4. **no_length** — no length feature at all

## Results

### Descriptive Statistics

| Metric | Austronesian | Non-mainstream | Delta |
|--------|:---:|:---:|:---:|
| Character length | 5.20 | 6.10 | +0.89 |
| **Syllable count** | **2.29** | **2.57** | **+0.29** |

Non-mainstream forms are ~0.3 syllables longer. The signal exists at syllable level.
Correlation between char length and syllable count: r = 0.87.

### Model Performance

| Variant | CV AUC | LOLO mean | LOLO ≥ 0.65 |
|---------|:---:|:---:|:---:|
| char_length | 0.768 ± 0.026 | 0.722 | 6/6 |
| **syllable_count** | **0.769 ± 0.026** | **0.728** | **6/6** |
| both | 0.768 ± 0.026 | 0.717 | 6/6 |
| no_length | 0.769 ± 0.025 | 0.732 | 6/6 |

**All four variants perform identically** (CV delta < 0.001). LOLO is slightly better with syllable count (+0.006) or even without any length feature (+0.010).

### Key Insight: Length is NOT the Primary Signal

The `no_length` variant (no length feature at all) performs as well as any variant with length. This means the phonological fingerprint does NOT depend on the length metric — it is carried by other features (consonant clusters, glottal stops, prefix patterns, semantic domain). Character count vs syllable count is a distinction without a difference for this model.

## Conclusion

**The model is robust to the choice of length metric.** Character count, syllable count, or no length feature at all — performance is identical. The "character count ≠ phonological metric" criticism is empirically refuted: the signal is not driven by length at all.

## Files

- `01_syllable_count.py` — Main experiment
- `results/syllable_validation_summary.json` — Summary statistics
