# E061: Script Simplification — Cross-Cultural Validation

**Status:** CONDITIONAL SUCCESS

## Hypothesis

All Indic-derived SE Asian scripts simplify from Sanskrit's 33 consonants toward their local phonological inventory. The degree of simplification correlates with geographic/temporal distance from India. This extends E036's finding that Hanacaraka (20 consonants) aligns with Proto-Austronesian (~17), not Sanskrit (33).

## Method

Compiled published data on 10 Brahmi-derived writing systems:
- **Baseline:** Devanagari (33C), Grantha (33C)
- **Mainland SE Asia:** Khmer (33C), Burmese (33C), Thai (44C), Tibetan (30C)
- **Maritime SE Asia (Austronesian):** Balinese (33C), Hanacaraka (20C), Lontara (23C), Baybayin (14C)

For each script: consonant count, vowel count, adoption date, distance from India, language family, local phonological consonant inventory.

### Statistical tests:
- **H1:** Austronesian vs non-Austronesian consonant counts (Mann-Whitney U, one-tailed)
- **H2:** Geographic distance from India vs consonant count (Spearman)
- **H3:** Adoption date vs consonant count (Spearman)
- **H4:** Phonological floor — do scripts stay above local consonant needs?

## Data Sources

- Hanacaraka: E036 (this project), Soemarmo 1995, Uhlenbeck 1978
- Baybayin: Santos 2002, Scott 1984
- Lontara: Noorduyn 1991, Pelras 1996
- Balinese: Fox 1993, Casparis 1975
- Thai: Royal Institute of Thailand standards
- Khmer: Huffman 1970
- Burmese: Okell 1971
- Tibetan: Beyer 1992
- Devanagari: Whitney 1889
- Grantha: Burnell 1878

## Results

### H1: Austronesian scripts simplify more — SUPPORTED
- Austronesian mean: 22.5 consonants vs Non-Austronesian mean: 34.3
- Mann-Whitney U = 3.0, **p = 0.027** (one-tailed)
- Sensitivity (excl. Balinese which retains full Sanskrit set): p = 0.011

### H2: Distance from India correlates with reduction — NOT SIGNIFICANT
- Spearman rho = -0.557, p = 0.119
- Direction correct but N too small for significance
- Thai (44C, expanded for tonal classes) is a major outlier

### H3: Later adoption correlates with reduction — NOT SIGNIFICANT (marginally with sensitivity)
- Spearman rho = -0.557, p = 0.119
- Sensitivity (excl. Thai): rho = -0.736, **p = 0.038**

### H4: Phonological floor — PARTIALLY SUPPORTED
- 9/10 scripts have >= their local consonant inventory
- One violation: Baybayin (14 script consonants < 16 Tagalog phonemes) — some phonemes share graphemes
- Mean excess graphemes: Austronesian +5.0, Non-Austronesian +11.0

### Key qualitative finding: Two adaptation strategies

| Strategy | Scripts | Family | Pattern |
|---|---|---|---|
| Conservative Encoders | Khmer, Burmese, Balinese | Mixed | Retain full 33 |
| Phonological Adapters | Hanacaraka, Lontara, Baybayin | All Austronesian | Reduce to local phonology |
| Tonal Expanders | Thai | Kra-Dai | Expand beyond 33 |

## Conclusion

**Hanacaraka's consonant reduction (33 to 20) is not unique — it is part of a systematic Austronesian pattern.** All three Austronesian scripts that underwent genuine adaptation (not mere copying) reduced their consonant inventories toward the local phonological floor. This contrasts sharply with mainland SE Asian scripts (Khmer, Burmese) that retained the full 33-consonant Sanskrit set regardless of phonological mismatch.

The Austronesian pattern suggests a pragmatic, phonologically-driven approach to script adoption: keep what the language needs, discard what it does not. Mainland scripts instead retained the full Indic inventory, likely due to stronger institutional ties to Sanskrit literary tradition (monastery systems, court Brahmins).

**Baybayin** (14 consonants) represents the extreme endpoint — the only script that dropped BELOW its local phonological inventory, indicating a culture that prioritized simplicity over completeness.

**Thai** is a fascinating counter-case: rather than simplifying, Thai EXPANDED the inventory to encode tonal distinctions through consonant class assignment — a fundamentally different adaptation strategy from either conservation or reduction.

**For P8:** This provides cross-cultural validation that E036's finding (Hanacaraka aligning with PAn rather than Sanskrit) reflects a broader Austronesian linguistic substrate phenomenon, not an isolated Javanese peculiarity.

## Limitations

1. **Small N** (10 scripts): statistical power is limited, especially for H2/H3 correlations.
2. **Consonant counts are debatable**: scholars disagree on whether to count the "full" inventory (including rarely-used characters for Sanskrit loans) or the "functional" native subset. Balinese is the clearest case — 33 in full inventory but only ~18 in native use.
3. **Distance is approximate**: straight-line km, not trade-route distance.
4. **Adoption dates are approximate**: most scripts evolved over centuries, not adopted at a single point.
5. **Confound: language family and distance are correlated** — Austronesian languages happen to be farther from India. H1 and H2 are not fully independent.

## Files

- `analyze.py` — Analysis script
- `results/E061_results.json` — Full statistical results
- `results/fig1_consonant_inventory_comparison.png` — Bar chart: script vs local phonological inventories
- `results/fig2_distance_vs_consonants.png` — Scatter: distance from India vs consonant count
- `results/fig3_script_vs_phonology.png` — Script graphemes vs local phoneme inventory
