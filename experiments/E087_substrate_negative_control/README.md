# E087: Substrate Detector Negative Control

**Date:** 2026-03-16
**Paper:** P8 (Linguistic Fossils) / L4 validation
**Status:** GREY ZONE (CONDITIONAL PASS — with major caveat)

## Hypothesis

If the E027 ML substrate detector (XGBoost/RF, AUC=0.762) is genuinely detecting pre-Austronesian substrate in Sulawesi languages, it should **fail** to detect "substrate" in closely related Austronesian language pairs where no substrate exists. If the detector achieves high AUC on any arbitrary language pair, it is detecting generic phylogenetic noise, not substrate.

## Method

### Three Feature Set Variants
Each control was tested with three progressively stripped feature sets:
1. **Full Model B** — phonological + semantic + language_id + language_cognacy_coverage (as in E027)
2. **No-LCOV** — same but without language_cognacy_coverage (known semi-circular feature)
3. **Pure Phonology** — word-level features only, NO language-level features at all

### Control Language Pairs

| Control | Languages | Subgroup | Substrate Expected? | N forms | Residual Rate |
|---------|-----------|----------|--------------------:|--------:|--------------:|
| **Reference** | 6 Sulawesi (Muna, Bugis, Makassar, Wolio, Toraja-Sadan, Tolaki) | South Sulawesi / Bungku-Tolaki / Muna-Buton | **YES** | 1,357 | 32.3% |
| **C1** | Tagalog + Cebuano | Central Philippine | No | 486 | 7.6% |
| **C2** | Malay + Minangkabau | Malayic | No | 456 | 6.6% |
| **C3** | Javanese + Sundanese (random labels) | Western Malayo-Polynesian | N/A | 434 | 32.3% (forced) |
| **C4** | Tagalog + Kapampangan | Central Luzon / Philippine | No | 459 | 9.8% |
| **C5** | Iban + Malay | Malayic | No (but high divergence) | 511 | 14.3% |
| **C6** | Acehnese + Toba Batak | Chamic + Batak | Acehnese: YES (Mon-Khmer) | 469 | 46.7% |

### Labeling
Same as E022/E027: words with ABVD cognacy = Austronesian (label=1); words without cognacy = residual/candidate substrate (label=0).

### Evaluation
Stratified 5-fold CV x 10 random seeds, RandomForest (n_estimators=500, balanced).

## Results

### Grand Comparison Table

| Test | Full Model B | No LCOV | Pure Phonology |
|------|:---:|:---:|:---:|
| **E027 Original (Sulawesi 6)** | **0.761** | **0.766** | **0.727** |
| C1: Tagalog + Cebuano | 0.611 | 0.597 | 0.568 |
| C2: Malay + Minangkabau | 0.690 | 0.691 | 0.674 |
| C3: Javanese + Sundanese (real labels) | 0.651 | 0.647 | 0.641 |
| C4: Tagalog + Kapampangan | 0.685 | 0.681 | 0.623 |
| **C5: Iban + Malay** | **0.794** | **0.788** | **0.713** |
| C6: Acehnese + Toba Batak | — | — | 0.660 |
| C3: Random labels (mean of 200) | 0.500 | — | — |

### Permutation Test (Pure Phonology, Sulawesi)
- Observed AUC: 0.727
- Permuted mean: 0.502
- Z-score: 10.00
- P-value: 0.0000
- **The Sulawesi signal is NOT noise** (10 SD above chance)

### Key Findings

1. **C3 (Random Labels) = CLEAN PASS.** Random labels produce AUC ~0.50. The features are not structurally biased.

2. **C1 (Tagalog + Cebuano) = PASS.** AUC = 0.568 (pure phon), near chance. Closely related, well-documented languages produce minimal false substrate signal.

3. **C2 (Malay + Minangkabau) = MARGINAL.** AUC = 0.674 (pure phon). Higher than ideal. Minangkabau has 10.9% residual rate vs Malay's 2.5%, suggesting the detector partly learns "which language has more missing cognacy data."

4. **C5 (Iban + Malay) = ALARMING.** AUC = 0.713 (pure phon), close to the original Sulawesi 0.727. Iban (24.4% residual) vs Malay (2.5%) — large coverage differential. The detector achieves near-Sulawesi AUC on a pair where NO substrate should exist.

5. **The core problem:** The E022 "residual" labeling method conflates **missing ABVD documentation** with **substrate**. Languages with lower ABVD coverage mechanically produce more "residuals." If those languages also have systematically different phonology (longer words, more consonant clusters — characteristics of less-documented, more peripheral languages), the classifier picks this up as "substrate."

## Diagnosis: Why C5 Is So High

Iban has:
- Lower ABVD cognacy coverage (75.6%) vs Malay (97.5%)
- More forms without cognates (24.4% residual vs 2.5%)
- Systematically different orthographic conventions (Borneo Malayic)
- Different phonological profile (more consonant clusters, different prefixation patterns)

The detector learns: "if a word looks like it's from a less-documented language, call it substrate." This is not wrong per se — less-documented languages DO retain more non-mainstream vocabulary — but it means the AUC=0.762 is partly a **documentation artifact**, not purely a **substrate signal**.

## Gap Analysis

| Comparison | AUC Gap (pure phon) |
|---|---|
| Sulawesi (0.727) - C1 (0.568) | +0.159 |
| Sulawesi (0.727) - C2 (0.674) | +0.053 |
| Sulawesi (0.727) - C5 (0.713) | +0.014 |

- The Sulawesi-C1 gap (+0.159) is substantial: the detector works MUCH better on Sulawesi than on Philippine pairs.
- The Sulawesi-C5 gap (+0.014) is negligible: Iban+Malay produces nearly identical AUC.
- The Sulawesi-C2 gap (+0.053) is small but may be meaningful.

## Verdict: GREY ZONE (CONDITIONAL PASS with major caveat)

### What PASSES:
- The Sulawesi signal is real (p=0.0000, z=10.0 in permutation test)
- The signal is significantly above the best closely-related control (C1: 0.568)
- Random labels produce chance AUC (0.500)
- The phonological fingerprint (longer forms, more consonant clusters, more glottal stops) is a genuine pattern in the data

### What FAILS:
- C5 (Iban+Malay) produces AUC=0.713 with no substrate, nearly matching Sulawesi's 0.727
- C2 (Malay+Minangkabau) produces AUC=0.674, uncomfortably high for a "no substrate" pair
- The detector CANNOT distinguish "genuine pre-Austronesian substrate" from "words missing from ABVD that happen to have non-mainstream phonology"

### Implication for P8:
The E027 ML substrate detection should be presented as detecting **phonological non-conformity** (words that don't match the mainstream Austronesian phonological profile), NOT as definitively detecting **pre-Austronesian substrate**. The method is valid as a **ranking tool** (which words are most phonologically anomalous?), but the AUC=0.762 should not be cited as proof that 32.3% of Sulawesi vocabulary is substrate.

### Honest framing for P8:
> "Our ML classifier identifies a phonological fingerprint in Sulawesi residual vocabulary (AUC=0.727 on pure phonological features, p<0.001 vs. permutation). This signal is stronger than in closely-related Philippine language pairs (C1 AUC=0.568) but comparable to other language pairs with similar ABVD coverage differentials (C5 AUC=0.713). We interpret this as evidence that non-cognate Sulawesi vocabulary has a distinct phonological profile consistent with substrate influence, while acknowledging that ABVD documentation gaps contribute to the signal."

## Files

| File | Description |
|---|---|
| `negative_control.py` | Main control tests (full Model B features) |
| `negative_control_no_lcov.py` | Controls without language_cognacy_coverage |
| `negative_control_pure_phon.py` | Controls with pure word-level phonology only |
| `results/negative_control_results.json` | Full Model B results |
| `results/negative_control_no_lcov.json` | No-LCOV results |
| `results/negative_control_pure_phon.json` | Pure phonology results |
| `results/control3_random_aucs.npy` | Random label AUC distribution (200 iterations) |
| `results/summary.txt` | Text summary of Full Model B run |

## Caveats

1. **Class imbalance:** Control pairs have 7-14% residual rate vs. Sulawesi's 32.3%. This makes the control classification harder, potentially deflating control AUCs. However, C5 achieves 0.713 with only 14.3% residual, suggesting the effect is robust.

2. **Orthographic confound:** ABVD data uses language-specific orthographies, not IPA. Orthographic differences between languages may masquerade as phonological differences.

3. **N is modest:** 486-511 forms per control pair vs. 1,357 for Sulawesi. Smaller N means noisier AUC estimates.

4. **No true positive control:** Ideally we would test on a language pair with KNOWN, independently confirmed substrate (e.g., Malay with confirmed Mon-Khmer substrate). Acehnese+Toba Batak (C6) was attempted but produces only 0.660 AUC — lower than expected if the detector were truly substrate-specific.
