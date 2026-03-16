# E085: ADV-4 Substrate Noise Permutation Test

**Status: SUCCESS — VOLCARCH L4 SUPPORTED**

**Type:** Adversarial test (ADV-4)
**Date:** 2026-03-13
**Pass criterion:** Empirical p < 0.05 (observed AUC in top 5% of permuted distribution)

## Hypothesis

The XGBoost substrate detection from E027 (AUC = 0.762) reflects genuine phonological differences between substrate and non-substrate words, not statistical noise or artifacts of the classification pipeline.

**Adversarial framing:** If randomly shuffled labels produce similar AUC values, then the "substrate detection" is meaningless and L4 (Cosmological Overwrite) loses its computational evidence base.

## Method

Four-test battery:

1. **Label Permutation Test (1000 iterations):** Shuffle substrate/non-substrate labels, train RandomForest each time, record AUC. If observed AUC falls within the permuted distribution → noise.

2. **Random Feature Baseline (100 iterations):** Replace real phonological features with random numbers, keep real labels. If random features achieve similar AUC → features are irrelevant.

3. **Frequency-Only Baseline:** Train using only word form length (no phonological features). If AUC is similar → phonological signal is just a length proxy.

4. **Circularity Check:** Test whether `language_cognacy_coverage` feature (which correlates with the labeling process) drives the result. Compare full model vs model without this feature.

## Data

- E027 feature matrix: `experiments/E027_ml_substrate_detection/data/features_matrix.csv`
- 1,357 lexical forms from 6 Sulawesi languages (Muna, Bugis, Makassar, Wolio, Toraja-Sadan, Tolaki)
- 919 Austronesian (cognate), 438 candidate substrate (residual)
- Model B features: 27 columns (10 phonological + 8 initial-char one-hot + 1 core-vocab + 7 semantic-domain one-hot + 2 language control)
- No distributional features (which would be circular — E027 Model A)
- Original E027 AUC: 0.7599 (XGBoost), 0.7618 (RandomForest), both 10-seed x 5-fold CV

## Results

### Test 1: Label Permutation — **PASS (p = 0.0000)**
| Metric | Value |
|--------|-------|
| Observed AUC | 0.762 |
| Permuted mean | 0.500 |
| Permuted std | 0.024 |
| Permuted 95th percentile | 0.539 |
| Permuted max (of 1000) | 0.584 |
| Z-score | **11.05** |
| Empirical p-value | **0.0000** (0/1000 permutations ≥ observed) |

The observed AUC is **11.1 standard deviations** above the permuted mean. Not a single random permutation out of 1000 came within 0.18 AUC of the observed value.

### Test 2: Random Features — **PASS (p = 0.0000)**
| Metric | Value |
|--------|-------|
| Random features mean AUC | 0.494 |
| Random features max | 0.559 |
| Z-score | **11.52** |

Real phonological features vastly outperform random noise features.

### Test 3: Frequency-Only Lift — **PASS (lift = +0.128)**
| Metric | Value |
|--------|-------|
| Full model AUC | 0.762 |
| Form length only (RF) | 0.634 |
| Form length only (LR) | 0.639 |
| AUC lift from phonological features | **+0.128** |

Word length alone achieves AUC 0.634 (substrate words tend to be shorter — expected for native vocabulary). Phonological features add +0.128 AUC on top, confirming the signal is not just a length proxy.

### Test 4: Circularity Check — **CLEAN**
| Metric | Value |
|--------|-------|
| Full model AUC | 0.762 |
| Without language_cognacy_coverage | 0.759 |
| language_cognacy_coverage alone | 0.680 |
| AUC loss from removing lcov | -0.003 |

Removing the potentially circular feature (`language_cognacy_coverage`) barely changes the AUC (0.762 → 0.759). The signal is carried by genuinely phonological features, not by metadata artifacts.

## Conclusion

**The substrate detection is NOT noise.** The observed AUC of 0.762 is:
- 11.1 standard deviations above the permuted null (p = 0.0000)
- 0.268 above random feature baseline
- 0.128 above word-length-only baseline
- Robust to removal of potentially circular features (AUC drops only 0.003)

This means L4 (Cosmological Overwrite) has genuine computational evidence: there ARE systematic phonological differences between words classified as Austronesian substrate and those classified as Sanskrit-influenced vocabulary. The ML classifier detects a real pattern in the data.

### Caveats

1. **Label quality:** The permutation test validates that *features discriminate the labels*, but does not validate the labels themselves. If E022's residual classification is systematically wrong (e.g., some Austronesian words misclassified as substrate due to missing cognacy data), then the "signal" could reflect data quality patterns rather than genuine substrate.

2. **Moderate effect size:** AUC=0.76 is moderate. The phonological fingerprint is *detectable* but not *definitive*. Individual word classifications should be treated as probabilistic.

3. **Form length dominance:** The frequency-only baseline shows form_length alone achieves AUC=0.634, accounting for ~51% of the full model's AUC above chance (0.134 of 0.262). Form length is the single strongest signal. The phonological features add genuine lift (+0.128) but should be interpreted as refining, not replacing, a length-based heuristic.

4. **Orthographic limitation:** Features are computed from orthographic forms, not IPA transcriptions. Language-specific spelling conventions could introduce systematic biases.

5. **The signal is real but moderate.** The phonological boundary between substrate and non-substrate is fuzzy (expected — centuries of contact would blur any boundary). Consistent with a gradual overwrite process rather than a sharp division.

## Adversarial Scorecard Update

| Test | Target | Result |
|------|--------|--------|
| ADV-1 (Japan comparanda) | L1 | TODO |
| ADV-2 (Non-volcanic control) | L1 | INCONCLUSIVE (p=0.760, N too small) |
| ADV-3 (Survey intensity) | L1 | **PASSED** (p=0.0015) |
| ADV-4 (Substrate noise) | L4 | **PASSED** (p=0.0000, z=11.05) |

## Files

- `adv4_substrate_noise.py` — Analysis script (4-test battery)
- `results/adv4_results.json` — Detailed results (permutation distributions)
- `results/adv4_summary.json` — Machine-readable summary
