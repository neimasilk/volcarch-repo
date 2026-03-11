# E027: ML-Based Linguistic Substrate Detection — Execution Plan

**Date:** 2026-03-10
**Paper:** P8 (Linguistic Fossils)
**Status:** PLANNED

## Hypothesis

An ML classifier trained on phonological and distributional features can distinguish inherited Austronesian vocabulary from potential pre-Austronesian substrate in Sulawesi languages, providing probability scores and interpretable feature importances that the rule-based E022 approach cannot.

## Data

- Source: `experiments/E022_linguistic_subtraction/data/abvd/cldf/`
  - `forms.csv` — 346,662 lexical forms (filter to 6 target languages → ~1,357 forms)
  - `cognates.csv` — 291,675 cognate set memberships
  - `parameters.csv` — concept names
- E022 residuals: `experiments/E022_linguistic_subtraction/results/poc_residuals_detail.csv`
- 6 target languages: Bare'e, Tae', Mori, Bungku, Tolaki, Wolio

## Labels

- **Positive (Austronesian):** 919 forms with ABVD cognacy assignments
- **Candidate Substrate:** 438 forms classified as residual by E022 enhanced pipeline
- This is a Positive-Unlabeled problem; treat E022 labels as noisy ground truth

## Two Model Variants (CRITICAL)

### Model A — Full Features (ranking tool)
- All 18 features including cognate set size
- Purpose: rank substrate candidates by confidence
- Expected AUC: high (>0.90) but CIRCULAR — cognate_set_size ≈ label

### Model B — Phonological-Only (scientific claim)
- 10 phonological + 2 semantic + 2 language features (NO distributional)
- Purpose: test if substrate has a detectable phonological "fingerprint"
- Expected AUC: lower but MEANINGFUL
- **THIS IS THE REAL EXPERIMENT**

## Features

### Phonological (10)
1. `form_length` — character count
2. `n_vowels` — vowel count (proxy for syllable count)
3. `vowel_ratio` — vowels / total chars
4. `ends_in_vowel` — binary, Austronesian canonical = yes
5. `initial_char` — one-hot: m, a, b, t, k, p, s, other
6. `has_glottal` — binary, contains ʔ or '
7. `has_nasal_cluster` — binary, contains ng/mb/nd/nj/mp/nk
8. `has_reduplication` — binary, contains hyphen or repeated pattern
9. `n_consonant_clusters` — count of CC+ sequences
10. `has_prefix_like` — binary, starts with ma-/me-/mo-/pa-/ka-/ta-/na-/po-

### Distributional (4) — Model A only
11. `max_cognate_set_size` — size of largest cognate set (0 for residuals)
12. `n_cognate_sets` — number of cognate sets assigned
13. `concept_residual_rate` — fraction of 6 langs with this concept as residual
14. `concept_cross_lang_count` — how many langs share this concept as residual

### Semantic (2)
15. `semantic_domain` — BODY/NATURE/ACTION/QUALITY/GRAMMAR/NUMBER/OTHER
16. `is_core_vocab` — binary, in Swadesh-100

### Language Control (2)
17. `language_id` — label encoded
18. `language_cognacy_coverage` — ABVD coverage rate for this language

## Models

1. **XGBoost** (primary) — n_estimators=300, max_depth=4, lr=0.05, reg_lambda=1.0
2. **Random Forest** (baseline) — n_estimators=500, min_samples_leaf=5
3. **Logistic Regression** (interpretability check) — L2 penalty, balanced class weights

## Validation

1. **Stratified 5-fold CV** (×10 seeds) — primary metric: AUC-ROC
2. **Leave-One-Language-Out (LOLO)** — 6 folds, tests cross-linguistic generalization
3. **E022 comparison** — do 8 Tier 1 candidates get high substrate probability?
4. **Semantic coherence** — do top-50 ML substrates cluster in expected domains?

## GO/NO-GO Criteria

### GO (AUC ≥ 0.75 Model B, LOLO ≥ 0.65 for 4/6 langs)
→ Proceed to P8 paper integration. ML substrate detection is a viable method.

### CONDITIONAL GO (AUC 0.65-0.75 Model B)
→ ML validates E022 but adds limited new insight. Report as "ML confirmation."

### NO-GO (AUC < 0.65 Model B)
→ Phonological features cannot distinguish substrate. Document as informative negative.

## Execution Steps

### Script 1: `00_prepare_features.py`
- Load ABVD CLDF forms for 6 languages
- Join with cognates, parameters
- Compute all 18 features
- Assign labels from E022 pipeline
- Output: `data/features_matrix.csv`

### Script 2: `01_train_and_evaluate.py`
- Load feature matrix
- Train Model A (full) and Model B (phon-only)
- 5-fold CV (×10 seeds) for both
- LOLO validation for both
- Output: `results/cv_results.json`, `results/lolo_results.json`

### Script 3: `02_shap_and_ranking.py`
- SHAP analysis on Model B (the scientific model)
- SHAP beeswarm plot
- Rank all 438 residuals by substrate probability
- Compare with E022 Tier 1/2/3
- Output: `results/shap_beeswarm.png`, `results/substrate_ranking.csv`

## Dependencies (all installed)
- sklearn 1.2.2, xgboost 3.0.3, shap 0.50.0
- pandas 2.3.3, numpy 1.26.4
- matplotlib 3.10.1

## Risks
1. Circularity → Model B (phon-only) avoids this
2. Small N (1,357) → regularized models, CV
3. Tolaki inflation → sensitivity analysis ±Tolaki
4. Orthographic not IPA → language_id as control
5. Label noise → frame as ranking, not classification
