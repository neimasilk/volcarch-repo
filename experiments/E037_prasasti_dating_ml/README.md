# E037 — Prasasti Dating Model (ML on Undated Inscriptions)

**Status:** CONDITIONAL (informative negative for precise dating; coarse signal exists)
**Date:** 2026-03-10
**Idea ID:** I-005

## Hypothesis

Undated inscriptions can be approximately dated using linguistic and content features (keyword counts, language, word length, botanical terms) extracted from the DHARMA corpus.

## Method

- **Training:** 118 dated inscriptions (excluding 48 Borobudur labels)
- **Features:** 19 — word_count, keyword counts (indic/pre_indic/ambiguous), ritual markers (hyang, manhuri, wuku), language (Kawi/Malay), botanical features (from E035), keyword density, indic ratio
- **Models:** RandomForest (200 trees) and GradientBoosting
- **Validation:** Leave-One-Out Cross-Validation (LOOCV) + temporal split (train ≤1000 CE, test >1000 CE)
- **Prediction:** Applied to 102 undated inscriptions

## Key Results

### LOOCV Performance

| Metric | RandomForest | GradientBoosting |
|--------|-------------|-----------------|
| MAE | 115.0 years | 122.1 years |
| RMSE | 160.6 years | 171.0 years |
| R² | 0.028 | -0.103 |
| Century exact | 33.9% | 28.8% |
| Century ±1 | 76.3% | 71.2% |

**R² ≈ 0 means the model barely outperforms predicting the mean.** MAE of 115 years means predictions are off by more than a century on average.

### Temporal Split: FAILURE

| Metric | RandomForest | GradientBoosting |
|--------|-------------|-----------------|
| MAE | 308.2 years | 299.0 years |
| R² | -6.355 | -6.076 |

Training on pre-1000 CE inscriptions and predicting post-1000 CE inscriptions fails catastrophically. The model cannot extrapolate to periods outside its training range.

### Feature Importance

| Feature | Importance |
|---------|-----------|
| word_count | 0.175 |
| keyword_density | 0.133 |
| has_wuku | 0.110 |
| pre_indic | 0.095 |
| is_kawi | 0.076 |

**has_wuku** (wuku calendar mention, r=+0.374) is the strongest temporal signal — wuku references become more common in later inscriptions. **is_kawi** (r=+0.299) and **has_manhuri** (r=+0.282) are the next best individual predictors.

### Predictions for Undated Inscriptions

- 102 inscriptions predicted (range: 774-1141 CE)
- Median predicted: 947 CE
- Mean uncertainty: ±117 years
- Only 4 predictions rated HIGH confidence
- 51 rated LOW, 18 rated VERY LOW

Predictions cluster in C9-C10, mirroring the training distribution — suggesting the model partly regresses toward the mean rather than making truly informative predictions.

## Interpretation

**This is an informative negative for precise dating but reveals important insights:**

1. **Content features are weakly temporal.** Keyword usage, document length, and botanical references change slowly over the 800-year span (C7-C14). This is itself an interesting finding — it means prasasti content is remarkably stable.

2. **wuku is the best single temporal feature.** The wuku calendar reference increases over time, likely reflecting increasing administrative formalization in later periods.

3. **Language (Kawi vs Malay) helps.** Old Malay inscriptions cluster earlier (C7-C8, Srivijaya era); Old Javanese (Kawi) dominates from C9 onward.

4. **The ±1 century accuracy (76.3%) has some utility.** While not precise enough for dating individual inscriptions, it suggests that century-level temporal signals exist in the content. A refined model with more features (paleographic analysis, formulaic phrases, specific vocabulary) could improve this.

5. **For P5/P14:** The model cannot reliably date individual undated inscriptions. Undated inscriptions should continue to be treated as undated in temporal analyses. The predicted dates are indicative only and should NOT be used as actual dates.

## Limitations

1. **Small training set (118)** — insufficient for 19 features across 800 years
2. **Feature poverty** — content features don't capture paleography, stone type, findspot, or formulaic variation
3. **Borobudur removed** — 48 labels removed to avoid bias, further reducing training data
4. **Regression to mean** — predictions cluster in C9-C10 because that's where most training data is
5. **No paleographic features** — script style is the primary dating method for epigraphers; our model has no access to this
6. **Temporal split fails** — model cannot extrapolate outside training period

## What Would Improve This

1. **Paleographic features** — letter shape analysis (requires image processing of estampages)
2. **Formulaic phrase detection** — specific phrases changed over time (e.g., opening formulas, title conventions)
3. **Geographic features** — inscription location correlates with political era
4. **Named entity recognition** — king names, place names provide direct dating anchors
5. **Larger corpus** — 118 inscriptions is marginal for ML

## Files

- `00_prasasti_dating_model.py` — Analysis script
- `results/dating_model_summary.json` — Model metrics
- `results/undated_predictions.csv` — Predicted dates for 102 undated inscriptions
- `results/dating_model_4panel.png` — Visualization

## Cross-Paper Implications

- **P5/P14:** Cannot use predictions as reliable dates. Treat as indicative only.
- **P8:** Content stability over time supports the "thin overlay" argument — underlying structure doesn't change much despite Indianization wave (E033).
- **Future:** A proper dating model needs paleographic + formulaic features, not just keyword counts.

## Conclusion

**CONDITIONAL.** The model achieves 76.3% century ±1 accuracy (LOOCV) but R²≈0 and temporal split failure mean it cannot reliably date individual inscriptions. Content features alone are too weakly temporal. The informative finding is that prasasti content is remarkably stable across 800 years — supporting P8's "thin overlay" argument. A useful dating model would require paleographic and formulaic features not available in the current pipeline.
