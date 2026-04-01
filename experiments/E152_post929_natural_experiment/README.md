# E152: Post-929 CE Mataram -> East Java Natural Experiment

**Date:** 2026-03-30  
**Status:** SUCCESS  
**Paper:** P1, P17  
**Layer:** L1 / L6 mechanism test

## Hypothesis

The 928-929 CE shift of Mataram's center from Central Java to East Java is a within-culture natural experiment. If volcanic context affects archaeological visibility, the surviving inscription record should change after the move.

## Data

- **E082:** 170 geocoded Java/Bali inscriptions
- **E030:** 166 dated inscriptions with NLP features
- **E084:** pre/post-929 volcano-distance split
- **E096:** BERTopic topic shift at 929 CE
- **E134:** century-level chronology
- `eruption_history.csv`: 11 medieval eruptions (VEI >= 3)

Sample sizes used in analysis:

- Geocoded + dated PRE-929: **100**
- Geocoded + dated POST-929: **27**
- NLP PRE-929: **130**
- NLP POST-929: **36**

## Method

1. Split inscription data at **929 CE**.
2. Compare Central vs East Java distribution.
3. Test change in mean volcano distance.
4. Compare word count and pre-Indic vocabulary ratio.
5. Cross-reference spatial shift with E096 topic shift and E084 split.

## Key Results

### 1. The Inscription Center Shifted East Dramatically

- Geographic center moved **187 km east**
- Mann-Whitney on longitude: **p = 3.89e-12**
- Region-by-period chi-square: **chi2 = 76.84, p = 1.85e-18**

PRE vs POST regional counts:

| Region | PRE-929 | POST-929 |
|--------|--------:|---------:|
| Central Java | 91 | 1 |
| East Java | 9 | 26 |

This is an extremely strong natural-experiment split.

### 2. Post-929 Inscriptions Are Farther from Volcanoes

| Period | Mean dist. to nearest volcano |
|--------|------------------------------:|
| PRE-929 | 22.78 km |
| POST-929 | 35.49 km |

- Difference: **+12.71 km**
- Mann-Whitney: **p = 0.000668**
- E084 confirmation: pre/post split **p = 5.27e-08**

This is the main taphonomic result.

### 3. The Surviving Record Becomes More Austronesian and More Complex

Pre-Indic vocabulary ratio:

| Period | Mean pre-Indic ratio |
|--------|---------------------:|
| PRE-929 | 0.088 |
| POST-929 | 0.231 |

- Mann-Whitney: **p = 0.000136**

Word count:

| Period | Mean word count |
|--------|----------------:|
| PRE-929 | 268.6 |
| POST-929 | 648.1 |

- Mann-Whitney: **p = 0.000025**

So the post-929 record is not only farther from volcanoes; it is also longer and lexically more pre-Indic.

### 4. Density Result is Mixed, Not Null

- Raw PRE-929 count: 88
- Raw POST-929 count: 78
- Raw density per century: 17.6 vs 15.6

But this is distorted by:

1. **C8 Borobudur inflation** (~48 relief labels)
2. **C10 peak** immediately after the shift (49 inscriptions)

Adjusted PRE-929 density excluding Borobudur falls to **8.0 per century**, so the density story is mixed rather than negative.

### 5. Topic Shift Aligns with the Spatial Shift

E096 already found a 929 CE topic shift:

- **chi-square p = 0.000251**
- PRE-929 dominant: administrative + ritual/calendrical
- POST-929 dominant: royal authority

E152 shows that this topic shift is not just political. It also sits inside a major geographic and taphonomic shift in where the surviving inscriptions come from.

## Interpretation

E152 supports a dual reading:

1. **Political:** new dynasty, new administrative center, new rhetorical priorities
2. **Taphonomic:** movement away from Merapi's high-deposition zone changed what survives

The experiment works because it is a within-tradition comparison. Same broad civilization, same inscription habit, different volcanic geography.

## Output Files

- `post929_analysis.py` - main analysis script
- `results/e152_results.json` - structured results
- `results/period_summary.csv` - PRE/POST summary table

## Limitations

1. DHARMA remains a survivorship-biased source corpus.
2. Many pre-929 inscriptions have generic Central Java geocoding.
3. The 929 cutoff simplifies a gradual transition.
4. Political and taphonomic effects cannot be perfectly separated without new non-DHARMA data.

## Conclusion

**SUCCESS.** The 929 CE shift is a real natural experiment and it produces measurable archaeological consequences. Post-929 inscriptions are **farther from volcanoes**, **farther east**, **longer**, and **more pre-Indic**. That does not prove a purely geological explanation, but it does show that volcanic geography is part of the mechanism shaping the surviving epigraphic record.
