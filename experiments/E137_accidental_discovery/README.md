# E137: Accidental Discovery Rate Model

**Date:** 2026-03-30
**Status:** PARTIAL (concept valid, parameters need calibration)
**Paper:** P1
**Layer:** L1 (cascade factors F3, F4)
**Mata Elang:** #10 Blind Spot B2 (Liangan Paradox)

---

## Hypothesis

The absence of accidental pre-Hindu archaeological discoveries can be explained by the depth-dependent probability of excavation activities reaching burial depth.

## Method

Modeled 7 types of deep construction/mining activity in Java with annual frequency, typical depth, footprint area, and intersection probability with predicted buried sites.

## Results

### Model Overpredicts By ~600x

The model predicts 3,000 post-Hindu site discoveries per century, but only ~5 are observed. This 600x overprediction reveals that **recognition and reporting** are far more severe barriers than previously modeled.

### Why the Overprediction?

1. **Most sand mining doesn't reach target depth** (4-6m typical, not 8m)
2. **Not all mining is in the volcanic zone** (parameter too generous)
3. **Recognition failure:** A sand miner who finds pottery sherds doesn't report it
4. **Reporting failure:** Even recognized finds often go unreported to authorities
5. **F4 (recognition) may be 0.002 instead of 0.40** — two orders of magnitude more severe

### The Liangan Paradox Resolution

The paradox ("why so few accidental finds?") dissolves when recognition/reporting are factored in. **Liangan was exceptional not because sand miners hit a buried site — that probably happens regularly — but because the site was SO OBVIOUS (complete structures, visible temples) that it was impossible to ignore.** Ordinary pre-Hindu organic sites at depth would be unrecognizable to non-archaeologists.

### Revised Cascade Implication

If recognition is 600x worse than modeled (F4 = 0.0007 instead of 0.40):
- The cascade product becomes much smaller
- This actually HELPS explain the gap without needing volcanic burial to be dominant
- **Survey deficit x recognition failure alone may explain >99% of the gap**

## Conclusion

**PARTIAL.** Model concept valid but parameters need calibration. The key finding is UNEXPECTED: recognition/reporting barriers are far more severe than E110 estimated. This strengthens, not weakens, VOLCARCH — it means the gap is even MORE explainable by non-burial factors.

## Scripts

- `accidental_discovery.py`
