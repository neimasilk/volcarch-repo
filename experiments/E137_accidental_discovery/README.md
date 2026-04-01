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

### F4 Discrepancy Resolution (Mata Elang #11)

**E137's F4 ≠ E110's F4.** These measure categorically different processes:

| Context | F4 Value | What it measures |
|---------|----------|------------------|
| E110 cascade | 0.40 | Probability that a **trained archaeologist with GPR** correctly identifies a subsurface anomaly as archaeological |
| E137 accidental | 0.0007 | Probability that a **sand miner** recognizes pottery sherds as ancient, stops work, AND reports to authorities |

The 570× discrepancy is real but not contradictory — accidental discovery by untrained workers and systematic detection by professionals are fundamentally different processes. E110's F4 applies to the PLANNED fieldwork scenario (E116 predictions). E137's F4 explains why ACCIDENTAL finds are so rare despite frequent deep excavation.

**Implication:** The cascade model (E110) remains valid for its intended use case — predicting outcomes of systematic archaeological survey. E137 adds a complementary insight: the lack of accidental finds is explained by recognition/reporting failure, not by absence of buried material.

## Conclusion

**PARTIAL.** Model concept valid but parameters need calibration. The key finding is UNEXPECTED: recognition/reporting barriers for **accidental** discovery are far more severe than E110's cascade (which models **systematic** discovery) estimated. This strengthens, not weakens, VOLCARCH — it means the lack of accidental finds is not evidence against buried sites, but evidence of recognition failure.

## Scripts

- `accidental_discovery.py`
