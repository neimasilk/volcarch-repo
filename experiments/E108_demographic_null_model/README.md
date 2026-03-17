# E108: Demographic Null Model — Pre-400 CE Java Carrying Capacity

**Date:** 2026-03-17
**Paper:** All (fundamental test of project premise)
**Status:** SUCCESS — Null hypothesis REJECTED

## Hypothesis

**Null hypothesis (never previously tested):** Pre-400 CE Java had a population too small to produce a detectable archaeological record. If true, the absence of pre-Hindu sites is expected, and taphonomic explanations (H1) are unnecessary.

## Method

Multi-scenario population model using:
- Java land area and terrain classification (129,000 km², 114,000 km² habitable)
- Five subsistence modes with ethnographic density analogues
- Contemporaneous comparanda (Thailand, Vietnam, Philippines, PNG, Sumatra)
- Sensitivity test: what if wet rice was absent before Indian contact?

### Subsistence Modes

| Mode | Low | Best | High | Land% | Source |
|------|:---:|:---:|:---:|:---:|---|
| Forest foraging | 0.1 | 0.2 | 0.5 | 10% | Headland & Reid 1989 |
| Coastal fishing | 1.0 | 2.5 | 5.0 | 5% | Kirch 2000 |
| Swidden agriculture | 5.0 | 12.0 | 25.0 | 40% | Bayliss-Smith 1980 |
| Early wet rice | 25.0 | 40.0 | 80.0 | 15% | Bray 1986; Higham 2014 |
| Mixed arboriculture | 10.0 | 20.0 | 40.0 | 30% | Kirch 2000 |

## Results

### Population Scenarios

| Scenario | Population | Density | Expected Settlements |
|----------|:---:|:---:|:---:|
| A (minimal) | **590,520** | 5.2/km² | 2,953-11,810 |
| B (moderate) | **1,931,730** | 16.9/km² | 9,659-38,635 |
| C (maximum) | **3,910,200** | 34.3/km² | 19,551-78,204 |

### Sensitivity: No Wet Rice
Without wet rice (swidden replaces): **1,452,930** (still 4,843x gap)

### Archaeological Gap
- Expected settlements (moderate): 9,659-38,635
- Known pre-400 CE Java sites: 0-3 (ambiguous)
- **Gap ratio: >3,220x**

### Contemporaneous Comparanda
Java's moderate estimate (1.93M) exceeds all contemporaneous ISEA polities:
- Thailand Dvaravati: 300-500K
- Vietnam Dong Son: 500K-1M
- Philippines: 200-500K
- Java's volcanic soils are among the most fertile in the world

## Conclusion

**THE NULL HYPOTHESIS IS REJECTED.**

Even the most conservative estimate (590K, foraging-dominant) implies thousands of settlements. The 3,220x gap between expected and observed pre-400 CE sites cannot be explained by low population. Three explanations remain:
1. Volcanic burial (H1)
2. Survey deficit (E086 Japan comparison)
3. Both (most likely)

## Caveats

1. **Population density estimates are extrapolated** from ethnographic analogues, not direct evidence
2. **Carrying capacity ≠ actual population** — disease, warfare, migration affect actual numbers
3. **Pre-400 CE agriculture in Java is poorly documented** — wet rice timing is debated
4. **0-3 known sites** may be undercount — some sites may exist but not be securely dated
5. **Kalimantan comparison** (low volcanic soil = low density = low sites) would strengthen argument but wasn't modeled

## Files

| File | Description |
|---|---|
| `demographic_null.py` | Population model script |
| `results/e108_results.json` | Full results with all scenarios |
