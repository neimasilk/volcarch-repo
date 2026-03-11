# E039: Volcanic Cultural Selection — Cross-Cultural Test

**Status:** INFORMATIVE NEGATIVE
**Date:** 2026-03-11
**Paper:** P11 (Volcanic Cultural Selection)
**IDEA_REGISTRY:** I-042

## Hypothesis

Austronesian cultures on volcanic high islands show higher ritual complexity than those on non-volcanic islands (atolls, continental, mainland).

## Method

- **Data:** Pulotu database (137 Austronesian cultures × 86 variables)
- **Classification:** Q32 (island type): volcanic high island (n=61) vs others (n=76)
- **Indices:** Broad ritual complexity (11 vars mean), mortuary index (Q4+Q5+Q10), costly sacrifices (Q35)
- **Test:** Mann-Whitney U, one-tailed (volcanic > non-volcanic)

## Results

| Test | Volcanic | Non-Volcanic | p-value | Cohen's d |
|------|----------|-------------|---------|-----------|
| Broad ritual complexity | 1.016 | 1.109 | 0.973 | -0.317 |
| Mortuary index (Q4+Q5+Q10) | 1.256 | 1.402 | 0.971 | -0.294 |
| Costly sacrifices (Q35) | 0.842 | 0.819 | 0.369 | — |
| Supernatural punishment (Q7) | 0.949 | 0.986 | 0.890 | — |

**All tests non-significant. Direction is OPPOSITE prediction (non-volcanic > volcanic).**

### One exception: Q21 (Mana as spiritual concept)
- Volcanic: 0.446, Non-volcanic: 0.232, **p = 0.006**
- Does NOT survive Bonferroni correction (threshold 0.05/11 = 0.0045)
- But intriguing: "spiritual power/energy" concept more prominent on volcanic islands

### Malagasy control
- Merina: ritual=1.273, mortuary=1.667 (HIGHER than volcanic mean)
- Tanala: ritual=0.909, mortuary=1.333
- VCS prediction (Malagasy < volcanic) is WRONG

### Classification problem (CRITICAL)
- Toraja (highest ritual: 1.545, mortuary: 2.000) classified as "mainland" not "volcanic_high"
- Toba Batak (ritual: 1.364, mortuary: 2.333) also "mainland"
- Q32 classifies by island geology, not by proximity to active volcanoes
- Java/Sumatra/Sulawesi = geologically complex islands classified as "mainland/continental"
- The cultures with the strongest VCS signatures are MISCLASSIFIED

### Confound: Pre-Austronesian contact
- Volcanic cultures: mean 1.000 vs Non-volcanic: 1.378 (p=0.028)
- Non-volcanic (mainland) cultures had MORE pre-Austronesian contact
- Pre-Austronesian contact → syncretic elaboration → higher ritual scores
- This confound may explain the reversed direction

## Interpretation

The VCS hypothesis is **NOT refuted** but **NOT supported by this crude test**.

The core problem: Q32 island-type classification is the wrong proxy for volcanic cultural selection pressure. VCS, if real, operates at a finer scale:

1. **Within-island variation** — communities closer to active volcanoes vs. further away (e.g., Tengger vs. coastal Java). This is P11's real prediction domain.
2. **Eruption frequency** — not just "volcanic island" but "how many eruptions per century." A volcanic island with 1 eruption per millennium is different from Java with 45 active volcanoes.
3. **Pre-Austronesian confound** — mainland/continental cultures had more contact with non-Austronesian populations, introducing syncretic ritual elaboration unrelated to volcanism.

The one promising signal (Q21: mana as spiritual concept, p=0.006) hints that volcanic landscapes may foster specific cosmological concepts (spiritual energy/power) without increasing overall ritual complexity.

## What This Means for P11

**DO NOT claim:** "Volcanic cultures have higher ritual complexity globally."
**DO claim:** "VCS is a local/regional phenomenon (Java, Bali, Toba) requiring fine-grained proximity-to-volcano analysis, not a pan-Austronesian island-type effect."

P11 should reframe: VCS operates at the scale of specific volcanic complexes (Merapi, Kelud, Tengger), not at the scale of "volcanic islands." E031 (candi siting within Java) and E032 (Pranata Mangsa within Java) provide the right granularity.

## E039b: Distance-Based Test (also INFORMATIVE NEGATIVE)

Refined test using continuous distance to nearest active volcano (854 GVP Holocene volcanoes):

| Test | rho | p | VCS Direction? |
|------|-----|---|----------------|
| Distance vs ritual | +0.145 | 0.092 | OPPOSITE (farther = more) |
| Distance vs mortuary | +0.003 | 0.968 | Null |
| Eruption count vs ritual | +0.065 | 0.450 | Null |
| Indo/Melanesian subset | +0.147 | 0.184 | OPPOSITE |

**Both tests reject VCS as global phenomenon. Direction consistently opposite.**

## Reframe for P11

VCS doesn't make cultures MORE ritually complex. It makes specific practices VOLCANICALLY INFORMED (where to build, when to plant, how to time death rituals). E031+E032 within Java remain valid.

## E039c: Subsistence/Cooperation Test (INFORMATIVE MIXED)

Tests whether volcanic proximity correlates with group-based subsistence and political complexity (I-044).

| Variable | rho | p | Direction |
|----------|-----|---|-----------|
| Q58 Group hunting | **-0.275** | **0.002** | **SUPPORTS** (closer = more) |
| Q44 Population | **-0.226** | **0.010** | **SUPPORTS** (closer = larger) |
| Q61 Group fishing | +0.238 | 0.007 | OPPOSITE (closer = less) |
| Q59 Agriculture | -0.052 | 0.553 | Null |
| Q16 Resource tapu | +0.026 | 0.787 | Null |
| Q50 Political community | +0.049 | 0.582 | Null |
| Q37 Polit-relig differentiation | +0.045 | 0.605 | Null |
| Q84 Religious authority | +0.071 | 0.437 | Null |
| Q86 Political authority | +0.113 | 0.207 | Null |

**Eruption count × agriculture:** rho = -0.248, p = 0.004 (more eruptions = MORE agriculture)

### Interpretation: Volcanic FERTILITY, Not Cultural Selection

The significant results (population, group hunting, agriculture × eruption count) reflect **volcanic soil fertility** driving agricultural surplus and larger populations — not cultural selection for ritual or cooperation. The Q61 (group fishing) being OPPOSITE confirms a distance-to-coast confound (volcanoes are inland).

**Verdict:** VCS as cultural selection mechanism remains unsupported at global scale. What volcanic proximity predicts is DEMOGRAPHIC: more people, more food, more group foraging — standard volcanic fertility effects known since Malthus.

## Summary of All Three Tests

| Test | Result | Key Finding |
|------|--------|-------------|
| E039 (binary island type) | INFORMATIVE NEGATIVE | Direction reversed. Classification problem. |
| E039b (continuous distance) | INFORMATIVE NEGATIVE | rho=+0.145. Farther = more ritual. |
| E039c (subsistence/cooperation) | INFORMATIVE MIXED | Population + group hunting significant, but = volcanic fertility, not VCS. |

**Overarching conclusion:** VCS is NOT a global Austronesian phenomenon. It is either (a) a local Java/Bali phenomenon operating at within-island scale (E031, E032), or (b) not a distinct mechanism at all — just volcanic fertility driving demographics. P11 must scope claims accordingly.

## Scripts

- `01_vcs_test.py` — binary island-type test
- `02_vcs_distance_test.py` — continuous distance test (854 GVP volcanoes)
- `03_vcs_subsistence_test.py` — subsistence/cooperation test
- `results/` — all JSON + CSV outputs
- `FULL_REPORT.md` — comprehensive analysis document (E039 + E039b)
