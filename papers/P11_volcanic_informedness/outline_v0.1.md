# VOLCARCH Paper 11 — Outline v0.1
# Volcanic Informedness: Evidence for Volcanically-Aware
# Cultural Practices in Java and Bali

**Status:** Outline — ready for author review before drafting
**Date:** 2026-03-12
**Author:** Mukhlis Amien, Lab Data Sains, Universitas Bhinneka Nusantara
**Co-author (TBC):** Go Frendi Gunawan, Universitas Bhinneka Nusantara
**Cross-ref:** P1, P5, P7, P9 (all submitted)
**Prior version:** `docs/drafts/P11_volcanic_cultural_selection.md` (concept draft, 2026-03-07 — SUPERSEDED)

---

## Reframing Note

The original P11 proposed **Volcanic Cultural Selection (VCS)** as a pan-Austronesian phenomenon: volcanic cultures develop higher ritual complexity. **E039 definitively rejected this** (binary p=0.973, direction reversed; continuous rho=+0.145, opposite prediction).

What DOES survive:
- **E031:** Candi siting on west sides of volcanoes (p<0.0001) — builders knew where was safer
- **E032:** Pranata Mangsa inadvertently encodes volcanic hazard seasonality (chi2 p=0.042)
- **E039c partial:** Group hunting and population density correlate with volcanic proximity (but = fertility, not cultural selection)

The new framing: **Volcanic Informedness** — communities living near active volcanoes in Java/Bali encode practical volcanic knowledge in architecture, calendars, and land-use decisions, without requiring the stronger claim that this knowledge was "selected for" at a population level.

This is a **local cultural ecology** paper, not a universal evolutionary argument.

---

## Target Journal

**Primary:** Indonesia (Cornell University Press)
- Free (no APC), Q2 Scopus, ISSN 0019-7289
- Area studies journal focused on Indonesia/Southeast Asia
- Word limit: ~15,000 words (generous)
- Accepts interdisciplinary work (cultural ecology + computational methods)
- Non-overlapping with P5 (BKI) and P9 (JSEAS) — different scope and audience

**Backup:** Asian Ethnology (Nanzan University)
- Free, Q2, cultural studies/anthropology focus

---

## Abstract (~250 words)

Java and Bali host 45 active volcanoes and over 74,000 years of continuous human habitation. We investigate whether this sustained volcanic exposure left detectable traces in local cultural practices — a phenomenon we term *volcanic informedness*. Using three independent computational tests, we find that (1) Hindu-Buddhist temples (candi) cluster preferentially on western flanks of volcanoes (Rayleigh p=3.4×10⁻⁸, n=142), consistent with tephra-sheltered positioning during southeast monsoon eruptions; (2) the traditional Javanese agricultural calendar Pranata Mangsa inadvertently encodes volcanic hazard seasonality, with peak eruption density during its wet-season month Kapitu (3.8× the lowest month; chi-squared p=0.042); and (3) these patterns are local rather than universal — a cross-cultural test across 116 Austronesian societies finds no global correlation between volcanic proximity and ritual complexity (p=0.973). We argue that volcanic informedness in Java and Bali represents practical environmental knowledge encoded in architecture and calendar rather than population-level cultural selection. This local phenomenon has been overlooked because archaeological models emphasize external cultural inputs (Indianization, Islamization) over endogenous landscape adaptation. Our findings complement recent work on volcanic taphonomic bias in Indonesian archaeology by showing that the same volcanic processes that bury archaeological evidence also shaped the cultural practices that produced it.

---

## 1. Introduction

### 1.1 Volcanic Landscapes and Cultural Adaptation

- Java-Bali = most volcanically active continuously inhabited region on Earth
- 45 active volcanoes (Java 34, Bali 2, adjacent islands 9)
- Standard narrative: culture shaped by Indianization, Islamization, colonialism
- Missing: endogenous volcanic landscape as shaping force
- Not arguing VCS (population-level selection) — arguing **informedness** (knowledge encoding)

### 1.2 Defining Volcanic Informedness

- **Volcanic Informedness (VI):** The encoding of practical volcanic knowledge in cultural practices (architecture, calendar, land use, cosmology) by communities living near active volcanoes
- Distinction from VCS: VI does not require differential survival of communities. It only requires that communities *observe* and *encode* volcanic patterns
- Analogous to "environmental knowledge" in Traditional Ecological Knowledge (TEK) literature
- VI can be detected computationally when systematic spatial or temporal patterns correlate with volcanic parameters

### 1.3 Scope and Limitations

- **Local claim only:** Java and Bali. Not pan-Austronesian, not universal
- **Computational evidence only:** No fieldwork, no ethnographic interviews
- **Complementary to VOLCARCH series:** P1 (burial rates), P5 (ritual substrate), P7 (spatial bias), P9 (peripheral conservatism)
- Does NOT require "Kawah Candradimuka" cultural memory framing (too speculative for this paper — park for future essay)

---

## 2. Study Area and Data

### 2.1 Volcanic Landscape of Java and Bali

- 45 active volcanoes (GVP Holocene database)
- Major systems: Merapi (central Java, pyroclastic), Kelud (east Java, lahar), Semeru (continuous), Tengger caldera, Agung/Batur (Bali)
- Eruption frequency: ~10-15 events per decade across Java
- Southeast monsoon dominance: tephra dispersal preferentially to west/northwest
- Map: volcano locations + candi distribution (adapt from E031)

### 2.2 Candi Dataset

- 142 candi with coordinates from published archaeological surveys
- Source: BPCB Jawa Timur + published gazetteers
- 20 candi with documented entrance orientation (from published studies)
- Penanggungan complex: 73 candi, densest cluster in Java

### 2.3 Eruption Seasonality Data

- GVP Holocene eruption database: 127 dated events from 4 Java volcanoes
- Kelud (32 events), Merapi (47 events), Semeru (28 events), Tengger-Bromo (20 events)
- **Limitation:** Merapi MISSING from GVP seasonal data (reporting gap)
- Pranata Mangsa: 12 traditional seasons mapped to Gregorian months (Daldjoeni 1984)

### 2.4 Cross-Cultural Comparative Data

- Pulotu Database of Pacific Religions: 116 Austronesian societies
- 9 ritual/cooperation variables (Q21 mana, Q32 island type, Q44 population, Q58 group hunting, etc.)
- GVP: 854 Holocene volcanoes worldwide for distance computation

---

## 3. Methods

### 3.1 Test 1: Candi Spatial Analysis (E031)

**Hypothesis:** If volcanic informedness exists, candi should cluster on volcanically sheltered flanks (west, given SE monsoon tephra dispersal).

- For each candi, compute bearing from nearest volcanic peak
- Rayleigh test for non-uniform circular distribution
- Quadrant chi-squared test (W/E/N/S expected 25% each)
- Control: population density map (does western clustering = more people, or volcanic preference?)
- Sub-analysis: Penanggungan complex (73 candi, single peak, controls for inter-volcano variation)

### 3.2 Test 2: Calendar-Eruption Seasonality (E032)

**Hypothesis:** If volcanic informedness is encoded in Pranata Mangsa, eruption frequency should correlate with specific mangsa periods.

- Map 127 eruptions to Pranata Mangsa months
- Null model: uniform distribution across 12 mangsa
- Chi-squared test for non-uniformity
- Rayleigh test for circular concentration
- Compute eruption density per mangsa (events per 30-day equivalent)
- Identify which mangsa carries highest volcanic hazard

### 3.3 Test 3: Cross-Cultural Falsification (E039)

**Purpose:** Test whether volcanic-cultural correlations are universal or local.

- **E039a:** Binary comparison — Pulotu volcanic vs non-volcanic islands
- **E039b:** Continuous — Haversine distance to nearest volcano vs ritual complexity
- **E039c:** Subsistence variables — disentangle volcanic fertility from cultural effects
- If E039a/b null → volcanic informedness is LOCAL (Java/Bali), not global
- This is not a failure — it **constrains** the claim to where it's valid

---

## 4. Results

### 4.1 Candi Cluster on Western Volcanic Flanks

- **All Java:** 142 candi, mean bearing = west-southwest. Rayleigh p = 3.4×10⁻⁸
- **Quadrant test:** West quadrant 1.89× expected (chi-squared p < 0.0001)
- **Penanggungan:** 73 candi, 46 west side (63.0%). Binomial p = 3.1×10⁻¹⁴
- **Orientation null:** 7/20 entrance faces volcano (35%, p = 0.94). Religious canon > volcanic direction
- **Interpretation:** WHERE to build = volcanically informed. HOW to orient = religiously determined

→ **Table 1:** Candi quadrant distribution per volcano
→ **Figure 1:** Polar plot of candi-volcano bearings (142 points)
→ **Figure 2:** Penanggungan detail map with west-clustering

### 4.2 Pranata Mangsa Encodes Volcanic Hazard Seasonality

- **Eruptions not uniformly distributed:** Chi-squared p = 0.042, Rayleigh p = 0.032
- **Kapitu (Dec–Jan, wet season):** 18.14 eruptions per 30-day period (rank 1/12)
- **Kapat (Sep, dry season):** 4.78 eruptions per 30-day period (rank 12/12)
- **Ratio:** 3.8× between highest and lowest mangsa
- **Monsoon coupling:** Wet season rainfall triggers phreatic eruptions (well-documented mechanism)
- **Calendar dual function:** Communities following Pranata Mangsa for agriculture were inadvertently tracking volcanic hazard

→ **Table 2:** Eruption density per mangsa month
→ **Figure 3:** Eruption seasonality polar plot with Pranata Mangsa overlay

### 4.3 Volcanic Informedness is Local, Not Global

- **E039a (binary):** Volcanic vs non-volcanic island ritual complexity: p = 0.973, direction REVERSED
- **E039b (continuous):** Distance to volcano vs ritual: rho = +0.145, p = 0.092 (OPPOSITE prediction)
- **E039c (subsistence):** Group hunting p=0.002, population p=0.010 — but = volcanic soil fertility, not cultural selection
- **Conclusion:** No global Austronesian pattern. Volcanic informedness is a LOCAL phenomenon of communities living in direct proximity to active volcanoes on Java and Bali

→ **Table 3:** E039 summary (3 sub-tests, all null for global VCS)
→ **Figure 4:** Scatterplot of volcanic distance vs ritual complexity (116 societies, no trend)

---

## 5. Discussion

### 5.1 What Volcanic Informedness Is (And Is Not)

- IS: practical environmental knowledge encoded in architecture and calendar
- IS: detectable computationally through spatial and temporal analysis
- IS NOT: population-level cultural selection (VCS rejected, §4.3)
- IS NOT: pan-Austronesian phenomenon (local to Java/Bali)
- IS NOT: deterministic (religious canon overrides volcanic logic for orientation, §4.1)

### 5.2 Why Western Siting? Tephra, Water, or Both?

- SE monsoon disperses tephra east/southeast → west flanks less exposed
- But western flanks also have better water access (orographic rainfall on windward slopes)
- Cannot fully disentangle — both mechanisms predict western clustering
- Population confound: more people live west of East Java volcanoes (historical pattern)
- **Honest statement:** Western siting is consistent with volcanic informedness but not conclusive evidence for it

### 5.3 Calendar as Environmental Knowledge System

- Pranata Mangsa already recognized as TEK (Traditional Ecological Knowledge) for agriculture (Daldjoeni 1984)
- Our contribution: showing that the calendar ALSO tracks volcanic hazard seasonality
- This is not intentional — it's a structural consequence of monsoon-eruption coupling
- Communities didn't need to "know about eruptions" — they just needed to follow the calendar
- **Implication for disaster risk reduction:** Traditional calendars may encode hazard information that modern systems miss

### 5.4 Connecting to the VOLCARCH Framework

- P1 shows volcanic processes BURY archaeological evidence (taphonomic bias)
- P11 shows volcanic processes SHAPE the cultural practices that produced that evidence
- **Feedback loop:** Volcanoes bury organic civilization (P1) → archaeological "blank" → scholars seek external explanations (Indianization) → endogenous volcanic adaptation overlooked → volcanic informedness invisible
- P11 closes the loop: the "missing" culture was shaped by the same force that hid it

### 5.5 E039c: Fertility, Not Selection

- Group hunting and population size correlate with volcanic proximity
- But this is volcanic soil fertility → more food → more people → more cooperation
- This is demographics, not cultural selection
- Important to distinguish: volcanic informedness (knowledge) vs volcanic fertility (demographics)
- Both are real; only informedness is cultural

---

## 6. Future Directions

### 6.1 Proposed Experiments (Not Yet Executed)

- **E045:** Candi-volcano distance vs ritual complexity (extend E031 with Pulotu-style ritual variables from epigraphic data). Test whether temples closer to volcanoes show more elaborate dedications.
- **E046:** Pranata Mangsa × additional calendar systems (Balinese Tika, Sasak calendar). Test whether volcanic informedness extends to Bali's calendar system.
- **E047:** Candi western siting × eruption survival. Test whether western-flank candi have lower damage rates in historical eruptions than eastern-flank candi.
- **E048 (if fieldwork):** Ethnographic interviews in Tengger and Bali Aga communities about volcanic knowledge encoded in practice.

### 6.2 Limitations Requiring Future Work

- Merapi MISSING from GVP seasonal data (most important Java volcano)
- Only 20 candi with documented entrance orientation (needs systematic survey)
- No ethnographic validation of "informedness" — all evidence is distributional
- Population confound for western siting not resolved

---

## 7. Conclusion

Three independent tests show that cultural practices in Java and Bali encode volcanic environmental knowledge:
1. Sacred architecture sites preferentially on sheltered volcanic flanks
2. The traditional agricultural calendar inadvertently tracks volcanic hazard seasonality
3. These patterns are local to Java/Bali, not a universal Austronesian trait

We term this **volcanic informedness** — the encoding of practical volcanic knowledge in architecture, calendar, and land-use decisions. Unlike the stronger claim of Volcanic Cultural Selection (rejected by our cross-cultural test), volcanic informedness requires only that communities observe and encode volcanic patterns, not that this knowledge was subject to population-level selection.

This finding complements the VOLCARCH project's demonstration that volcanic processes create systematic archaeological biases (Papers 1, 7): the same volcanoes that bury evidence of past civilizations also shaped the cultural practices of those civilizations. Understanding volcanic informedness is essential for correct interpretation of Java's archaeological record.

---

## Figure List

| # | Description | Source |
|---|-------------|--------|
| 1 | Polar plot: candi-to-volcano bearings (n=142) | E031 |
| 2 | Penanggungan detail map with west-clustering | E031 (new) |
| 3 | Eruption seasonality × Pranata Mangsa polar overlay | E032 |
| 4 | Scatterplot: volcanic distance vs ritual complexity (116 societies) | E039 |
| 5 | Conceptual diagram: volcanic informedness feedback loop | New |

## Table List

| # | Description | Source |
|---|-------------|--------|
| 1 | Candi quadrant distribution per major volcano | E031 |
| 2 | Eruption density per Pranata Mangsa month | E032 |
| 3 | Cross-cultural test summary (E039a/b/c) | E039 |

---

## Experiment Foundation

| Experiment | Status | Key Result | Role in P11 |
|-----------|--------|------------|-------------|
| E031 | SUCCESS (split) | Siting p<0.0001; orientation null | §4.1 — core evidence |
| E032 | CONDITIONAL SUCCESS | Chi2 p=0.042; Kapitu peak 3.8× | §4.2 — core evidence |
| E039 | INFORMATIVE NEG | Binary p=0.973; distance rho=+0.145 | §4.3 — constrains scope |
| E033 | SUCCESS | Indianization curve, rho=-0.211 | §5.4 — contextual support |
| E040 | SUCCESS | Organic 63.4% vs lithic 27.2% | §5.4 — taphonomic bridge |
| **E051** | **SUCCESS** | **57.7% pre-Hindu toponyms, court rho=0.387** | **§5.6 — toponymic overwriting** |
| **E056** | **SUCCESS** | **Candi in MORE Indianized areas, MW p=0.007** | **§5.4 — dual Indianization signature** |
| **E065** | **SUCCESS** | **Zone A 17.9× overrepresented, West 47.2%** | **§4.1 — strengthens spatial analysis** |

---

## Writing Notes

- **Tone:** Cultural ecology, not evolutionary psychology. Avoid "selection" language.
- **Honesty:** §5.2 and §5.5 must be transparently honest about alternative explanations.
- **Scope discipline:** LOCAL only. Every time the text drifts toward "Austronesian," pull it back.
- **No Kawah Candradimuka:** Beautiful metaphor, but too speculative for a data paper. Save for essay or P5 revision.
- **Word count target:** ~8,000-10,000 (Indonesia journal allows up to 15,000 but brevity is strength).

---

*Outline v0.1 — 2026-03-12*
*Replaces concept draft P11_volcanic_cultural_selection.md (2026-03-07)*
*"Pertanyaannya bukan 'apakah mereka tahu ada gunung berapi' — mereka memasukkannya ke dalam kalender."*
