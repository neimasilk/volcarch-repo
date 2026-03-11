# VOLCARCH Paper 10 — DRAFT v0.1
# Archaeological Biosignatures for Detecting Buried 
# Civilizations in Volcanic Landscapes: 
# A Geochemical Disequilibrium Framework for Java

**Status:** Concept draft — ideas capture, not yet for submission  
**Date:** 2026-03-06  
**Author:** Mukhlis Amien, Lab Data Sains, Universitas Bhinneka Nusantara  
**Requires:** Fieldwork collaboration (PVMBG / UGM Geologi / ANU)

---

## The Core Analogy

In exoplanet science, the co-presence of oxygen and methane in an 
atmosphere constitutes a *biosignature* — a chemical disequilibrium 
that cannot be maintained abiotically. The presence of life is inferred 
not from direct observation, but from the thermodynamic impossibility 
of the observed signal without biological intervention.

We propose an archaeological equivalent: a set of subsurface geochemical 
signals that, in combination, cannot occur naturally in volcanic plains 
but are predictably produced by sustained human occupation. We call this 
an **archaeological disequilibrium signature (ADS)**.

```
Exoplanet logic:
O₂ + CH₄ coexistence = thermodynamically impossible without life
→ Infer: life present

Archaeological logic:
Phosphorus cluster + anthropogenic charcoal + 
rice phytolith + non-local strontium
= statistically impossible without human occupation
→ Infer: settlement present at depth
```

---

## 1. Introduction

### 1.1 The Missing Positive Evidence Problem

VOLCARCH Papers 1–9 establish a framework for explaining *why* 
pre-400 CE archaeological evidence is absent from volcanic Java:

- Papers 1, 7: taphonomic burial at 3.6–4.4 mm/yr
- Paper 3: settlement suitability model identifies high-probability zones
- Paper 7 (E019): spatial segregation confirms burial signature
- Paper 9: peripheral conservatism provides cultural substrate evidence

However, the framework remains primarily *negative* — it explains 
absence, but does not demonstrate presence. The missing piece is 
**positive subsurface evidence** of pre-400 CE Homo sapiens occupation 
on volcanic Java plains.

This paper proposes a methodology to obtain that evidence.

### 1.2 Why Standard Archaeology Cannot Solve This

```
Standard surface survey: effective to ~1m depth
Ground-penetrating radar: effective to ~5-10m
Excavation: expensive, requires permit, 
            point-specific (where to dig?)

Problem: Evidence at 5-100m depth across
         thousands of km² of volcanic plain

Solution needed: 
(1) Tell us WHERE the signal should be
    → Paper 3 XGBoost model (done)
(2) Tell us WHAT signal to look for
    → This paper
(3) Test it
    → Paper 11 (requires fieldwork)
```

---

## 2. The Four Archaeological Biosignatures

### 2.1 Signature 1: Phosphorus Anomaly

**Mechanism:**
Human activity concentrates phosphorus through:
- Food waste and middens
- Human and animal excrement
- Bone decomposition
- Ash from cooking fires

Volcanic soil baseline phosphorus is HIGH (fertile), but it is 
spatially uniform. Human occupation creates **clustered phosphorus 
anomalies** that stand out against the uniform volcanic background.

```
Natural volcanic plain:
→ Phosphorus: high but spatially uniform
→ Pattern: random variation

Human-occupied site:
→ Phosphorus: elevated IN CLUSTERS
→ Pattern: non-random, structured
→ Detectable: yes, even after 10,000+ years
   (phosphorus bonds permanently to soil minerals)
```

**Precedent:** Phosphorus mapping has successfully identified:
- Maya settlements in Yucatan (Parnell et al. 2002)
- Iron Age settlements in Scandinavia
- Pre-Columbian villages in Amazonia

**Java application:** No systematic phosphorus survey of volcanic 
plains in East Java has been conducted for archaeological purposes. 
This is an immediate gap.

**Critical question:** Does volcanic burial CONCENTRATE or DILUTE 
phosphorus signatures over time? This requires lab calibration 
using known buried sites (Sambisari, Kedulan) as ground truth.

---

### 2.2 Signature 2: Anthropogenic Charcoal

**Mechanism:**
All fire produces charcoal, but cooking fires and wildfires produce 
morphologically and chemically distinct charcoal:

```
Wildfire charcoal:
→ Large fragments, irregular
→ Mixed species (whatever burned)
→ Single deposition event
→ No spatial clustering

Cooking fire / anthropogenic charcoal:
→ Small, rounded fragments 
  (repeated mechanical disturbance)
→ Specific species selection 
  (preferred fuel wood)
→ Multiple deposition layers
→ Spatially clustered at hearth locations
→ Associated with other ADS markers
```

**Volcanic context advantage:**
Charcoal survives volcanic burial extremely well — it is chemically 
inert and resistant to compaction. Charcoal from 40,000+ year old 
hearths has been recovered from volcanic contexts in Sulawesi.

**Radiocarbon potential:**
Anthropogenic charcoal at depth can be directly dated via AMS 
radiocarbon — providing absolute chronology for occupation events.

---

### 2.3 Signature 3: Rice Phytoliths

**Mechanism:**
Plants deposit silica bodies (phytoliths) in their cells. These 
survive in soil long after the plant has decomposed. Rice (*Oryza sativa*) 
produces distinctive phytolith morphologies that are:
- Species-identifiable
- Distinguishable from wild grass phytoliths
- Stable in volcanic soil for tens of thousands of years

```
Natural volcanic plain:
→ Phytoliths: C4 grasses dominant
→ Pattern: background assemblage

Human cultivation site:
→ Rice phytoliths (C3): present in concentration
→ Weedy species associated with cultivation
→ Pattern: cultivation assemblage
→ Directly implies: sedentary agriculture
   = sedentary population
   = settlement
```

**Why this is powerful for Java:**
Rice cultivation requires sedentary settlement. The presence of rice 
phytoliths at depth is not just evidence of *humans* — it is evidence 
of *organized agricultural communities*, which is precisely the 
"civilization" signal we are looking for.

**Java precedent:** Rice phytoliths have been recovered from sites 
in mainland Southeast Asia at >6,000 BP. No systematic phytolith 
survey of volcanic plain cores in East Java exists.

---

### 2.4 Signature 4: Strontium Isotope Anomaly

**Mechanism:**
Strontium isotope ratios (⁸⁷Sr/⁸⁶Sr) in human teeth and bones 
reflect the geology of the region where a person grew up. Volcanic 
rocks have distinct strontium signatures.

A non-local strontium signature in skeletal remains = person who 
grew up elsewhere = migration, trade, or urban centre drawing 
population from multiple geological zones.

```
Isolated small community:
→ Strontium: matches local volcanic geology
→ All individuals: same ratio

Settlement / urban centre:
→ Strontium: MIXED ratios
→ Some individuals: non-local
→ Implies: long-distance movement, trade,
           complex social organization
```

**Application to VOLCARCH:**
If skeletal remains are recovered from Zone B/C cores (even fragmentary), 
strontium analysis can determine whether the population was local or 
drew from a wider region — distinguishing a small farmstead from 
a proto-urban centre.

**Limitation:** Requires skeletal material, which may not be present 
if burial practice was exposed/surface (Paper 9 — mepasah analog). 
In that case, strontium from food residue on ceramic sherds 
(if present) can substitute.

---

## 3. The Disequilibrium Argument

The power of the ADS framework is not in any single signature, 
but in their **co-occurrence**:

```
Signature 1 alone (phosphorus):
→ Could be natural volcanic anomaly
→ Probability of false positive: moderate

Signatures 1 + 2 (phosphorus + charcoal):
→ Less likely natural
→ Probability of false positive: low

Signatures 1 + 2 + 3 (+ rice phytolith):
→ Cannot occur naturally
→ Probability of false positive: negligible

Signatures 1 + 2 + 3 + 4 (+ strontium):
→ Impossible without organized human settlement
→ Probability of false positive: ~0
```

This is the disequilibrium logic: the combination is the signal, 
not any individual component.

---

## 4. Integration with Paper 3 (XGBoost Model)

```
Paper 3 output:
→ Settlement suitability probability map
→ Zone A: sites present (shallow burial)
→ Zone B: no sites, moderate burial (100-300cm)
→ Zone C: no sites, deep burial (>300cm)

Paper 10 application:
→ Select 10-20 Zone B/C cells with 
  highest suitability scores
→ These are WHERE to take soil cores
→ Drill to burial depth (based on Paper 1 model)
→ Analyze for ADS markers

Prediction (falsifiable):
Zone B/C high-suitability cores will show
significantly higher ADS signal than:
(a) Zone E low-suitability cores
(b) Zone B/C low-suitability cores

If prediction confirmed:
→ First positive evidence of buried settlement
   on volcanic Java plains
→ Papers 1-9 framework validated

If prediction not confirmed:
→ Either: taphonomic destruction of 
          geochemical signals at depth
          (important negative result)
→ Or: genuine absence of settlement
      in predicted zones
      (requires framework revision)
```

---

## 5. Proposed Fieldwork Design

### Phase 1: Calibration (6 months, low cost)

**Use existing cores from PVMBG:**

```
PVMBG already has soil cores from volcanic 
hazard assessment across East Java.

Request: access to existing core samples 
from Zone B/C coordinates for 
re-analysis (phosphorus, charcoal, phytolith)

Cost: lab analysis only (~Rp 50-100 juta)
No new drilling required
No excavation permit required
```

**Use known buried sites as positive controls:**

```
Sambisari, Kedulan, Kimpulan (Merapi system)
= known occupation sites, known burial depth

Request: soil core samples from adjacent 
areas (not the temples themselves)

Test: do ADS markers appear in cores 
adjacent to known buried temples?

If yes: ADS methodology validated on 
        known sites → apply to unknown
```

### Phase 2: Prediction Test (12 months)

```
Select top 10 Zone B/C high-suitability cells 
from Paper 3 model in East Java

New soil cores at predicted coordinates:
→ Drill to predicted burial depth
→ Full ADS panel analysis
→ AMS radiocarbon dating of charcoal if found

Compare with 10 Zone E control cores
(low suitability, should show no ADS signal)
```

### Phase 3: Targeted Investigation (if Phase 2 positive)

```
If ADS signal detected:
→ High-resolution coring grid around positive hit
→ GPR survey at depth (if <10m)
→ Possible pilot excavation (requires permit)
→ Paper 11
```

---

## 6. Required Collaborations

```
Geochemistry lab:
→ Phosphorus analysis
→ Strontium isotope (expensive, ~$50-100/sample)
→ UGM Geologi / ITB / PVMBG lab

Palynology/phytolith:
→ Phytolith identification requires specialist
→ LIPI Botany / herbarium connection?
→ Or: ANU SEA archaeology program

Radiocarbon dating:
→ AMS facility needed
→ BATAN (Indonesia) has AMS capability
→ Or: outsource to ANSTO (Australia)
      or Beta Analytic (Florida)

Field access:
→ PVMBG for existing cores
→ BPCB for known site areas
→ Local landowner permission for new cores
```

---

## 7. Why This Paper Matters Beyond VOLCARCH

```
If ADS methodology works in volcanic Java:

→ Transferable to ALL volcanic archaeological
  landscapes globally:
  
  Central America (Maya lowlands + 
                   volcanic highlands)
  East Africa (Rift Valley volcanics)
  Mediterranean (Santorini, Vesuvius)
  Japan (volcanic plains, Jomon culture)
  New Zealand (Maori + volcanic North Island)

→ This is a GENERAL METHOD paper
  dressed as a JAVA paper

→ Target journal: Quaternary Science Reviews
  or Journal of Archaeological Science
  (not Antiquity — too method-heavy)

→ Impact: every geoarchaeologist working
  in volcanic landscape will cite this
```

---

## 8. Significance Statement (draft)

The absence of pre-400 CE archaeological evidence from volcanic Java 
has long been interpreted as evidence of cultural absence. VOLCARCH 
Papers 1–9 demonstrate that taphonomic burial provides an alternative 
explanation. This paper completes the framework by proposing the first 
systematic methodology for detecting positive evidence of buried 
occupation through geochemical disequilibrium signatures preserved 
in volcanic stratigraphy. The approach — combining machine-learning 
site prediction (Paper 3) with multi-proxy geochemical analysis — 
offers a pathway to testing whether the absence of evidence is 
indeed evidence of absence, or merely the expected result of 
landscape-scale taphonomic processes operating over millennia.

---

## 4b. Tephrochronology as Stratigraphic Calendar

### The Iceland Principle Applied to Java

Iceland archaeology uses volcanic ash layers as precise dating tools —
each eruption creates a dateable horizon, and archaeological deposits
BETWEEN layers are chronologically bracketed without radiocarbon dating.

Java has the same capability, but has never been systematically applied
to deep-time archaeological investigation.

**The Java Tephra Calendar (Jawa Timur):**

```
Depth (approx)  | Event              | Date        | Datable?
────────────────────────────────────────────────────────────
~2 cm           | Kelud 2014         | 2014 CE     | ✓ exact
~8 cm           | Kelud 1990         | 1990 CE     | ✓ exact
~30 cm          | Krakatau ash       | 1883 CE     | ✓ exact
~72 cm          | Tambora ash        | 1815 CE     | ✓ exact
~80 cm          | Samalas/Rinjani    | 1257 CE     | ✓ ±5yr
~185 cm         | [Dwarapala anchor] | ~1268 CE    | ✓ historical
[unknown]       | Tengger cycles     | variable    | ✓ radiometric
~266 m          | Toba ash layer     | ~74,000 BP  | ✓ radiometric
```

**Why Toba Matters for VOLCARCH:**

Toba eruption (~74,000 BP) deposited detectable cryptotephra across
ALL of Southeast Asia including Java. This creates a hard stratigraphic
floor: any anthropogenic signal ABOVE the Toba layer = post-74ka human
activity. Any signal BELOW = pre-Toba, which would be extraordinary.

Independent verification of Paper 7 burial depth estimate:
```
74,000 years × 3.6 mm/yr = 266 meters

Paper 7 estimate from biogeographic argument: 163-326 meters
Tephrochronology estimate: ~266 meters

CONVERGENCE FROM TWO INDEPENDENT METHODS ✓
```

**Operational Implication for Paper 10 Phase 1:**

Before running ADS geochemical analysis on soil cores, identify
tephra layers first. This provides:
1. Free chronological framework (no radiocarbon needed for bracketing)
2. Independent depth calibration vs Paper 1 burial model
3. Stratigraphic context for any anthropogenic signal found

**Key reference:** Dugmore et al. (2007) tephrochronology methodology
(Iceland) — directly transferable protocol to Java context.

---

## Notes / Parking Lot

- Check: phytolith survival rates in volcanic soil specifically — 
  some literature suggests alkaline volcanic ash actually ENHANCES 
  preservation (opposite of acidic taphonomy concern)
- Liangan site (sand mining discovery 2008) — cores from adjacent 
  areas? This is the most accessible deeply buried site
- LiDAR coverage of East Java — what exists already?
- BATAN AMS capability — what is current turnaround + cost?
- Phosphorus survey of Majapahit heartland (Trowulan) — 
  already done? Literature check needed
- Consider: drone multispectral for crop mark detection 
  in Zone B/C areas — low cost, non-invasive first pass

---

## Paper Series Position

```
Paper 1  (intarch):   Sedimentation rates — THE BURIAL CLOCK
Paper 3  (jrs):       XGBoost model — WHERE TO LOOK  
Paper 7  (Antiquity): Spatial segregation — THE SIGNAL IS REAL
Paper 9  (draft):     Peripheral substrate — CULTURAL EVIDENCE
Paper 10 (this):      ADS methodology — HOW TO FIND IT
Paper 11 (future):    Field results — WHAT WE FOUND
```

---
*Draft v0.1 — concept capture 2026-03-06*  
*Core idea: archaeological disequilibrium signature (ADS) =*  
*biosignature logic applied to buried civilizations*  
*Next: literature check on phosphorus survey methodology,*  
*phytolith survival in volcanic soil, PVMBG core access*
