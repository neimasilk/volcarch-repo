# VOLCARCH Paper 12 — DRAFT v0.1
# Computational Mythology: A Classification Framework
# for Distinguishing Pre-Hindu Substrate Residue 
# from Hindu Overlay in Nusantaran Oral Tradition

**Status:** Concept draft — ideas capture, not yet for submission
**Date:** 2026-03-07
**Author:** Mukhlis Amien, Lab Data Sains, Universitas Bhinneka Nusantara
**Cross-ref:** Papers 8, 9, 11
**NLP angle:** This is fundamentally a binary classification problem

---

## The Core Insight

Humanists ask: "Is this myth indigenous or imported?"
This paper answers: "This is a classification problem. Here is the classifier."

```
INPUT:  One unit of mythology / ritual / narrative
OUTPUT: P(residue) vs P(overlay)

Features:
1. Indian corpus parallel (yes/no + similarity score)
2. Malagasy presence (yes/no)
3. Geographic distribution (central/peripheral)
4. Sanskrit vocabulary ratio (0.0 - 1.0)
5. Volcanic/natural phenomena linkage (yes/no)
6. Pre-400 CE attestation (yes/no)
7. Multi-region distribution (count)

This is a tractable NLP + cultural analytics task.
The humanities community has not framed it this way.
We do.
```

---

## 1. Introduction

### 1.1 The Problem

When Hindu-Buddhist influence arrived in Nusantara (~400 CE),
it did not arrive into a cultural vacuum. It arrived into a
landscape already inhabited by populations with their own
cosmologies, rituals, narratives, and oral traditions —
traditions that had been shaped by tens of thousands of years
of volcanic landscape pressure (Paper 11).

The resulting cultural record is a **palimpsest**: Hindu overlay
written on top of indigenous substrate. In some places the
overlay is thin and the substrate shows through clearly.
In others, the overlay is thick and the substrate is nearly
invisible. In most places, they are fused into something new
that is neither purely Hindu nor purely indigenous.

The standard humanistic approach — close reading of individual
texts, comparative mythology, historical linguistics — has
produced important insights but cannot scale to the full
corpus of Nusantaran oral tradition (thousands of texts,
multiple languages, multiple regional variants).

**We propose a computational classification framework**
that can process this corpus at scale, assign probability
scores to each unit, and identify the highest-confidence
substrate candidates for deeper investigation.

### 1.2 The Calibration Anchors

Before building a classifier, we need ground truth —
units we are confident about in BOTH directions.

```
HIGH-CONFIDENCE RESIDUE (Class A):
→ Slametan ritual (Paper 5/9):
  - Volcanic calibration confirmed
  - No Hindu canonical parallel
  - Present in peripheral communities
  - Non-Sanskrit core vocabulary
  - Malagasy parallel: TBD (research question)

→ Trunyan mepasah burial (Paper 9):
  - No Hindu canonical parallel
  - Bali Aga = pre-Majapahit community
  - Volcanic context explicit
  - Non-Sanskrit vocabulary throughout

HIGH-CONFIDENCE OVERLAY (Class B):
→ Ramayana Jawa version:
  - Direct Indian parallel (obvious)
  - Sanskrit-heavy vocabulary
  - Appears in court/prestige contexts
  - Not present in peripheral communities
  - Not in Malagasy (departed ~1200 CE,
    but Ramayana already present by then —
    interesting edge case)

→ Yupa inscriptions (Kutai, ~400 CE):
  - Explicitly Sanskrit
  - Direct Vedic ritual parallel
  - Power center origin
  - Clear dating
```

These anchors allow us to calibrate the classifier before
applying it to ambiguous cases.

---

## 2. The Two-Class Framework

### 2.1 Class A — Pre-Hindu Substrate Residue

Defining characteristics:

```
1. NO INDIAN PARALLEL
   Cross-reference with:
   - Mahabharata corpus
   - Ramayana corpus
   - Puranas (18 major + 18 minor)
   - Vedic hymns
   → Absence = strong Class A signal

2. MULTI-REGION NUSANTARA DISTRIBUTION
   Present in ≥3 of:
   [Jawa, Sulawesi, Sumatra, Kalimantan,
    Bali, Malagasy]
   → Wide distribution before Hindu contact
     = must predate Hindu contact

3. MALAGASY PRESENCE (strongest single test)
   If present in Malagasy:
   → Must predate ~1200 CE departure
   → If also absent from Indian corpus:
     → Almost certainly pre-Hindu substrate
   → This is the MOST POWERFUL single filter

4. PERIPHERAL COMMUNITY PRESERVATION
   Present in Bali Aga, Tengger, Osing,
   Tegal-Banyumas, Toraja, Muna
   → Peripheral conservatism (Paper 9)
   → Survived because peripheral communities
     received less Hindu overlay pressure

5. NON-SANSKRIT CORE VOCABULARY
   KawiKupas analysis:
   → Sanskrit ratio < threshold (TBD)
   → Vocabulary from Austronesian root stock
   → Or: vocabulary without ANY known etymology
     (= possible pre-Austronesian substrate)

6. VOLCANIC / NATURAL PHENOMENA LINKAGE
   Semantically connected to:
   → Volcanic events, mountains, fire
   → Decomposition, soil, earth
   → Tidal/coastal phenomena
   → Monsoon, seasonal patterns
   → (consistent with VCS hypothesis, Paper 11)
```

### 2.2 Class B — Hindu/Buddhist Overlay

Defining characteristics:

```
1. DIRECT INDIAN PARALLEL
   Identifiable source in Indian corpus
   → Obvious: Ramayana, Mahabharata
   → Subtle: specific deity attributes,
     ritual structure, narrative motif

2. SANSKRIT-HEAVY VOCABULARY
   KawiKupas Sanskrit ratio > threshold
   → Especially in key narrative terms
     (character names, ritual objects,
      cosmological concepts)

3. POWER CENTER ORIGIN
   Attested first in:
   → Court poetry (kakawin)
   → Royal inscriptions
   → Prestige religious contexts
   NOT in folk/peripheral contexts first

4. POST-400 CE ATTESTATION ONLY
   No evidence in record before
   Hindu contact period
   → Though absence of evidence ≠
     evidence of absence (taphonomy!)
     → requires additional confirmation

5. ABSENT FROM MALAGASY
   If not present in Malagasy tradition:
   → Either entered after ~1200 CE
   → Or was not portable to new settlement
   → Weaker Class B signal (not definitive)
```

### 2.3 Class C — Syncretic (Neither Pure)

Reality: most Nusantaran mythology is neither purely A nor B.
The classifier should output a PROBABILITY DISTRIBUTION:

```
P(Class A) + P(Class B) + P(Class C) = 1.0

Examples:

Nyai Roro Kidul:
P(A)=0.75, P(B)=0.15, P(C)=0.10
→ Probably pre-Hindu sea deity,
  partially Hinduized in retelling

Wayang Panji cycle:
P(A)=0.80, P(B)=0.10, P(C)=0.10
→ Highest Class A score of any
  major narrative tradition

Ramayana Jawa:
P(A)=0.05, P(B)=0.85, P(C)=0.10
→ Clear overlay, minor local adaptation

Slametan:
P(A)=0.90, P(B)=0.05, P(C)=0.05
→ Highest Class A score overall
  (calibration anchor confirmed)
```

---

## 3. Test Cases

### 3.1 Cerita Panji — The Strongest Substrate Candidate

**Why Panji is the gold standard:**

```
Cerita Panji is a cycle of romantic adventure
stories centered on Prince Panji Asmarabangun
and Princess Candra Kirana — their separation,
disguises, and eventual reunion.

Class A tests:
□ Indian parallel?    → NONE
  (exhaustively searched by Dutch scholars
   since 1800s — no source found in India)
□ Multi-region?       → YES
  Jawa, Bali, Kamboja, Thailand, Malaysia,
  Philippines — widest distribution of ANY
  Nusantaran narrative tradition
□ Malagasy presence?  → UNKNOWN (research question)
□ Peripheral?         → YES, survives in Bali,
  Banyuwangi (Osing), Tengger variants
□ Sanskrit vocabulary? → LOW
  Core names: Panji (Javanese), 
  Asmarabangun (Sanskrit — overlay?)
  Candra Kirana (Sanskrit — overlay?)
  But narrative structure: non-Sanskrit
□ Volcanic linkage?    → INDIRECT
  Panji's wandering = displacement narrative?
  = encoded memory of volcanic disruption
  forcing population movement?

VERDICT: P(Class A) = 0.80-0.85
Cerita Panji = highest-confidence pre-Hindu
narrative tradition in Nusantara

Research priority:
→ KawiKupas on Panji texts:
  Sanskrit ratio vs Ramayana Jawa
→ Malagasy search: any Panji parallel?
→ If Malagasy has Panji = pre-1200 CE confirmed
→ If not = entered 1200-1500 CE window
```

### 3.2 Nyai Roro Kidul — The Displaced Deity

```
Sea goddess of the South Sea (Laut Selatan)
Every Sultan of Yogyakarta is her "husband"
Controls the Indian Ocean south of Java

Class A tests:
□ Indian parallel?    → NONE
  No Hindu sea goddess with this profile
  Dewi Sri = rice goddess (different)
  Varuna = sea god but male + Vedic
□ Multi-region?       → YES
  Jawa (all variants), Sunda, Bali
□ Malagasy?           → UNKNOWN
□ Peripheral?         → YES
  Strongest among coastal/fishing communities
  NOT primarily a court deity
□ Sanskrit vocab?     → LOW
  Nyai (Javanese/Malay) + Roro (OJ) +
  Kidul (Javanese "south") = 0% Sanskrit
□ Volcanic linkage?   → YES (strong)
  Associated with:
  - Krakatau eruption narratives
  - Coastal volcanic activity
  - Ratu Kidul = ruler of displaced
    (Dewata Cengkar went to sea — Paper 12 §4)

VERDICT: P(Class A) = 0.75
Nyai Roro Kidul = encoded memory of:
(a) Pre-Hindu coastal deity
(b) Displaced pre-Hindu population
    (Dewata Cengkar → sea → Ratu Kidul)
(c) Volcanic coastal phenomena
```

### 3.3 Dewata Cengkar — The Cannibal King

```
Pre-Aji Saka ruler of Java
Characterized as: cannibal, tyrant, raksasa
Defeated by: Aji Saka (incoming Hindu culture)
Fate: pushed into the sea

Class A tests:
□ Indian parallel?    → PARTIAL
  "Evil king defeated by civilized hero"
  is universal motif — but specific
  details (cannibal + sea fate) = no Indian source
□ Decoding:
  "Cannibal" = possibly misrepresentation of
  exposed burial practice (Paper 9 §2.2)
  "Pushed to sea" = population displacement
  to coastal/maritime zones
□ Volcanic linkage?   → INDIRECT
  Raksasa = mountain/earth beings
  in pre-Hindu cosmology

VERDICT: P(Class A) = 0.65
Dewata Cengkar = real historical memory of
pre-Hindu Javanese population encounter
with Hindu Indian cultural influence,
encoded as colonization myth
```

### 3.4 Gunungan (Kayon) — The Cosmic Mountain

```
The tree/mountain symbol that opens and
closes every wayang performance.
Shape: mountain with tree, surrounded by
       fire, animals, supernatural beings

Class A tests:
□ Indian parallel?    → PARTIAL
  Mount Meru in Hindu cosmology = mountain
  as cosmic axis — but Gunungan's specific
  iconography has no direct Indian source
□ Volcanic linkage?   → EXTREMELY STRONG
  Gunungan = stylized VOLCANO
  Fire surrounding = volcanic fire
  Animals fleeing = eruption behavior
  Tree on summit = vegetation above
  treeline (pre-eruption calm)
□ Function: marks transitions (beginning/end)
  → Volcano as temporal marker
  → "World resets when volcano speaks"
  → VCS hypothesis (Paper 11) confirmed
    in iconographic form

VERDICT: P(Class A) = 0.70
Gunungan = pre-Hindu volcanic cosmology
incorporated into Hindu wayang frame
The most visually obvious substrate residue
hiding in plain sight in every wayang
performance
```

---

## 4. The Raksasa Question: Giants as Encoded History

### 4.1 The Universal Pattern

Across world mythologies, incoming cultural groups
consistently encode existing populations as:

```
Greek:    Titans, Cyclops, Giants
          → pre-Greek Mediterranean populations

Biblical: Nephilim, Goliath, Anakim
          → pre-Israelite Canaan populations

Norse:    Jotnar (Frost Giants)
          → pre-Norse Scandinavian populations

Vedic:    Asura, Rakshasa
          → pre-Vedic Indian populations
          (Varuna was originally Asura →
           reclassified as Deva later)

Javanese: Raksasa, Buto
          → pre-Hindu Javanese populations
```

This is **narrative technology of colonization** —
not unique to any culture, but universal.

### 4.2 What "Giant" Actually Encodes

```
"Giant" characteristics in mythology
decoded as colonial narrative elements:

LARGE BODY:
→ "They were here before us"
→ Physical power = prior territorial claim
→ We had to overcome something formidable

CANNIBAL / VIOLENT:
→ Dehumanization justifies displacement
→ "They ate people" = we were right to replace them
→ Possibly: misinterpretation of
  exposed burial practice (Paper 9)

PRIMITIVE / STUPID (often):
→ "They lacked civilization"
→ Justifies cultural replacement
→ "We brought writing, law, agriculture"
  (= Aji Saka narrative exactly)

ASSOCIATED WITH EARTH / MOUNTAINS:
→ They belong to the "old world"
→ Chthonic (earth-based) vs
  Hindu sky/divine orientation
→ Consistent with pre-Hindu volcanic
  landscape cosmology (Paper 11)
```

### 4.3 The Asura Import

Hindu colonizers arrived with their OWN
pre-existing "giant/demon" mythology:

```
India: Vedic Aryan arrival →
       existing Dravidian/pre-Vedic
       populations = Asura/Rakshasa

Same narrative, exported to Nusantara:

Java: Hindu arrival →
      existing Javanese population
      = Raksasa (using same category)

This means Java's "raksasa" problem is
DOUBLY encoded:
1. Real pre-Hindu Javanese population
2. Filtered through Indian narrative
   template that ALREADY had a category
   for "the people we replaced"

Disentangling these two layers requires:
→ Finding raksasa narratives that
  DON'T match Indian templates
→ Local-specific details = real memory
→ Indian template details = overlay
```

---

## 5. Computational Pipeline

### 5.1 Corpus Construction

```
MYTHOLOGY CORPUS (Nusantara):
Source 1: Babad Tanah Jawi
          (Javanese "history" = mythology)
Source 2: Cerita Panji cycle
          (all regional variants)
Source 3: Wayang stories NOT in
          Indian Mahabharata/Ramayana
          (Javanese original episodes = carangan)
Source 4: Kidung (Old Javanese poetry,
          some pre-Hindu)
Source 5: La Galigo (Bugis epic —
          possibly largest oral literature
          in the world, ~6000 pages)
Source 6: Malagasy oral tradition
          (Tantara ny Andriana,
           Fomba Malagasy)

CONTROL CORPUS (Indian):
Source 7: Mahabharata (full)
Source 8: Ramayana (full)
Source 9: Major Puranas
→ Used for parallel detection
```

### 5.2 Feature Extraction

```
For each narrative unit:

F1: Indian_parallel_score (0.0-1.0)
    → Semantic similarity to Indian corpus
    → Using multilingual sentence embeddings

F2: Malagasy_presence (0/1)
    → Exact or near-match in Malagasy corpus

F3: Geographic_spread (integer 1-8)
    → Number of distinct regions with variant

F4: Sanskrit_ratio (0.0-1.0)
    → KawiKupas applied to key vocabulary
    → Proportion of Sanskrit-derived terms

F5: Volcanic_semantic_linkage (0.0-1.0)
    → Semantic similarity to volcanic
      phenomena vocabulary
    → Custom dictionary: gunung, kawah,
      lahar, abu, api, tanah, lava...

F6: Peripheral_attestation (0/1)
    → Present in Bali Aga / Tengger /
      Osing / Tegal / Toraja / Muna

F7: Pre400_attestation (0/1)
    → Any evidence before 400 CE
    → Mostly 0 (taphonomy problem)
    → But Malagasy can substitute as
      pre-1200 CE proxy
```

### 5.3 Classification Model

```
Simple logistic regression first
(interpretable, defensible to reviewers):

P(Class A) = sigmoid(β₀ + 
  β₁×F1_inverse +    # low Indian parallel
  β₂×F2 +            # Malagasy presence
  β₃×F3 +            # geographic spread
  β₄×F4_inverse +    # low Sanskrit ratio
  β₅×F5 +            # volcanic linkage
  β₆×F6)             # peripheral attestation

Training data: calibration anchors
(slametan, mepasah = Class A gold standard;
Ramayana Jawa, Yupa = Class B gold standard)

Validation: cross-reference with
expert judgment from existing
comparative mythology literature
```

### 5.4 Expected Outputs

```
HIGH P(Class A) candidates (predictions):
→ Cerita Panji: ~0.80-0.85
→ Nyai Roro Kidul: ~0.75
→ Gunungan: ~0.70
→ Specific La Galigo episodes: TBD
→ Malagasy ritual parallels: TBD

HIGH P(Class B) (should confirm):
→ Ramayana Jawa: ~0.85
→ Arjuna Wiwaha (kakawin): ~0.80
→ Nagarakretagama royal sections: ~0.75

INTERESTING MIDDLE CASES:
→ Dewata Cengkar: ~0.65 (A)
   (hybrid of real memory + Indian template)
→ Batara Guru (Javanese Shiva):
   Hindu origin but heavily localized
   → How much of Batara Guru is pre-Hindu?
```

---

## 6. The Slametan-Mythology Bridge

This paper's most important connection to
the broader VOLCARCH framework:

```
VOLCARCH has established:
Slametan = pre-Hindu volcanic substrate
           (Papers 5, 9, 11)

This paper asks:
Which MYTHOLOGICAL elements share
the same substrate origin as slametan?

Prediction:
Mythological units with high P(Class A)
should have SEMANTIC OVERLAP with
slametan's volcanic calibration themes:

- Time and waiting (1000 days)
- Decomposition and renewal
- Community obligation and sharing
- Volcanic soil and fertility
- Mountain as sacred/dangerous axis

If Cerita Panji, Nyai Roro Kidul, and
Gunungan all semantically cluster WITH
slametan (not with Ramayana):
→ They share substrate origin
→ They form a COHERENT pre-Hindu
  cosmological system
→ That system was volcanically shaped
  (Paper 11 VCS confirmed mythologically)

This is the full circle:
Geology (Papers 1,7) →
  Archaeology (Papers 3,10) →
    Culture (Papers 9,11) →
      Mythology (Paper 12) →
        All driven by same volcanic mechanism
```

---

## 7. Significance

```
Standard view:
"We cannot know pre-Hindu Nusantaran
 mythology because there are no
 pre-400 CE written sources"

This paper:
"Written sources are not the only evidence.
 Computational analysis of:
 (a) what is absent from Indian corpus
 (b) what is present in Malagasy corpus
 (c) what survives in peripheral communities
 (d) what has low Sanskrit vocabulary ratio
 
 can identify pre-Hindu substrate with
 quantifiable confidence scores —
 despite zero pre-400 CE written attestation"

Taphonomy argument applied to mythology:
Just as volcanic burial explains absence of
physical archaeological evidence (Papers 1-7),
cultural overlay explains absence of
written mythological evidence.

But both have residues.
And residues can be found computationally.
```

---

## 8. Ideas Parking Lot

- La Galigo (Bugis epic): possibly world's
  longest oral literature, ~6000 pages,
  pre-Islamic, set in a cosmology with
  ZERO Hindu pantheon → strongest
  candidate corpus after Panji
  
- Carangan wayang: Javanese-original episodes
  NOT in Indian Mahabharata — what are these?
  Systematically unanalyzed computationally

- Batara Kala (time-devouring deity):
  Kala = Sanskrit (time) but Batara Kala's
  specific role in Javanese cosmology
  (eating people born on wrong calendar days)
  has no direct Indian parallel →
  Class A candidate in Hindu dress

- Ruwatan ritual (purification from Batara Kala):
  pre-Hindu structure, post-Hindu vocabulary →
  good test case for syncretic Class C

- Check: does any version of Cerita Panji
  appear in Malagasy tradition?
  This single data point would transform
  our understanding of the narrative's age

- Gamelan tuning systems: some researchers
  note pelog scale does not appear in
  Indian classical music → pre-Hindu?
  Computational ethnomusicology angle?

---

## Paper Series Position

```
Paper 8:  Linguistic substrate (Sulawesi 6 languages)
Paper 9:  Peripheral conservatism (ritual + botanical)
Paper 11: Volcanic Cultural Selection (VCS)
Paper 12: Computational mythology (THIS)
          — mythology as fourth evidence channel

Together Papers 9 + 11 + 12 form a trilogy:
PHYSICAL substrate (ritual, botanical)
BEHAVIORAL substrate (volcanic adaptation)
NARRATIVE substrate (mythology classification)

All converging on same conclusion:
Pre-Hindu Nusantaran civilization was real,
sophisticated, volcanically shaped,
and recoverable through computational methods
despite zero direct written attestation
```

---
*Draft v0.1 — concept capture 2026-03-07*
*Core method: binary classification of mythology units*
*Key test case: Cerita Panji (no Indian parallel,*
*widest geographic distribution, low Sanskrit ratio)*
*Critical single data point: does Cerita Panji*
*exist in Malagasy tradition? Check immediately.*
*Next: La Galigo corpus access,*
*carangan wayang episode list,*
*Malagasy narrative database search*
