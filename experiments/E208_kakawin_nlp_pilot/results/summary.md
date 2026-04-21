# E208 Phase 1 — OJW Domain Classification Results

**Date:** 2026-04-20 (autonomous execution)
**Input:** `data/raw/old_javanese_wordnet/wn-kaw.tab` (5019 synsets)
**Method:** Lookup each OJW synset in Princeton WordNet 3.0 by (pos, offset); classify via WordNet lexname → VOLCARCH 9-domain schema.

## POS Distribution (input)

| POS | Count |
|---|---:|
| n (noun) | 3229 |
| a (adjective) | 1081 |
| v (verb) | 695 |
| r (adverb) | 14 |

Total: 5019

## Match Rate

- Princeton WordNet 3.0 lookup hits: **5018 of 5019 (100.0%)**
- Misses (synset offsets not in WordNet 3.0): **1 (0.0%)**

Interpretation: OJW synset offsets that do not resolve in WordNet 3.0 likely reflect version drift (OJW built against a specific WordNet release whose offsets may differ from NLTK's WordNet 3.0). These misses do not indicate bad data; they would require rebuilding against the exact matching WordNet release. For domain distribution analysis, the hits are representative.

## Domain Distribution (VOLCARCH 9-domain schema)

| Domain | OJW Count | OJW % of hits | E058 % of 189 (for comparison) |
|---|---:|---:|---:|
| Attributes | 1453 | 29.0% | — |
| Craft/Technology | 626 | 12.5% | 12.2% |
| Social/Governance | 548 | 10.9% | 11.1% |
| Actions/States | 493 | 9.8% | — |
| Knowledge/Cognition | 465 | 9.3% | 5.8% |
| Body/Medicine | 442 | 8.8% | 6.9% |
| Nature/Environment | 430 | 8.6% | 14.3% |
| Spatial/Navigation | 237 | 4.7% | 7.4% |
| Agriculture/Plants | 180 | 3.6% | 7.7% |
| Ritual/Cosmology | 144 | 2.9% | 17.5% |

## Key Comparative Observations (OJW vs E058)

E058 used 189 literary terms curated by frequency in Old Javanese literature (Zoetmulder, Kakawin). OJW has 5,020 synsets covering the full dictionary vocabulary — richer and broader.

**Notable differences:**

- **Agriculture/Plants**: OJW 3.6% vs E058 7.7% (SMALLER in OJW)
- **Attributes**: OJW 29.0% vs E058 0.0% (LARGER in OJW)
- **Nature/Environment**: OJW 8.6% vs E058 14.3% (SMALLER in OJW)
- **Knowledge/Cognition**: OJW 9.3% vs E058 5.8% (LARGER in OJW)
- **Fishing/Maritime**: OJW 0.0% vs E058 5.3% (SMALLER in OJW)
- **Actions/States**: OJW 9.8% vs E058 0.0% (LARGER in OJW)
- **Ritual/Cosmology**: OJW 2.9% vs E058 17.5% (SMALLER in OJW)

## Top 10 Lexname Categories (finer-grained)

| Lexname | Count | Pct |
|---|---:|---:|
| adj.all | 1079 | 21.5% |
| noun.artifact | 474 | 9.4% |
| noun.person | 331 | 6.6% |
| noun.animal | 228 | 4.5% |
| noun.communication | 210 | 4.2% |
| noun.act | 205 | 4.1% |
| noun.object | 204 | 4.1% |
| noun.attribute | 177 | 3.5% |
| noun.body | 166 | 3.3% |
| noun.cognition | 163 | 3.2% |

## Sample Lemmas per Domain (first 5)

**Actions/States:**
- *abhyudaya* → increase.n.03 — a process of becoming larger or longer or more numerous or more important
- *abhyudaya* → prosperity.n.01 — an economic state of growth with rising profits and full employment
- *adhyāya* → lesson.n.01 — a unit of instruction
- *adhyāya* → lecture.n.03 — teaching by giving a discourse on some subject (typically to a class)
- *adyan-adyan* → blessing.n.01 — the formal act of approving

**Agriculture/Plants:**
- *aṇḍa* → egg.n.02 — oval reproductive body of a fowl (especially a hen) used as food
- *asin* → salt.n.02 — white crystalline form of especially sodium chloride used to season and preserve
- *bhojana* → meal.n.01 — the food served and eaten at one time
- *bwah* → fruit.n.01 — the ripened reproductive body of a seed plant
- *cacar* → meal.n.01 — the food served and eaten at one time

**Attributes:**
- *abrĕsih* → clean.a.01 — free from dirt or impurities; or having clean habits
- *abrĕsih* → pure.a.06 — (used of persons or behaviors) having no faults; sinless; - Sylvia Plath
- *abrĕsih* → pure.a.01 — free of extraneous elements of any kind
- *agöṅ* → large.a.01 — above average in size or number or quantity or magnitude or extent
- *agöṅ* → great.s.01 — relatively large in size or number or extent; larger than others of its kind

**Body/Medicine:**
- *ākṛti* → human_body.n.01 — alternative names for the body of a human being
- *aṇḍah* → duck.n.01 — small wild or domesticated web-footed broad-billed swimming bird usually having 
- *aṇḍaja* → bird.n.01 — warm-blooded egg-laying vertebrates characterized by feathers and forelimbs modi
- *aṇḍoja* → bird.n.01 — warm-blooded egg-laying vertebrates characterized by feathers and forelimbs modi
- *aṅguṣṭha* → thumb.n.01 — the thick short innermost digit of the forelimb

**Craft/Technology:**
- *ādhāra* → support.n.10 — any device that bears the weight of another thing
- *alaṅkara* → decoration.n.01 — something used to beautify
- *alaṅkṛta* → decoration.n.01 — something used to beautify
- *ālekhana* → drawing.n.02 — a representation of forms or objects on a surface by means of lines
- *ālekhana* → painting.n.01 — graphic art consisting of an artistic composition made by applying paints to a s

**Knowledge/Cognition:**
- *ābhādha* → pain.n.03 — a somatic sensation of acute discomfort
- *abhidhana* → name.n.01 — a language unit by which a person or thing is known
- *abhimata* → purpose.n.01 — an anticipated outcome that is intended or that guides your planned actions
- *abhiprāya* → purpose.n.01 — an anticipated outcome that is intended or that guides your planned actions
- *ādeśa* → guidance.n.01 — something that provides direction or advice as to a decision or course of action

**Nature/Environment:**
- *abdhi* → ocean.n.01 — a large body of water constituting a principal part of the hydrosphere
- *abu* → ash.n.01 — the residue that remains when something is burned
- *acala* → mountain.n.01 — a land mass that projects well above its surroundings; higher than a hill
- *adri* → mountain.n.01 — a land mass that projects well above its surroundings; higher than a hill
- *aho* → day.n.01 — time for Earth to make a complete rotation on its axis

**Ritual/Cosmology:**
- *ābhādha* → pain.n.02 — emotional distress; a fundamental feeling that people try to avoid
- *abhilāṣa* → desire.n.01 — the feeling that accompanies an unsatisfied state
- *abhimata* → desire.n.01 — the feeling that accompanies an unsatisfied state
- *aṅgā* → desire.n.01 — the feeling that accompanies an unsatisfied state
- *ātura* → suffering.n.04 — feelings of mental or physical pain

**Social/Governance:**
- *abĕṭĕk* → cook.n.01 — someone who cooks food
- *abhyāgata* → guest.n.01 — a visitor to whom hospitality is extended
- *adhikārapuruṣa* → hero.n.01 — a man distinguished by exceptional courage and nobility and strength
- *adhipati* → king.n.01 — a male sovereign; ruler of a kingdom
- *agraśekhara* → general.n.01 — a general officer of the highest rank

**Spatial/Navigation:**
- *adhirājya* → capital.n.03 — a seat of government
- *ala* → side.n.04 — a surface forming part of the outside of an object
- *antya* → end.n.09 — a boundary marking the extremities of something
- *awasawya* → right.n.02 — location near or direction toward the right side; i.e. the side to the south whe
- *āyatanasthāna* → residence.n.01 — any address at which you dwell more than temporarily


## Interpretation for VOLCARCH

- The OJW domain distribution provides a FULL-CORPUS picture of Old Javanese vocabulary by semantic domain, where E058 only sampled 189 frequency-curated terms.
- If the OJW profile shows substantially richer coverage in Agriculture/Plants, Craft/Technology, and Body/Medicine than E058 implies, it strengthens the "indigenous material culture substrate" argument central to P0 Channel 3 (linguistic reconstruction).
- Unmapped lexnames (if any large) indicate our 9-domain schema may need extension; the WordNet lexname level is the fallback.
- **This pilot is exploratory**: the OJW was built from Zoetmulder dictionary (1982), which has its own curatorial biases. A fuller analysis would require Phase 2 (Sanskrit-vs-native tagging) and Phase 3 (frequency-weighted kakawin corpus NLP).

---
*Produced autonomously by Claude, E208 Phase 1, 2026-04-20. Verified manually before citation.*
