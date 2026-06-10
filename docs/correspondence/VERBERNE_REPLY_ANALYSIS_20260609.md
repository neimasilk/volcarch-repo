# Verberne reply — objective analysis of the two questions

**Date:** 2026-06-09
**Purpose:** Standing objective record behind the reply draft (`EMAIL_VERBERNE_REPLY_DRAFT_20260609.md`). Facts + risks, no spin.
**Trigger:** Verberne email 2026-06-08, two questions after consulting a TU Delft colleague.

---

## Q1 — "Not familiar with the proposed dataset … ready for NER?"

### Finding 1 — The dataset IS GLOBALISE (confirmed), and the repo files are genuine
- Repo files `data/processed/globalise_voc/clean_1053.txt …` and `clean_10098A …10130B` match GLOBALISE's official released inventory ranges (**inv. 1053–4454 + 7527–11024**).
- GLOBALISE provenance (verified 2026-06-09): ~5 million handwritten VOC pages ("Overgekomen brieven en papieren"), transcribed by **HTR (Loghi), not OCR**; released **CC0**; downloadable as running text, one file per inventory number; Huygens Institute (KNAW).
- So the data is real, legal, machine-readable. The PI processed a 500-file subset (E211): 548,929 paragraphs, ~146M words.

### Finding 2 — "Not familiar with the dataset" is partly a PROPOSAL-CLARITY problem, not just her unfamiliarity
- Proposal §Corpus lists **three** corpora (Delpher + GLOBALISE + Nationaal Archief) without designating one primary working set.
- The brief's headline "~22,000 references … VOC archive … 1602–1800" is **misattributed**: the 22,162 mentions are from **E091 = OV (Oudheidkundig Verslag), 1912–1929**, not VOC 1602–1800. Dates don't match the source. A careful reader cannot tell what the working corpus is.
- **Action:** name the dataset crisply with provenance + readiness; fix this misattribution in future proposal versions.

### Finding 3 — GLOBALISE ALREADY does NER → RQ1 redundancy risk (the sharp one)
- GLOBALISE itself builds/releases NER for VOC entities: **persons, places, commodities, ships, events** (published fine-grained NER; NER datasets on their Dataverse/GitHub).
- Vossen (named in proposal as collaborator) is the GLOBALISE PI.
- **Risk:** if RQ1 = "generic NER for colonial Dutch," it reads as duplicating GLOBALISE's own pipeline.
- **Resolution:** the PI's genuinely-novel entity types are archaeology-specific — **DEPTH, MATERIAL, FIND_EVENT, SOIL_CONTEXT** — which are NOT in GLOBALISE's schema.

### Finding 4 — But those novel entities are SPARSE in VOC, DENSE in OV/Delpher (E211)
- E211: early VOC dagregisters are trade/admin, archaeology-thin (naive keyword precision <15%; pagode=currency, arca=Latin "chest", Candi=Kandy).
- The archaeology-rich corpus is the later printed material (OV 1912–1949, Delpher 1854–1942) — **not GLOBALISE**.

### Corpus ↔ RQ mapping (objective, defensible)
| RQ | Right corpus | Position vs GLOBALISE |
|----|-------------|----------------------|
| RQ1 archaeology NER (depth/material/find) | OV + Delpher (dense) | Complementary — entities they don't cover |
| RQ3 place-name disambiguation + settlement geography | GLOBALISE VOC | Extends their place entities, not competes |
| RQ4 depth validation | OV/Delpher depth records | — |

### Q1 reply requirements
1. Name GLOBALISE explicitly + provenance (HTR/Loghi/CC0/5M pages/inv ranges).
2. Acknowledge GLOBALISE already does base-entity NER → reframe it as foundation, not competitor.
3. Position contribution = archaeology entities + settlement geography + physical validation, in the right corpus.
4. Honest readiness: machine-readable yes; turnkey NER-ready no (HTR noise, early-modern orthography, code-switching, domain shift; <15% naive baseline).

### Open item before finalising Q1
- Read GLOBALISE NER paper for EXACT entity schema + performance, so "what they don't cover" is precise:
  `https://anthology.ach.org/volumes/vol0003/fine-grained-named-entity-recognition-for-east/10.63744@DRbhWNTzqNzR.pdf`

### Sources (verified 2026-06-09)
- VOC transcriptions — GLOBALISE (KNAW Pure): https://pure.knaw.nl/portal/en/datasets/voc-transcriptions-globalise/
- Contribute data — GLOBALISE (entity types): https://globalise.huygens.knaw.nl/contribute-data/
- Fine-grained NER for the East-India Company archives (paper): https://anthology.ach.org/volumes/vol0003/fine-grained-named-entity-recognition-for-east/10.63744@DRbhWNTzqNzR.pdf
- GLOBALISE GitHub: https://github.com/globalise-huygens

---

## Q2 — "Physical validation requires archaeologists … your approach?"

### The question's real content (three worries behind one question)
1. **Supervisory scope** — an NLP professor cannot supervise a dig or guarantee an archaeologist.
2. **Scientific validity** — validating NLP against a *model* is not ground truth; risk of circularity.
3. **Flexibility** — her opening reservation; both her questions target the archaeology-dependent parts.

### Rejected approach: "build an AI framework so we don't need archaeologists"
PI floated this. Rejected, objectively, for three reasons:
1. **Misreads the question** — she asks how you'll *handle* validation, not how to *eliminate* it. Answering "AI so I don't need archaeologists" confirms her worst read (computational person substituting model output for ground truth).
2. **Worsens circularity** — model-validating-model removes the last tether to physical reality → echo chamber. Repo already carries `feedback_confirmation_architecture` risk + the Submission Integrity Gate. **E214 palynology just disconfirmed part of the thesis** = the disconfirmation channel is real and working; removing it is anti-science.
3. **Contradicts Verberne's own research model** — ArcheoBERTje ("Can BERT dig it?") was built *with* archaeologists (Brandsen, Lambers, Wansleeben — her co-authors, cited in the PI's own proposal references). Pitching expert-replacement to her is tone-deaf.

**Governing distinction:** you can avoid **new fieldwork**; you cannot avoid **all domain expertise** without losing credibility.

### Adopted approach: reframe as institutional fit (two-way value), two-tier
- **Reframe:** validation-by-expert is not a weakness coped with — it is *why Leiden fits*, and where the PI brings what Leiden lacks.
- **Leiden has:** historical NLP (Verberne) + computational archaeology (Lambers/Wansleeben) co-located. Rare (VU = NLP without archaeology faculty; Edinburgh = pure informatics). → in-house validation.
- **PI brings:** a credible pathway to Indonesian field institutions — **UGM** (archaeology dept in the Central Java volcanic zone, near Liangan) and **BRIN** (absorbed Arkenas). Credible *because PI is an Indonesian academic*; impossible for a Europe-only project. Field reality + ground-truth + postcolonial/ethical grounding.
- **Two tiers:**
  - *Near (guaranteed):* Leiden in-house lightweight expert audit of a sample + existing-data validation (model / cross-source / published reports). **Thesis completes here.**
  - *Far (prospective, enrichment):* UGM/BRIN. Honestly **not yet formal** — a pathway to build. Future fieldwork/impact, never a deliverable.
- **AI's role:** efficiency multiplier for scarce expertise (ranked falsifiable predictions; ~3.5× search efficiency, E116/E118), **not** replacement.

### Honesty guardrails (binding)
1. State Indonesian access as **prospective** — do not claim a partnership that doesn't exist (Verberne knows Vossen; she can check).
2. Keep collaboration as **enrichment, not critical path** — else it re-creates the dependency she flagged.

### Strategic bonus (whole application, not just Q2)
PI = **bridge** between Dutch archival infrastructure (GLOBALISE/Leiden) and Indonesian field institutions (UGM/BRIN). Genuine candidate differentiator; aligns with funder demand (NWO/Horizon/MSCA) for local partners + postcolonial ethics in heritage projects.

### Naming decision (resolved 2026-06-09)
- Name **UGM + BRIN** explicitly (command of the Indonesian terrain).
- Reference Leiden archaeology via the **ArcheoBERTje connection**, NOT by naming Lambers/Wansleeben (avoids appearing to arrange her household). Verify their availability privately before relying on them.

### Q2 reply bottom line
Thesis completes as a computational thesis (existing-data validation + light in-house audit). Indonesian collaboration = upside the PI is positioned to pursue. New survey = future work. AI = efficiency multiplier, not replacement.
