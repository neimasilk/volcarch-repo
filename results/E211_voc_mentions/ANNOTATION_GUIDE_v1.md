# E211 Phase 2 — Annotation Guide v1.0

**Task:** Annotate VOC colonial Dutch sentences for archaeological mention relevance.
**File to annotate:** `annotation_sample_v1.csv` (65 sentences, Phase 2 pilot)
**Goal:** Estimate precision of Phase 1 extraction + build training data for NER

---

## Annotation Fields

For each sentence in `annotation_sample_v1.csv`, fill in three columns:

### `ann_is_archaeological`
- `YES` — the sentence genuinely describes an archaeological feature or observation (a monument, grave, ruin, artifact, or inscription that is physically present in the landscape and dates to pre-Dutch or early colonial period)
- `NO` — false positive (keyword matches but the sentence is about trade, administration, navigation, or non-archaeological context)
- `MAYBE` — ambiguous; the keyword might refer to something archaeological but the context is unclear

### `ann_correct_types`
If `ann_is_archaeological = YES`, specify which entity types are correctly identified. Use the same labels:
- `MONUMENT` — standing structure (temple, shrine, stupa, idol, statue)
- `GRAVE` — burial feature (grave pit, tomb, burial mound)
- `RUIN` — collapsed/abandoned structure
- `ARTIFACT` — portable object (coin, ceramic, metal object, relic)
- `INSCRIPTION` — inscribed object (stone tablet, metal plate, inscribed column)
- `DEPTH` — depth measurement in archaeological context

If the auto-detected type is wrong but something IS archaeological, write the correct type(s).

### `ann_notes`
Free text. Explain WHY it's YES/NO/MAYBE, or note any ambiguity. Especially note:
- What the keyword actually refers to in context
- Geographic location (Java? Ceylon? India? Unspecified?)
- Approximate date if inferable from context

---

## Decision Rules

### When to say NO (false positive patterns to reject)

| Pattern | Reason |
|---|---|
| `pagode` + price/quantity (rd., gulden, pag.) | "pagode" = gold coin, not temple |
| `arca` in Latin/Portuguese/Spanish sentence | "arca" = chest/box, not statue |
| `opschrift` on a letter or administrative document | "opschrift" = letter heading, not inscription |
| `Candi` near "Koninck", "Ceylon", "Ceijlon" | "Candi" = Kingdom of Kandy (Sri Lanka) |
| `graven` as verb meaning "to dig" (moats, canals) | "graven" ≠ "graves" |
| `vervallen` meaning "lapsed/expired" (contract, right) | "vervallen" ≠ "ruins" |
| `tempel` + context about India or Persia | Not Indonesian archaeology |
| `pura` in "Singapura" | Place name, not Balinese temple |

### When to say YES (genuine archaeological signal)

- Sentence describes finding/observing a physical structure, statue, or artifact
- The object is described as old, buried, or pre-dating Dutch presence
- Context mentions Java, Bali, Sumatra, or other Indonesian islands explicitly
- Depth is mentioned in context of excavation or burial (not construction/military)
- Dutch colonial officer is reporting a local landmark or discovery

### Examples

**YES (genuine):**
> "Op 3 voet diepte werden eenige steenen gevonden welke tot een oud tempel schijnen te behooren nabij de dessa Trowulan."
→ YES. MONUMENT + DEPTH. Physical stones found at 3 feet depth, described as belonging to an old temple, near Java location (Trowulan).

**NO (false positive):**
> "den last Noten op 30 pagodes de picol gerekend ende 12 pagodes voor de Sappan"
→ NO. "pagode" = currency unit. Price list for nutmeg and sappanwood.

**NO (false positive):**
> "una arca, in qua supra-specificata vasa argentea vestes R. recondita sunt"
→ NO. Latin sentence. "arca" = chest/box containing silver vessels. Not Indonesian.

---

## Priority Sentence Types for Phase 2 NER Training

After annotation, sentences marked `YES` become positive training examples. We need to annotate:
- 100–200 `YES` examples (balanced across entity types)
- 100–200 `NO` examples (hard negatives — sentences with the right keywords but wrong context)

The full pool for annotation (if 65-sentence sample yields insufficient positives) is the 14,626 Java-filtered CSV.

---

## Who Should Annotate

| Option | Pro | Con |
|---|---|---|
| **Pak Amien (self)** | Domain expert, fast judgment, free | Time cost (~5 hrs for 500 sentences) |
| **Go Frendi (co-author)** | NLP expertise, understands task | Coordination overhead |
| **Fiverr Dutch linguist** | Linguistic expertise in colonial Dutch | No archaeological domain knowledge; cost ~$50–100 |
| **Hybrid** | Pak Amien: YES/NO; Fiverr: entity boundaries | Best quality, moderate cost |

**Recommendation:** Start with Pak Amien annotating this 65-sentence sample (1–2 hours). Use the precision estimate to decide if Fiverr annotation is worth the cost.

---

*Annotation guide v1.0 — E211 Phase 2 — 2026-04-23*
