# Line 05 — ARCHIVAL NLP (Colonial Records & Pipelines)

> **Question:** What did colonial administrators actually record, and can NLP extract archaeological
> signal from it at scale?

**Recommended model:** Sonnet for pipeline construction and corpus runs (mechanical, high volume);
Opus for schema design, evaluation design, and any claim. **Effort:** medium.

---

## Scope

Dutch colonial and VOC archives: Delpher, *Oudheidkundig Verslag* (OV), dagregisters, GLOBALISE.
Dutch/historical NER, OCR normalisation, entity extraction, and the software product built to do it.
Reviewer communities: **digital humanities / historical NLP** (JOAD, LREC/CLARIN-adjacent venues) and
**Scopus DH journals**.

**This is the PI's home turf.** Their core expertise is NLP — on this line Claude assists with
scaffolding and evaluation, and the PI leads on data and interpretation. It is the inverse of
[06_thesis](../06_thesis/).

**Out of scope:** Old Javanese / Sanskrit / indigenous-language corpora
(→ [04_language_text](../04_language_text/)); the PhD applications this work supports
(→ [07_career](../07_career/)).

---

## The product

**VOC-ArchNLP v1.0.0** — `tools/voc_archnlp/`. Built as an **HKI Hak Cipta (Program Komputer)**
deliverable, not just a research script. Four modules behind a unified CLI:

```
downloader → preprocessor → normalizer → extractor
```

DJKI registration paperwork: 4 documents in `docs/HKI/`.
Related: `tools/globalise_pipeline/`.

**This is the project's only shippable software artefact**, and it counts toward the KUM/career track
independently of any paper acceptance — see [07_career](../07_career/).

---

## Papers & datasets

| Item | Folder | Status |
|---|---|---|
| **D1** Colonial register | `papers/D1_colonial_register/` | Data paper, **52 entries**. ✅ **PUBLISHED on Zenodo 2026-08-11** — `10.5281/zenodo.21882007` (v1.0). JOAD no longer a dependency; the register lives as an open dataset. |
| **P21** ColonialMine | `papers/P21_colonialmine/` | Proposal (a *mudik* idea). Not started. |

---

## Experiments

| Experiment | What it is | Status |
|---|---|---|
| `E211_voc_dagregister_nlp` | **Phase 1 pipeline = VOC-ArchNLP.** 500 files downloaded. | 🛑 **awaiting PI approval to run (112 days as of 2026-08-13)** |
| `E207_globalise_voc_pilot` | GLOBALISE pilot | done — schema verified |
| `E200_dutch_ner_baseline` | Dutch NER baseline | done |
| `E206_archeobert_colonial_gap` | ArcheoBERT on the colonial gap | done |
| `E091_ov_nlp_mining` | *Oudheidkundig Verslag* mining — **breaks the DHARMA monoculture** | done |
| `E128_ov_depth_analysis` | OV depth evidence (also cited by [02_taphonomy](../02_taphonomy/)) | independent replication |
| `E141`–`E143` | Delpher extraction / fulltext / spatial | done |
| `E125_delpher_pilot` | Delpher pilot | SUPERSEDED |
| `E070`, `E093`, `E098` | colonial & Indonesian literature mining, lit database | done |
| `E197_colonial_depth_validation` | colonial depth validation | done |

**14 experiments** are assigned to this line (12 primary). Authoritative list:
`docs/EXPERIMENT_INDEX.md` §"By Line of Inquiry" — regenerate with
`python tools/scan_experiments.py`.

### Hard constraints on the data

- **GLOBALISE NER schema has 7 entity types and *none* of them are archaeological** (verified). Any
  archaeological extraction requires a custom layer — do not assume the schema covers it.
- **Delpher / KB.nl API:** available, but under **GDPR and data-sharing restrictions**. A **standard
  contractual clause (SCC) with the institution is needed** before bulk redistribution. Local
  analysis is fine; republishing extracted full text is not.

---

## Line rules

1. **E211 does not run without PI approval.** 500 files are downloaded and waiting; that decision is
   the PI's, and it has been pending since April.
2. **Never redistribute Delpher/KB full text.** Derived counts, entities, and coordinates are fine;
   source text is not, absent the SCC.
3. **Evaluate against a held-out annotated set before reporting any extraction number.** The lesson
   from [01_spatial](../01_spatial/) transfers exactly: a metric computed on the data you tuned on is
   not a result.
4. **Version the product properly.** VOC-ArchNLP is registered software — a breaking change to the
   CLI is a versioning event, not a refactor.
5. Absence of archaeological entities in a colonial record is **evidence about the recorder**, not
   about the ground. State which of the two you are claiming.
