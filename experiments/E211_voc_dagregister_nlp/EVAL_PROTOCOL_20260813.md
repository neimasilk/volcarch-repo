# E211 — Evaluation Protocol (PRE-REGISTERED before the authorised full run)

**Date:** 2026-08-13 · **Status:** DECLARED IN ADVANCE of the full 500-file run (PI authorised same
day, decision hour D2). This protocol is frozen BEFORE the run, per line 01's lesson (E217–E223): a
metric computed on data you tuned on is not a result.

**Pipeline:** VOC-ArchNLP v1.0.0 (`tools/voc_archnlp/`): downloader → preprocessor → normalizer →
extractor. Corpus: 500 GLOBALISE dagregister files (~845 MB, 146M words), already on disk.

---

## 1. Entity types (frozen)

| Type | Definition | Example surface forms |
|---|---|---|
| `ARCHAEOLOGICAL_FEATURE` | temple, statue, ruin, inscription, grave, relic, antique structure | *tempel, beeld, puin/ruïne, opschrift/inscriptie, graf, oudheden* |
| `LOCATION` | place mentioned in find context | toponyms, *op de berg X* |
| `DEPTH_MEASUREMENT` | numeric depth + unit | *3 voet onder de grond* |
| `DATE` | date of find/report | *den 12en maart 1675* |
| `PERSON` | finder, reporter, informant | official names, *een Javaan* |
| `ORGANISATION` | VOC office, local polity | *Compagnie, Bantam* |
| `MATERIAL` | stone, bronze, ceramic, bone, gold | *steen, koper, aardewerk* |

Unit conversion table (frozen): voet = 0.3048 m · el = 0.6858 m · palm = 0.10 m · duim = 0.0254 m.

## 2. Annotated held-out sets (frozen design)

Two sets, one annotation guide (`results/E211_voc_mentions/ANNOTATION_GUIDE_v1.md`, v1 exists —
refresh to v2 with the type definitions above):

- **Precision set:** 300 sentences sampled at random from the extractor's candidate output,
  stratified by source file (one sentence max per file) → estimates candidate precision per type.
- **Recall set:** 200 sentences sampled at random from the full corpus (all paragraphs, not
  keyword-filtered) → estimates false-negative rate (recall).
- **Agreement subset:** the first 100 annotated sentences double-annotated (PI + second annotator)
  → Cohen's κ per type; target κ ≥ 0.7. Below 0.6: tighten the guide, re-annotate the subset.
- Held-out rule: **no tuning on these 500 sentences.** Any pattern/rule change after seeing them
  invalidates the eval → re-sample 100 new sentences as a fresh held-out set.

## 3. Metrics (frozen)

Per type: precision, recall, F1. Overall: micro-F1 across types. Sentence-level: exact-span match,
token-level IOB match (report both). Depth: unit-conversion accuracy on all DEPTH_MEASUREMENT spans
(manual recompute check). Geocoding (Phase 3): exact / ±10 km / region-only accuracy, reported
separately, never pooled.

## 4. Selection rule for publication (frozen)

- Publish (in the dataset + paper) entity types with **F1 ≥ 0.70** on the held-out sets.
- Types with 0.40 ≤ F1 < 0.70: published with an explicit "reduced confidence" flag.
- Types with F1 < 0.40: **excluded from the published dataset; reported as limitations.**
- The full candidate CSV may be released separately as raw output, clearly labelled unvalidated.

## 5. Pass/kill criteria (frozen)

| Outcome | Action |
|---|---|
| Micro-F1 ≥ 0.70 | Proceed: geocoding phase, dataset deposit, paper |
| 0.40 ≤ micro-F1 < 0.70 | Proceed with NER fine-tuning (ArcheoBERTje, 500+ sentence training set per E211 README Phase 2); re-eval on fresh held-out set |
| Micro-F1 < 0.40 | Keyword extraction insufficient — documented as informative negative; pivot to fine-tuned NER as the paper's method |

Negative results are publishable (E211 README risk register): the paper is method + dataset, not
thesis confirmation.

## 6. Smoke test — 10 files (immediately before the full run)

- Files: the first 10 by stable filename order (`ls data/*/dagregister* | head`), fixed list
  recorded in the run log.
- Checks: (a) all four modules chain without error; (b) extractor output schema matches v1.0.0;
  (c) per-file wall time → full-run time estimate (500 × t); (d) at least one candidate mention in
  ≥3 of the 10 files (sanity — the corpus contains archaeological vocabulary).
- The smoke test is not the run and does not need approval; its log is the run's preamble.

## 7. Run log requirements

Record: git HEAD, python version, `pip freeze` (voc_archnlp deps), seed of any random sampling,
file list, start/end time, output checksums. Everything needed to reproduce the CSV.

---

*Protocol written 2026-08-13 (E211 authorised, D2). Annotation guide v2 + sampling script are the
next Claude-owned steps; the full run follows the smoke test.*
