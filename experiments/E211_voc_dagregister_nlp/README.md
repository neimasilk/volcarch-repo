# E211: VOC Dagregister NLP at Scale — Archaeological Mentions Extraction

**Date:** 2026-04-22 (Session 21)
**Status:** Phase 0 — corpus secured (500 files), scoping for extraction pipeline
**Paper target:** TBD — likely DHQ (Digital Humanities Quarterly, Diamond OA) or Journal of Cultural Analytics. Secondary target: Digital Scholarship in the Humanities (Oxford, Q1).
**PhD-pitch alignment:** **MAXIMUM** — directly in Verberne's "Digging in Documents" scope (Leiden LIACS), Vossen's GLOBALISE project (VU Amsterdam), Lamqaddam's OCR+LLM for historical catalogues (UvA), Cohen's structured prediction for historical NLP (Edinburgh). This is the diamond-hunt most aligned with the active PhD tracks.
**Risk profile:** **LOWEST** of the 5 diamond-hunts per Gemini 3 Pro review (2026-04-22): "The Dutch colonial administrators were meticulous, geographically bound, and physically present. If an NLP pipeline identifies 500 spatial mentions of 'oudheden' or 'verborgen' structures in 17th-century texts, you have generated a genuine, first-order historical dataset. It does not require taphonomic assumptions."

---

## Why E211 is now priority #1

Three cross-model reviews (DeepSeek, Gemini Pro, ChatGPT Go) converged 2026-04-22 on the conclusion that:
- E209 satellite ML = circular (trained on Hindu-Buddhist sites, cannot detect pre-Hindu)
- E210 InSAR = highest risk of "compelling-but-nonsensical" pattern (groundwater/tectonic noise)
- E212 genomic Ne(t) = equifinality (any dip has multiple explanations)
- E208 Phase 3 kakawin = tangential to primary PhD targets
- **E211 VOC NLP = lowest risk, highest PhD-pitch alignment, most epistemologically sound**

Additionally, ChatGPT Go's observation that "BPI/Dikti is credibility-first, novelty-second — tightly-argued incremental papers outcompete unproven engine" applies directly: a clean VOC archaeological-mentions spatial database is exactly the kind of incremental-methodological-contribution Dutch PIs can evaluate against their existing grant frameworks.

---

## Research question

Can systematic NLP processing of the Dutch East India Company administrative corpus (1602–1798, ~5M+ pages via GLOBALISE Dataverse) surface a spatially-indexed database of archaeological encounters (monuments, antiquities, hidden structures, excavation reports, ruins) that tests VOLCARCH's predicted burial-depth geography against a genuinely independent corpus not used in prior P1 / P0 work?

### Specifically

- **H1:** Mentions of archaeological material (Dutch keywords: *oudheden*, *verborgen*, *beeld*, *tempel*, *puing*, *graf*) cluster spatially, and after geocoding, their density correlates with VOLCARCH Channel 5 predictions (volcano-distance gradient; burial-depth model projected onto observable colonial-era depth ranges 1–4 m).
- **H2:** The colonial-era find density gradient extends spatially through the full Java interior, not only edge sites that already appear in E091 (OV 1925–1949) or E141 (Delpher 1854–1942). The VOC dagregister covers 150–300 years earlier (1600s–1700s) and reports finds in locations where the later OV/Delpher record is thin.
- **H3:** Comparing the three corpora (VOC 1602–1798, Delpher 1854–1942, OV 1925–1949) reveals a temporal evolution in what counts as "archaeological" for Dutch administrators — with implications for colonial knowledge-production studies (a Lamqaddam-aligned angle).

### If successful

Primary output: a spatial-temporal database of 1,000–10,000+ Dutch-colonial-era archaeological mentions in Java, published with the paper, citable, and reusable. Secondary: methodological contribution on applying modern NLP (XLM-R or ArcheoBERTje fine-tuned on colonial Dutch) to historical administrative text at scale.

---

## Current state — PHASE 1 EXECUTING (2026-04-23)

**Status:** Phase 1 pipeline running on 500-file corpus.

| Stage | Status | Output |
|---|---|---|
| Download | **COMPLETE** (2026-04-22) | 500 files, ~845 MB, 0 failures |
| Preprocess (Stage 1) | **COMPLETE** (2026-04-23) | 548,929 paragraphs, 145,971,146 words |
| Normalize (Stage 2) | **RUNNING** (2026-04-23) | Colonial Dutch → modern Dutch, 500 paras_ files |
| Extract (Stage 3) | **RUNNING** (2026-04-23) | `results/E211_voc_mentions/voc_archaeological_mentions.csv` |

**VOC-ArchNLP v1.0.0** (HKI Hak Cipta 2026) is the pipeline package. Registered at `tools/voc_archnlp/`.

- **Pipeline components:**
  - `tools/voc_archnlp/` — unified package (download + preprocess + normalize + **extract [NEW]**)
  - `tools/globalise_pipeline/download_globalise.py` — corpus downloader
  - `tools/globalise_pipeline/preprocess_voc.py` — HTR cleanup
  - `tools/globalise_pipeline/normalize_colonial_dutch.py` — spelling normalisation

**Extractor entity types:** MONUMENT, GRAVE, RUIN, ARTIFACT, INSCRIPTION, DEPTH
**Depth conversion:** voet (0.3048m), el (0.6858m), palm (0.10m), duim (0.0254m)

### Prior related experiments

- **E091** — 22,162 settlement-mentions from 16 OV volumes (1925–1949)
- **E141** — 1,768 articles from Delpher (1854–1942), 165 geocoded, 33 with depth
- **E197** — colonial depth records 1.0–4.0 m matching sedimentation calibration
- **E206** — ArcheoBERTje evaluation: 60% missing entity types on VOC (needs fine-tuning)
- **E207** — GLOBALISE pilot (50 files initial; now at 500)

E211 is the scale-up of this line of work.

---

## Pipeline design

### Phase 1: Corpus preparation (next 1–2 sessions)

1. **Expand corpus** to 1000–2000 files if needed (current 500 may be adequate pilot; re-run `download_globalise.py --n 1500`).
2. **Preprocess** — run existing `preprocess_voc.py` on all files → clean text.
3. **Normalise colonial Dutch orthography** — run `normalize_colonial_dutch.py` → modern Dutch equivalents.
4. **Tokenise + sentence-split** → Stanza or spaCy `nl_core_news_lg`.
5. **Build keyword index** for quick archaeological-relevance filtering: *oudheden, oudheidkundig, verborgen, beeld, tempel, puing, graf, antik, ruïne, stenen, inscriptie, penning, begraven, gedelfd, ontgraven* + Javanese loanwords *candi, prasasti, arca, yoni, lingga, stupa*.

### Phase 2: NER + entity linking (next 2–3 weeks)

1. **Fine-tune ArcheoBERTje** or train from scratch a custom NER model for 7 entity types:
   - ARCHAEOLOGICAL_FEATURE (temple, statue, ruin, inscription, grave, relic)
   - LOCATION (place name, mentioned in find context)
   - DEPTH_MEASUREMENT (numerical depth with unit — "3 voet onder de grond")
   - DATE (date of find report)
   - PERSON (finder, reporter, local informant)
   - ORGANISATION (VOC office, local polity)
   - MATERIAL (stone, bronze, ceramic, bone)
2. **Training data:** manually annotate 500 sentences from VOC corpus (~1 week of focused work, can be split with co-author candidate or Fiverr linguistic annotator). Use PASCAL format.
3. **Evaluate** held-out accuracy.
4. **Apply** to full 1000+ file corpus → structured entity tables.

### Phase 3: Geocoding + spatial database (next 3–4 weeks)

1. **Gazetteer** construction: merge existing place names from (a) Java modern toponymy (OSM), (b) `data/processed/east_java_sites_*.geojson`, (c) colonial-era place name variants from OV / Delpher experiments, (d) direct extraction from VOC corpus.
2. **Entity linking:** resolve LOCATION mentions against gazetteer; where multiple candidates, apply contextual disambiguation (nearby rivers, distance markers, polity names).
3. **Output:** `data/processed/voc_archaeological_mentions.geojson` — one feature per archaeological-mention-event with attributes: mention_text, date_normalised, location_resolved, location_confidence, entity_types, source_file, source_page_or_section.

### Phase 4: Analysis + paper draft (next 4–8 weeks)

1. **Spatial analysis:** density by distance-to-volcano, comparison with E091 and E141 distributions, temporal trend 1600s–1700s.
2. **Validation:** cross-check ≥20 mentions against known archaeological record + verify reasonable coordinates.
3. **Paper drafting:** methods paper on historical-Dutch NLP pipeline + methodological contribution of cross-corpus comparison, emphasising the dataset as primary output.
4. **External review:** Pak Amien + Fiverr/academic reviewer + ideally a Verberne/Vossen informal read (PhD-track aligned).

---

## Risk register

| Risk | Mitigation |
|---|---|
| OCR quality in GLOBALISE may be poor (17th-c. Dutch handwriting transcription) | `normalize_colonial_dutch.py` handles partial fixes; evaluate against sample before scaling |
| Colonial-Dutch NER is a thin-data domain | Fine-tune ArcheoBERTje (Dutch archaeological text model, 2024) as starting point; budget for annotation |
| Geocoding ambiguity of colonial toponyms | Multi-source gazetteer + confidence scoring; flag low-confidence mentions |
| Dataset may contradict VOLCARCH predictions | **Welcome** — that's exactly what a genuine independent test should allow. Frame accordingly: "we use VOC NLP to test spatial predictions of our earlier work" — null or negative results are publishable. |
| Scope creep — pipeline is long | Commit to "method + dataset" paper first; further interpretive analysis is Phase 5+ or follow-up paper |

---

## Budget

- **Compute:** 0 (local, RTX 4080 for BERT fine-tuning)
- **Data:** 0 (GLOBALISE is CC0)
- **Storage:** ~5–10 GB for full 6893-file corpus (comfortable)
- **Annotation:** Either 1 week of Pak Amien time OR ~$50–150 Fiverr for 500-sentence annotation in PASCAL format
- **External review:** $50–200 Fiverr methodology reviewer (optional but recommended per ME#16 non-negotiable co-author / validation critique)

Total within existing operating budget.

---

## Phase 1 Output Files (2026-04-23)

| File | Description | Rows |
|---|---|---|
| `results/E211_voc_mentions/voc_archaeological_mentions.csv` | Full keyword extraction | 33,930 |
| `results/E211_voc_mentions/voc_mentions_java_filtered.csv` | Java/Indonesia geographic filter | 14,626 |
| `results/E211_voc_mentions/voc_mentions_high_precision.csv` | MONUMENT+INSCRIPTION+Java | 871 |
| `results/E211_voc_mentions/voc_mentions_normalized.csv` | Extraction on normalized text | pending |
| `results/E211_voc_mentions/annotation_sample_v1.csv` | 65-sentence annotation sample | 65 |
| `results/E211_voc_mentions/ANNOTATION_GUIDE_v1.md` | Annotation instructions | — |
| `experiments/E211_voc_dagregister_nlp/FINDINGS_v1_20260423.md` | Full Phase 1 analysis | — |

**Key Phase 1 finding:** Estimated precision <15%. `oudheden` = 0 occurrences. Primary false positives: `pagode` (currency), `arca` (Latin for chest), `opschrift` (document label). Phase 2 requires NER fine-tuning.

---

## Deliverables (Revised)

- **D1 (Phase 2 end):** Annotated dataset (200+ sentences) + fine-tuned NER model
- **D2 (Phase 3 end):** `voc_archaeological_mentions_filtered.geojson` — geocoded, precision-filtered
- **D3 (Phase 3 end):** Cross-corpus comparison (VOC 1600s vs Delpher 1850s vs OV 1920s)
- **D4 (Phase 4):** Paper targeting DHQ or Journal of Cultural Analytics
- **D5 (ongoing):** Zenodo dataset deposit (pilot 871 high-precision + annotation) for citation

---

## Success conditions for PhD pitch (revised)

Email to Verberne/Vossen/Lamqaddam by ~2026-06-01:
- "I've built a systematic NLP pipeline (VOC-ArchNLP v1.0, HKI registered) on 500 GLOBALISE files (146M words). Phase 1 yields 33,930 candidate sentences; after geographic filtering, 871 high-precision candidates. Phase 2 annotation has started. I'd welcome your perspective on the NER architecture and annotation protocol."
- Supported by Zenodo deposit of `annotation_sample_v1.csv` + pilot results.
- The HKI registration is the institutional anchor that makes this a citable, formalized deliverable.

---

## Relation to other experiments

- Supersedes E207 as the "production" VOC NLP line.
- Complements (not replaces) E091 (OV 1925-1949) and E141 (Delpher 1854-1942). Full temporal triad: VOC 1600s-1700s → Delpher 1800s-1940s → OV 1920s-1940s.
- Feeds masterpiece incubation as a validated-method/independent-dataset resource, without being the masterpiece itself.
- Does NOT conflict with E209 satellite (can run in parallel).

---

*E211 scoped 2026-04-22 per Pak Amien's decision + 3-AI review convergence. Phase 1 execution begins next session.*
