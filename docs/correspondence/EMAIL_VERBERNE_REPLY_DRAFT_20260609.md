# Reply to Verberne — Delft colleague's two questions (DRAFT v2)

**Date drafted:** 2026-06-09
**In reply to:** Verberne email 2026-06-08 22:02 (after consulting Delft colleague)
**Status:** DRAFT v4 — SEND-READY (2026-06-22). v3 humanised + tightened (~410 words, em-dashes stripped, bullet/structure removed, GLOBALISE-NER point folded into Q1). v4 closes the last open item: GLOBALISE's exact NER schema verified (15 labels / 7 types — persons, locations, organisations, polities, commodities, ships, documents + dates; NO archaeological types), so the Q1 entity list is now accurate and the "what they don't cover" claim is backed. Reads as plain academic prose to reduce AI-detection risk. **No open items.**

---

## The two questions being answered
1. Is the colonial Dutch archive dataset already digitised and pre-processed, ready for downstream NER?
2. The last work package (physical validation by archaeologists) can get very complicated — what is the approach?

## Strategic read (why this draft is shaped the way it is)
- Both questions target the **archaeology-dependent** parts of the project — the parts an NLP professor cannot supervise/fund/control. Combined with her opening flexibility concern, the likely subtext is: *is there a supervisable NLP-core thesis here, and are you willing to let the archaeology scale down?*
- Therefore the reply does NOT defend RQ4 harder. It makes **RQ1–RQ3 the self-contained thesis** and reframes **RQ4 as a falsifiable, non-blocking validation layer**. This answers the flexibility worry directly and is the honest scoping anyway.
- It volunteers, rather than hides, the inconvenient pilot finding (early VOC = archaeology-thin; rich archaeological signal is in later OV/Delpher). Hiding it and being caught by a GLOBALISE-literate Delft colleague would cost trust.
- It leads RQ4 with the **non-significant** pilot (Wilcoxon p=0.131), NOT the brief's "99.94% / r=0.951". Given P7's content rejection (volcano-distance artifact) + E214 palynology counter-evidence, the model is calibrated-but-contested; do not oversell it.

## Decision status (updated 2026-06-09)
- **Q2 reframed (DONE):** validation-by-expert recast as institutional fit + two-way value (Leiden has NLP+computational-archaeology; PI brings Indonesian field pathway). Two-tier (near = Leiden in-house, guaranteed; far = UGM/BRIN, prospective enrichment). AI = efficiency multiplier, not replacement. Rejected the "AI so we don't need archaeologists" framing (circularity + contradicts Verberne's own ArcheoBERTje model). See `VERBERNE_REPLY_ANALYSIS_20260609.md` §Q2.
- **Naming (DECIDED):** UGM + BRIN named (shows command of Indonesian terrain); Leiden archaeology referenced via the ArcheoBERTje connection, NOT by naming Lambers/Wansleeben (avoids appearing to arrange her household).
- **Flexibility (EMBODIED):** answered through the reframe, not by ceding the vision. Dial further toward NLP if you wish.
- **Falsifiability sentence (KEPT):** the "independent channel came back against my strongest claim and I revised" line stays — cleanest proof of the flexibility she asked about. Cut only if you'd rather not surface that the thesis is under revision.

### Honesty guardrails (binding — do not violate when sending)
1. Indonesian access is stated as **prospective** ("I do not yet have a formal arrangement"). Do NOT upgrade to a claimed partnership — Verberne knows Vossen and can check.
2. Collaboration is **enrichment, not critical path**. The thesis must complete on near-tier (existing data + light in-house audit) alone.
3. Lead RQ4 with the **non-significant** pilot (p=0.131), not the brief's 99.94%/r=0.951 (P7-contested).

### STILL OPEN — none (closed 2026-06-22)
- **Q1 GLOBALISE-NER repositioning — CLOSED.** GLOBALISE's published fine-grained NER schema verified: **15 labels across 7 entity types — persons, locations, organisations, polities, commodities, ships, documents — plus dates** (Fine-grained NER for the East-India Company archives; corroborated by the GLOBALISE contribute-data page: "persons, places, commodities and ships … events"). **None of their types are archaeological** — no depth, material/find-object, find-event, or soil/burial context. So the proposal's novel entities (DEPTH, MATERIAL, FIND_EVENT, SOIL_CONTEXT) sit precisely outside their schema; the Q1 sentence now lists their entities accurately and the contribution = archaeology entities + settlement geography is backed, not asserted.

## Context for you (NOT for the email)
- The gate is still **BPI Dosen funding** (competitive, not secured). A good reply converts interest, it does not secure a position. Keep expectations calibrated.
- Do not import the brief's headline numbers (99.94% invisible, r=0.951) into this reply. They are exactly what P7 reviewers contested. The reply stays scoped to data-readiness + validation method.

---

## DRAFT EMAIL

Subject: Re: PhD Inquiry — NLP for Colonial Dutch Archives and Geospatial Settlement Reconstruction

Dear Suzan,

Thank you for this, and please thank your colleague in Delft too. Both questions go straight to the parts of the project I worry about most, so let me give you proper answers rather than reassurances.

On the dataset. What I'm building on is GLOBALISE, the Huygens Institute's machine transcription of the handwritten VOC archive, about five million pages, all released under CC0. It is certainly digitised and readable, and I've already cleaned a 500-volume slice of it, roughly 146 million words. What it isn't is ready for NER as it stands. So instead of just asserting that, I ran a quick keyword baseline over the slice to see how bad things were, and the answer was: quite bad. Well under 15% precision on inspection, drowning in false positives. The false positives are the telling part. In this corpus "pagode" almost always means a coin, not a temple; "arca" turns out to be the Latin word for a chest; and Dutch, Portuguese, Latin and Malay sit on the same page as often as not. It needs precisely the work I'm proposing — language identification, a domain-adapted NER model, normalisation — before it gives up anything usable. I've put the pipeline, the cleaning statistics and that baseline output online, in case you or your colleague would sooner see it than take my word for it: https://huggingface.co/datasets/neimasilk/voc-archnlp-mentions

One thing I should be honest about, since you know the team well: GLOBALISE already recognises the entities its own historians need — persons, places, organisations, polities, ships, commodities, documents and dates — and I've no wish to redo any of that. What I would add sits where their schema stops — archaeological mentions like depth, material and finds, and the tying of settlement names to coordinates. And for that archaeological vocabulary the richer material is really in the later printed reports, the Oudheidkundig Verslag and the Delpher newspapers; the VOC registers themselves carry much less of it. So I would be building on GLOBALISE, not alongside it.

On the validation, which is the fair worry, and honestly part of why I'm writing to Leiden in particular. The first thing to say is that it needs no new excavation at all. It compares depth figures already written into the colonial texts against an independent burial model — one calibrated from geological sedimentation rates, not from the records it is then tested on — and my small pilot of 33 such records held up against it (Wilcoxon p = 0.131). I try to treat that as a real test rather than a box to tick; in fact another, independent strand of the project recently went against one of my central claims, and I revised the claim.

Where it does need genuine archaeological judgement, I'd approach it at two levels. At Leiden, the computational-archaeology people you built ArcheoBERTje with could check a sample of the extractions, which is enough to keep the thesis self-contained. In Indonesia, the natural partners for ground truth are places like UGM (Universitas Gadjah Mada) in Yogyakarta, whose archaeologists work right in the Central Java volcanic zone, and BRIN. I'll be straight that I have nothing formal with them yet; what I can offer is a realistic way in as an Indonesian academic, which a project run only from Europe wouldn't have. I'd keep that firmly as future fieldwork the thesis doesn't depend on. The job of the NLP, throughout, is to point a scarce archaeologist at where to look, not to take their place.

So, put plainly: this can be finished as a computational thesis, checked against data that already exists and a light audit at Leiden; the Indonesian side is an opportunity I'm well placed to build, and one the thesis can manage without; and any real digging stays as future work.

I'd be very glad to talk it through whenever suits you and your colleague, or to send a one-page revision of that last work package if that would help.

Best regards,
Mukhlis Amien

---

## Source grounding (all from repo, not invented)
- GLOBALISE subset 500 files / 548,929 paragraphs / 145,971,146 words; naive pilot 33,930 candidates, <15% precision; pagode/arca/Candi false positives → `experiments/E211_voc_dagregister_nlp/FINDINGS_v1_20260423.md`
- E197 depth validation: 33 records, Wilcoxon p=0.131 → proposal §RQ4
- E091: 16 OV volumes, 22,162 mentions, recall vs 52 manual entries → proposal §Preliminary Results
- E141: 1,768 Delpher records → proposal §Preliminary Results
- GLOBALISE = HTR (handwriting), not OCR; Delpher = OCR (print) — correctness point for a GLOBALISE-literate Delft colleague
- Flexibility/revision example = E214 palynology counter-evidence → `project_palynology_counterevidence` memory + JOURNAL 2026-06-08
- Model is calibrated-but-contested post-P7 (volcano-distance artifact) → do not oversell

## Verification — "already done" claims checked against repo (2026-06-09)
Triggered by PI integrity check. Each "already done" statement in the email traced to a concrete artifact:

| Email claim | Status | Evidence |
|---|---|---|
| cleaned 500-volume subset, ~146M words | ✅ solid | `data/processed/globalise_voc/preprocessing_stats.json` → `total_files:500`, `total_words:145,971,146`; 500 `clean_*.txt` on disk (859 MB); per-file cleaning stages recorded (metadata-strip → rejoin → filter) |
| keyword baseline run, ~34k candidate mentions | ✅ solid | `results/E211_voc_mentions/voc_archaeological_mentions.csv` = 33,930 data rows; java_filtered 14,626; high_precision 871 |
| pagode=coin / arca=chest false positives | ✅ solid | pagode 2,239×, arca 613× in output CSV; analysis in `experiments/E211_voc_dagregister_nlp/FINDINGS_v1_20260423.md` |
| precision < 15% | ⚠️ ESTIMATE (not gold-measured) | FINDINGS states "estimated precision <15%" from FP analysis; gold annotation = E211 Phase 2, PENDING. Email wording softened to "on inspection … well under 15%" so it does not imply a measured metric. |
| pilot 33 records, Wilcoxon p=0.131 | ✅ exact | `experiments/E197_colonial_depth_validation/results/e197_results.json` → `n_combined:33`, `wilcoxon_p:0.1309`, `model_consistent:true` |

Pipeline that produced the cleaning: `tools/voc_archnlp/` (E211 Phase 1, Session 22, 2026-04-23).

## Public validation reference (2026-06-09)
The data claim was NOT validatable on GitHub (E211 results/code untracked; committed `preprocessing_stats.json` showed the stale 50-file/6.26M-word state; four result files exceed GitHub's 100 MB limit). Instead, curated artifacts were published as a **HuggingFace dataset**: **https://huggingface.co/datasets/neimasilk/voc-archnlp-mentions** (public). Contents: `preprocessing_stats.json` (the 500-file/146M-word evidence), the pipeline code (`voc_archnlp/`), `FINDINGS_v1`, the 871-row high-precision sample, and the full 33,930-row extraction + 14,626-row Java-filtered set (LFS). Card states the <15% precision is an **estimate**, not gold-measured. This is the email's footnote [1]. The main `volcarch-repo` is deliberately NOT linked (avoids exposing the four-track PhD strategy + internal critiques).
