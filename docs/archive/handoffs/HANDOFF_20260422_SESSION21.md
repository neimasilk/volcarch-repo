# HANDOFF — Session 21 (2026-04-22)

**Duration:** Extended pipeline-driven session (afternoon → evening)
**Mode:** Autonomous per Pak Amien mandate — ME#16 produced, multi-AI review panel established, strategic pivots locked, three concrete next-step deliverables scaffolded.
**Trigger:** Pak Amien reframe: P1 = masterpiece (the paper that keeps getting rejected IS the masterpiece candidate), rejections were a test, kegelisahan manifesto belum terjawab landmark, mandate untuk sisi yang belum tergali + system/research-designer critique + pipeline execution.

---

## 1. Strategic decisions locked this session

1. **Mata Elang #16 produced** — `docs/research_notes/MATA_ELANG_16_2026_04_22.md`. Diagnosis: **discovery deficit** (200+ experiments = re-read of same data; zero new physical-grade observations) is root cause of all rejections. Pivot proposed: discovery-first diamond-hunt program.
2. **ME#16 pivot rebalanced** (after Pak Amien PhD-pitching constraint clarification): NOT "stop all polish and pause papers"; instead "maintenance-mode papers continue, diamond-hunts run in parallel, co-author pursuit upgraded to blocker, cascade retired, E211 VOC NLP promoted to diamond-hunt #1."
3. **Three AI skeptical reviews completed** on ME#16:
   - DeepSeek via API ($0.003) — flagged E209 classifier circularity, co-author non-negotiable, diamond-hunt = confirmation bias in tech.
   - Gemini 3 Pro via Playwright MCP (personal account `neimasilk@gmail.com`) — flagged materials-assumption (bamboo/wood = no RS signature), diamond-hunt risk ranking (E210 highest risk, E211 lowest), "pausing P0/P1 = tactical suicide", "pivot from Discovery to Engine" reframe.
   - ChatGPT Go via Playwright MCP (wife's account `cprastiasih@gmail.com`) — flagged Indonesian/Dutch academic sociology specifics, "posterior stacking on correlated likelihoods", "manifesto = interpretive elasticity masquerading as theory", "optimization target drift", and the **meta-finding** that running more AI reviews is itself intellectual procrastination.
4. **Masterpiece Protocol formalised** — Pak Amien decision: P1 is the masterpiece, handled slow-cooked with a 4-AI consensus gate (Claude + DeepSeek + Gemini + ChatGPT must all return "masterpiece level / accept / minor revision"); any REJECT sends it back to incubation; other papers continue pipeline normally; no drafting until Phase 0 fallow (reading only) ends.
5. **Discovery-first pivot rebalanced to balanced execution:**
   - SUBMIT P0 v0.4 to Journal of Anthropological Archaeology (balance: don't pause submission queue = PhD pitch supporting material).
   - HOLD P1 = move to `papers/MASTERPIECE/` Phase 0 fallow.
   - EXECUTE E211 VOC NLP (lowest risk, highest PhD-pitch alignment).
   - STOP AI-only review loop (3 reviews is the saturation point per ChatGPT meta-finding).

---

## 2. Artifacts produced this session

### Strategic / meta
- `docs/research_notes/MATA_ELANG_16_2026_04_22.md` — full architect's critique (discovery-first pivot rationale, 5-hunt portfolio, drop list, testing framework for critique selection)
- `docs/research_notes/ME16_DEEPSEEK_REVIEW_20260422.md` — cross-model critical review #1
- `docs/research_notes/ME16_GEMINI_PRO_REVIEW_20260422.md` — cross-model critical review #2
- `docs/research_notes/ME16_CHATGPT_GO_REVIEW_20260422.md` — cross-model critical review #3 (with meta-finding on the review process itself)

### Folder / protocol establishment
- `papers/MASTERPIECE/` (new folder) + `README.md` — 4-AI consensus gate, Phase 0 fallow, 6-rule protocol
- `papers/P1_taphonomic_framework/ARCHIVE_STATUS.md` — P1 archived; pointer to MASTERPIECE
- `memory/project_masterpiece_protocol.md` — formalised protocol documented for future sessions
- `memory/user_google_accounts.md` — 3 Google accounts (EDU, Personal Pro for Gemini, wife's for ChatGPT Go), speed observations, how-to-use

### Diamond-hunt #1 (E209 Satellite ML Classifier)
- `experiments/E209_satellite_ml_classifier/README.md` — full scoping
- `experiments/E209_satellite_ml_classifier/scripts/01_prepare_training_data.py` — 121 sites built (6 hard-pos + 109 soft-pos + 5 hard-neg)
- `experiments/E209_satellite_ml_classifier/scripts/01b_add_random_negatives.py` — +200 random negatives (class balance fix)
- `experiments/E209_satellite_ml_classifier/scripts/02_extract_s2_features.py` — Sentinel-2 STAC pipeline, 190 sites with dry+wet features extracted
- `experiments/E209_satellite_ml_classifier/scripts/03_train_classifier.py` — **AUC 0.844 ± 0.060, leave-one-hard-positive-out 0.865**
- `experiments/E209_satellite_ml_classifier/scripts/04_predict_landscape.py` — scaffold (needs multi-scene mosaic fix for full basin coverage; composite returned 50km×1.15km strip instead of 44×44km full Malang)
- `experiments/E209_satellite_ml_classifier/FINDINGS_v1_20260422.md` — findings doc
- `experiments/E209_satellite_ml_classifier/external_reviews/critical_deepseek_findings_20260422.md` — DeepSeek review (flagged Hindu-Buddhist training circularity — the classifier detects temple-like sites, not pre-Hindu specifically)

### Diamond-hunt #2 (E210 InSAR) — scoping only
- `experiments/E210_insar_subsidence/README.md` — deferred per ChatGPT risk ranking

### Diamond-hunt promoted #1 (E211 VOC NLP)
- `experiments/E211_voc_dagregister_nlp/README.md` — full scoping with 4-phase pipeline, risk register, PhD-pitch alignment
- VOC corpus expanded from 50 → 500 files (845.6 MB, 0 failures) via parallel background shell

### P0 submission prep
- `papers/P0_invisible_civilization/SUBMISSION_CHECKLIST.md` — JAnthArch pre-flight audit, declarations, cover letter sketch, reviewer suggestions, 10-day timeline

### Memory / WORKSTATE / JOURNAL
- WORKSTATE mode updated to "DISCOVERY-FIRST" → then balanced per rebalance
- JOURNAL Session 21 entry
- MEMORY.md index refreshed (added Masterpiece Protocol and Google Accounts references)

---

## 3. Numeric outcomes

| Metric | Value |
|---|---|
| Budget spent total | ~$0.006 (DeepSeek API only; Gemini + ChatGPT = subscription, $0 marginal) |
| Budget remaining | ~$3.27 |
| Background shells completed (3) | E209 initial extraction (241 features), VOC 50→500 download, E209 random-negative extraction |
| E209 classifier AUC | 0.844 ± 0.060 (5-fold) |
| E209 leave-one-hard-positive-out | 0.865 (5/6 buried candi >0.85 when held out) |
| AI reviews produced | 3 (DeepSeek, Gemini 3 Pro, ChatGPT Go) |
| Papers paused → masterpiece | 1 (P1) |
| Papers ready to submit | 1 (P0 v0.4, 48pp compile clean) |

---

## 4. Known limits and risks flagged

- E209 classifier is (per DeepSeek + Gemini): a Hindu-Buddhist site detector, not a pre-Hindu detector. "Discovery" framing retired; reframe as "site-detection methodology paper" per Gemini's "Engine not Discovery" suggestion.
- Landscape prediction (task #30) blocked on multi-scene mosaic fix. First attempt returned 50×1.15 km strip because Sentinel-2 tile cut across 44×44 km basin. Documented.
- ChatGPT's meta-finding on intellectual procrastination — running further AI reviews is self-defeating. Commitment: no more AI skeptical reviews until actual external peer review returns from P0 submission.
- Co-author pursuit is flagged as non-negotiable by all 3 AI reviewers and should be actively advanced at PhD-track level (via Verberne/Lamqaddam positive responses already received).

---

## 5. What Pak Amien decides next session (options only, no action queued)

1. **P0 submission pathway** — review SUBMISSION_CHECKLIST.md, decide on (a) title softening, (b) CRediT split, (c) reviewer suggestions, (d) external-reviewer budget ($50–200 Fiverr), (e) submit direct or via external-reviewer gate first.
2. **E211 Phase 1 execution** — give Claude approval to run preprocess_voc.py on 500 files + scope annotation approach (Pak Amien vs Fiverr vs Gunawan).
3. **MASTERPIECE Phase 0 reading** — set reading schedule for Wolters 1999, Lombard, Bloembergen, Pollock, Ali (Pak Amien manual work).
4. **E209 landscape mosaic fix** — approve Claude to implement multi-scene mosaic (1 hour coding) for full basin probability maps.
5. **Lamqaddam reply** — user task; not in Claude's scope but flagged as time-sensitive.

---

## 6. State at end of session

- 6 papers under review unchanged (P2 JCAA, P7 Antiquity, P8 OL, P11 Archipel, P17 ArchCalc, plus P0 pending submission).
- Core stack unchanged.
- PhD tracks unchanged: Verberne proposal sent; Lamqaddam awaiting reply; Vossen drafted; Cohen CV sent.
- 208+ experiments registry unchanged (E209, E210 scoped added; E211 scoped).

---

## 7. For the next-session Claude (one-paragraph orientation)

Read this HANDOFF first. State at session end 2026-04-22: Pak Amien has formalised the Masterpiece Protocol with a 4-AI consensus gate (Claude + DeepSeek + Gemini 3 Pro + ChatGPT Go must unanimously approve before the masterpiece paper submits), moved P1 to `papers/MASTERPIECE/` Phase 0 fallow, approved P0 v0.4 for submission to JAnthArch with a detailed checklist in `papers/P0_invisible_civilization/SUBMISSION_CHECKLIST.md`, promoted E211 VOC NLP to diamond-hunt #1 with full scoping at `experiments/E211_voc_dagregister_nlp/README.md` and 500 GLOBALISE files already downloaded, and explicitly stopped running additional AI-only skeptical reviews (3 is saturation). E209 satellite ML classifier achieved AUC 0.844 leave-one-HP-out 0.865 but is now correctly framed as Hindu-Buddhist site detector (not pre-Hindu discovery tool) per cross-model review; landscape prediction blocked on multi-scene mosaic fix (task #30). Next session should begin with Pak Amien's decisions on P0 submission pathway options, or execute E211 Phase 1 preprocessing if given approval. Do not add more AI reviews, do not polish P0 further without specific request, do not attempt to draft the masterpiece (Phase 0 fallow = reading only).

---

*HANDOFF produced 2026-04-22 end-of-session-21. Next session consumes this first.*

