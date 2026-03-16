# WORKSTATE — Session Continuity Contract

**READ THIS FIRST. Continue in-progress items before starting new work.**
**Last updated:** 2026-03-16

---

## IN PROGRESS

- **E090 v5 full run** — Script written. v5 corpus (200 entries, 12 traditions). BERTopic reactivated, 8 concept groups.
  - Files: `experiments/E090_transformer_textual_nlp/e090_v5_full.py`
  - Next: USER RUNS ON GPU (RTX 4080). ~2 min. Produces BERTopic topics, 8-group convergence, v2→v5 delta.
  - After: Update E090 README with v5 results, feed into P16 assessment.
- **E094 DHARMA semantic search** — Script written. First SBERT on Old Javanese epigraphy.
  - Files: `experiments/E094_dharma_semantic_search/dharma_semantic_search.py`
  - Next: USER RUNS ON GPU (~2 min). UMAP clusters, 7 semantic queries, temporal drift.
  - After: Update README with results. P5/P8 revision ammo.
- **E096 DHARMA diachronic BERTopic** — Script written. First BERTopic on any epigraphy.
  - Files: `experiments/E096_dharma_diachronic_bertopic/dharma_diachronic_bertopic.py`
  - Next: USER RUNS ON GPU (~1 min). Pre/post-929 CE topic comparison.
  - After: Update README with results. P5/P8/P16.
- **E076 v2 satellite analysis** — Script written, needs user to run (network-dependent)
  - Files: `experiments/E076_satellite_ndvi/02_multi_tile_analysis.py`
  - Next: Run script (requires internet for Planetary Computer STAC API), ~30 min
- **Colonial data verification** — Open 10 E070 entries on Delpher.nl, verify manually
  - Files: `data/raw/colonial_sources/colonial_site_register_v1.0.csv`
  - Next: Pick 10 entries, find original OV page, confirm extraction accuracy
- **P11 finalization** — v0.3 drafted (18pp), needs user manual review then submission
  - Files: `papers/P11_volcanic_informedness/draft_v0.3.tex`, `SUBMISSION_PREP.md`
  - Next: User manual review → Chicago 17th citations → submit to indonesia-journal@cornell.edu

## COMPLETED THIS SESSION (Senter v3 — Computational Deepening)

- **E097 anomaly detection** — SUCCESS. Isolation Forest on 378 sites, 589K grid cells. **65% overlap** with E080 targets. Kelud focus. TRI top feature (0.294). 195K site-like cells with >1m burial.
- **E092 volcanic comparanda** — SUCCESS. 28 worldwide buried sites, methodology blueprint by depth range. GPR/ERT/coring cost estimates.
- **E093 Indonesian lit mining** — SUCCESS. 65 publications. GPR leads: Trowulan, Liyangan, Sambisari. Validation opportunities documented.
- **E098 systematic lit database** — SUCCESS. 69 sedimentation rates, 29 buried sites, 20 GPR surveys in tropical/volcanic soils. Meta-analysis.
- **E090 v5 script** — Written. BERTopic reactivated (200 entries), 3 new concept groups, delta comparison.
- **E094 script** — Written. SBERT + UMAP + HDBSCAN on DHARMA inscriptions.
- **E096 script** — Written. Diachronic BERTopic, pre/post-929 CE comparison.
- **EXPERIMENT_INDEX** — Updated to 98 experiments. E090/EXP4 revisit addressed.
- **E089 v5 corpus** — 200 entries reached (done in previous expansion). BERTopic threshold met.
- **E080 infographic** — Fieldwork convergence map (E080 targets x E097 anomalies). `experiments/E080_fieldwork_targets/results/fieldwork_infographic.html` (575KB)
- **Dokumen Jembatan v0.2** — Integrated E092 comparanda, E097 65% convergence, E098 GPR limitation, ERT recommendation.
- **L3 EXECUTION** — Sprint 11, GPU tasks listed.
- **Handoff document** — `docs/plans/senter_v3_handoff.md` with prompts for tomorrow.

## COMPLETED PREVIOUS SESSIONS

- **Consolidation Sprint (Sessions 2-5)** — P7 preprint submitted. Dokumen Jembatan v0.1. Dissemination roadmap v1.0. E089 v4 (162). P11 v0.3 (18pp). Code review done.
- **Senter v2** — E091 OV NLP (22K mentions). E089 v3 (106 entries). E076 v2 script. E090 corpus update.

## DISSEMINATION ROADMAP — Phase 1

- Dissemination Roadmap v1.0 — `docs/VOLCARCH_Dissemination_Roadmap_v1.0.md`
- Phase 1 (Foundation): Dokumen Jembatan, infographic, preprint strategy
- Phase 2 (Targeting): WAIT for 1 acceptance first
- **Gate:** minimal 1 paper accepted (P1 or P2, estimate 2-4 months)

## BLOCKED

- **D1+D2 JOAD submission** — Blocked on APC waiver decision (£374 each)
- **P7 preprint DOI** — Submitted to Authorea/ESSOAr 2026-03-16, screening up to 4 business days
- **Dissemination Phase 2 emails** — Blocked on paper acceptance

## SESSION PROMPT

STATUS: **98 experiments**, 6 papers submitted, P11 v0.3 drafted, D1+D2 drafted. Code review DONE. Dissemination roadmap v1.0.
PHASE: CONTRACTION + VALIDATION + DISSEMINATION PREP (post-structural critique).
PRIORITIES:
1. USER GPU: E090 v5 (`e090_v5_full.py`, v5 corpus 200 entries) — ~2 min
2. USER GPU: E094 DHARMA semantic search — ~2 min
3. USER GPU: E096 DHARMA diachronic BERTopic — ~1 min
4. USER NETWORK: E076 satellite (`02_multi_tile_analysis.py`) — ~30 min
5. USER MANUAL: Cek P8 arXiv status (submit/7351261, on hold)
6. USER MANUAL: Upload Dokumen Jembatan ke NotebookLM → generate Audio Overview
7. USER REVIEW: P11 manual review → Word cleanup → submit to Cornell
8. CLAUDE: After GPU results → update READMEs, assess P16 viability
9. CLAUDE: Dokumen Jembatan v0.2 (integrate E092 comparanda + E097 anomaly map)

## DO NOT WORK ON

- New paper submissions (6 already under review — wait)
- Framework expansion (CONTRACTION phase)
- Phase 2 dissemination emails (wait for 1 acceptance first)
- E095 cross-lingual XLM-R (deferred — needs E090 v5 results first)
