# HANDOFF — Session 18 (2026-04-20)

**Duration:** Extended pipeline-driven session (Sunday, full day)
**Mode:** User authorised max-effort + autonomous + pivot-eligible
**Trigger:** Pak Amien's dissatisfaction with P1 pre-submission state; request for masterpiece-quality review
**Outcome:** Path B adopted (split P1, build P0 masterpiece), SLR launched and completed all 10 subfields, E208 NLP pipeline executed

---

## 1. Strategic decisions locked this session

1. **NO JASREP submission Monday 2026-04-21.** Submission suspended pending Path B execution.
2. **Path B adopted:** Split P1 v2.0 into (a) P1-core (technical calibration, ~15pp) and (b) P0 "The Invisible Civilization" (synthesis masterpiece, ~25-30pp).
3. **P0 target:** *Journal of Anthropological Archaeology* (Elsevier Q1, subscription route = zero APC). Backup: Current Anthropology, Antiquity, Cambridge Archaeological Journal.
4. **Full SLR launched** — subfield-driven, symmetric-bias-mitigated, pivot-eligible. Completed all 10 subfields in one extended session.
5. **PhD-waiting-period priority rule installed:** prioritise autonomous NLP-heavy computational work during Verberne response wait; skip manual-human-bandwidth tasks.
6. **Landmark definition operationalised** in ME#15: 7 explicit criteria for P0 quality bar.

---

## 2. Artifacts produced this session

### Strategic / meta
- `docs/research_notes/MATA_ELANG_15_2026_04_20.md` — 10-section architect's critique + critique-selection protocol + three-path analysis with Path B recommendation
- `docs/LITERATURE_SLR_PROTOCOL.md` — full SLR protocol with 10-subfield scope + tagging schema + bias-mitigation rules
- `docs/LITERATURE_SLR_PROGRESS.md` — living progress tracker (session 18a/b/c)
- `tools/critical_reviewer_prompt.md` — cross-model review prompt + special-focus addenda for P1-core and P0

### P0 masterpiece (new paper)
- `papers/P0_invisible_civilization/SKELETON_v0.1.md` — 10-section scaffold (8-10K words target)
- `papers/P0_invisible_civilization/draft_v0.1.tex` — §1 Introduction + §2 Demographic Puzzle + §3.1 Sedimentation + §3.2 Philippines (partial, ~3,500 words, compiles to 12pp)
- `papers/P0_invisible_civilization/references.bib` — bibliography derived from P1's + new entries (jia_etal_2024, reid1988, amien_p1core, amien_e201, bellwood_dizon2013, erasmus2005 placeholder)
- `papers/P0_invisible_civilization/revision_ammo/JATIM_BEADS_DATONG.md` — 2,400+ word cathedral-evidence documentation

### P1-core (surgical cut from v2.0)
- `papers/P1_taphonomic_framework/submission_jasrep_v3.0.tex` — surgical cut removing §2.2 last 2 paragraphs (demographic → P0), §2.5 West Java (→ P0), §5.5 cascade (→ P0), §5.4 Kutai (merged), §5.2 near-volcano (compressed), plus added compaction/erosion/spatial autocorrelation/monument-vs-settlement caveats in §5.6. **Compiles clean, 21pp with refs.**
- `papers/P1_taphonomic_framework/submission_jasrep_v3.0.pdf` — compiled
- `papers/P1_taphonomic_framework/references.bib` — +Reid 1988 + placeholder for amien_synthesis_forthcoming (P0 cite)
- `papers/P1_taphonomic_framework/external_reviews/critical_review_claude_persona_20260420.md` — simulated skeptical reviewer pass, 9 actionable items

### SLR bibliography (12 files)
```
docs/bibliography/
├── 01_glass_bead_archaeometry/
│   ├── jia_etal_2024_datong_jatim_beads.md          ← CATHEDRAL anchor
│   ├── jia_etal_2025_northern_wei_blue_vessels.md   ← methodological nuance
│   ├── wang_etal_2023_taiwan_maritime_glass.md      ← Taiwan context
│   └── wang_etal_2021_guishan_iron_age_taiwan.md    ← multi-workshop evidence
├── 02_trans_eurasian_trade/
│   └── hoppal_bellina_dussubieux_2023_se_asia_mediterranean.md
├── 03_chinese_historical_texts/
│   └── wolters_1967_early_indonesian_commerce.md    ← CATHEDRAL anchor (Ye-tiao 132 CE)
├── 04_indonesian_archaeometry/
│   └── _SUBFIELD_SUMMARY_indonesian_archaeometry_session18.md (GAP finding)
├── 05_paleogenomics/
│   └── _SUBFIELD_SUMMARY_paleogenomics_session18.md (risk zone CONFIRMED)
├── 06_volcanic_taphonomy_global/
│   └── _SUBFIELD_SUMMARY_volcanic_taphonomy_global_session18.md (risk zone CONFIRMED)
├── 07_austronesian_metallurgy/
│   └── _SUBFIELD_SUMMARY_metallurgy_session18.md (Pejeng drum lead isotopes)
├── 08_korean_japanese_tombs/
│   └── _SUBFIELD_SUMMARY_korea_japan_session18.md   ← CATHEDRAL (10+ Silla tomb Jatim beads)
├── 09_berenike_red_sea/
│   └── _SUBFIELD_SUMMARY_berenike_session18.md (Sidebotham confirmation)
└── 10_indianization_historiography/
    └── ali_2011_inscriptions_sanskrit_cosmopolis.md (positioning vs Pollock)
```

### E208 Kakawin NLP Pilot
- `experiments/E208_kakawin_nlp_pilot/README.md` — full documentation + honest interpretation
- `experiments/E208_kakawin_nlp_pilot/scripts/phase1_domain_classification.py`
- `experiments/E208_kakawin_nlp_pilot/scripts/phase2a_sanskrit_heuristic.py`
- `experiments/E208_kakawin_nlp_pilot/results/` — `domain_distribution.csv`, `lexname_distribution.csv`, `domain_samples.json`, `summary.md`, `phase2a_domain_by_etymology.csv`, `phase2a_summary.md`
- `data/raw/old_javanese_wordnet/wn-kaw.tab` — 5,019 OJW synsets downloaded from GitHub

### Logs / memory
- `docs/JOURNAL.md` — 3 new session entries (SLR launch, session 18b risk zones, session 18c Fase B complete + E208)
- `docs/WORKSTATE.md` — urgent decision flag added; mode marker updated
- Memory `project_phd_leiden.md` — updated with priority rule during waiting period

---

## 3. Key findings summary

### SLR: zero material counter-evidence across 10 subfields
3 risk-zone subfields (paleogenomics, global volcanic taphonomy, Indianization critical historiography) all confirmed or nuanced VOLCARCH, did not contradict it.

### 3 cathedral-grade external anchors added
- **Jia et al. 2024** (Nature Heritage Science): Datong Jatim beads chemically attributed to Java, 398-494 CE
- **Wolters 1967** (canonical): Ye-tiao 132 CE Javanese embassy to Han China attestation
- **Lankton & Bernbaum** + 10+ Jatim beads in Silla royal tombs (late 4th - mid 6th c. CE Gyeongju)

### Channel 6 corpus expanded from 1 to 4+ terminal sites
Datong (China) + Gyeongju (Korea) + Berenike (Egypt) + Palau (Micronesia) + Kofun Japan (tbd) + Europe terminus — ~15+ peer-reviewed Jatim bead finds across 8,000+ km network, all in the period VOLCARCH claims invisible civilisation existed in Java.

### Channel 6 extended to second archaeometric sub-channel (bronze)
Calo's lead-isotope analysis of Pejeng drums confirms local Bali/Java bronze production 1st-2nd c. CE, adding bronze as a sibling to glass beads in the archaeometric channel.

### E208 Phase 2a nuances E058
E058's extreme native/Sanskrit figures (91% Agriculture native, 14% Ritual native from 189 curated kakawin terms) do NOT reproduce at 5,019-synset dictionary scale. Directional pattern holds (material-culture > prestige for native %) but extremes dampened. Three explanations: heuristic undercounts Sanskrit, E058 literary-register bias, token-vs-type scale difference. **Implication:** E058 should be reframed as kakawin-frequency-weighted, not language-wide. P0 Channel 3 reports both scales.

### Two Facebook post claims independently verified
Jia 2024 Jatim beads (via peer-reviewed paper) + Wolters 1967 Ye-tiao 132 CE embassy (via canonical Sinology). Both check out.

---

## 4. State of the project at end of session

### Papers
- **P1-core v3.0:** surgical cut ready, compiles clean, awaits Pak Amien review → JASREP submission ~2-3 weeks
- **P0 (new):** §1-3.2 drafted, §3.3-9 pending, needs SLR Fase D synthesis before continuing
- **5 under review:** P2 JCAA, P7 Antiquity, P8 OL, P11 Archipel, P17 ArchCalc — no change
- **P16, P18, P19:** drafting, unchanged

### SLR
- **Fase A + B:** COMPLETE across 10 subfields
- **Fase C** (CSV inventory extraction): PENDING — next session
- **Fase D** (synthesis → revised P0 evidence inventory): PENDING — next session

### Experiments
- **E208 Phase 1 + 2a:** COMPLETE
- **E208 Phase 2b** (ACD validation): PENDING
- **E208 Phase 3** (SEAlang kakawin corpus frequency): PENDING
- **E208 Phase 4** (merged etymology database): PENDING
- Total experiment count: **208** (E001–E208, E180 skipped)

### PhD track
- Verberne proposal SENT 2026-04-16, waiting response (~1-2 weeks expected)
- During waiting: autonomous NLP work prioritised (this session delivered substantially)

---

## 5. Decisions still pending from Pak Amien

1. **Review P1-core v3.0** — surgical cut completed, compiles clean; Pak Amien read + approve/revise
2. **Approve Path B formally** (implicit from "sesuai rekomendasi semua" but worth confirming in writing)
3. **Budget $50-200 for external statistics reviewer** before P0 submission
4. **DeepSeek API access** if/when cross-model critical review desired
5. **E208 Phase 2b execution approval** (would require ~2-3 hours and ACD access — free)

---

## 6. Next session priority queue (in order)

### Tier 1 — pipeline work (Claude-executable in isolation)
1. **SLR Fase C: CSV inventory extraction** — mechanical consolidation of 12 bibliography files into `docs/bibliography/_INVENTORY.csv` with structured columns (citekey, title, year, subfield, relation, chronology, method, quality, volcarch_use). Probably 1-2 hours.
2. **SLR Fase D: synthesis** — cluster analysis of findings, counter-evidence audit, revised P0 evidence inventory. Produces `docs/bibliography/_SYNTHESIS_for_P0.md`. Probably 2-3 hours.
3. **P0 §3.3 Linguistic channel draft** — now with nuanced E058/E208 interpretation. Uses findings from both. ~1-2 hours.
4. **P0 §3.4 Genomic channel draft** — uses SLR subfield 05 findings. ~1 hour.
5. **P0 §3.5 Colonial archive channel draft** — uses E091/E141. ~1 hour.
6. **P0 §3.6 Archaeometric channel draft** — NEW channel, combines Jia 2024 + Lankton Korean corpus + Berenike + Pejeng drums. ~2-3 hours.

### Tier 2 — After Tier 1
7. **P0 §4 Selective Survival** — bronze drums + glass beads reframe
8. **P0 §5 Wayang Living Evidence** — uses E205
9. **P0 §6 Six-Layer Framework** — consolidation of manifesto 6 layers
10. **P0 §7 Pre-registered Predictions** — falsification criteria
11. **P0 §8 Limitations** — honest caveats
12. **P0 §9 Conclusions**

### Tier 3 — Blocked by external
- Verberne PhD response (no action needed from Claude, just wait)
- Pak Amien P1-core v3.0 review
- Reviewer responses on P2/P7/P8/P11/P17

### Tier 4 — Lower priority / deferred
- E208 Phase 2b ACD validation
- E208 Phase 3 kakawin corpus frequency
- Borehole data mining (ESDM/PVMBG access unclear)
- DEM depression detection
- Satellite time-series analysis

---

## 7. Files Pak Amien should review before next session

1. `docs/research_notes/MATA_ELANG_15_2026_04_20.md` — strategic critique + Path B rationale
2. `papers/P1_taphonomic_framework/submission_jasrep_v3.0.pdf` — surgical cut result, 21pp
3. `papers/P0_invisible_civilization/SKELETON_v0.1.md` — masterpiece architecture
4. `papers/P0_invisible_civilization/draft_v0.1.pdf` — first 12pp draft
5. `docs/LITERATURE_SLR_PROGRESS.md` — SLR status + final scorecard
6. `experiments/E208_kakawin_nlp_pilot/README.md` — NLP pipeline nuanced finding

---

## 8. Open risks and unknowns

- **E208 heuristic Sanskrit undercount:** the 34.7% Sanskrit figure is a lower bound. Phase 2b ACD validation would constrain it. Current P0 Channel 3 language must flag this uncertainty.
- **SLR is light-to-medium depth, not exhaustive PRISMA.** Possibility remains that a framework-breaking paper exists in a subfield we did not deeply penetrate. Should be flagged as limitation in P0 §8.
- **P0 word target may drift.** 25-30K words is ambitious. Currently §1-3.2 is ~3,500 words; scaling up to target needs discipline.
- **Verberne's response framing unknown.** If she wants a specific NLP deliverable for follow-up, Claude should be ready to produce it fast. Current E208 pipeline is one candidate demo.

---

## 9. For next-session Claude (one-paragraph orientation)

Read MATA_ELANG_15, WORKSTATE, and this HANDOFF first. Path B is locked: P1-core v3.0 ready for Pak Amien review (JASREP submit in 2-3 weeks), P0 masterpiece being built with SLR evidence base (target JAnthArch, submit ~mid-July). SLR Fase A+B complete (10 subfields surveyed, zero counter-evidence, 3 cathedral anchors added — Jia 2024, Wolters 1967, Korean Jatim corpus). E208 NLP pipeline ran Phase 1+2a and produced a nuanced result (E058 extremes dampened at corpus scale — needs honest framing in P0). Next priority is SLR Fase C+D (CSV extraction + synthesis), then P0 §3.3-§3.6 drafting (remaining four channels). Pak Amien is at lunch; PhD-waiting-period rule applies (prioritise autonomous NLP-visible work, skip manual-bandwidth tasks).

---

*HANDOFF document produced 2026-04-20 end-of-session 18. Next session will consume it first.*

