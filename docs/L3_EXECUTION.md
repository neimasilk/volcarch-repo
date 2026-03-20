# L3: EXECUTION (Active Tasks)

**Status:** ACTIVE
**Sprint:** Sprint 11 — Senter v3 Computational Deepening
**Last updated:** 2026-03-16

---

## Current Focus

**Phase:** CONTRACTION + VALIDATION (post-Mata Elang #8 structural critique).
6 papers awaiting review. No new submissions. Build datasets, validate existing work, computational deepening.

**Experiments:** See `docs/EXPERIMENT_INDEX.md` for the complete **119-experiment** registry (E001-E115, minus 6 unused IDs).

---

## Papers Under Review (3 active, 3 rejected)

| Paper | Journal | MS# | Submitted | Status |
|-------|---------|-----|-----------|--------|
| ~~P1~~ | ~~Asian Perspectives~~ | 019A-0326 | 2026-03-10 | **REJECTED** 2026-03-17 (AI flag) |
| P2 | JCAA (Diamond OA) | #280 | 2026-03-11 | Under review |
| ~~P5~~ | ~~BKI~~ | — | 2026-03-09 | **REJECTED** 2026-03-19 ("too narrow") |
| P7 | Antiquity Project Gallery (Q1) | — | 2026-03-06 | Under review |
| P8 | Oceanic Linguistics (Q1) | OL-03-2026-11 | 2026-03-11 | Under review |
| ~~P9~~ | ~~JSEAS~~ | JSEAS-202603-051 | 2026-03-11 | **REJECTED** 2026-03-20 ("not suitable") |

### Retargeting (Diamond OA only — zero APC)
| Paper | Rejected From | Target Options |
|-------|---------------|----------------|
| P1 | Asian Perspectives | **EGQSJ** (Copernicus, ready now) |
| P5 | BKI | Archeologia e Calcolatori / J. Pacific Archaeology / Wacana |
| P9 | JSEAS | DHQ / Wacana / Archipel (HOLD until P2/P8 outcome) |

## Papers In Progress

| Paper | Target | Status | Next Step |
|-------|--------|--------|-----------|
| P11 | Wacana (UI, Scopus Q2, Diamond OA) or Indonesia (Cornell) | v0.3 drafted (18pp) | Manual review → submit post-mudik |
| P16 | DHQ (ADHO, Scopus+WoS, Diamond OA) | Draft v0.1 (19pp, 6 figs) | User review → expand → submit |
| P17 | Archeologia e Calcolatori / J. Pacific Arch. (Diamond OA) | Draft v0.2 (22pp) | User review → submit |
| P18 | HOLD — wait for 1 acceptance | Draft v0.1 (16pp) | Strengthen, do not submit yet |
| D1 | Zenodo (free) | Draft ready | Zenodo deposit |
| D2 | Zenodo (free) | Draft ready | Zenodo deposit |

---

## Active Tasks

> **Tactical next-actions → see `docs/WORKSTATE.md`** (session continuity contract with concrete file paths and next steps).

### Structural Critique Follow-up (highest priority)
1. **Colonial data verification** — Open 10 E070 entries on Delpher.nl, verify manually
2. ~~**Code review**~~ — DONE (E027, E065, E069, E082, E083). E065 path fix applied. 4/5 ready, all functional.
3. ~~**Dependency freeze**~~ — DONE (`requirements_freeze.txt` created 2026-03-16)
4. ~~**Consilience reframing**~~ — DONE (L1 Section 5 added, manifesto v3.4, 2026-03-16)

### Paper Tasks
5. ~~**P11 v0.3**~~ — DONE (18pp, E084+E083+E086 integrated, self-citations removed, 10 refs)
6. **D1+D2** — Zenodo deposit + JOAD submission (APC waiver decision needed)
7. **P9 Word file** — Send to Eileen Shen (JSEAS) if not already done
8. **Cross-citation strategy** — P5↔P9, P8↔P9 differentiation statements

### If Reviewer Responses Arrive
9. **P1 revision** — Japan paragraph ready (`ADV1_japan_comparanda.md`)
10. **P8 revision** — Negative control reframing ready (`ADV5_negative_control.md`)

### Senter v3 — GPU Runs ~~Pending~~ COMPLETED (2026-03-17)
11. ~~**E090 v5**~~ — DONE. 16 BERTopic topics, 8/8 converge, VOLCANO z=7.39.
12. ~~**E094**~~ — DONE. Volcanic silence 0.244 (lowest), C11→C12 rupture.
13. ~~**E096**~~ — DONE. 929 CE shift p=0.0003. Royal surges, ritual vanishes.
14. **E076 v2** — `02_multi_tile_analysis.py`. Satellite NDVI. USER NETWORK ~30 min. **STILL PENDING.**
15. ~~**E095**~~ — DONE (#99). Cross-lingual XLM-R/ML-SBERT. Validates E094 (rho=0.336).

### Dissemination (see `docs/VOLCARCH_Dissemination_Roadmap_v1.0.md`)
16. ~~**Dokumen Jembatan v0.2**~~ — DONE. PDF generated. NotebookLM slides extracted (14 slides).
16. **Infographic** — E080 fieldwork convergence map. Generated (Senter v3).
17. **Preprint strategy** — P7 → EarthArXiv, P8 → arXiv cs.CL. Can start now.
18. **Zenodo survey paper** — Indonesian-language research agenda paper. Target: May-June 2026.
19. **Phase 2 outreach** — BALARJATIM, ITB/UGM Geologi, UB Malang. **GATE: 1 paper accepted first.**

### Backlog
- Survey intensity normalization (BPCB coverage per km²)
- GPR/LiDAR partnership for field validation → now part of Dissemination Phase 4
- Conference presentation (EHPA or Berkala Arkeologi) → now part of Dissemination Phase 3
- Berkala Arkeologi Indonesian-language gateway paper → Dissemination Phase 3B

---

## User Decisions Needed

- [ ] JOAD APC £374: proceed with waiver request, or Zenodo? (JOAD has waiver fund)
- [ ] Preprint: post P7 to EarthArXiv now? (Roadmap recommends yes)
- [ ] Preprint: submit P8 to arXiv cs.CL? (Endorsement already available)
- [ ] D1/D2: JOAD with waiver, or deposit directly on Zenodo?
- [ ] Dissemination: start Dokumen Jembatan draft now, or wait for acceptance?

---

<details>
<summary>Archived Tasks (completed, resolved, or killed)</summary>

All completed tasks from Sprint 0-9 are archived here. See JOURNAL.md for full history.

**Sprint 9 (completed):**
- TASK-055: ADV-3 Survey Intensity → COMPLETE (VOLCARCH SUPPORTED, p=0.0015)
- TASK-056: E070 Colonial Register → COMPLETE (52 entries)
- TASK-057: P11 Reframe → COMPLETE (v0.2 methodology paper)
- TASK-052: E048-E050 block → COMPLETE
- TASK-053: E048-E060 Mata Elang #5 → COMPLETE (14 experiments)
- TASK-054: E061-E065 block → COMPLETE
- TASK-048: Revision ammo audit → DONE
- TASK-044: P8 submission → DONE (MS# OL-03-2026-11)
- TASK-041: P7 submission → DONE (Antiquity)
- TASK-045: P5 submission → DONE (BKI)
- TASK-036/037: P1/P2 submission → DONE
- TASK-032: ORCID → DONE (0000-0002-1848-167X)

**Killed:**
- TASK-028: P3 scoping → KILLED (E017 FAILED)
- P14 → KILLED (Bonferroni)
- P15 → DISSOLVED into P5 ammo

**Sprint 0-3:** See git history for TASK-001 through TASK-025.

</details>

---

*This document tracks ONLY what needs to happen NOW. For experiment history, see `docs/EXPERIMENT_INDEX.md`.
For research log, see `docs/JOURNAL.md`. For strategy, see `docs/L2_STRATEGY.md`.*
