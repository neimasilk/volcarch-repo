# WORKSTATE — Session Continuity Contract

**READ THIS FIRST. Continue in-progress items before starting new work.**
**Last updated:** 2026-03-18 (P1 Zenodo published + EGQSJ Copernicus reformat done — pre-Lebaran pause)

---

## IN PROGRESS

- **P1 → ZENODO PUBLISHED, NEXT: EGQSJ**
  - Files: `papers/P1_taphonomic_framework/submission_v1.0.tex`, `references.bib`
  - **Zenodo DOI: 10.5281/zenodo.19081502** — published 2026-03-18, CC-BY 4.0
  - Authors: Mukhlis Amien + Go Frendi Gunawan (Universitas Bhinneka Nusantara)
  - Target journal: **EGQSJ** (E&G Quaternary Science Journal, Copernicus, Diamond OA, Scopus+WoS)
  - **Copernicus reformat: DONE** — `submission_egqsj_v1.0.tex` compiles clean (1.22 MiB)
  - Next: register manuscript di editor.copernicus.org → submit EGQSJ
  - Fallback: Berkala Arkeologi (BRIN, gratis, tapi tanpa Scopus/WoS)
  - Manual verification needed: gertisser2012 DOI, baylisssmith1980 details, manguin2011 book title/pages, ov1925 specificity, miksic2004 correct DOI, french2003 correct DOI
- **P11 finalization** — v0.3 drafted (18pp), closest to submission-ready
  - Files: `papers/P11_volcanic_informedness/draft_v0.3.tex`, `SUBMISSION_PREP.md`
  - Next: User manual review → Chicago 17th citations → submit to indonesia-journal@cornell.edu
- **P17 drafting** — Draft v0.2 (22pp, 5 figures, ~7K words, 30 refs). Compiles cleanly.
  - Files: `papers/P17_two_javas/draft_v0.2.tex`, `p17_references.bib`, `figures/`
  - Next: User review → add Fig 6 (conceptual map) → check Antiquity submission guidelines → submit
- **P16 drafting** — Draft v0.1 EXPANDED (27pp, ~8K words, 6 figures). Compiles cleanly.
  - Files: `papers/P16_computational_textual_archaeology/draft_v0.1.tex`, `p16_references.bib`, `figures/`
  - Next: User review → check DSH submission guidelines → rename v0.2 → submit
- **P18 drafting (HOLD)** — Draft v0.1 (16pp, 6 figures, 15 refs). "What Words Remember." Compiles cleanly. **Do NOT submit yet** — wait for 1-2 acceptances.
  - Files: `papers/P18_invisible_civilization/draft_v0.1.tex`, `p18_references.bib`, `figures/`, `generate_figures.py`
  - Needs: Expand to ~9K words (Background + E113/E114 integration), Fig 6 (West Java map), literature verification, World Archaeology guidelines
- **E076 v2 satellite** — Script written, needs internet (~30 min)
  - Files: `experiments/E076_satellite_ndvi/02_multi_tile_analysis.py`
- **Colonial data verification** — 10 E070 entries on Delpher.nl (user manual task)

## PAPERS UNDER REVIEW (WAIT)

| Paper | Journal | MS# | Submitted | Revision Ammo |
|-------|---------|-----|-----------|:---:|
| ~~P1~~ | ~~Asian Perspectives~~ | 019A-0326 | REJECTED 2026-03-18 (AI flag) |
| **P1** | **EGQSJ** (Copernicus, Diamond OA) | — | **READY TO SUBMIT** — `submission_v1.0.tex` |
| P2 | JCAA (Diamond OA) | #280 | 2026-03-11 | **3 files** (incl. E109 confound) |
| P5 | BKI (Diamond OA) | — | 2026-03-09 | **5 files** (incl. E112 domain gradient) |
| P7 | Antiquity Project Gallery | — | 2026-03-06 | 1 file |
| P8 | Oceanic Linguistics (Q1) | OL-03-2026-11 | 2026-03-11 | **5 files** (incl. E107 ADV-5 RESOLVED) |
| P9 | JSEAS (NUS Press) | JSEAS-202603-051 | 2026-03-11 | **3 files** (incl. E114 comparative) |

**Total revision ammo: 26 files.** All 6 papers have pre-computed responses to anticipated critiques + new findings from E107-E114.

## KEY FINDINGS THIS SESSION (reference only)

| Finding | Experiment | Impact |
|---------|-----------|--------|
| ADV-5 resolved: C5 = Mon-Khmer substrate | E107 | L4 upgraded, P8 framing restored |
| Demographic gap 3,220× | E108 | Null hypothesis rejected |
| 5-factor cascade P=0.058% matches data | E110 | Core theoretical model |
| PAN \*surat indigenous (~5000 BP) | E112 | Writing concept pre-dates India |
| Agriculture 91% native, Religion 86% Sanskrit | E112 | Sanskritization = elite overlay |
| No inscription learning curve | E113 | Pre-existing organic-media tradition |
| Nusantara #1/10 pre-literate societies | E114 | CCI=23, z=2.12 |
| Survey deficit = 40× leverage, burial = 1.7× | E110 | Reframe: survey first, burial second |
| West Java smoking gun (Buni + Batujaya) | E110 | Within-island taphonomic control |

## BLOCKED

- **D1+D2 JOAD** — APC waiver decision (£374 each) or Zenodo (free)
- **Dissemination Phase 2** — Wait for 1 acceptance

## COMPLETED PREVIOUS SESSIONS

- **P1 Final Review (2026-03-18)** — Review fixes (duplicate content, textbook filler, West Java claim, 3220x transparency, ov1925 reframe), AI disclosure trimmed, internal jargon cleaned (E083/RQ/H labels), DOIs verified via Playwright (3 corrected, 1 removed), versioning cleanup (renamed v1.0, archived obsolete variants).
- **Structural Audit + Vocabulary Archaeology (2026-03-17)** — 9 new experiments (E107-E114), P18 draft, manifesto v4.0, 6 new revision ammo files, cascade model, West Java comparandum. Total: 115 experiments.
- **Two Javas Sprint (2026-03-17)** — E099-E106, P17 v0.2 (22pp), P16 expanded (27pp), E090/E094/E096 GPU runs, P5 revision ammo.
- **Senter v3 (2026-03-16)** — E092-E098, anomaly detection, GPU scripts, Dokumen Jembatan v0.2.
- **Consolidation (Sessions 2-5)** — P7 preprint DOI live. E089 v4. P11 v0.3. Code review.
- **Senter v2** — E091 OV NLP. E089 v3. E076 v2 script.

## SESSION PROMPT

STATUS: **115 experiments**. 6 under review + 4 drafting. 26 revision ammo. Phase = CONSOLIDATION.
P1 preprint live on Zenodo (DOI: 10.5281/zenodo.19081502). Copernicus format ready.
PRIORITIES: P1 submit EGQSJ → P11 submit → P17 submit → P16 submit → P18 strengthen (hold).
**NEXT ACTION:** Register manuscript at editor.copernicus.org, upload `submission_egqsj_v1.0.tex` + PDF + figures, submit to EGQSJ.

## DO NOT WORK ON

- New experiments (115 is comprehensive)
- New paper drafts beyond strengthening P18
- P18 submission (wait for acceptances)
- Phase 2 dissemination (wait for 1 acceptance)
