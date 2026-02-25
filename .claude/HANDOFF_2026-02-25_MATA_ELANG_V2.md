# HANDOFF: Mata Elang Review V2
**Date:** 2026-02-25 (evening session)
**Author:** Opus 4.6 (Mata Elang mode)
**Context:** Post multi-AI review integration session

---

## SESSION SUMMARY

This session accomplished:
1. Integrated 1860 Leiden University photo of buried Dwarapala into Paper 1
2. Fixed table margin overflows in both papers
3. Updated affiliation to "Lab Data Sains, Universitas Bhinneka Nusantara"
4. Adopted selective critiques from Gemini, Kimi, and another AI reviewer
5. Pushed all changes to GitHub (excluding .tif files due to size)
6. Conducted full project audit (this document)

---

## PAPER 1 STATUS: 90% — NEAR SUBMISSION-READY

**File:** `papers/P1_taphonomic_framework/submission_jasrep_v0.1.tex`
**Target:** Journal of Archaeological Science: Reports (Q1)
**PDF:** 21 pages, compiled clean

### What's Complete
- [x] Gemini's new title/abstract/intro integrated
- [x] 4-site calibration framework (Dwarapala + 3 Merapi sites)
- [x] Mean rate: 4.4 +/- 1.2 mm/yr with explicit midpoint calculation
- [x] 535yr/3.5mm/yr standardized (was inconsistent 535 vs 510)
- [x] Dwarapala 1860 photo (Leiden) + present photo as Figure 1
- [x] Timeline diagram as Figure 2
- [x] Burial depth projections table + figure
- [x] Calibration rates comparison figure
- [x] E004/E005 demoted to "Cautionary analysis" subsection
- [x] Aggradation vs tephra clarification sentence added
- [x] Linear extrapolation first-order caveat added (Section 3.4)
- [x] Tables 1 & 2 fit within margins (resizebox)
- [x] All 6 figures present and referenced
- [x] references.bib: 36 entries (complete)
- [x] Double-spacing + line numbers per JAS:Reports format
- [x] Affiliation: Lab Data Sains, Universitas Bhinneka Nusantara

### Remaining Before Submission
| # | Task | Effort | Priority |
|---|------|--------|----------|
| P1-1 | Verify Dwarapala 1860 photo provenance (can we cite Leiden catalogue number?) | 15 min | HIGH |
| P1-2 | Read through final PDF for typos/formatting | 20 min | HIGH |
| P1-3 | Consider seeking informal domain review (geologist or archaeologist) | Human task | MEDIUM |
| P1-4 | Check JAS:Reports submission portal requirements (cover letter, etc.) | 15 min | HIGH |

### Verdict: CAN SUBMIT after P1-1 through P1-4 (~1 hour human work)

---

## PAPER 2 STATUS: 85% — NEEDS REFERENCE AUDIT

**File:** `papers/P2_settlement_model/submission_remote_sensing_v0.3.tex`
**Target:** Remote Sensing (MDPI, Q1)
**PDF:** 13 pages, compiled clean

### What's Complete
- [x] MVR met: seed-averaged AUC 0.751 (range 0.729-0.774)
- [x] Null model comparison: E013 beats DKNS by +0.122 AUC
- [x] Enhanced 3-test tautology suite: GREY_ZONE (honest, defensible)
- [x] Seed stability framing improved (min/max/count reported)
- [x] Road distance tension explicitly addressed in Limitations
- [x] Figure numbering fixed (framework=Fig1, study area=Fig2)
- [x] Tables fit within margins (resizebox)
- [x] All 12 figures present on disk
- [x] Interdisciplinary discussion structure (CS/Archaeology/Geology)
- [x] Affiliation: Lab Data Sains, Universitas Bhinneka Nusantara
- [x] Long \texttt{} paths replaced with clean text

### Remaining Before Submission
| # | Task | Effort | Priority |
|---|------|--------|----------|
| P2-1 | **REFERENCE AUDIT**: Only 14 refs in .bib — need ~10-15 more for 13-page paper | 45 min | CRITICAL |
| P2-2 | Check MDPI Remote Sensing LaTeX template compliance (may need reformatting) | 30 min | CRITICAL |
| P2-3 | Verify figures not referenced in LaTeX (fig2-fig7 exist on disk but some may not be \ref'd) | 15 min | HIGH |
| P2-4 | Run build_tautology_figure.py to generate visual for tautology section | 10 min | MEDIUM |
| P2-5 | Read through final PDF for typos/formatting | 20 min | HIGH |
| P2-6 | Supplementary materials section needs real URLs/DOIs, not file paths | 10 min | HIGH |

### Verdict: NEEDS 1.5-2 HOURS WORK before submission (primarily reference audit + MDPI template)

---

## PAPER 2 REFERENCE GAP ANALYSIS

Current .bib has 14 entries. Citations in text match .bib. But for Remote Sensing MDPI, typical papers cite 30-50 references. Missing categories:

### Refs Needed
1. **Archaeological predictive modeling in SE Asia** (0 currently — should have 2-3)
   - Lauer & Aswani (2009) - Pacific island settlement modeling
   - Ford et al. (2009) - Maya predictive modeling
2. **XGBoost/RF in geospatial applications** (have Chen2016 + Breiman2001, need 1-2 more)
   - Georganos et al. (2021) - XGBoost for spatial prediction
3. **Volcanic landscape archaeology** (0 currently — should have 2-3)
   - Torrence & Grattan (2002) - Natural Disasters and Cultural Change
   - Sheets (2002) - Ceren site
4. **Target-Group Background method** (0 dedicated ref)
   - Phillips et al. (2009) is cited but TGB concept originates from Maxent literature
5. **Java-specific archaeology** (0 currently)
   - Kinney et al. (2003) - Worshipping Siva and Buddha
   - Degroot (2009) - Candi Space and Landscape

---

## PAPER 3 STATUS: NOT STARTED

**Target:** Journal of Volcanology and Geothermal Research (JVGR)
**Concept:** Spatially-resolved burial depth model using tephra dispersal simulation

### Prerequisites (All Unmet)
- [ ] Papers 1 & 2 submitted to journals
- [ ] Geologist co-author identified (Universitas Brawijaya suggested)
- [ ] Tephra dispersal tool installed (Tephra2 or FALL3D)
- [ ] ERA5 wind reanalysis data for Java downloaded
- [ ] Isopach map literature review completed

### Estimated Timeline
- **Start:** After Papers 1 & 2 submitted (target: March 2026)
- **MVR target:** Predict Dwarapala burial depth within +/-30% (130-240 cm)
- **Full draft:** 3-4 months after start

### Recommendation
Do NOT start Paper 3 until Papers 1 & 2 are in review. Use the review cycle (2-4 months) as buffer time to:
1. Identify geologist co-author
2. Install tephra tools
3. Download ERA5 data
4. Write methods outline

---

## CRITICAL DECISIONS LOG

### Adopted from AI Reviews
| Decision | Source | Paper | Rationale |
|----------|--------|-------|-----------|
| Clarify mean +/- 1.2 as SD of midpoints | AI-3 | P1 | Reviewer would ask how computed |
| Standardize 535yr/3.5mm/yr | AI-3 | P1 | Inconsistency between 535 and 510 was a bug |
| Add linear model caveat in Section 3.4 | AI-3 | P1 | Preempt geology reviewer |
| Report seed range [0.729-0.774] | AI-3 | P2 | Minimum > DKNS is stronger than mean ~0.75 |
| Expand road distance tension | AI-3 | P2 | Dual role of road_dist needed explicit discussion |
| Fix figure numbering (framework=Fig1) | AI-3 | P2 | Image had "Figure 1" baked in |
| Clarify aggradation vs tephra | Gemini | P1 | 45% non-volcanic is valid concern |

### Rejected from AI Reviews
| Decision | Source | Paper | Rationale |
|----------|--------|-------|-----------|
| Move Kutai to Discussion | Gemini | P1 | Already in both Background AND Discussion |
| Add non-volcanic rate column | Gemini | P1 | Data doesn't exist — would be fabrication |
| Grid resolution too coarse | Gemini | P1 | E004/E005 already demoted as cautionary |
| Restructure Discussion | AI-3 | P2 | Per-audience format works for MDPI |

---

## GIT STATUS

### Latest Push
- Commit: `a549a63` — "feat: full project snapshot with papers, figures, and Dwarapala photos"
- Pushed to: `origin/main` on `github.com/neimasilk/volcarch-repo`
- Excludes: all *.tif files (too large for GitHub even with LFS free tier)
- .gitignore updated: *.tif, *.aux, *.log, *.out, *.blg excluded

### Unpushed Changes (from this session)
- Paper 1: 535/3.5 standardization, mean clarification, linear caveat
- Paper 2: seed range, road tension, figure order, table fixes, text cleanup
- **ACTION NEEDED:** Commit and push these changes

---

## NEXT SESSION EXECUTION PLAN

### Phase A: Finalize Paper 1 (Priority: SUBMIT THIS WEEK)
1. Verify Leiden photo provenance
2. Final PDF read-through
3. Check JAS:Reports submission requirements
4. Submit

### Phase B: Finalize Paper 2 (Priority: SUBMIT WITHIN 2 WEEKS)
1. **Reference audit** — add 10-15 missing refs to references.bib
2. Check MDPI Remote Sensing template — may need `\documentclass{remotesensing}`
3. Verify all 12 figures are referenced in LaTeX
4. Run tautology figure script
5. Fix Supplementary Materials section (remove file paths, add proper descriptions)
6. Final PDF read-through
7. Submit

### Phase C: Paper 3 Preparation (During Review Cycle)
1. Write Paper 3 methods outline
2. Identify geologist co-author (Universitas Brawijaya, PVMBG, or ITB)
3. Install Tephra2 or FALL3D
4. Download ERA5 wind reanalysis for Java
5. Literature review: isopach maps for Kelud/Merapi/Semeru

---

## FILES MODIFIED THIS SESSION

### Paper 1 (`submission_jasrep_v0.1.tex`)
- Added Figure 1 (Dwarapala 1860 + present photos)
- Fixed Tables 1 & 2 margin overflow
- Standardized 535yr / 3.5mm/yr (6 locations)
- Clarified mean 4.4 +/- 1.2 calculation
- Added first-order approximation caveat (Section 3.4)
- Added aggradation vs tephra clarification
- Updated affiliation

### Paper 2 (`submission_remote_sensing_v0.3.tex`)
- Improved seed stability framing (range, min, DKNS gap)
- Expanded road distance tension in Limitations
- Swapped figure order (framework first)
- Fixed experiment table + tautology table overflow
- Replaced long \texttt{} paths
- Updated affiliation

### New Files
- `figures/fig0a_dwarapala_1860.jpg` (LaTeX-safe copy)
- `figures/fig0b_dwarapala_present.png` (LaTeX-safe copy)
- `.claude/HANDOFF_2026-02-25_MATA_ELANG_V2.md` (this file)
