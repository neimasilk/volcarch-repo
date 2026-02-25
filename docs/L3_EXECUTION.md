# L3: EXECUTION (Active Tasks)

**Status:** ACTIVE -- Updates per week or per experiment.
**Sprint:** Sprint 3 -- Submission Preparation
**Last updated:** 2026-02-25

---

## Current Focus

**Immediate goal:** Submit Paper 1 and Paper 2 to journals.

### Paper 1 (JAS:Reports): 90% READY
- LaTeX + PDF compiled clean (21 pages)
- All figures complete (including Dwarapala 1860 photo)
- Remaining: verify photo provenance, final read-through, submit
- **Target: submit this week**

### Paper 2 (Remote Sensing MDPI): 85% READY
- LaTeX + PDF compiled clean (13 pages)
- MVR met: seed-averaged AUC 0.751, range [0.729-0.774]
- Remaining: **reference audit (CRITICAL)**, MDPI template check, final read-through
- **Target: submit within 2 weeks**

### Paper 3 (JVGR): NOT STARTED
- Blocked pending Paper 1+2 submission
- Needs geologist co-author, tephra dispersal tools, ERA5 wind data

## Sprint 0 Summary: COMPLETE

All data collection and analysis scripts written and executed:
- E001: 666 sites collected, 297 geocoded (296 from OSM/Wikidata + 1 duplicate)
- E002: 8 eruption records (seed data; full GVP still needs manual download)
- E003: Copernicus GLO-30 DEM downloaded for Malang Raya + full Jawa Timur
- E004: H1 first test â€” rho=-0.991, INCONCLUSIVE (dominated by survey bias)
- E005: H1 terrain-controlled test â€” rho=-0.364, H1 not supported from distribution data

**Key decision:** Paper 1 reframed as METHODOLOGICAL framework.
- Argument: observable site distribution cannot test taphonomic bias due to survey + survivorship bias
- Core contribution: Dwarapala calibration (3.6 mm/year sedimentation) + computational framework
- H1 proposed as hypothesis requiring fieldwork to confirm, not a proven statistical finding

---

## Active Tasks

### TASK-008: Increase geocoded site count
**Status:** COMPLETE (2026-02-24)
**Description:** 369/666 sites in east_java_sites.geojson have accuracy_level='no_coords'.
Geocode using OSM Nominatim API â€” query "<name>, Jawa Timur, Indonesia" with bbox filter.
**Expected:** 30â€“60% success rate â†’ ~100â€“220 additional geocoded sites
**Output:** Updated `data/processed/east_java_sites.geojson` + `data/processed/geocoding_report.txt`

### TASK-009: Re-run E004 + E005 with enriched site data
**Status:** COMPLETE (2026-02-24) â€” rho=-0.955 (was -0.991), terrain rho=-0.358 (was -0.364)
**Description:** After geocoding, re-run E004 (density analysis) and E005 (terrain suitability)
with the larger geocoded dataset to see if the pattern changes.

### TASK-010: Paper 1 first draft
**Status:** COMPLETE (2026-02-24) — full draft v0.2 in `papers/P1_taphonomic_framework/draft_v0.1.md`
**Description:** Write full first draft of Paper 1.
**Target journal:** Journal of Archaeological Science: Reports
**File:** `papers/P1_taphonomic_framework/`

### TASK-025: Paper 1 LaTeX submission package
**Status:** COMPLETE (2026-02-25, updated evening)
**Description:** Full submission package for Paper 1 (JAS:Reports).
- LaTeX manuscript: `papers/P1_taphonomic_framework/submission_jasrep_v0.1.tex`
- Compiled PDF: `papers/P1_taphonomic_framework/submission_jasrep_v0.1.pdf` (21 pages)
- BibTeX references: `papers/P1_taphonomic_framework/references.bib` (36 entries)
- Figures (6 total):
  - `fig0a_dwarapala_1860.jpg` + `fig0b_dwarapala_present.png` — before/after burial photos
  - `fig1_dwarapala_timeline.png` — burial timeline diagram
  - `fig2_burial_depth_projections.png` — depth by era with GPR range
  - `fig3_calibration_rates.png` — 4-site cross-system sedimentation rates
  - `fig4_density_vs_distance.png` — E004 results + interpretive panel
- Author: Mukhlis Amien, Lab Data Sains, Universitas Bhinneka Nusantara
- Format: double-spaced, line-numbered, elsarticle-harv bibliography style
- Revisions applied: 535yr/3.5mm/yr standardization, mean clarification, linear caveat, aggradation note

### TASK-026: Paper 1 submission (NEXT)
**Status:** PENDING
**Description:** Submit Paper 1 to JAS:Reports.
- [ ] Verify Dwarapala 1860 photo provenance (Leiden catalogue number)
- [ ] Final PDF read-through for typos
- [ ] Check JAS:Reports submission portal requirements
- [ ] Submit via journal portal

### TASK-027: Paper 2 reference audit + submission
**Status:** PENDING
**Description:** Complete Paper 2 for Remote Sensing MDPI submission.
- [ ] Add 10-15 missing references to references.bib (currently 14, need ~25-30)
- [ ] Check MDPI Remote Sensing LaTeX template compliance
- [ ] Verify all figures referenced in LaTeX (fig2-fig7 may be unreferenced)
- [ ] Fix Supplementary Materials section (remove file paths)
- [ ] Final PDF read-through
- [ ] Submit via MDPI portal

### TASK-028: Paper 3 scoping and co-author search
**Status:** PENDING (blocked on P1+P2 submission)
**Description:** Begin Paper 3 preparation during review cycle.
- [ ] Identify geologist co-author (Universitas Brawijaya / PVMBG / ITB)
- [ ] Install tephra dispersal tool (Tephra2 or FALL3D)
- [ ] Download ERA5 wind reanalysis for Java
- [ ] Literature review: isopach maps for Kelud/Merapi/Semeru
- [ ] Write Paper 3 methods outline

### TASK-015: E008 â€” Settlement Suitability Model v2 (feature tuning)
**Status:** COMPLETE (2026-02-24) â€” AUC=0.695, REVISIT (not kill signal)
- Added river distance raster from OSM Overpass (9,730 waterways); AUC improved +0.036
- River_dist ranks 3rd in importance. Trend: E007=0.659 â†’ E008=0.695 â†’ E009=0.664
- Weak folds 2â€“3 suggest survey-bias root cause (positive samples cluster in surveyed areas)

### TASK-017: E009 â€” Settlement Suitability Model v3
**Status:** REVISIT (2026-02-24) â€” Path A complete, MVR not met
**Description:** Path A implemented (SoilGrids clay+silt added to E008 feature set).
**Result:**
- XGBoost AUC=0.664 Â± 0.049 (TSS=0.337 Â± 0.083)
- RandomForest AUC=0.643 Â± 0.054 (TSS=0.312 Â± 0.072)
- Challenge 1 PASS: rho=-0.266 (tautology-free)
- Delta vs E008: -0.031 (performance drop)
**Decision:** Move to Path B (Target-Group Background pseudo-absences).

### TASK-018: E010 â€” Target-Group Background pseudo-absences
**Status:** REVISIT (2026-02-24) â€” Path B complete, MVR not met
**Description:** Replaced random pseudo-absences with road-accessibility weighted TGB sampling.
**Result:**
- XGBoost AUC=0.711 Â± 0.085 (TSS=0.384 Â± 0.150)
- RandomForest AUC=0.699 Â± 0.081 (TSS=0.380 Â± 0.130)
- Challenge 1 PASS: rho=-0.142 (tautology-free)
- Delta vs E008: +0.016 (improved, but below 0.75)
**Decision:** Continue with TGB tuning (parameter sweep + richer accessibility proxy).

### TASK-019: E011 - TGB tuning and proxy enrichment
**Status:** REVISIT (2026-02-24) - parameter sweep complete, MVR not met
**Description:** Tuned TGB sampling over 12 configurations with fixed spatial CV splits.
**Result:**
- Best config: decay=16km, max_road_dist=60km
- XGBoost AUC=0.725 +- 0.084 (TSS=0.447 +- 0.184)
- RandomForest AUC=0.716 +- 0.081 (TSS=0.408 +- 0.147)
- Challenge 1 PASS: rho=-0.169 (tautology-free)
- Delta vs E010: +0.014, vs E008: +0.030
**Decision:** Continue with proxy enrichment (expanded road classes / survey polygons).

### TASK-020: E012 - TGB proxy enrichment
**Status:** REVISIT (2026-02-24) - proxy enrichment complete, MVR not met
**Description:** Added expanded road classes (`unclassified`, `residential`, `service`)
to accessibility proxy and reran fixed-split TGB sweep.
**Result:**
- Best config: decay=12km, max_road_dist=20km
- XGBoost AUC=0.730 +/- 0.085 (TSS=0.420 +/- 0.170)
- RandomForest AUC=0.724 +/- 0.081 (TSS=0.413 +/- 0.152)
- Challenge 1 PASS: rho=-0.160 (tautology-free)
- Delta vs E011: +0.005, vs E008: +0.035
**Decision:** Continue to hybrid bias-correction strategy (E013).

### TASK-021: E013 - Hybrid bias-corrected background
**Status:** SUCCESS (2026-02-24) - MVR met
**Description:** Combined expanded-road TGB with hybrid controls (regional quota blend + hard negatives).
**Result:**
- Best config: region_blend=0.00, hard_frac_target=0.30
- XGBoost AUC=0.768 +/- 0.069 (TSS=0.507 +/- 0.167)
- RandomForest AUC=0.742 +/- 0.070 (TSS=0.458 +/- 0.126)
- Challenge 1 PASS: rho=-0.229 (tautology-free)
- Delta vs E012: +0.038, vs E008: +0.073
**Decision:** Paper 2 GO (threshold achieved).

### TASK-022: Paper 2 outline and draft kickoff
**Status:** COMPLETE (2026-02-24) - draft package assembled
**Description:** Start Paper 2 writing package after E013 success.
- Draft outline at `papers/P2_settlement_model/outline.md`
- Draft manuscript v0.3 at `papers/P2_settlement_model/draft_v0.3.md` (submission-prep baseline)
- Discussion + Limitations sections integrated for internal review baseline
- Figure assets + caption callouts prepared in draft (`build_figures.py` pipeline)
- Robustness supplement completed (`robustness_checks.py`): 20 alternate-seed runs + bootstrap CI
  archived at `papers/P2_settlement_model/supplement/e013_seed_stability.csv` and
  `papers/P2_settlement_model/supplement/e013_robustness_summary.txt`
- Block-size sensitivity completed (`block_size_sensitivity.py`): ~40km/~50km/~60km
  outputs at `papers/P2_settlement_model/supplement/e013_blocksize_summary.csv`
  and `papers/P2_settlement_model/figures/fig7_e013_blocksize_sensitivity.png`
- Journal-format pass completed: references and data/code availability section added
  to `papers/P2_settlement_model/draft_v0.3.md`

### TASK-023: Paper 2 internal review and submission packaging
**Status:** COMPLETE (2026-02-24) - submission package baseline ready
**Description:** Prepare draft for journal submission workflow.
- Internal review pass 1 completed (claim tightening + robustness-consistent abstract)
- Submission checklist initialized at `papers/P2_settlement_model/submission_checklist.md`
- Journal-style metadata sections added (Data/Code availability, Funding, COI)
- v0.3 template-aligned draft completed (`papers/P2_settlement_model/draft_v0.3.md`)
- Remote Sensing section mapping prepared at
  `papers/P2_settlement_model/remote_sensing_template_map.md`
- Author contributions template prepared at
  `papers/P2_settlement_model/author_contributions_template.md`
- Author Contributions placeholder inserted in
  `papers/P2_settlement_model/draft_v0.3.md`
- DOI/URL verification completed and logged at
  `papers/P2_settlement_model/reference_verification_2026-02-24.md`
- Submission-formatted file created:
  `papers/P2_settlement_model/submission_remote_sensing_v0.1.md`
- Dependency lock file generated:
  `papers/P2_settlement_model/requirements_submission_lock.txt`
- Author Contributions finalized with confirmed author identity:
  Mukhlis Amien (`amien@ubhinus.ac.id`)
- All non-optional submission checklist items completed.
- Milestone committed and pushed to GitHub (`main`, commit `453f36b`).

### TASK-024: Multidisciplinary LaTeX manuscript package (CS + Archaeology + Geology)
**Status:** COMPLETE (2026-02-24)
**Description:** Reframe Paper 2 for mixed-discipline readability and deliver full LaTeX draft.
- Installed MiKTeX via winget (`MiKTeX.MiKTeX 25.12`)
- Created full manuscript in LaTeX:
  `papers/P2_settlement_model/submission_remote_sensing_v0.2.tex`
- Expanded introduction with explicit interdisciplinary literature framing:
  - computer science (spatial CV, background bias handling)
  - archaeology (predictive modeling and sampling process)
  - geology (tephra transport/deposition and volcanic context)
- Added dedicated visual assets (illustration + diagrams):
  - `fig1_interdisciplinary_framework.png`
  - `fig8_pipeline_overview.png`
  - `fig9_interpretation_bridge.png`
  generated by `papers/P2_settlement_model/build_interdisciplinary_visuals.py`
- Compiled output PDF:
  `papers/P2_settlement_model/submission_remote_sensing_v0.2.pdf`

### TASK-011: Download full GVP eruption data
**Status:** COMPLETE (2026-02-25)
**Description:** GVP eruption database downloaded and processed.
- `tools/scrape_gvp.py` created — downloads + processes GVP eruption database
- Full GVP database: 9,902 global eruptions in `data/raw/gvp/GVP_Eruption_Search_Result.xlsx`
- Filtered to 4 target volcanoes: **168 confirmed eruptions** (was 8 seed records)
  - Kelud: 37, Semeru: 63, Bromo: 67, Arjuno-Welirang: 1
- Output: `data/processed/eruption_history.csv` (168 records with ashfall estimates)
- Ashfall since 1268 CE: 174.6 cm from 41 events → implied rate 2.30 mm/yr (lower bound)

---

## Upcoming Tasks (Backlog â€” Sprint 2)

- TASK-012: **Survey intensity normalization** — BPCB excavation report coverage per km²
  (needed to properly normalize E004 site density by survey effort)
- TASK-014: **GPR/LiDAR partnership** â€” identify "blank" high-suitability zones as test sites
  for field campaign (confirm burial, not absence)
- TASK-016: **Literature review** â€” volcanic taphonomy precedents (Pompeii, Minoan Thera,
  Toba super-eruption, Mt Pinatubo burial studies)

---

## Completed Tasks

- TASK-001: Repo structure, requirements.txt, experiment dirs, paper dirs (2026-02-23)
- TASK-002: E001 scripts written + executed â†’ 666 sites, 297 geocoded (2026-02-23)
- TASK-003: E002 script written + executed â†’ 8 seed eruption records (2026-02-23)
- TASK-004: E003 scripts written + executed â†’ Malang + full Jawa Timur DEM (2026-02-23)
- TASK-005: E004 script written + executed â†’ rho=-0.991, INCONCLUSIVE (2026-02-23)
- TASK-006: E005 scripts written + executed â†’ rho=-0.364, H1 not supported (2026-02-23)
- TASK-007: Paper 1 reframing decision documented in JOURNAL.md + E005 README (2026-02-23)

---

## Experiment Queue

| ID | Name | Status | Paper | Notes |
|----|------|--------|-------|-------|
| E001 | Archaeological site geocoding | COMPLETE | P1 | 297 geocoded / 666 total |
| E002 | Eruption history compilation | COMPLETE | P1, P3 | 168 records from GVP (Kelud 37, Semeru 63, Bromo 67, Arjuno-Welirang 1) |
| E003 | DEM acquisition and processing | COMPLETE | P2, P3 | Malang + full Jawa Timur |
| E004 | Site density vs volcanic proximity | COMPLETE | P1 | rho=-0.991, survey bias dominates |
| E005 | Terrain suitability H1 test | COMPLETE | P1, P2 | rho=-0.364, H1 inconclusive |
| E006 | Re-run E004/E005 with enriched geocoding | COMPLETE | P1 | rho stable; n=383 |
| E007 | Settlement suitability model (baseline) | REVISIT | P2 | AUC=0.659; MVR not met; Challenge 1 PASSED |
| E008 | Settlement suitability model v2 (+ river dist) | REVISIT | P2 | AUC=0.695 (+0.036); MVR not met; Challenge 1 PASSED |
| E009 | Settlement suitability model v3 (+ soil data) | REVISIT | P2 | AUC=0.664; Challenge 1 PASSED; Path A complete |
| E010 | Settlement suitability model v4 (TGB pseudo-absences) | REVISIT | P2 | AUC=0.711 (+0.016 vs E008); Challenge 1 PASSED |
| E011 | Settlement suitability model v5 (TGB tuning) | REVISIT | P2 | AUC=0.725 (+0.014 vs E010); best config decay=16km, max=60km |
| E012 | Settlement suitability model v6 (TGB proxy enrichment) | REVISIT | P2 | AUC=0.730 (+0.005 vs E011); best config decay=12km, max=20km |
| E013 | Settlement suitability model v7 (hybrid bias correction) | SUCCESS | P2 | AUC=0.768 (>0.75); Challenge 1 PASSED; Paper 2 GO |

---

*Update this document whenever tasks change status. Keep it honest â€” if something is stuck, say so.*


