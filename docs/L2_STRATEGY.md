# L2: STRATEGY (Active Phase)

**Status:** ACTIVE — Updates per phase/quarter.
**Current Phase:** Phase 1 — Foundation (Computational + Interdisciplinary)
**Last updated:** 2026-03-11

---

## 1. Current Phase: Phase 1 (Computational + Exploratory Literature)

**Duration:** ~6–12 months (Q1 2026 – Q4 2026)
**Resources:** 4× RTX 4080, Claude Code, public data, literature
**Dependencies:** None (P1+P2 submitted 2026-03-10/11)
**Goal:** Establish the VOLCARCH framework through computational papers (P1, P2) and exploratory interdisciplinary contributions (P5, P7, and others as opportunities arise)

**Research philosophy:** Exploratory rapid prototyping — multiple experiments run in parallel, fail fast, keep what works. All papers serve one manifesto: the four layers of invisibility (volcanic burial, coastal submersion, historiographic bias, cosmological displacement). See `docs/drafts/manifesto.md`.

**Preprint strategy:** P1 and P2 will be posted to **EarthArXiv** as preprints (free DOI) before or concurrent with journal submission, to establish priority and provide citable references for P5/P7/future papers.

## 2. Active Papers

### Paper 1: Taphonomic Bias Framework
**Status:** SUBMITTED to Asian Perspectives (2026-03-10) — MS# 019A-0326, Submission #773
**Type:** Literature review + quantitative analysis + position paper
**Target journal:** Asian Perspectives (Q1, $0, Univ Hawaii Press)
**File:** `papers/P1_taphonomic_framework/submission_asianperspectives_v0.1.docx`
**Single-author:** Mukhlis Amien
**MVR:** Multi-site calibration of sedimentation rates (4.4 ± 1.2 mm/yr across 2 volcanic systems) — **MET**
**Key tasks:**
- [x] Scrape and geocode all known archaeological sites in East Java (666 sites, 383 geocoded)
- [x] Compile volcanic eruption histories for Kelud, Semeru, Arjuno-Welirang (37+ eruptions)
- [x] Compute site density vs volcanic proximity statistics (rho=-0.955)
- [x] Write Dwarapala case study with multi-site calibration
- [x] Reformat for Asian Perspectives (.docx, 6 JPG figures)
- [x] Submit (2026-03-10)
- [x] ORCID: `0000-0002-1848-167X`
- [ ] Post preprint to EarthArXiv

### Paper 2: Settlement Suitability Model
**Status:** SUBMITTED to JCAA (2026-03-11) — Submission #280
**Type:** ML/GIS computational paper
**Target journal:** JCAA (Diamond OA, $0, Ubiquity Press, Scopus-indexed)
**File:** `papers/P2_settlement_model/submission_jcaa_v0.1.tex` (24 pages)
**Compile:** `pdflatex → bibtex → pdflatex → pdflatex`
**Authors:** Mukhlis Amien (corresponding), Go Frendi Gunawan (both Universitas Bhinneka Nusantara)
**MVR:** AUC > 0.75 on spatial cross-validation — **MET** (AUC=0.768, seed-avg 0.751)
**Key tasks:**
- [x] Model development (E007-E013, AUC progression 0.659→0.768)
- [x] Tautology test suite (CONDITIONAL PASS)
- [x] Temporal validation E014 (AUC=0.755)
- [x] SHAP analysis E015 (rho=0.943)
- [x] Reference audit (40 references)
- [x] Adapt to JCAA format (natbib/plainnat, Harvard citations, double-spaced)
- [x] Write cover letter (AI disclosure included)
- [x] Submit (2026-03-11)
- [ ] Post preprint to EarthArXiv

### Paper 3: Volcanic Burial Depth Model
**Status:** KILLED (2026-03-10, Mata Elang review #2)
**Type:** Geospatial modeling
**Target journal:** Was J. Volcanology & Geothermal Research (Q1)
**MVR:** Model predicts Dwarapala burial depth within ±30% (130–240 cm vs actual ~185 cm) — **NOT MET**
**Kill reason:** E017 Tephra POC FAILED (1/4 calibration sites pass). Pyle 1989 generic model insufficient; requires per-volcano calibration with Tephra2/FALL3D and geologist co-author. Resurface only if geologist collaborator joins.
**Data preserved:** E002 eruption records (168), E017 results documented.

## 3. Data Strategy

**Priority data to acquire first (for Paper 1):**

| Data | Source | Status | Experiment |
|------|--------|--------|------------|
| Archaeological site locations (East Java) | BPCB publications, literature, OSM | COMPLETE (666 sites, 383 geocoded) | E001 |
| Kelud eruption history + ashfall data | GVP Smithsonian | COMPLETE (37 eruptions) | E002 |
| Semeru eruption history | GVP Smithsonian | COMPLETE (63 eruptions) | E002 |
| DEM Malang Raya + Jawa Timur | Copernicus GLO-30 | COMPLETE | E003 |

## 4. Collaboration Strategy

**Phase 1 (now):** Work solo or with students. No external dependencies.
**Phase 1 (late):** Approach potential collaborators informally:
- Geologist: Universitas Brawijaya (vulkanologi)
- Archaeologist: Balai Arkeologi Yogyakarta or BPCB Jawa Timur
- Present Paper 1 draft as conversation starter

## 5. Computational Strategy

| Tool | Purpose | Phase |
|------|---------|-------|
| Python + geopandas + rasterio | GIS data processing | 1 |
| scikit-learn / xgboost | Settlement ML models | 1 |
| elapid (or MaxEnt Java) | Presence-only modeling | 1 |
| Tephra2 or FALL3D | Tephra dispersal simulation | 1 |
| Folium / Kepler.gl | Interactive map visualization | 1 |
| Claude Code | Scripting, scraping, analysis automation | 1 |
| RTX 4080 × 4 | DL models, large raster processing | 1–2 |

## 6. Phase Transition Criteria

**Move to Phase 2 when ALL of:**
- [x] Paper 1 submitted (or accepted) — **SUBMITTED** to Asian Perspectives (2026-03-10)
- [x] Paper 2 model achieves MVR (AUC > 0.75) — **MET** (AUC=0.768)
- ~~Paper 3 model achieves MVR~~ — **KILLED** (E017 POC failed; removed from gate criteria)
- [x] Challenge 1 (tautology test) passed — **CONDITIONAL PASS** (T3-T4 robust; definitive proof requires fieldwork/GPR)
- [ ] At least one domain collaborator engaged
- [ ] BIMA or equivalent funding proposal drafted

## 7. Formal Challenges (Must-Pass Before Phase 2)

### Challenge 1: Tautology Elimination
**Question:** Does the settlement model learn *suitability* or *visibility*?
**Design:** Train model on environmental features ONLY. Test if it predicts high suitability in high-burial zones.
**Pass:** Spatial AUC > 0.70 AND predicts some high-suitability in high-burial areas.
**Fail:** Model only predicts where sites are already found → redesign required.
**See:** `docs/EVAL.md` section 3.

### Challenge 2: Multi-Source Calibration
**Status:** KILLED (2026-03-10) — E017 POC failed. Challenge cannot be met with current tools/data.
**Question:** Can burial depth be predicted from eruption data across multiple volcanic systems?
**Result:** 1/4 calibration sites passed. Generic Pyle 1989 model insufficient.
**Resurrection condition:** Geologist collaborator + Tephra2/FALL3D access.
**See:** `docs/EVAL.md` section 2, `experiments/E017_tephra_poc/`.

**Phase 2 will require:**
- Funding for GPR equipment rental or partnership
- Archaeologist co-PI for fieldwork permits
- Geologist co-PI for stratigraphic interpretation
- Ethics clearance if working near cultural/spiritual sites

## 8. Papers Pipeline (Updated 2026-03-10)

### Submitted (under review)
- **P1** — Taphonomic Bias Framework → Asian Perspectives (2026-03-10)
- **P2** — Settlement Suitability Model → JCAA (2026-03-11)
- **P5** — The Volcanic Ritual Clock → BKI (2026-03-09)
- **P7** — Temporal Overlay Matrix → Antiquity Project Gallery (2026-03-06)

### Active exploration
- **P8** — Linguistic Fossils (draft v0.1 complete, 5 experiments, needs red-team before submission)
- **P9/P09alt** — Borehole Archaeology / Peripheral Substrate (E024 POC SUCCESS)

### Incubation (see `docs/drafts/README.md`)
- **P11** — Volcanic Cultural Selection (partial draft, needs P5+P9 as foundation)

### Dissolved
- **P15** — TOM-R content absorbed into `papers/P5_volcanic_ritual_clock/revision_ammo/` (2026-03-10)

### Killed
- **P3** — Volcanic Burial Depth Model (E017 POC FAILED, 2026-03-10)
- **P4** — Estuarine Hybrids (stub, no data, no path to execution)
- **P6** — Linguistic Phylogenetics (depends on P8 + linguist, too speculative for 2026)
- **P10** — Archaeological Biosignatures (requires fieldwork, no partner)
- **P12** — Computational Mythology (requires corpus construction, no corpus)
- **P14** — Pararaton Volcanic Collapse (Bonferroni kills significance; E026 folded into P5 revision ammo, 2026-03-11)
- **P-coastal** — The Invisible Shore (stub, no data, no method)

**Pipeline summary (post Mata Elang #4):** 4 under review (P1, P2, P5, P7) + 2 drafting (P8, P9) + 1 incubating (P11) = **7 active papers**.

See `docs/drafts/README.md` for full catalog.

---

*This document reflects the current plan. It will be updated when phases change or pivots occur.*
*Last updated: 2026-03-11 (post Mata Elang #4 — P1+P2 submitted)*
