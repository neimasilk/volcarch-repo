# L2: STRATEGY (Active Phase)

**Status:** ACTIVE — Updates per phase/quarter.
**Current Phase:** Phase 1 — Computational Foundation
**Last updated:** 2026-02-23

---

## 1. Current Phase: Phase 1 (Computational & Literature Only)

**Duration:** ~6–12 months (Q1 2026 – Q4 2026)
**Resources:** 4× RTX 4080, Claude Code, public data, literature
**Dependencies:** None (no funding, no collaborators required)
**Goal:** Produce 2–3 papers establishing the computational framework

## 2. Active Papers

### Paper 1: Taphonomic Bias Framework
**Status:** DRAFT COMPLETE (submission package ready 2026-02-25)
**Type:** Literature review + quantitative analysis + position paper
**Target journal:** Journal of Archaeological Science: Reports (Q1)
**MVR:** Show statistically significant negative correlation between known site density and volcanic sediment deposition potential in East Java
**Key tasks:**
- [ ] Scrape and geocode all known archaeological sites in East Java
- [ ] Compile volcanic eruption histories for Kelud, Semeru, Arjuno-Welirang
- [ ] Compute site density vs volcanic proximity statistics
- [ ] Write Dwarapala case study with full calculation
- [ ] Comparative analysis: Kalimantan vs Java preservation conditions
- [ ] Draft and circulate for feedback

### Paper 2: Settlement Suitability Model
**Status:** DRAFT COMPLETE (MVR met at E013, submission package in progress)
**Type:** ML/GIS computational paper
**Target journal:** Remote Sensing (MDPI, Q1)
**MVR:** AUC > 0.75 on spatial cross-validation
**Methodology:**
- Features: DEM-derived (slope, aspect, TWI, TRI), hydrological (river distance, confluence distance), soil, land cover
- Labels: known sites (positive), pseudo-absence (negative)
- Algorithms: MaxEnt (primary — suited for presence-only), Random Forest, XGBoost
- Validation: spatial k-fold cross-validation, AUC-ROC, TSS
- Study area: Malang Raya (Kota Malang + Kabupaten Malang)

### Paper 3: Volcanic Burial Depth Model
**Status:** NOT STARTED (can run parallel with Paper 2)
**Type:** Geospatial modeling
**Target journal:** J. Volcanology & Geothermal Research (Q1)
**MVR:** Model predicts Dwarapala burial depth within ±30% (130–240 cm vs actual ~185 cm)
**Methodology:**
- Empirical approach: accumulate tephra from known eruptions using dispersal models + wind data
- Calibration: Dwarapala ground truth
- Data: GVP eruption records, ERA5 wind patterns, isopach maps from literature
**Note:** Seek geologist co-author for domain credibility

## 3. Data Strategy

**Priority data to acquire first (for Paper 1):**

| Data | Source | Status | Experiment |
|------|--------|--------|------------|
| Archaeological site locations (East Java) | BPCB publications, literature, OSM | NOT STARTED | E001 |
| Kelud eruption history + ashfall data | GVP Smithsonian | NOT STARTED | E002 |
| Semeru eruption history | GVP Smithsonian | NOT STARTED | E002 |
| DEM Malang Raya | DEMNAS (BIG) or SRTM | NOT STARTED | E003 |

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
- [ ] Paper 1 submitted (or accepted)
- [ ] Paper 2 model achieves MVR (AUC > 0.75)
- [ ] Paper 3 model achieves MVR (Dwarapala prediction ±30%)
- [ ] Challenge 1 (tautology test) passed
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
**Question:** Can burial depth be predicted from eruption data across multiple volcanic systems?
**Design:** Accumulate tephra from known eruptions using VEI + distance + wind. Calibrate to all anchors (Dwarapala, Sambisari, Kedulan, Kimpulan).
**Pass:** Predictions within ±30% for at least 3 of 4 calibration points.
**Fail:** Cannot predict Dwarapala within ±50% → model fundamentally wrong.
**See:** `docs/EVAL.md` section 2.

**Phase 2 will require:**
- Funding for GPR equipment rental or partnership
- Archaeologist co-PI for fieldwork permits
- Geologist co-PI for stratigraphic interpretation
- Ethics clearance if working near cultural/spiritual sites

---

*This document reflects the current plan. It will be updated when phases change or pivots occur.*
