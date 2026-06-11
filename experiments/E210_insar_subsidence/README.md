# E210: InSAR Time-Series Subsidence Detection for Buried Archaeological Structures

**Date started:** 2026-04-22 (Session 21)
**Status:** SCOPING — Phase 0 (pre-pipeline design)
**Paper target:** Remote Sensing of Environment (Q1) or Journal of Archaeological Science (Q1)
**Layer:** L1 (volcanic burial, independent angle from E209)
**ME#16 rank:** Diamond-hunt #2 — highest novelty + AI-leverage after E209 flagship

---

## Hypothesis

Subsurface archaeological structures (buried walls, foundations, cavities, compacted fill) produce small but systematic differential compaction relative to surrounding undisturbed soil over multi-year timescales. This differential compaction is detectable as millimetre-scale vertical displacement patterns in Sentinel-1 Interferometric SAR (InSAR) time-series spanning 2014–present.

Specifically:

- **H1 (detectability):** Known discovered-buried sites (Sambisari, Kedulan, Kimpulan, Liangan) show InSAR subsidence signatures statistically distinguishable from surrounding control areas (per-pixel velocity variance, seasonal coherence, anomaly score).
- **H2 (landscape signal):** Volcanic-interior basins contain spatial clusters of anomalous subsidence patterns that (a) are not explained by known geological processes (active volcanism, groundwater extraction, landslide zones) and (b) correlate with settlement-suitability predictions (E013, E097).
- **H3 (complementarity with E209):** InSAR anomalies provide an independent signal from optical/SAR-backscatter features used in E209. Combined multi-modal classifier (E209 features + E210 InSAR anomaly score) achieves higher AUC than either alone.

---

## Why This Matters (per ME#16)

ME#16 identified **AI-unique work under-exploited** as a structural gap. InSAR archaeology has precedent (Tapete & Cigna 2019 global review; Chen et al. 2020 China; Stewart et al. 2018 Italy) but **zero published Indonesian applications**. This is a genuine global-first opportunity.

The method is uniquely AI-enabled because:
- Sentinel-1 has 6-day revisit since 2014 → ~720 acquisitions per area
- Processing requires complex interferogram stacking (SBAS/PS-InSAR) — computationally intensive but standard pipelines exist
- Indonesian volcanic soils are typically decoded as "high atmospheric noise, decorrelated" in InSAR literature; demonstrating useful archaeological signal requires targeted analysis
- Output = cm-precision vertical displacement maps, far more structurally informative than single-epoch optical

---

## Method

### Data

| Source | Type | Resolution | Access | Notes |
|---|---|---|---|---|
| Sentinel-1 SLC | SAR C-band (VV) | 5m × 20m (spatial), 6-day repeat | Free (ASF DAAC or Copernicus Open Hub) | Full archive 2014–present |
| Copernicus GLO-30 DEM | Elevation | 30m | Free | For topographic phase correction |
| SRTM 30m DEM | Elevation | 30m | Free (NASA) | Backup/comparison |
| Ground Deformation (from MintPy) | Derived time-series | 100m | Computed output | Main analysis product |

Acquisition stack target: 2018-01-01 to 2026-03-01 = ~8 years × 60 acquisitions/yr × ~20 overlapping slices = ~500 SLC scenes per basin.

### Processing pipeline

Two options:

**Option A: MintPy (local processing, RTX 4080)**
- InSAR pairs generated via ISCE2 or SNAP
- SBAS (Small Baseline Subset) stacking via MintPy
- Python-based, well-documented, open-source
- Compute: ~24–48 hours per basin on RTX 4080
- Storage: ~100 GB per basin intermediate products

**Option B: COMET-LiCSAR (free UK consortium processing)**
- Global Sentinel-1 time-series already processed
- LiCSBAS time-series analysis tool
- Output: pre-computed displacement rasters for many global areas
- Coverage of Indonesia: to verify
- If available, saves 24–48 hours compute per basin

**Option C: Google Earth Engine (free cloud compute)**
- Sentinel-1 ingested to GEE
- Hyp3 service provides InSAR products on demand (Alaska Satellite Facility)
- Custom python scripts for time-series aggregation

**Recommendation:** Start with Option B (LiCSAR) to check Indonesian coverage. Fall back to Option A if needed.

### Feature engineering (per pixel / per 100m tile)

- Mean velocity (mm/yr)
- Velocity standard deviation (temporal noise)
- Seasonal amplitude (wet-dry volumetric response)
- Anomaly score vs local neighbourhood (e.g., Z-score of velocity in a 5 km radius)
- Coherence statistics (decorrelation magnitude)
- Acceleration (2nd derivative of displacement)
- Correlation with groundwater-extraction / rainfall time series (rule out hydrology)

### Classifier

- Transfer learning from E209 architecture — same RF/GBM baseline, InSAR features added as extra columns
- Standalone InSAR classifier: same feature set sans E209 inputs
- Combined E209+E210: assess whether multi-modal improves AUC substantially

### Validation

- Known discovered-buried sites as hard positives (same as E209)
- Active volcanic zones (Merapi, Kelud, Semeru) as hard negatives (known subsidence from geology)
- Groundwater extraction zones (cities) as hard negatives (anthropogenic subsidence)

---

## Kill criteria

Diamond-hunt killed if:

- InSAR processing for Java volcanic basins fails systematically due to vegetation decorrelation (C-band coherence < 0.3 across >80% of area)
- Known discovered-buried sites show no distinguishable InSAR signature at >1 cm/yr precision
- LiCSAR products unavailable AND local Option A processing fails on RTX 4080

Any of above → document as informative negative ("InSAR archaeology limits in volcanic tropical Indonesia"). Proceed to E211 / E208P3.

---

## Timeline

- **Phase 0 (scoping, this session):** README + data source verification → **COMPLETE (this document)**
- **Phase 1 (~2 weeks):** LiCSAR coverage check + fallback plan; if LiCSAR usable, begin baseline extraction at known buried sites
- **Phase 2 (~2–4 weeks):** Feature engineering + baseline classifier; compare with E209
- **Phase 3 (~2 weeks):** Landscape application + top-K candidate extraction
- **Phase 4 (~2 weeks):** Paper drafting

Gated on E209 outcome: if E209 AUC ≥ 0.75, E210 begins as independent confirmatory line. If E209 AUC < 0.65, E210 becomes primary diamond-hunt.

---

## Relation to other experiments

- **E209** (satellite ML classifier): parallel diamond-hunt; features combine
- **E013** (settlement model): geographic prior, classifier input
- **E097** (anomaly detection): prediction target validation
- **E116** (GPR prediction): E210 top-K coords could guide future GPR fieldwork
- **P23** (satellite archaeology paper): E210 could extend OR replace P23 scope

---

## Budget

- Compute: 0 ($) with LiCSAR; ~48 hr RTX 4080 if local Option A
- Data: 0 ($) all free satellite
- Storage: ~100 GB per basin intermediate (manageable)
- Time: ~8–12 weeks to publication

No budget request needed.

---

## References

- Tapete, D. & Cigna, F. (2019). "Detection of archaeological looting from space: Methods, achievements and challenges." *Remote Sensing* 11(20).
- Chen, F. et al. (2020). "InSAR time-series subsidence monitoring in East China lowlands." *Remote Sensing of Environment*.
- LiCSAR: Lazecký et al. (2020). "LiCSAR: An automatic InSAR tool for measuring and monitoring tectonic and volcanic activity." *Remote Sensing*.
- MintPy: Yunjun et al. (2019). "Small baseline InSAR time series analysis: Unwrapping error correction and noise reduction." *Computers & Geosciences*.

---

*E210 scoped 2026-04-22 per ME#16 §4 Candidate 2. Will execute after E209 Phase 1 evaluation.*
