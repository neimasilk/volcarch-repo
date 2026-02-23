# L1: CONSTITUTION (UUD)

**Status:** STABLE — This document changes only if core assumptions are proven wrong.
**Last updated:** 2026-02-23

---

## 1. Mission

Build a computational framework that predicts where ancient settlements in volcanic Java are buried underground, and at what depth — enabling archaeologists to find what volcanic activity has hidden.

## 2. Core Hypotheses

**H1 — Taphonomic Bias:** The distribution of known archaeological sites in Java is systematically biased by volcanic sedimentation. Sites in high-deposition zones are underrepresented.

**H2 — Discovery Bias:** The perceived "oldest kingdom" status of Kutai (Kalimantan, ~400 CE) may reflect differential preservation (zero volcanism) rather than genuine chronological primacy over Javanese polities.

**H3 — Predictability:** Locations of undiscovered buried archaeological sites in volcanic Java can be predicted with useful accuracy by combining settlement suitability modeling with volcanic sediment accumulation estimation.

## 3. Research Philosophy

- **Simple is better.** Prefer the simplest model that works. Add complexity only when simplicity demonstrably fails.
- **Fail fast, pivot early.** Define Minimum Viable Results (MVR) before each experiment. If MVR is not met, pivot or abandon — don't sink more time.
- **Santai dalam waktu, serius dalam standar ilmiah.** No rushed deadlines. But every claim must be defensible, every number traceable, every method reproducible.
- **Computational contribution.** We are informaticians. We build tools and models. We do not replace archaeologists or geologists — we empower them.
- **Honest reporting.** Negative results and failed experiments are documented with equal rigor. Publication bias starts with us choosing not to practice it.

## 4. Empirical Anchor: The Dwarapala Case

Our entire framework is calibrated against a single, verifiable data point:

The Dwarapala statues of Singosari (built ~1268 CE, discovered 1803 CE with half their 370 cm height buried) yield a measured sedimentation rate of approximately 3.6 mm/year at that specific location in the Malang basin. This is consistent with known volcanic ash deposition from Gunung Kelud (~20 eruptions in that period, each depositing 2–20 cm at Malang distance).

**This is not a universal rate.** It is a calibration point. Spatial variation is the subject of the research itself.

## 5. Scope Boundaries

### In Scope
- Computational modeling (ML, GIS, remote sensing analysis)
- Literature-based data collection and synthesis
- Open-data analysis (DEM, satellite imagery, eruption records)
- Producing probability maps and predictions
- Proposing fieldwork targets for domain experts

### Out of Scope
- Conducting archaeological excavations ourselves
- Making definitive archaeological claims without domain expert validation
- Building commercial products
- Fieldwork without proper permits and institutional partnerships

## 6. Ethical Boundaries

- **Site protection.** Precise predicted coordinates of potential sites are shared ONLY with authorized bodies (BPCB, Balai Arkeologi). Public outputs use coarse resolution.
- **Cultural sensitivity.** Archaeological sites may have spiritual significance. Engage local communities before any fieldwork.
- **No sensationalism.** We do not claim to have "found lost civilizations." We build tools to help look.
- **Open science.** Code and non-sensitive data are open source. Papers target open-access journals where feasible.

## 7. Kill Criteria

This research line should be **abandoned** if:
- Paper 1 analysis shows NO correlation between known site density and volcanic deposition (H1 falsified)
- Settlement suitability models consistently perform below AUC 0.65 despite feature engineering (H3 falsified)
- Domain experts (archaeologists, geologists) judge the framework fundamentally flawed after review

This research line should be **pivoted** if:
- Volcanic burial depth proves unmodelable from available data → pivot to purely settlement suitability modeling
- Training data (known sites) is too sparse for ML → pivot to rule-based/expert-system approach
- A better empirical anchor than Dwarapala is found → recalibrate

---

*This document is the foundation. Everything else can change; this should not — unless reality demands it.*
