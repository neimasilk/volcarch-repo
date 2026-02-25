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

## 4. Empirical Anchors

Our framework is calibrated against multiple verifiable data points across Java's volcanic landscape. These are not universal rates — they are calibration points. Spatial variation is the subject of the research itself.

### Primary Anchor: Dwarapala Singosari (Malang Basin / Kelud system)
The Dwarapala statues of Singosari (built ~1268 CE, discovered 1803 CE with half their 370 cm height buried) yield a measured sedimentation rate of approximately **3.6 mm/year**. Consistent with Gunung Kelud's ~20 eruptions in that period.

### Secondary Anchors (Central Java / Merapi system — to be compiled)

| Site | Built (CE) | Found | Depth (cm) | System | Rate (mm/yr) | Source |
|------|-----------|-------|-----------|--------|-------------|--------|
| Dwarapala Singosari | ~1268 | 1803 | ~185 | Kelud (E. Java) | **3.5** | BPCB Jawa Timur |
| Candi Sambisari | ~835 | 1966 | 500–650 | Merapi (C. Java) | **4.4–5.7** | Wanua Tengah III inscription; BPCB DIY |
| Candi Kedulan | ~869 | 1993 | 600–700 | Merapi (C. Java) | **5.3–6.2** | Sumundul inscription (791 Saka); BPCB DIY |
| Candi Kimpulan (UII) | ~900 | 2009 | 270–500 | Merapi (C. Java) | **2.4–4.5** | Putra & Setyastuti (BEFEO 105); UII |
| Candi Liangan | ~9th c. | 2008 | 500–900 | Sundoro (C. Java) | N/A (catastrophic) | Abbas (2016); C14: 590 CE |

**Summary:** Four independent calibration points from two volcanic systems yield sedimentation rates of **2.4–6.2 mm/yr** (mean 4.4 ± 1.2 mm/yr). Merapi-system sites show higher rates (~4.8 mm/yr mean) than the Kelud-system Dwarapala (3.5 mm/yr), consistent with Merapi's higher eruption frequency. Liangan is excluded from rate calculation (single catastrophic burial event) but confirms that 5–9 m burial depths occur in Central Java.

**Critical note:** Sambisari, Kedulan, and Kimpulan are Merapi-system sites (Central Java), while Dwarapala is a Kelud-system site (East Java). Having calibration points from *different volcanic systems* proves the burial phenomenon is Java-wide, not volcano-specific. The rate consistency across systems (same order of magnitude) is itself a key finding.

## 5. Known Methodological Risks (Baked Into Design)

### The Tautology Trap
Our settlement model (Paper 2) trains on *discovered* sites. But discovered sites are biased toward low-burial-depth locations — which is literally our hypothesis (H1). There is a risk the model learns "visibility to modern archaeologists" rather than "suitability for ancient settlement."

**Mitigation (mandatory):** The settlement model must be trained ONLY on environmental features (slope, river distance, soil, etc.) — never on burial-depth or volcanic-proximity features. Then we separately test whether the model's predictions correlate with volcanic burial zones. If the model predicts sites in high-burial zones that have no known discoveries — that's the finding. If it only predicts sites where things were already found — it has learned the tautology and is useless.

### Single-Point Extrapolation
The Dwarapala rate of 3.6 mm/year is one point. Extrapolating spatially from one point is dangerous. This is why secondary anchors (Sambisari, Kedulan, Kimpulan) are critical — they provide independent calibration from a different volcanic system (Merapi vs Kelud).

## 6. Scope Boundaries

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

## 7. Ethical Boundaries

- **Site protection.** Precise predicted coordinates of potential sites are shared ONLY with authorized bodies (BPCB, Balai Arkeologi). Public outputs use coarse resolution (minimum 500m grid). No raw GPS coordinates in public papers.
- **Cultural sensitivity.** Archaeological sites may have spiritual significance. Engage local communities before any fieldwork.
- **No sensationalism.** We do not claim to have "found lost civilizations." We build tools to help look.
- **Open science.** Code and non-sensitive data are open source. Papers target open-access journals where feasible.

## 8. Kill Criteria

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
