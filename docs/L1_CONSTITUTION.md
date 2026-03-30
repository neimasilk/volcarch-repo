# L1: CONSTITUTION (UUD)

**Status:** STABLE — This document changes only if core assumptions are proven wrong.
**Last updated:** 2026-03-20 (kill criteria updated — originals were obsolete after E005 pivot)

---

## 1. Mission

Investigate how volcanic processes have shaped — and hidden — the archaeological, linguistic, and cultural record of pre-modern Nusantara. Using computational methods (ML, GIS, NLP, corpus analysis), we aim to:

1. **Predict** where ancient settlements in volcanic Java are buried underground (Papers 1–2);
2. **Detect** pre-Austronesian and pre-Indic substrates preserved in language and ritual (Papers 5, 8);
3. **Test** whether volcanic events drove political and cultural transitions (Papers 7, 14).

The core thesis: **the perceived absence of pre-4th century civilisation in western Indonesia reflects taphonomic loss compounded by insufficient survey intensity, not genuine cultural absence.** Volcanic burial, coastal submersion, and historiographic bias have rendered an entire civilisational stratum invisible — and Indonesia's archaeological infrastructure has been insufficient to recover what volcanism has hidden.

## 2. Core Hypotheses

**H1 — Taphonomic Bias:** The distribution of known archaeological sites in Java is systematically biased by volcanic sedimentation compounded by insufficient survey intensity. Sites in high-deposition zones are underrepresented because volcanic burial renders them invisible to surface survey, and Indonesia's archaeological infrastructure has been insufficient to recover them through subsurface methods.

> **Status note (2026-03):** E004 and E005 could not confirm H1 from observable site-distribution data alone — the signal is dominated by survey-effort bias (see JOURNAL 2026-02-23). The project pivoted to treating H1 as a *motivating hypothesis* that requires fieldwork (GPR) to test directly, while the computational contribution became a *methodological framework* for tautology-free settlement modeling (Papers 1–2).
>
> **ADV-2 (E081, 2026-03-13):** Cave/open-air site ratio is IDENTICAL in volcanic and non-volcanic regions (Fisher p=0.760). Cave bias is universal where karst exists. The L1 argument must be built on **burial depth data** (E083: 24 sites, mean 3.41m), not site-type distribution.
>
> **ADV-3 (E069, 2026-03-13):** After controlling for survey intensity proxies (road distance, BPCB proximity), volcanic proximity retains a significant independent effect on site density (quasi-Poisson beta=-0.477, **p=0.0015**). Volcanic signal survives survey-effort control.
>
> **ADV-1 Japan Constraint (E086, 2026-03-16):** Japan — equally volcanic — has a rich 38,000-year record because it invests 100-200× more in archaeological survey (8,300 excavations/year, 460,000 registered sites, mandatory rescue archaeology since 1950). Volcanic burial is a *necessary but not sufficient* condition for archaeological invisibility. The sufficient condition is volcanic burial combined with inadequate survey intensity. Java's tropical lahar regime creates 32× deeper sustained burial than Japan's temperate system, but the primary difference is institutional capacity, not geology alone.

**H2 — Discovery Bias (Motivating Observation):** The perceived "oldest kingdom" status of Kutai (Kalimantan, ~400 CE) may reflect differential preservation (zero volcanism) rather than genuine chronological primacy over Javanese polities. *(Note: This is a motivating observation that contextualizes the research, not a testable hypothesis within the current computational scope.)*

**H3 — Predictability:** Locations of undiscovered buried archaeological sites in volcanic Java can be predicted with useful accuracy by combining settlement suitability modeling with volcanic sediment accumulation estimation.

## 3. Research Philosophy

- **Simple is better.** Prefer the simplest model that works. Add complexity only when simplicity demonstrably fails.
- **Fail fast, pivot early.** Define Minimum Viable Results (MVR) before each experiment. If MVR is not met, pivot or abandon — don't sink more time.
- **Santai dalam waktu, serius dalam standar ilmiah.** No rushed deadlines. But every claim must be defensible, every number traceable, every method reproducible.
- **Interdisciplinary with computational core.** We are data scientists applying computational methods (ML, GIS, NLP, corpus analysis) to archaeological, linguistic, and cultural questions. We also pursue exploratory literature-based research when it yields novel, testable findings. We do not replace domain experts — we offer new tools and perspectives.
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

## 5. Evidential Structure (Consilience)

Our argument follows the principle of **consilience of inductions** (Whewell 1840): multiple independent lines of evidence converge on the same conclusion through different methods and datasets.

**Analytical lenses:** 4 — geological/taphonomic, linguistic, epigraphic, spatial-architectural.

**Core datasets (~5):** DHARMA inscriptions (268), East Java archaeological sites (666), ABVD wordlists, candi locations (142), OV colonial register.

**Genuinely independent datasets (3):**
1. **E083** — Colonial-era burial depth measurements (51 eruption-site pairs, 24 measured depths). Source: OV field reports + published volcanological literature. Zero overlap with statistical pipeline.
2. **E088/E089** — Textual archaeology corpus (106 passages from 12 ancient traditions). Source: translated primary texts (Greek, Chinese, Indian, Arab, etc.). Zero overlap with DHARMA/ABVD.
3. **E091** — OV NLP mining (22,162 structured mentions from 16 OV volumes). Source: automated extraction from colonial Dutch text. 94.2% cross-validation against manual dataset.

**Dataset honesty (2026-03-20):** 21 of 122 experiments depend on the same 268 DHARMA inscriptions. The three genuinely independent datasets above mitigate this concentration. Claims of "independent channels" refer to analytical methods, not necessarily independent data sources, except where explicitly noted.

## 6. Known Methodological Risks (Baked Into Design)

### The Tautology Trap
Our settlement model (Paper 2) trains on *discovered* sites. But discovered sites are biased toward low-burial-depth locations — which is literally our hypothesis (H1). There is a risk the model learns "visibility to modern archaeologists" rather than "suitability for ancient settlement."

**Mitigation (mandatory):** The settlement model must be trained ONLY on environmental features (slope, river distance, soil, etc.) — never on burial-depth or volcanic-proximity features. Then we separately test whether the model's predictions correlate with volcanic burial zones. If the model predicts sites in high-burial zones that have no known discoveries — that's the finding. If it only predicts sites where things were already found — it has learned the tautology and is useless.

### Single-Point Extrapolation
The Dwarapala rate of 3.6 mm/year is one point. Extrapolating spatially from one point is dangerous. This is why secondary anchors (Sambisari, Kedulan, Kimpulan) are critical — they provide independent calibration from a different volcanic system (Merapi vs Kelud).

## 7. Scope Boundaries

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

## 8. Ethical Boundaries

- **Site protection.** Precise predicted coordinates of potential sites are shared ONLY with authorized bodies (BPCB, Balai Arkeologi). Public outputs use coarse resolution (minimum 500m grid). No raw GPS coordinates in public papers.
- **Cultural sensitivity.** Archaeological sites may have spiritual significance. Engage local communities before any fieldwork.
- **No sensationalism.** We do not claim to have "found lost civilizations." We build tools to help look.
- **Open science.** Code and non-sensitive data are open source. Papers target open-access journals where feasible.

## 9. Kill Criteria

> **Update 2026-03-20:** Original kill criteria (H1 correlation, AUC threshold) were tested in E005 and E007-E013. E005 failed to show direct H1 correlation from observable data; the project pivoted rather than killed (correctly — the signal was masked by survey bias, confirmed by E069 p=0.0015). AUC threshold was met (0.768). The criteria below replace the originals with conditions that reflect the project's current state.

### Abandon the research line if:
- **Cascade falsification:** The E110 5-factor model is shown to be off by >2 orders of magnitude from observed gap, AND no reasonable parameter adjustment restores agreement (E115 currently shows 92% of runs within 10×)
- **Within-island control fails:** New evidence shows pre-400 CE sites ARE present in volcanic East Java at comparable density to non-volcanic West Java (Buni/Batujaya), eliminating the taphonomic signal
- **External comparandum falsification:** A volcanic tropical island (e.g., Philippines' volcanic zones) shows rich pre-400 CE archaeology despite comparable sedimentation rates, proving volcanism alone does not explain absence
- **3 or more peer reviews** (not desk rejects) provide substantive methodological critiques that cannot be addressed — i.e., the *method* is judged flawed, not just the *framing* or *journal fit*
- **Domain expert consensus** (≥2 independent archaeologists or geologists with Java expertise) judges the framework fundamentally unsound

### Pivot the research line if:
- **All computational journals reject P1:** Reframe from "taphonomic bias framework" to "survey priority model" — keep E110 cascade + E080/E097 fieldwork targeting, drop the civilizational claims
- **P2 settlement model fails peer review on tautology grounds:** Redesign with presence-only methods (MaxEnt) or abandon ML approach for rule-based expert system
- **No acceptance after 6 submissions across different journals:** The work may not be publishable as currently structured — consider consolidating into 1-2 comprehensive papers instead of 6+
- **Fieldwork partner found:** Pivot from computational-only to computational+empirical — this is the DESIRED pivot that would resolve the zero-ground-truth problem

---

*This document is the foundation. Everything else can change; this should not — unless reality demands it.*
