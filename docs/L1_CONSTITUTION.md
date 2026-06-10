# L1: CONSTITUTION (UUD)

**Status:** STABLE — This document changes only if core assumptions are proven wrong.
**Last updated:** 2026-06-08 (§1 core thesis amended → *peradaban vulkanik* character framing, per PI + E214/E215; see `docs/research_notes/L1_AMENDMENT_PROPOSAL_2026_06_08.md`). Prior: 2026-03-20 (falsification criteria).

---

## 1. Mission

Investigate how volcanic processes have shaped — and hidden — the archaeological, linguistic, and cultural record of pre-modern Nusantara. Using computational methods (ML, GIS, NLP, corpus analysis), we aim to:

1. **Predict** where ancient settlements in volcanic Java are buried underground (Papers 1–2);
2. **Detect** pre-Austronesian and pre-Indic substrates preserved in language and ritual (Papers 5, 8);
3. **Test** whether volcanic events drove political and cultural transitions (Papers 7, 14).

The core thesis (amended 2026-06-08): **pre-Indic Nusantara developed a distinct *volcanic civilisation* (*peradaban vulkanik*) — a society organised around and adapted to volcanic landscapes (fertile andisols, eruption cycles, mountain cosmology, dispersed upland settlement), structurally unlike the great river-valley civilisations of Egypt, Mesopotamia, Persia, and China.** Its apparent "late start" — the first Indian-script inscriptions of ~400 CE — marks not its inception but the moment it became *textually visible*. The civilisation itself was rendered hard to see by three compounding invisibilities: **archaeological** (dispersed settlement, largely perishable material culture, and taphonomic burial under volcanic sediment — the original H1, compounded by insufficient survey intensity); **palaeoecological** (volcanic-adapted swidden/arboriculture rather than landscape-scale forest clearance, leaving a weak pollen signature); and **historiographic** (oral/perishable transmission until Indian script was adopted). This is a claim about the *character and visibility* of a civilisation, not merely the burial of a stratum. It is falsifiable — it predicts distinctive volcanic-adapted material culture, cosmology, and cultivation with pre-Indic roots — and contrary evidence is reported honestly (e.g., the palynological channel does not currently support a *large* forest-clearing pre-400 CE population; see JOURNAL 2026-06-08 / E214). H1–H3 below are the supporting mechanisms of the archaeological-invisibility leg, not the headline.

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

**Dataset honesty (2026-04-01):** ~25 of 175 experiments depend on the same 268 DHARMA inscriptions. The three genuinely independent datasets above mitigate this concentration. Claims of "independent channels" refer to analytical methods, not necessarily independent data sources, except where explicitly noted.

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
- Proposing fieldwork candidates for domain experts

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

## 9. Falsification Criteria

> **Update 2026-03-20:** Original falsification criteria (H1 correlation, AUC threshold) were tested in E005 and E007-E013. E005 failed to show direct H1 correlation from observable data; the project pivoted rather than discontinued (correctly — the signal was masked by survey bias, confirmed by E069 p=0.0015). AUC threshold was met (0.768). The criteria below replace the originals with conditions that reflect the project's current state.

> **Update 2026-04-21 (Session 19):** Criteria #1 (cascade) and #3 (comparandum) were audited in `docs/research_notes/STOP_CRITERION_AUDIT_2026_04_21.md` and found to be either incoherent with current state (#1 after E176 over-parameterization) or non-operational (#3 without measured rate comparison). Session 19 also executed first cross-model critical review (DeepSeek) which surfaced methodological concerns not caught by self-review. The updated criteria below address both issues and add a new cross-model trigger. These updates were explicitly approved by Pak Amien ("saya percaya kamu").

### Abandon the research line if:
- **Cascade underdetermination (UPDATED 2026-04-21):** The 5-factor cascade has been demonstrated by E176 to be over-parameterized — 83.8% of random 5-factor draws bracket observation, and 3-factor minimal models achieve the same fit (AIC 6.73 vs 5-factor AIC 6.25). The cascade is therefore **retained only as pedagogical illustration of compound taphonomy, not as validation of the specific multi-factor framework.** Papers that argue from specific numerical cascade match (e.g., legacy P1 §5.5) must reframe to argue from the six-filter structure instead. The cascade-as-explanatory-model criterion is considered **PARTIALLY TRIGGERED as of 2026-04-21.** This does not discontinue the research line — it retires one of its rhetorical supports.
- **Within-island control fails:** New evidence shows pre-400 CE sites ARE present in volcanic East Java at comparable density to non-volcanic West Java (Buni/Batujaya), eliminating the taphonomic signal. *Current state (2026-04-21): HOLDS. Reinforced by Session 19 Batujaya documentation.*
- **External comparandum falsification (REFINED 2026-04-21):** Discovery of pre-400 CE open-air archaeological sites in a volcanic tropical setting with both: (a) measured total tephra deposition rate within 2× of Java's 4.4 mm/yr, **and** (b) karst cover below 5% of terrain (matching Java's volcanic interior). *Current state (2026-04-21): Philippines has pre-400 CE volcanic sites, but neither (a) nor (b) is operationally verified — Philippines has lower volcanic density (0.07 vs 0.35 active volcanoes per 1{,}000 km²) and abundant karst. The comparison therefore does NOT falsify, but this is a preliminary status until (a) and (b) are measured.*
- **3 or more peer reviews** (not desk rejects) provide substantive methodological critiques that cannot be addressed — i.e., the *method* is judged flawed, not just the *framing* or *journal fit*. *Current state (2026-04-21): UNTESTED. All rejections have been desk-level.*
- **Domain expert consensus** (≥2 independent archaeologists or geologists with Java expertise) judges the framework fundamentally unsound. *Current state (2026-04-21): UNTESTED. No domain expert has engaged.*
- **Cross-model methodology critique (NEW 2026-04-21):** Two or more independent cross-model skeptical reviews (different training corpus, e.g., DeepSeek + Gemini + GPT) converge on the same methodological flaw that cannot be addressed by revision. *Current state (2026-04-21): Session 19 DeepSeek critical review of P1-core v3.0 identified "calibration is not a calibration" (monument as proxy for landscape rate). This is a **single-model finding**; criterion triggers only if independently replicated by a second model. Action: run Gemini or GPT critical review before JASREP submission to confirm or falsify this concern.*

### Pivot the research line if:
- **All computational journals reject P1:** Reframe from "taphonomic bias framework" to "survey priority model" — keep E110 cascade + E080/E097 fieldwork targeting, drop the civilizational claims
- **P2 settlement model fails peer review on tautology grounds:** Redesign with presence-only methods (MaxEnt) or abandon ML approach for rule-based expert system
- **No acceptance after 6 submissions across different journals:** The work may not be publishable as currently structured — consider consolidating into 1-2 comprehensive papers instead of 6+. *Current state (2026-04-21): 5 rejected + 5 under review. Any one of current 5 rejecting crosses threshold — watch closely.*
- **Fieldwork partner found:** Pivot from computational-only to computational+empirical — this is the DESIRED pivot that would resolve the zero-ground-truth problem
- **Skeptical reviews recommend reframe (NEW 2026-04-21):** If cross-model critical review recommends reframing a paper as "critical review + research proposal" rather than empirical finding (both Session 19 DeepSeek reviews of P1-core v3.0 and P0 draft v0.1 did this), adopt that reframe for that specific paper before submission. This is a paper-level pivot, not a program-level pivot.

---

*This document is the foundation. Everything else can change; this should not — unless reality demands it.*
