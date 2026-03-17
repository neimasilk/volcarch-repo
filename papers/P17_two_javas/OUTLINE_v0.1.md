# P17: Two Javas — Paper Outline v0.1

**Working Title:** "Two Javas: Spatial Segregation of Sacred and Administrative Landscapes in Volcanic Java and Its Consequences for Archaeological Inference"

**Alternative Titles:**
- "Volcano Java and Court Java: How Geographic Taphonomy Creates Two Parallel Archaeological Records"
- "The Court Zone Hypothesis: Why the Visible Inscriptional Record Misrepresents Ancient Java"

**Author:** Mukhlis Amien (single-author)
**Target journal:** Antiquity (Q1, Cambridge UP) — full article, not Project Gallery
**Backup:** Journal of Archaeological Science, or World Archaeology
**Date:** 2026-03-17

---

## Core Argument (1 paragraph)

The archaeological record of Java is not uniformly dark — it has a spatial structure that maps onto volcanic geography. Analysis of 142 temples (candi), 176 geocoded inscriptions, and 391 archaeological sites reveals two distinct archaeological worlds separated by a 13km median distance gap. "Volcano Java" (0-15km from active volcanoes) is dominated by sacred architecture, relatively indigenous vocabulary, and progressive burial at 2.4-6.2 mm/year. "Court Java" (15-30km) is dominated by administrative inscriptions, Sanskrit-heavy content, and the dramatic 929 CE discursive shift. The temporal trend of increasing pre-Indic vocabulary (rho=0.781, p<0.0001) occurs exclusively in the court zone. Post-929 CE epigraphy abandons the court zone entirely, shifting to the periphery with indigenous-rich content. These findings demonstrate that using inscription counts as proxies for population density or cultural activity conflates two fundamentally different archaeological regimes, producing systematic misrepresentation of pre-Hindu Java.

---

## Structure

### 1. Introduction (~1500 words)
- The "uniform darkness" assumption in Indonesian archaeology
- Why spatial structure matters for inference
- Research questions:
  1. Do sacred architecture and administrative inscriptions occupy the same geographic space?
  2. Does volcanic proximity predict vocabulary composition?
  3. Is the 929 CE discursive shift spatially uniform or zone-specific?
  4. What does this mean for archaeological inference?

### 2. Background (~1000 words)
- Volcanic geography of Java (45 active volcanoes, sedimentation rates)
- The inscriptional record: 268 DHARMA inscriptions, distribution patterns
- Candi distribution: 142 temples, known westward clustering (E065)
- The 929 CE Mataram collapse: geographic and cultural shift
- Prior spatial analyses: Degroot 2009 (candi landscape), Miksic 2004 (Trowulan)

### 3. Data (~800 words)
- **Dataset 1:** 142 candi with coordinates and nearest-volcano distances (E031)
- **Dataset 2:** 176 geocoded inscriptions with vocabulary analysis (E082 + E074 + E030)
- **Dataset 3:** 391 archaeological sites with DEM-extracted elevation (E001 + E003)
- **Dataset 4:** 27 burial sites with measured depth (E083)
- All datasets are independently constructed and documented

### 4. Methods (~1000 words)
- Distance-to-nearest-volcano computation (haversine, 10 major Java volcanoes)
- Zone classification: Volcano (<15km), Court (15-30km), Periphery (>30km)
- Elevation zone analysis (DEM extraction, 5 bins from coastal to mountain)
- Pre-Indic vocabulary ratio (E030 methodology)
- Spearman correlations, Mann-Whitney tests, Fisher exact, chi-square
- Partial correlation controlling for inscription length

### 5. Results (~2500 words)

#### 5.1 Spatial segregation of candi and inscriptions (E104)
- Candi peak: 0-10km (42.3%), median 14.6km
- Inscription peak: 20-30km (39.2%), median 27.6km
- Mann-Whitney p < 0.000001
- Fisher exact: inscriptions 1.86× more court-concentrated (p=0.012)
- Figure 1: Dual histogram of distance distributions

#### 5.2 Elevation gradient (E100)
- Site density increases monotonically: 1.96 → 18.61 per 1000km²
- Mountain zones = volcano survivors, not representative sample
- Chi-square p < 0.000001 (non-uniform distribution)
- Figure 2: Elevation zone bar chart with density

#### 5.3 Vocabulary × burial depth interaction (E102)
- Indigenous ratio × nearest burial depth: rho = 0.456 (length-controlled), p < 0.0001
- 5.8× jump: shallow (9.3%) → deep (56.4%) indigenous vocabulary
- Effect driven by Sanskrit inscriptions (rho=0.512) not Old Javanese (rho=0.138, NS)
- Figure 3: Depth-binned vocabulary profile

#### 5.4 The court zone as Indianization epicenter (E103)
- Temporal pre-Indic trend: ONLY at 20-40km (rho=0.781, p<0.0001)
- Near-volcano: no trend (rho=0.106, NS) — always indigenous
- Far-volcano: no trend (rho=0.045, NS) — always indigenous
- 929 CE shift: pre=0.012 → post=0.196, MW p<0.0001 — ONLY in court zone
- Figure 4: Temporal trend split by zone (3 panels)

#### 5.5 Topic geography: the post-929 relocation (E105)
- Sanskrit-dominant: 72% in court zone (56/78)
- Post-929 shift: court → periphery, Sanskrit → indigenous
- Pre-929: 57% of inscriptions in court zone (91% Sanskrit)
- Post-929: 53% in periphery (89% mixed/indigenous)
- Figure 5: Pre/Post-929 zone × topic stacked bars

### 6. Discussion (~2000 words)

#### 6.1 Two Javas: a spatial model of archaeological darkness
- Volcano Java: physical heritage (candi), indigenous culture, buried at 3-9m
- Court Java: textual heritage (inscriptions), Sanskrit overlay, visible but biased
- Peripheral Java: post-929 indigenous recovery zone
- The "double filter": Volcano Java hidden by geology, Court Java biased by genre

#### 6.2 Consequences for archaeological inference
- Inscription counts ≠ population density (inscriptions measure COURTS, not PEOPLE)
- The "Hindu period" was a COURT phenomenon, not a civilization-wide transformation
- Pre-Hindu Java was never absent — it was ALWAYS present in volcano and peripheral zones
- What we call "Indianization" is a 20-30km-wide band around selected volcanoes

#### 6.3 The 929 CE collapse as natural experiment
- The collapse didn't just move the political center east
- It EXPOSED the indigenous substrate that was always underneath the Sanskrit overlay
- E103: post-929 pre-Indic ratio jumps from 0.012 to 0.196 — the substrate was always there
- Analogy: peeling back paint to find the original wall underneath

#### 6.4 Implications for fieldwork
- Zone B/C targets (E080, E097) are in VOLCANO JAVA — the buried sacred landscape
- GPR/ERT surveys should target 0-10km zone, not court zone
- Collaborators should be volcanologists (for burial context) not epigraphers (for text)

#### 6.5 Limitations
- Single-island analysis (Java only; Bali, Sumatra untested)
- Inscription geocoding has ±5km uncertainty for some entries
- Pre-Indic ratio as vocabulary proxy has limitations (E030 methodology)
- Burial depth matching is to NEAREST burial site, not SAME site

### 7. Conclusion (~500 words)
- Java's archaeological record is not uniformly dark — it is structurally biased
- Sacred architecture clusters on volcano slopes; administrative texts cluster on fertile plains
- The 929 CE collapse reveals the indigenous substrate by removing the Sanskrit overlay
- Archaeological inference must account for this spatial structure or risk systematic error

---

## Figures (6)

| # | Content | Source | Type |
|---|---------|--------|------|
| 1 | Candi vs inscription distance distributions | E104 | Data (dual histogram) |
| 2 | Elevation × site density (5 zones) | E100 | Data (bar chart) |
| 3 | Depth-binned vocabulary profile | E102 | Data (grouped bars) |
| 4 | Temporal pre-Indic trend × zone (3 panels) | E103 | Data (scatter + regression) |
| 5 | Pre/Post-929 zone × topic | E105 | Data (stacked bars) |
| 6 | "Two Javas" conceptual map | New (or NotebookLM) | Conceptual |

## Experiments → Paper Mapping

| Section | Experiments | Key Stats |
|---------|-----------|-----------|
| 5.1 Segregation | E104, E031, E065 | MW p<0.0001, Fisher OR=1.86 |
| 5.2 Elevation | E100 | Density 1.96→18.61, chi2 p<0.0001 |
| 5.3 Vocabulary | E102 (confound-checked) | rho=0.456 (partial), p<0.0001 |
| 5.4 Court zone | E103 | rho=0.781 (court only), 929 shift p<0.0001 |
| 5.5 Topic geography | E105 | Sanskrit 72% court, post-929→periphery |

## What Makes This Novel

1. **First spatial decomposition** of Java's archaeological record into volcanic distance zones
2. **"Two Javas" model** — quantitative evidence that candi and inscriptions represent different worlds
3. **Court zone as Indianization epicenter** — the "Hindu period" was a 15km-wide phenomenon
4. **929 CE as natural experiment** — political collapse reveals spatial structure of cultural overlay
5. **Burial depth × vocabulary interaction** — volcanic burial preferentially hides indigenous content

## Risk Assessment

- **STRONG:** All five results are significant (p < 0.01) and cross-validated
- **MEDIUM:** E102 confound (length-driven) — acknowledged, partial correlation still significant
- **LOW:** Overlap with P7 (submitted to Antiquity) — P17 goes MUCH deeper with vocabulary + topics
- **NOTE:** P7 is Project Gallery (short); P17 is full article. Different scope, different journal.

## Estimated Length
- 8,000-10,000 words
- 6 figures, 3-4 tables
- ~40 references
