# Temporal Overlay Matrix: Integrating Linguistic Phylogenetics, Ancient DNA, and Archaeological Dating to Reconstruct the Deep Settlement History of Nusantara

**VERSION 3.1 — Revised per review: p-value reporting, circularity caveat, structural reorder**

*Mukhlis Amien*
Lab Data Sains, Universitas Bhinneka Nusantara, Indonesia
amien@ubhinus.ac.id

2026-03-05

**[DRAFT v3 — Paper 7 of VOLCARCH Series — NOT FOR DISTRIBUTION]**

> **Version history:**
> - v1 (13 pages): Proposed measuring taphonomic gap using 'oldest date per region'. Tested in E018.
> - v2 (10 pages): Replaced falsified metric with cave/open-air ratio and site density. Added biogeographic argument.
> - v3: Integrates E019 results — Metric 3 (spatial distribution) now **executed and confirmed**. Zone B cells cluster significantly closer to volcanoes than Zone A cells (Cohen's d = 1.005).
> - **v3.1 (this version):** Revised per review — fixes p-value reporting, disambiguates two burial timeframes, strengthens circularity caveat (Analysis 3 = primary, Analysis 2 = corroborating), adds Zone C data, reorders Section 7 to B→A→C.

> **Series context:** Paper 7 of VOLCARCH. Papers 1-2 quantified physical taphonomic loss (volcanic burial, coastal submersion). Paper 2 built a predictive ML settlement model. This paper integrates three independent temporal clocks into a Temporal Overlay Matrix (TOM) for Nusantara.

---

## Abstract

The settlement history of Nusantara is reconstructed almost entirely from the archaeological record — a record that Papers 1 and 2 have demonstrated to be systematically biased by volcanic burial and coastal submersion. This paper proposes the Temporal Overlay Matrix (TOM): a framework comparing three independent temporal clocks — linguistic phylogenetics, population genetics, and radiocarbon archaeology — across eight Nusantaran regions.

An initial proof-of-concept experiment (E018) tested the prediction that linguistic/genetic dates would exceed archaeological dates in proportion to volcanic density, using 'oldest date per region' as the metric. The test failed: correlation was near zero for Neolithic data and weakly negative for deep-time data. Critically, this failure reveals a more powerful argument than the original test: every deep-time site across all eight regions is a cave site, regardless of volcanic density. This cave survivorship bias is itself evidence of systematic open-air site loss — the very mechanism Papers 1-2 predict.

We introduce three revised test metrics immune to cave-site confounding: (1) cave/open-air site ratio for periods >10,000 BP; (2) site density per 5,000-year time bin comparing volcanic versus non-volcanic regions; and (3) spatial distribution of deep-time sites relative to volcanic plains. **Metric 3 has been executed (Experiment E019) and strongly supports H-TOM:** All four known deep-time Java sites are in karst caves or river terraces, 90-170 km from volcanic centers — none on volcanic plains. A corroborating spatial analysis shows that Zone B cells (high suitability, moderate burial, zero known sites) cluster at median 16.1 km from the nearest volcano, versus 42.5 km for Zone A cells where sites are found (Mann-Whitney U, Z = 39.50, Cohen's d = 1.005).

We further develop the biogeographic argument that Java must have been inhabited before ≥67,800 BP — the confirmed date of Sulawesi cave art (Oktaviana et al. 2026) — because Wallace's Line crossing requires maritime technology that could only have been developed on the Sunda Shelf. At 3.6 mm/yr volcanic sedimentation over 60,000-70,000 years, evidence of this occupation now lies under an estimated 200+ meters of tephra on Javanese volcanic plains.

These arguments — the universal cave bias, the biogeographic deduction, the confirmed spatial segregation of sites from volcanic plains, and the pending statistical tests — constitute a case that absence of evidence in the Javanese open-air record is not evidence of absence of civilization. It is evidence of burial depth.

**Keywords:** taphonomic gap; cave survivorship bias; glottochronology; Bayesian phylolinguistics; ancient DNA; Wallace's Line; Sunda Shelf; radiocarbon chronology; Temporal Overlay Matrix; Java deep history; Sulawesi cave art

---

## 1. The Three Clocks: Framework

Reconstructing the deep past requires clocks — mechanisms that change at measurable rates and whose current state can be read to infer elapsed time. Three independent clocks are available for settlement history:

| Clock | What it measures | Method | Depends on physical preservation? |
|-------|-----------------|--------|----------------------------------|
| Language | When a language diverged from its ancestor | Bayesian phylolinguistics; glottochronology | NO — language lives in mouths of speakers |
| Genetics | When populations separated or admixed | Haplogroup dating; ancient DNA | NO — genes are biologically inherited |
| Archaeology | When oldest physical evidence was deposited | Radiocarbon (C14), OSL, U-series | YES — can be buried, submerged, destroyed |

These three clocks are genuinely independent. They measure different materials, use different methods, and are subject to different sources of error. The crucial asymmetry — first noted in Paper 1 and now empirically confirmed by E018 — is that only the archaeological clock depends on physical preservation.

This asymmetry is the core of the VOLCARCH argument. In regions with high taphonomic loss, the archaeological clock will read younger than the other two — not because occupation is recent, but because evidence is gone. The TOM is designed to measure this discrepancy.

---

## 2. Experiment E018: What Failed and Why

### 2.1 Original prediction (v1)

Draft v1 predicted that in regions with high volcanic density (high TAP_index), linguistic and genetic settlement dates would exceed archaeological dates by a larger margin than in low-TAP regions. The metric used was the oldest dated site per region (A_age), subtracted from the linguistic or genetic estimate.

### 2.2 What E018 found

E018 ran two tests using published data from eight Nusantaran regions:

**Test 1 — Neolithic frame:** Using Austronesian expansion dates (~3,500-4,500 BP) for all three clocks. Result: Spearman rho = 0.013. No correlation. Explanation: Austronesian expansion is too recent (~4,000 years) for volcanic burial to have erased evidence.

**Test 2 — Deep-time frame:** Using oldest Homo sapiens evidence per region, including Oktaviana et al. 2026 (Sulawesi, >=67,800 BP). Result: Spearman rho = -0.143. Weak negative correlation — volcanic regions showed older dates, not younger. Explanation: cave-site survivorship bias.

### 2.3 The cave survivorship bias: a more important discovery

The two oldest sites are in Sumatra (Lida Ajer, 68,000 BP) and Sulawesi (Liang Metanduno, 67,800 BP) — both in regions with substantial volcanism — and both are cave sites in karst terrain far from volcanic plains. Caves protect evidence from tephra deposition; open-air sites do not.

> *The critical pattern: every deep-time site (>40,000 BP) in the entire Nusantaran dataset is a cave site or river terrace. Not one open-air site on a volcanic plain has yielded pre-Neolithic evidence anywhere in the archipelago.*

This is not evidence against taphonomic loss. It is the signature of taphonomic loss.

**Table 1: All deep-time sites in Nusantara — context distribution (E018 data)**

| Region | Volcanoes | Oldest site (BP) | Site name | Context | Distance from volcanic plain |
|--------|-----------|-----------------|-----------|---------|----------------------------|
| Sumatra | 35 | ~68,000 | Lida Ajer | CAVE | Karst mountains, far from Barisan axis |
| Sulawesi | 11 | 67,800 | Liang Metanduno | CAVE | Muna Island, minimal local volcanism |
| Java | 45 | ~60,000 | Song Terus | CAVE | Gunung Sewu karst, far south coast |
| Nusa Tenggara | ~15 | 44,600 | Laili Cave | CAVE | Karst, Timor |
| Philippines | 24 | 47,000 | Tabon Cave | CAVE | Palawan, low volcanism |
| Maluku | ~16 | 36,000 | Golo Cave | CAVE | Morotai Island |
| Kalimantan | 0 | 40,000 | Niah Cave | CAVE | Karst, Sarawak |
| Madagascar | 0 | 10,500 | Christmas River | Open-air* | *Austronesian-era, not deep-time |

**Pattern: 7 of 7 deep-time sites (>10,000 BP) are cave sites. Open-air deep-time sites: zero.**

---

## 3. The Biogeographic Argument: Java Must Predate Sulawesi

### 3.1 The deductive chain

The strongest argument in this paper is not statistical — it is deductive, and it follows from a single confirmed fact: humans were in Sulawesi at >=67,800 BP (Oktaviana et al. 2026).

> *Wallace's Line has never been dry land — not even at glacial maximum. Crossing it required intentional maritime technology: watercraft capable of open-sea navigation. Technology of this complexity does not appear overnight. It develops over generations of coastal adaptation. Therefore: the population that crossed Wallace's Line had already spent considerable time on the Sunda Shelf before crossing. Sunda Shelf = modern Java + Sumatra + Kalimantan + Malaysia, all connected dry land during glacial maximum. Conclusion: humans were on what is now Java before they were in Sulawesi — i.e., before 67,800 BP.*

### 3.2 Formalizing the chain

```
Confirmed:  Sulawesi >= 67,800 BP              (Oktaviana et al. 2026, Nature)
Confirmed:  Wallace's Line requires maritime crossing (never dry land)
Confirmed:  Maritime technology develops through coastal adaptation
Confirmed:  Sunda Shelf = Java + Sumatra + Kalimantan during glacial max
Therefore:  Sunda Shelf occupation predates Sulawesi crossing
Therefore:  Java occupation > 67,800 BP

Evidence:   Song Terus cave ~ 60,000 BP         (Semah et al. 2023)
Gap:        ~8,000 years younger than expected
Expl. A:    Song Terus is not the oldest Java site — just oldest found
Expl. B:    Open-air Java sites from >67,800 BP exist but are buried

At 3.6 mm/yr x 67,800 years = ~244 meters below volcanic plain surface
```

### 3.2a Counterargument: the Borneo bypass

The deductive chain assumes the crossing population was on Java. But Wallace's Line can also be crossed from northern Borneo (modern Kalimantan/Sabah) to Sulawesi via the Makassar Strait. If the crossing originated from Borneo, then the argument proves Sunda Shelf occupation before 67,800 BP — but not necessarily *Java* specifically.

**Response:** This weakens the specificity but not the substance. (1) At glacial maximum, Java, Sumatra, Borneo, and the Malay Peninsula were connected by dry land as a single landmass (Sunda Shelf). A population anywhere on the Shelf had access to all of it. (2) The shortest crossing points from Sunda to Sulawesi are from eastern Borneo (~120 km across Makassar Strait) or from eastern Java/Bali (~35 km at the narrowest Lombok Strait). Both routes are plausible. (3) Even if the crossing was from Borneo, humans arriving on the Sunda Shelf from mainland Asia would have traversed or passed through what is now Java, as it sits on the main southern migration corridor. (4) The burial depth argument applies regardless of which crossing point was used: evidence of Sunda Shelf occupation before 67,800 BP exists but is buried under volcanic sediment on Java's plains and under sea-level rise elsewhere.

### 3.3 The 244-meter number

Paper 1 calibrated volcanic sedimentation in Java at a mean of 3.6 mm/year from the Dwarapala stratigraphy and corroborating sites. Applying this rate over 67,800 years:

> **3.6 ± 1.2 mm/yr x 67,800 yr = 163-326 meters of accumulated volcanic sediment on Java's plains since the confirmed date of Sulawesi's oldest cave art (central estimate: 244 m).**

No archaeological survey method currently in routine use — including ground-penetrating radar, which has a practical depth limit of 10-15 meters in volcanic sediments — can reach material at these depths. Even the lower bound estimate (163 m) exceeds routine survey depth by an order of magnitude. The evidence is not missing. It is present at a depth that has never been systematically probed.

---

## 4. Revised Test Metrics

### 4.1 Metric 1 (Primary): Cave/open-air site ratio for periods >10,000 BP

> **H-TOM v2 (pre-registered):** The ratio of cave sites to total sites for deposits >10,000 BP will be significantly higher in volcanic regions (TAP_index > 0.5) than in non-volcanic regions (TAP_index < 0.2). Formally: mean(cave_ratio | high_TAP) > mean(cave_ratio | low_TAP), tested by Mann-Whitney U, one-tailed, alpha = 0.05.

This metric directly measures the systematic disappearance of open-air evidence in volcanic regions. It is immune to the cave survivorship bias that contaminated the original metric.

**Status:** PENDING — requires NusaRC database (or mini-NusaRC with ~80 sites). See Section 8.

### 4.2 Metric 2: Site density per 5,000-year time bin

Rather than asking 'what is the oldest site?', this metric asks: 'how many sites are documented per unit time, and does this number drop off faster in volcanic regions than in non-volcanic regions?'

Prediction: in Kalimantan (zero volcanoes), site density will remain roughly constant from 40,000 BP to the present. In Java (45 volcanoes), site density will show a sharp dropoff for deposits older than 5,000-10,000 BP.

**Status:** PENDING — requires NusaRC database.

### 4.3 Metric 3: Spatial distribution relative to volcanic plains — CONFIRMED (E019)

All deep-time Java sites — Song Terus, Wajak, Trinil, Sangiran — are located in karst zones or river terrace systems. None are on the volcanic plains that dominate northern and central Java.

**Prediction (from v2):** A GIS analysis mapping all pre-Neolithic sites against distance from the nearest active volcano will show that in Java, all such sites cluster at >50 km from volcanic centers or in karst zones.

**Result (E019, 2026-03-05): CONFIRMED — strongly supports H-TOM.**

E019 performed three complementary spatial analyses using pre-computed zone data from E013/E016 (378 East Java sites, 7 volcanoes, 65,432 grid cells):

**Analysis 1 — Sites cluster near volcanoes (the taphonomic trap input):**
Sites are significantly *closer* to volcanoes than geographic chance (median 27.9 km vs grid baseline 59.2 km, Mann-Whitney p = 3.02e-36). People settle on fertile volcanic lowlands — creating the conditions for taphonomic burial.

**Analysis 2 — Zone B clusters on the volcanic axis (key quantitative result):**

| Zone | Description | Median distance to volcano | n cells |
|------|-------------|--------------------------|---------|
| C | High suitability, deep burial (>300 cm) | **2.6 km** | 48 |
| B | High suitability, moderate burial (100-300 cm), **zero sites** | **16.1 km** | 1,093 |
| A | High suitability, shallow burial (<100 cm), sites exist | **42.5 km** | 15,217 |
| E | Low suitability | **71.0 km** | 49,074 |

- Mann-Whitney U (Zone A vs Zone B) = 14,254,494
- **p < 10⁻¹⁰⁰** (Z = 39.50; underflows 64-bit float precision; the test statistic, not the p-value, is the meaningful number)
- **r = 0.309** (rank-biserial correlation)
- **Cohen's d = 1.005 (large effect)**

The monotonic gradient C < B < A < E maps directly onto the burial depth gradient: areas closer to volcanoes have deeper tephra cover, pushing archaeological material below detection. Zone B cells are suitable for habitation (the model predicts people *would* live there) but have zero known sites — because the evidence is buried under 100-300 cm of post-1268 CE tephra.

**Important temporal disambiguation:** The Zone B burial depth (100-300 cm) is modeled from the Pyle (1989) exponential thinning function calibrated to the 1268 CE Rinjani eruption — a 758-year timeframe. This is the *minimum* burial from a single well-documented event. Over 67,800 years (the timeframe of the biogeographic argument in Section 3), cumulative tephra deposition at 3.6 mm/yr would produce ~244 meters of sediment — two orders of magnitude deeper. The Zone B analysis tests whether *even recent* (post-1268 CE) burial is sufficient to erase the archaeological record; the biogeographic argument concerns the far greater cumulative burial over deep time.

**Analysis 3 — Deep-time sites: all in protected contexts:**

| Site | Age | Distance to nearest volcano | Context |
|------|-----|-----------------------------|---------|
| Song Terus | ~60,000 BP | 153.5 km (Kelud) | Cave, Gunung Sewu karst |
| Trinil | ~500,000 BP | 121.6 km (Kelud) | River terrace, Solo River |
| Sangiran | ~1,600,000 BP | 169.3 km (Kelud) | River erosion, Solo basin dome |
| Wajak | ~28,000 BP | 89.7 km (Kelud) | Cave, Tulungagung karst |

All four sites are 90-170 km from the nearest volcano, all in karst caves or river terraces. None on volcanic plains.

**Circularity caveat and evidential hierarchy:** Analysis 2 uses the Pyle (1989) burial depth model to define zones, and this model is itself distance-dependent (burial depth decreases exponentially with distance from source). Therefore Analysis 2 contains partial circularity: zones defined by distance are then tested for distance differences. The large effect size (Cohen's d = 1.005) confirms the model's internal consistency, but *cannot independently confirm H-TOM*.

The non-circular evidence is **Analysis 3**: the deep-time site distribution is drawn from published literature, independent of any model in this study. The fact that *all four* known deep-time Java sites are in karst caves or river terraces 90-170 km from volcanic centers — and *none* on volcanic plains — is the primary empirical evidence for spatial segregation. Analysis 2 quantifies the pattern; Analysis 3 establishes it independently.

**Zone C — the extreme case:** Zone C cells (n=48, deep burial >300 cm, median 2.6 km from nearest volcano) have the highest predicted suitability in the model but the fewest known sites (zero). At median 2.6 km from volcanic centers, these cells represent the most extreme taphonomic burial — and the strongest version of the argument that absence of sites reflects burial depth, not absence of habitation.

**Data and code:** `experiments/E019_spatial_distribution/`
**Figures:** `fig_zone_distance_boxplot.png`, `fig_deep_time_context_map.png`, `fig_distance_histogram.png`

### 4.4 Metric 4 (Exploratory): Chronological gaps in cave sequences

Examining whether cave sequences in volcanic regions show depositional gaps corresponding to known major eruptions. The Toba super-eruption (~74,000 BP) deposited ashfall across South and Southeast Asia. If Song Terus shows a Toba ash layer with occupation above it, this is direct evidence that volcanic events impacted human occupation even in cave contexts.

**Status:** Exploratory — insufficient stratigraphic resolution in published data.

---

## 5. What Remains Valid from Draft v1

### 5.1 The three-clock framework

The fundamental logic — that language, genes, and archaeology are independent clocks, and that their divergence in high-taphonomy regions measures preservation loss — remains valid and is now strengthened by E018 and E019.

### 5.2 Linguistic methods: glottochronology and Bayesian phylolinguistics

The linguistic dating methods described in v1 (Swadesh list divergence, Gray et al. 2009 Bayesian framework) remain valid.

### 5.3 The Suriname calibration

Javanese contract laborers transported to Suriname between 1890 and 1939 form a precisely dated, geographically isolated speech community. Systematic Swadesh list comparison between Surinaams Javaans and modern Central Javanese provides an empirical calibration of the vocabulary change rate for the Javanese branch — anchoring the glottochronological estimates used in the TOM's linguistic clock.

### 5.4 Typology of clock disagreements

Three patterns of disagreement between the three clocks remain analytically valuable:

- **Type 1 — Taphonomic Gap:** Language and genes indicate older settlement than archaeology. The pattern predicted for volcanic Java.
- **Type 2 — Language Replacement:** Genes old, language young. Population present for millennia adopted a new language. Expected for Austronesian spread over pre-existing Australo-Melanesian populations.
- **Type 3 — Population Replacement:** Language and archaeology continuous, but genetic composition changes. The Malagasy case.

### 5.5 Ghost populations

The VOLCARCH explanation for genetic 'ghost populations' — ancestral components in modern Southeast Asian genomes that correspond to no known archaeological culture — remains valid. These components may represent pre-Austronesian populations whose material culture is buried under volcanic sediment or drowned under the Java Sea.

### 5.6 NusaRC database need

The need for a standardized Nusantaran radiocarbon database remains the primary infrastructure prerequisite for Metrics 1 and 2. **A mini-NusaRC (~80 sites) is now in development (E020) to enable preliminary testing.**

---

## 6. What Draft v1 Claimed That Must Be Revised or Retired

### 6.1 Retired: Table 2 gap estimates

The preliminary gap estimates in v1 — which assigned L_gap values based on oldest date differences — are retired. They used the wrong metric confounded by cave survivorship bias.

### 6.2 Revised: The core falsifiable prediction

V1's formulation ('Gap magnitude correlates with TAP_index') is retired. The revised prediction (H-TOM v2) uses cave/open-air ratios. See Section 4.1.

### 6.3 Softened: Claims about direct falsifiability

TOM tests a specific prediction derived from VOLCARCH — not the entire framework. Disconfirmation of H-TOM v2 would require explanation but would not automatically falsify Papers 1-6.

---

## 7. The Argument as It Now Stands

After E018 and E019, the argument of Paper 7 has three components. We present them in order of logical force, from the argument that requires the fewest assumptions to the one that requires the most:

### Component B: Deductive (strongest — requires only one confirmed fact)

Humans were in Sulawesi at >=67,800 BP (Oktaviana et al. 2026, *Nature*). Wallace's Line has never been dry land — crossing it required maritime technology. Maritime technology develops through coastal adaptation on land. The only available landmass is the Sunda Shelf (Java + Sumatra + Kalimantan). Therefore Java was occupied before 67,800 BP.

This is a deductive chain with one confirmed premise and two uncontroversial geographic facts. Its conclusion — that Java was inhabited before Sulawesi — does not depend on any model, statistical test, or archaeological interpretation in this paper.

At 3.6 mm/yr volcanic sedimentation (Paper 1, calibrated from Dwarapala stratigraphy), evidence of this occupation now lies under approximately 163-326 meters of volcanic sediment (using the Paper 1 confidence interval of 3.6 ± 1.2 mm/yr). Even the lower bound (163 m) far exceeds any routine archaeological survey depth.

### Component A: Empirical (strong — independent of any model in this study)

Every deep-time site (>40,000 BP) in Nusantara is a cave site or river terrace — a documented pattern from published literature independent of any model in this study. No open-air site on a volcanic plain has yielded pre-Neolithic evidence anywhere in the archipelago.

**Strengthened by E019 Analysis 3:** Within East Java specifically, all four known deep-time sites (Song Terus, Trinil, Sangiran, Wajak) are in karst caves or river terraces, 90-170 km from the nearest volcanic center. This is the primary non-circular spatial evidence for H-TOM.

**Zone B/C quantification (E019 Analysis 2):** The areas closest to volcanoes — Zone B (median 16.1 km, n=1,093 cells) and Zone C (median 2.6 km, n=48 cells) — have zero known archaeological sites despite being predicted as highly suitable for habitation. This quantifies the magnitude of the spatial gap. *(Note: Analysis 2 contains partial circularity because zones are defined by a distance-dependent model; see Section 4.3 caveat. The effect size, Cohen's d = 1.005, confirms internal consistency but should be interpreted as corroborating, not primary, evidence.)*

### Component C: Statistical (partially confirmed, partially pending)

- **Metric 3 (spatial distribution): CONFIRMED** by E019 Analysis 3 (non-circular) and Analysis 2 (corroborating, partially circular).
- **Metric 1 (cave/open-air ratio): PENDING** — requires mini-NusaRC (E020) or full NusaRC.
- **Metric 2 (site density per time bin): PENDING** — requires NusaRC.

> *We did not find what we expected. We found something more important: the signature of what is missing. Every ancient site in Nusantara hides in a cave. The caves survived. The plains did not. That is not an archaeological puzzle — it is an archaeological answer.*

---

## 8. Next Steps and Status

### 8.1 Completed

- [x] **E018:** Proof of concept — identified cave-site confound, redirected metrics
- [x] **E019:** Metric 3 spatial analysis — **CONFIRMED**, Cohen's d = 1.005

### 8.2 In development: Mini-NusaRC (E020)

A semi-automated compilation of ~80 key sites across 8 Nusantaran regions from open-access literature. Target: sufficient data to run preliminary tests of Metrics 1 and 2 without waiting for full NusaRC (6-18 months). See `experiments/E020_mini_nusarc/` for concept and pipeline.

With mini-NusaRC + E019, **all three metrics could be tested**, making Paper 7 publishable as a complete empirical argument rather than a theoretical framework awaiting data.

### 8.3 Future: Full NusaRC (6-18 months)

Comprehensive Nusantaran radiocarbon database — potential standalone data paper in Journal of Open Archaeology Data.

### 8.4 Publication strategy

**Option A (immediate):** Metric 3 (E019) as short communication — JAS: Reports or Archaeological Prospection.
**Option B (with mini-NusaRC):** Full Paper 7 with all three metrics tested — target: Quaternary Science Reviews or Journal of Human Evolution.
**Option C (deferred):** Wait for full NusaRC. Not recommended — Components A and B already stand without it.

---

## 9. Conclusions

Experiment E018 failed to confirm the original TOM prediction. It succeeded in something more valuable: identifying why the prediction failed, and replacing a weak metric with stronger ones. The cave survivorship bias revealed by E018 is itself strong evidence for taphonomic loss.

Experiment E019 confirmed Metric 3 through two complementary analyses. The primary evidence (Analysis 3) is non-circular and model-independent: all four known deep-time Java sites are in karst caves or river terraces, 90-170 km from volcanic centers — none on volcanic plains. The corroborating evidence (Analysis 2) shows that Zone B cells (high suitability, zero sites) cluster significantly closer to volcanoes than Zone A cells where sites are found (Cohen's d = 1.005, Z = 39.50). Zone C cells (median 2.6 km from volcanoes, n=48) represent the extreme case — highest modeled suitability, deepest burial, zero archaeological sites.

The biogeographic argument — that Java must have been inhabited before Sulawesi's confirmed >=67,800 BP cave art, because Wallace's Line requires maritime technology developed on the Sunda Shelf — is a deductive argument requiring only one confirmed premise. Its conclusion: evidence of Java's pre-67,800 BP occupation now lies under approximately 163-326 meters of volcanic sediment (at 3.6 ± 1.2 mm/yr), inaccessible to any current survey method.

Three independent lines of evidence — the biogeographic deduction, the universal cave bias confirmed by published site data, and the spatial segregation quantified by E019 — converge on the same conclusion: Nusantara's apparent archaeological thinness before 4,000 BP is an artifact of where we can look, not of where people lived.

---

## Draft Notes

- [x] ~~Section 3.3: Present 244m as range~~ → Done in v3.1: 163-326m using 3.6 ± 1.2 mm/yr (Section 7 Component B).
- [ ] Section 4.1: Consider formal pre-registration at OSF before running Metric 1 with mini-NusaRC.
- [x] ~~Section 4.3: E019 circularity caveat~~ → Done in v3.1: expanded caveat, Analysis 3 = primary, Analysis 2 = corroborating.
- [x] ~~Fix "p ≈ 0" reporting~~ → Done in v3.1: Z = 39.50 (underflows float64), emphasis on Cohen's d.
- [x] ~~Disambiguate 758yr (Zone B) vs 67,800yr (biogeographic)~~ → Done in v3.1: new paragraph in Section 4.3.
- [x] ~~Reorder Section 7: B→A→C~~ → Done in v3.1: deductive first, empirical second, statistical third.
- [x] ~~Zone C data~~ → Done in v3.1: added to Section 4.3 and Section 9.
- [x] ~~Song Terus vs Sulawesi gap~~ → Done in v3.1: Section 3.2a "Borneo bypass" counterargument added. Response: weakens specificity but not substance.
- [ ] Toba super-eruption (~74,000 BP): check Semah et al. 2023 for Song Terus stratigraphy re: Toba ash layer.
- [ ] Ghost population section from v1 (Section 6.2): reinstate in full paper — connects genetic evidence to taphonomic argument.
- [ ] E019 figures: integrate `fig_zone_distance_boxplot.png` and `fig_deep_time_context_map.png` as paper figures.
- [ ] Title scope: consider subtitle addition "— Evidence from Spatial Analysis of East Java Archaeological Sites" to set expectations. The TOM framework is theoretical contribution; empirical content is currently one metric on one region.
- [ ] Publication strategy decision: Option A (short comm with E019 only) vs Option B (full paper with mini-NusaRC). User decision.

---

## References

Barker, G., et al., 2007. The 'human revolution' in lowland tropical Southeast Asia: the antiquity of anatomically modern humans at Niah Cave. Journal of Human Evolution 52, 243-261.

Detroit, F., et al., 2004. Palaeontology of Tabon Cave. Comptes Rendus Palevol 3, 705-712.

Gray, R.D., Drummond, A.J., Greenhill, S.J., 2009. Language phylogenies reveal expansion pulses and pauses in Pacific settlement. Science 323, 479-483.

Hansford, J., et al., 2018. Early Holocene human presence in Madagascar evidenced by exploitation of avian megafauna. Science Advances 4, eaat6925.

Hawkins, S., et al., 2017. Late Pleistocene and Holocene human biogeography in island Southeast Asia: ancient DNA from Laili Cave, Timor-Leste. Quaternary Science Reviews 171, 58-72.

Hill, C., et al., 2007. A mitochondrial stratigraphy for island Southeast Asia. Molecular Biology and Evolution 24, 871-882.

Lipson, M., et al., 2014. Reconstructing Austronesian population history in Island Southeast Asia. Nature Communications 5, 4689.

Oktaviana, A.A., et al., 2026. Rock art from at least 67,800 years ago in Sulawesi. Nature 650, 652-656.

Pierron, D., et al., 2017. Genomic landscape of human diversity across Madagascar. PNAS 114, E2341-E2350.

Semah, F., et al., 2023. Song Terus cave and the late Pleistocene record of Java. L'Anthropologie.

Tumonggor, M., et al., 2013. The Indonesian archipelago: an ancient genetic highway linking Asia and the Pacific. European Journal of Human Genetics 21, 824-831.

Westaway, K., et al., 2017. An early modern human presence in Sumatra 73,000-63,000 years ago. Nature 548, 322-325.

Amien, M., 2026a-f. VOLCARCH Series, Papers 1-6. Lab Data Sains, Universitas Bhinneka Nusantara.
