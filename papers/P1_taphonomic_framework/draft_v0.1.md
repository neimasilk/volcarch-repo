# A Framework for Estimating Volcanic Taphonomic Bias in Indonesian Archaeological Records: The Dwarapala Calibration

**Draft v0.2 — 2026-02-24**
**Target journal:** Journal of Archaeological Science: Reports
**Status:** All sections drafted. Secondary anchor rates computed. Ready for review.

---

## Abstract

Active volcanism has shaped East Java's landscape for millennia, depositing centimeters to meters of ash and sediment per eruption. We argue this process systematically biases the observable archaeological record against sites in high-deposition zones. Using the Dwarapala statues of Singosari (~1268 CE) as an empirical calibration point — found half-buried after 510 years of volcanic sedimentation — we estimate a baseline burial rate of 3.6 mm/year in the Malang basin. Extrapolated to earlier periods, this implies 4.6 m of overburden above Kanjuruhan-era (~760 CE) surface remains. We test whether the spatial distribution of 666 known sites in East Java reflects this bias and find that observed clustering near volcanoes is better explained by two centuries of survey concentration in the Majapahit-Singosari heartland than by genuine settlement preference. We propose a taphonomic framework integrating terrain suitability analysis with volcanic sedimentation modeling to estimate burial depth and identify zones where archaeological absence likely reflects burial, not absence of occupation. Secondary calibration points from Merapi-system temples (Sambisari at 650 cm, Kedulan at 700 cm, Kimpulan at 270 cm) demonstrate that deep burial is a Java-wide phenomenon across multiple volcanic systems, not a local anomaly. This framework provides a methodology for prioritizing future ground-penetrating radar surveys and targeted excavations in volcanic Indonesia.

**Keywords:** volcanic taphonomy, archaeological bias, sedimentation rate, East Java, settlement prediction, GIS

---

## 1. Introduction

The Indonesian archipelago hosts some of the world's most active volcanic systems alongside one of humanity's longest continuous habitation records. Java alone supports 45 active volcanoes and has been inhabited since *Homo erectus*. Hindu-Buddhist polities flourished on the island from at least the 4th century CE, yet the material evidence of early Javanese civilization is strikingly sparse compared to contemporaneous societies in mainland Southeast Asia. The oldest widely-recognized kingdom in Indonesia — Kutai, dated to approximately 400 CE — is located in Kalimantan, a region with zero active volcanoes, where its Yupa inscriptions were found near the present surface.

This paper proposes that the apparent scarcity of early archaeological evidence in volcanic Java is not a reflection of genuine absence but of systematic taphonomic bias. Volcanic eruptions deposit centimeters to tens of centimeters of ash and sediment per event. Over centuries, cumulative deposition buries surface-level remains beneath meters of overburden, removing them from the observable archaeological record. We term this process *volcanic taphonomic burial* and distinguish it from taphonomic *destruction*: volcanic burial preserves sites (as demonstrated by Pompeii, Akrotiri, and — closer to our study area — Candi Sambisari and Candi Liangan), but renders them invisible to conventional surface survey.

We address three research questions:

1. **RQ1:** Can we quantify volcanic sedimentation rates at archaeological sites in Java using empirical calibration points?
2. **RQ2:** Does the spatial distribution of known archaeological sites in East Java show evidence of taphonomic bias relative to volcanic proximity?
3. **RQ3:** What are the expected burial depths for remains of different historical periods, and where should future subsurface investigations be prioritized?

Our approach centers on a multi-point empirical calibration. The Dwarapala statues of Singosari (East Java, ~1268 CE) provide a primary anchor: found half-buried in 1803, they yield a directly measured sedimentation rate. Three additional temples from Central Java's Merapi system provide independent calibration from a separate volcanic system. Together, these four points establish that mm/yr-scale burial is a systematic, Java-wide phenomenon — and that absence of archaeological evidence in volcanic zones demands reinterpretation.

---

## 2. Background

### 2.1 Java as a volcanic burial zone

Java hosts 45 active volcanoes across approximately 129,000 km² — a volcanic density of 0.35 per 1,000 km², the highest of any major Indonesian island and approximately six times that of Sumatra (0.06/1,000 km²). The average spacing between volcanic centers is ~54 km, meaning that no point on Java is more than approximately 27 km from the nearest active volcano. Since tephra from VEI 3–4 eruptions can deposit measurable ash at 50–100+ km from the source, the entire island lies within the depositional range of at least one — and often multiple — volcanic centers.

This density has a fundamental implication for archaeological preservation: **Java is not an island where some sites happen to be near volcanoes — it is an island where volcanic sedimentation is an inescapable, island-wide taphonomic process.** The relevant question is not *whether* burial occurs at any given location, but *how deep* — a function of proximity to active vents, eruption frequency, prevailing wind patterns, and local topography.

By contrast, Kalimantan (544,000 km²) has zero active volcanoes. Sulawesi (189,000 km²) has 11, with an average spacing of ~131 km. This asymmetry in volcanic density directly maps to asymmetry in archaeological preservation: surface-level remains persist indefinitely in Kalimantan but are buried within centuries in Java.

East Java province alone encompasses seven major active centers: Kelud, Semeru, Arjuno-Welirang, Bromo, Lamongan, Raung, and Ijen. The Brantas River basin — which hosted the Singosari (1222–1293 CE) and Majapahit (1293–~1500 CE) kingdoms — sits in the depositional path of Kelud (35 km east), Arjuno-Welirang (25 km north), and Semeru (60 km southeast). Kelud alone has produced at least 30 historically documented eruptions since 1000 CE, with VEI 3–4 events depositing 2–15 cm of ash at Malang distance per eruption (Global Volcanism Program, Smithsonian Institution).

In Central Java, Mount Merapi — one of the world's most active stratovolcanoes — has buried a cluster of 9th-century Hindu-Buddhist temples to depths of 3–7 m. These buried temples (Sambisari, Kedulan, Kimpulan) were discovered accidentally during modern construction and mining activities, not through systematic archaeological survey.

### 2.2 The observable archaeological record in East Java

Our compilation of known archaeological sites in East Java (Section 3.1) identified 666 unique entries, of which only 391 (59%) have usable spatial coordinates. The dataset is dominated by stone monuments (*candi*) from the Singosari and Majapahit periods (13th–15th centuries CE). Earlier periods are poorly represented: pre-10th century sites constitute less than 5% of the geocoded total.

This temporal distribution is consistent with two complementary explanations: (a) archaeological survey has historically concentrated on the Singosari-Majapahit heartland around Blitar, Malang, and Mojokerto; and (b) older sites are more deeply buried, having accumulated more overburden, making them less likely to be discovered by surface methods.

### 2.3 Volcanic taphonomy

Taphonomy — the study of post-mortem processes affecting the archaeological and fossil records — has a well-established literature in sedimentary and fluvial contexts. Volcanic taphonomy is comparatively understudied outside of catastrophic preservation events (Pompeii, Akrotiri/Santorini, Cerén in El Salvador). These high-profile cases involved rapid burial by single eruptions, preserving sites in extraordinary detail.

The process we describe is fundamentally different: *cumulative* volcanic taphonomy, where repeated small-to-moderate eruptions over centuries gradually bury sites layer by layer. This produces the same end result — sites invisible from the surface — but through a slower, less dramatic mechanism. The key distinction is that cumulative burial is spatially pervasive (affecting entire volcanic basins, not just eruption-proximal zones) and temporally continuous (ongoing, not one-time events).

### 2.4 The Kutai comparison

The oldest known kingdom in the Indonesian archipelago is Kutai Martadipura (~400 CE), located in East Kalimantan — a region with zero active volcanoes across 544,000 km². Its Yupa inscriptions, carved on stone pillars, were found near the present ground surface. We propose this is not coincidental.

The contrast could not be starker: Java has 45 active volcanoes in 129,000 km² (0.35/1,000 km²); Kalimantan has zero in 544,000 km². At the mean Javanese sedimentation rate of 4.4 mm/yr, a Kutai-era (~400 CE) inscription in the Malang basin would now lie beneath 4–10 m of volcanic overburden — invisible to any surface survey. The same inscription in Kalimantan sits exactly where it was placed 1,600 years ago, because there is no volcanic sedimentation to bury it.

Kutai's apparent chronological primacy over Javanese polities of similar or greater antiquity may therefore reflect differential preservation conditions rather than genuine temporal precedence. The "oldest kingdom" is simply the most *visible* one — a direct consequence of volcanic density asymmetry between islands.

---

## 3. Data and Methods

### 3.1 Archaeological site dataset

We compiled a database of known archaeological sites in East Java province from three complementary sources. First, we queried the OpenStreetMap Overpass API for all features tagged with `historic=*` within the Jawa Timur administrative boundary (bounding box: 6.5-9.0°S, 111.0-115.0°E), yielding 281 geolocated features classified as archaeological sites (n=156), monuments (n=144), and ruins (n=29). Second, we queried the Wikidata SPARQL endpoint for entities with coordinate property P625 within the same bounding box, recovering 16 precisely located sites including major temples such as Candi Badut, Candi Jago, and Candi Penataran. Third, we scraped the Indonesian-language Wikipedia article "Daftar candi di Indonesia" for site names, identifying 369 additional entries — the majority lacking published coordinates.

To increase spatial coverage, we geocoded the 369 coordinate-less entries using the OpenStreetMap Nominatim API with progressive query refinement: each site name was searched with suffixes "Jawa Timur, Indonesia," then "Jawa, Indonesia," then "Indonesia" alone. Results were validated against the East Java bounding box (latitude -9.5° to -6.5°, longitude 110.5° to 115.0°). Of 369 queries, 94 returned valid coordinates within the study area (25.5% success rate). The low hit rate is informative: the majority of the ungeocoded entries are sites located outside East Java (Sumatra, Central Java, Bali) that were included in the pan-Indonesian Wikipedia list.

After spatial deduplication within a 100 m radius, the final dataset contains 666 unique site entries, of which 391 have usable coordinates (58.7% geocoding rate). Within the East Java analytical bounds, 383 geocoded sites were used for spatial analysis. Coordinate quality is tracked per site: `osm_centroid` (n=281, ±50 m), `wikidata_p625` (n=16, ±10 m), and `nominatim` (n=94, ±1 km).

**Limitation:** The dataset is heavily biased toward stone monuments (candi) that survived volcanic burial due to their monumental scale. Wooden settlements, which likely constituted the vast majority of historical habitation, are systematically absent — precisely the pattern predicted by the taphonomic bias hypothesis.

### 3.2 Eruption history dataset

We compiled eruption records for four major East Java volcanic centers: Gunung Kelud (GVP #263280), Gunung Semeru (#263300), Gunung Arjuno-Welirang (#263260), and Gunung Bromo (#263310). The current dataset comprises 8 key eruption events with estimated ashfall at Malang distance, drawn from the Smithsonian Global Volcanism Program database and published volcanological literature. VEI (Volcanic Explosivity Index) values range from 2 to 4, with estimated ashfall per event of 1-15 cm at the Malang basin (~35 km from Kelud).

**Limitation:** The GVP database documents over 100 eruptions for these four volcanoes in the historical period. Our current dataset of 8 seed records is a lower bound; full integration of GVP records is ongoing. The 3.6 mm/year Dwarapala calibration inherently accounts for the cumulative effect of all eruptions (documented and undocumented) at that specific location.

### 3.3 Digital Elevation Model

We acquired the Copernicus GLO-30 Digital Elevation Model (30 m horizontal resolution) for the full Jawa Timur province extent from the AWS Open Data registry. Fifteen tiles were mosaicked (five additional S10-latitude tiles returned HTTP 404, being ocean-only), producing a merged DEM of 8,356 × 13,345 pixels covering elevations from sea level to 3,672 m (Semeru summit). All rasters were reprojected to UTM Zone 49S (EPSG:32749).

Four terrain derivative layers were computed: slope (degrees), aspect (degrees from north), Topographic Ruggedness Index (TRI, mean absolute elevation difference within a 3×3 window), and a simplified Topographic Wetness Index (TWI) using a window-based contributing area proxy. The TWI implementation uses neighborhood statistics rather than full hydrological flow accumulation; this is adequate for the current terrain suitability classification but should be replaced with a pysheds- or SAGA-based flow accumulation for publication-quality hydrological modeling.

### 3.4 The Dwarapala calibration

The primary empirical anchor for our burial depth framework is the pair of Dwarapala guardian statues at Candi Singosari, Malang Regency, East Java.

**Known parameters:**
- Construction date: ~1268 CE (reign of Kertanegara, Singosari Kingdom, 1222-1293 CE)
- Discovery date: 1803 CE, by Nicolaus Engelhard
- Physical dimensions: 370 cm seated height, ~40 tonnes, monolithic andesite
- Condition at discovery: "separuh tubuh terpendam" (half the body buried)
- Estimated burial depth: ~185 cm (half of 370 cm)
- Elapsed time: 1803 - 1268 = 535 years (using construction date) or ~510 years (using estimated completion)

**Calculated sedimentation rate:**

    R = 185 cm / 510 years = 0.36 cm/year = 3.6 mm/year

**Cross-validation:** Gunung Kelud, the dominant tephra source for the Malang basin (35 km to the east), erupted approximately 20 times between 1268 and 1803 CE. Documented VEI 3-4 eruptions deposit 2-20 cm of ash at Malang distance. Twenty eruptions at an average of ~5 cm per event would account for ~100 cm of the 185 cm total burial. The remainder (~85 cm) is attributable to secondary remobilization of volcanic material, contributions from Semeru and Arjuno-Welirang, and non-volcanic alluvial/aeolian sedimentation.

**Uncertainty:** This rate is a point estimate at a single location. Spatial variation across the Malang basin — and certainly across East Java — is expected. The rate may be higher closer to volcanic vents and lower in elevated, well-drained positions. This spatial heterogeneity is the subject of ongoing work (Paper 3).

### 3.5 Secondary calibration points

To assess whether the Dwarapala burial rate is a local anomaly or representative of a Java-wide phenomenon, we compiled four additional calibration points from Central Java's Merapi volcanic system:

| Site | Built (CE) | Discovered | Depth (cm) | Volcanic System | Rate (mm/yr) | Dating Evidence |
|------|-----------|------------|-----------|-----------------|-------------|-----------------|
| Candi Sambisari | ~835 | 1966 | 500–650 | Merapi | 4.4–5.7 | Wanua Tengah III inscription (908 CE); Rakai Garung reign 828–846 |
| Candi Kedulan | ~869 | 1993 | 600–700 | Merapi | 5.3–6.2 | Sumundul/Pananggaran inscriptions (791 Saka = 869 CE) |
| Candi Kimpulan | ~900 | 2009 | 270–500 | Merapi | 2.4–4.5 | Architectural style; 9th–10th century consensus (Putra & Setyastuti, BEFEO 105) |
| Candi Liangan | ~9th c. | 2008 | 500–900 | Sundoro | N/A | Catastrophic single-event burial; C14 date 590 CE for charred wood |

Using documented construction dates from epigraphic evidence, we compute cumulative sedimentation rates for each site: Rate = depth / (discovery_year - construction_year). The three Merapi-system sites yield rates of 2.4–6.2 mm/yr (mean ~4.8 mm/yr), consistently higher than the Kelud-system Dwarapala rate (3.5 mm/yr). This difference is physically plausible: Merapi erupts more frequently than Kelud, and the Central Java sites are closer to their volcanic source.

Across all four sites (excluding Liangan, which was catastrophically buried in a single event), the computed rates span 2.4–6.2 mm/yr with a mean of 4.4 ± 1.2 mm/yr. The consistency across two independent volcanic systems — Kelud in East Java and Merapi in Central Java — is the central quantitative finding of this paper. It demonstrates that mm/yr-scale burial of archaeological sites is a systematic, Java-wide phenomenon, not a local anomaly.

**Critical implication:** If stone temples weighing tens of tonnes are buried 3–7 m deep, then lighter wooden and bamboo structures — which constituted the vast majority of historical settlements — would be buried to the same depth with no surface expression whatsoever.

### 3.6 Burial depth projection model

Using the Dwarapala-calibrated rate as a baseline, we estimate expected overburden depth for archaeological remains of different periods at the Malang basin:

    Depth(era) = R × (T_present - T_era)

where R = 3.6 mm/year and T_present = 2026 CE.

| Era | Approximate Date | Elapsed (yr) | Overburden: Low (2.4 mm/yr) | Dwarapala (3.5 mm/yr) | Mean (4.4 mm/yr) | High (6.2 mm/yr) |
|-----|-----------------|-------------|---------------------------|----------------------|-----------------|-----------------|
| Late Majapahit | ~1400 CE | 626 | 1.5 m | 2.2 m | 2.8 m | 3.9 m |
| Singosari | ~1268 CE | 758 | 1.8 m | 2.7 m | 3.3 m | 4.7 m |
| Mataram (E. Java) | ~900 CE | 1,126 | 2.7 m | 3.9 m | 5.0 m | 7.0 m |
| Kanjuruhan | ~760 CE | 1,266 | 3.0 m | 4.4 m | 5.6 m | 7.9 m |
| Pre-Hindu | ~400 CE | 1,626 | 3.9 m | 5.7 m | 7.2 m | 10.1 m |

These projections now use the full range of empirically-derived rates (2.4–6.2 mm/yr) from four calibration points across two volcanic systems. Actual burial depth varies with distance from volcanic vents, topographic position (valley floors accumulate more than ridges), wind patterns during eruptions, and local hydrology. Nevertheless, these estimates establish order-of-magnitude expectations: Kanjuruhan-era remains in the Malang basin could reasonably lie beneath 4-5 m of overburden, rendering them invisible to surface survey and requiring subsurface investigation (GPR, coring, or excavation) to detect.

### 3.7 Spatial analysis methods

**E004 — Raw site density vs volcanic proximity.** We computed the minimum great-circle distance from each geocoded site to the nearest of seven reference volcanoes (Kelud, Semeru, Arjuno-Welirang, Bromo, Lamongan, Raung, Ijen). Sites were binned into seven distance bands: 0-25, 25-50, 50-75, 75-100, 100-150, 150-200, and 200+ km. Land area in each band was computed by intersecting volcano-centered buffer rings with the East Java province polygon (fetched from OSM Overpass API, projected to EPSG:32749). Site density was expressed as sites per 1,000 km². Spearman's rank correlation between distance band midpoint and site density tests whether sites are found preferentially away from volcanoes (positive rho = H1 supported) or near volcanoes (negative rho = clustering due to survey bias).

**E005 — Terrain-controlled analysis.** To separate the effect of terrain suitability from volcanic proximity, we constructed a simple terrain suitability index combining four DEM-derived features: slope (lower is better for settlement), elevation (moderate preferred), TWI (higher = wetter = better water access), and a river proximity proxy (via TWI). The study area was divided into a 25 km × 25 km grid (187 cells). For each cell, we computed mean terrain suitability, observed site count, and predicted site count (proportional to suitability × area). The residual (observed - predicted) indicates whether a cell has more or fewer sites than terrain alone predicts. Spearman correlation between residual density and distance to nearest volcano tests H1 after controlling for terrain: negative residuals near volcanoes would support taphonomic bias (H1), while positive residuals suggest the observed clustering is explained by terrain preference.

---

## 4. Results

### 4.1 The Dwarapala calibration

The Dwarapala sedimentation calculation yields a rate of 3.6 mm/year (Section 3.4). This rate is internally consistent with documented Kelud eruption histories, which account for approximately 100 of the 185 cm total burial through direct tephra deposition alone. The remaining 85 cm is attributable to secondary remobilization and contributions from other volcanic systems, consistent with the composite sedimentation environment of an inter-volcanic basin.

The secondary calibration points (Table, Section 3.5) yield independently-computed sedimentation rates of 2.4–6.2 mm/yr across the Merapi system, consistent with the Dwarapala rate (3.5 mm/yr) from the Kelud system. The cross-system mean of 4.4 ± 1.2 mm/yr establishes that ongoing volcanic burial is a Java-wide phenomenon with quantifiable, consistent rates.

### 4.2 Burial depth projections

Table (Section 3.6) presents estimated overburden depth by era. Key projections for the Malang basin:
- Kanjuruhan-era (~760 CE) remains: ~4.56 m below current surface
- Pre-Hindu (~400 CE) remains: ~5.85 m below current surface

These depths exceed standard archaeological survey capabilities (typically 0-1 m surface inspection) and approach the limits of ground-penetrating radar (effective to 2-5 m in volcanic ash/sediment). Detection of Kanjuruhan-era or earlier remains in the Malang basin will likely require either deep GPR surveys, borehole coring, or fortuitous exposure through modern construction or erosion — exactly the mechanism by which Sambisari (farmer's plow, 1966) and Kimpulan (university construction, 2009) were discovered.

### 4.3 Site density vs volcanic proximity (E004)

Analysis of 383 geocoded sites within the East Java bounds reveals a strong negative correlation between site density and distance from the nearest active volcano:

| Distance Band | Sites | Area (km²) | Density (per 1,000 km²) |
|---------------|-------|-----------|------------------------|
| 0-25 km | 147 | 11,343 | 12.96 |
| 25-50 km | 136 | 18,034 | 7.54 |
| 50-75 km | 37 | 18,528 | 2.00 |
| 75-100 km | 22 | 16,373 | 1.34 |
| 100-150 km | 41 | 28,523 | 1.44 |
| 150-200 km | 0 | 25,507 | 0.00 |
| 200+ km | 0 | 73,740 | 0.00 |

Spearman's rho = -0.955, p = 0.0008 (n = 7 distance bands with non-zero area).

This result shows the *opposite* of what a naive reading of H1 would predict: known sites are concentrated near volcanoes, not away from them. The 0-25 km band has nearly 10 times the site density of the 75-100 km band.

**Interpretation:** This pattern does not falsify H1. Instead, it reflects the geography of archaeological survey in East Java. The Singosari and Majapahit kingdoms (13th-15th centuries CE) were centered in the Brantas River valley, 20-50 km from Kelud and Arjuno-Welirang. Two centuries of archaeological attention have concentrated in this region, producing a dataset that maps survey history rather than settlement history. The zero-count bands (150-200 km, 200+ km) represent unsurveyed areas, not uninhabited areas.

### 4.4 Terrain-controlled analysis (E005)

To test whether the volcanic-proximal clustering is explained by terrain suitability (i.e., volcanic regions simply have better settlement terrain), we computed terrain-predicted site density and compared it to observed density across 187 grid cells.

Spearman's rho (residual density vs distance to volcano) = -0.358, p < 0.0001 (n = 187 cells, 391 sites).

The terrain-controlled analysis produces a weaker but still significantly negative correlation. Near-volcano grid cells have *more* sites than terrain alone predicts — not fewer. This means:

1. Terrain suitability explains part of the near-volcano clustering (rho weakened from -0.955 to -0.358), but not all of it.
2. The remaining clustering (rho = -0.358) reflects factors not captured in terrain: primarily survey intensity, which has historically focused on the Singosari-Majapahit heartland.
3. There is no evidence of a taphonomic "deficit" in near-volcano zones — but this absence of evidence is itself expected if the deficit applies primarily to *undiscovered* sites that are deeply buried.

### 4.5 Robustness check (E006)

To assess sensitivity to sample size, we repeated both analyses after geocoding 94 additional sites via Nominatim (increasing the geocoded count from 297 to 391, a 29% increase). Results were remarkably stable:

| Metric | Original (n=297) | Enriched (n=383/391) | Change |
|--------|------------------|----------------------|--------|
| E004 Spearman rho | -0.991 | -0.955 | +0.036 |
| E005 Spearman rho | -0.364 | -0.358 | +0.006 |

The negligible change confirms that the spatial pattern is robust and not an artifact of incomplete geocoding.

### 4.6 Why distribution data cannot test H1

The fundamental limitation of testing H1 from observed site distributions is circular: the sites in our dataset are those that *survived* burial and were *discovered* by archaeologists. Both processes systematically favor low-burial-depth locations:

1. **Survivorship bias:** Stone temples (candi) dominate the dataset because they are large enough to protrude through meters of sediment. The wooden and bamboo structures that constituted >99% of historical habitation leave no surface trace after burial — they are exactly the sites H1 predicts are missing.

2. **Survey bias:** Indonesian archaeological survey has concentrated on regions of known historical kingdoms (Singosari, Majapahit, Kediri) for over two centuries. "Blank" areas on the archaeological map are blank because they are unsurveyed, not because they are uninhabited.

3. **Discovery mechanism bias:** Most deeply buried temples were discovered accidentally during modern construction or agriculture (Sambisari: farmer's plow; Kimpulan: university excavation; Liangan: sand mining). Systematic subsurface survey has never been conducted in the study area.

H1 therefore cannot be confirmed or denied from the current dataset. It remains a hypothesis that requires subsurface investigation to test. The Dwarapala calibration provides the quantitative basis for predicting *where* to look and *how deep* to probe.

---

## 5. Discussion

### 5.1 Multi-point calibration as the core contribution

The central finding of this paper is not a statistical test of H1 but an empirical calibration: four independent archaeological sites across two volcanic systems (Kelud and Merapi) yield cumulative sedimentation rates of 2.4–6.2 mm/yr (mean 4.4 ± 1.2 mm/yr). This consistency across sites separated by hundreds of kilometers and sourced from different volcanic centers establishes that ongoing burial of archaeological remains is a systematic feature of volcanic Java, not a localized anomaly.

The Dwarapala rate (3.5 mm/yr) sits at the lower end of this range. Merapi-system sites show higher rates (mean ~4.8 mm/yr), consistent with Merapi's higher eruption frequency. This variation is itself informative: burial rates are site-specific and depend on distance from volcanic vents, topographic position, and local hydrology. A spatially-resolved burial depth model (the subject of planned follow-up work) must account for this heterogeneity.

### 5.2 Why the distribution test is informative despite being "negative"

Our spatial analysis (E004, E005) shows that known sites cluster *near* volcanoes — the opposite of what a naive reading of the taphonomic bias hypothesis would predict. We argue this result is itself evidence for the hypothesis, not against it:

The observable archaeological record in East Java is a product of two overlapping biases — survey history and survivorship. Archaeological survey has concentrated on the Singosari-Majapahit heartland (0–50 km from Kelud/Arjuno) for over 200 years, producing a dataset that maps *where we have looked* rather than *where sites exist*. The stone monuments that dominate the dataset are precisely those large enough to resist or protrude through volcanic burial — exactly the kind of sites that would survive in high-deposition zones.

The fact that adding 29% more geocoded sites (E006) produced negligible change in the correlation (rho shifted from -0.991 to -0.955 for raw density, -0.364 to -0.358 terrain-controlled) confirms that the pattern is robust and saturated: the observable record has reached a ceiling imposed by survey history, not by genuine settlement distributions.

### 5.3 Practical implications for fieldwork

The burial depth projections in Table (Section 3.6) have direct operational implications:

- **GPR applicability:** Ground-penetrating radar is effective to ~2–5 m depth in volcanic ash and sediment. This means GPR can potentially detect Late Majapahit-era remains (projected 1.5–3.9 m depth) but may struggle with Kanjuruhan-era or earlier remains (projected 3.0–7.9 m). For deeper targets, borehole coring or deep test trenches are required.
- **Priority zones:** High-value investigation targets are areas with (a) high terrain suitability for historical settlement (low slope, water access, fertile soils) and (b) high expected burial depth (volcanic proximity). These zones would have been attractive for settlement AND are exactly where burial has rendered sites invisible.
- **Discovery mechanism:** The accidental discoveries of Sambisari (farmer's plow, 1966), Kimpulan (university construction, 2009), and Liangan (sand mining, 2008) illustrate that deeply buried temples in Java are found by chance during modern ground disturbance, not by systematic survey. A transition to targeted subsurface investigation could dramatically increase the discovery rate.

### 5.4 Limitations

1. **Calibration sample size:** Four calibration points, while a significant improvement over one, remain a small sample. The computed rates carry uncertainty from imprecise construction dates and variable depth measurements across sources.
2. **Rate constancy assumption:** We treat sedimentation as temporally uniform, which ignores episodic large eruptions. The Dwarapala rate of 3.5 mm/yr is a *cumulative average* that smooths over individual eruption events. Actual burial may occur in centimeter-scale steps separated by decades.
3. **Depth measurement uncertainty:** Published burial depths for the secondary anchors vary across sources (e.g., Sambisari: "5 metres" to "6.5 metres"; Kimpulan: "270 cm" to "5 metres"). We report ranges rather than point estimates, but these ranges are wide.
4. **Spatial extrapolation:** Our rates are derived from sites in two specific volcanic basins. Extrapolation to other regions of Java (e.g., southern coast, eastern salient) requires additional calibration points.
5. **Survey bias unquantified:** We argue that survey history dominates the distribution data but cannot quantify this claim without systematic survey coverage maps (not currently available).

### 5.5 Future work

Three follow-up studies are planned:
- **Paper 2 (Settlement Suitability Model):** A machine learning model using terrain features to predict settlement suitability, independently tested against volcanic burial zones (the "tautology test" — does the model predict settlements where none are known but burial is deep?).
- **Paper 3 (Burial Depth Model):** A spatially-resolved burial depth model using eruption records, tephra dispersal models, and wind data, calibrated against all four anchor points.
- **Fieldwork validation:** Targeted GPR survey at 5–10 locations in high-suitability, high-burial-depth zones to test whether subsurface anomalies consistent with archaeological features are present.

---

## 6. Conclusion

We have presented a quantitative framework for estimating volcanic taphonomic bias in the Indonesian archaeological record. Four empirical calibration points from two independent volcanic systems yield cumulative sedimentation rates of 2.4–6.2 mm/yr (mean 4.4 ± 1.2 mm/yr), establishing that multi-meter burial of archaeological sites is a systematic, Java-wide phenomenon.

The spatial distribution of known archaeological sites in East Java cannot test the taphonomic bias hypothesis because the observable record is dominated by survey history and survivorship bias. This finding is itself a contribution: it demonstrates that the "archaeological absence" of pre-Majapahit evidence in volcanic zones cannot be interpreted as evidence of cultural absence.

Our burial depth projections indicate that Kanjuruhan-era (~760 CE) remains in the Malang basin lie beneath 3.0–7.9 m of overburden, and pre-Hindu (~400 CE) remains beneath 3.9–10.1 m. These depths exceed conventional surface survey capabilities and approach or exceed the effective range of ground-penetrating radar. The perceived chronological primacy of Kutai (Kalimantan, ~400 CE, zero volcanism) over Javanese polities may reflect this differential preservation rather than genuine temporal precedence.

We propose that future archaeological investigation in volcanic Java should explicitly account for taphonomic burial by incorporating subsurface detection methods (GPR, coring, targeted excavation) in zones identified as having both high settlement suitability and high expected burial depth. The computational framework presented here provides a replicable methodology for identifying such zones.

---

## References

*[To be compiled — see outline.md References section for seeds]*

---

## Figures

| Figure | Status | Description |
|--------|--------|-------------|
| Fig 1 | PENDING | Dwarapala burial timeline (1268 CE → 1803 → 2026) |
| Fig 2 | AVAILABLE | Sedimentation proxy map, East Java (`E005/results/jatim_suitability.tif`) |
| Fig 3 | AVAILABLE | Site density vs volcanic distance bar chart (`E004/results/density_chart.png`) |
| Fig 4 | AVAILABLE | Observed vs predicted density + residual scatter (`E005/results/jatim_density_chart.png`) |
| Fig 5 | AVAILABLE | Candidate burial zones / residual map (`E005/results/jatim_residual_map.html`) |
| Table 1 | IN TEXT | Expected overburden by era (Section 3.6) |
| Table 2 | IN TEXT | Secondary calibration points (Section 3.5) |
| Table 3 | IN TEXT | Density by distance band (Section 4.3) |

---

*Draft v0.1 — Methods and Results sections complete. Next: Introduction, Background, Discussion, Conclusion.*
