# Provenance Notes — E018 TOM Data

All dates are in years Before Present (BP, where Present = 1950 CE).

## Provenance Classes
- **A_published**: Value directly from a published peer-reviewed paper
- **B_derived**: Value calculated from published data (method documented)
- **C_estimated**: Expert estimate based on indirect evidence (flagged)

---

## Linguistic Ages (L_age)

All linguistic dates derive from the Bayesian phylolinguistic analysis of Austronesian languages by Gray, Drummond & Greenhill (2009), "Language Phylogenies Reveal Expansion Pulses and Pauses in Pacific Settlement," *Science* 323:479-483.

Gray et al. used a relaxed molecular clock applied to cognate data from 400 Austronesian languages (ABVD database), calibrated against historical and archaeological dates. We use node dates corresponding to the estimated arrival/diversification of language subgroups in each region.

| Region | L_age | Source node | Notes |
|--------|-------|-------------|-------|
| Java | 3500 BP | Javanese-Madurese split | Median posterior estimate |
| Sumatra | 3400 BP | Malay-Chamic clade | Includes Sumatran Malay diversification |
| Sulawesi | 3800 BP | South Sulawesi subgroup | Bugis-Makassarese ancestor |
| Nusa Tenggara | 3600 BP | Bima-Sumba clade | Flores-Timor branch |
| Philippines | 4000 BP | Malayo-Polynesian root | Cross-checked with Blust 1995 reconstruction |
| Maluku | 3500 BP | Central-Eastern MP clade | Higher uncertainty |
| Kalimantan | 3600 BP | Land Dayak / Barito node | Borneo diversification |
| Madagascar | 1300 BP | Malagasy split from SE Borneo | Adelaar 2009 Lingua 119:1707-1727 cross-check |

## Genetic Ages (G_age)

Genetic dates come from mitochondrial DNA (mtDNA) and Y-chromosome haplogroup coalescence/divergence analyses. Primary sources:

- **Hill et al. (2007)** "A Mitochondrial Stratigraphy for Island Southeast Asia." *Mol Biol Evol* 24:871-882.
- **Tumonggor et al. (2013)** "The Indonesian archipelago: an ancient genetic highway linking Asia and the Pacific." *Eur J Hum Genet* 21:824-831.
- **Lipson et al. (2014)** "Reconstructing Austronesian population history in Island Southeast Asia." *Nat Commun* 5:3513.
- **Pierron et al. (2017)** "Genomic landscape of human diversity across Madagascar." *Proc Natl Acad Sci* 114:E2341-E2350.

These studies track mtDNA haplogroup B4a1a (the "Polynesian motif" precursor), along with M7c3c, E lineages, and Y-chromosome O-M175 subclades. Coalescence dates represent the estimated time of population expansion into each region.

| Region | G_age | Dominant haplogroup tracked | Uncertainty |
|--------|-------|-----------------------------|-------------|
| Java | 4000 BP | B4a1a, M7c3c | ±1000 yr (coalescence uncertainty) |
| Sumatra | 4500 BP | M7c3c (Batak), B4a1a | Hill et al. 2007 Batak deep coalescence |
| Sulawesi | 4000 BP | B4a1a (South Sulawesi) | Cox 2008 cross-check |
| Nusa Tenggara | 4000 BP | B4a1a (Flores-Timor) | Lansing et al. 2011 |
| Philippines | 4500 BP | E, B4a1a | Delfin et al. 2011 Philippines-specific |
| Maluku | 3500 BP | B4a1a (Moluccan) | Lower signal, higher uncertainty |
| Kalimantan | 4000 BP | B4a1a (Land Dayak) | Jinam et al. 2012 |
| Madagascar | 1500 BP | B4a1a1b (Malagasy) | Pierron et al. 2017 aDNA confirmation |

**Important note:** Genetic coalescence dates are inherently broader (±1000 yr is typical) than C14 dates. This asymmetric uncertainty is accounted for in the sensitivity analysis.

## Archaeological Ages (A_age)

Archaeological dates represent the oldest **reliable C14 dates** for human presence in each region. For ISEA regions, we focus on Neolithic (Austronesian-associated) dates rather than pre-Neolithic (Homo erectus or early Homo sapiens) dates, since the TOM hypothesis concerns the same population traced by linguistic and genetic data.

**Exception: Kalimantan** — Niah Cave's 40,000 BP date represents continuous modern human occupation, providing a true negative control (deep record preserved due to zero volcanism).

### Run 1 (Neolithic-only, original)

| Region | A_age | Key site | Source |
|--------|-------|----------|--------|
| Java | 3500 BP | Kalumpang-type pottery | Simanjuntak 2002; Noerwidi 2017 |
| Sumatra | 3500 BP | Bukit Arat | Forestier et al. 2005; Bellwood 2017 |
| Sulawesi | 3500 BP | Minanga Sipakko | Anggraeni et al. 2014 Antiquity 88:740 |
| Nusa Tenggara | 3500 BP | Lewoleba cave | Bellwood et al. 1998; Brumm et al. 2006 |
| Philippines | 4000 BP | Nagsabaran | Hung et al. 2011; Bellwood & Dizon 2013 |
| Maluku | 3500 BP | Uattamdi cave | Bellwood 1998; Spriggs 1998 |
| Kalimantan | 40000 BP | Niah Cave | Barker et al. 2007 J Hum Evol 52:243 |
| Madagascar | 1200 BP | Lakaton'i Anja | Crowther et al. 2016 Azania 51:517 |

### Run 2 (Deep-time, oldest H. sapiens evidence)

| Region | A_age | Key site | Dating method | Source |
|--------|-------|----------|---------------|--------|
| Java | 60,000 BP | Song Terus (tooth ST04) | ESR/U-series | Semah et al. 2023 L'Anthropologie; <60 ka terminus ante quem |
| Sumatra | 68,000 BP | Lida Ajer (2 H. sapiens teeth) | Coupled U-series/ESR + luminescence | Westaway et al. 2017 Nature 548:322-325 |
| Sulawesi | 67,800 BP | Liang Metanduno (hand stencil) | LA-U-series on calcite | Oktaviana et al. 2026 Nature 650:652-656 |
| Nusa Tenggara | 44,600 BP | Laili cave, Timor-Leste | Radiocarbon calibrated | Hawkins et al. 2017 Quat Sci Rev 171:58-72 |
| Philippines | 47,000 BP | Tabon Cave (H. sapiens tibia) | Direct U-series on bone | Detroit et al. 2004 C.R. Palevol 3:705-712 |
| Maluku | 36,000 BP | Golo Cave, Gebe Island | Radiocarbon calibrated | Bellwood 1998 Spice Islands in Prehistory ANU Press |
| Kalimantan | 40,000 BP | Niah Cave ("Deep Skull") | AMS radiocarbon ABOX + U-series | Barker et al. 2007 J Hum Evol 52:243-261 |
| Madagascar | 10,500 BP | Christmas River (cut-marked bone) | Radiocarbon | Hansford et al. 2018 Science Advances 4:eaat6925 |

**Critical notes on deep-time dates:**
- **All dates come from CAVE sites.** This is a fundamental confound — caves are sheltered from tephra deposition, meaning they specifically survive the volcanic destruction mechanism that H-TOM predicts.
- **Callao Cave (Philippines, 67K BP)** was excluded because the hominin found there is *Homo luzonensis*, not *H. sapiens* (Detroit et al. 2019 Nature).
- **Sumatra's Lida Ajer** (68K) is in highland rainforest far from the volcanic axis.
- **Java's Song Terus** is in the Gunung Sewu karst of southern Java, away from the volcanic plains.
- **Sulawesi's Liang Metanduno** is on Muna Island, with minimal local volcanic activity.

## Taphonomic Pressure Components (TAP)

### Volcano counts (n_volcanoes_holocene)
Source: Smithsonian Global Volcanism Program (GVP), Holocene Volcano List v5.2.8.
Counts include all volcanoes with confirmed Holocene eruptions within each region's geographic boundary.

### VEI 3+ eruption counts (n_eruptions_vei3plus)
Source: GVP eruption database. Counts include all eruptions with assigned VEI ≥ 3 during the Holocene.

### Land area (area_km2)
Source: Standard geographic reference values.

### Shelf exposure fraction (shelf_exposure_frac)
Source: Voris (2000) "Maps of Pleistocene sea levels in Southeast Asia: shorelines, river systems and time durations." *J Biogeography* 27:1153-1167. Sathiamurthy & Voris (2006) update.

This represents the approximate fraction of each region's current area that was exposed as dry land during the Last Glacial Maximum (LGM, ~20 ka). Higher values = more land area lost to sea-level rise = more potential archaeological sites submerged.

| Region | shelf_frac | Rationale |
|--------|-----------|-----------|
| Java | 0.15 | Northern coast shelf, narrow |
| Sumatra | 0.20 | Strait of Malacca was dry land |
| Sulawesi | 0.10 | Narrow shelves, deep surrounding seas |
| Nusa Tenggara | 0.08 | Deep Lombok/Flores Strait, minimal shelf |
| Philippines | 0.12 | Palawan shelf notable, rest limited |
| Maluku | 0.05 | Almost no shelf (Wallacea) |
| Kalimantan | 0.35 | Extensive Sunda Shelf — massive area lost |
| Madagascar | 0.05 | Limited shelf on western coast |

---

## Known Limitations

1. **Neolithic vs. pre-Neolithic framing:** The L_age and G_age track Austronesian expansion (~4000-3000 BP), while archaeological preservation bias affects ALL time periods. The strongest TOM signal should appear when comparing deep-time archaeological records (>10,000 BP) rather than Neolithic-only.

2. **Java paradox:** Java's Neolithic archaeological age (~3500 BP) matches its linguistic age, suggesting no TOM effect for the Neolithic period. But Java's PRE-Neolithic record is anomalously shallow compared to Kalimantan/Philippines, which IS consistent with H-TOM. This experiment captures only the Neolithic comparison.

3. **Genetic date uncertainty:** mtDNA coalescence dates have wide confidence intervals (±1000 yr), making them the weakest "clock." The sensitivity analysis perturbs these dates to test robustness.

4. **Regional boundaries:** Treating Java/Sumatra/Sulawesi as uniform regions ignores within-region variation (e.g., East Java is much more volcanic than West Java). This is a POC simplification.

5. **n=8 limitation:** With only 8 data points, formal statistical significance (p<0.05) is nearly impossible. We rely on effect size, direction consistency, and sensitivity robustness.

6. **Cave-site survivorship bias (identified in Run 2):** All deep-time archaeological dates come from cave sites. Caves are specifically protected from tephra burial — the exact mechanism H-TOM proposes for destroying open-air sites. This means the deep-time oldest-date metric measures CAVE PRESERVATION, not volcanic destruction. The test is confounded at a fundamental level.

7. **The correct test for H-TOM:** Should examine site density per time period, open-air vs cave site ratios, spatial coverage gaps, and chronological continuity — not just the oldest single date per region.
