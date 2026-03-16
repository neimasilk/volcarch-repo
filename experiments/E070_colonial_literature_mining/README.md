# E070: Colonial Literature Mining — Independent Dataset Construction

**Status:** SUCCESS (DS-1 complete; DS-2–DS-6 pending)
**Date:** 2026-03-13
**Type:** DATASET CONSTRUCTION
**Papers:** P1, P2, P7, P11 (and revision ammo for all)

## Rationale

VOLCARCH's Mata Elang #6 structural critique identified that 67 experiments rely on ~4 core datasets (DHARMA 268 prasasti, 666 E.Java sites from modern sources, ABVD wordlists, 142 candi locations). The "11 independent channels" claim is weakened by dataset dependence.

**Solution:** Construct genuinely independent datasets from untapped colonial-era sources (1800s-1945). These predate modern archaeological biases, are in Dutch (barrier that prevented systematic mining), and contain primary observations unavailable in any digital database.

## Source Inventory

### Tier 1: CRITICAL (free, digitized, directly relevant)

| Source | URL | Content | Language | Status |
|--------|-----|---------|----------|--------|
| **KITLV/Leiden TBG** | kitlv-docs.library.leiden.edu | Vols 1-86 (1853-1958), core colonial scholarship | Dutch | FREE, PDF scans |
| **Internet Archive: Krom** | archive.org | *Barabudur* (1927), *Hindoe-Javaansche Kunst* (1920), *HJ Geschiedenis* | Dutch | FREE |
| **Internet Archive: Verbeek** | archive.org | *Krakatau* (1885), *Géologie de Java* (1896) | Dutch/French | FREE |
| **Internet Archive: Kern** | archive.org | *Verspreide Geschriften* vols 3-9 (epigraphy, linguistics) | Dutch | FREE |
| **Internet Archive: Brandes** | archive.org | *Oud-Javaansche Oorkonden* (1913) | Dutch | FREE |
| **OV (Oudheidkundig Verslag)** | archive.org + Leiden | Annual archaeological survey reports (1913-1950) | Dutch | FREE (partial) |
| **Persee: BEFEO** | persee.fr/collection/befeo | French scholarship (Coedès, Damais) 1901-2024 | French | FREE |
| **Djawa journal** | archive.org | Java Institute journal (1921-1941) | Dutch | FREE |
| **David Rumsey Maps** | davidrumsey.com | Verbeek geological map of Java (26 sheets) | — | FREE |

### Tier 2: HIGH VALUE (accessible with effort)

| Source | URL | Content | Language |
|--------|-----|---------|----------|
| **Delpher.nl** | delpher.nl | 130M pages colonial newspapers incl. Dutch East Indies | Dutch |
| **JSTOR/Brill BKI** | jstor.org / brill.com | BKI historical volumes (1853+) | Dutch/English |
| **UGM Langka** | langka.lib.ugm.ac.id | Rare colonial archaeological monographs | Dutch |
| **Wereldmuseum catalog** | collectie.wereldmuseum.nl | 450K objects, Javanese provenance data | Dutch/English |
| **Nationaal Archief** | nationaalarchief.nl | Memories van Overgave (colonial admin reports) | Dutch |

### Tier 3: SUPPLEMENTARY

| Source | Content |
|--------|---------|
| Wikimedia Commons KITLV donation | 5,574 historical photos (candi, excavations) |
| Gallica (BnF) | French colonial scientific works |
| Leiden OD photograph collection | 21,800 photos (15% digitized) |

## Datasets to Construct

### DS-1: Colonial Archaeological Site Register (targets P1, P7)

**Goal:** Extract site locations + burial depth + volcanic deposit observations from OV reports (1913-1950) and Krom/Stutterheim publications.

**Why independent:** These observations predate modern BPCB registry. Colonial archaeologists documented burial conditions at excavation time — data not in any modern database.

**Key data points:**
- Site name, location (can be georeferenced from Dutch maps)
- Discovery date and excavation date
- **Burial depth at time of discovery** (the gold standard for taphonomy)
- Volcanic deposit description (if mentioned)
- Material found (organic vs lithic vs metal)

**Method:**
1. Download OV volumes from Internet Archive (1913-1950, ~37 volumes)
2. AI-assisted Dutch OCR + extraction of site reports
3. Geocode using colonial maps + modern coordinates
4. Cross-reference with E001 modern database → identify sites IN colonial record but NOT in modern database (lost sites)

**Expected yield:** 200-500 site records with burial depth data (vs current 5 calibration points)

**Estimated effort:** 2-4 weeks for extraction, 1 week for geocoding and validation

---

### DS-2: Verbeek Volcanic Geology Map (targets P1, P2)

**Goal:** Digitize and georeference Verbeek & Fennema (1896) 26-sheet geological map of Java.

**Why independent:** Colonial geological mapping predates modern PVMBG data. Contains volcanic deposit extent not in modern digital datasets. Direct overlay with archaeological sites creates independent taphonomic test.

**Method:**
1. Download high-res scans from David Rumsey Collection
2. Georeference each sheet to modern CRS (EPSG:4326)
3. Digitize volcanic deposit polygons (lahar, pyroclastic flow, tuff)
4. Overlay with E001 site database

**Expected yield:** First GIS-ready volcanic deposit map of Java from pre-modern geological survey.

**Estimated effort:** 1-2 weeks for georeferencing, 1 week for digitization

---

### DS-3: Colonial Ethnographic Observations (targets P5, P9)

**Goal:** Extract pre-modern observations of Javanese/Balinese ritual, agriculture, and material culture from TBG, Djawa, and colonial monographs.

**Why independent:** Colonial ethnographers observed practices before 20th century standardization. Their descriptions of burial customs, agricultural rituals, and material culture provide a temporal baseline unavailable in modern ethnography.

**Key sources:**
- TBG articles on Javanese customs (1853-1958)
- Djawa journal articles on temple rituals, wayang, gamelan (1921-1941)
- Kern's collected works on epigraphy and linguistics
- Raffles (1817) *History of Java* — earliest comprehensive English account

**Method:**
1. AI-assisted reading of Dutch sources (Claude can read Dutch)
2. Structured extraction: ritual name, description, location, date observed, indigenous vs Indic elements
3. Cross-reference with P5 slametan analysis and E023 prasasti ritual corpus

**Expected yield:** 100-300 ethnographic observations predating modern sources

**Estimated effort:** 3-5 weeks (reading-intensive)

---

### DS-4: Museum Provenance Database (targets P1, P7)

**Goal:** Extract provenance data (find location, depth, material type, date) for Javanese archaeological objects in Dutch museum collections.

**Why independent:** Museum catalogs contain find context information that may not appear in published archaeological reports.

**Sources:**
- Wereldmuseum online catalog (450K objects)
- Rijksmuseum van Oudheden (now integrated into Wereldmuseum)
- Published catalogs: Bernet Kempers' inventory

**Method:**
1. Query Wereldmuseum API/catalog for "Java" + "archaeological"
2. Extract provenance fields: find location, excavation context, depth if recorded
3. Geocode find locations

**Expected yield:** 50-200 objects with usable provenance data

**Estimated effort:** 1-2 weeks

---

### DS-5: Colonial Newspaper Event Reports (targets P1, P5)

**Goal:** Extract contemporary accounts of volcanic eruptions and their effects on settlements/sites from colonial Dutch newspapers (Delpher.nl).

**Why independent:** Contemporary newspaper reports provide primary source evidence for volcanic burial events as they happened — not retroactive modeling.

**Key searches:** "Kelut" / "Kloet", "Merapi", "Smeroe" / "Semeru", eruption years + "begräbnis" / "bedolven" (buried)

**Method:**
1. Search Delpher.nl for volcanic event reports (Dutch East Indies newspapers)
2. Extract: date, volcano, description of damage, mention of buried structures
3. Cross-reference with GVP eruption catalog (E002)

**Expected yield:** 50-200 newspaper reports spanning 1800-1945

**Estimated effort:** 1-2 weeks

---

### DS-6: Damais Chronological Tables (targets P5, P8)

**Goal:** Digitize L.-C. Damais' comprehensive chronological tables of dated Javanese inscriptions from BEFEO (1950s-1960s).

**Why independent:** Damais' epigraphic chronology is the gold standard but has never been digitized into a structured database. Currently VOLCARCH uses DHARMA (268 inscriptions); Damais covers ~400+ dated records.

**Method:**
1. Access Damais articles on Persee.fr (BEFEO)
2. AI-assisted extraction of inscription dates, locations, and classifications
3. Create structured CSV compatible with VOLCARCH experiment pipeline

**Expected yield:** 400+ dated inscription records (vs current 268 DHARMA)

**Estimated effort:** 2-3 weeks

---

## Phase 1.5 Execution Plan

| Week | Dataset | Activity |
|------|---------|----------|
| 1-2 | DS-1 (Colonial Sites) | Download OV volumes, begin extraction |
| 1-2 | DS-2 (Verbeek Map) | Download, georeference |
| 2-3 | DS-5 (Newspapers) | Delpher.nl searches |
| 3-4 | DS-6 (Damais Tables) | BEFEO extraction |
| 3-5 | DS-3 (Ethnographic) | TBG + Djawa reading |
| 4-5 | DS-4 (Museum) | Wereldmuseum catalog |

## Success Criteria

- At least 2 datasets that are genuinely independent from DHARMA, ABVD, and E001
- At least 50 colonial-era site records with burial depth data
- Verbeek map georeferenced and overlayable with modern data
- Damais epigraphy digitized (expanding inscription corpus by >50%)
- All datasets documented in `data/sources.md` with full provenance

## Risk Assessment

- **OCR quality:** Colonial Dutch typefaces may have OCR errors → manual validation needed
- **Georeferencing accuracy:** Colonial coordinates use different systems → ±500m uncertainty acceptable
- **Completeness:** Not all OV volumes are on Internet Archive → may need Leiden library access
- **Time:** Reading-intensive work, slower than computational experiments

---

## Execution Results (DS-1)

### DS-1: Colonial Archaeological Site Register — COMPLETE

**Output:** `data/raw/colonial_sources/colonial_site_register_v1.0.csv`

| Metric | Value |
|--------|-------|
| Total entries | 52 |
| Entries with burial depth | 32 |
| Depth range | 0.60–9.14 m |
| Geocoded entries | 43 |
| Volcanic association | 44 |
| Source: OV reports | majority |
| Period covered | 1912–1929 |

**Key findings:**
- Mean burial depth: 3.2 m (consistent with L1 sedimentation rates)
- Deepest: 9.14 m (Prambanan Vishnu, OV 1925)
- 84.6% of entries have volcanic association
- Dataset is genuinely independent from DHARMA and ABVD

**Cross-references:**
- E083 uses this register for tephra-archaeological correlation (51 pairs)
- D1 data paper drafted for JOAD publication

### DS-2 through DS-6: PENDING

See Phase 1.5 Execution Plan above. DS-2 (Verbeek Map) and DS-6 (Damais Tables) are highest priority remaining datasets.
