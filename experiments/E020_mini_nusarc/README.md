# E020: Mini-NusaRC — Preliminary Radiocarbon Database for H-TOM Testing

## Status: PRELIMINARY RESULTS (v2 — 48 sites, merged from v1 + agent harvest)

### Dataset Versions
- **v1** (41 sites): Manual compilation from known literature
- **v2** (48 sites): Merged v1 + 7 new sites from agent-harvested nusarc_v0.1.csv (6 Sulawesi, 1 Kalimantan)
- **Gaps**: Sumatra (2/5 min), Philippines (3/4 min)

### Metric 1 Result (v2)
- Fisher's exact p = 0.761 — NOT SIGNIFICANT
- Volcanic regions: 82.4% cave; Non-volcanic: 85.7% cave
- **Interpretation**: Cave bias is universal across tropical regions, not specific to volcanic ones. The H-TOM signal is in DISCOVERY METHOD (erosion/construction exposure vs surface survey), not simple site type binary. Metric needs refinement.

## Strategic Rationale

Paper 7 has three test metrics for H-TOM:
- **Metric 3 (spatial distribution): DONE** — E019 confirmed, Cohen's d = 1.005
- **Metric 1 (cave/open-air ratio): BLOCKED** on radiocarbon site database
- **Metric 2 (site density per time bin): BLOCKED** on radiocarbon site database

Full NusaRC (comprehensive Nusantaran radiocarbon database) = 6-18 months.
**Mini-NusaRC** = ~80 sites from open-access literature in 2-4 weeks, sufficient for preliminary tests of Metrics 1 & 2.

If mini-NusaRC + E019 together show all three metrics supporting H-TOM, Paper 7 becomes publishable WITHOUT waiting for full NusaRC.

## Hypothesis

Same as Paper 7 H-TOM v2:

> "The ratio of cave sites to total sites for deposits >10,000 BP will be significantly higher in volcanic regions (TAP_index > 0.5) than in non-volcanic regions (TAP_index < 0.2)."

Additionally for Metric 2:

> "Site density per 5,000-year time bin will decline more steeply in volcanic regions than in non-volcanic regions for periods >5,000 BP."

## Data Schema

Each record in mini-NusaRC:

```
site_id:          Unique identifier (e.g., NUSARC-0001)
site_name:        Published name
lat:              Decimal degrees (WGS84)
lon:              Decimal degrees (WGS84)
coord_precision:  exact / approximate / regional
region:           One of 8 TOM regions (Java, Sumatra, Sulawesi,
                  Kalimantan, Nusa_Tenggara, Philippines, Maluku, Madagascar)
country:          ISO 3166-1
date_bp:          Radiocarbon date (uncalibrated BP, or cal BP if specified)
date_type:        C14 / OSL / U-series / TL / relative
date_error:       +/- years
lab_code:         Laboratory code (e.g., OxA-12345, Beta-67890)
cal_bp_2sigma_lo: Calibrated range lower (cal BP), if available
cal_bp_2sigma_hi: Calibrated range upper (cal BP), if available
site_type:        cave / rockshelter / open_air / river_terrace /
                  shell_midden / burial / other
context_detail:   Free text (e.g., "karst cave in Gunung Sewu")
material_dated:   charcoal / bone / shell / sediment / other
cultural_period:  Lower_Paleolithic / Middle_Paleolithic / Upper_Paleolithic /
                  Mesolithic / Neolithic / Metal_Age / Historical / unknown
species:          Homo_sapiens / Homo_erectus / unknown / fauna_only
n_volcanoes_200km: Number of active volcanoes within 200 km
dist_nearest_volcano_km: Distance to nearest active volcano (km)
tap_index:        Taphonomic pressure index (computed)
source_doi:       DOI of source publication
source_citation:  Short citation (e.g., "Semah et al. 2023")
source_table:     Which table/figure in the paper (e.g., "Table 3")
confidence:       high / medium / low
notes:            Free text
```

## Minimum Viable Dataset

For H-TOM testing, the minimum requirement is:

| Region | Target sites | Min sites | Key requirement |
|--------|-------------|-----------|-----------------|
| Java | 15 | 8 | Mix of deep-time cave + historical open-air |
| Sumatra | 10 | 5 | Include Lida Ajer + Neolithic sites |
| Sulawesi | 15 | 8 | Rich cave record (Maros-Pangkep, Muna) |
| Kalimantan | 10 | 5 | KEY CONTROL — zero volcanoes |
| Nusa Tenggara | 8 | 4 | Mix of volcanic (Flores) + non-volcanic (Timor) |
| Philippines | 8 | 4 | Tabon + other cave sites |
| Maluku | 5 | 3 | Golo Cave + others |
| Madagascar | 5 | 3 | KEY CONTROL — zero volcanoes, Austronesian-era |
| **Total** | **~76** | **~40** | **Min 40 for preliminary stats** |

## Seed Dataset (from P7 Table 1 + E019)

These 12 sites are already documented in the repo with verified coordinates and citations:

| site_name | region | date_bp | site_type | source |
|-----------|--------|---------|-----------|--------|
| Song Terus | Java | 60,000 | cave | Semah et al. 2023 |
| Trinil | Java | 500,000 | river_terrace | Dubois 1894; Huffman et al. 2010 |
| Sangiran | Java | 1,600,000 | river_terrace | von Koenigswald 1940; Larick et al. 2001 |
| Wajak | Java | 28,000 | cave | Dubois 1922; Storm et al. 2013 |
| Lida Ajer | Sumatra | 68,000 | cave | Westaway et al. 2017 |
| Liang Metanduno | Sulawesi | 67,800 | cave | Oktaviana et al. 2026 |
| Laili Cave | Nusa_Tenggara | 44,600 | cave | Hawkins et al. 2017 |
| Tabon Cave | Philippines | 47,000 | cave | Detroit et al. 2004 |
| Golo Cave | Maluku | 36,000 | cave | Bellwood 1998 |
| Niah Cave | Kalimantan | 40,000 | cave | Barker et al. 2007 |
| Liang Bua | Nusa_Tenggara | 95,000 | cave | Sutikna et al. 2016 |
| Christmas River | Madagascar | 10,500 | open_air | Hansford et al. 2018 |

**Current bias: 11/12 = cave. This is the pattern H-TOM predicts — but we need Neolithic/historical open-air sites to compute ratios properly.**

## Pipeline Architecture

### Phase 1: Automated Harvest (Week 1-2)

```
INPUT:  Search queries for C14 dates in Nusantara
TOOLS:  Semantic Scholar API, OpenAlex API, Unpaywall
OUTPUT: ~200-500 candidate papers with metadata

Steps:
1. Query APIs with archaeological + chronometric keywords
2. Filter by region (Indonesia, Philippines, Madagascar)
3. Check open access status
4. Download accessible full texts
5. Extract tables (Camelot/GROBID for PDFs)
6. NER for site names, dates, lab codes
```

**Realistic yield:** ~30-50% of key papers are open access. The rest need institutional access.

### Phase 2: LLM-Assisted Extraction (Week 2-3)

```
INPUT:  Open-access paper full texts
TOOLS:  Claude (read PDF + extract structured data)
OUTPUT: Structured records per schema

Per paper:
- Read paper (Claude can read PDFs directly)
- Extract all C14 dates into schema format
- Classify site_type from text description
- Flag low-confidence extractions

Key advantage: Claude can read Indonesian-language papers
```

### Phase 3: Manual Verification (Week 3-4)

```
INPUT:  Extracted records with confidence flags
TOOLS:  Domain expertise (user)
OUTPUT: Verified mini-NusaRC dataset

Tasks:
- Verify coordinate accuracy (GIS check)
- Confirm site_type classification
- Assess dating reliability
- Fill gaps from paywalled papers (if institutional access available)
```

## Bottleneck Analysis

### What CAN be automated:
- Paper discovery and metadata extraction
- Open-access full text download
- Table extraction from structured PDFs
- Named entity recognition for dates and lab codes
- Geocoding from site names
- TAP_index and volcano distance computation

### What CANNOT be automated (the real bottleneck):
1. **Paywalled papers (~60-70%):** Asian Perspectives, Berkala Arkeologi, Bijdragen, government reports
2. **Non-digitized sources:** Indonesian-language excavation reports, theses, Balai Arkeologi reports
3. **Context classification:** "cave or open-air?" requires reading text, not just metadata
4. **Dating reliability:** requires domain expertise to assess
5. **Coordinate verification:** Indonesian site names often ambiguous

### Mitigation:
- Start with open-access papers only — sufficient for ~40+ sites
- Use Claude to read full texts and classify context (Phase 2)
- User provides domain verification (Phase 3)
- Expand to paywalled sources later if preliminary results warrant

## Priority Open-Access Papers

Papers likely to contain multiple C14 dates for Nusantara (to verify and expand):

| Paper | Region | Expected dates | Access |
|-------|--------|---------------|--------|
| Barker et al. 2007 (Niah) | Kalimantan | 10+ | Open (JHE) |
| Sutikna et al. 2016 (Liang Bua) | NTT | 15+ | Open (Nature) |
| Oktaviana et al. 2026 (Sulawesi) | Sulawesi | 5+ | Open (Nature) |
| Westaway et al. 2017 (Lida Ajer) | Sumatra | 10+ | Open (Nature) |
| Hawkins et al. 2017 (Laili) | NTT | 8+ | QSR (check access) |
| Bellwood et al. 1998 (Golo) | Maluku | 5+ | Open? |
| Aubert et al. 2014 (Maros) | Sulawesi | 10+ | Open (Nature) |
| Brumm et al. 2006 (Mata Menge) | NTT | 8+ | Open (Nature) |
| O'Connor et al. 2011 (Timor) | NTT | 10+ | Various |
| Storm et al. 2013 (Wajak) | Java | 5+ | Check access |
| Simanjuntak et al. (Song Terus seq.) | Java | 10+ | Various |

**Estimated yield from open-access papers alone: 80-120 date records.**

## Analyses Enabled by Mini-NusaRC

### Test 1: Cave/Open-Air Ratio (Metric 1)

```
Group sites by:
  - volcanic (TAP_index > 0.5): Java, Sumatra, Sulawesi, NTT, Philippines
  - non-volcanic (TAP_index < 0.2): Kalimantan, Madagascar

For sites with date_bp > 10,000:
  - Compute cave_ratio = n_cave / n_total per region
  - Mann-Whitney U: volcanic cave_ratio vs non-volcanic cave_ratio
  - Prediction (H-TOM): volcanic regions have higher cave_ratio

For sites with date_bp > 5,000 AND date_bp < 10,000:
  - Same test — effect should be weaker (less time for burial)
```

### Test 2: Site Density Dropoff (Metric 2)

```
Bin sites by 5,000-year intervals:
  0-5K, 5-10K, 10-15K, 15-20K, 20-25K, 25-30K, 30-35K, 35-40K+

Per region, compute: n_sites per bin / total_sites_in_region
Plot: normalized density curves for each region

Prediction (H-TOM):
  - Kalimantan: relatively flat curve (no systematic burial)
  - Java: steep dropoff for bins > 5-10K BP
  - Sulawesi: intermediate (some volcanism, many caves)
```

### Test 3: Combined spatial + temporal (NEW — integrates E019)

```
For Java sites specifically:
  - Plot date_bp vs distance_to_nearest_volcano
  - Prediction: older sites exclusively at large distances
    (karst/river contexts far from volcanic axis)
  - Younger sites (Neolithic/historical) can be anywhere
  - This directly tests whether volcanic burial preferentially
    removes old evidence near volcanoes
```

## Kill / Pass Criteria

| Outcome | Finding | Decision |
|---------|---------|----------|
| Strong support | Cave ratio significantly higher in volcanic regions (p<0.05); density dropoff steeper in Java than Kalimantan | **H-TOM v2 SUPPORTED** — publish Paper 7 with all 3 metrics |
| Moderate | Trend visible but p>0.05 (underpowered with n=80) | **PARTIAL** — expand to full NusaRC |
| No signal | Cave ratio similar across all regions | **NEUTRAL** — H-TOM may still hold but requires different test |
| Counter | Non-volcanic regions have higher cave ratio | **WEAKENS H-TOM** |

## Publication Potential

**Mini-NusaRC as standalone data contribution:**
- Target: Journal of Open Archaeology Data, Data in Brief
- Value: no compiled radiocarbon database exists for Island Southeast Asia
- Even ~80 sites with standardized schema = citable contribution
- This becomes the seed for full NusaRC

**Mini-NusaRC + E019 + P7 framework = complete Paper 7:**
- All three metrics tested (even if Metrics 1&2 are preliminary)
- Target: Quaternary Science Reviews, Journal of Human Evolution
- Much stronger than framework-only paper

## Implementation Notes

- All scripts go in this experiment directory
- Seed CSV in `data/seed_sites.csv` (the 12 known sites)
- Harvested records in `data/harvested_raw.csv`
- Verified records in `data/mini_nusarc_v1.csv`
- Analysis scripts: `01_harvest.py`, `02_extract.py`, `03_analyze.py`
- Results in `results/`
