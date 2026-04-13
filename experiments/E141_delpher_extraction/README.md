# E141: Delpher Colonial Newspaper Extraction Pipeline

**Date:** 2026-03-30
**Status:** SUCCESS (Phase 1 — metadata extraction)
**Paper:** P21 ColonialMine
**Layer:** Cross-cutting (independent dataset)

---

## Hypothesis

Dutch colonial newspapers (1800-1942) contain hundreds of incidental archaeological observations that can be systematically extracted via the KB SRU API.

## Method

12 targeted queries via KB SRU API (`https://jsru.kb.nl/sru/sru`, collection `DDD_artikel`). Each query combines archaeological, depth, volcanic, and location keywords in Dutch. Results deduplicated and classified by relevance (archaeology, depth, material, volcanic context, Java location).

## Results

| Metric | Value |
|--------|:---:|
| Queries run | 12 |
| Total unique records | **529** |
| High relevance (score >= 4) | 30 |
| Archaeological context | 196 |
| Singosari-specific | 43 |
| Mojokerto-specific | 49 |

### Notable Finds

- "OOST-JAVA Oudheidkundige Vondsten STEENEN, MESSEN, SPEREN EN BIJLEN" (1939) — stone tools in East Java
- "Het sprekende Vrouwenbeeld OPGRAVING IN KAMPONG" (1938) — statue excavated from village
- "OUDHEIDKUNDIG BODEM ONDERZOEK MADJAPAHIT" (1941) — Majapahit archaeological survey
- "OUDHEDEN HEBBEN DOOR BEVING..." (1937) — artifacts exposed by earthquake
- "Natuur en Historie op het Diëng-Plateau" (1935) — Dieng volcanic + historical

### Phase 2 COMPLETED (2026-04-13)

Full-text fetched + NLP extraction for 96 high-relevance records.

| Metric | Value |
|--------|:---:|
| Full-text fetched | 96/96 |
| With geocoded locations | 68 |
| With depth values (archaeological) | 2 (after filtering oil exploration) |
| With volcanic context | 22 |
| **Near E080 fieldwork targets (25km)** | **19 (28%)** |

**Key finds:**
- **Penataran/Kelud slope, 1.0m depth** (1939) — validates burial model prediction
- **Singosari cluster** — 4 reports 1938-1941, 13km from E080 target
- **28% of geocoded colonial reports are near VOLCARCH's predicted fieldwork zones**
- **22 reports mention volcanic context** — colonial observers already noted the volcano-archaeology link

Materials: statues (55), temples (47), stone (42), metal (22), tools (19), inscriptions (11), bone (9), pottery (4).

Top locations: Batavia (32), Modjokerto (22), Malang (16), Kediri (15), Boroboedoer (12), Prambanan (11), Singosari (10), Blitar (10).

Files: `results/delpher_phase2_fulltext.csv`, `results/delpher_phase2_summary.json`

## API Details

- **Endpoint:** `https://jsru.kb.nl/sru/sru`
- **No registration required** for public domain collections
- **Collection:** `DDD_artikel` (newspaper articles)
- **Query syntax:** CQL with AND operators

## Scripts

- `delpher_extract.py` — Full pipeline (12 queries, dedup, classify, export)

## Output

- `results/delpher_extraction.csv` — 529 records with title, date, source, relevance, tags
- `results/delpher_summary.json` — Summary statistics
