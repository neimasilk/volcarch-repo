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

### Phase 2 Needed

Depth values not extracted yet — DC metadata doesn't include full OCR text. Phase 2: fetch individual article full-text via resolver API, then apply NLP depth/location extraction (reuse E091 pipeline).

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
