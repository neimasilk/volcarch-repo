# Data Sources & Provenance

All data used in this project must be documented here with source, license, and access method.

## Open Datasets

| Dataset | Source | Resolution | License | Status |
|---------|--------|-----------|---------|--------|
| SRTM DEM | NASA EarthData | 30 m | Public domain | ACQUIRED (via Copernicus GLO-30, E003) |
| DEMNAS | Badan Informasi Geospasial (BIG) | 8.5 m | Indonesian gov (free, registration) | NOT ACQUIRED |
| Sentinel-2 imagery | Copernicus / ESA | 10 m | Free & open | NOT ACQUIRED |
| Eruption records | Global Volcanism Program (Smithsonian) | — | Free (citation required) | ACQUIRED (168 records, E002) |
| ERA5 wind reanalysis | Copernicus Climate Data Store | 31 km / hourly | Free (registration) | NOT ACQUIRED |
| FAO soil map (HWSD) | FAO | ~1 km | Free | ACQUIRED (E007+, used in settlement model) |
| River network | OpenStreetMap / HydroSHEDS | Varies | ODbL / Public domain | ACQUIRED (river distance features, E007+) |

## Literature-Derived Data

| Dataset | Compiled From | Status |
|---------|--------------|--------|
| East Java archaeological sites | BPCB publications, Wikipedia, OSM, papers | COMPLETE (666 sites, 383 geocoded, E001/E006) |
| Eruption history (structured) | GVP Smithsonian, published papers | COMPLETE (168 records, E002) |
| Kelud isopach maps | Published volcanological papers | PARTIAL (E017 used Pyle 1989 generic model; per-volcano isopachs not acquired) |
| Dwarapala measurements | BPCB Jawa Timur, news sources | COMPLETE (see JOURNAL.md) |
| Mini-NusaRC v0.1 | 10+ published papers (Nature, JHE, Science, PLoS ONE) | v0.1 COMPLETE (51 records) |

### Mini-NusaRC v0.1 (`data/raw/nusarc_v0.1.csv`)

Radiocarbon/U-series/ESR dated archaeological site database for Island Southeast Asia. 51 records from 16 unique sites across 5 regions (Sulawesi, Kalimantan, Nusa Tenggara, Java, Philippines). Date range: 840 ka to 4.5 ka BP. Sources:

1. Aubert et al. 2014 Nature 514:223-227 (Maros cave art, Sulawesi)
2. Aubert et al. 2018 Nature 564:254-257 (Borneo cave art)
3. Brumm et al. 2006 Nature 441:624-628 (Mata Menge, Flores)
4. Sutikna et al. 2016 Nature 532:366-369 (Liang Bua revised chronology)
5. Detroit et al. 2019 Nature 568:181-186 (Callao Cave, Philippines)
6. O'Connor et al. 2011 Science 334:1117-1121 (Jerimalai, Timor-Leste)
7. Brumm et al. 2018 PLoS ONE 13:e0193025 (Leang Burung 2, Sulawesi)
8. Simanjuntak 2002; Forestier 2007 (Song Keplek, Java)
9. Westaway et al. 2007 JAS 34:1953-1969 (Song Gupuh, Java)
10. Storm et al. 2013 JHE 64:356-365 (Wajak, Java)
11. Barker et al. 2007 JHE 52:243-261 (Niah Cave, Borneo)
12. Hameau et al. 2007 Quat Geochronol 2:356-362 (Song Terus, Java)
13. Brumm et al. 2017 JHE 114:200-217 (Leang Bulu Bettue, Sulawesi)
14. O'Connor 2007 Asian Perspectives 46:180-208 (Lene Hara, Timor)
15. O'Connor et al. 2017 Quat Sci Rev 171:58-72 (Laili, Timor)

**Coordinate quality:** Mixed. Some from publications (marked `from_publication`), some from Wikipedia/Wikidata (marked `from_wikipedia`), some approximate from regional maps (marked `approximate`). All coordinates should be verified before formal analysis.

## Data Rules

1. **Raw data is immutable.** Never modify files in `data/raw/`. Process into `data/processed/`.
2. **Every dataset must have a source.** No anonymous data.
3. **Respect licenses.** Some datasets require citation or restrict commercial use.
4. **Geocoding accuracy.** When geocoding sites from literature, record the accuracy level (exact coordinates from GPS, approximate from map, or estimated from description).
