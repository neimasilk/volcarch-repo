# Data Sources & Provenance

All data used in this project must be documented here with source, license, and access method.

## Open Datasets

| Dataset | Source | Resolution | License | Status |
|---------|--------|-----------|---------|--------|
| SRTM DEM | NASA EarthData | 30 m | Public domain | NOT ACQUIRED |
| DEMNAS | Badan Informasi Geospasial (BIG) | 8.5 m | Indonesian gov (free, registration) | NOT ACQUIRED |
| Sentinel-2 imagery | Copernicus / ESA | 10 m | Free & open | NOT ACQUIRED |
| Eruption records | Global Volcanism Program (Smithsonian) | — | Free (citation required) | NOT ACQUIRED |
| ERA5 wind reanalysis | Copernicus Climate Data Store | 31 km / hourly | Free (registration) | NOT ACQUIRED |
| FAO soil map (HWSD) | FAO | ~1 km | Free | NOT ACQUIRED |
| River network | OpenStreetMap / HydroSHEDS | Varies | ODbL / Public domain | NOT ACQUIRED |

## Literature-Derived Data

| Dataset | Compiled From | Status |
|---------|--------------|--------|
| East Java archaeological sites | BPCB publications, Wikipedia, OSM, papers | NOT STARTED |
| Eruption history (structured) | GVP Smithsonian, published papers | NOT STARTED |
| Kelud isopach maps | Published volcanological papers | NOT STARTED |
| Dwarapala measurements | BPCB Jawa Timur, news sources | COMPLETE (see JOURNAL.md) |

## Data Rules

1. **Raw data is immutable.** Never modify files in `data/raw/`. Process into `data/processed/`.
2. **Every dataset must have a source.** No anonymous data.
3. **Respect licenses.** Some datasets require citation or restrict commercial use.
4. **Geocoding accuracy.** When geocoding sites from literature, record the accuracy level (exact coordinates from GPS, approximate from map, or estimated from description).
