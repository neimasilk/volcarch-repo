# Colonial Archaeological Register of Java (CARJ) v1.0

## Description

A georeferenced database of 52 archaeological site observations extracted from Dutch colonial *Oudheidkundig Verslag* (OV) reports, 1912-1929. Each record includes site identification, location (WGS84), burial depth measurements, material descriptions, volcanic context, and condition assessments as documented by colonial surveyors of the Netherlands East Indies Archaeological Service.

## Files

- `colonial_site_register_v1.0.csv` — Main dataset (52 records, 21 fields, UTF-8)
- `REGISTER_NOTES.md` — Data documentation, version history, key findings, limitations

## Schema (21 fields)

| Field | Type | Description |
|-------|------|-------------|
| source | string | OV volume reference |
| year_report | integer | Year of OV report |
| site_name | string | Dutch colonial site name |
| modern_name | string | Modern Indonesian name |
| location | string | Location from OV text |
| regency | string | Colonial regentschap / modern kabupaten |
| province | string | Modern province |
| lat | float | Latitude (WGS84, approximate) |
| lon | float | Longitude (WGS84, approximate) |
| built_ce | string | Estimated construction date (CE) |
| found_year | integer | Year of discovery |
| burial_depth_m | float | Burial depth in meters |
| depth_type | string | measured / estimated / qualitative |
| material | string | Material description |
| volcanic_system | string | Associated volcanic system |
| volcano_dist_km | float | Distance to volcano (km) |
| rate_mm_yr | float | Calculated burial rate (mm/year) |
| condition | string | Site condition at observation |
| context | string | Depositional context |
| notes | string | Key observations from colonial text |
| ov_page | string | Page reference in OV volume |

## Key Statistics

- Entries with depth measurements: 32 (62%)
- Depth range: 0.60-9.14 m (mean 2.88 m, median 2.00 m)
- Entries with coordinates: 43 (83%)
- Entries with volcanic system: 44 (85%)

## Citation

Amien, M. (2026). The Colonial Archaeological Register of Java: A Digitized Database of Site Observations from Dutch *Oudheidkundig Verslag* Reports (1912-1929). *Journal of Open Archaeology Data* [submitted].

## License

CC BY 4.0

## Author

Mukhlis Amien, Department of Computer Science, Universitas Muhammadiyah Malang, Indonesia. ORCID: 0000-0002-1848-167X
