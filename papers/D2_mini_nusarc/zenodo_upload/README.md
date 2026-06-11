# Mini-NusaRC v3: Archaeological Site Database for Island Southeast Asia and Madagascar

## Description

A georeferenced compilation of 80 dated archaeological sites from eight geographic regions of Island Southeast Asia (Indonesia, Malaysia, Philippines, Timor-Leste) and Madagascar, spanning 1,200 to 1,600,000 BP. Covers five hominin species and seven dating methods. Designed for comparative analyses of site discovery patterns across volcanic and non-volcanic landscapes.

## Files

- `mini_nusarc_v3.csv` — Main dataset (80 records, 17 fields, UTF-8)

## Schema (17 fields)

| Field | Type | Description |
|-------|------|-------------|
| site_id | string | Unique identifier (NUSARC-NNNN) |
| site_name | string | Published site name |
| lat | float | Latitude (WGS84) |
| lon | float | Longitude (WGS84) |
| coord_precision | string | exact / approximate / regional |
| region | string | Geographic region (one of 8) |
| country | string | ISO 3166-1 alpha-2 code |
| date_bp | integer | Age in years before present |
| date_type | string | Dating method |
| date_error | integer | Age uncertainty (± years) |
| site_type | string | cave / open_air / river_terrace / rockshelter |
| context_detail | string | Free-text site description |
| cultural_period | string | Archaeological period |
| species | string | Associated hominin species |
| source_citation | string | Primary literature reference |
| confidence | string | high / medium / low |
| notes | string | Additional observations |

## Key Statistics

- 8 regions: Java (19), Sulawesi (18), Nusa Tenggara (12), Kalimantan (8), Sumatra (7), Philippines (6), Maluku (5), Madagascar (5)
- 5 species: H. sapiens (64), H. erectus (8), H. floresiensis (3), unknown (4), H. luzonensis (1)
- 7 dating methods: C14 (45), U-series (14), relative (12), luminescence (4), Ar-Ar (2), fission track (1), laser ablation (1)
- 4 site types: cave (43), open_air (20), river_terrace (9), rockshelter (8)

## Citation

Amien, M. (2026). Mini-NusaRC: A Georeferenced Archaeological Site Database for Island Southeast Asia and Madagascar (1,200-1,600,000 BP) (Version 3) [Data set]. Zenodo. DOI to be assigned on publication.

A data paper describing this dataset is in preparation.

## License

CC BY 4.0

## Author

Mukhlis Amien, Department of Computer Science, Universitas Bhinneka Nusantara, Indonesia. ORCID: 0000-0002-1848-167X

## AI Disclosure

Compilation from published literature was AI-assisted (large language model). Each record carries its primary literature reference (`source_citation`) and a confidence rating so entries can be checked against the published source.
