# Archaeological Site Data Schema

## File: `data/processed/sites.csv`

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| site_id | string | yes | Unique identifier (e.g., "SING001") |
| name | string | yes | Site name (e.g., "Candi Singosari") |
| lat | float | yes | Latitude (WGS84) |
| lon | float | yes | Longitude (WGS84) |
| coord_quality | enum | yes | "gps" / "map" / "estimated" — how precise are the coordinates |
| type | string | yes | "candi" / "arca" / "prasasti" / "settlement" / "other" |
| period_label | string | no | Kingdom/era (e.g., "Singosari", "Kanjuruhan", "Majapahit") |
| construction_year | int | no | Estimated construction year (CE, negative for BCE) |
| discovery_year | int | no | Year of modern discovery/documentation |
| burial_depth_cm | float | no | Measured burial depth at discovery (cm) — GOLD DATA |
| burial_depth_source | string | no | Citation for burial depth measurement |
| buried_by | string | no | Volcanic system responsible (e.g., "Kelud", "Merapi") |
| sedimentation_rate_mm_yr | float | no | Computed rate if both depth and dates are known |
| source | string | yes | Where this data point came from (URL, paper citation, etc.) |
| notes | string | no | Free text notes |

## Quality Flags

- **coord_quality = "gps"**: Coordinates from GPS measurement or precise map. ±10m accuracy.
- **coord_quality = "map"**: Coordinates estimated from published map or satellite imagery. ±100m accuracy.
- **coord_quality = "estimated"**: Coordinates inferred from text description. ±1km accuracy.

## Example Rows

```csv
site_id,name,lat,lon,coord_quality,type,period_label,construction_year,discovery_year,burial_depth_cm,burial_depth_source,buried_by,sedimentation_rate_mm_yr,source,notes
SING001,Arca Dwarapala Singosari,-7.8893,112.7184,map,arca,Singosari,1268,1803,185,"BPCB Jawa Timur",Kelud,3.6,"kebudayaan.kemdikbud.go.id","Primary calibration anchor"
SAMB001,Candi Sambisari,-7.7528,110.4448,map,candi,,,,650,"BPCB DIY",Merapi,,"Wikipedia; BPCB DIY","Discovered 1966 by farmer"
KEDU001,Candi Kedulan,-7.7439,110.4556,map,candi,,,,700,"BPCB DIY",Merapi,,"BPCB DIY","13 stratigraphic layers visible"
KIMP001,Candi Kimpulan,-7.6873,110.4208,map,candi,,,,270,"BPCB DIY; UII",Merapi,,"UII perpustakaan","Glass box showing 19 eruption layers"
LIAN001,Candi Liangan,-7.3214,110.0169,estimated,candi,,800,2008,600,estimated,Sundoro,,"Various","9th century; discovered 2008"
```

## Rules

1. Sites with `burial_depth_cm` values are **calibration gold** — treat with extra care.
2. Every row must have a `source`. No anonymous data points.
3. When in doubt about coordinates, use `coord_quality = "estimated"` and note the uncertainty.
4. Negative `construction_year` = BCE (e.g., -800 = 800 BCE).
