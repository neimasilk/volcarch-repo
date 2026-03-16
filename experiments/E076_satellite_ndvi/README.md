# E076: Satellite NDVI Anomaly Detection at Candi Sites

## Hypothesis
Buried archaeological structures at candi sites produce detectable vegetation anomalies in Sentinel-2 NDVI imagery (10m resolution). Buried walls reduce soil moisture (lower NDVI), while buried ditches retain moisture (higher NDVI).

## Method
- Sentinel-2 L2A data via Microsoft Planetary Computer STAC API (no registration needed)
- 15 target candi + 5 control sites
- NDVI = (B08-B04)/(B08+B04) in 500m buffer around each site
- Center vs ring anomaly analysis + local variance measurement
- Dry season 2024 (July-September), cloud cover <10%

## Preliminary Results

**STATUS: INCONCLUSIVE (insufficient coverage)**

### Data Coverage
- 5/15 candi sites extracted (Jawi, Kidal, Singosari, Sumberawan, Songgoriti)
- 2/5 control sites extracted
- 10 candi sites (Trowulan cluster, Kediri, Blitar) fell outside available tile (49MGM)

### Promising Trends
| Metric | Candi (n=5) | Control (n=2) |
|--------|------------|---------------|
| Mean center-ring NDVI diff | +0.026 | +0.005 |
| Mean local variance | 0.0029 | 0.0012 |
| Variance ratio | 2.5× higher | baseline |

### Statistical Test
- Mann-Whitney U = 39.0, p = 0.46 (NOT significant — too few samples)

## Next Steps
1. Add more Sentinel-2 tiles (49MFM for Kediri/Blitar, other tiles for Trowulan)
2. Increase sample size to ≥20 candi and ≥10 controls
3. Multi-temporal composites (multiple dry-season dates)
4. Test SAR data (ALOS PALSAR L-band) for subsurface penetration

## Methodological Note
No published study applies NDVI crop-mark detection to candi in Java. This would be methodologically novel — most crop-mark archaeology uses temperate-climate data. Tropical volcanic terrain is unexplored territory.

## Data
- `results/ndvi_anomaly_results.csv` — Per-site NDVI statistics
- `results/e076_results.json` — Summary
