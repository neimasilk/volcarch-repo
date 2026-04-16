# E202: DEM Depression Detection for Buried Archaeological Structures

**Date:** 2026-04-16
**Status:** INCONCLUSIVE (practically FAILED)
**Paper:** P1, P11 (resolution requirements), P18 (satellite archaeology)
**Layer:** L1

---

## Hypothesis

Volcanic tephra burial of pre-colonial structures in East Java creates subtle surface depressions detectable in DEM data. The mechanism: differential compaction between tephra filling a structure void vs surrounding undisturbed soil produces a topographic signature — the same principle used by Canuto et al. (2018, *Nature*) with LiDAR in Maya archaeology and Evans (2016) in Angkor.

**Specific prediction:** Known candi sites should show lower TPI (Topographic Position Index) and deeper depressions compared to random terrain points.

## Method

### Data
- Copernicus GLO-30 DEM (`data/processed/dem/jatim_dem.tif`), 30m resolution, EPSG:32749
- Study area: -8.15 to -7.55 lat, 112.05 to 112.80 lon (covers Malang, Kelud, Trowulan)
- 2155 x 2704 pixels, 5.8M valid cells

### Calibration Sites (9 candi, all in-bounds)
| Site | Lat | Lon | Elevation |
|------|-----|-----|-----------|
| Singosari | -7.889 | 112.636 | 561m |
| Kidal | -8.076 | 112.613 | 387m |
| Jago | -8.069 | 112.613 | 389m |
| Tikus | -7.769 | 112.445 | 1476m |
| Sumberawan | -7.819 | 112.609 | 1289m |
| Jawi | -7.820 | 112.593 | 1351m |
| Badut | -7.970 | 112.613 | 471m |
| Kotes | -7.818 | 112.272 | 302m |
| Gambar Wetan | -7.920 | 112.240 | 612m |

### Target Sites (8 E080 high-probability locations, all in-bounds)
All near Kelud and Arjuno-Welirang volcanoes (E080 fieldwork targets).

### Controls
30 random points, seeded, constrained to 50-1000m elevation within the study area.

### Depression Detection Methods
1. **Fill-sink (local minima):** Identifies cells lower than all 8 neighbors
2. **TPI 150m:** Cell minus mean of circular neighborhood (r=5 pixels)
3. **TPI 300m:** Same, r=10 pixels
4. **Local relief deviation:** Cell minus mean in 330m window
5. **Relief range:** Max minus min in 330m window
6. **Multi-scale TPI:** Average of z-normalized TPI at 90/150/210/300/450m

### Statistical Test
Mann-Whitney U (one-sided): are candi sites more depressed than controls?

## Results

### Detection Rates (multiscale TPI < -0.5 z)

| Category | Detections | Total | Rate |
|----------|-----------|-------|------|
| Candi (known sites) | 1 | 9 | 11.1% |
| Controls (random) | 3 | 30 | 10.0% |
| E080 targets | 2 | 8 | 25.0% |
| Borehole targets | 1 | 8 | 12.5% |

**True positive rate (11.1%) is essentially equal to false positive rate (10.0%).** The method cannot discriminate.

### Statistical Tests (Candi vs Control)

| Metric | U | p-value | Candi mean | Control mean | Cohen's d | Significant? |
|--------|---|---------|------------|--------------|-----------|-------------|
| Depression depth | 135 | 1.000 | 0.000 | 0.000 | 0.000 | NO |
| TPI 150m | 137 | 0.533 | -1.53 | -0.12 | -0.19 | NO |
| TPI 300m | 121 | 0.326 | -5.79 | -0.74 | -0.37 | NO |
| Relief deviation | 140 | 0.573 | -2.33 | -0.43 | -0.22 | NO |
| Relief range | 139 | 0.560 | 44.9 | 41.5 | 0.08 | NO |
| Multi-scale TPI | 138 | 0.546 | -0.28 | -0.03 | -0.27 | NO |

**No metric achieves p < 0.05.** Best candidate: TPI 300m (p=0.326, d=-0.37) — the direction is correct (candi are slightly more negative) but the effect is drowned in noise.

### E080 Targets vs Control
Slightly more suggestive but still not significant. Best: TPI 300m (p=0.075). This marginal signal likely reflects **terrain position** (E080 targets are on volcanic flanks = naturally lower TPI) rather than buried structures.

### Why It Failed: Resolution and Signal-to-Noise Analysis

**Horizontal resolution problem:**

| Structure | Footprint | Pixels at 30m | Min. Resolution |
|-----------|-----------|---------------|-----------------|
| Candi Singosari | 14x14m | 0.22 | 2.8m |
| Candi Kidal | 8x8m | 0.07 | 1.6m |
| Candi Tikus (bath) | 28x22m | 0.68 | 4.4m |
| Village compound | 50x50m | 2.78 | 10m |
| Trowulan city block | 200x200m | 44.4 | 40m |

Individual candi are **sub-pixel** at 30m resolution. They literally do not exist in the DEM.

**Vertical signal-to-noise problem:**

| Burial Depth | Expected Depression | GLO-30 Noise | SNR |
|-------------|--------------------|--------------|----|
| 5m | 0.25-0.75m | ~3.5m RMSE | 0.14 |
| 10m | 0.50-1.50m | ~3.5m RMSE | 0.29 |
| 20m | 1.00-3.00m | ~3.5m RMSE | 0.57 |

Even at 20m burial (maximum predicted by E075), the depression signal is **less than the DEM noise floor**. The signal is invisible.

## Conclusion

**INCONCLUSIVE (practically FAILED).** The Copernicus GLO-30 DEM at 30m resolution cannot detect surface depressions from buried candi in East Java. This is a **resolution and SNR limitation**, not a methodological failure:

1. Individual candi (8-28m) are sub-pixel at 30m resolution
2. Expected depression amplitude (0.25-1.5m) is below the ~3.5m vertical RMSE
3. No statistical test discriminates candi from random terrain
4. The slight negative TPI trend at E080 targets (p=0.075) likely reflects volcanic flank topography, not buried structures

### What WOULD Be Needed

| Target | Minimum Resolution | Technology | Cost |
|--------|-------------------|------------|------|
| Individual candi | 1-5m | Airborne LiDAR | $5K-50K per km2 |
| Village compound | 5-10m | High-res satellite DEM (WorldDEM, Pléiades) | $500-5K per km2 |
| Settlement cluster | 10-15m | TanDEM-X (commercial) | $200-2K per km2 |
| City-scale (Trowulan) | 30-40m | **GLO-30 (this study)** | Free |

**The only features potentially detectable at 30m are city-scale (Trowulan, 1km+).** A dedicated Trowulan analysis would be a worthwhile follow-up.

### Positive Contributions
Despite the failure, this experiment provides:
1. **Quantified resolution requirements** for archaeological DEM analysis in volcanic Java
2. **Signal-to-noise framework** showing exactly why 30m fails and what would succeed
3. **Baseline metrics** at known sites, useful when higher-resolution data becomes available
4. **Validation of E189 strategy:** SAR/spectral satellite methods (E189) are more promising than DEM morphometry for this application

## Scripts
- `depression_analysis.py` — Full analysis pipeline

## Output Files
- `results/e202_results.json` — Complete results with all samples and statistics
- `results/depression_analysis_maps.png` — Six-panel map of depression metrics
- `results/depression_comparison.png` — Boxplot comparison across site categories
- `results/resolution_feasibility.png` — Resolution requirements chart
- `results/signal_noise_analysis.png` — Signal-to-noise diagram

## References
- Canuto, M.A. et al. (2018). Ancient lowland Maya complexity as revealed by airborne laser scanning of northern Guatemala. *Science*, 361(6409).
- Evans, D. (2016). Airborne laser scanning as a method for exploring long-term socio-ecological dynamics in Cambodia. *Journal of Archaeological Science*, 74, 164-175.
- Copernicus DEM GLO-30: https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model
- E080: Fieldwork Targeting (20 priority sites)
- E097: Anomaly Detection (65% overlap with E080)
- E189: Satellite Spectral Feasibility (Phase A, NDVI/NDWI)

## Relation to Other Experiments
- Builds on: E003 (DEM download), E005 (terrain suitability), E080 (fieldwork targets), E097 (anomaly detection)
- Related: E189 (satellite spectral — alternative remote sensing approach), E076 (satellite archaeology)
- Informs: P18 (resolution requirements for "invisible civilization" detection)
- Next: Apply to Trowulan (city-scale, might work at 30m), or wait for LiDAR data
