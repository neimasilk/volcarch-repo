# E003: DEM Acquisition and Processing (Malang Raya)

**Date:** 2026-02-23
**Status:** QUEUED
**Paper:** P2 (Settlement Suitability Model)
**Author:** Amien + Claude

## Hypothesis

Terrain features derived from a DEM (slope, aspect, TWI, TRI, curvature) are informative predictors
of ancient settlement locations in the Malang Raya study area.

## Method

1. Download SRTM 30m DEM for Malang Raya bounding box (~7.5°S–8.5°S, 112°E–123.5°E)
   - Source: NASA EarthData (public domain, no registration for basic SRTM)
   - Alternative: OpenTopography API (free, SRTM available)
2. Clip to study area polygon (Kota Malang + Kabupaten Malang)
3. Derive terrain layers:
   - Slope (degrees)
   - Aspect (degrees)
   - TWI = Topographic Wetness Index = ln(a / tan(β))
   - TRI = Terrain Ruggedness Index
   - Plan/profile curvature
4. Mosaic, reproject to WGS84 UTM Zone 49S (EPSG:32749)
5. Write all layers to `data/processed/dem/`

## Data

- Input: SRTM 30m (raw tiles → `data/raw/dem/`)
- Output: `data/processed/dem/malang_dem.tif`, `malang_slope.tif`, `malang_aspect.tif`,
  `malang_twi.tif`, `malang_tri.tif`

## Results

*Pending — experiment not yet run.*

## Conclusion

*Pending.*

## Next Steps

- Terrain features feed directly into Paper 2 settlement suitability model
- If DEMNAS 8.5m becomes available, re-run at higher resolution

## Failure Notes (if applicable)

*N/A — not yet attempted.*
