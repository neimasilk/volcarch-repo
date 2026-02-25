"""
E003 / Step 2: Download Copernicus GLO-30 DEM for full Jawa Timur province.

This is required for E005 (full H1 test) and E002 (provincial volcanic influence map).
Malang Raya pilot DEM (step 1) is insufficient for province-wide analysis.

Bounding box (Jawa Timur province + buffer):
  South: -9.1, North: -6.7, West: 110.9, East: 114.6

Tiles needed (~12 tiles × ~25MB avg = ~300MB total):
  S07 (N=-7): E111, E112, E113, E114
  S08 (N=-8): E111, E112, E113, E114
  S09 (N=-9): E111, E112, E113, E114

Output:
  data/raw/dem/cop30_jatim_*.tif       (individual tiles)
  data/processed/dem/jatim_dem.tif     (merged + projected)
  data/processed/dem/jatim_slope.tif
  data/processed/dem/jatim_twi.tif
  data/processed/dem/jatim_tri.tif

Run from repo root:
    python experiments/E003_dem_acquisition/02_download_full_jatim_dem.py
"""

import sys
from pathlib import Path

# Import reusable functions from step 1
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

# We reuse functions from 01_download_dem.py
# Load them by importing the module directly
import importlib.util

step1_path = Path(__file__).parent / "01_download_dem.py"
spec = importlib.util.spec_from_file_location("step1", step1_path)
step1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step1)

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

REPO_ROOT = Path(__file__).parent.parent.parent
RAW_DEM_DIR = REPO_ROOT / "data" / "raw" / "dem"
PROC_DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"

# Full Jawa Timur bounding box (with 0.1deg buffer)
JATIM_BBOX = {
    "south": -9.1,
    "north": -6.7,
    "west": 110.9,
    "east": 114.6,
}

TARGET_CRS = "EPSG:32749"


def main():
    raw_path = RAW_DEM_DIR / "jatim_cop30.tif"
    proc_path = PROC_DEM_DIR / "jatim_dem.tif"

    print("=" * 60)
    print("E003 Step 2: Full Jawa Timur DEM download")
    print("=" * 60)
    print(f"Bounding box: {JATIM_BBOX}")

    tiles = step1.get_cop30_tiles(JATIM_BBOX)
    total_size_est = len(tiles) * 25
    print(f"Tiles to download: {len(tiles)} (~{total_size_est} MB estimated)")
    for t in tiles:
        print(f"  {t['name']}")

    # Download all tiles
    if raw_path.exists():
        size_mb = raw_path.stat().st_size / 1e6
        print(f"\nMerged DEM already exists ({size_mb:.1f} MB): {raw_path}")
        print("  Delete it to re-download.")
    else:
        print("\nDownloading...")
        success = step1.download_dem(JATIM_BBOX, RAW_DEM_DIR, raw_path)
        if not success:
            sys.exit(1)

    # Reproject
    if not proc_path.exists():
        print("\nReprojecting to UTM 49S...")
        step1.reproject_to_utm(raw_path, proc_path)
    else:
        print(f"\nReprojected DEM exists: {proc_path}")

    # Terrain derivatives
    print("\nComputing terrain derivatives for full Jawa Timur...")
    with rasterio.open(proc_path) as src:
        dem = src.read(1).astype(float)
        profile = src.profile
        nodata = src.nodata if src.nodata is not None else -9999
        res_m = abs(src.transform.a)

    print(f"  DEM shape: {dem.shape}, resolution: {res_m:.1f} m")
    print(f"  Elevation range: {dem[dem != nodata].min():.0f} - {dem[dem != nodata].max():.0f} m")

    from scipy.ndimage import sobel, uniform_filter, generic_filter

    slope, aspect = step1.compute_slope_aspect(dem, res_m)
    step1.save_layer(slope,  profile, PROC_DEM_DIR / "jatim_slope.tif",  nodata=nodata)
    step1.save_layer(aspect, profile, PROC_DEM_DIR / "jatim_aspect.tif", nodata=nodata)

    twi = step1.compute_twi(dem, slope, res_m, nodata=nodata)
    step1.save_layer(twi, profile, PROC_DEM_DIR / "jatim_twi.tif", nodata=nodata)

    tri = step1.compute_tri(dem, nodata=nodata)
    step1.save_layer(tri, profile, PROC_DEM_DIR / "jatim_tri.tif", nodata=nodata)

    print(f"\nAll layers saved to {PROC_DEM_DIR}")
    print("\nNext step: re-run E005 with jatim_dem.tif for full H1 test:")
    print("  python experiments/E005_terrain_suitability/02_full_jatim_analysis.py")


if __name__ == "__main__":
    main()
