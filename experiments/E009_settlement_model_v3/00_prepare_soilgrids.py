"""
E009 helper: fetch SoilGrids clay/silt (0-5cm mean) and align to East Java DEM grid.

Run from repo root:
    py experiments/E009_settlement_model_v3/00_prepare_soilgrids.py
"""

import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
DEM_PATH = REPO_ROOT / "data" / "processed" / "dem" / "jatim_dem.tif"
DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"

SOIL_URLS = {
    "clay": "https://files.isric.org/soilgrids/latest/data/clay/clay_0-5cm_mean.vrt",
    "silt": "https://files.isric.org/soilgrids/latest/data/silt/silt_0-5cm_mean.vrt",
}


def reproject_soil(url: str, out_path: Path, ref_meta: dict) -> None:
    ref_h = ref_meta["height"]
    ref_w = ref_meta["width"]
    ref_transform = ref_meta["transform"]
    ref_crs = ref_meta["crs"]
    dst_arr = np.full((ref_h, ref_w), np.nan, dtype=np.float32)

    with rasterio.open(url) as src:
        print(f"  Source CRS: {src.crs}")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    out_meta = ref_meta.copy()
    out_meta.update(
        count=1,
        dtype="float32",
        compress="lzw",
        nodata=np.nan,
    )
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(dst_arr, 1)

    finite = dst_arr[np.isfinite(dst_arr)]
    print(
        f"  Saved: {out_path.name} | valid pixels={finite.size:,} "
        f"range=[{finite.min():.2f}, {finite.max():.2f}]"
    )


def main() -> None:
    if not DEM_PATH.exists():
        print(f"ERROR: Missing reference DEM: {DEM_PATH}")
        sys.exit(1)

    with rasterio.open(DEM_PATH) as dem:
        ref_meta = dem.meta.copy()
        print(f"DEM grid: {dem.width}x{dem.height}, CRS={dem.crs}")
        print(f"DEM bounds: {dem.bounds}")

    print("\nPreparing SoilGrids layers...")
    for feat, url in SOIL_URLS.items():
        out_name = f"jatim_{feat}.tif"
        out_path = DEM_DIR / out_name
        print(f"\nProcessing {feat} from:")
        print(f"  {url}")
        reproject_soil(url, out_path, ref_meta)

    print("\nDone.")


if __name__ == "__main__":
    main()
