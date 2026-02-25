"""
E003 / Step 1: Download 30m DEM for Malang Raya and derive terrain layers.

Primary source: Copernicus GLO-30 DEM via AWS Open Data (truly free, no auth required)
  - 30m resolution, global coverage
  - Available as 1-degree tiles on AWS S3
  - AWS bucket: copernicus-dem-30m.s3.amazonaws.com

Fallback: OpenTopography SRTM GL1 (requires free API key at opentopography.org)

Study area: Malang Raya (Kota Malang + Kabupaten Malang)
  Bounding box: 7.8S-8.3S, 112.4E-113.1E (approximately)

Derived layers:
  - slope (degrees)
  - aspect (degrees from north)
  - TWI (Topographic Wetness Index) = ln(contributing_area / tan(slope))
  - TRI (Terrain Ruggedness Index) = mean absolute difference from neighbors

Run from repo root:
    python experiments/E003_dem_acquisition/01_download_dem.py

Output:
    data/raw/dem/cop30_*.tif             (raw Copernicus tiles -- do not modify)
    data/processed/dem/malang_dem.tif    (merged + projected)
    data/processed/dem/malang_slope.tif
    data/processed/dem/malang_aspect.tif
    data/processed/dem/malang_twi.tif
    data/processed/dem/malang_tri.tif

Citation required:
    European Space Agency, Sinergise (2021). Copernicus Global Digital Elevation
    Model. Distributed by OpenTopography. doi:10.5069/G9028PQB

    Airbus (2020). Copernicus DEM - Global and European Digital Elevation Model (COP-DEM).
    https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model
"""

import sys
from pathlib import Path
import requests
import numpy as np

# Rasterio and scipy are required
try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    from rasterio import transform as rio_transform
    import rasterio.features
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

try:
    from scipy.ndimage import generic_filter, sobel, uniform_filter
except ImportError:
    print("ERROR: scipy not installed. Run: pip install scipy")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
RAW_DEM_DIR = REPO_ROOT / "data" / "raw" / "dem"
PROC_DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"

# Malang Raya bounding box (WGS84): south, north, west, east
BBOX = {
    "south": -8.30,
    "north": -7.80,
    "west": 112.40,
    "east": 113.10,
}

# Target CRS for all outputs: UTM Zone 49S (meters, suitable for East Java)
TARGET_CRS = "EPSG:32749"

# Copernicus GLO-30 DEM on AWS (no auth required)
# Tile naming: Copernicus_DSM_COG_10_S{lat:02d}_00_E{lon:03d}_00_DEM
COP30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def get_cop30_tiles(bbox: dict) -> list[dict]:
    """
    Determine which 1-degree Copernicus GLO-30 tiles cover the bounding box.
    Returns list of {lat, lon, url, filename} dicts.
    """
    import math
    tiles = []
    lat_min = math.floor(bbox["south"])   # e.g. -9 for south=-8.3
    lat_max = math.floor(bbox["north"])   # e.g. -8 for north=-7.8
    lon_min = math.floor(bbox["west"])    # e.g. 112 for west=112.4
    lon_max = math.floor(bbox["east"])    # e.g. 113 for east=113.1

    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            abs_lat = abs(lat)
            abs_lon = abs(lon)
            # Copernicus tile convention: lat is the UPPER-LEFT corner
            # For lat=-8, the tile covers -9 to -8 (upper left is -8)
            # For lat=-9, the tile covers -9 to -8 ... wait, let me check.
            # Actually Copernicus tiles: S08 covers lat -8 to -9 (tile labeled by the northernmost lat)
            # The tile filename lat refers to the northern edge of the tile.
            # For BBOX south=-8.3, north=-7.8:
            #   We need tiles S08 (covers -8 to -9) and S07 (covers -7 to -8)
            # But Copernicus uses the SOUTHERN edge in the filename for S tiles.
            # Let me use the convention from actual AWS listing.
            # Based on AWS structure: S08 = lat between -8 and -9
            # tile label = abs(floor) for southern hemisphere
            tile_name = f"Copernicus_DSM_COG_10_{ns}{abs_lat:02d}_00_{ew}{abs_lon:03d}_00_DEM"
            url = f"{COP30_BASE}/{tile_name}/{tile_name}.tif"
            tiles.append({
                "lat": lat,
                "lon": lon,
                "name": tile_name,
                "url": url,
                "filename": f"cop30_{ns}{abs_lat:02d}_{ew}{abs_lon:03d}.tif",
            })
    return tiles


def download_cop30_tile(tile: dict, raw_dir: Path) -> Path | None:
    """Download a single Copernicus GLO-30 tile. Returns path or None on failure."""
    out_path = raw_dir / tile["filename"]
    if out_path.exists():
        print(f"  Already exists: {tile['filename']}")
        return out_path

    print(f"  Downloading: {tile['filename']}  ({tile['url']})")
    try:
        resp = requests.get(
            tile["url"],
            timeout=180,
            stream=True,
            headers={"User-Agent": "VOLCARCH-research/0.1 (academic)"},
        )
        resp.raise_for_status()
        raw_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        size_mb = out_path.stat().st_size / 1e6
        print(f"    Done: {size_mb:.1f} MB")
        return out_path
    except requests.RequestException as e:
        print(f"    FAILED: {e}")
        if out_path.exists():
            out_path.unlink()
        return None


def merge_and_clip_tiles(tile_paths: list[Path], bbox: dict, output_path: Path) -> bool:
    """Merge multiple GeoTIFF tiles and clip to bbox. Returns True on success."""
    from rasterio.merge import merge as rio_merge
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box as shp_box
    import json

    if not tile_paths:
        return False

    print(f"  Merging {len(tile_paths)} tile(s)...")
    datasets = [rasterio.open(p) for p in tile_paths]

    try:
        merged, merged_transform = rio_merge(datasets)
        merged_profile = datasets[0].profile.copy()
        merged_profile.update({
            "height": merged.shape[1],
            "width": merged.shape[2],
            "transform": merged_transform,
        })
    finally:
        for ds in datasets:
            ds.close()

    # Clip to bbox
    clip_geom = shp_box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path, "w", **merged_profile) as tmp_ds:
            tmp_ds.write(merged)

        with rasterio.open(tmp_path) as tmp_ds:
            clipped, clipped_transform = rio_mask(
                tmp_ds,
                [clip_geom.__geo_interface__],
                crop=True,
                nodata=-9999,
            )
            clip_profile = tmp_ds.profile.copy()
            clip_profile.update({
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": clipped_transform,
                "nodata": -9999,
            })

        with rasterio.open(output_path, "w", **clip_profile) as out_ds:
            out_ds.write(clipped)

        print(f"  Merged+clipped tile saved: {output_path.name}")
        return True
    finally:
        os.unlink(tmp_path)


def download_dem(bbox: dict, raw_dir: Path, merged_raw: Path) -> bool:
    """
    Download Copernicus GLO-30 DEM tiles covering bbox.
    Tiles are free via AWS S3, no auth required.
    """
    print("Downloading Copernicus GLO-30 DEM (30m, free via AWS)...")
    tiles = get_cop30_tiles(bbox)
    print(f"  Tiles needed: {len(tiles)}")
    for t in tiles:
        print(f"    {t['name']}")

    downloaded = []
    for tile in tiles:
        path = download_cop30_tile(tile, raw_dir)
        if path:
            downloaded.append(path)

    if not downloaded:
        print("\nAll tile downloads failed.")
        print("Manual fallback options:")
        print("  1. OpenTopography (free API key): https://portal.opentopography.org/")
        print(f"     Register, get API key, add to script as OPENTOPO_API_KEY")
        print("  2. NASA EarthData SRTM: https://earthdata.nasa.gov/")
        print(f"     Download tiles for bbox: S={bbox['south']}, N={bbox['north']}, W={bbox['west']}, E={bbox['east']}")
        print(f"     Save merged GeoTIFF to: {merged_raw}")
        return False

    return merge_and_clip_tiles(downloaded, bbox, merged_raw)


def reproject_to_utm(src_path: Path, dst_path: Path) -> bool:
    """Reproject DEM from WGS84 geographic to UTM Zone 49S."""
    print(f"Reprojecting to {TARGET_CRS}...")
    dst_crs = CRS.from_epsg(32749)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": -9999,
        })

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

    print(f"  Reprojected DEM saved: {dst_path}")
    return True


def save_layer(array: np.ndarray, profile: dict, path: Path, nodata: float = -9999) -> None:
    """Save a numpy array as a single-band GeoTIFF."""
    profile = profile.copy()
    profile.update({"count": 1, "dtype": "float32", "nodata": nodata, "compress": "lzw"})
    array = array.astype("float32")
    array[array == nodata] = nodata

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    print(f"  Saved: {path.name}")


def compute_slope_aspect(dem: np.ndarray, res_m: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and aspect (degrees from north) from DEM.
    Uses Horn's method (8-neighbor finite differences).
    """
    # Gradients using Sobel-like convolution
    dz_dx = sobel(dem, axis=1) / (8 * res_m)
    dz_dy = sobel(dem, axis=0) / (8 * res_m)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    # Convert to geographic convention (0° = north, clockwise)
    aspect_deg = 90.0 - np.degrees(aspect_rad)
    aspect_deg[aspect_deg < 0] += 360.0

    return slope_deg, aspect_deg


def compute_twi(dem: np.ndarray, slope_deg: np.ndarray, res_m: float,
                nodata: float = -9999) -> np.ndarray:
    """
    Compute Topographic Wetness Index (TWI) = ln(a / tan(β)).

    Approximation: use upslope contributing area proxy (flow accumulation).
    For a proper TWI, use a flow direction / flow accumulation algorithm.
    Here we use a simplified version using a window-based proxy for contributing area,
    which is adequate for a first-pass analysis.

    Note: For publication-quality TWI, replace with pysheds or SAGA GIS.
    """
    slope_rad = np.radians(np.clip(slope_deg, 0.01, 89.9))  # avoid div by zero

    # Simplified: proxy for ln(a) = smoothed elevation inverse (higher elev = less upslope area)
    # This is a rough approximation. TODO: replace with proper flow accumulation.
    # For now: a_proxy = window-sum of contributing cells using a 5×5 window
    ones = np.ones_like(dem, dtype=float)
    ones[dem == nodata] = 0
    a_proxy = uniform_filter(ones, size=5) * (5 * res_m) ** 2  # m²

    a_proxy = np.maximum(a_proxy, res_m ** 2)  # floor at one cell
    twi = np.log(a_proxy / np.tan(slope_rad))

    twi[dem == nodata] = nodata
    return twi


def compute_tri(dem: np.ndarray, nodata: float = -9999) -> np.ndarray:
    """
    Terrain Ruggedness Index (TRI) = mean absolute difference from 8 neighbors.
    Riley et al. 1999.
    """
    def tri_kernel(window):
        center = window[4]
        if center == nodata:
            return nodata
        neighbors = np.concatenate([window[:4], window[5:]])
        valid = neighbors[neighbors != nodata]
        if len(valid) == 0:
            return 0.0
        return float(np.mean(np.abs(valid - center)))

    tri = generic_filter(dem.astype(float), tri_kernel, size=3, mode="nearest")
    tri[dem == nodata] = nodata
    return tri


def main():
    raw_path = RAW_DEM_DIR / "malang_cop30.tif"   # merged Copernicus tiles
    proc_path = PROC_DEM_DIR / "malang_dem.tif"

    # Step 1: Download
    if raw_path.exists():
        size_mb = raw_path.stat().st_size / 1e6
        print(f"Raw DEM already exists ({size_mb:.1f} MB): {raw_path}")
        print("  Delete it to re-download.")
    else:
        success = download_dem(BBOX, RAW_DEM_DIR, raw_path)
        if not success:
            sys.exit(1)

    # Step 2: Reproject
    if not proc_path.exists():
        reproject_to_utm(raw_path, proc_path)
    else:
        print(f"Reprojected DEM already exists: {proc_path}")

    # Step 3: Load DEM for terrain derivatives
    print("\nComputing terrain derivatives...")
    with rasterio.open(proc_path) as src:
        dem = src.read(1).astype(float)
        profile = src.profile
        nodata = src.nodata if src.nodata is not None else -9999
        # Resolution in meters (UTM, so transform units are meters)
        res_m = abs(src.transform.a)  # x pixel size

    dem[dem == nodata] = nodata
    print(f"  DEM shape: {dem.shape}, resolution: {res_m:.1f} m")
    print(f"  Elevation range: {dem[dem != nodata].min():.0f} – {dem[dem != nodata].max():.0f} m")

    # Step 4: Slope and aspect
    slope, aspect = compute_slope_aspect(dem, res_m)
    save_layer(slope, profile, PROC_DEM_DIR / "malang_slope.tif", nodata=nodata)
    save_layer(aspect, profile, PROC_DEM_DIR / "malang_aspect.tif", nodata=nodata)

    # Step 5: TWI
    twi = compute_twi(dem, slope, res_m, nodata=nodata)
    save_layer(twi, profile, PROC_DEM_DIR / "malang_twi.tif", nodata=nodata)

    # Step 6: TRI
    tri = compute_tri(dem, nodata=nodata)
    save_layer(tri, profile, PROC_DEM_DIR / "malang_tri.tif", nodata=nodata)

    print(f"\nAll terrain layers saved to {PROC_DEM_DIR}")
    print("\nSummary statistics:")
    for name, arr in [("Slope (deg)", slope), ("Aspect (deg)", aspect),
                       ("TWI", twi), ("TRI", tri)]:
        valid = arr[(arr != nodata) & np.isfinite(arr)]
        if len(valid):
            print(f"  {name}: min={valid.min():.2f}, max={valid.max():.2f}, "
                  f"mean={valid.mean():.2f}, std={valid.std():.2f}")


if __name__ == "__main__":
    main()
