"""
compute_river_distance.py — Download OSM waterways and compute distance raster.

Produces: data/processed/dem/jatim_river_dist.tif
  - Pixel values = distance in metres to nearest named river/stream
  - Grid matches jatim_dem.tif (same extent, CRS, resolution)

Run from repo root:
    python tools/compute_river_distance.py

Dependencies: requests, geopandas, rasterio, shapely, scipy, numpy
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import requests

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.features import rasterize
    from shapely.geometry import LineString, MultiLineString, shape
    from scipy.ndimage import distance_transform_edt
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent
DEM_DIR   = REPO_ROOT / "data" / "processed" / "dem"
OUT_PATH  = DEM_DIR / "jatim_river_dist.tif"

# East Java bounding box (WGS84): south, west, north, east
BBOX = (-9.0, 111.0, -6.5, 115.0)

# Overpass API endpoint (public instance)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Waterway types to include — major rivers only for first pass
# Using 'river' and 'canal' (not 'stream') to keep dataset manageable
WATERWAY_FILTER = '["waterway"~"river|canal"]'


def download_osm_waterways() -> gpd.GeoDataFrame:
    """
    Download waterway linestrings from OSM Overpass API for East Java.
    Returns GeoDataFrame in WGS84 (EPSG:4326).
    """
    south, west, north, east = BBOX

    # Overpass QL query — ways with waterway tag in bbox
    query = f"""
[out:json][timeout:300][bbox:{south},{west},{north},{east}];
(
  way{WATERWAY_FILTER};
  relation{WATERWAY_FILTER};
);
out geom;
"""
    print("Downloading waterways from OSM Overpass API...")
    print(f"  Query bbox: {BBOX}")
    print(f"  Waterway filter: {WATERWAY_FILTER}")

    for attempt in range(3):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=360)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < 2:
                print(f"  Attempt {attempt+1} failed: {e}. Retrying in 10s...")
                time.sleep(10)
            else:
                print(f"  All attempts failed: {e}")
                sys.exit(1)

    data = resp.json()
    elements = data.get("elements", [])
    print(f"  Downloaded {len(elements)} OSM elements")

    # Parse way geometries
    geometries = []
    names = []
    types = []

    for el in elements:
        el_type = el.get("type")
        tags = el.get("tags", {})
        wtype = tags.get("waterway", "")
        name = tags.get("name", "")

        if el_type == "way":
            coords = el.get("geometry", [])
            if len(coords) < 2:
                continue
            pts = [(c["lon"], c["lat"]) for c in coords]
            geometries.append(LineString(pts))
            names.append(name)
            types.append(wtype)

        elif el_type == "relation":
            # Relations can have member ways
            members = el.get("members", [])
            member_lines = []
            for m in members:
                if m.get("type") == "way":
                    geom_pts = m.get("geometry", [])
                    if len(geom_pts) >= 2:
                        member_lines.append([(c["lon"], c["lat"]) for c in geom_pts])
            if member_lines:
                # Create separate linestrings per member
                for pts in member_lines:
                    geometries.append(LineString(pts))
                    names.append(name)
                    types.append(wtype)

    if not geometries:
        print("ERROR: No waterway geometries found in OSM response")
        sys.exit(1)

    gdf = gpd.GeoDataFrame(
        {"name": names, "waterway": types, "geometry": geometries},
        crs="EPSG:4326"
    )
    print(f"  Parsed {len(gdf)} waterway lines (rivers + canals)")
    return gdf


def compute_distance_raster(waterways_gdf: gpd.GeoDataFrame) -> None:
    """
    Burn waterway lines onto the DEM grid, then compute Euclidean distance
    to nearest river pixel for every cell.

    Saves result to OUT_PATH as float32 GeoTIFF (values in metres).
    """
    dem_path = DEM_DIR / "jatim_dem.tif"
    if not dem_path.exists():
        print(f"ERROR: DEM not found at {dem_path}. Run E003 first.")
        sys.exit(1)

    # Open DEM to get grid metadata
    with rasterio.open(dem_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        bounds = src.bounds

    print(f"DEM grid: {height}x{width}, CRS: {crs}")
    pixel_size_m = abs(transform.a)  # metres per pixel (UTM)
    print(f"  Pixel size: {pixel_size_m:.1f} m")

    # Reproject waterways to match DEM CRS
    print("Reprojecting waterways to DEM CRS...")
    waterways_proj = waterways_gdf.to_crs(crs)

    # Filter to DEM bounds (should already be within bbox, but be safe)
    from shapely.geometry import box
    dem_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    waterways_proj = waterways_proj[waterways_proj.intersects(dem_box)]
    print(f"  {len(waterways_proj)} waterway lines within DEM extent")

    if len(waterways_proj) == 0:
        print("WARNING: No waterways overlap DEM extent. Check CRS and bbox.")
        sys.exit(1)

    # Clip to DEM bounds
    waterways_proj = waterways_proj.copy()
    waterways_proj["geometry"] = waterways_proj.intersection(dem_box)
    waterways_proj = waterways_proj[~waterways_proj.is_empty]

    # Rasterize: burn 1 where waterway exists, 0 elsewhere
    print("Rasterizing waterway lines...")
    shapes_iter = [(geom, 1) for geom in waterways_proj.geometry if geom is not None and not geom.is_empty]

    river_binary = rasterize(
        shapes=shapes_iter,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    n_river_pixels = river_binary.sum()
    print(f"  River pixels burned: {n_river_pixels:,} ({n_river_pixels/height/width*100:.1f}% of grid)")

    if n_river_pixels == 0:
        print("WARNING: No river pixels burned. Distance raster will be all-zeros.")

    # Distance transform: distance in pixels to nearest 1
    # Invert: 0 → background (True = background), 1 → river (False = river pixel)
    print("Computing Euclidean distance transform...")
    bg_mask = river_binary == 0   # True where NOT a river pixel
    dist_pixels = distance_transform_edt(bg_mask)

    # Convert pixels → metres
    dist_metres = (dist_pixels * pixel_size_m).astype(np.float32)

    print(f"  Distance range: {dist_metres.min():.0f}–{dist_metres.max():.0f} m")
    print(f"  Mean distance: {dist_metres.mean():.0f} m")

    # Save as GeoTIFF
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        OUT_PATH, "w",
        driver="GTiff",
        dtype="float32",
        count=1,
        height=height, width=width,
        crs=crs,
        transform=transform,
        nodata=None,
        compress="lzw"
    ) as dst:
        dst.write(dist_metres, 1)

    print(f"\nSaved: {OUT_PATH}")


def main():
    print("=" * 60)
    print("compute_river_distance.py")
    print("Downloads OSM waterways + computes proximity raster")
    print("=" * 60)

    if OUT_PATH.exists():
        print(f"\nOutput already exists: {OUT_PATH}")
        print("Delete it and re-run to regenerate.")
        return

    waterways = download_osm_waterways()
    compute_distance_raster(waterways)

    print("\nDone. jatim_river_dist.tif is ready for E008.")


if __name__ == "__main__":
    main()
