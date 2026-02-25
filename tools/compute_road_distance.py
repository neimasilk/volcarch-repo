"""
compute_road_distance.py - Download OSM roads and compute distance raster.

Produces: data/processed/dem/jatim_road_dist.tif
  - Pixel values = distance in metres to nearest major road
  - Grid matches jatim_dem.tif (same extent, CRS, resolution)

Run from repo root:
    py tools/compute_road_distance.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import requests

try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import LineString, box
    from scipy.ndimage import distance_transform_edt
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent
DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"
OUT_PATH = DEM_DIR / "jatim_road_dist.tif"

# East Java bounding box (WGS84): south, west, north, east
BBOX = (-9.0, 111.0, -6.5, 115.0)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Major road classes as survey accessibility proxy.
ROAD_FILTER = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'


def download_osm_roads() -> gpd.GeoDataFrame:
    south, west, north, east = BBOX
    query = f"""
[out:json][timeout:300][bbox:{south},{west},{north},{east}];
(
  way{ROAD_FILTER};
  relation{ROAD_FILTER};
);
out geom;
"""
    print("Downloading roads from OSM Overpass API...")
    print(f"  Query bbox: {BBOX}")
    print(f"  Road filter: {ROAD_FILTER}")

    for attempt in range(3):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=360)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < 2:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying in 10s...")
                time.sleep(10)
            else:
                print(f"ERROR: Overpass request failed after retries: {e}")
                sys.exit(1)

    data = resp.json()
    elements = data.get("elements", [])
    print(f"  Downloaded {len(elements)} OSM elements")

    geometries = []
    names = []
    classes = []

    for el in elements:
        etype = el.get("type")
        tags = el.get("tags", {})
        hwy = tags.get("highway", "")
        name = tags.get("name", "")

        if etype == "way":
            coords = el.get("geometry", [])
            if len(coords) < 2:
                continue
            pts = [(c["lon"], c["lat"]) for c in coords]
            geometries.append(LineString(pts))
            names.append(name)
            classes.append(hwy)

        elif etype == "relation":
            members = el.get("members", [])
            for m in members:
                if m.get("type") != "way":
                    continue
                geom_pts = m.get("geometry", [])
                if len(geom_pts) < 2:
                    continue
                pts = [(c["lon"], c["lat"]) for c in geom_pts]
                geometries.append(LineString(pts))
                names.append(name)
                classes.append(hwy)

    if not geometries:
        print("ERROR: No road geometries parsed from Overpass response")
        sys.exit(1)

    gdf = gpd.GeoDataFrame(
        {"name": names, "highway": classes, "geometry": geometries},
        crs="EPSG:4326",
    )
    print(f"  Parsed {len(gdf)} road lines")
    return gdf


def compute_distance_raster(roads_gdf: gpd.GeoDataFrame) -> None:
    dem_path = DEM_DIR / "jatim_dem.tif"
    if not dem_path.exists():
        print(f"ERROR: Missing DEM at {dem_path}. Run E003 first.")
        sys.exit(1)

    with rasterio.open(dem_path) as src:
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width
        bounds = src.bounds

    pixel_size_m = abs(transform.a)
    print(f"DEM grid: {height}x{width}, CRS={crs}, pixel={pixel_size_m:.2f}m")

    roads_proj = roads_gdf.to_crs(crs)
    dem_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    roads_proj = roads_proj[roads_proj.intersects(dem_box)]
    roads_proj = roads_proj.copy()
    roads_proj["geometry"] = roads_proj.intersection(dem_box)
    roads_proj = roads_proj[~roads_proj.is_empty]

    if len(roads_proj) == 0:
        print("ERROR: No road geometries overlap DEM extent")
        sys.exit(1)

    print(f"  Roads in DEM extent: {len(roads_proj)}")
    shapes = [(geom, 1) for geom in roads_proj.geometry if geom is not None and not geom.is_empty]
    road_binary = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    n_road_pixels = int(road_binary.sum())
    print(f"  Road pixels burned: {n_road_pixels:,} ({n_road_pixels / (height * width) * 100:.2f}%)")

    bg_mask = road_binary == 0
    dist_pixels = distance_transform_edt(bg_mask)
    dist_metres = (dist_pixels * pixel_size_m).astype(np.float32)

    with rasterio.open(
        OUT_PATH,
        "w",
        driver="GTiff",
        dtype="float32",
        count=1,
        height=height,
        width=width,
        crs=crs,
        transform=transform,
        nodata=None,
        compress="lzw",
    ) as dst:
        dst.write(dist_metres, 1)

    print(f"  Distance range: {float(dist_metres.min()):.0f}-{float(dist_metres.max()):.0f} m")
    print(f"Saved: {OUT_PATH}")


def main() -> None:
    print("=" * 60)
    print("compute_road_distance.py")
    print("Downloads OSM major roads + computes proximity raster")
    print("=" * 60)

    if OUT_PATH.exists():
        print(f"\nOutput already exists: {OUT_PATH}")
        print("Delete it and re-run to regenerate.")
        return

    roads = download_osm_roads()
    compute_distance_raster(roads)
    print("\nDone. jatim_road_dist.tif is ready.")


if __name__ == "__main__":
    main()
