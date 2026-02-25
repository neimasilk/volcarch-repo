"""
E012 helper: build expanded road-distance proxy raster for TGB.

Produces:
    data/processed/dem/jatim_road_dist_expanded.tif

Road classes:
    motorway, trunk, primary, secondary, tertiary,
    unclassified, residential, service

Run from repo root:
    py experiments/E012_settlement_model_v6/00_prepare_road_proxy_expanded.py
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
    from scipy.ndimage import distance_transform_edt
    from shapely.geometry import LineString, box
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"
DEM_PATH = DEM_DIR / "jatim_dem.tif"
OUT_PATH = DEM_DIR / "jatim_road_dist_expanded.tif"

# East Java bbox: south, west, north, east
BBOX = (-9.0, 111.0, -6.5, 115.0)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ROAD_FILTER = '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"]'


def download_roads() -> gpd.GeoDataFrame:
    south, west, north, east = BBOX
    query = f"""
[out:json][timeout:300][bbox:{south},{west},{north},{east}];
(
  way{ROAD_FILTER};
  relation{ROAD_FILTER};
);
out geom;
"""
    print("Downloading expanded OSM road classes...")
    print(f"  BBOX: {BBOX}")
    print(f"  Filter: {ROAD_FILTER}")

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
                print(f"ERROR: Overpass request failed after retries: {e}")
                sys.exit(1)

    data = resp.json()
    elements = data.get("elements", [])
    print(f"  Downloaded {len(elements)} OSM elements")

    geoms, hwy_types, names = [], [], []
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
            geoms.append(LineString(pts))
            hwy_types.append(hwy)
            names.append(name)
        elif etype == "relation":
            for m in el.get("members", []):
                if m.get("type") != "way":
                    continue
                geom_pts = m.get("geometry", [])
                if len(geom_pts) < 2:
                    continue
                pts = [(c["lon"], c["lat"]) for c in geom_pts]
                geoms.append(LineString(pts))
                hwy_types.append(hwy)
                names.append(name)

    if not geoms:
        print("ERROR: No road geometries parsed.")
        sys.exit(1)

    gdf = gpd.GeoDataFrame(
        {"name": names, "highway": hwy_types, "geometry": geoms},
        crs="EPSG:4326",
    )
    print(f"  Parsed {len(gdf)} road lines")
    return gdf


def rasterize_distance(roads: gpd.GeoDataFrame) -> None:
    if not DEM_PATH.exists():
        print(f"ERROR: Missing DEM at {DEM_PATH}")
        sys.exit(1)

    with rasterio.open(DEM_PATH) as src:
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width
        bounds = src.bounds

    pixel_size = abs(transform.a)
    dem_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    roads_proj = roads.to_crs(crs)
    roads_proj = roads_proj[roads_proj.intersects(dem_box)].copy()
    roads_proj["geometry"] = roads_proj.intersection(dem_box)
    roads_proj = roads_proj[~roads_proj.is_empty]
    if len(roads_proj) == 0:
        print("ERROR: No roads overlap DEM extent")
        sys.exit(1)

    print(f"  Roads in DEM extent: {len(roads_proj)}")
    road_counts = roads_proj["highway"].value_counts().to_dict()
    print("  Highway class counts in extent:")
    for cls, n in sorted(road_counts.items(), key=lambda kv: kv[0]):
        print(f"    {cls}: {n:,}")

    shapes = [(geom, 1) for geom in roads_proj.geometry if geom is not None and not geom.is_empty]
    road_mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    n_road_pixels = int(road_mask.sum())
    print(f"  Road pixels burned: {n_road_pixels:,} ({n_road_pixels/(height*width)*100:.2f}%)")

    dist_pix = distance_transform_edt(road_mask == 0)
    dist_m = (dist_pix * pixel_size).astype(np.float32)

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
        dst.write(dist_m, 1)

    print(f"  Distance range: {dist_m.min():.0f}-{dist_m.max():.0f} m")
    print(f"Saved: {OUT_PATH}")


def main() -> None:
    print("=" * 60)
    print("E012 road proxy preparation")
    print("=" * 60)

    if OUT_PATH.exists():
        print(f"\nOutput already exists: {OUT_PATH}")
        print("Delete it and re-run if you want to regenerate.")
        return

    roads = download_roads()
    rasterize_distance(roads)
    print("\nDone.")


if __name__ == "__main__":
    main()
