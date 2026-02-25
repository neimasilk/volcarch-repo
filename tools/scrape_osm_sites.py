"""
tools/scrape_osm_sites.py

Scrape archaeological sites in East Java from the OpenStreetMap Overpass API.

Usage:
    python tools/scrape_osm_sites.py

Output:
    data/processed/east_java_sites_osm.geojson

Notes:
    - Uses Overpass API (overpass-api.de) — free, no auth required.
    - Query covers Jawa Timur province bounding box.
    - Tags collected: historic=archaeological_site, historic=ruins,
      historic=monument, historic=candi (custom East Java usage)
    - OSM data is incomplete; treat as partial source, to be merged with
      Wikipedia and BPCB data in E001.
"""

import json
import time
import requests
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# --- Config ---
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Jawa Timur bounding box: south, west, north, east
JATIM_BBOX = (-8.8, 110.9, -6.8, 114.5)

# Tags to collect
TARGET_TAGS = [
    "archaeological_site",
    "ruins",
    "candi",        # Indonesian temples — sometimes tagged this way
    "monument",
]

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "east_java_sites_osm.geojson"


def build_overpass_query(bbox: tuple[float, float, float, float]) -> str:
    """Build an Overpass QL query for archaeological historic features in bbox."""
    s, w, n, e = bbox
    bbox_str = f"{s},{w},{n},{e}"

    tag_filters = "\n".join(
        f'  node["historic"="{tag}"]({bbox_str});'
        f'\n  way["historic"="{tag}"]({bbox_str});'
        f'\n  relation["historic"="{tag}"]({bbox_str});'
        for tag in TARGET_TAGS
    )

    query = f"""
[out:json][timeout:90];
(
{tag_filters}
);
out center tags;
"""
    return query


def fetch_overpass(query: str, retries: int = 3) -> dict:
    """POST query to Overpass API with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=120,
                headers={"User-Agent": "VOLCARCH-research/0.1 (academic; contact: volcarch-research)"},
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def parse_element(el: dict) -> dict | None:
    """
    Parse a single Overpass element into a flat feature dict.
    Returns None if no usable coordinates.
    """
    tags = el.get("tags", {})

    # Coordinates: nodes have lat/lon directly; ways/relations have center
    if el["type"] == "node":
        lat = el.get("lat")
        lon = el.get("lon")
    else:
        center = el.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")

    if lat is None or lon is None:
        return None

    name_id = tags.get("name") or tags.get("name:en") or tags.get("name:id") or f"osm_{el['id']}"

    # Try to infer archaeological type from tags
    historic_val = tags.get("historic", "unknown")
    site_type = {
        "archaeological_site": "archaeological_site",
        "ruins": "ruins",
        "candi": "candi",
        "monument": "monument",
    }.get(historic_val, historic_val)

    return {
        "name": name_id,
        "type": site_type,
        "period": tags.get("start_date") or tags.get("era") or "unknown",
        "lat": lat,
        "lon": lon,
        "source": "OpenStreetMap",
        "osm_id": str(el["id"]),
        "osm_type": el["type"],
        "discovery_year": None,
        "accuracy_level": "osm_centroid",  # centroid of OSM polygon, not field GPS
        "notes": tags.get("description") or tags.get("inscription") or "",
        "wikipedia": tags.get("wikipedia") or "",
        "wikidata": tags.get("wikidata") or "",
    }


def main():
    print("Building Overpass query...")
    query = build_overpass_query(JATIM_BBOX)

    print("Fetching from Overpass API (this may take 30–60 seconds)...")
    data = fetch_overpass(query)

    elements = data.get("elements", [])
    print(f"Raw elements returned: {len(elements)}")

    features = []
    for el in elements:
        parsed = parse_element(el)
        if parsed:
            features.append(parsed)

    print(f"Valid geocoded features: {len(features)}")

    if not features:
        print("WARNING: No features returned. Check bounding box or Overpass API status.")
        return

    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(
        features,
        geometry=[Point(f["lon"], f["lat"]) for f in features],
        crs="EPSG:4326",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"\nType breakdown:\n{gdf['type'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
