"""
tools/geocode_sites.py — Geocode 'no_coords' sites in east_java_sites.geojson
using OSM Nominatim API.

Strategy (in order):
  1. Nominatim query: "<name>, Jawa Timur, Indonesia"
  2. If no match: "<name>, Jawa, Indonesia"
  3. If no match: "<name>, Indonesia" (last resort — accept only if result in East Java bbox)

All results validated against East Java bbox (lat -9.5 to -6.5, lon 110.5 to 115.0).
Rate limit: 1 request/second (Nominatim ToS).

Usage:
    python tools/geocode_sites.py

Output:
    data/processed/east_java_sites.geojson  (updated in-place, existing coords preserved)
    data/processed/geocoding_report.txt     (per-site results)
"""

import json
import time
import sys
from pathlib import Path

try:
    import requests
    from shapely.geometry import Point
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# ---- Configuration ------------------------------------------------------------

REPO_ROOT   = Path(__file__).parent.parent
SITES_PATH  = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
REPORT_PATH = REPO_ROOT / "data" / "processed" / "geocoding_report.txt"

# East Java bounding box (with small buffer)
BBOX_LAT_MIN = -9.5
BBOX_LAT_MAX = -6.5
BBOX_LON_MIN = 110.5
BBOX_LON_MAX = 115.0

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {
    "User-Agent": "VOLCARCH-Research/1.0 (academic volcanic taphonomy study; contact via GitHub volcarch-repo)"
}
SLEEP_SEC = 1.1   # slightly over 1s to stay within ToS

# ---- Functions ----------------------------------------------------------------

def nominatim_search(query: str) -> list[dict]:
    """Query Nominatim and return list of result dicts."""
    params = {
        "q": query,
        "format": "json",
        "limit": 5,
        "addressdetails": 0,
    }
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return []


def in_east_java(lat: float, lon: float) -> bool:
    return (BBOX_LAT_MIN <= lat <= BBOX_LAT_MAX) and (BBOX_LON_MIN <= lon <= BBOX_LON_MAX)


def try_geocode(name: str) -> tuple[float | None, float | None, str]:
    """
    Try geocoding a site name using progressively looser queries.
    Returns (lat, lon, query_used) or (None, None, '').
    """
    queries = [
        f"{name}, Jawa Timur, Indonesia",
        f"{name}, Jawa, Indonesia",
        f"{name}, Indonesia",
    ]

    for query in queries:
        results = nominatim_search(query)
        time.sleep(SLEEP_SEC)

        for r in results:
            lat = float(r["lat"])
            lon = float(r["lon"])
            if in_east_java(lat, lon):
                return lat, lon, query

    return None, None, ""


def main():
    print("=" * 60)
    print("VOLCARCH Nominatim Geocoder")
    print("=" * 60)

    # Load GeoJSON
    with open(SITES_PATH, encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson["features"]
    print(f"Total features: {len(features)}")

    no_coords = [f for f in features if f["properties"].get("accuracy_level") == "no_coords"]
    print(f"Sites needing geocoding: {len(no_coords)}")

    if not no_coords:
        print("Nothing to do.")
        return

    # Geocode
    report_lines = [
        "VOLCARCH Geocoding Report",
        f"Date: 2026-02-23",
        f"Total sites to geocode: {len(no_coords)}",
        "-" * 60,
    ]

    found = 0
    not_found = 0

    for i, feat in enumerate(no_coords, 1):
        name = feat["properties"]["name"] or ""
        if not name.strip():
            not_found += 1
            report_lines.append(f"[{i:3d}] SKIP (no name)")
            continue

        print(f"[{i:3d}/{len(no_coords)}] {name[:50]:<50}", end=" ... ", flush=True)

        lat, lon, query_used = try_geocode(name)

        if lat is not None:
            # Update feature
            feat["geometry"] = {"type": "Point", "coordinates": [lon, lat]}
            feat["properties"]["lat"] = lat
            feat["properties"]["lon"] = lon
            feat["properties"]["accuracy_level"] = "nominatim"
            feat["properties"]["notes"] = (
                (feat["properties"].get("notes") or "") +
                f" | Geocoded via Nominatim: query='{query_used}'"
            ).strip(" |")

            found += 1
            print(f"FOUND ({lat:.4f}, {lon:.4f})")
            report_lines.append(f"[{i:3d}] FOUND  {name[:45]:<45} lat={lat:.4f} lon={lon:.4f}  query='{query_used}'")
        else:
            not_found += 1
            print("not found")
            report_lines.append(f"[{i:3d}] MISS   {name[:45]}")

    # Save updated GeoJSON
    with open(SITES_PATH, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"\nUpdated GeoJSON saved: {SITES_PATH}")

    # Save report
    report_lines += [
        "-" * 60,
        f"Found: {found}",
        f"Not found: {not_found}",
        f"Success rate: {found / len(no_coords) * 100:.1f}%",
        f"New total geocoded: {found + sum(1 for f in features if f['properties'].get('accuracy_level') in ('osm_centroid', 'wikidata_p625', 'nominatim'))}",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report saved: {REPORT_PATH}")

    print(f"\nResult: {found} / {len(no_coords)} geocoded ({found / len(no_coords) * 100:.1f}%)")


if __name__ == "__main__":
    main()
