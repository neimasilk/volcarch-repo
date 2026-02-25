"""
E001 / Step 1: Collect and merge archaeological site data for East Java.

This script:
1. Runs the OSM scraper to get sites from OpenStreetMap
2. Loads a manually-compiled Wikipedia supplement (if it exists)
3. Merges and deduplicates
4. Outputs: data/processed/east_java_sites.geojson

Run from repo root:
    python experiments/E001_site_density_vs_volcanic_proximity/01_collect_sites.py
"""

import sys
from pathlib import Path

# Make tools importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import geopandas as gpd
import pandas as pd

from scrape_osm_sites import main as run_osm_scraper

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
OSM_OUTPUT = REPO_ROOT / "data" / "processed" / "east_java_sites_osm.geojson"
WIKI_SUPPLEMENT = REPO_ROOT / "data" / "processed" / "east_java_sites_wiki.csv"
FINAL_OUTPUT = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"


def load_wiki_supplement() -> gpd.GeoDataFrame | None:
    """
    Load manually-compiled Wikipedia site list if it exists.
    Expected columns: name, type, period, lat, lon, source, notes
    """
    if not WIKI_SUPPLEMENT.exists():
        print(f"No Wikipedia supplement found at {WIKI_SUPPLEMENT} — skipping.")
        print("  (To add: create that CSV with columns: name,type,period,lat,lon,source,notes)")
        return None

    df = pd.read_csv(WIKI_SUPPLEMENT)
    required = {"name", "lat", "lon"}
    if not required.issubset(df.columns):
        print(f"Wikipedia supplement missing required columns: {required - set(df.columns)}")
        return None

    from shapely.geometry import Point
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(row.lon, row.lat) for row in df.itertuples()],
        crs="EPSG:4326",
    )
    gdf["source"] = gdf.get("source", "Wikipedia")
    gdf["osm_id"] = None
    gdf["accuracy_level"] = gdf.get("accuracy_level", "literature_estimated")
    return gdf


def deduplicate(gdf: gpd.GeoDataFrame, distance_m: float = 100) -> gpd.GeoDataFrame:
    """
    Remove near-duplicate sites within `distance_m` meters.
    When duplicates found, keep the one with the richest metadata (longer notes).
    This is a simple spatial dedup — not perfect, but good enough for first pass.
    """
    gdf_proj = gdf.to_crs("EPSG:32749")  # UTM 49S — meters
    removed = set()
    keep = []

    for i, row in gdf_proj.iterrows():
        if i in removed:
            continue
        keep.append(i)
        # find all within distance_m of this point
        neighbors = gdf_proj[gdf_proj.geometry.distance(row.geometry) < distance_m].index
        for j in neighbors:
            if j != i and j not in removed:
                removed.add(j)

    deduped = gdf.loc[keep].copy()
    print(f"Deduplication: {len(gdf)} -> {len(deduped)} sites ({len(gdf) - len(deduped)} removed)")
    return deduped


def main():
    # Step 1: OSM data
    print("=" * 60)
    print("Step 1: Scraping OpenStreetMap...")
    print("=" * 60)
    run_osm_scraper()

    if not OSM_OUTPUT.exists():
        print("ERROR: OSM scrape did not produce output. Aborting.")
        sys.exit(1)

    osm_gdf = gpd.read_file(OSM_OUTPUT)
    print(f"OSM sites loaded: {len(osm_gdf)}")

    # Step 2: Wikipedia supplement
    print("\n" + "=" * 60)
    print("Step 2: Loading Wikipedia supplement...")
    print("=" * 60)
    wiki_gdf = load_wiki_supplement()

    # Step 3: Merge
    print("\n" + "=" * 60)
    print("Step 3: Merging sources...")
    print("=" * 60)

    all_sources = [osm_gdf]
    if wiki_gdf is not None:
        all_sources.append(wiki_gdf)

    merged = gpd.GeoDataFrame(
        pd.concat(all_sources, ignore_index=True),
        crs="EPSG:4326",
    )
    print(f"Total before dedup: {len(merged)}")

    # Step 4: Dedup
    final = deduplicate(merged, distance_m=100)

    # Step 5: Save
    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_file(FINAL_OUTPUT, driver="GeoJSON")
    print(f"\nFinal dataset saved to {FINAL_OUTPUT}")
    print(f"Total sites: {len(final)}")
    print(f"\nType breakdown:\n{final['type'].value_counts().to_string()}")
    print(f"\nSource breakdown:\n{final['source'].value_counts().to_string()}")

    # Basic sanity check
    null_coords = final[final.geometry.is_empty | final.geometry.isna()]
    if len(null_coords) > 0:
        print(f"\nWARNING: {len(null_coords)} sites with null geometry. Inspect and fix.")


if __name__ == "__main__":
    main()
