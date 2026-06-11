#!/usr/bin/env python3
"""
E209 Phase 1, Step 02: Download Sentinel-2, Sentinel-1, and Copernicus DEM bands for
each training site.

Inputs:
  - data/training_sites.csv (from step 01)

Outputs:
  - data/s2_bands/{site_id}_{season}.tif          # 10m Sentinel-2 L2A stack (7 bands)
  - data/s1_bands/{site_id}_{season}.tif          # Sentinel-1 GRD VV+VH
  - data/dem/{site_id}.tif                        # Copernicus GLO-30 tile

Per-site: 1000m × 1000m tile centred on (lat, lon) — gives 100×100 at 10m Sentinel
resolution, 33×33 at 30m DEM.

Seasons:
  - dry_season: July–September (SE Asia dry)
  - wet_season: January–March
  Uses 3-year median to reduce cloud noise.

Data sources:
  - Sentinel-2 L2A via Microsoft Planetary Computer STAC API (free, signed URLs)
  - Sentinel-1 GRD via Planetary Computer STAC
  - Copernicus GLO-30 DEM via Planetary Computer STAC

Runtime estimate:
  - 121 sites × 2 seasons × ~3 bands = ~726 tile requests
  - At ~5 sec/tile download + process = ~1 hour real time
  - Network and disk: ~10-20 GB total

This script is checkpointed: re-running skips already-downloaded tiles.

TO EXECUTE:
  python scripts/02_download_satellite_bands.py [--sites N] [--seasons dry,wet]

Dependencies:
  pip install rasterio requests planetary-computer pystac-client

NOT executed in initial scaffolding commit — run when ready for full data pull.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Placeholder imports — activate when running
# import rasterio
# import requests
# from pystac_client import Client
# import planetary_computer

REPO_ROOT = Path(__file__).resolve().parents[3]
E209_DIR = Path(__file__).resolve().parents[1]
SITES_CSV = E209_DIR / "data" / "training_sites.csv"
S2_DIR = E209_DIR / "data" / "s2_bands"
S1_DIR = E209_DIR / "data" / "s1_bands"
DEM_DIR = E209_DIR / "data" / "dem"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

SEASONS = {
    "dry_season": {"months": [7, 8, 9], "years": [2022, 2023, 2024]},
    "wet_season": {"months": [1, 2, 3], "years": [2023, 2024, 2025]},
}

# Sentinel-2 bands to retrieve
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]  # blue, green, red, NIR, SWIR1, SWIR2, scene classification

# Tile size (metres) — 1000m × 1000m
TILE_SIZE_M = 1000.0


def load_sites(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def make_bbox(lat: float, lon: float, size_m: float = TILE_SIZE_M) -> tuple[float, float, float, float]:
    """Create a bbox around (lat, lon) of approximately `size_m` × `size_m` metres."""
    # Approximate: 1 deg latitude ~ 111 km; longitude ~ 111 * cos(lat) km
    import math
    dlat = (size_m / 2.0) / 111_000.0
    dlon = (size_m / 2.0) / (111_000.0 * max(0.1, math.cos(math.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def download_s2_stack(site: dict, season: str) -> Path | None:
    """Download a Sentinel-2 L2A median composite for a site+season. Returns path or None."""
    site_id = site["site_id"]
    out = S2_DIR / f"{site_id}_{season}.tif"
    if out.exists():
        return out  # checkpointed
    # ...actual STAC search + download logic here; see E189/run_core.py for pattern
    # Placeholder — not implemented in scaffolding commit
    return None


def download_s1_stack(site: dict, season: str) -> Path | None:
    """Download a Sentinel-1 GRD VV+VH median composite. Returns path or None."""
    site_id = site["site_id"]
    out = S1_DIR / f"{site_id}_{season}.tif"
    if out.exists():
        return out
    return None


def download_dem(site: dict) -> Path | None:
    """Download Copernicus GLO-30 DEM tile for a site. Returns path or None."""
    site_id = site["site_id"]
    out = DEM_DIR / f"{site_id}.tif"
    if out.exists():
        return out
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="E209 Step 02 — satellite data pull")
    parser.add_argument("--sites", type=int, default=None,
                        help="Limit to first N sites (default: all).")
    parser.add_argument("--seasons", default="dry_season,wet_season",
                        help="Comma-separated season names.")
    parser.add_argument("--skip-s2", action="store_true", help="Skip Sentinel-2 download.")
    parser.add_argument("--skip-s1", action="store_true", help="Skip Sentinel-1 download.")
    parser.add_argument("--skip-dem", action="store_true", help="Skip DEM download.")
    args = parser.parse_args()

    for d in [S2_DIR, S1_DIR, DEM_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    sites = load_sites(SITES_CSV)
    if args.sites:
        sites = sites[: args.sites]

    seasons = args.seasons.split(",")

    print(f"E209 Step 02: Satellite data pull for {len(sites)} sites × {len(seasons)} seasons")
    print("=" * 60)
    print(f"Sentinel-2:  {'SKIP' if args.skip_s2 else 'ON'}")
    print(f"Sentinel-1:  {'SKIP' if args.skip_s1 else 'ON'}")
    print(f"Copernicus DEM: {'SKIP' if args.skip_dem else 'ON'}")
    print()

    print("NOTE: This scaffolding commit does NOT execute downloads.")
    print("To activate, implement download_s2_stack / download_s1_stack / download_dem")
    print("using the STAC client pattern from E189 run_core.py.")
    print()
    print("For now, the script validates the site list and directory structure.")

    # Validate
    for site in sites:
        lat, lon = float(site["lat"]), float(site["lon"])
        bbox = make_bbox(lat, lon, TILE_SIZE_M)
        # Just print first 5
        if sites.index(site) < 5:
            print(f"  {site['site_id']:10s} {site['name'][:30]:30s} ({lat:.4f}, {lon:.4f})")
            print(f"    bbox: {bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}")

    print(f"  ... ({len(sites) - 5} more sites)" if len(sites) > 5 else "")
    print()
    print(f"Ready for download execution. Total sites: {len(sites)}")
    print(f"Estimated runtime at 5 sec/tile: ~{len(sites) * len(seasons) * 3 * 5 / 60:.0f} minutes")


if __name__ == "__main__":
    main()
