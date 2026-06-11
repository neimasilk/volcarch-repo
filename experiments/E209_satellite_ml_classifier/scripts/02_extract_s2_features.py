#!/usr/bin/env python3
"""
E209 Phase 1, Step 02: Extract Sentinel-2 spectral features per site (dry + wet season).

Ports the working STAC pipeline from E189/run_core.py and applies it to the full
E209 training-site list (121 sites), computing multi-season spectral features
for classifier training.

Inputs:
  - data/training_sites.csv (from step 01)

Outputs:
  - data/features_s2.csv
    Columns: site_id, name, lat, lon, label, class, <feature_1>, <feature_2>, ...

Features per site (~30 from S2 alone, per season):
  - NDVI mean/std/center/ring/diff/local_variance
  - NDWI mean/std/center/ring/diff/local_variance
  - MSAVI mean/std/center/diff
  - Clay ratio (B11/B12) mean
  - Iron oxide (B04/B03) mean
  - Cloud cover at best scene
  - Seasonal delta (wet_season_value - dry_season_value) for all above

Design notes:
  - 1000m × 1000m tiles (100×100 at 10m Sentinel-2 resolution)
  - Planetary Computer STAC API (free, signed URLs)
  - Per-scene nodata rejection (threshold 50%)
  - Checkpointing: re-running skips sites already in features_s2.csv
  - Failures logged but don't halt the run

Runtime estimate: ~60-90 min for 121 sites × 2 seasons on typical home connection.
Test run: `python 02_extract_s2_features.py --limit 11` first to validate.

Usage:
  python 02_extract_s2_features.py [--limit N] [--seasons dry,wet]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from scipy.ndimage import zoom

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

E209_DIR = Path(__file__).resolve().parents[1]
SITES_CSV = E209_DIR / "data" / "training_sites.csv"
FEATURES_CSV = E209_DIR / "data" / "features_s2.csv"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"

SEASONS = {
    "dry": [
        "2024-07-01/2024-09-30",
        "2023-07-01/2023-09-30",
        "2022-07-01/2022-09-30",
    ],
    "wet": [
        "2024-01-01/2024-03-31",
        "2023-01-01/2023-03-31",
        "2025-01-01/2025-03-31",
    ],
}

TILE_BUFFER_M = 500  # 1000m × 1000m window


def load_sites(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def search_scenes(bbox: list, date_ranges: list, min_scenes: int = 5) -> list:
    """Search STAC for Sentinel-2 L2A scenes in bbox + any of the date ranges.

    Uses limit=100 per range + paginates via MGRS tile to ensure coverage across
    multiple tiles (Java spans multiple S2 granules).
    """
    scenes = []
    for dr in date_ranges:
        body = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": dr,
            "limit": 100,
            "query": {"eo:cloud_cover": {"lt": 25}},
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        }
        try:
            resp = requests.post(f"{STAC_URL}/search", json=body, timeout=60)
            found = resp.json().get("features", [])
            scenes.extend(found)
        except Exception as e:
            print(f"    STAC search error for {dr}: {e}")
        # Ensure tile diversity: stop when we have scenes from >= 5 distinct MGRS tiles
        tiles = {s["properties"].get("s2:mgrs_tile", "?") for s in scenes}
        if len(tiles) >= 5 and len(scenes) >= min_scenes * 2:
            break
    scenes.sort(key=lambda s: s["properties"].get("eo:cloud_cover", 100))
    return scenes


def get_sas_token() -> str:
    """Fetch a Planetary Computer SAS token for Sentinel-2 L2A."""
    try:
        return requests.get(SIGN_URL, timeout=10).json().get("token", "")
    except Exception as e:
        print(f"    Token fetch failed: {e}")
        return ""


def read_band(scene, band_name, cx, cy, token, buffer_m=TILE_BUFFER_M):
    """Read a band window from a signed STAC asset. Returns array or None."""
    import rasterio
    from rasterio.windows import from_bounds

    href = scene["assets"].get(band_name, {}).get("href")
    if not href:
        return None
    signed = f"{href}&{token}" if "?" in href else f"{href}?{token}"
    try:
        with rasterio.open(signed) as src:
            if not (src.bounds.left <= cx <= src.bounds.right and
                    src.bounds.bottom <= cy <= src.bounds.top):
                return None
            w = from_bounds(cx - buffer_m, cy - buffer_m,
                            cx + buffer_m, cy + buffer_m, src.transform)
            data = src.read(1, window=w).astype(float)
            valid_frac = np.count_nonzero(data) / max(data.size, 1)
            if valid_frac < 0.5:
                return None
            return data
    except Exception:
        return None


def resize_match(src, tgt):
    if src.shape == tgt.shape:
        return src
    return zoom(src,
                (tgt.shape[0] / src.shape[0], tgt.shape[1] / src.shape[1]),
                order=0)


def compute_site_features(site: dict, scenes: list, token: str,
                          transformer) -> Optional[dict]:
    """For one site, find first usable scene and compute feature dict."""
    lat = float(site["lat"])
    lon = float(site["lon"])
    cx, cy = transformer.transform(lon, lat)

    for scene in scenes:
        b04 = read_band(scene, "B04", cx, cy, token)
        if b04 is None:
            continue
        b08 = read_band(scene, "B08", cx, cy, token)
        b03 = read_band(scene, "B03", cx, cy, token)
        if b08 is None or b03 is None:
            continue
        # Try SWIR for clay ratio (optional)
        b11 = read_band(scene, "B11", cx, cy, token)
        b12 = read_band(scene, "B12", cx, cy, token)

        # Compute indices
        d = b08 + b04
        ndvi = np.where(d > 0, (b08 - b04) / d, np.nan)
        b03r = resize_match(b03, b08)
        d2 = b03r + b08
        ndwi = np.where(d2 > 0, (b03r - b08) / d2, np.nan)
        term = (2 * b08 + 1) ** 2 - 8 * (b08 - b04)
        term = np.maximum(term, 0)
        msavi = (2 * b08 + 1 - np.sqrt(term)) / 2

        h, w = ndvi.shape
        if h < 10 or w < 10:
            continue

        # Center vs ring
        ch, cw = max(1, h // 5), max(1, w // 5)
        nc = ndvi[2 * ch:3 * ch, 2 * cw:3 * cw]
        nr = np.concatenate([
            ndvi[:ch, :].flatten(),
            ndvi[-ch:, :].flatten(),
            ndvi[ch:-ch, :cw].flatten(),
            ndvi[ch:-ch, -cw:].flatten(),
        ])
        wc = ndwi[2 * ch:3 * ch, 2 * cw:3 * cw]
        wr = np.concatenate([
            ndwi[:ch, :].flatten(),
            ndwi[-ch:, :].flatten(),
            ndwi[ch:-ch, :cw].flatten(),
            ndwi[ch:-ch, -cw:].flatten(),
        ])
        mc = msavi[2 * ch:3 * ch, 2 * cw:3 * cw]
        mr = np.concatenate([
            msavi[:ch, :].flatten(),
            msavi[-ch:, :].flatten(),
            msavi[ch:-ch, :cw].flatten(),
            msavi[ch:-ch, -cw:].flatten(),
        ])

        # Local variance (3x3)
        lvars_ndvi = []
        lvars_ndwi = []
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                v = np.nanvar(ndvi[i - 1:i + 2, j - 1:j + 2])
                if np.isfinite(v):
                    lvars_ndvi.append(v)
                v2 = np.nanvar(ndwi[i - 1:i + 2, j - 1:j + 2])
                if np.isfinite(v2):
                    lvars_ndwi.append(v2)

        feats = {
            "ndvi_mean": float(np.nanmean(ndvi)),
            "ndvi_std": float(np.nanstd(ndvi)),
            "ndvi_center": float(np.nanmean(nc)),
            "ndvi_ring": float(np.nanmean(nr)),
            "ndvi_diff": float(np.nanmean(nc) - np.nanmean(nr)),
            "ndvi_lvar": float(np.mean(lvars_ndvi)) if lvars_ndvi else np.nan,
            "ndwi_mean": float(np.nanmean(ndwi)),
            "ndwi_std": float(np.nanstd(ndwi)),
            "ndwi_center": float(np.nanmean(wc)),
            "ndwi_ring": float(np.nanmean(wr)),
            "ndwi_diff": float(np.nanmean(wc) - np.nanmean(wr)),
            "ndwi_lvar": float(np.mean(lvars_ndwi)) if lvars_ndwi else np.nan,
            "msavi_mean": float(np.nanmean(msavi)),
            "msavi_center": float(np.nanmean(mc)),
            "msavi_diff": float(np.nanmean(mc) - np.nanmean(mr)),
            "cloud_pct": float(scene["properties"].get("eo:cloud_cover", -1)),
            "scene_date": str(scene["properties"].get("datetime", ""))[:10],
        }

        # Clay + iron ratios if SWIR available
        if b11 is not None and b12 is not None:
            b11r = resize_match(b11, b04)
            b12r = resize_match(b12, b04)
            # Safe division
            clay = np.where(b12r > 0, b11r / b12r, np.nan)
            iron = np.where(b03 > 0, b04 / b03, np.nan)
            feats["clay_ratio"] = float(np.nanmean(clay))
            feats["iron_oxide"] = float(np.nanmean(iron))
        else:
            feats["clay_ratio"] = np.nan
            feats["iron_oxide"] = np.nan

        return feats

    return None


def load_existing_features(path: Path) -> set:
    """Return set of (site_id, season) already processed for checkpointing."""
    if not path.exists():
        return set()
    done = set()
    try:
        with open(path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                done.add((row.get("site_id", ""), row.get("season", "")))
    except Exception:
        pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N sites (for test runs).")
    parser.add_argument("--seasons", default="dry,wet",
                        help="Comma-separated season keys.")
    parser.add_argument("--min-class", type=int, default=-2,
                        help="Minimum site class to process (exclude class=0 etc.).")
    args = parser.parse_args()

    from pyproj import Transformer
    # Java UTM zone 49S
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)

    sites = [s for s in load_sites(SITES_CSV)
             if int(s.get("class", "0") or "0") >= args.min_class
             and int(s.get("class", "0") or "0") != 0]
    if args.limit:
        # Prioritise hard positives/negatives first
        sites.sort(key=lambda s: (abs(int(s["class"] or "0")) != 2, s["site_id"]))
        sites = sites[: args.limit]

    seasons = args.seasons.split(",")
    done = load_existing_features(FEATURES_CSV)

    print(f"E209 Step 02: Sentinel-2 feature extraction")
    print("=" * 60)
    print(f"Sites: {len(sites)}")
    print(f"Seasons: {seasons}")
    print(f"Already done: {len(done)} site-season pairs (skipped)")
    print()

    # Fetch SAS token once
    print("Fetching PC SAS token...")
    token = get_sas_token()
    print(f"  Token length: {len(token)}")
    if not token:
        print("ABORT: no token")
        return

    # Field order for CSV
    base_fields = ["site_id", "name", "lat", "lon", "label", "class",
                   "season", "scene_date", "cloud_pct"]
    feat_fields = ["ndvi_mean", "ndvi_std", "ndvi_center", "ndvi_ring",
                   "ndvi_diff", "ndvi_lvar",
                   "ndwi_mean", "ndwi_std", "ndwi_center", "ndwi_ring",
                   "ndwi_diff", "ndwi_lvar",
                   "msavi_mean", "msavi_center", "msavi_diff",
                   "clay_ratio", "iron_oxide"]
    all_fields = base_fields + feat_fields

    # Prepare output (append mode, write header if fresh)
    mode = "a" if FEATURES_CSV.exists() and done else "w"
    with open(FEATURES_CSV, mode, encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=all_fields)
        if mode == "w":
            writer.writeheader()

        # Loop: for each season, find scenes covering all sites, then extract
        for season in seasons:
            date_ranges = SEASONS.get(season, [])
            if not date_ranges:
                print(f"  Unknown season: {season}")
                continue

            # Global bbox covering all sites (simpler than per-site scene lookup)
            lats = [float(s["lat"]) for s in sites]
            lons = [float(s["lon"]) for s in sites]
            bbox = [min(lons) - 0.1, min(lats) - 0.1,
                    max(lons) + 0.1, max(lats) + 0.1]

            print(f"\n=== Season: {season} ===")
            print(f"BBox: {bbox}")
            print("Searching scenes...")
            scenes = search_scenes(bbox, date_ranges, min_scenes=10)
            print(f"  Found {len(scenes)} scenes")
            if scenes:
                for i, s in enumerate(scenes[:3]):
                    p = s["properties"]
                    print(f"  [{i}] {p['datetime'][:10]} cloud={p.get('eo:cloud_cover', '?')}%")

            if not scenes:
                print(f"  NO SCENES for {season}, skipping season")
                continue

            ok = 0
            fail = 0
            t0 = time.time()
            for i, site in enumerate(sites):
                key = (site["site_id"], season)
                if key in done:
                    continue
                print(f"  [{i+1}/{len(sites)}] {site['site_id']:10s} {site['name'][:25]:25s} ({season}) ...",
                      end=" ", flush=True)
                feats = compute_site_features(site, scenes, token, transformer)
                if feats is None:
                    print("SKIP (no valid scene)")
                    fail += 1
                    continue
                row = {
                    "site_id": site["site_id"],
                    "name": site["name"],
                    "lat": site["lat"],
                    "lon": site["lon"],
                    "label": site["label"],
                    "class": site["class"],
                    "season": season,
                }
                row.update(feats)
                writer.writerow({k: row.get(k, "") for k in all_fields})
                fout.flush()
                ok += 1
                print(f"OK ndvi_diff={feats['ndvi_diff']:+.3f}")

            dt = time.time() - t0
            print(f"  Season {season}: {ok} ok, {fail} fail, {dt/60:.1f} min")

    print("\n" + "=" * 60)
    print(f"Features written to: {FEATURES_CSV}")
    print("Next: scripts/03_train_classifier.py")


if __name__ == "__main__":
    main()
