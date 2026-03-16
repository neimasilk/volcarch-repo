#!/usr/bin/env python3
"""
E076: Satellite NDVI Anomaly Detection at Candi Sites
=======================================================
Uses Sentinel-2 L2A data (10m resolution) from Microsoft Planetary Computer
to detect vegetation anomalies at known candi sites in East Java.

Buried archaeological structures affect surface vegetation through:
- Walls/foundations → reduced soil moisture → LOWER NDVI (stressed vegetation)
- Ditches/moats → increased moisture retention → HIGHER NDVI (vigorous vegetation)

Method:
1. Search for cloud-free dry-season Sentinel-2 imagery (July-September)
2. Extract NDVI windows around known candi sites
3. Compare NDVI statistics inside vs outside known candi footprints
4. Identify anomalous NDVI patterns at predicted buried-site locations

This is methodologically novel for Java — no published NDVI crop-mark
detection has been applied to candi sites.
"""

import json
import math
import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import requests

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Planetary Computer STAC API ───────────────────────────────────────
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"

# ── Target candi sites ───────────────────────────────────────────────
# Well-known candi with documented buried/partially buried structures
# Priority sites chosen for:
# (a) known subsurface features (from excavation reports)
# (b) proximity to volcanoes (Zone A from E065)
# (c) diverse terrain (volcanic slope, plain, river valley)

TARGET_CANDI = [
    # Name, lat, lon, description
    ("Candi Jawi", -7.7619, 112.7281, "Penanggungan slope, partially buried base"),
    ("Candi Tikus", -7.5589, 112.3756, "Trowulan sunken bathing complex, excavated from burial"),
    ("Candi Brahu", -7.5481, 112.3856, "Trowulan, large brick temple"),
    ("Candi Bajang Ratu", -7.5561, 112.3978, "Trowulan gate, partially restored"),
    ("Candi Wringin Lawang", -7.5397, 112.3908, "Trowulan, split gate"),
    ("Candi Kidal", -8.0156, 112.5928, "Malang, 13th century"),
    ("Candi Singosari", -7.8894, 112.6592, "Malang, Singhasari period"),
    ("Candi Sumberawan", -7.8403, 112.5456, "Malang, Buddhist stupa on slope"),
    ("Candi Songgoriti", -7.8714, 112.4894, "Batu, thermal spring temple"),
    ("Candi Surawana", -7.7756, 112.1494, "Kediri, 14th century"),
    ("Candi Penataran", -7.9250, 112.2069, "Blitar, largest temple complex in East Java"),
    ("Candi Sawentar", -7.9500, 112.1806, "Blitar, near Kelud"),
    ("Candi Gambar Wetan", -7.9769, 112.2597, "Near Kelud, volcanic burial site"),
    ("Candi Tegowangi", -7.7733, 112.1194, "Kediri, relief panels"),
    ("Situs Trowulan", -7.5500, 112.3800, "Capital of Majapahit, large complex"),
]

# Control sites: locations with NO known archaeological sites
# (for baseline NDVI comparison)
CONTROL_SITES = [
    ("Control_volc_1", -7.85, 112.40, "Volcanic slope, no known site"),
    ("Control_volc_2", -7.95, 112.85, "Volcanic plain, no known site"),
    ("Control_plain_1", -7.50, 112.50, "Alluvial plain, no known site"),
    ("Control_plain_2", -7.60, 112.20, "River valley, no known site"),
    ("Control_forest_1", -8.00, 113.00, "Forested hill, no known site"),
]


def search_sentinel2(bbox, date_range="2024-07-01/2024-09-30", max_cloud=15, max_items=10):
    """Search Planetary Computer STAC for Sentinel-2 scenes."""
    search_body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": date_range,
        "limit": max_items,
        "query": {
            "eo:cloud_cover": {"lt": max_cloud}
        },
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]
    }

    resp = requests.post(f"{STAC_URL}/search", json=search_body, timeout=30)
    if resp.status_code != 200:
        print(f"  STAC search failed: {resp.status_code}")
        return []

    features = resp.json().get('features', [])
    return features


def sign_url(asset_url):
    """Sign a Planetary Computer URL for access."""
    try:
        resp = requests.get(SIGN_URL, timeout=10)
        if resp.status_code == 200:
            token_data = resp.json()
            token = token_data.get('token', '')
            if '?' in asset_url:
                return f"{asset_url}&{token}"
            else:
                return f"{asset_url}?{token}"
    except Exception as e:
        print(f"  Token signing failed: {e}")
    return asset_url


def extract_ndvi_window(b04_url, b08_url, center_lat, center_lon, buffer_m=500):
    """
    Extract NDVI from B04 (Red) and B08 (NIR) bands around a point.
    Uses rasterio's windowed reading of Cloud-Optimized GeoTIFFs.
    Returns NDVI array and metadata.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds

        # Sign URLs
        b04_signed = sign_url(b04_url)
        b08_signed = sign_url(b08_url)

        # Convert lat/lon to UTM (Sentinel-2 data is in UTM)
        # For East Java, UTM Zone 49S (EPSG:32749)
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
        center_x, center_y = transformer.transform(center_lon, center_lat)

        # Define window bounds (buffer in meters)
        xmin = center_x - buffer_m
        xmax = center_x + buffer_m
        ymin = center_y - buffer_m
        ymax = center_y + buffer_m

        with rasterio.open(b04_signed) as src_b04:
            # Check if the point falls within this tile
            if not (src_b04.bounds.left <= center_x <= src_b04.bounds.right and
                    src_b04.bounds.bottom <= center_y <= src_b04.bounds.top):
                return None, "Point outside tile bounds"

            window = from_bounds(xmin, ymin, xmax, ymax, src_b04.transform)
            b04 = src_b04.read(1, window=window).astype(float)

        with rasterio.open(b08_signed) as src_b08:
            window = from_bounds(xmin, ymin, xmax, ymax, src_b08.transform)
            b08 = src_b08.read(1, window=window).astype(float)

        # Compute NDVI
        denominator = b08 + b04
        ndvi = np.where(denominator > 0, (b08 - b04) / denominator, 0)

        return ndvi, None

    except Exception as e:
        return None, str(e)


def analyze_ndvi_anomaly(ndvi, site_name):
    """
    Analyze NDVI array for archaeological anomalies.

    Approach: Compare center vs periphery.
    Archaeological features tend to create spatial anomalies —
    lower NDVI over buried walls, higher over buried ditches.
    """
    if ndvi is None or ndvi.size == 0:
        return None

    h, w = ndvi.shape
    if h < 10 or w < 10:
        return None

    # Define center zone (inner 40%) vs ring zone (outer 60%)
    ch, cw = h // 5, w // 5
    center = ndvi[2*ch:3*ch, 2*cw:3*cw]
    ring = np.concatenate([
        ndvi[:ch, :].flatten(),
        ndvi[-ch:, :].flatten(),
        ndvi[ch:-ch, :cw].flatten(),
        ndvi[ch:-ch, -cw:].flatten(),
    ])

    if center.size == 0 or ring.size == 0:
        return None

    # Statistics
    result = {
        'site': site_name,
        'ndvi_mean': round(float(np.mean(ndvi)), 4),
        'ndvi_std': round(float(np.std(ndvi)), 4),
        'ndvi_center_mean': round(float(np.mean(center)), 4),
        'ndvi_ring_mean': round(float(np.mean(ring)), 4),
        'ndvi_center_std': round(float(np.std(center)), 4),
        'ndvi_ring_std': round(float(np.std(ring)), 4),
        'center_ring_diff': round(float(np.mean(center) - np.mean(ring)), 4),
        'anomaly_ratio': round(float(np.std(center) / max(np.std(ring), 0.001)), 4),
        'n_pixels': int(ndvi.size),
        'center_pixels': int(center.size),
    }

    # Spatial heterogeneity (archaeological sites often have higher local variance)
    # Compute local variance in 3x3 windows
    if h >= 3 and w >= 3:
        local_vars = []
        for i in range(1, h-1):
            for j in range(1, w-1):
                patch = ndvi[i-1:i+2, j-1:j+2]
                local_vars.append(np.var(patch))
        result['mean_local_variance'] = round(float(np.mean(local_vars)), 6)
    else:
        result['mean_local_variance'] = None

    return result


def main():
    print("=" * 70)
    print("E076: Satellite NDVI Anomaly Detection at Candi Sites")
    print("=" * 70)

    all_sites = [(name, lat, lon, desc, "candi") for name, lat, lon, desc in TARGET_CANDI]
    all_sites += [(name, lat, lon, desc, "control") for name, lat, lon, desc in CONTROL_SITES]

    print(f"\nTarget sites: {len(TARGET_CANDI)} candi + {len(CONTROL_SITES)} controls")

    # ── Step 1: Search for imagery ────────────────────────────────────
    print("\n--- Step 1: Searching for cloud-free dry-season imagery ---")

    # Use bounding box covering all sites
    all_lats = [s[1] for s in all_sites]
    all_lons = [s[2] for s in all_sites]
    bbox = [min(all_lons) - 0.1, min(all_lats) - 0.1,
            max(all_lons) + 0.1, max(all_lats) + 0.1]

    print(f"  Search bbox: {bbox}")

    # Try multiple date ranges (2024 dry season, then 2023)
    scenes = []
    for date_range in ["2024-07-01/2024-09-30", "2023-07-01/2023-09-30", "2025-07-01/2025-09-30"]:
        print(f"  Searching {date_range}...")
        found = search_sentinel2(bbox, date_range=date_range, max_cloud=10, max_items=5)
        if found:
            scenes.extend(found)
            print(f"    Found {len(found)} scenes")
            if len(scenes) >= 5:
                break

    if not scenes:
        print("\n  WARNING: No scenes found. Trying with higher cloud cover...")
        for date_range in ["2024-06-01/2024-10-31", "2023-06-01/2023-10-31"]:
            found = search_sentinel2(bbox, date_range=date_range, max_cloud=30, max_items=5)
            if found:
                scenes.extend(found)
                break

    if not scenes:
        print("\n  ERROR: Could not find any Sentinel-2 scenes.")
        print("  This may be a network issue. Saving search diagnostics.")
        with open(RESULTS_DIR / "e076_search_diagnostic.json", "w") as f:
            json.dump({"status": "no_scenes_found", "bbox": bbox}, f, indent=2)
        return

    # Sort by cloud cover
    scenes.sort(key=lambda s: s['properties'].get('eo:cloud_cover', 100))

    print(f"\n  Total scenes found: {len(scenes)}")
    for i, scene in enumerate(scenes[:5]):
        props = scene['properties']
        print(f"  [{i}] {props.get('datetime', '?')[:10]} | "
              f"Cloud: {props.get('eo:cloud_cover', '?'):.1f}% | "
              f"Tile: {props.get('s2:mgrs_tile', '?')}")

    # ── Step 2: Extract NDVI at each site ─────────────────────────────
    print("\n--- Step 2: Extracting NDVI at candi sites ---")

    results = []
    for name, lat, lon, desc, site_type in all_sites:
        print(f"\n  Processing: {name} ({lat:.4f}, {lon:.4f})")

        # Find best scene that covers this site
        ndvi = None
        scene_used = None
        error_msg = None

        for scene in scenes:
            try:
                assets = scene.get('assets', {})
                b04_asset = assets.get('B04', assets.get('red', {}))
                b08_asset = assets.get('B08', assets.get('nir', {}))

                b04_url = b04_asset.get('href', '')
                b08_url = b08_asset.get('href', '')

                if not b04_url or not b08_url:
                    continue

                ndvi, err = extract_ndvi_window(b04_url, b08_url, lat, lon, buffer_m=500)

                if ndvi is not None:
                    scene_used = scene['properties'].get('datetime', '')[:10]
                    break
                elif err and "outside tile" in err.lower():
                    continue  # Try next scene
                else:
                    error_msg = err

            except Exception as e:
                error_msg = str(e)
                continue

        if ndvi is not None:
            analysis = analyze_ndvi_anomaly(ndvi, name)
            if analysis:
                analysis['lat'] = lat
                analysis['lon'] = lon
                analysis['description'] = desc
                analysis['type'] = site_type
                analysis['scene_date'] = scene_used
                results.append(analysis)
                print(f"    NDVI: mean={analysis['ndvi_mean']:.3f}, "
                      f"center-ring={analysis['center_ring_diff']:+.4f}, "
                      f"local_var={analysis['mean_local_variance']:.6f}" if analysis['mean_local_variance'] else "")
            else:
                print(f"    Analysis failed (insufficient data)")
        else:
            print(f"    No data available. Error: {error_msg or 'No covering scene'}")

    # ── Step 3: Compare candi vs control ──────────────────────────────
    if results:
        print("\n" + "=" * 70)
        print("RESULTS: Candi vs Control Site NDVI Comparison")
        print("=" * 70)

        candi_results = [r for r in results if r['type'] == 'candi']
        control_results = [r for r in results if r['type'] == 'control']

        if candi_results:
            print(f"\n  Candi sites (n={len(candi_results)}):")
            candi_diffs = [r['center_ring_diff'] for r in candi_results]
            candi_vars = [r['mean_local_variance'] for r in candi_results if r['mean_local_variance']]
            print(f"    Mean center-ring NDVI diff: {np.mean(candi_diffs):+.4f}")
            print(f"    Std center-ring diff: {np.std(candi_diffs):.4f}")
            if candi_vars:
                print(f"    Mean local variance: {np.mean(candi_vars):.6f}")

        if control_results:
            print(f"\n  Control sites (n={len(control_results)}):")
            ctrl_diffs = [r['center_ring_diff'] for r in control_results]
            ctrl_vars = [r['mean_local_variance'] for r in control_results if r['mean_local_variance']]
            print(f"    Mean center-ring NDVI diff: {np.mean(ctrl_diffs):+.4f}")
            print(f"    Std center-ring diff: {np.std(ctrl_diffs):.4f}")
            if ctrl_vars:
                print(f"    Mean local variance: {np.mean(ctrl_vars):.6f}")

        if candi_results and control_results:
            # Mann-Whitney U test
            from scipy import stats
            if len(candi_diffs) >= 3 and len(ctrl_diffs) >= 3:
                u_stat, mw_p = stats.mannwhitneyu(
                    [abs(d) for d in candi_diffs],
                    [abs(d) for d in ctrl_diffs],
                    alternative='greater'
                )
                print(f"\n  Mann-Whitney U (|center-ring diff|): U={u_stat:.1f}, p={mw_p:.4f}")
                print(f"  Interpretation: {'SIGNIFICANT' if mw_p < 0.05 else 'NOT significant'} "
                      f"anomaly difference")

            if candi_vars and ctrl_vars and len(candi_vars) >= 3 and len(ctrl_vars) >= 3:
                u2, p2 = stats.mannwhitneyu(candi_vars, ctrl_vars, alternative='greater')
                print(f"  Mann-Whitney U (local variance): U={u2:.1f}, p={p2:.4f}")
                print(f"  Interpretation: {'SIGNIFICANT' if p2 < 0.05 else 'NOT significant'} "
                      f"heterogeneity difference")

    # ── Save results ──────────────────────────────────────────────────
    if results:
        with open(RESULTS_DIR / "ndvi_anomaly_results.csv", "w", newline='') as f:
            fieldnames = ['site', 'type', 'lat', 'lon', 'description', 'scene_date',
                         'ndvi_mean', 'ndvi_std', 'ndvi_center_mean', 'ndvi_ring_mean',
                         'center_ring_diff', 'anomaly_ratio', 'mean_local_variance',
                         'n_pixels']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        summary = {
            "experiment": "E076",
            "title": "Satellite NDVI Anomaly Detection at Candi Sites",
            "n_candi_analyzed": len(candi_results) if results else 0,
            "n_controls_analyzed": len(control_results) if results else 0,
            "n_scenes_searched": len(scenes),
            "methodology": "Sentinel-2 L2A 10m NDVI via Planetary Computer STAC",
        }
        with open(RESULTS_DIR / "e076_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {RESULTS_DIR}/")
    else:
        print("\nNo results to save.")
        # Save diagnostic info
        with open(RESULTS_DIR / "e076_diagnostic.json", "w") as f:
            json.dump({
                "status": "no_ndvi_extracted",
                "scenes_found": len(scenes),
                "scene_tiles": [s['properties'].get('s2:mgrs_tile', '?') for s in scenes[:10]],
            }, f, indent=2)


if __name__ == "__main__":
    main()
