#!/usr/bin/env python3
"""
E076 v2: Multi-tile Satellite NDVI Analysis for Candi Sites
============================================================
Fixes the v1 single-tile limitation by:
1. Querying per-site (not one big bbox) to get the correct tile for each site
2. Expanding to 2023-2024 dry seasons for composite
3. Implementing proper SAS token signing for Planetary Computer
4. Running Mann-Whitney U on full 15 candi vs 5 control dataset

Hardware: CPU only (no GPU needed). Requires: rasterio, pyproj, requests, scipy.
Network: Downloads ~20 COG windows (~5-10 MB each) from Planetary Computer.
"""

import json
import sys
import csv
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import requests

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Planetary Computer API ─────────────────────────────────────────────
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ── Target sites ───────────────────────────────────────────────────────
# All 15 candi + 5 controls from v1, with corrected coordinates.
TARGET_CANDI = [
    ("Candi Jawi", -7.7619, 112.7281, "Penanggungan slope, partially buried base"),
    ("Candi Tikus", -7.5589, 112.3756, "Trowulan sunken bathing complex"),
    ("Candi Brahu", -7.5481, 112.3856, "Trowulan, large brick temple"),
    ("Candi Bajang Ratu", -7.5561, 112.3978, "Trowulan gate"),
    ("Candi Wringin Lawang", -7.5397, 112.3908, "Trowulan, split gate"),
    ("Candi Kidal", -8.0156, 112.5928, "Malang, 13th century"),
    ("Candi Singosari", -7.8894, 112.6592, "Malang, Singhasari period"),
    ("Candi Sumberawan", -7.8403, 112.5456, "Buddhist stupa on slope"),
    ("Candi Songgoriti", -7.8714, 112.4894, "Batu, thermal spring temple"),
    ("Candi Surawana", -7.7756, 112.1494, "Kediri, 14th century"),
    ("Candi Penataran", -7.9250, 112.2069, "Blitar, largest E.Java complex"),
    ("Candi Sawentar", -7.9500, 112.1806, "Blitar, near Kelud"),
    ("Candi Gambar Wetan", -7.9769, 112.2597, "Near Kelud, volcanic burial"),
    ("Candi Tegowangi", -7.7733, 112.1194, "Kediri, relief panels"),
    ("Situs Trowulan", -7.5500, 112.3800, "Majapahit capital"),
]

CONTROL_SITES = [
    ("Control_volc_1", -7.85, 112.40, "Volcanic slope, no known site"),
    ("Control_volc_2", -7.95, 112.85, "Volcanic plain, no known site"),
    ("Control_plain_1", -7.50, 112.50, "Alluvial plain, no known site"),
    ("Control_plain_2", -7.60, 112.20, "River valley, no known site"),
    ("Control_forest_1", -8.00, 113.00, "Forested hill, no known site"),
]


def get_sas_token():
    """Get a SAS token from Planetary Computer for Sentinel-2 data."""
    try:
        resp = requests.get(
            "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a",
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json().get('token', '')
    except Exception as e:
        print(f"  SAS token error: {e}")
    return None


def sign_href(href, token):
    """Append SAS token to asset URL."""
    if not token:
        return href
    separator = '&' if '?' in href else '?'
    return f"{href}{separator}{token}"


def search_scenes_for_site(lat, lon, date_ranges, max_cloud=15):
    """Search Planetary Computer STAC for Sentinel-2 scenes covering a specific point."""
    # Small bbox around the point (0.01° ≈ 1.1 km)
    bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]

    all_scenes = []
    for date_range in date_ranges:
        search_body = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": date_range,
            "limit": 5,
            "query": {
                "eo:cloud_cover": {"lt": max_cloud}
            },
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]
        }

        try:
            resp = requests.post(f"{STAC_URL}/search", json=search_body, timeout=30)
            if resp.status_code == 200:
                features = resp.json().get('features', [])
                all_scenes.extend(features)
        except Exception as e:
            print(f"    STAC search error for {date_range}: {e}")

    # Sort by cloud cover
    all_scenes.sort(key=lambda s: s['properties'].get('eo:cloud_cover', 100))
    return all_scenes


def extract_ndvi_window(scene, lat, lon, buffer_m=500, token=None):
    """Extract NDVI from a scene around a point. Returns NDVI array or None."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from pyproj import Transformer
    except ImportError as e:
        return None, f"Missing dependency: {e}"

    assets = scene.get('assets', {})
    b04_asset = assets.get('B04', assets.get('red', {}))
    b08_asset = assets.get('B08', assets.get('nir', {}))

    b04_url = b04_asset.get('href', '')
    b08_url = b08_asset.get('href', '')

    if not b04_url or not b08_url:
        return None, "Missing band URLs"

    # Sign URLs
    b04_url = sign_href(b04_url, token)
    b08_url = sign_href(b08_url, token)

    try:
        # Determine the CRS from the scene metadata
        scene_crs = scene.get('properties', {}).get('proj:epsg')
        if not scene_crs:
            # Default to UTM zone based on longitude
            utm_zone = int((lon + 180) / 6) + 1
            hemisphere = 'south' if lat < 0 else 'north'
            scene_crs = 32700 + utm_zone if hemisphere == 'south' else 32600 + utm_zone

        epsg_code = f"EPSG:{scene_crs}"
        transformer = Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
        center_x, center_y = transformer.transform(lon, lat)

        xmin = center_x - buffer_m
        xmax = center_x + buffer_m
        ymin = center_y - buffer_m
        ymax = center_y + buffer_m

        with rasterio.open(b04_url) as src_b04:
            # Verify point is within tile
            if not (src_b04.bounds.left <= center_x <= src_b04.bounds.right and
                    src_b04.bounds.bottom <= center_y <= src_b04.bounds.top):
                return None, "Point outside tile bounds"

            window = from_bounds(xmin, ymin, xmax, ymax, src_b04.transform)
            b04 = src_b04.read(1, window=window).astype(float)

        with rasterio.open(b08_url) as src_b08:
            window = from_bounds(xmin, ymin, xmax, ymax, src_b08.transform)
            b08 = src_b08.read(1, window=window).astype(float)

        # NDVI
        denom = b08 + b04
        ndvi = np.where(denom > 0, (b08 - b04) / denom, 0)

        return ndvi, None

    except Exception as e:
        return None, str(e)


def analyze_ndvi(ndvi, site_name):
    """Analyze NDVI array for center-ring anomaly and local variance."""
    if ndvi is None or ndvi.size == 0:
        return None

    h, w = ndvi.shape
    if h < 10 or w < 10:
        return None

    # Center zone (inner 20%) vs ring zone (outer 80%)
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
    }

    # Local variance in 3×3 windows
    if h >= 3 and w >= 3:
        local_vars = []
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                patch = ndvi[i-1:i+2, j-1:j+2]
                local_vars.append(np.var(patch))
        result['mean_local_variance'] = round(float(np.mean(local_vars)), 6)
    else:
        result['mean_local_variance'] = None

    return result


def main():
    print("=" * 70)
    print("E076 v2: MULTI-TILE NDVI ANALYSIS AT CANDI SITES")
    print("=" * 70)

    all_sites = [(n, la, lo, d, "candi") for n, la, lo, d in TARGET_CANDI]
    all_sites += [(n, la, lo, d, "control") for n, la, lo, d in CONTROL_SITES]

    print(f"\nTarget: {len(TARGET_CANDI)} candi + {len(CONTROL_SITES)} controls = {len(all_sites)} sites")

    # Date ranges: two dry seasons for robustness
    date_ranges = [
        "2024-07-01/2024-09-30",
        "2023-07-01/2023-09-30",
    ]

    # Get SAS token
    print("\nGetting Planetary Computer SAS token...")
    token = get_sas_token()
    if token:
        print("  Token obtained.")
    else:
        print("  WARNING: No token. Will try unsigned URLs.")

    # ── Process each site individually ─────────────────────────────────
    print(f"\n--- Processing {len(all_sites)} sites ---")
    results = []
    failed_sites = []

    for idx, (name, lat, lon, desc, site_type) in enumerate(all_sites):
        print(f"\n  [{idx+1}/{len(all_sites)}] {name} ({lat:.4f}, {lon:.4f}) [{site_type}]")

        # Search for scenes covering this specific site
        scenes = search_scenes_for_site(lat, lon, date_ranges, max_cloud=15)
        if not scenes:
            # Retry with higher cloud tolerance
            scenes = search_scenes_for_site(lat, lon, date_ranges, max_cloud=30)

        if not scenes:
            print(f"    No scenes found.")
            failed_sites.append(name)
            continue

        print(f"    Found {len(scenes)} scene(s). Best cloud: {scenes[0]['properties'].get('eo:cloud_cover', '?'):.1f}%")
        tile = scenes[0]['properties'].get('s2:mgrs_tile', '?')
        print(f"    Tile: {tile}")

        # Try extracting NDVI from best scene
        ndvi = None
        scene_used = None
        error_msg = None

        for scene in scenes[:3]:  # Try top 3 scenes
            ndvi, err = extract_ndvi_window(scene, lat, lon, buffer_m=500, token=token)
            if ndvi is not None:
                scene_used = scene['properties'].get('datetime', '')[:10]
                break
            error_msg = err

        if ndvi is not None:
            analysis = analyze_ndvi(ndvi, name)
            if analysis:
                analysis['lat'] = lat
                analysis['lon'] = lon
                analysis['description'] = desc
                analysis['type'] = site_type
                analysis['scene_date'] = scene_used
                analysis['tile'] = tile
                results.append(analysis)
                print(f"    NDVI: mean={analysis['ndvi_mean']:.3f}, "
                      f"center-ring={analysis['center_ring_diff']:+.4f}, "
                      f"local_var={analysis.get('mean_local_variance', 'N/A')}")
            else:
                print(f"    Analysis failed (insufficient pixels)")
                failed_sites.append(name)
        else:
            print(f"    NDVI extraction failed: {error_msg or 'unknown'}")
            failed_sites.append(name)

        # Rate limit to be polite to Planetary Computer
        time.sleep(1)

    # ── Statistical comparison ─────────────────────────────────────────
    if results:
        print("\n" + "=" * 70)
        print("RESULTS: CANDI vs CONTROL NDVI COMPARISON")
        print("=" * 70)

        candi_results = [r for r in results if r['type'] == 'candi']
        control_results = [r for r in results if r['type'] == 'control']

        print(f"\n  Successfully analyzed: {len(candi_results)} candi, {len(control_results)} controls")
        print(f"  Failed: {len(failed_sites)} sites")
        if failed_sites:
            print(f"    {', '.join(failed_sites)}")

        if candi_results:
            candi_diffs = [abs(r['center_ring_diff']) for r in candi_results]
            candi_vars = [r['mean_local_variance'] for r in candi_results if r.get('mean_local_variance')]
            print(f"\n  Candi (n={len(candi_results)}):")
            print(f"    Mean |center-ring diff|: {np.mean(candi_diffs):.4f}")
            if candi_vars:
                print(f"    Mean local variance: {np.mean(candi_vars):.6f}")

        if control_results:
            ctrl_diffs = [abs(r['center_ring_diff']) for r in control_results]
            ctrl_vars = [r['mean_local_variance'] for r in control_results if r.get('mean_local_variance')]
            print(f"\n  Control (n={len(control_results)}):")
            print(f"    Mean |center-ring diff|: {np.mean(ctrl_diffs):.4f}")
            if ctrl_vars:
                print(f"    Mean local variance: {np.mean(ctrl_vars):.6f}")

        # Mann-Whitney U tests
        if len(candi_results) >= 3 and len(control_results) >= 2:
            from scipy import stats

            candi_diffs = [abs(r['center_ring_diff']) for r in candi_results]
            ctrl_diffs = [abs(r['center_ring_diff']) for r in control_results]

            u_stat, mw_p = stats.mannwhitneyu(candi_diffs, ctrl_diffs, alternative='greater')
            print(f"\n  Mann-Whitney U (|center-ring diff|):")
            print(f"    U = {u_stat:.1f}, p = {mw_p:.4f}")
            print(f"    {'SIGNIFICANT (p<0.05)' if mw_p < 0.05 else 'NOT significant'}")

            candi_vars = [r['mean_local_variance'] for r in candi_results if r.get('mean_local_variance')]
            ctrl_vars = [r['mean_local_variance'] for r in control_results if r.get('mean_local_variance')]

            if len(candi_vars) >= 3 and len(ctrl_vars) >= 2:
                u2, p2 = stats.mannwhitneyu(candi_vars, ctrl_vars, alternative='greater')
                print(f"\n  Mann-Whitney U (local variance):")
                print(f"    U = {u2:.1f}, p = {p2:.4f}")
                print(f"    {'SIGNIFICANT (p<0.05)' if p2 < 0.05 else 'NOT significant'}")

        # Effect size
        if candi_results and control_results:
            c_mean = np.mean([abs(r['center_ring_diff']) for r in candi_results])
            k_mean = np.mean([abs(r['center_ring_diff']) for r in control_results])
            if k_mean > 0:
                print(f"\n  Candi/Control ratio: {c_mean/k_mean:.2f}x")

    # ── Save results ───────────────────────────────────────────────────
    print("\n--- Saving results ---")

    if results:
        csv_path = RESULTS_DIR / "ndvi_full_results.csv"
        fieldnames = ['site', 'type', 'lat', 'lon', 'tile', 'scene_date', 'description',
                      'ndvi_mean', 'ndvi_std', 'ndvi_center_mean', 'ndvi_ring_mean',
                      'center_ring_diff', 'anomaly_ratio', 'mean_local_variance', 'n_pixels']
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"  Saved: {csv_path.name}")

    summary = {
        'experiment': 'E076_v2',
        'title': 'Multi-tile NDVI Analysis at Candi Sites',
        'date': __import__('datetime').datetime.now().strftime('%Y-%m-%d'),
        'n_candi_target': len(TARGET_CANDI),
        'n_control_target': len(CONTROL_SITES),
        'n_candi_analyzed': len([r for r in results if r['type'] == 'candi']),
        'n_control_analyzed': len([r for r in results if r['type'] == 'control']),
        'n_failed': len(failed_sites),
        'failed_sites': failed_sites,
        'methodology': 'Per-site STAC query, Sentinel-2 L2A 10m NDVI, 2023-2024 dry seasons',
    }
    with open(RESULTS_DIR / "e076_v2_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: e076_v2_results.json")

    print("\n" + "=" * 70)
    print("E076 v2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
