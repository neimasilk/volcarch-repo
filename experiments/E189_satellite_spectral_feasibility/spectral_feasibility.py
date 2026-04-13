#!/usr/bin/env python3
"""
E189: Satellite Spectral Feasibility — Can Sentinel-2 See Buried Candi?
========================================================================
Builds on E076 (proven Planetary Computer STAC pipeline).
Expands to multi-index analysis: NDVI, NDWI, MSAVI, clay ratio, iron oxide.
Tests spectral anomalies at known candi, E080 fieldwork targets, E097 anomaly
cells, and control sites.

Usage:
    python spectral_feasibility.py
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import requests

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REPO_ROOT = Path(__file__).parent.parent.parent

# ── Planetary Computer STAC API ───────────────────────────────────────
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"

# ── Site definitions ──────────────────────────────────────────────────

# Category 1: Known candi (positive reference)
# From E076 — sites with documented subsurface archaeology
KNOWN_CANDI = [
    ("Candi Jawi", -7.7619, 112.7281, "Penanggungan slope, partially buried base"),
    ("Candi Tikus", -7.5589, 112.3756, "Trowulan sunken bathing complex"),
    ("Candi Brahu", -7.5481, 112.3856, "Trowulan, large brick temple"),
    ("Candi Bajang Ratu", -7.5561, 112.3978, "Trowulan gate"),
    ("Candi Kidal", -8.0156, 112.5928, "Malang, 13th century"),
    ("Candi Singosari", -7.8894, 112.6592, "Malang, Singhasari period"),
    ("Candi Sumberawan", -7.8403, 112.5456, "Buddhist stupa on slope"),
    ("Candi Songgoriti", -7.8714, 112.4894, "Batu, thermal spring temple"),
    ("Candi Penataran", -7.9250, 112.2069, "Blitar, largest East Java complex"),
    ("Candi Sawentar", -7.9500, 112.1806, "Blitar, near Kelud"),
    ("Candi Gambar Wetan", -7.9769, 112.2597, "Near Kelud, volcanic burial"),
    ("Candi Tegowangi", -7.7733, 112.1194, "Kediri, relief panels"),
    ("Candi Surawana", -7.7756, 112.1494, "Kediri, 14th century"),
    ("Situs Trowulan", -7.5500, 112.3800, "Capital of Majapahit"),
    ("Candi Wringin Lawang", -7.5397, 112.3908, "Trowulan, split gate"),
]

# Category 2: Control sites (negative reference — no known archaeology)
CONTROL_SITES = [
    ("Control_volc_kelud", -7.95, 112.45, "Open farmland S of Kelud, no sites"),
    ("Control_volc_arjuno", -7.80, 112.55, "Slope NE Arjuno, no sites"),
    ("Control_plain_north", -7.45, 112.40, "Alluvial plain north of Trowulan"),
    ("Control_plain_east", -7.60, 112.80, "Eastern lowland, no sites"),
    ("Control_forest_south", -8.05, 113.00, "Forested hill, Ijen area"),
]


def load_e080_targets():
    """Load E080 fieldwork target zones."""
    path = REPO_ROOT / "experiments" / "E080_fieldwork_targets" / "results" / "top20_targets.csv"
    targets = []
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                name = f"E080_T{i+1:02d}_{row.get('nearest_volcano', 'unk')}"
                targets.append((
                    name,
                    float(row['lat']),
                    float(row['lon']),
                    f"Predicted buried site near {row.get('nearest_candi', '?')}, "
                    f"burial ~{row.get('estimated_burial_m', '?')}m"
                ))
    except FileNotFoundError:
        print("  WARNING: E080 targets not found")
    return targets


def load_e097_anomalies(max_sites=20):
    """Load E097 top anomaly cells."""
    path = REPO_ROOT / "experiments" / "E097_anomaly_detection" / "results" / "top50_anomaly_cells.csv"
    targets = []
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_sites:
                    break
                name = f"E097_A{i+1:02d}"
                targets.append((
                    name,
                    float(row['lat']),
                    float(row['lon']),
                    f"Anomaly cell, burial ~{float(row.get('burial_depth_cm', 0))/100:.0f}m, "
                    f"volcano {float(row.get('volcano_dist_km', 0)):.1f}km"
                ))
    except FileNotFoundError:
        print("  WARNING: E097 anomalies not found")
    return targets


def search_sentinel2(bbox, date_range="2024-07-01/2024-09-30", max_cloud=15, max_items=10):
    """Search Planetary Computer STAC for Sentinel-2 L2A scenes."""
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
    try:
        resp = requests.post(f"{STAC_URL}/search", json=search_body, timeout=30)
        if resp.status_code != 200:
            print(f"  STAC search failed: {resp.status_code} {resp.text[:200]}")
            return []
        return resp.json().get('features', [])
    except requests.exceptions.RequestException as e:
        print(f"  Network error: {e}")
        return []


def sign_url(asset_url):
    """Sign a Planetary Computer URL for data access."""
    try:
        resp = requests.get(SIGN_URL, timeout=10)
        if resp.status_code == 200:
            token_data = resp.json()
            token = token_data.get('token', '')
            sep = '&' if '?' in asset_url else '?'
            return f"{asset_url}{sep}{token}"
    except Exception as e:
        print(f"  Token signing failed: {e}")
    return asset_url


def extract_bands(scene, center_lat, center_lon, buffer_m=500):
    """
    Extract multiple Sentinel-2 bands around a point.
    Returns dict of band arrays: B03 (green), B04 (red), B08 (NIR),
    B11 (SWIR1), B12 (SWIR2).
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from pyproj import Transformer
    except ImportError as e:
        return None, f"Missing dependency: {e}"

    assets = scene.get('assets', {})

    # Band mapping — Planetary Computer uses different naming
    band_map = {
        'B03': ['B03', 'green'],
        'B04': ['B04', 'red'],
        'B08': ['B08', 'nir'],
        'B11': ['B11', 'swir16'],
        'B12': ['B12', 'swir22'],
    }

    # For East Java: UTM Zone 49S (EPSG:32749)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
    center_x, center_y = transformer.transform(center_lon, center_lat)

    xmin, xmax = center_x - buffer_m, center_x + buffer_m
    ymin, ymax = center_y - buffer_m, center_y + buffer_m

    bands = {}
    for band_name, aliases in band_map.items():
        url = None
        for alias in aliases:
            if alias in assets and 'href' in assets[alias]:
                url = assets[alias]['href']
                break
        if not url:
            continue

        signed = sign_url(url)
        try:
            with rasterio.open(signed) as src:
                # Check bounds
                if not (src.bounds.left <= center_x <= src.bounds.right and
                        src.bounds.bottom <= center_y <= src.bounds.top):
                    return None, "Point outside tile bounds"

                window = from_bounds(xmin, ymin, xmax, ymax, src.transform)
                data = src.read(1, window=window).astype(float)

                # Check for nodata — Sentinel-2 uses 0 as nodata at tile edges
                valid_frac = np.count_nonzero(data) / max(data.size, 1)
                if valid_frac < 0.5:
                    return None, f"Mostly nodata ({valid_frac:.0%} valid)"

                # B11/B12 are 20m resolution, others 10m — handle size mismatch
                bands[band_name] = data
        except Exception as e:
            # Skip this band but continue with others
            print(f"    Band {band_name} read error: {e}")
            continue

    if 'B04' not in bands or 'B08' not in bands:
        return None, "Missing essential bands (B04/B08)"

    return bands, None


def compute_indices(bands):
    """
    Compute spectral indices from band arrays.
    Returns dict of index arrays.
    """
    indices = {}

    b04 = bands.get('B04')  # Red
    b08 = bands.get('B08')  # NIR
    b03 = bands.get('B03')  # Green
    b11 = bands.get('B11')  # SWIR1
    b12 = bands.get('B12')  # SWIR2

    # NDVI — vegetation health
    if b04 is not None and b08 is not None:
        denom = b08 + b04
        indices['NDVI'] = np.where(denom > 0, (b08 - b04) / denom, 0)

    # NDWI — water content (Green-NIR)
    if b03 is not None and b08 is not None:
        # Resize b03 to match b08 if needed
        b03r = _resize_to_match(b03, b08)
        if b03r is not None:
            denom = b03r + b08
            indices['NDWI'] = np.where(denom > 0, (b03r - b08) / denom, 0)

    # MSAVI — Modified Soil-Adjusted Vegetation Index
    if b04 is not None and b08 is not None:
        term = (2 * b08 + 1)**2 - 8 * (b08 - b04)
        term = np.maximum(term, 0)  # avoid sqrt of negative
        indices['MSAVI'] = (2 * b08 + 1 - np.sqrt(term)) / 2

    # Clay ratio (B11/B12) — mineral composition
    if b11 is not None and b12 is not None:
        b12r = _resize_to_match(b12, b11)
        if b12r is not None:
            indices['clay_ratio'] = np.where(b12r > 0, b11 / b12r, 0)

    # Iron oxide ratio (B04/B03) — laterite detection
    if b04 is not None and b03 is not None:
        b03r = _resize_to_match(b03, b04)
        if b03r is not None:
            indices['iron_oxide'] = np.where(b03r > 0, b04 / b03r, 0)

    return indices


def _resize_to_match(source, target):
    """Resize source array to match target dimensions (nearest-neighbor)."""
    if source.shape == target.shape:
        return source
    try:
        from scipy.ndimage import zoom
        zy = target.shape[0] / source.shape[0]
        zx = target.shape[1] / source.shape[1]
        return zoom(source, (zy, zx), order=0)
    except Exception:
        return None


def analyze_site(indices, site_name):
    """
    Analyze spectral indices at a site.
    Compare center (inner 40%) vs ring (outer periphery).
    Compute local variance as archaeological heterogeneity proxy.
    """
    result = {'site': site_name}

    for idx_name, arr in indices.items():
        if arr is None or arr.size == 0:
            continue

        h, w = arr.shape
        if h < 6 or w < 6:
            continue

        # Overall stats
        result[f'{idx_name}_mean'] = round(float(np.nanmean(arr)), 5)
        result[f'{idx_name}_std'] = round(float(np.nanstd(arr)), 5)

        # Center vs ring
        ch, cw = h // 5, w // 5
        center = arr[2*ch:3*ch, 2*cw:3*cw]
        ring = np.concatenate([
            arr[:ch, :].flatten(),
            arr[-ch:, :].flatten(),
            arr[ch:-ch, :cw].flatten(),
            arr[ch:-ch, -cw:].flatten(),
        ])

        if center.size > 0 and ring.size > 0:
            c_mean = float(np.nanmean(center))
            r_mean = float(np.nanmean(ring))
            result[f'{idx_name}_center'] = round(c_mean, 5)
            result[f'{idx_name}_ring'] = round(r_mean, 5)
            result[f'{idx_name}_diff'] = round(c_mean - r_mean, 5)

        # Local variance (3x3 window) — key archaeological signal
        if h >= 5 and w >= 5:
            local_vars = []
            for i in range(1, h-1):
                for j in range(1, w-1):
                    patch = arr[i-1:i+2, j-1:j+2]
                    v = np.nanvar(patch)
                    if np.isfinite(v):
                        local_vars.append(v)
            if local_vars:
                result[f'{idx_name}_local_var'] = round(float(np.mean(local_vars)), 8)

        result[f'{idx_name}_pixels'] = int(arr.size)

    return result


def run_statistical_tests(results_by_category):
    """Compare spectral profiles across site categories."""
    from scipy import stats as sp_stats

    tests = {}
    key_indices = ['NDVI', 'NDWI', 'MSAVI']

    categories = list(results_by_category.keys())
    if 'candi' not in categories or 'control' not in categories:
        return tests

    for idx in key_indices:
        diff_key = f'{idx}_diff'
        lvar_key = f'{idx}_local_var'

        # Candi vs control: center-ring difference
        candi_diffs = [r[diff_key] for r in results_by_category['candi']
                       if diff_key in r and r[diff_key] is not None]
        ctrl_diffs = [r[diff_key] for r in results_by_category['control']
                      if diff_key in r and r[diff_key] is not None]

        if len(candi_diffs) >= 3 and len(ctrl_diffs) >= 3:
            u, p = sp_stats.mannwhitneyu(
                [abs(d) for d in candi_diffs],
                [abs(d) for d in ctrl_diffs],
                alternative='greater'
            )
            tests[f'{idx}_diff_candi_vs_control'] = {
                'U': round(float(u), 2), 'p': round(float(p), 5),
                'n_candi': len(candi_diffs), 'n_control': len(ctrl_diffs),
                'significant': p < 0.05
            }

        # Candi vs control: local variance
        candi_vars = [r[lvar_key] for r in results_by_category['candi']
                      if lvar_key in r and r[lvar_key] is not None]
        ctrl_vars = [r[lvar_key] for r in results_by_category['control']
                     if lvar_key in r and r[lvar_key] is not None]

        if len(candi_vars) >= 3 and len(ctrl_vars) >= 3:
            u, p = sp_stats.mannwhitneyu(candi_vars, ctrl_vars, alternative='greater')
            tests[f'{idx}_lvar_candi_vs_control'] = {
                'U': round(float(u), 2), 'p': round(float(p), 5),
                'n_candi': len(candi_vars), 'n_control': len(ctrl_vars),
                'significant': p < 0.05
            }

        # E080 targets vs control
        if 'e080' in results_by_category:
            tgt_diffs = [r[diff_key] for r in results_by_category['e080']
                         if diff_key in r and r[diff_key] is not None]
            if len(tgt_diffs) >= 3 and len(ctrl_diffs) >= 3:
                u, p = sp_stats.mannwhitneyu(
                    [abs(d) for d in tgt_diffs],
                    [abs(d) for d in ctrl_diffs],
                    alternative='greater'
                )
                tests[f'{idx}_diff_e080_vs_control'] = {
                    'U': round(float(u), 2), 'p': round(float(p), 5),
                    'n_e080': len(tgt_diffs), 'n_control': len(ctrl_diffs),
                    'significant': p < 0.05
                }

        # E097 anomalies vs control
        if 'e097' in results_by_category:
            anom_diffs = [r[diff_key] for r in results_by_category['e097']
                          if diff_key in r and r[diff_key] is not None]
            if len(anom_diffs) >= 3 and len(ctrl_diffs) >= 3:
                u, p = sp_stats.mannwhitneyu(
                    [abs(d) for d in anom_diffs],
                    [abs(d) for d in ctrl_diffs],
                    alternative='greater'
                )
                tests[f'{idx}_diff_e097_vs_control'] = {
                    'U': round(float(u), 2), 'p': round(float(p), 5),
                    'n_e097': len(anom_diffs), 'n_control': len(ctrl_diffs),
                    'significant': p < 0.05
                }

    return tests


def main():
    print("=" * 70)
    print("E189: Satellite Spectral Feasibility")
    print("Can Sentinel-2 See Buried Candi in Volcanic Java?")
    print("=" * 70)

    # ── Load all site categories ──────────────────────────────────────
    e080_targets = load_e080_targets()
    e097_anomalies = load_e097_anomalies(max_sites=20)

    all_sites = []
    for name, lat, lon, desc in KNOWN_CANDI:
        all_sites.append((name, lat, lon, desc, 'candi'))
    for name, lat, lon, desc in CONTROL_SITES:
        all_sites.append((name, lat, lon, desc, 'control'))
    for name, lat, lon, desc in e080_targets:
        all_sites.append((name, lat, lon, desc, 'e080'))
    for name, lat, lon, desc in e097_anomalies:
        all_sites.append((name, lat, lon, desc, 'e097'))

    print(f"\nSites: {len(KNOWN_CANDI)} candi, {len(CONTROL_SITES)} control, "
          f"{len(e080_targets)} E080, {len(e097_anomalies)} E097")
    print(f"Total: {len(all_sites)} locations to analyze")

    # ── Search for imagery ────────────────────────────────────────────
    print("\n--- Step 1: Searching for dry-season Sentinel-2 imagery ---")

    all_lats = [s[1] for s in all_sites]
    all_lons = [s[2] for s in all_sites]
    bbox = [min(all_lons) - 0.1, min(all_lats) - 0.1,
            max(all_lons) + 0.1, max(all_lats) + 0.1]
    print(f"  Bounding box: {[round(b, 2) for b in bbox]}")

    scenes = []
    for date_range in ["2024-07-01/2024-09-30", "2023-07-01/2023-09-30"]:
        print(f"  Searching {date_range}...")
        found = search_sentinel2(bbox, date_range=date_range, max_cloud=15, max_items=50)
        if found:
            scenes.extend(found)
            print(f"    Found {len(found)} scenes")
        if len(scenes) >= 30:
            break

    if not scenes:
        print("  Relaxing cloud threshold to 30%...")
        for date_range in ["2024-06-01/2024-10-31", "2023-06-01/2023-10-31"]:
            found = search_sentinel2(bbox, date_range=date_range, max_cloud=30, max_items=50)
            if found:
                scenes.extend(found)
                break

    if not scenes:
        print("\n  ERROR: No Sentinel-2 scenes found. Check network connection.")
        diag = {"status": "no_scenes", "bbox": bbox, "sites": len(all_sites)}
        with open(RESULTS_DIR / "diagnostic.json", "w") as f:
            json.dump(diag, f, indent=2)
        return

    scenes.sort(key=lambda s: s['properties'].get('eo:cloud_cover', 100))
    print(f"\n  Scenes available: {len(scenes)}")
    for i, s in enumerate(scenes[:5]):
        p = s['properties']
        print(f"  [{i}] {p.get('datetime', '?')[:10]} | "
              f"Cloud: {p.get('eo:cloud_cover', '?'):.1f}% | "
              f"Tile: {p.get('s2:mgrs_tile', '?')}")

    # ── Extract multi-index profiles ──────────────────────────────────
    print(f"\n--- Step 2: Extracting spectral indices at {len(all_sites)} sites ---")

    results_by_cat = defaultdict(list)
    all_results = []
    success_count = 0

    for name, lat, lon, desc, category in all_sites:
        print(f"\n  [{category}] {name} ({lat:.4f}, {lon:.4f})")

        bands = None
        scene_used = None

        for scene in scenes:
            bands, err = extract_bands(scene, lat, lon, buffer_m=500)
            if bands is not None:
                scene_used = scene['properties'].get('datetime', '')[:10]
                break
            elif err and "outside" in err.lower():
                continue
            else:
                if err:
                    print(f"    Error: {err}")
                break

        if bands is None:
            print(f"    SKIP: no data available")
            continue

        indices = compute_indices(bands)
        if not indices:
            print(f"    SKIP: could not compute indices")
            continue

        analysis = analyze_site(indices, name)
        analysis['lat'] = lat
        analysis['lon'] = lon
        analysis['category'] = category
        analysis['description'] = desc
        analysis['scene_date'] = scene_used
        analysis['n_bands'] = len(bands)
        analysis['n_indices'] = len(indices)

        results_by_cat[category].append(analysis)
        all_results.append(analysis)
        success_count += 1

        # Print key metrics
        ndvi_diff = analysis.get('NDVI_diff', None)
        ndvi_lvar = analysis.get('NDVI_local_var', None)
        ndwi_diff = analysis.get('NDWI_diff', None)
        print(f"    NDVI diff: {ndvi_diff:+.5f}" if ndvi_diff is not None else "    NDVI: N/A", end="")
        print(f" | NDWI diff: {ndwi_diff:+.5f}" if ndwi_diff is not None else " | NDWI: N/A", end="")
        print(f" | local_var: {ndvi_lvar:.8f}" if ndvi_lvar is not None else " | lvar: N/A")

    print(f"\n  Success: {success_count}/{len(all_sites)} sites analyzed")

    # ── Category summaries ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS: Spectral Profile Comparison")
    print("=" * 70)

    for cat_name, cat_label in [('candi', 'Known Candi'), ('control', 'Control'),
                                 ('e080', 'E080 Targets'), ('e097', 'E097 Anomalies')]:
        cat_results = results_by_cat.get(cat_name, [])
        if not cat_results:
            continue

        print(f"\n  {cat_label} (n={len(cat_results)}):")
        for idx in ['NDVI', 'NDWI', 'MSAVI']:
            diffs = [r.get(f'{idx}_diff') for r in cat_results if r.get(f'{idx}_diff') is not None]
            lvars = [r.get(f'{idx}_local_var') for r in cat_results if r.get(f'{idx}_local_var') is not None]
            if diffs:
                print(f"    {idx}: mean diff={np.mean(diffs):+.5f} (std={np.std(diffs):.5f})", end="")
            if lvars:
                print(f" | local_var={np.mean(lvars):.8f}", end="")
            print()

    # ── Statistical tests ─────────────────────────────────────────────
    print("\n--- Statistical Tests ---")
    tests = run_statistical_tests(results_by_cat)

    if tests:
        for test_name, result in tests.items():
            sig = "***" if result['significant'] else "   "
            print(f"  {sig} {test_name}: U={result['U']}, p={result['p']:.5f}")
    else:
        print("  Insufficient data for statistical tests (need >=3 per category)")

    # ── Convergence analysis ──────────────────────────────────────────
    # Do E080/E097 targets look more like candi or control?
    print("\n--- Convergence Analysis ---")
    print("  Q: Do predicted buried-site zones have candi-like spectral profiles?")

    for idx in ['NDVI', 'NDWI']:
        lvar_key = f'{idx}_local_var'
        candi_lv = [r[lvar_key] for r in results_by_cat.get('candi', []) if lvar_key in r]
        ctrl_lv = [r[lvar_key] for r in results_by_cat.get('control', []) if lvar_key in r]
        e080_lv = [r[lvar_key] for r in results_by_cat.get('e080', []) if lvar_key in r]
        e097_lv = [r[lvar_key] for r in results_by_cat.get('e097', []) if lvar_key in r]

        if candi_lv and ctrl_lv:
            candi_mean = np.mean(candi_lv)
            ctrl_mean = np.mean(ctrl_lv)
            print(f"\n  {idx} local variance:")
            print(f"    Candi mean:   {candi_mean:.8f}")
            print(f"    Control mean: {ctrl_mean:.8f}")
            if e080_lv:
                e080_mean = np.mean(e080_lv)
                # Which is it closer to?
                d_candi = abs(e080_mean - candi_mean)
                d_ctrl = abs(e080_mean - ctrl_mean)
                closer = "CANDI" if d_candi < d_ctrl else "CONTROL"
                print(f"    E080 mean:    {e080_mean:.8f} (closer to {closer})")
            if e097_lv:
                e097_mean = np.mean(e097_lv)
                d_candi = abs(e097_mean - candi_mean)
                d_ctrl = abs(e097_mean - ctrl_mean)
                closer = "CANDI" if d_candi < d_ctrl else "CONTROL"
                print(f"    E097 mean:    {e097_mean:.8f} (closer to {closer})")

    # ── Save results ──────────────────────────────────────────────────
    print("\n--- Saving results ---")

    # CSV with all site results
    if all_results:
        # Collect all possible keys
        all_keys = set()
        for r in all_results:
            all_keys.update(r.keys())
        # Order: identifiers first, then indices
        id_keys = ['site', 'category', 'lat', 'lon', 'description', 'scene_date', 'n_bands', 'n_indices']
        idx_keys = sorted([k for k in all_keys if k not in id_keys])
        fieldnames = id_keys + idx_keys

        with open(RESULTS_DIR / "spectral_profiles.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print(f"  Saved spectral_profiles.csv ({len(all_results)} sites)")

    # JSON summary
    summary = {
        "experiment": "E189",
        "title": "Satellite Spectral Feasibility",
        "date": "2026-04-13",
        "sites_analyzed": success_count,
        "sites_total": len(all_sites),
        "by_category": {cat: len(res) for cat, res in results_by_cat.items()},
        "scenes_used": len(scenes),
        "indices": ["NDVI", "NDWI", "MSAVI", "clay_ratio", "iron_oxide"],
        "statistical_tests": tests,
        "methodology": "Sentinel-2 L2A 10m via Planetary Computer STAC, dry season 2023-2024",
    }
    with open(RESULTS_DIR / "e189_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved e189_results.json")

    # ── Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    n_sig = sum(1 for t in tests.values() if t.get('significant'))
    n_tests = len(tests)
    print(f"VERDICT: {n_sig}/{n_tests} tests significant at p<0.05")

    if n_sig == 0:
        print("STATUS: NO SIGNAL DETECTED in Sentinel-2 multispectral data")
        print("IMPLICATION: Andosol may be too homogeneous for passive optical detection.")
        print("NEXT: Proceed to Phase B — SAR (Sentinel-1) may penetrate where optical fails.")
        verdict = "NO_SIGNAL"
    elif n_sig <= n_tests // 3:
        print("STATUS: WEAK SIGNAL — some indices show differences but inconclusive")
        print("NEXT: Multi-temporal analysis (dry vs wet) or SAR complementary analysis.")
        verdict = "WEAK_SIGNAL"
    else:
        print("STATUS: SIGNAL DETECTED — spectral anomalies at candi sites differ from controls")
        print("NEXT: Phase C — train ML model on these spectral signatures.")
        verdict = "SIGNAL_DETECTED"

    summary['verdict'] = verdict
    with open(RESULTS_DIR / "e189_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
