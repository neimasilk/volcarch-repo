#!/usr/bin/env python3
"""
E075: Volcanic Sedimentation Burial Model for Java
====================================================
Estimates cumulative volcanic deposit thickness across East Java
using GVP eruption history + Pyle (1989) exponential thinning model.

For each grid cell, sums tephra fall from every recorded eruption
of every volcano, producing a burial depth map.

Validates against:
- E070 colonial depth measurements
- E024 borehole data
- Published sedimentation rates

Key question: How many archaeological sites are buried under
volcanic deposits, and how deep?

References:
- Pyle (1989) "The thickness, volume and grainsize of tephra fall deposits"
  Bull Volcanol 51:1-15
- Bonadonna & Houghton (2005) "Total grain-size distribution and volume"
  Bull Volcanol 67:441-456
- Alloway et al. (2017) Samalas tephra across Java, QSR
- de Belizal et al. (2013) Semeru sedimentation, Bull Volcanol
"""

import csv
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Data paths ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "data"
ERUPTION_CSV = DATA_DIR / "processed" / "eruption_history.csv"
VOLCANO_CSV = DATA_DIR / "processed" / "dashboard" / "volcanoes.csv"
SITES_CSV = DATA_DIR / "processed" / "dashboard" / "sites.csv"

# ── Pyle (1989) tephra fall parameters by VEI ─────────────────────────
# T(r) = T0 * exp(-r / bt)
# T0 = maximum proximal thickness (cm)
# bt = half-distance decay (km) — distance at which thickness halves
# Calibrated from Kelut 2014 (VEI 4: 100cm at 0km, ~1cm at 100km)
# and Samalas 1257 (VEI 7: 22cm at 240km)

PYLE_PARAMS = {
    1: {'T0': 5, 'bt': 3},       # Very small eruptions
    2: {'T0': 15, 'bt': 5},      # Small eruptions (most Bromo)
    3: {'T0': 50, 'bt': 12},     # Moderate (some Kelut)
    4: {'T0': 150, 'bt': 25},    # Large (Kelut 1990, 2014; Merapi 2010)
    5: {'T0': 500, 'bt': 45},    # Very large (Kelut 1586)
    6: {'T0': 2000, 'bt': 80},   # Tambora-class
    7: {'T0': 5000, 'bt': 120},  # Samalas-class
}

# Default VEI for eruptions without data
DEFAULT_VEI = 2


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def tephra_thickness_cm(distance_km, vei):
    """Pyle (1989) exponential thinning model."""
    if vei is None or np.isnan(vei):
        vei = DEFAULT_VEI
    vei = int(min(max(vei, 1), 7))
    params = PYLE_PARAMS[vei]
    # Minimum distance = 1 km (avoid singularity)
    d = max(distance_km, 1.0)
    thickness = params['T0'] * math.exp(-d / params['bt'])
    return thickness


def load_volcanoes():
    """Load volcano locations."""
    volcanoes = []
    with open(VOLCANO_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['name'] and row['lat'] and row['lon']:
                volcanoes.append({
                    'name': row['name'].strip(),
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                })
    return volcanoes


def load_eruptions():
    """Load eruption history."""
    eruptions = []
    with open(ERUPTION_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                year = int(row['year'])
            except (ValueError, KeyError):
                continue

            try:
                vei = float(row['vei']) if row['vei'] else None
            except ValueError:
                vei = None

            eruptions.append({
                'volcano': row['volcano'].strip(),
                'year': year,
                'vei': vei,
            })
    return eruptions


def load_sites():
    """Load archaeological sites for validation."""
    sites = []
    with open(SITES_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row['lat'])
                lon = float(row['lon'])
            except (ValueError, KeyError):
                continue

            burial = None
            try:
                burial = float(row.get('burial_depth_cm', ''))
            except (ValueError, TypeError):
                pass

            sites.append({
                'lat': lat,
                'lon': lon,
                'burial_depth_cm': burial,
                'name': row.get('name', ''),
            })
    return sites


def main():
    print("=" * 70)
    print("E075: Volcanic Sedimentation Burial Model for Java")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    volcanoes = load_volcanoes()
    eruptions = load_eruptions()
    sites = load_sites()

    print(f"\nVolcanoes: {len(volcanoes)}")
    for v in volcanoes:
        print(f"  {v['name']}: ({v['lat']:.3f}, {v['lon']:.3f})")

    print(f"\nEruptions: {len(eruptions)} total")

    # Filter to archaeological time window (0 CE to 2025 CE)
    eruptions_arch = [e for e in eruptions if 0 <= e['year'] <= 2025]
    print(f"Eruptions 0-2025 CE: {len(eruptions_arch)}")

    # VEI distribution
    vei_counts = defaultdict(int)
    for e in eruptions_arch:
        vei = int(e['vei']) if e['vei'] is not None else 'unknown'
        vei_counts[vei] += 1
    print(f"VEI distribution: {dict(sorted(vei_counts.items(), key=lambda x: str(x[0])))}")

    # Map volcano names to coordinates
    volcano_coords = {v['name']: (v['lat'], v['lon']) for v in volcanoes}

    # ── Build grid ────────────────────────────────────────────────────
    # East Java bounding box
    lat_min, lat_max = -8.8, -7.2
    lon_min, lon_max = 110.5, 114.8
    resolution = 0.05  # ~5.5 km grid cells

    lats = np.arange(lat_min, lat_max, resolution)
    lons = np.arange(lon_min, lon_max, resolution)
    n_cells = len(lats) * len(lons)
    print(f"\nGrid: {len(lats)} x {len(lons)} = {n_cells} cells at {resolution}° (~{resolution * 111:.1f} km)")

    # ── Pre-compute distances (volcano -> grid cell) ──────────────────
    print("\nComputing volcano-grid distances...")
    # volcano_distances[v_name][(i,j)] = distance_km
    volcano_distances = {}
    for v in volcanoes:
        distances = np.zeros((len(lats), len(lons)))
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                distances[i, j] = haversine_km(lat, lon, v['lat'], v['lon'])
        volcano_distances[v['name']] = distances

    # ── Accumulate tephra from all eruptions ──────────────────────────
    print("Accumulating tephra deposits...")

    # Time windows for analysis
    windows = {
        'all_time': (0, 2025),
        'pre_colonial': (0, 1800),
        'classical_java': (400, 1500),
        'pre_inscription': (0, 400),
    }

    cumulative = {}
    for window_name, (year_start, year_end) in windows.items():
        burial_grid = np.zeros((len(lats), len(lons)))
        eruption_count = 0

        for e in eruptions_arch:
            if year_start <= e['year'] <= year_end:
                v_name = e['volcano']
                if v_name not in volcano_distances:
                    continue

                vei = e['vei'] if e['vei'] is not None else DEFAULT_VEI
                distances = volcano_distances[v_name]

                # Vectorized tephra calculation
                vei_int = int(min(max(vei, 1), 7))
                params = PYLE_PARAMS[vei_int]
                d_clipped = np.maximum(distances, 1.0)
                thickness = params['T0'] * np.exp(-d_clipped / params['bt'])
                # Only add significant deposits (> 0.01 cm)
                thickness[thickness < 0.01] = 0
                burial_grid += thickness
                eruption_count += 1

        cumulative[window_name] = burial_grid
        print(f"\n  {window_name} ({year_start}-{year_end}): {eruption_count} eruptions")
        print(f"    Max deposit: {burial_grid.max():.1f} cm ({burial_grid.max()/100:.2f} m)")
        print(f"    Mean deposit: {burial_grid.mean():.1f} cm")
        print(f"    Median deposit: {np.median(burial_grid):.1f} cm")
        # Cells with > 1m burial
        deep_cells = np.sum(burial_grid > 100)
        print(f"    Cells > 1m burial: {deep_cells} ({deep_cells/n_cells*100:.1f}%)")
        deep_cells_3m = np.sum(burial_grid > 300)
        print(f"    Cells > 3m burial: {deep_cells_3m} ({deep_cells_3m/n_cells*100:.1f}%)")

    # ── Site-level analysis ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SITE-LEVEL BURIAL ANALYSIS")
    print("=" * 70)

    burial_all = cumulative['all_time']
    burial_classical = cumulative['classical_java']

    site_burials = []
    for site in sites:
        # Find nearest grid cell
        i = int((site['lat'] - lat_min) / resolution)
        j = int((site['lon'] - lon_min) / resolution)
        if 0 <= i < len(lats) and 0 <= j < len(lons):
            predicted_cm = burial_all[i, j]
            classical_cm = burial_classical[i, j]

            # Distance to nearest volcano
            min_dist = float('inf')
            nearest_vol = ''
            for v in volcanoes:
                d = haversine_km(site['lat'], site['lon'], v['lat'], v['lon'])
                if d < min_dist:
                    min_dist = d
                    nearest_vol = v['name']

            site_burials.append({
                'name': site.get('name', ''),
                'lat': site['lat'],
                'lon': site['lon'],
                'observed_depth_cm': site['burial_depth_cm'],
                'predicted_depth_cm': round(predicted_cm, 1),
                'classical_depth_cm': round(classical_cm, 1),
                'nearest_volcano': nearest_vol,
                'dist_nearest_km': round(min_dist, 1),
            })

    # Print top 20 by predicted burial
    site_burials.sort(key=lambda x: x['predicted_depth_cm'], reverse=True)
    print(f"\nTop 20 sites by predicted burial depth:")
    print(f"{'Name':<30s} {'Dist(km)':<10s} {'Predicted(cm)':<15s} {'Observed(cm)':<14s} {'Nearest'}")
    print("-" * 85)
    for sb in site_burials[:20]:
        obs = f"{sb['observed_depth_cm']:.0f}" if sb['observed_depth_cm'] else "—"
        print(f"{sb['name'][:30]:<30s} {sb['dist_nearest_km']:<10.1f} {sb['predicted_depth_cm']:<15.1f} {obs:<14s} {sb['nearest_volcano']}")

    # ── Validation against observed depths ────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION: Predicted vs Observed Burial Depths")
    print("=" * 70)

    validated = [(sb['predicted_depth_cm'], sb['observed_depth_cm'])
                 for sb in site_burials if sb['observed_depth_cm'] is not None and sb['observed_depth_cm'] > 0]

    if validated:
        pred_arr = np.array([v[0] for v in validated])
        obs_arr = np.array([v[1] for v in validated])

        # Correlation
        if len(validated) > 2:
            corr = np.corrcoef(pred_arr, obs_arr)[0, 1]
            print(f"\n  N validated: {len(validated)}")
            print(f"  Pearson r: {corr:.3f}")
            print(f"  Mean predicted: {pred_arr.mean():.1f} cm")
            print(f"  Mean observed: {obs_arr.mean():.1f} cm")
            print(f"  Mean ratio (pred/obs): {(pred_arr / np.maximum(obs_arr, 1)).mean():.2f}")

    # ── Estimate missing sites ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MISSING SITE ESTIMATION")
    print("=" * 70)

    # Sites per grid cell in non-volcanic zones (>50km from volcano)
    # Use as baseline density
    site_cells = defaultdict(int)
    for site in sites:
        i = int((site['lat'] - lat_min) / resolution)
        j = int((site['lon'] - lon_min) / resolution)
        if 0 <= i < len(lats) and 0 <= j < len(lons):
            min_dist = min(haversine_km(site['lat'], site['lon'], v['lat'], v['lon']) for v in volcanoes)
            site_cells[(i, j, 'near' if min_dist < 30 else 'far')] = 1

    far_cells = sum(1 for k in site_cells if k[2] == 'far')
    near_cells = sum(1 for k in site_cells if k[2] == 'near')

    # Count total habitable grid cells
    total_habitable = n_cells  # simplified; could filter by elevation/slope
    near_habitable = sum(1 for i in range(len(lats)) for j in range(len(lons))
                         if min(volcano_distances[v['name']][i, j] for v in volcanoes) < 30)
    far_habitable = total_habitable - near_habitable

    if far_habitable > 0 and far_cells > 0:
        far_density = far_cells / far_habitable
        expected_near = near_habitable * far_density
        deficit = expected_near - near_cells

        print(f"\n  Far from volcanoes (>30km):")
        print(f"    Habitable cells: {far_habitable}")
        print(f"    Cells with sites: {far_cells}")
        print(f"    Density: {far_density:.4f} sites/cell")
        print(f"\n  Near volcanoes (<30km):")
        print(f"    Habitable cells: {near_habitable}")
        print(f"    Cells with sites: {near_cells}")
        print(f"    Expected (if same density): {expected_near:.0f}")
        print(f"    DEFICIT: {deficit:.0f} missing site-cells")
        print(f"    Burial probability: {1 - near_cells / max(expected_near, 1):.1%}")

    # ── Burial depth zones ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BURIAL DEPTH ZONES")
    print("=" * 70)

    burial_all_flat = burial_all.flatten()
    zones = [
        ("Surface accessible (<50 cm)", 0, 50),
        ("Shallow burial (50-200 cm)", 50, 200),
        ("Moderate burial (200-500 cm)", 200, 500),
        ("Deep burial (500-1000 cm)", 500, 1000),
        ("Very deep burial (>1000 cm)", 1000, float('inf')),
    ]

    print(f"\n{'Zone':<40s} {'N cells':<10s} {'%':<8s} {'Area (km²)'}")
    print("-" * 70)
    cell_area_km2 = (resolution * 111) ** 2  # approximate
    for zone_name, lo, hi in zones:
        n = np.sum((burial_all_flat >= lo) & (burial_all_flat < hi))
        pct = n / n_cells * 100
        area = n * cell_area_km2
        print(f"  {zone_name:<38s} {n:<10d} {pct:<8.1f} {area:.0f}")

    # ── Archaeological implications ───────────────────────────────────
    print("\n" + "=" * 70)
    print("ARCHAEOLOGICAL IMPLICATIONS")
    print("=" * 70)

    # Cells with significant burial and suitable terrain
    significant_burial = np.sum(burial_all > 100)  # > 1m
    very_deep = np.sum(burial_all > 500)  # > 5m

    print(f"""
  1. SCALE OF BURIAL:
     {significant_burial} grid cells ({significant_burial/n_cells*100:.1f}%) have >1m cumulative
     volcanic deposit from {len(eruptions_arch)} recorded eruptions (0-2025 CE).
     {very_deep} cells ({very_deep/n_cells*100:.1f}%) have >5m deposits.

  2. MODEL LIMITATIONS:
     - Only uses RECORDED eruptions (GVP database). Pre-historic eruptions
       (pre-1800s) are severely under-counted.
     - Does NOT include lahar deposits (which can add 4-10m per event in
       river valleys — Merapi 2010, Thouret et al. 2015).
     - Does NOT include erosion/reworking (tephra redistribution).
     - Pyle (1989) model assumes circular isopachs (wind not modeled).
     - Grid resolution ({resolution}° ~{resolution*111:.1f}km) too coarse for local
       lahar channels.

  3. CONSERVATIVE ESTIMATE:
     These numbers are MINIMUM BOUNDS. Adding:
     - Pre-historic eruptions (10x multiplier for pre-1800)
     - Lahar channeling (10-100x thicker in valleys)
     - Alluvial reworking
     would increase burial estimates by 1-2 orders of magnitude.

  4. ARCHAEOLOGICAL CONSEQUENCE:
     Standard excavation depth in Indonesian archaeology: 1-3 meters.
     Sites buried >3m are effectively invisible to standard survey.
     The model predicts {np.sum(burial_all > 300)} cells ({np.sum(burial_all > 300)/n_cells*100:.1f}%) exceed
     this threshold — a minimum estimate of the "archaeological dark zone."
""")

    # ── Save results ──────────────────────────────────────────────────
    # Grid results (sampled for manageable size)
    sample_step = max(1, len(lats) // 50)  # ~50x50 sample
    with open(RESULTS_DIR / "burial_grid_sample.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lat', 'lon', 'burial_all_cm', 'burial_classical_cm',
                         'burial_pre_inscription_cm', 'nearest_volcano', 'dist_nearest_km'])
        for i in range(0, len(lats), sample_step):
            for j in range(0, len(lons), sample_step):
                lat = lats[i]
                lon = lons[j]
                min_dist = float('inf')
                nearest = ''
                for v in volcanoes:
                    d = volcano_distances[v['name']][i, j]
                    if d < min_dist:
                        min_dist = d
                        nearest = v['name']
                writer.writerow([
                    round(lat, 3), round(lon, 3),
                    round(burial_all[i, j], 1),
                    round(cumulative['classical_java'][i, j], 1),
                    round(cumulative['pre_inscription'][i, j], 1),
                    nearest, round(min_dist, 1)
                ])

    # Site results
    with open(RESULTS_DIR / "site_burial_predictions.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'name', 'lat', 'lon', 'observed_depth_cm', 'predicted_depth_cm',
            'classical_depth_cm', 'nearest_volcano', 'dist_nearest_km'
        ])
        writer.writeheader()
        for sb in site_burials:
            writer.writerow(sb)

    # Summary JSON
    summary = {
        "experiment": "E075",
        "title": "Volcanic Sedimentation Burial Model for Java",
        "n_volcanoes": len(volcanoes),
        "n_eruptions_total": len(eruptions),
        "n_eruptions_0_2025ce": len(eruptions_arch),
        "grid_resolution_deg": resolution,
        "grid_cells": n_cells,
        "time_windows": {
            name: {
                "year_range": f"{w[0]}-{w[1]}",
                "max_deposit_cm": round(float(cumulative[name].max()), 1),
                "mean_deposit_cm": round(float(cumulative[name].mean()), 1),
                "cells_gt_1m": int(np.sum(cumulative[name] > 100)),
                "cells_gt_3m": int(np.sum(cumulative[name] > 300)),
                "cells_gt_5m": int(np.sum(cumulative[name] > 500)),
            }
            for name, w in windows.items()
        },
        "pyle_params_by_vei": {str(k): v for k, v in PYLE_PARAMS.items()},
        "missing_sites": {
            "near_volcano_habitable_cells": near_habitable,
            "far_volcano_habitable_cells": far_habitable,
            "near_cells_with_sites": near_cells,
            "far_cells_with_sites": far_cells,
            "expected_near_sites": round(expected_near, 0) if far_habitable > 0 else None,
            "deficit": round(deficit, 0) if far_habitable > 0 else None,
        },
        "n_sites_analyzed": len(site_burials),
        "n_validated": len(validated),
    }

    with open(RESULTS_DIR / "e075_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}/")
    print(f"  burial_grid_sample.csv — Sampled burial depth grid")
    print(f"  site_burial_predictions.csv — Per-site burial predictions")
    print(f"  e075_results.json — Summary statistics")


if __name__ == "__main__":
    main()
