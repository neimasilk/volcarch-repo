"""
E100 — Coastal-Highland Archaeological Visibility Inversion
============================================================
Tests whether L1 (volcanic burial, highland) and L2 (coastal submersion)
create a "double blind spot" leaving only the middle zone archaeologically
visible. If true: site density should show an inverse U-shape by elevation.

Experiment #101 in the VOLCARCH series.
"""

import numpy as np
import json
import geopandas as gpd
import rasterio
from scipy import stats
from collections import Counter

print("=" * 70)
print("E100 — COASTAL-HIGHLAND ARCHAEOLOGICAL VISIBILITY INVERSION")
print("=" * 70)

# --- 1. Load sites and extract elevation ---
print("\n[1/5] Loading sites and extracting elevation from DEM...")

sites = gpd.read_file("data/processed/east_java_sites.geojson")
sites_with_geom = sites[sites.geometry.notna()].copy()
print(f"  Total sites: {len(sites)}, with geometry: {len(sites_with_geom)}")

# Extract elevation from DEM
dem_path = "data/processed/dem/jatim_dem.tif"
with rasterio.open(dem_path) as dem:
    dem_data = dem.read(1)
    dem_transform = dem.transform
    dem_nodata = dem.nodata
    dem_bounds = dem.bounds
    dem_crs = dem.crs
    print(f"  DEM: {dem.width}x{dem.height}, CRS: {dem_crs}, bounds: {dem_bounds}")

# Reproject sites to DEM CRS if needed
from pyproj import Transformer
if dem_crs and str(dem_crs) != 'EPSG:4326':
    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    print(f"  Reprojecting sites from EPSG:4326 to {dem_crs}...")
else:
    transformer = None

elevations = []
valid_sites = []
for idx, row in sites_with_geom.iterrows():
    lon, lat = row.geometry.x, row.geometry.y
    if transformer:
        x, y = transformer.transform(lon, lat)
    else:
        x, y = lon, lat
    # Check if within DEM bounds
    if (dem_bounds.left <= x <= dem_bounds.right and
        dem_bounds.bottom <= y <= dem_bounds.top):
        # Convert to pixel coordinates
        col, row_px = ~dem_transform * (x, y)
        col, row_px = int(col), int(row_px)
        if 0 <= row_px < dem_data.shape[0] and 0 <= col < dem_data.shape[1]:
            elev = dem_data[row_px, col]
            if elev != dem_nodata and elev > -100:
                elevations.append(float(elev))
                valid_sites.append(idx)

sites_elev = sites_with_geom.loc[valid_sites].copy()
sites_elev['elevation'] = elevations
print(f"  Sites with elevation: {len(sites_elev)}")
print(f"  Elevation range: {min(elevations):.0f} - {max(elevations):.0f} m")

# --- 2. Elevation zone classification ---
print("\n[2/5] Classifying sites by elevation zone...")

# Define zones
# Coastal: 0-50m (potentially affected by sea level change, low-lying)
# Lowland: 50-200m (transition zone)
# Midslope: 200-500m (middle zone — predicted BRIGHTEST)
# Highland: 500-1000m (volcanic slopes — predicted DARK)
# Mountain: >1000m (near summits — very dark)

zones = {
    'Coastal (0-50m)': (0, 50),
    'Lowland (50-200m)': (50, 200),
    'Midslope (200-500m)': (200, 500),
    'Highland (500-1000m)': (500, 1000),
    'Mountain (>1000m)': (1000, 5000),
}

zone_counts = {}
zone_sites = {}
for zone_name, (lo, hi) in zones.items():
    mask = (sites_elev['elevation'] >= lo) & (sites_elev['elevation'] < hi)
    zone_counts[zone_name] = mask.sum()
    zone_sites[zone_name] = sites_elev[mask]

print(f"\n  {'Zone':<25} {'Sites':>8} {'Pct':>8}")
print(f"  {'-'*25} {'-'*8} {'-'*8}")
total = sum(zone_counts.values())
for zone_name, count in zone_counts.items():
    pct = 100 * count / total if total > 0 else 0
    print(f"  {zone_name:<25} {count:>8} {pct:>7.1f}%")

# --- 3. Compute area per zone (from DEM) ---
print("\n[3/5] Computing area per elevation zone from DEM...")

# Count pixels per zone
# DEM is in UTM (EPSG:32749), pixel size in meters
pixel_w = abs(dem_transform[0])  # meters
pixel_h = abs(dem_transform[4])  # meters
pixel_area_km2 = pixel_w * pixel_h / 1e6  # convert m² to km²
print(f"  Pixel size: {pixel_w:.1f}m x {pixel_h:.1f}m = {pixel_area_km2:.6f} km²")

zone_area = {}
for zone_name, (lo, hi) in zones.items():
    mask = (dem_data >= lo) & (dem_data < hi) & (dem_data != dem_nodata)
    zone_area[zone_name] = float(mask.sum() * pixel_area_km2)

print(f"\n  {'Zone':<25} {'Area (km2)':>12} {'Sites':>8} {'Density':>12}")
print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*12}")
zone_density = {}
for zone_name in zones:
    area = zone_area[zone_name]
    count = zone_counts[zone_name]
    density = count / area * 1000 if area > 0 else 0  # sites per 1000 km2
    zone_density[zone_name] = density
    print(f"  {zone_name:<25} {area:>11.0f} {count:>8} {density:>10.2f}/1000km2")

# --- 4. Statistical test: inverse U-shape ---
print("\n[4/5] Testing inverse U-shape hypothesis...")

# Assign numeric zone order for trend test
zone_order = list(zones.keys())
densities = [zone_density[z] for z in zone_order]
midpoints = [25, 125, 350, 750, 1500]  # zone midpoint elevations

# Test: is midslope density significantly higher than highland + coastal?
coastal_density = zone_density['Coastal (0-50m)']
lowland_density = zone_density['Lowland (50-200m)']
midslope_density = zone_density['Midslope (200-500m)']
highland_density = zone_density['Highland (500-1000m)']
mountain_density = zone_density['Mountain (>1000m)']

# Margins (coastal + highland) vs middle
margin_density = (coastal_density + highland_density + mountain_density) / 3
middle_density = (lowland_density + midslope_density) / 2

print(f"\n  Margin zones mean density: {margin_density:.2f}/1000km2")
print(f"  Middle zones mean density: {middle_density:.2f}/1000km2")
print(f"  Ratio (middle/margin): {middle_density/margin_density:.2f}x" if margin_density > 0 else "  Margin density = 0")

# Quadratic test: fit parabola to density vs elevation
# If coefficient of x^2 is negative → inverse U-shape
from numpy.polynomial import polynomial as P
coeffs = np.polyfit(midpoints, densities, 2)
print(f"\n  Quadratic fit: {coeffs[0]:.6f}x^2 + {coeffs[1]:.4f}x + {coeffs[2]:.2f}")
print(f"  x^2 coefficient: {coeffs[0]:.6f} ({'NEGATIVE = inverse U' if coeffs[0] < 0 else 'POSITIVE = U-shape'})")

# Peak of parabola
if coeffs[0] != 0:
    peak_elev = -coeffs[1] / (2 * coeffs[0])
    print(f"  Predicted peak elevation: {peak_elev:.0f} m")

# Kruskal-Wallis test across zones
zone_elevations_list = []
zone_labels = []
for i, (zone_name, (lo, hi)) in enumerate(zones.items()):
    n = zone_counts[zone_name]
    zone_elevations_list.extend([i] * n)
    zone_labels.extend([zone_name] * n)

# Mann-Whitney: midslope vs highland
if zone_counts['Midslope (200-500m)'] > 5 and zone_counts['Highland (500-1000m)'] > 5:
    # Compare density directly is tricky with count data
    # Use chi-square: observed vs expected (proportional to area)
    observed = [zone_counts[z] for z in zone_order]
    total_area = sum(zone_area.values())
    expected = [zone_area[z] / total_area * total for z in zone_order]
    # Remove zones with 0 expected
    obs_filt = [o for o, e in zip(observed, expected) if e > 0]
    exp_filt = [e for e in expected if e > 0]
    chi2, p_chi2 = stats.chisquare(obs_filt, f_exp=exp_filt)
    print(f"\n  Chi-square (observed vs area-proportional expected): chi2={chi2:.2f}, p={p_chi2:.6f}")
    print(f"  Sites are {'NON-UNIFORMLY' if p_chi2 < 0.05 else 'uniformly'} distributed across elevation zones")

# --- 5. Volcano distance interaction ---
print("\n[5/5] Volcano distance × elevation interaction...")

# Major volcanoes
volcanoes = {
    'Kelud': (-7.93, 112.31),
    'Arjuno-Welirang': (-7.73, 112.58),
    'Semeru': (-8.11, 112.92),
    'Bromo': (-7.94, 112.95),
    'Penanggungan': (-7.62, 112.63),
}

# Compute distance to nearest volcano for each site
from math import radians, sin, cos, sqrt, atan2
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

sites_elev['volcano_dist_km'] = sites_elev.apply(
    lambda row: min(haversine(row.geometry.y, row.geometry.x, v[0], v[1]) for v in volcanoes.values()),
    axis=1
)

# Correlation: elevation vs volcano distance
rho_ev, p_ev = stats.spearmanr(sites_elev['elevation'], sites_elev['volcano_dist_km'])
print(f"  Elevation × volcano distance: rho={rho_ev:.4f}, p={p_ev:.6f}")

# Sites in "double blind spot" (high elevation + close to volcano)
blind_spot = sites_elev[(sites_elev['elevation'] > 500) & (sites_elev['volcano_dist_km'] < 15)]
print(f"  Sites in blind spot (>500m, <15km from volcano): {len(blind_spot)}/{len(sites_elev)} ({100*len(blind_spot)/len(sites_elev):.1f}%)")

# Sites in "visible middle" (200-500m, >15km from volcano)
visible_middle = sites_elev[(sites_elev['elevation'] >= 200) & (sites_elev['elevation'] < 500) & (sites_elev['volcano_dist_km'] >= 15)]
print(f"  Sites in visible middle (200-500m, >15km): {len(visible_middle)}/{len(sites_elev)} ({100*len(visible_middle)/len(sites_elev):.1f}%)")

# 2D density: elevation bins × distance bins
print("\n  Elevation × Distance density matrix (sites per cell):")
elev_bins = [0, 100, 300, 500, 1000, 3000]
dist_bins = [0, 10, 20, 30, 50, 100]
elev_labels = ['0-100', '100-300', '300-500', '500-1000', '1000+']
dist_labels = ['0-10km', '10-20km', '20-30km', '30-50km', '50+km']

matrix = np.zeros((len(elev_labels), len(dist_labels)))
for i in range(len(elev_bins)-1):
    for j in range(len(dist_bins)-1):
        mask = ((sites_elev['elevation'] >= elev_bins[i]) & (sites_elev['elevation'] < elev_bins[i+1]) &
                (sites_elev['volcano_dist_km'] >= dist_bins[j]) & (sites_elev['volcano_dist_km'] < dist_bins[j+1]))
        matrix[i, j] = mask.sum()

print(f"  {'':>10}", end='')
for dl in dist_labels:
    print(f"  {dl:>8}", end='')
print()
for i, el in enumerate(elev_labels):
    print(f"  {el:>10}", end='')
    for j in range(len(dist_labels)):
        print(f"  {int(matrix[i,j]):>8}", end='')
    print()

# --- Save results ---
results = {
    'meta': {
        'experiment': 'E100',
        'date': '2026-03-17',
        'n_sites_total': len(sites),
        'n_sites_with_elevation': len(sites_elev),
        'dem': 'jatim_dem.tif (GLO-30)',
    },
    'zone_analysis': {
        z: {
            'sites': int(zone_counts[z]),
            'area_km2': float(zone_area[z]),
            'density_per_1000km2': float(zone_density[z]),
        } for z in zone_order
    },
    'inverse_u_test': {
        'quadratic_x2_coeff': float(coeffs[0]),
        'is_inverse_u': bool(coeffs[0] < 0),
        'peak_elevation_m': float(peak_elev) if coeffs[0] != 0 else None,
        'middle_vs_margin_ratio': float(middle_density / margin_density) if margin_density > 0 else None,
        'chi2': float(chi2),
        'chi2_p': float(p_chi2),
    },
    'volcano_interaction': {
        'elev_dist_rho': float(rho_ev),
        'elev_dist_p': float(p_ev),
        'blind_spot_sites': int(len(blind_spot)),
        'visible_middle_sites': int(len(visible_middle)),
    },
    'density_matrix': {
        'elev_bins': elev_labels,
        'dist_bins': dist_labels,
        'matrix': matrix.tolist(),
    },
}

with open("experiments/E100_coastal_highland_visibility/results/e100_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("E100 SUMMARY")
print("=" * 70)
print(f"  Sites with elevation: {len(sites_elev)}")
print(f"  Quadratic x^2: {coeffs[0]:.6f} ({'INVERSE U' if coeffs[0] < 0 else 'U-SHAPE'})")
print(f"  Peak elevation: {peak_elev:.0f} m")
print(f"  Middle/Margin ratio: {middle_density/margin_density:.2f}x" if margin_density > 0 else "")
print(f"  Chi-square: {chi2:.2f}, p={p_chi2:.6f}")
print(f"  Blind spot (<15km, >500m): {len(blind_spot)} sites ({100*len(blind_spot)/len(sites_elev):.1f}%)")
print("=" * 70)
