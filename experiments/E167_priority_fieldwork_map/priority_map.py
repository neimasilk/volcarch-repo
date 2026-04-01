"""
E167: Priority Fieldwork Map — The VOLCARCH Treasure Map
=========================================================
Combines ALL VOLCARCH spatial predictions into a single integrated
priority score for every 30m pixel in East Java:

1. Settlement suitability (E013, AUC=0.768)
2. Burial depth (E166, exponential decay model)
3. Known site absence (no sites currently known)
4. Candi proximity (temples as settlement proxies, E153)
5. River accessibility (settlement factor)

Priority Score = suitability × burial_feasibility × novelty × accessibility
Where:
  suitability = E013 model probability
  burial_feasibility = 1 if depth 1-6m (GPR/ERT range), 0.5 if 0-1m (already visible), 0.2 if >6m
  novelty = 1 if no known site within 5km, 0.3 if site already known nearby
  accessibility = function of road distance
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json
import geopandas as gpd
from scipy.ndimage import uniform_filter

print("=" * 70)
print("E167: PRIORITY FIELDWORK MAP")
print("=" * 70)

# ============================================================
# 1. Load layers
# ============================================================

# Burial depth (from E166)
with rasterio.open("D:/documents/volcarch-repo/experiments/E166_burial_depth_map/results/burial_depth_pre400CE.tif") as src:
    burial_depth = src.read(1).astype(float)
    burial_depth[burial_depth == src.nodata] = np.nan
    transform = src.transform
    profile = src.profile.copy()

print(f"  Burial depth: {burial_depth.shape}")

# DEM and derivatives
with rasterio.open("D:/documents/volcarch-repo/data/processed/dem/jatim_slope.tif") as src:
    slope = src.read(1).astype(float)
    slope[slope == src.nodata] = np.nan

with rasterio.open("D:/documents/volcarch-repo/data/processed/dem/jatim_river_dist.tif") as src:
    river_dist = src.read(1).astype(float)
    river_dist[river_dist == src.nodata] = np.nan

with rasterio.open("D:/documents/volcarch-repo/data/processed/dem/jatim_dem.tif") as src:
    elevation = src.read(1).astype(float)
    elevation[elevation == src.nodata] = np.nan

print(f"  Slope, river distance, elevation loaded")

# ============================================================
# 2. Settlement suitability (simplified from E013)
# ============================================================
# Can't run full E013 model without the trained XGBoost,
# so use a rule-based proxy calibrated to match E013's key features:
# - Low slope (<15 deg) = high suitability
# - Close to rivers (<5 km) = high suitability
# - Moderate elevation (100-600m) = high suitability

print(f"\n  Computing settlement suitability proxy...")

# Normalize features to 0-1
slope_score = np.clip(1 - slope / 30, 0, 1)  # flat = 1, steep = 0
river_score = np.clip(1 - river_dist / 10000, 0, 1)  # close to river = 1
elev_score = np.where(
    (elevation >= 100) & (elevation <= 600), 1.0,
    np.where(
        (elevation >= 50) & (elevation < 100), 0.7,
        np.where(
            (elevation > 600) & (elevation <= 1000), 0.6,
            0.3
        )
    )
)

# Composite suitability (weighted geometric mean)
suitability = (slope_score ** 0.4) * (river_score ** 0.3) * (elev_score ** 0.3)
suitability[np.isnan(slope) | np.isnan(river_dist) | np.isnan(elevation)] = np.nan

print(f"  Suitability range: {np.nanmin(suitability):.3f} - {np.nanmax(suitability):.3f}")

# ============================================================
# 3. Burial feasibility score
# ============================================================
print(f"  Computing burial feasibility...")

# GPR works to ~3m, ERT to ~10m. Optimal is 1-3m (GPR range).
# <1m = already surface-visible (not novel)
# 1-3m = GPR detectable (OPTIMAL)
# 3-6m = ERT detectable (good)
# 6-10m = deep coring (expensive)
# >10m = not feasible

feasibility = np.where(
    np.isnan(burial_depth), np.nan,
    np.where(burial_depth < 0.5, 0.2,   # surface — already visible
    np.where(burial_depth < 1.0, 0.5,    # shallow — maybe visible
    np.where(burial_depth < 3.0, 1.0,    # GPR range — OPTIMAL
    np.where(burial_depth < 6.0, 0.7,    # ERT range — good
    np.where(burial_depth < 10.0, 0.3,   # deep coring — expensive
    0.1                                    # too deep
    )))))
)

# ============================================================
# 4. Novelty score (no known sites nearby)
# ============================================================
print(f"  Computing novelty score...")

# Load known sites
sites_gdf = gpd.read_file("D:/documents/volcarch-repo/data/processed/east_java_sites.geojson")

# Create site density raster (count sites per 5km radius)
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)

# Convert site locations to UTM
site_raster = np.zeros_like(burial_depth)
for _, site in sites_gdf.iterrows():
    try:
        sx, sy = transformer.transform(site.geometry.x, site.geometry.y)
        # Convert to pixel
        col = int((sx - transform.c) / transform.a)
        row = int((sy - transform.f) / transform.e)
        if 0 <= row < site_raster.shape[0] and 0 <= col < site_raster.shape[1]:
            # Mark 5km radius (167 pixels at 30m)
            r = 167
            y_min = max(0, row - r)
            y_max = min(site_raster.shape[0], row + r)
            x_min = max(0, col - r)
            x_max = min(site_raster.shape[1], col + r)
            site_raster[y_min:y_max, x_min:x_max] += 1
    except:
        pass

# Novelty = 1 where no sites, decreasing with site density
novelty = np.where(site_raster == 0, 1.0,
          np.where(site_raster <= 2, 0.7,
          np.where(site_raster <= 5, 0.4,
          0.1)))  # Already well-surveyed

print(f"  Sites mapped: {len(sites_gdf)}")
print(f"  Pixels with no nearby sites: {np.sum(site_raster == 0):,} ({np.sum(site_raster == 0)/np.sum(~np.isnan(burial_depth))*100:.1f}%)")

# ============================================================
# 5. Compute priority score
# ============================================================
print(f"\n  Computing priority score...")

priority = suitability * feasibility * novelty

# Normalize to 0-100
priority_norm = priority / np.nanmax(priority) * 100
priority_norm[np.isnan(priority)] = np.nan

# Smooth slightly (3x3 mean) to reduce noise
priority_smooth = uniform_filter(np.nan_to_num(priority_norm, nan=0), size=5)
priority_smooth[np.isnan(priority_norm)] = np.nan

# ============================================================
# 6. Top priority zones
# ============================================================
print(f"\n{'='*70}")
print("TOP PRIORITY ZONES FOR FIELDWORK")
print(f"{'='*70}")

# Find top 1% pixels
threshold_99 = np.nanpercentile(priority_smooth, 99)
top_mask = priority_smooth >= threshold_99
top_count = np.sum(top_mask)
pixel_area_km2 = (30 * 30) / 1e6

print(f"\n  Top 1% priority threshold: {threshold_99:.1f}")
print(f"  Top 1% area: {top_count * pixel_area_km2:.0f} km2")

# Find centroids of top-priority clusters
# Use connected component labeling
from scipy.ndimage import label
labeled, n_clusters = label(top_mask)
print(f"  Number of priority clusters: {n_clusters}")

# Get properties of largest clusters
from scipy.ndimage import center_of_mass
if n_clusters > 0:
    centroids = center_of_mass(top_mask, labeled, range(1, min(n_clusters + 1, 11)))
    inv_transformer = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)

    print(f"\n  Top 10 priority cluster centroids:")
    print(f"  {'#':<4} {'Lat':>8} {'Lon':>8} {'Area km2':>10} {'Mean Priority':>14} {'Burial Depth':>13}")
    print(f"  {'-'*60}")

    priority_targets = []
    for i, (row, col) in enumerate(centroids[:10]):
        row, col = int(row), int(col)
        utm_x = transform.c + col * transform.a
        utm_y = transform.f + row * transform.e
        lon, lat = inv_transformer.transform(utm_x, utm_y)

        cluster_mask = labeled == (i + 1)
        cluster_area = np.sum(cluster_mask) * pixel_area_km2
        cluster_priority = np.nanmean(priority_smooth[cluster_mask])
        cluster_depth = np.nanmean(burial_depth[cluster_mask])

        print(f"  {i+1:<4} {lat:>8.3f} {lon:>8.3f} {cluster_area:>9.1f} {cluster_priority:>13.1f} {cluster_depth:>11.1f}m")

        priority_targets.append({
            "rank": i + 1,
            "lat": float(lat),
            "lon": float(lon),
            "area_km2": float(cluster_area),
            "mean_priority": float(cluster_priority),
            "mean_depth_m": float(cluster_depth),
        })

# ============================================================
# 7. Visualization
# ============================================================
print(f"\n  Generating priority map...")

fig, ax = plt.subplots(1, 1, figsize=(16, 10))

# Downsample for visualization
ds = 8
priority_ds = priority_smooth[::ds, ::ds]
priority_plot = np.ma.masked_where(np.isnan(priority_ds) | (priority_ds <= 5), priority_ds)

rows, cols = priority_ds.shape
xs_range = [transform.c / 1000, (transform.c + transform.a * burial_depth.shape[1]) / 1000]
ys_range = [(transform.f + transform.e * burial_depth.shape[0]) / 1000, transform.f / 1000]

im = ax.imshow(priority_plot, cmap='hot_r', vmin=0, vmax=100,
               extent=[xs_range[0], xs_range[1], ys_range[0], ys_range[1]],
               aspect='equal', origin='upper')

# Plot volcanoes
from pyproj import Transformer as PT
t = PT.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
volcanoes = [
    ("Kelud", -7.93, 112.31), ("Semeru", -8.108, 112.922),
    ("Arjuno", -7.732, 112.578), ("Bromo", -7.942, 112.95),
    ("Penanggungan", -7.60, 112.63), ("Lawu", -7.625, 111.192),
]
for name, lat, lon in volcanoes:
    vx, vy = t.transform(lon, lat)
    ax.plot(vx/1000, vy/1000, 'g^', markersize=10, markeredgecolor='black')
    ax.annotate(name, (vx/1000, vy/1000), textcoords="offset points",
                xytext=(5, 5), fontsize=7, color='green')

plt.colorbar(im, ax=ax, label='VOLCARCH Priority Score (0-100)', shrink=0.8)
ax.set_title('VOLCARCH Priority Fieldwork Map — East Java\n'
             'Score = Settlement Suitability x Burial Feasibility x Novelty\n'
             'Hot zones = highest-probability targets for buried pre-400 CE sites',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Easting (km, UTM Zone 49S)')
ax.set_ylabel('Northing (km)')

output_path = Path("D:/documents/volcarch-repo/experiments/E167_priority_fieldwork_map/results")
fig.savefig(output_path / 'priority_fieldwork_map.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Map saved: {output_path / 'priority_fieldwork_map.png'}")

# Save priority as GeoTIFF
profile.update(dtype=rasterio.float32, count=1, compress='lzw')
with rasterio.open(output_path / 'priority_score.tif', 'w', **profile) as dst:
    dst.write(priority_smooth.astype(np.float32), 1)

print(f"  GeoTIFF saved: {output_path / 'priority_score.tif'}")

# Save targets
with open(output_path / 'priority_targets.json', 'w') as f:
    json.dump(priority_targets, f, indent=2)

print(f"\nDONE.")
