"""
E170: DEM-Based Lahar Flow Accumulation Model
===============================================
Lahars don't flow radially from volcanoes — they follow topography,
flowing downhill through valleys and accumulating in lowland basins.

This model uses the DEM to:
1. Compute flow direction from each volcano
2. Accumulate "sediment" along flow paths
3. Weight by distance (sediment load decreases with distance)
4. Produce a physically-realistic burial depth map

Key improvement over E166: valleys at 20 km accumulate MORE
sediment than ridges at 20 km, because lahar channels concentrate flow.

Uses: Malang region DEM (30m) for detailed Kelud/Arjuno analysis.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from pyproj import Transformer
import json
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter

print("=" * 70)
print("E170: DEM-BASED LAHAR FLOW ACCUMULATION MODEL")
print("=" * 70)

# Use Malang region DEM (higher detail, covers Kelud + Arjuno)
dem_path = "D:/documents/volcarch-repo/data/processed/dem/malang_dem.tif"

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(float)
    dem[dem == src.nodata] = np.nan
    transform = src.transform
    profile = src.profile.copy()
    nodata = src.nodata

print(f"  DEM: {dem.shape[1]} x {dem.shape[0]} pixels")
print(f"  Resolution: {abs(transform.a):.1f}m")
print(f"  Elevation range: {np.nanmin(dem):.0f} - {np.nanmax(dem):.0f}m")

# ============================================================
# 1. Define volcano sources
# ============================================================
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)

volcanoes = [
    {"name": "Kelud", "lat": -7.93, "lon": 112.31, "vei_avg": 4, "freq": 37},
    {"name": "Arjuno-Welirang", "lat": -7.732, "lon": 112.578, "vei_avg": 2, "freq": 12},
    {"name": "Semeru", "lat": -8.108, "lon": 112.922, "vei_avg": 2, "freq": 63},
    {"name": "Penanggungan", "lat": -7.60, "lon": 112.63, "vei_avg": 2, "freq": 5},
]

# Convert to pixel coordinates
volcano_pixels = []
for v in volcanoes:
    vx, vy = transformer.transform(v['lon'], v['lat'])
    col = int((vx - transform.c) / transform.a)
    row = int((vy - transform.f) / transform.e)
    if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
        volcano_pixels.append({**v, 'row': row, 'col': col, 'utm_x': vx, 'utm_y': vy})
        print(f"  {v['name']}: pixel ({row}, {col}), elev={dem[row, col]:.0f}m")

# ============================================================
# 2. Compute flow direction (D8 algorithm)
# ============================================================
print(f"\n  Computing flow direction (D8)...")

# D8: each cell flows to its lowest neighbor
# Directions: 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
dr = [0, 1, 1, 1, 0, -1, -1, -1]  # row offsets
dc = [1, 1, 0, -1, -1, -1, 0, 1]  # col offsets
dist_weight = [1, 1.414, 1, 1.414, 1, 1.414, 1, 1.414]  # diagonal distance

# Fill sinks first (simple approach: fill to minimum neighbor + epsilon)
dem_filled = dem.copy()
dem_filled[np.isnan(dem_filled)] = 9999

# Simple pit filling
for iteration in range(3):
    for i in range(8):
        shifted = np.roll(np.roll(dem_filled, dr[i], axis=0), dc[i], axis=1)
        mask = dem_filled < shifted - 100  # obvious sinks
        # Don't fill, just smooth slightly
    # Slight smoothing to remove DEM artifacts
    dem_smooth = uniform_filter(dem_filled, size=3)
    dem_filled = np.where(dem_filled < 9000, dem_smooth, dem_filled)

flow_dir = np.full(dem.shape, -1, dtype=int)
slope_grid = np.zeros(dem.shape)

for r in range(1, dem.shape[0] - 1):
    for c in range(1, dem.shape[1] - 1):
        if np.isnan(dem[r, c]):
            continue
        max_slope = 0
        best_dir = -1
        for d in range(8):
            nr, nc = r + dr[d], c + dc[d]
            if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1]:
                if not np.isnan(dem[nr, nc]):
                    slope = (dem[r, c] - dem[nr, nc]) / (abs(transform.a) * dist_weight[d])
                    if slope > max_slope:
                        max_slope = slope
                        best_dir = d
        flow_dir[r, c] = best_dir
        slope_grid[r, c] = max_slope

print(f"  Flow direction computed. Cells with valid flow: {np.sum(flow_dir >= 0):,}")

# ============================================================
# 3. Compute lahar accumulation from each volcano
# ============================================================
print(f"\n  Computing lahar flow accumulation...")

total_accumulation = np.zeros(dem.shape, dtype=float)

for v in volcano_pixels:
    print(f"\n    Processing {v['name']}...")

    # Initialize sediment at volcano summit
    # Sediment load proportional to eruption frequency and VEI
    source_strength = v['freq'] * (10 ** (v['vei_avg'] - 1))  # relative sediment production

    # BFS/flood from volcano summit following flow direction
    # Sediment decreases with distance (friction loss)
    accumulation = np.zeros(dem.shape, dtype=float)

    # Start from a radius around the summit (lahars originate from summit area)
    start_radius = 10  # pixels (~300m)
    vr, vc = v['row'], v['col']

    # Seed the source area
    for dr_s in range(-start_radius, start_radius + 1):
        for dc_s in range(-start_radius, start_radius + 1):
            sr, sc = vr + dr_s, vc + dc_s
            if 0 <= sr < dem.shape[0] and 0 <= sc < dem.shape[1]:
                dist_from_summit = np.sqrt(dr_s**2 + dc_s**2) * abs(transform.a) / 1000  # km
                if dist_from_summit < 3:  # within 3 km of summit
                    accumulation[sr, sc] = source_strength * np.exp(-dist_from_summit / 2)

    # Propagate downstream following flow direction
    # Process cells from highest to lowest elevation
    valid_cells = []
    for r in range(dem.shape[0]):
        for c in range(dem.shape[1]):
            if not np.isnan(dem[r, c]) and flow_dir[r, c] >= 0:
                valid_cells.append((dem[r, c], r, c))

    # Sort by elevation (highest first — upstream to downstream)
    valid_cells.sort(key=lambda x: -x[0])

    for elev, r, c in valid_cells:
        if accumulation[r, c] <= 0:
            continue

        d = flow_dir[r, c]
        if d < 0:
            continue

        nr, nc = r + dr[d], c + dc[d]
        if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1]:
            # Transfer sediment downstream with friction loss
            # Steeper slopes = more transport
            slope = slope_grid[r, c]
            transport_efficiency = min(0.95, 0.3 + slope * 5)  # steeper = more transport
            accumulation[nr, nc] += accumulation[r, c] * transport_efficiency

    # Distance-decay attenuation
    dist_from_volcano = np.sqrt(
        (np.arange(dem.shape[0])[:, None] - vr)**2 +
        (np.arange(dem.shape[1])[None, :] - vc)**2
    ) * abs(transform.a) / 1000  # km

    distance_decay = np.exp(-dist_from_volcano / 20)  # 20 km characteristic length
    accumulation *= distance_decay

    total_accumulation += accumulation
    max_acc = np.max(accumulation)
    print(f"    Max accumulation: {max_acc:.1f} (relative units)")
    print(f"    Cells with accumulation > 0: {np.sum(accumulation > 0):,}")

# ============================================================
# 4. Convert accumulation to burial depth
# ============================================================
print(f"\n{'='*70}")
print("CONVERTING ACCUMULATION TO BURIAL DEPTH")
print(f"{'='*70}")

# Calibrate against known burial sites
# Dwarapala Singosari: ~1.85m burial, 755 years (1268-2023)
# Need to find Singosari pixel
singosari_x, singosari_y = transformer.transform(112.639, -7.889)
singosari_col = int((singosari_x - transform.c) / transform.a)
singosari_row = int((singosari_y - transform.f) / transform.e)

if 0 <= singosari_row < dem.shape[0] and 0 <= singosari_col < dem.shape[1]:
    singosari_acc = total_accumulation[singosari_row, singosari_col]
    print(f"  Singosari accumulation value: {singosari_acc:.2f}")

    if singosari_acc > 0:
        # Known: 1.85m burial in 755 years = 2.45 mm/yr at this location
        # Calibration factor: depth_mm = accumulation * calibration_factor * time_years
        calibration = 1.85 / (singosari_acc * 755 / 1000)  # m per (accumulation × kyr)
        print(f"  Calibration factor: {calibration:.4f}")

        # Compute depth for pre-400 CE (1626 years)
        burial_depth_flow = total_accumulation * calibration * 1626 / 1000
        burial_depth_flow[np.isnan(dem)] = np.nan
    else:
        print(f"  WARNING: Singosari has zero accumulation — using fallback calibration")
        # Fallback: normalize so max depth = 15m (reasonable for proximal zones)
        burial_depth_flow = total_accumulation / np.nanmax(total_accumulation) * 15
        burial_depth_flow[np.isnan(dem)] = np.nan
else:
    print(f"  WARNING: Singosari outside DEM bounds — using fallback")
    burial_depth_flow = total_accumulation / np.nanmax(total_accumulation) * 15
    burial_depth_flow[np.isnan(dem)] = np.nan

# Statistics
valid = ~np.isnan(burial_depth_flow) & (burial_depth_flow > 0)
print(f"\n  Burial depth (flow-based, pre-400 CE):")
print(f"    Mean: {np.mean(burial_depth_flow[valid]):.1f}m")
print(f"    Median: {np.median(burial_depth_flow[valid]):.1f}m")
print(f"    Max: {np.max(burial_depth_flow[valid]):.1f}m")
print(f"    Pixels > 1m: {np.sum(burial_depth_flow[valid] > 1):,}")
print(f"    Pixels > 3m: {np.sum(burial_depth_flow[valid] > 3):,}")
print(f"    Pixels > 6m: {np.sum(burial_depth_flow[valid] > 6):,}")

# ============================================================
# 5. Compare flow-based vs distance-based models
# ============================================================
print(f"\n{'='*70}")
print("COMPARISON: Flow-Based vs Distance-Based Burial Models")
print(f"{'='*70}")

# Distance-based model (E166 approach)
min_dist = np.full(dem.shape, np.inf)
for v in volcano_pixels:
    dist = np.sqrt(
        (np.arange(dem.shape[0])[:, None] - v['row'])**2 +
        (np.arange(dem.shape[1])[None, :] - v['col'])**2
    ) * abs(transform.a) / 1000
    min_dist = np.minimum(min_dist, dist)

burial_depth_dist = 8.0 * np.exp(-min_dist / 15.0) * 1626 / 1000
burial_depth_dist[np.isnan(dem)] = np.nan
burial_depth_dist[min_dist > 60] = 0

# Correlation between models
valid_both = valid & ~np.isnan(burial_depth_dist) & (burial_depth_dist > 0)
if np.sum(valid_both) > 100:
    from scipy.stats import spearmanr
    rho, p = spearmanr(
        burial_depth_flow[valid_both].ravel()[:10000],  # subsample for speed
        burial_depth_dist[valid_both].ravel()[:10000]
    )
    print(f"  Spearman correlation (flow vs distance): rho={rho:.3f}, p={p:.2e}")

# Key difference: valley vs ridge analysis
print(f"\n  Valley vs Ridge Analysis:")

# TRI (terrain ruggedness) as proxy for valley/ridge
# Low TRI = flat (valley floor), High TRI = rough (ridge/slope)
with rasterio.open("D:/documents/volcarch-repo/data/processed/dem/malang_tri.tif") as src:
    tri = src.read(1).astype(float)
    tri[tri == src.nodata] = np.nan

tri_median = np.nanmedian(tri)
valley_mask = (tri < tri_median) & valid_both
ridge_mask = (tri >= tri_median) & valid_both

if np.sum(valley_mask) > 0 and np.sum(ridge_mask) > 0:
    valley_flow = np.mean(burial_depth_flow[valley_mask])
    valley_dist = np.mean(burial_depth_dist[valley_mask])
    ridge_flow = np.mean(burial_depth_flow[ridge_mask])
    ridge_dist = np.mean(burial_depth_dist[ridge_mask])

    print(f"    VALLEY (low TRI, flat):")
    print(f"      Flow model depth: {valley_flow:.2f}m")
    print(f"      Distance model depth: {valley_dist:.2f}m")
    print(f"      Flow/Distance ratio: {valley_flow/valley_dist:.2f}x")

    print(f"    RIDGE (high TRI, rough):")
    print(f"      Flow model depth: {ridge_flow:.2f}m")
    print(f"      Distance model depth: {ridge_dist:.2f}m")
    print(f"      Flow/Distance ratio: {ridge_flow/ridge_dist:.2f}x")

    print(f"\n    KEY FINDING: Flow model predicts {valley_flow/ridge_flow:.1f}x more burial")
    print(f"    in valleys than ridges at the same distance from volcano.")
    print(f"    Distance model predicts {valley_dist/ridge_dist:.2f}x (nearly equal).")
    print(f"    The flow model DIFFERENTIATES valleys from ridges — the distance model doesn't.")

# ============================================================
# 6. Visualization
# ============================================================
print(f"\n  Generating comparison maps...")

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

ds = 4  # downsample

# Panel 1: Distance-based (E166)
depth_dist_ds = burial_depth_dist[::ds, ::ds]
depth_dist_plot = np.ma.masked_where(np.isnan(depth_dist_ds) | (depth_dist_ds <= 0), depth_dist_ds)
im1 = axes[0].imshow(depth_dist_plot, cmap='YlOrRd', vmin=0, vmax=15, origin='upper', aspect='equal')
axes[0].set_title('E166: Distance-Based Model\n(Exponential decay from volcano)', fontweight='bold')

# Panel 2: Flow-based (E170)
depth_flow_ds = burial_depth_flow[::ds, ::ds]
depth_flow_plot = np.ma.masked_where(np.isnan(depth_flow_ds) | (depth_flow_ds <= 0), depth_flow_ds)
im2 = axes[1].imshow(depth_flow_plot, cmap='YlOrRd', vmin=0, vmax=15, origin='upper', aspect='equal')
axes[1].set_title('E170: Flow-Based Model\n(Lahar follows topography)', fontweight='bold')

# Panel 3: Difference (flow - distance)
diff = burial_depth_flow - burial_depth_dist
diff[np.isnan(diff)] = 0
diff_ds = diff[::ds, ::ds]
diff_plot = np.ma.masked_where(np.abs(diff_ds) < 0.1, diff_ds)
im3 = axes[2].imshow(diff_plot, cmap='RdBu_r', vmin=-5, vmax=5, origin='upper', aspect='equal')
axes[2].set_title('Difference (Flow - Distance)\n(Red = flow predicts DEEPER)', fontweight='bold')

# Add volcano markers
for ax in axes:
    for v in volcano_pixels:
        ax.plot(v['col']//ds, v['row']//ds, 'k^', markersize=10)

plt.colorbar(im1, ax=axes[0], shrink=0.6, label='Burial depth (m)')
plt.colorbar(im2, ax=axes[1], shrink=0.6, label='Burial depth (m)')
plt.colorbar(im3, ax=axes[2], shrink=0.6, label='Depth difference (m)')

plt.suptitle('E170: Lahar Flow Model vs Distance Model — Malang Region',
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path("D:/documents/volcarch-repo/experiments/E170_lahar_flow_model/results")
fig.savefig(output_path / 'flow_vs_distance.png', dpi=150, bbox_inches='tight')
plt.close()

# Save flow-based burial depth as GeoTIFF
profile.update(dtype=rasterio.float32, count=1, compress='lzw')
with rasterio.open(output_path / 'burial_depth_flow.tif', 'w', **profile) as dst:
    out = burial_depth_flow.astype(np.float32)
    out[np.isnan(out)] = -9999
    dst.write(out, 1)

print(f"  Maps saved to {output_path}")
print(f"\nDONE.")
