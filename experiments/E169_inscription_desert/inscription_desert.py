"""
E169: Inscription Desert Analysis
===================================
Given the spatial patterns of where inscriptions ARE found (E084),
model where inscriptions SHOULD be found but AREN'T.

These 'inscription deserts' are zones where:
- Settlement suitability is high (people lived there)
- Volcanic burial depth is significant (evidence is hidden)
- No inscriptions exist (cultural production is invisible)

The deserts are the dark heart of VOLCARCH's argument.
"""

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from pyproj import Transformer
from scipy.stats import gaussian_kde
import json

print("=" * 70)
print("E169: INSCRIPTION DESERT ANALYSIS")
print("=" * 70)

# ============================================================
# 1. Load inscription and candi locations
# ============================================================
insc_df = pd.read_csv("D:/documents/volcarch-repo/experiments/E082_inscription_georeferencing/results/geocoded_inscriptions.csv")
candi_df = pd.read_csv("D:/documents/volcarch-repo/experiments/E031_candi_orientation/results/candi_volcano_pairs.csv")

# Filter to Java
insc_java = insc_df[
    (insc_df['lat'] > -9) & (insc_df['lat'] < -5.5) &
    (insc_df['lon'] > 105) & (insc_df['lon'] < 115)
].copy()

print(f"  Java inscriptions: {len(insc_java)}")
print(f"  Candi: {len(candi_df)}")

# Convert to UTM
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)

insc_x, insc_y = [], []
for _, row in insc_java.iterrows():
    x, y = transformer.transform(row['lon'], row['lat'])
    insc_x.append(x)
    insc_y.append(y)

candi_x, candi_y = [], []
for _, row in candi_df.iterrows():
    x, y = transformer.transform(row['lon'], row['lat'])
    candi_x.append(x)
    candi_y.append(y)

# ============================================================
# 2. Load DEM extent for grid
# ============================================================
with rasterio.open("D:/documents/volcarch-repo/data/processed/dem/jatim_dem.tif") as src:
    dem = src.read(1).astype(float)
    dem[dem == src.nodata] = np.nan
    transform = src.transform
    profile = src.profile.copy()

# Create coarse grid (1 km resolution = every ~33 pixels)
step = 33  # ~1km at 30m resolution
rows_c = np.arange(0, dem.shape[0], step)
cols_c = np.arange(0, dem.shape[1], step)
grid_y, grid_x = np.meshgrid(rows_c, cols_c, indexing='ij')

# Convert to UTM
grid_utm_x = transform.c + grid_x * transform.a
grid_utm_y = transform.f + grid_y * transform.e

print(f"  Grid: {grid_x.shape[0]} x {grid_x.shape[1]} cells (1 km resolution)")

# ============================================================
# 3. Compute inscription density (KDE)
# ============================================================
print(f"\n  Computing inscription density (KDE)...")

if len(insc_x) > 10:
    # KDE with bandwidth = 15 km
    insc_positions = np.vstack([insc_x, insc_y])
    kde_insc = gaussian_kde(insc_positions, bw_method=0.05)  # bandwidth relative to data range

    # Evaluate on grid
    grid_positions = np.vstack([grid_utm_x.ravel(), grid_utm_y.ravel()])
    insc_density = kde_insc(grid_positions).reshape(grid_x.shape)

    # Normalize to 0-1
    insc_density_norm = insc_density / np.max(insc_density)
else:
    insc_density_norm = np.zeros(grid_x.shape)

# ============================================================
# 4. Compute "expected inscription density" from model
# ============================================================
print(f"  Computing expected inscription density...")

# Model: inscriptions concentrate 15-30 km from volcanoes (E084 finding)
# Expected density follows a Gaussian centered at 25 km from nearest volcano

volcanoes = [
    (-7.93, 112.31), (-8.108, 112.922), (-7.732, 112.578),
    (-7.942, 112.95), (-7.60, 112.63), (-7.625, 111.192),
    (-8.125, 114.042),
]

# Compute distance to nearest volcano for each grid cell
min_dist_grid = np.full(grid_x.shape, np.inf)
for vlat, vlon in volcanoes:
    vx, vy = transformer.transform(vlon, vlat)
    dist = np.sqrt((grid_utm_x - vx)**2 + (grid_utm_y - vy)**2) / 1000  # km
    min_dist_grid = np.minimum(min_dist_grid, dist)

# Expected model: Gaussian peak at 25 km, sigma = 10 km
expected_density = np.exp(-0.5 * ((min_dist_grid - 25) / 10)**2)

# Mask ocean/invalid pixels
elev_coarse = dem[::step, ::step][:grid_x.shape[0], :grid_x.shape[1]]
valid_mask = ~np.isnan(elev_coarse)
expected_density[~valid_mask] = np.nan

# ============================================================
# 5. Compute DESERT score = expected - observed
# ============================================================
print(f"  Computing inscription desert score...")

# Desert = high expected density - low observed density
# Normalized to same scale
expected_norm = expected_density / np.nanmax(expected_density)

desert_score = expected_norm - insc_density_norm
desert_score[~valid_mask] = np.nan

# Only count deserts where expected > 0.3 (meaningful expectation)
desert_significant = np.where(expected_norm > 0.3, desert_score, np.nan)

# ============================================================
# 6. Find the biggest deserts
# ============================================================
print(f"\n{'='*70}")
print("TOP INSCRIPTION DESERTS (zones where inscriptions SHOULD be but AREN'T)")
print(f"{'='*70}")

# Find contiguous desert regions
from scipy.ndimage import label
desert_mask = desert_significant > 0.5  # strong desert (expected >> observed)
desert_mask[np.isnan(desert_significant)] = False
labeled_desert, n_deserts = label(desert_mask)

print(f"\n  Number of inscription deserts (score > 0.5): {n_deserts}")

from scipy.ndimage import center_of_mass
inv_transformer = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)

if n_deserts > 0:
    # Get desert properties
    desert_info = []
    for i in range(1, min(n_deserts + 1, 21)):
        mask_i = labeled_desert == i
        if np.sum(mask_i) < 5:  # skip tiny deserts
            continue
        centroid = center_of_mass(mask_i)
        row_c, col_c = int(centroid[0]), int(centroid[1])

        utm_x = grid_utm_x[row_c, col_c]
        utm_y = grid_utm_y[row_c, col_c]
        lon, lat = inv_transformer.transform(utm_x, utm_y)

        area = np.sum(mask_i)  # km2 (1 km grid)
        mean_score = np.nanmean(desert_significant[mask_i])
        mean_dist = np.nanmean(min_dist_grid[mask_i])
        mean_elev = np.nanmean(elev_coarse[mask_i])

        desert_info.append({
            "rank": len(desert_info) + 1,
            "lat": float(lat),
            "lon": float(lon),
            "area_km2": int(area),
            "mean_desert_score": float(mean_score),
            "mean_volcano_dist_km": float(mean_dist),
            "mean_elevation_m": float(mean_elev) if not np.isnan(mean_elev) else None,
        })

    print(f"\n  {'#':<4} {'Lat':>8} {'Lon':>8} {'Area':>6} {'Score':>7} {'V.Dist':>7} {'Elev':>6} Interpretation")
    print(f"  {'-'*75}")

    for d in desert_info[:15]:
        # Interpret location
        if d['lon'] < 111:
            region = "W. E.Java (Lawu zone)"
        elif d['lon'] < 112:
            region = "C. E.Java (transition)"
        elif d['lon'] < 113:
            region = "Malang/Kelud zone"
        else:
            region = "E. E.Java (Semeru/Bromo)"

        elev_str = f"{d['mean_elevation_m']:.0f}m" if d['mean_elevation_m'] else "N/A"
        print(f"  {d['rank']:<4} {d['lat']:>8.3f} {d['lon']:>8.3f} {d['area_km2']:>5} {d['mean_desert_score']:>6.2f} "
              f"{d['mean_volcano_dist_km']:>6.1f} {elev_str:>6} {region}")

# ============================================================
# 7. The interpretation
# ============================================================
print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")

total_desert_area = np.sum(desert_mask)
total_expected_area = np.sum(expected_norm > 0.3)
desert_fraction = total_desert_area / total_expected_area * 100 if total_expected_area > 0 else 0

print(f"""
  Total area where inscriptions are expected (score > 0.3): {total_expected_area} km2
  Total inscription desert area (expected but absent, score > 0.5): {total_desert_area} km2
  Desert fraction: {desert_fraction:.1f}% of expected inscription zone is EMPTY

  These deserts are NOT empty because nobody lived there.
  They are empty because:
  1. The court system that produced inscriptions was concentrated in specific zones (E084)
  2. Volcanic burial concealed inscriptions in proximal zones (E102)
  3. Organic-media writing (palm leaf) in non-court zones left no trace (E113)
  4. Survey effort concentrated on already-known court-zone sites (E069, E129)

  The inscription deserts are the SHADOW of the Two Javas divide.
  Court Java has inscriptions. Volcano Java has deserts.
  The deserts don't mean absence — they mean invisibility.
""")

# ============================================================
# 8. Visualization
# ============================================================
print(f"  Generating inscription desert map...")

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Panel 1: Observed inscription density
ax = axes[0]
im1 = ax.imshow(insc_density_norm, cmap='Blues', vmin=0, vmax=1,
                extent=[grid_utm_x.min()/1000, grid_utm_x.max()/1000,
                        grid_utm_y.min()/1000, grid_utm_y.max()/1000],
                origin='upper', aspect='equal')
ax.set_title('Observed Inscription Density\n(KDE of 174 geocoded inscriptions)', fontweight='bold')
plt.colorbar(im1, ax=ax, shrink=0.6, label='Density (normalized)')

# Panel 2: Expected inscription density
ax = axes[1]
im2 = ax.imshow(expected_norm, cmap='Oranges', vmin=0, vmax=1,
                extent=[grid_utm_x.min()/1000, grid_utm_x.max()/1000,
                        grid_utm_y.min()/1000, grid_utm_y.max()/1000],
                origin='upper', aspect='equal')
ax.set_title('Expected Inscription Density\n(Gaussian model, peak at 25 km from volcano)', fontweight='bold')
plt.colorbar(im2, ax=ax, shrink=0.6, label='Expected density')

# Panel 3: Desert score (expected - observed)
ax = axes[2]
desert_plot = np.ma.masked_where(np.isnan(desert_significant), desert_significant)
im3 = ax.imshow(desert_plot, cmap='RdYlGn_r', vmin=-0.5, vmax=1.0,
                extent=[grid_utm_x.min()/1000, grid_utm_x.max()/1000,
                        grid_utm_y.min()/1000, grid_utm_y.max()/1000],
                origin='upper', aspect='equal')
ax.set_title('Inscription Desert Score\n(Red = expected but absent)', fontweight='bold')
plt.colorbar(im3, ax=ax, shrink=0.6, label='Desert score (expected - observed)')

# Add volcanoes to all panels
for ax in axes:
    for vlat, vlon in volcanoes:
        vx, vy = transformer.transform(vlon, vlat)
        ax.plot(vx/1000, vy/1000, 'k^', markersize=8)
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')

plt.suptitle('E169: Inscription Desert Analysis — Where Are the Missing Inscriptions?',
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path("D:/documents/volcarch-repo/experiments/E169_inscription_desert/results")
fig.savefig(output_path / 'inscription_desert_map.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Map saved: {output_path / 'inscription_desert_map.png'}")

# Save results
results = {
    "n_inscriptions_java": len(insc_java),
    "n_deserts": int(n_deserts),
    "total_desert_area_km2": int(total_desert_area),
    "desert_fraction_pct": float(desert_fraction),
    "top_deserts": desert_info[:15] if desert_info else [],
}

with open(output_path / 'inscription_desert.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nDONE.")
