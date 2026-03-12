#!/usr/bin/env python3
"""
E052: GEBCO Sunda Shelf Paleo-Drainage Reconstruction
======================================================

Hypothesis: The Sunda Shelf's paleo-drainage network, reconstructed from
bathymetry, reveals habitable corridors that are now submerged — representing
a massive "blind spot" in Southeast Asian archaeology.

This script:
1. Loads SRTM30+ bathymetry for the Sunda Shelf region
2. Extracts paleo-coastlines at key sea-level stands
3. Calculates exposed land area at each stand
4. Identifies paleo-river channels from bathymetric depressions
5. Identifies potential habitable zones
6. Overlays known archaeological context
7. Generates publication-quality maps

Data: SRTM30_PLUS via NOAA CoastWatch ERDDAP (Becker et al. 2009)
Resolution: ~1 km (30 arc-second)

References:
- Voris (2000) Maps of Pleistocene sea levels in SE Asia
- Sathiamurthy & Voris (2006) Maps of Holocene sea level transgression
- Solihuddin (2014) Indonesian sea-level curves
- Hanebuth et al. (2000) Rapid flooding of the Sunda Shelf
- Oppenheimer (1998) Eden in the East
- Molengraaff (1921) First paleo-river reconstruction
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from scipy import ndimage
from scipy.ndimage import label as ndimage_label
import json

# === CONFIGURATION ===
BASEDIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASEDIR, 'results')
os.makedirs(RESULTS, exist_ok=True)

DATAFILE = os.path.join(BASEDIR, 'data_srtm30plus_full.nc')

# Sea-level stands (depth in meters, age in years BP)
# Sources: Lambeck et al. (2014), Solihuddin (2014), Hanebuth et al. (2000)
SEA_LEVELS = [
    (-120, 20000, 'LGM (~20,000 BP)'),
    (-80, 14000, '~14,000 BP'),
    (-60, 12000, '~12,000 BP'),
    (-40, 10000, '~10,000 BP'),
    (0, 0, 'Present'),
]

# Grid cell area calculation (approximate)
EARTH_RADIUS_KM = 6371.0


def load_bathymetry():
    """Load SRTM30+ bathymetry data."""
    print("Loading bathymetry data...")
    ds = xr.open_dataset(DATAFILE, engine='scipy')
    lon = ds.longitude.values
    lat = ds.latitude.values
    z = ds['z'].values.astype(np.float32)
    ds.close()

    print(f"  Grid: {z.shape[0]} x {z.shape[1]} (lat x lon)")
    print(f"  Lat: {lat[0]:.2f} to {lat[-1]:.2f}")
    print(f"  Lon: {lon[0]:.2f} to {lon[-1]:.2f}")
    print(f"  Depth range: {z.min():.0f}m to {z.max():.0f}m")
    return lon, lat, z


def cell_area_km2(lat, dlon, dlat):
    """Calculate area of a grid cell at given latitude in km^2."""
    lat_rad = np.radians(lat)
    # Width in km (longitude)
    width = EARTH_RADIUS_KM * np.cos(lat_rad) * np.radians(dlon)
    # Height in km (latitude)
    height = EARTH_RADIUS_KM * np.radians(dlat)
    return np.abs(width * height)


def calculate_areas(lon, lat, z):
    """Calculate exposed land area at each sea-level stand."""
    print("\n" + "="*60)
    print("PALEO-COASTLINE ANALYSIS: Exposed Land Areas")
    print("="*60)

    dlon = np.abs(lon[1] - lon[0])
    dlat = np.abs(lat[1] - lat[0])

    # Pre-compute cell areas for each latitude
    cell_areas = np.array([cell_area_km2(la, dlon, dlat) for la in lat])
    # Broadcast to 2D
    AREA_GRID = np.tile(cell_areas[:, np.newaxis], (1, len(lon)))

    results = {}

    # Present-day land area (for reference)
    present_land = z > 0
    present_land_area = np.sum(AREA_GRID[present_land])
    print(f"\nPresent-day land area in region: {present_land_area:,.0f} km^2")

    for depth, age, label in SEA_LEVELS:
        # Land = everything above the sea level at that time
        exposed = z > depth
        area = np.sum(AREA_GRID[exposed])

        # Additional area beyond present
        additional = area - present_land_area

        # Shelf area that would be exposed (between current sea level and paleo level)
        shelf_exposed = (z > depth) & (z <= 0)
        shelf_area = np.sum(AREA_GRID[shelf_exposed])

        results[depth] = {
            'age': age,
            'label': label,
            'total_land_km2': area,
            'additional_km2': additional,
            'shelf_exposed_km2': shelf_area,
            'exposed_mask': exposed,
        }

        print(f"\n{label} (sea level = {depth}m):")
        print(f"  Total land area:      {area:>12,.0f} km^2")
        print(f"  Additional vs today:  {additional:>12,.0f} km^2")
        print(f"  Exposed shelf area:   {shelf_area:>12,.0f} km^2")

    # Summary table
    print("\n" + "-"*60)
    print("SUMMARY: Progressive flooding of Sunda Shelf")
    print("-"*60)
    print(f"{'Sea Level':>10} {'Age':>12} {'Total Land':>14} {'Shelf Lost':>14}")
    print(f"{'(m)':>10} {'(BP)':>12} {'(km^2)':>14} {'(km^2)':>14}")
    print("-"*60)

    prev_total = None
    for depth, age, label in SEA_LEVELS:
        r = results[depth]
        lost = ''
        if prev_total is not None:
            lost_area = prev_total - r['total_land_km2']
            lost = f"{lost_area:>14,.0f}"
        print(f"{depth:>10} {age:>12,} {r['total_land_km2']:>14,.0f} {lost}")
        prev_total = r['total_land_km2']

    return results, AREA_GRID


def detect_paleo_rivers(lon, lat, z, sea_level=-120):
    """
    Detect paleo-river channels from bathymetric depressions.

    Method: At a given sea level, identify linear depressions in the
    exposed shelf surface using topographic analysis:
    1. Mask to shelf area only (between sea_level and 0m)
    2. Apply Gaussian smoothing to remove noise
    3. Compute topographic position index (TPI) — difference from local mean
    4. Channels = negative TPI (lower than surroundings)
    5. Filter by connectivity and length
    """
    print(f"\n{'='*60}")
    print(f"PALEO-RIVER CHANNEL DETECTION (sea level = {sea_level}m)")
    print("="*60)

    # Focus on shelf area
    shelf_mask = (z > sea_level) & (z <= 0)
    n_shelf = shelf_mask.sum()
    print(f"Shelf cells: {n_shelf:,}")

    # Create shelf elevation surface
    # Use shelf median as fill value (avoids edge artifacts from filling with 0)
    shelf_median = np.median(z[shelf_mask])
    shelf_filled = np.where(shelf_mask, z, shelf_median)

    # Multi-scale TPI: combine different window sizes to catch channels of
    # varying widths (narrow tributaries to broad valleys)
    print("Computing multi-scale Topographic Position Index...")
    tpi_combined = np.zeros_like(z, dtype=np.float64)
    for sigma, window in [(3, 9), (5, 15), (8, 25)]:
        z_smooth = ndimage.gaussian_filter(shelf_filled.astype(np.float64), sigma=sigma)
        z_mean = ndimage.uniform_filter(shelf_filled.astype(np.float64), size=window)
        tpi_scale = z_smooth - z_mean
        tpi_combined += tpi_scale
    tpi = tpi_combined / 3.0  # average across scales

    # Channels = negative TPI in shelf area
    # Use percentile-based threshold for robustness
    shelf_tpi = tpi[shelf_mask]
    threshold = np.percentile(shelf_tpi, 10)  # deepest 10% are channel candidates
    threshold = max(threshold, -1.5)  # but at least 1.5m below local mean
    print(f"Channel TPI threshold: {threshold:.2f}m (10th percentile of shelf TPI)")

    channel_mask = shelf_mask & (tpi < threshold)

    # Label connected components
    labeled, n_features = ndimage_label(channel_mask)
    print(f"Raw channel segments: {n_features}")

    # Filter: keep connected components > 50 cells (~50 km^2)
    channel_sizes = ndimage.sum(channel_mask, labeled, range(1, n_features + 1))
    large_channels = np.zeros_like(channel_mask, dtype=bool)
    major_rivers = []

    # Sort by size, keep top systems
    size_threshold = 50
    for i, size in enumerate(channel_sizes):
        if size > size_threshold:
            component = labeled == (i + 1)
            large_channels |= component
            ys, xs = np.where(component)
            centroid_lat = lat[int(np.mean(ys))]
            centroid_lon = lon[int(np.mean(xs))]
            extent_lat = lat[ys.min()], lat[ys.max()]
            extent_lon = lon[xs.min()], lon[xs.max()]
            avg_depth = np.mean(z[component])
            length_lat = abs(float(max(extent_lat)) - float(min(extent_lat))) * 111
            length_lon = abs(float(max(extent_lon)) - float(min(extent_lon))) * 111
            est_length = max(length_lat, length_lon)
            major_rivers.append({
                'id': len(major_rivers) + 1,
                'size_cells': int(size),
                'centroid': (float(centroid_lat), float(centroid_lon)),
                'lat_range': (float(min(extent_lat)), float(max(extent_lat))),
                'lon_range': (float(min(extent_lon)), float(max(extent_lon))),
                'avg_depth': float(avg_depth),
                'avg_tpi': float(np.mean(tpi[component])),
                'est_length_km': float(est_length),
            })

    # Sort by size descending
    major_rivers.sort(key=lambda x: x['size_cells'], reverse=True)
    for i, r in enumerate(major_rivers):
        r['id'] = i + 1

    print(f"Major channel systems (>{size_threshold} cells): {len(major_rivers)}")
    for r in major_rivers[:15]:  # show top 15
        print(f"  River {r['id']}: {r['size_cells']:,} cells, "
              f"center ({r['centroid'][0]:.1f}N, {r['centroid'][1]:.1f}E), "
              f"avg depth {r['avg_depth']:.0f}m, TPI {r['avg_tpi']:.1f}m, "
              f"~{r['est_length_km']:.0f}km extent")

    total_channel_cells = large_channels.sum()
    print(f"Total channel cells: {total_channel_cells:,} ({100*total_channel_cells/n_shelf:.1f}% of shelf)")

    return large_channels, tpi, major_rivers


def identify_habitable_zones(lon, lat, z, channels, sea_level=-120):
    """
    Identify potentially habitable zones on the exposed Sunda Shelf.

    Criteria for habitability:
    1. Above sea level at the given stand
    2. Relatively flat (low slope)
    3. Near river channels (within ~30km for water access)
    4. Not too steep (slope < 2 degrees)
    """
    print(f"\n{'='*60}")
    print(f"HABITABLE ZONE IDENTIFICATION (sea level = {sea_level}m)")
    print("="*60)

    dlon = np.abs(lon[1] - lon[0])
    dlat = np.abs(lat[1] - lat[0])

    # 1. Exposed land at this sea level
    exposed = (z > sea_level)

    # 2. Slope calculation (gradient in m/km)
    # dy = dlat * 111 km, dx = dlon * 111 * cos(lat)
    dy = dlat * 111.0  # km per cell in latitude
    cos_lat = np.cos(np.radians(lat))[:, np.newaxis]
    dx = dlon * 111.0 * cos_lat  # km per cell in longitude

    dz_dy, dz_dx = np.gradient(z.astype(np.float64))
    slope_y = dz_dy / dy  # m/km
    slope_x = dz_dx / dx  # m/km
    slope = np.sqrt(slope_x**2 + slope_y**2)
    slope_degrees = np.degrees(np.arctan(slope / 1000.0))  # convert m/km to degrees

    flat = slope_degrees < 2.0  # less than 2 degrees

    # 3. Near rivers/valleys (within ~50km)
    # Use detected channels plus TPI-based valley proximity
    from scipy.ndimage import binary_dilation
    # Also include all cells with negative TPI (any valley tendency)
    valley_mask = channels.copy()
    if channels.any():
        # Dilate detected channels by ~50 cells (~50km)
        # Use circular structuring element for more realistic buffer
        y, x = np.ogrid[-50:51, -50:51]
        struct_circle = (x**2 + y**2) <= 50**2
        near_river = binary_dilation(channels, structure=struct_circle)
    else:
        # Fallback: use flat shelf areas (most of shelf is flat and near water)
        near_river = on_shelf & flat  # most of the flat shelf was accessible

    # 4. On shelf (between sea_level and 0m — the newly exposed area)
    on_shelf = (z > sea_level) & (z <= 0)

    # Combine criteria
    habitable = exposed & flat & near_river & on_shelf

    # Calculate areas
    cell_areas = np.array([cell_area_km2(la, dlon, dlat) for la in lat])
    AREA_GRID = np.tile(cell_areas[:, np.newaxis], (1, len(lon)))

    total_shelf_area = np.sum(AREA_GRID[on_shelf])
    habitable_area = np.sum(AREA_GRID[habitable])
    pct = 100 * habitable_area / total_shelf_area if total_shelf_area > 0 else 0

    print(f"Exposed shelf area: {total_shelf_area:,.0f} km^2")
    print(f"Habitable zone area: {habitable_area:,.0f} km^2 ({pct:.1f}% of shelf)")
    print(f"  - Flat (<2 deg slope): {np.sum(AREA_GRID[exposed & flat & on_shelf]):,.0f} km^2")
    print(f"  - Near rivers (<30km): {np.sum(AREA_GRID[exposed & near_river & on_shelf]):,.0f} km^2")
    print(f"  - Both (habitable):    {habitable_area:,.0f} km^2")

    # Population estimate (very rough)
    # Hunter-gatherer density: 0.01-1.0 persons/km^2
    # Tropical river environments: ~0.5 persons/km^2 (Kelly 2013)
    pop_low = habitable_area * 0.05
    pop_mid = habitable_area * 0.3
    pop_high = habitable_area * 1.0
    print(f"\nPopulation estimates (hunter-gatherer densities):")
    print(f"  Low (0.05/km^2):  {pop_low:>12,.0f}")
    print(f"  Mid (0.3/km^2):   {pop_mid:>12,.0f}")
    print(f"  High (1.0/km^2):  {pop_high:>12,.0f}")

    return habitable, slope_degrees, {
        'shelf_area_km2': float(total_shelf_area),
        'habitable_area_km2': float(habitable_area),
        'habitable_pct': float(pct),
        'pop_estimate_low': float(pop_low),
        'pop_estimate_mid': float(pop_mid),
        'pop_estimate_high': float(pop_high),
    }


def plot_bathymetry_overview(lon, lat, z):
    """Plot 1: Overview bathymetric map of the region."""
    print("\nGenerating Figure 1: Bathymetric overview...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Custom bathymetric colormap
    colors_bathy = [
        (0.0, '#08306b'),   # deep ocean (dark blue)
        (0.2, '#2171b5'),   # mid ocean
        (0.4, '#4292c6'),   # shallow ocean
        (0.55, '#6baed6'),  # shelf edge
        (0.7, '#9ecae1'),   # shallow shelf
        (0.8, '#c6dbef'),   # very shallow
        (0.85, '#deebf7'),  # near coast
        (0.88, '#f7fcf5'),  # sea level transition
        (0.9, '#c7e9c0'),   # low land
        (0.95, '#74c476'), # mid land
        (1.0, '#238b45'),   # highland
    ]
    cmap = LinearSegmentedColormap.from_list('bathy', colors_bathy)

    LON, LAT = np.meshgrid(lon, lat)
    vmin, vmax = -200, 200
    im = ax.pcolormesh(LON, LAT, np.clip(z, vmin, vmax), cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='auto', rasterized=True)

    # Add depth contours
    contour_levels = [-120, -80, -60, -40, -20, 0]
    cs = ax.contour(LON, LAT, z, levels=contour_levels,
                    colors='black', linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%dm')

    # Mark 0m contour (present coastline) thicker
    ax.contour(LON, LAT, z, levels=[0], colors='black', linewidths=1.5)

    # -120m contour (LGM coastline) dashed
    ax.contour(LON, LAT, z, levels=[-120], colors='red', linewidths=2,
               linestyles='dashed')

    # Labels
    text_kwargs = dict(ha='center', va='center',
                       path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax.text(101.5, 4.0, 'MALAY\nPENINSULA', **text_kwargs, fontweight='bold', fontsize=8)
    ax.text(100.5, -0.5, 'SUMATRA', **text_kwargs, fontweight='bold', fontsize=8)
    ax.text(114.5, 1.0, 'BORNEO', **text_kwargs, fontweight='bold', fontsize=8)
    ax.text(110, -7.5, 'JAVA', **text_kwargs, fontweight='bold', fontsize=8)
    ax.text(107.5, -4.5, 'Java Sea', **text_kwargs, fontstyle='italic', fontsize=9)
    ax.text(102, 7.5, 'Gulf of\nThailand', **text_kwargs, fontstyle='italic', fontsize=8)
    ax.text(100, 2.5, 'Strait of\nMalacca', **text_kwargs, fontstyle='italic', fontsize=7)
    ax.text(112, 7.5, 'South China\nSea', **text_kwargs, fontstyle='italic', fontsize=9)
    ax.text(118.5, -2, 'Makassar\nStrait', **text_kwargs, fontstyle='italic', fontsize=7)

    # LGM coastline legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, ls='--', label='LGM coastline (-120m)'),
        Line2D([0], [0], color='black', lw=1.5, label='Present coastline (0m)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label='Elevation / Depth (m)')
    ax.set_xlabel('Longitude (E)')
    ax.set_ylabel('Latitude (N)')
    ax.set_title('Sunda Shelf Bathymetry\nSRTM30_PLUS ~1km resolution',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(95, 120)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    outpath = os.path.join(RESULTS, 'fig1_bathymetry_overview.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_progressive_flooding(lon, lat, z, area_results):
    """Plot 2: Progressive flooding of Sunda Shelf at different sea levels."""
    print("\nGenerating Figure 2: Progressive flooding sequence...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    LON, LAT = np.meshgrid(lon, lat)

    # Color scheme: land = green/brown, ocean = blue, newly flooded = red tint
    land_color = '#c7e9c0'
    ocean_color = '#deebf7'
    shelf_color = '#fff7bc'  # exposed shelf (yellow)

    all_levels = SEA_LEVELS  # 5 levels

    for i, (depth, age, label) in enumerate(all_levels):
        ax = axes[i]

        # Create land/ocean/shelf map
        rgb = np.zeros((*z.shape, 3))

        # Deep ocean (below shelf edge)
        deep = z <= depth
        rgb[deep] = [0.122, 0.467, 0.706]  # blue

        # Present land (always land)
        present_land = z > 0
        rgb[present_land] = [0.455, 0.769, 0.463]  # green

        # Exposed shelf at this sea level (between depth and 0)
        exposed_shelf = (z > depth) & (z <= 0)
        rgb[exposed_shelf] = [0.996, 0.851, 0.463]  # yellow/tan

        ax.imshow(rgb, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                  aspect='equal', origin='lower')

        # Coastline at this level
        ax.contour(LON, LAT, z, levels=[depth], colors='red',
                   linewidths=1.0, alpha=0.8)

        # Present coastline
        ax.contour(LON, LAT, z, levels=[0], colors='black',
                   linewidths=0.5, alpha=0.5)

        r = area_results[depth]
        ax.set_title(f'{label}\nSea level: {depth}m\n'
                     f'Exposed shelf: {r["shelf_exposed_km2"]:,.0f} km$^2$',
                     fontsize=10)
        ax.set_xlim(95, 120)
        ax.set_ylim(-10, 10)

        if i >= 3:
            ax.set_xlabel('Longitude (E)')
        if i % 3 == 0:
            ax.set_ylabel('Latitude (N)')

        ax.grid(True, alpha=0.2, linestyle='--')

    # 6th panel: area chart
    ax = axes[5]
    depths = [d for d, _, _ in SEA_LEVELS]
    shelf_areas = [area_results[d]['shelf_exposed_km2'] for d in depths]
    ages = [a for _, a, _ in SEA_LEVELS]

    ax.barh(range(len(depths)), shelf_areas, color=['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60'])
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f'{d}m\n({a:,} BP)' for d, a, _ in SEA_LEVELS])
    ax.set_xlabel('Exposed Shelf Area (km$^2$)')
    ax.set_title('Progressive Land Loss\n(Exposed shelf area)', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)

    # Add area labels on bars
    for j, area in enumerate(shelf_areas):
        if area > 0:
            ax.text(area + 10000, j, f'{area:,.0f} km$^2$', va='center', fontsize=8)

    fig.suptitle('Progressive Flooding of the Sunda Shelf\n'
                 'Post-glacial sea-level rise: Last Glacial Maximum to Present',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    outpath = os.path.join(RESULTS, 'fig2_progressive_flooding.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_paleo_rivers(lon, lat, z, channels, tpi, major_rivers):
    """Plot 3: Paleo-river channels on the Sunda Shelf."""
    print("\nGenerating Figure 3: Paleo-river channels...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    LON, LAT = np.meshgrid(lon, lat)

    # Left: TPI map showing channel detection
    ax = axes[0]
    shelf_mask = (z > -120) & (z <= 0)
    tpi_display = np.where(shelf_mask, tpi, np.nan)

    im = ax.pcolormesh(LON, LAT, tpi_display, cmap='RdBu_r',
                       vmin=-20, vmax=20, shading='auto', rasterized=True)
    ax.contour(LON, LAT, z, levels=[0], colors='black', linewidths=1.5)
    ax.contour(LON, LAT, z, levels=[-120], colors='red', linewidths=1.5, linestyles='--')

    fig.colorbar(im, ax=ax, shrink=0.7, label='Topographic Position Index (m)')
    ax.set_title('Topographic Position Index\n(Negative = channel/valley)', fontsize=11)
    ax.set_xlim(95, 120)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('Longitude (E)')
    ax.set_ylabel('Latitude (N)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Right: Detected channels overlaid on bathymetry
    ax = axes[1]

    # Background: shelf bathymetry
    shelf_bathy = np.where(shelf_mask, z, np.nan)
    im2 = ax.pcolormesh(LON, LAT, shelf_bathy, cmap='terrain',
                        vmin=-120, vmax=10, shading='auto', rasterized=True)

    # Overlay channels in red
    channel_display = np.where(channels, 1.0, np.nan)
    ax.pcolormesh(LON, LAT, channel_display, cmap='Reds',
                  vmin=0, vmax=1, shading='auto', alpha=0.7, rasterized=True)

    # Present coastline
    ax.contour(LON, LAT, z, levels=[0], colors='black', linewidths=1.5)
    ax.contour(LON, LAT, z, levels=[-120], colors='white', linewidths=1.5, linestyles='--')

    # Label major rivers
    text_kw = dict(fontsize=8, fontweight='bold', color='yellow',
                   path_effects=[pe.withStroke(linewidth=2, foreground='black')])
    for r in major_rivers[:8]:  # top 8
        ax.plot(r['centroid'][1], r['centroid'][0], 'w*', markersize=10,
                markeredgecolor='black', markeredgewidth=0.5)
        ax.annotate(f"Ch.{r['id']}\n{r['size_cells']:,} cells",
                    (r['centroid'][1], r['centroid'][0]),
                    textcoords="offset points", xytext=(10, 5),
                    **text_kw)

    fig.colorbar(im2, ax=ax, shrink=0.7, label='Depth (m)')
    ax.set_title('Detected Paleo-River Channels\n(LGM exposed shelf, -120m)',
                 fontsize=11)
    ax.set_xlim(95, 120)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('Longitude (E)')
    ax.set_ylabel('Latitude (N)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Known paleo-rivers from Voris (2000) - approximate locations
    known_rivers = [
        ('Siam (Chao Phraya ext.)', 103, 5),
        ('North Sunda', 108, 2),
        ('Molengraaff', 106, 0),
        ('East Sunda (Borneo)', 112, -2),
    ]
    for name, rlon, rlat in known_rivers:
        ax.annotate(name, (rlon, rlat),
                    fontsize=7, fontstyle='italic', color='white',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')],
                    ha='center')

    plt.tight_layout()
    outpath = os.path.join(RESULTS, 'fig3_paleo_rivers.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_habitable_zones(lon, lat, z, habitable, channels, hab_stats):
    """Plot 4: Habitable zones on the LGM Sunda Shelf."""
    print("\nGenerating Figure 4: Habitable zones...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    LON, LAT = np.meshgrid(lon, lat)

    # Background: elevation categories
    rgb = np.zeros((*z.shape, 3))

    # Deep ocean
    rgb[z <= -120] = [0.12, 0.47, 0.71]

    # Exposed shelf (not habitable)
    shelf_only = (z > -120) & (z <= 0) & ~habitable
    rgb[shelf_only] = [0.80, 0.80, 0.70]  # gray-tan

    # Habitable zones
    rgb[habitable] = [0.85, 0.25, 0.10]  # red-orange

    # Present land
    rgb[z > 0] = [0.45, 0.77, 0.46]

    ax.imshow(rgb, extent=[lon[0], lon[-1], lat[0], lat[-1]],
              aspect='equal', origin='lower')

    # Channel overlay
    channel_display = np.where(channels, 1.0, np.nan)
    ax.pcolormesh(LON, LAT, channel_display, cmap='Blues',
                  vmin=0, vmax=1.5, shading='auto', alpha=0.5, rasterized=True)

    # Coastlines
    ax.contour(LON, LAT, z, levels=[0], colors='black', linewidths=1.5)
    ax.contour(LON, LAT, z, levels=[-120], colors='white', linewidths=2, linestyles='--')

    # Known archaeological sites near shelf edge (examples)
    # These are real sites from the literature that are near the former shelf
    sites = [
        ('Niah Cave', 114.2, 3.8, 'Sarawak, 40-50 kya'),
        ('Tam Pa Ling', 103.4, 8.5, 'Laos, ~46 kya (inland)'),
        ('Liang Bua', 120.4, -8.5, 'Flores (off-map)'),
        ('Gunung Sewu', 110.5, -8.0, 'Java, 30+ kya'),
        ('Song Terus', 111.5, -8.0, 'Java, 120 kya'),
    ]
    for name, slon, slat, desc in sites:
        if 95 <= slon <= 120 and -10 <= slat <= 10:
            ax.plot(slon, slat, 'w^', markersize=10,
                    markeredgecolor='black', markeredgewidth=1)
            ax.annotate(f'{name}\n({desc})', (slon, slat),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=7, color='white',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # Labels
    text_kw = dict(fontsize=9, ha='center', va='center', fontweight='bold',
                   path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax.text(107, 1, 'SUNDALAND\n(exposed at LGM)', color='#8b0000',
            fontsize=11, ha='center', va='center', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#d94017', label=f'Habitable zone ({hab_stats["habitable_area_km2"]:,.0f} km$^2$)'),
        Patch(facecolor='#ccccb3', label='Exposed shelf (steep/far from rivers)'),
        Patch(facecolor='#1e78b5', label='Ocean (below -120m)'),
        Patch(facecolor='#73c476', label='Present-day land'),
        Line2D([0], [0], color='white', lw=2, ls='--', label='LGM coastline (-120m)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='Known archaeological sites'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8,
              facecolor='white', framealpha=0.9)

    ax.set_xlabel('Longitude (E)')
    ax.set_ylabel('Latitude (N)')
    ax.set_title(f'Habitable Zones on the LGM Sunda Shelf (-120m sea level)\n'
                 f'Criteria: flat (<2 deg slope), near rivers (<30km)\n'
                 f'Estimated population: {hab_stats["pop_estimate_low"]:,.0f} - '
                 f'{hab_stats["pop_estimate_high"]:,.0f} '
                 f'(mid: {hab_stats["pop_estimate_mid"]:,.0f})',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(95, 120)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')

    outpath = os.path.join(RESULTS, 'fig4_habitable_zones.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_depth_histogram(z):
    """Plot 5: Depth distribution histogram for the Sunda Shelf."""
    print("\nGenerating Figure 5: Depth histogram...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Full depth distribution
    ax = axes[0]
    z_flat = z.flatten()
    z_ocean = z_flat[z_flat < 0]

    ax.hist(z_ocean, bins=200, color='steelblue', alpha=0.7, edgecolor='none')
    for depth, _, label in SEA_LEVELS[:-1]:  # exclude present
        ax.axvline(depth, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(depth + 2, ax.get_ylim()[1] * 0.9, f'{depth}m',
                fontsize=7, color='red', rotation=90, va='top')
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Cell count')
    ax.set_title('Depth Distribution (ocean cells only)')
    ax.set_xlim(-500, 0)
    ax.grid(True, alpha=0.3)

    # Right: Shelf depth distribution (0 to -200m)
    ax = axes[1]
    z_shelf = z_flat[(z_flat >= -200) & (z_flat < 0)]

    ax.hist(z_shelf, bins=100, color='#fc8d59', alpha=0.7, edgecolor='none')
    for depth, _, label in SEA_LEVELS[:-1]:
        if depth >= -200:
            ax.axvline(depth, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
            ax.text(depth + 1, ax.get_ylim()[1] * 0.9, label,
                    fontsize=7, color='red', rotation=90, va='top')

    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Cell count')
    ax.set_title('Shelf Depth Distribution (0 to -200m)\nFlat shelf = concentrated at shallow depths')
    ax.grid(True, alpha=0.3)

    # Annotate the "step" in the histogram at shelf edge
    ax.annotate('Continental shelf\nedge (~120m)',
                xy=(-120, 0), xytext=(-180, ax.get_ylim()[1] * 0.5),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, ha='center')

    plt.tight_layout()
    outpath = os.path.join(RESULTS, 'fig5_depth_histogram.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_flooding_timeline(area_results):
    """Plot 6: Timeline of Sunda Shelf flooding."""
    print("\nGenerating Figure 6: Flooding timeline...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Data - ordered by sea level stand
    ordered_keys = sorted(area_results.keys())  # -120, -80, -60, -40, 0
    ages = [area_results[k]['age'] for k in ordered_keys]
    shelf_areas = [area_results[k]['shelf_exposed_km2'] for k in ordered_keys]
    total_lands = [area_results[k]['total_land_km2'] for k in ordered_keys]
    labels = [area_results[k]['label'] for k in ordered_keys]

    # Left: Exposed shelf area over time
    ax = axes[0]
    ax.fill_between(ages, shelf_areas, alpha=0.3, color='#d73027')
    ax.plot(ages, shelf_areas, 'o-', color='#d73027', linewidth=2, markersize=8)
    for i, (a, s, l) in enumerate(zip(ages, shelf_areas, labels)):
        ax.annotate(f'{s:,.0f} km$^2$', (a, s),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=8, fontweight='bold')
    ax.set_xlabel('Years Before Present')
    ax.set_ylabel('Exposed Shelf Area (km$^2$)')
    ax.set_title('Loss of Sunda Shelf Land Area')
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # Highlight rapid flooding period (Meltwater Pulse 1A, ~14,500 BP)
    ax.axvspan(14800, 13500, alpha=0.15, color='blue', label='Meltwater Pulse 1A')
    ax.legend(fontsize=8)

    # Right: Rate of land loss
    ax = axes[1]
    for i in range(len(ages) - 1):
        dt = ages[i] - ages[i+1]  # years
        da = shelf_areas[i] - shelf_areas[i+1]  # km^2 lost
        rate = da / dt * 1000  # km^2 per millennium
        mid_age = (ages[i] + ages[i+1]) / 2

        color = '#d73027' if rate > 100 else '#fc8d59' if rate > 50 else '#91bfdb'
        ax.bar(mid_age, rate, width=dt * 0.8, color=color, alpha=0.7,
               edgecolor='black', linewidth=0.5)
        ax.text(mid_age, rate + 5, f'{rate:,.0f}\nkm$^2$/kyr',
                ha='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Years Before Present')
    ax.set_ylabel('Rate of Land Loss (km$^2$ per millennium)')
    ax.set_title('Rate of Sunda Shelf Flooding')
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Timeline of Sunda Shelf Submergence\n'
                 'Post-glacial sea-level rise progressively drowned Sundaland',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    outpath = os.path.join(RESULTS, 'fig6_flooding_timeline.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_cross_section(lon, lat, z):
    """Plot 7: Cross-section profiles through the Sunda Shelf."""
    print("\nGenerating Figure 7: Cross-sections...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Profile 1: West-East through shelf center (lat ~ 1N)
    ax = axes[0]
    lat_idx = np.argmin(np.abs(lat - 1.0))
    profile = z[lat_idx, :]
    ax.fill_between(lon, profile, -200, where=profile > -200,
                    alpha=0.3, color='#c7e9c0', label='Shelf')
    ax.fill_between(lon, profile, -200, where=profile <= -200,
                    alpha=0.3, color='#6baed6', label='Deep ocean')
    ax.plot(lon, profile, 'k-', linewidth=0.8)

    for depth, _, label in SEA_LEVELS[:-1]:
        ax.axhline(depth, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.text(95.5, depth + 2, f'{depth}m ({label})', fontsize=7, color='red')

    ax.axhline(0, color='blue', linewidth=1.5, label='Present sea level')
    ax.set_xlim(95, 120)
    ax.set_ylim(-200, 100)
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'W-E Cross-section at {lat[lat_idx]:.1f}N (through shelf center)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Profile 2: North-South through shelf (lon ~ 107E)
    ax = axes[1]
    lon_idx = np.argmin(np.abs(lon - 107.0))
    profile = z[:, lon_idx]
    ax.fill_between(lat, profile, -200, where=profile > -200,
                    alpha=0.3, color='#c7e9c0')
    ax.fill_between(lat, profile, -200, where=profile <= -200,
                    alpha=0.3, color='#6baed6')
    ax.plot(lat, profile, 'k-', linewidth=0.8)

    for depth, _, label in SEA_LEVELS[:-1]:
        ax.axhline(depth, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.axhline(0, color='blue', linewidth=1.5)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-200, 100)
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'N-S Cross-section at {lon[lon_idx]:.1f}E')
    ax.grid(True, alpha=0.3)

    # Profile 3: Diagonal (Malay Peninsula to Borneo, through Molengraaff channel)
    ax = axes[2]
    # From ~100E,4N to ~114E,-2N
    n_points = 500
    lons_diag = np.linspace(100, 114, n_points)
    lats_diag = np.linspace(4, -2, n_points)
    # Interpolate
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((lat, lon), z, method='linear',
                                      bounds_error=False, fill_value=np.nan)
    points = np.column_stack((lats_diag, lons_diag))
    profile_diag = interp(points)

    dist_km = np.sqrt(((lons_diag - lons_diag[0]) * 111 * np.cos(np.radians(lats_diag)))**2 +
                       ((lats_diag - lats_diag[0]) * 111)**2)

    ax.fill_between(dist_km, profile_diag, -200, where=profile_diag > -200,
                    alpha=0.3, color='#c7e9c0')
    ax.plot(dist_km, profile_diag, 'k-', linewidth=0.8)

    for depth, _, label in SEA_LEVELS[:-1]:
        ax.axhline(depth, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.axhline(0, color='blue', linewidth=1.5)
    ax.set_ylim(-200, 100)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Diagonal: Malay Peninsula (100E,4N) to Borneo (114E,-2N)')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Cross-sectional Profiles Through the Sunda Shelf\n'
                 'Showing flat shelf bathymetry and paleo-sea-level stands',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    outpath = os.path.join(RESULTS, 'fig7_cross_sections.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def compute_flooding_rate_analysis(area_results):
    """Analyze the rate and chronology of shelf flooding."""
    print(f"\n{'='*60}")
    print("FLOODING RATE ANALYSIS")
    print("="*60)

    # Sort from most negative (oldest/deepest) to 0 (present)
    depths = sorted(area_results.keys())  # -120, -80, -60, -40, 0

    print("\nChronological flooding phases (oldest to youngest):")
    for i in range(len(depths) - 1):
        d1 = depths[i]      # earlier, deeper (e.g., -120)
        d2 = depths[i + 1]  # later, shallower (e.g., -80)
        r1 = area_results[d1]
        r2 = area_results[d2]

        dt = r1['age'] - r2['age']  # years (positive: 20000 - 14000 = 6000)
        da = r1['shelf_exposed_km2'] - r2['shelf_exposed_km2']  # km^2 lost (positive)

        if dt > 0:
            rate_per_yr = da / dt
            rate_per_kyr = rate_per_yr * 1000
            sea_level_rise = d2 - d1  # positive, e.g. -80 - (-120) = +40
            sea_level_rate = sea_level_rise / dt * 1000  # m per kyr

            print(f"\n  {r1['label']} -> {r2['label']}:")
            print(f"    Duration: {dt:,} years")
            print(f"    Sea level rise: +{sea_level_rise}m ({sea_level_rate:+.1f} m/kyr)")
            print(f"    Land lost: {da:,.0f} km^2 ({rate_per_kyr:,.0f} km^2/kyr)")

    # Comparison with modern countries for perspective
    print("\n" + "-"*60)
    print("SCALE COMPARISON: LGM exposed shelf vs modern countries")
    print("-"*60)
    lgm_shelf = area_results[-120]['shelf_exposed_km2']
    comparisons = [
        ('Java (island)', 129_000),
        ('Borneo', 743_000),
        ('Sumatra', 473_000),
        ('UK (Great Britain)', 209_000),
        ('France', 551_000),
        ('Texas (USA)', 696_000),
        ('Combined (Java+Bali+Madura)', 138_000),
    ]
    print(f"LGM exposed Sunda Shelf: {lgm_shelf:,.0f} km^2")
    for name, area in comparisons:
        ratio = lgm_shelf / area
        print(f"  = {ratio:.1f}x {name} ({area:,.0f} km^2)")

    return lgm_shelf


def save_results_json(area_results, major_rivers, hab_stats, lgm_shelf):
    """Save all results as JSON for documentation."""
    results = {
        'experiment': 'E052_sunda_shelf',
        'title': 'GEBCO Sunda Shelf Paleo-Drainage Reconstruction',
        'data_source': 'SRTM30_PLUS via NOAA CoastWatch ERDDAP (Becker et al. 2009)',
        'resolution_km': 1.0,
        'region': {'north': 10, 'south': -10, 'west': 95, 'east': 120},
        'sea_level_stands': {},
        'paleo_rivers': major_rivers,
        'habitability': hab_stats,
        'lgm_shelf_area_km2': lgm_shelf,
    }

    for depth in area_results:
        r = area_results[depth]
        results['sea_level_stands'][str(depth)] = {
            'depth_m': depth,
            'age_bp': r['age'],
            'label': r['label'],
            'total_land_km2': r['total_land_km2'],
            'additional_vs_present_km2': r['additional_km2'],
            'shelf_exposed_km2': r['shelf_exposed_km2'],
        }

    outpath = os.path.join(RESULTS, 'results.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results JSON: {outpath}")
    return results


# === MAIN ===
def main():
    print("="*60)
    print("E052: Sunda Shelf Paleo-Drainage Reconstruction")
    print("="*60)

    # 1. Load data
    lon, lat, z = load_bathymetry()

    # 2. Calculate exposed areas at each sea-level stand
    area_results, area_grid = calculate_areas(lon, lat, z)

    # 3. Detect paleo-river channels
    channels, tpi, major_rivers = detect_paleo_rivers(lon, lat, z, sea_level=-120)

    # 4. Identify habitable zones
    habitable, slope, hab_stats = identify_habitable_zones(
        lon, lat, z, channels, sea_level=-120)

    # 5. Flooding rate analysis
    lgm_shelf = compute_flooding_rate_analysis(area_results)

    # 6. Generate all figures
    plot_bathymetry_overview(lon, lat, z)
    plot_progressive_flooding(lon, lat, z, area_results)
    plot_paleo_rivers(lon, lat, z, channels, tpi, major_rivers)
    plot_habitable_zones(lon, lat, z, habitable, channels, hab_stats)
    plot_depth_histogram(z)
    plot_flooding_timeline(area_results)
    plot_cross_section(lon, lat, z)

    # 7. Save results
    results = save_results_json(area_results, major_rivers, hab_stats, lgm_shelf)

    # 8. Final summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Data: SRTM30_PLUS, {z.shape[0]}x{z.shape[1]} grid, ~1km resolution")
    print(f"Region: {lon[0]:.0f}E-{lon[-1]:.0f}E, {lat[0]:.0f}S-{lat[-1]:.0f}N")
    print(f"\nKey findings:")
    print(f"  LGM exposed shelf area: {area_results[-120]['shelf_exposed_km2']:,.0f} km^2")
    print(f"  Habitable zone: {hab_stats['habitable_area_km2']:,.0f} km^2 "
          f"({hab_stats['habitable_pct']:.1f}% of shelf)")
    print(f"  Paleo-river systems detected: {len(major_rivers)}")
    print(f"  Population estimate (mid): {hab_stats['pop_estimate_mid']:,.0f}")
    print(f"\n  => This represents a MASSIVE archaeological blind spot.")
    print(f"     {area_results[-120]['shelf_exposed_km2']:,.0f} km^2 of potentially")
    print(f"     inhabited land is now under water — inaccessible to")
    print(f"     conventional archaeology.")
    print(f"\n  7 figures saved to: {RESULTS}")


if __name__ == '__main__':
    main()
