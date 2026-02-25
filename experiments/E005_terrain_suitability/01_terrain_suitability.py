"""
E005 / Step 1: Terrain Suitability Model as null hypothesis for H1.

Builds a simple terrain suitability index from DEM derivatives, then tests whether
near-volcano zones show a NEGATIVE residual (observed < predicted), which would
support H1 (taphonomic burial bias) even when raw site counts are high near volcanoes.

Requires:
  - data/processed/dem/malang_slope.tif    (from E003)
  - data/processed/dem/malang_twi.tif      (from E003)
  - data/processed/dem/malang_dem.tif      (from E003)
  - data/processed/east_java_sites.geojson (from E001)

For full East Java analysis, DEM should cover all of Jawa Timur, not just Malang Raya.
The Malang Raya DEM gives a PILOT result; full analysis needs full-province DEM.

Run from repo root:
    python experiments/E005_terrain_suitability/01_terrain_suitability.py

Output:
  results/suitability_map.tif          -- rasterized suitability index
  results/grid_analysis.csv            -- 25km grid: observed, predicted, residual
  results/residual_vs_distance.csv     -- per-cell residual vs volcano distance
  results/h1_test_results.txt          -- Spearman test on residuals
  results/map_residuals.html           -- Folium map of residuals
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    from scipy.stats import spearmanr
    from shapely.geometry import Point, box
    import matplotlib.pyplot as plt
    import folium
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"
SITES_PATH = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
RESULTS_DIR = Path(__file__).parent / "results"

# Volcano coordinates (same as E004)
VOLCANOES = {
    "Kelud":           {"lat": -7.9300, "lon": 112.3080},
    "Semeru":          {"lat": -8.1080, "lon": 112.9220},
    "Arjuno-Welirang": {"lat": -7.7290, "lon": 112.5750},
    "Bromo":           {"lat": -7.9420, "lon": 112.9500},
    "Lamongan":        {"lat": -7.9770, "lon": 113.3430},
    "Raung":           {"lat": -8.1250, "lon": 114.0420},
}

GRID_SIZE_M = 25000  # 25 km grid cells for analysis
TARGET_CRS = "EPSG:32749"  # UTM 49S

SUITABILITY_WEIGHTS = {
    "slope":     0.40,
    "elevation": 0.30,
    "twi":       0.20,
    "river":     0.10,  # placeholder: distance to river (fallback: uniform 0.5)
}


def load_raster_as_array(path: Path) -> tuple[np.ndarray, dict]:
    """Load a single-band raster, return (array, profile)."""
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}\nRun E003 first.")
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata if src.nodata is not None else -9999
        arr[arr == nodata] = np.nan
        return arr, src.profile


def normalize_score(arr: np.ndarray, low_good: bool = True,
                    clip_min: float = None, clip_max: float = None) -> np.ndarray:
    """Normalize array to 0–1, handling NaNs. low_good=True means smaller = better."""
    out = arr.copy()
    if clip_min is not None:
        out = np.clip(out, clip_min, None)
    if clip_max is not None:
        out = np.clip(out, None, clip_max)
    vmin = np.nanmin(out)
    vmax = np.nanmax(out)
    if vmax == vmin:
        return np.where(np.isnan(out), np.nan, 0.5)
    score = (out - vmin) / (vmax - vmin)
    if low_good:
        score = 1.0 - score
    return score


def compute_slope_score(slope_arr: np.ndarray) -> np.ndarray:
    """
    Suitability from slope (degrees).
    0–10°: ideal (1.0)
    10–25°: transitional (linear falloff)
    >25°: unsuitable (0.0)
    """
    score = np.where(slope_arr <= 10, 1.0,
            np.where(slope_arr <= 25, 1.0 - (slope_arr - 10) / 15.0, 0.0))
    score[np.isnan(slope_arr)] = np.nan
    return score


def compute_elevation_score(dem_arr: np.ndarray) -> np.ndarray:
    """
    Suitability from elevation (meters).
    50–800m: ideal (1.0)
    Below 50m or above 800m: ramp to 0.0
    Archaeological sites in Java are concentrated in mid-elevation river valleys.
    """
    score = np.where(
        (dem_arr >= 50) & (dem_arr <= 800), 1.0,
        np.where(
            dem_arr < 50, np.clip(dem_arr / 50.0, 0, 1),
            np.where(dem_arr <= 1500, 1.0 - (dem_arr - 800) / 700.0, 0.0)
        )
    )
    score[np.isnan(dem_arr)] = np.nan
    return np.clip(score, 0, 1)


def compute_twi_score(twi_arr: np.ndarray) -> np.ndarray:
    """
    Suitability from TWI (Topographic Wetness Index).
    Higher TWI = wetter, more suitable for agriculture/settlement.
    Normalize to 0–1 (higher = better).
    """
    return normalize_score(twi_arr, low_good=False, clip_min=0, clip_max=20)


def build_suitability_index(dem_arr, slope_arr, twi_arr, profile) -> np.ndarray:
    """Weighted sum of terrain suitability scores."""
    s_slope = compute_slope_score(slope_arr)
    s_elev  = compute_elevation_score(dem_arr)
    s_twi   = compute_twi_score(twi_arr)

    # River proximity: no river raster yet → use TWI as proxy (high TWI ≈ near water)
    # This is a simplification; replace with actual river distance raster when available
    s_river = s_twi.copy()  # proxy

    suitability = (
        SUITABILITY_WEIGHTS["slope"]     * s_slope +
        SUITABILITY_WEIGHTS["elevation"] * s_elev  +
        SUITABILITY_WEIGHTS["twi"]       * s_twi   +
        SUITABILITY_WEIGHTS["river"]     * s_river
    )

    # Propagate NaNs
    nan_mask = np.isnan(s_slope) | np.isnan(s_elev) | np.isnan(s_twi)
    suitability[nan_mask] = np.nan
    return suitability


def save_suitability_raster(suit_arr: np.ndarray, profile: dict, path: Path) -> None:
    p = profile.copy()
    p.update({"count": 1, "dtype": "float32", "nodata": -9999, "compress": "lzw"})
    suit_out = suit_arr.astype("float32")
    suit_out[np.isnan(suit_arr)] = -9999
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(suit_out, 1)
    print(f"  Suitability raster saved: {path}")


def build_analysis_grid(profile: dict, suit_arr: np.ndarray,
                         sites: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create a 25km × 25km analysis grid over the DEM extent.
    For each cell: compute mean suitability and count known sites.
    """
    from rasterio.transform import AffineTransformer
    from shapely.geometry import box as shp_box

    transform = profile["transform"]
    crs = profile["crs"]
    height, width = suit_arr.shape
    res = abs(transform.a)  # pixel size in meters

    # Grid cell size in pixels
    cell_px = int(GRID_SIZE_M / res)
    if cell_px < 1:
        cell_px = 1

    sites_proj = sites.to_crs(crs)

    cells = []
    for row_start in range(0, height, cell_px):
        for col_start in range(0, width, cell_px):
            row_end = min(row_start + cell_px, height)
            col_end = min(col_start + cell_px, width)

            # Cell extent in projected coords
            x_min, y_max = rasterio.transform.xy(transform, row_start, col_start, offset="ul")
            x_max, y_min = rasterio.transform.xy(transform, row_end, col_end, offset="lr")
            cell_geom = shp_box(x_min, y_min, x_max, y_max)

            # Mean suitability in cell
            cell_suit = suit_arr[row_start:row_end, col_start:col_end]
            valid = cell_suit[~np.isnan(cell_suit)]
            if len(valid) == 0:
                continue
            mean_suit = float(np.mean(valid))

            # Count sites in cell
            sites_in = sites_proj[sites_proj.geometry.within(cell_geom)]
            site_count = len(sites_in)

            # Cell centroid
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

            cells.append({
                "geometry": cell_geom,
                "cx": cx,
                "cy": cy,
                "mean_suitability": mean_suit,
                "site_count": site_count,
                "cell_area_km2": (cell_geom.area / 1e6),
            })

    gdf = gpd.GeoDataFrame(cells, crs=crs)
    return gdf


def compute_predicted_density(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Predict site count per cell assuming density is proportional to suitability.
    Calibration: scale predicted total to match observed total.
    """
    total_observed = grid["site_count"].sum()
    total_suit_weight = grid["mean_suitability"].sum()

    if total_suit_weight == 0:
        grid["predicted_count"] = 0.0
    else:
        grid["predicted_count"] = (
            grid["mean_suitability"] / total_suit_weight * total_observed
        )

    grid["residual"] = grid["site_count"] - grid["predicted_count"]
    grid["residual_density"] = grid["residual"] / grid["cell_area_km2"] * 1000
    return grid


def compute_grid_volcano_distance(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute distance from each cell centroid to nearest volcano."""
    volc_rows = []
    for name, info in VOLCANOES.items():
        volc_rows.append({"name": name, "geometry": Point(info["lon"], info["lat"])})
    volcs_gdf = gpd.GeoDataFrame(volc_rows, crs="EPSG:4326").to_crs(grid.crs)

    centroids = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(grid["cx"], grid["cy"])},
        crs=grid.crs
    )

    min_dists = []
    nearest = []
    for _, row in centroids.iterrows():
        dists = volcs_gdf.geometry.distance(row.geometry) / 1000
        min_dists.append(float(dists.min()))
        nearest.append(volcs_gdf.iloc[dists.idxmin()]["name"])

    grid["dist_to_volcano_km"] = min_dists
    grid["nearest_volcano"] = nearest
    return grid


def run_h1_residual_test(grid: gpd.GeoDataFrame) -> dict:
    """
    Spearman correlation: residual density vs distance to volcano.
    H1 predicts: more negative residuals near volcanoes → positive rho.
    (cells near volcanoes have fewer sites than terrain predicts)
    """
    # Only cells with non-zero suitability
    valid = grid[grid["mean_suitability"] > 0.05].copy()

    rho, pval = spearmanr(valid["dist_to_volcano_km"], valid["residual_density"])

    h1_supported = rho > 0.3 and pval < 0.05

    return {
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "n_cells": len(valid),
        "h1_supported": h1_supported,
        "interpretation": (
            f"rho = {rho:.3f}, p = {pval:.4f}. "
            + ("H1 SUPPORTED: Near-volcano zones have fewer sites than terrain predicts."
               if h1_supported
               else "H1 NOT SUPPORTED: No deficit of sites near volcanoes after controlling for terrain.")
        ),
    }


def plot_residual_map(grid: gpd.GeoDataFrame, output_path: Path) -> None:
    """Interactive Folium map of residuals."""
    grid_wgs = grid.to_crs("EPSG:4326")
    center_lat = float(grid_wgs.geometry.centroid.y.mean())
    center_lon = float(grid_wgs.geometry.centroid.x.mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8,
                   tiles="CartoDB positron")

    vmin = float(grid["residual_density"].quantile(0.05))
    vmax = float(grid["residual_density"].quantile(0.95))

    for _, row in grid_wgs.iterrows():
        val = row["residual_density"]
        # Red = negative residual (fewer than expected = potential burial zone)
        # Blue = positive residual (more than expected = dense discovery area)
        norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
        norm = max(0, min(1, norm))
        r = int(255 * (1 - norm))
        b = int(255 * norm)
        color = f"#{r:02x}00{b:02x}"

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda f, c=color: {
                "fillColor": c, "color": "#444", "weight": 0.3, "fillOpacity": 0.6
            },
            tooltip=(
                f"Suitability: {row['mean_suitability']:.2f} | "
                f"Observed: {row['site_count']} | "
                f"Predicted: {row['predicted_count']:.1f} | "
                f"Residual: {row['residual']:.1f} | "
                f"Dist volcano: {row['dist_to_volcano_km']:.0f} km"
            ),
        ).add_to(m)

    # Add volcano markers
    for name, info in VOLCANOES.items():
        folium.Marker(
            [info["lat"], info["lon"]],
            tooltip=f"Volcano: {name}",
            icon=folium.Icon(color="red", icon="fire", prefix="fa"),
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px;border:2px solid #ccc;font-size:11px;">
    <b>H1 Test: Residual Site Density</b><br>
    <span style="color:#ff0000">&#9632;</span> Fewer sites than terrain predicts (burial?)<br>
    <span style="color:#0000ff">&#9632;</span> More sites than terrain predicts (survey bias?)<br>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"  Residual map saved: {output_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("E005: Terrain Suitability Model — H1 Proper Test")
    print("=" * 60)

    # Load DEM derivatives
    print("\nLoading DEM derivatives (from E003)...")
    try:
        dem_arr, profile = load_raster_as_array(DEM_DIR / "malang_dem.tif")
        slope_arr, _     = load_raster_as_array(DEM_DIR / "malang_slope.tif")
        twi_arr, _       = load_raster_as_array(DEM_DIR / "malang_twi.tif")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  DEM shape: {dem_arr.shape}")
    print(f"  Coverage: Malang Raya pilot study area")
    print(f"  NOTE: Full East Java analysis requires province-wide DEM (future work)")

    # Build suitability index
    print("\nBuilding terrain suitability index...")
    suit_arr = build_suitability_index(dem_arr, slope_arr, twi_arr, profile)
    valid_suit = suit_arr[~np.isnan(suit_arr)]
    print(f"  Suitability range: {valid_suit.min():.3f} – {valid_suit.max():.3f}")
    print(f"  Mean suitability: {valid_suit.mean():.3f}")

    save_suitability_raster(suit_arr, profile, RESULTS_DIR / "suitability_map.tif")

    # Load sites
    print("\nLoading site dataset (from E001)...")
    sites = gpd.read_file(SITES_PATH)
    sites = sites[sites.geometry.notna() & ~sites.geometry.is_empty]
    sites = sites.to_crs(profile["crs"])

    # Filter to DEM extent
    with rasterio.open(DEM_DIR / "malang_dem.tif") as src:
        bounds = src.bounds
    from shapely.geometry import box as shp_box
    dem_extent = shp_box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    dem_extent_gdf = gpd.GeoDataFrame({"geometry": [dem_extent]}, crs=profile["crs"])
    sites_in_dem = gpd.sjoin(sites, dem_extent_gdf, how="inner", predicate="within")
    print(f"  Sites in Malang Raya DEM extent: {len(sites_in_dem)}")

    if len(sites_in_dem) < 5:
        print("  WARNING: Very few sites in Malang Raya extent.")
        print("  This is expected — pilot study. Full East Java DEM needed for meaningful test.")

    # Build analysis grid
    print(f"\nBuilding {GRID_SIZE_M/1000:.0f}km x {GRID_SIZE_M/1000:.0f}km analysis grid...")
    grid = build_analysis_grid(profile, suit_arr, sites_in_dem)
    print(f"  Grid cells: {len(grid)}")
    print(f"  Cells with sites: {(grid['site_count'] > 0).sum()}")

    # Predicted density
    grid = compute_predicted_density(grid)

    # Volcano distances
    print("\nComputing distances to volcanoes...")
    grid = compute_grid_volcano_distance(grid)

    # Save grid
    grid_csv = grid.drop(columns="geometry")
    grid_csv.to_csv(RESULTS_DIR / "grid_analysis.csv", index=False)
    print(f"  Grid saved: {RESULTS_DIR / 'grid_analysis.csv'}")

    # H1 test
    print("\nRunning H1 residual test (Spearman correlation)...")
    stats = run_h1_residual_test(grid)
    print(f"  {stats['interpretation']}")
    print(f"  H1 {'SUPPORTED' if stats['h1_supported'] else 'NOT SUPPORTED'} at p < 0.05, rho > 0.3")

    # Write stats
    stats_text = f"""
E005 - H1 Residual Test (Terrain Suitability-Controlled)
=========================================================
Date: 2026-02-23
Study area: Malang Raya (PILOT - limited extent)
Grid size: {GRID_SIZE_M/1000:.0f}km x {GRID_SIZE_M/1000:.0f}km
Total grid cells analyzed: {stats['n_cells']}
Sites in study area: {len(sites_in_dem)}

Method: Spearman correlation between:
  - residual site density (observed - terrain-predicted)
  - distance to nearest active volcano

Spearman rho: {stats['spearman_rho']:.4f}
p-value: {stats['p_value']:.6f}

{stats['interpretation']}

H1 status: {'SUPPORTED' if stats['h1_supported'] else 'NOT SUPPORTED / INCONCLUSIVE'}

IMPORTANT CAVEATS:
1. Study area limited to Malang Raya (pilot). Full East Java analysis needed.
2. Suitability model is simplified (slope + elevation + TWI proxy).
   No actual river distance data, no soil quality, no aspect.
3. River proximity uses TWI as proxy - may inflate suitability in upland areas.
4. Site data is sparse in this area (major sites are concentrated in Brantas valley
   just outside the Malang Raya DEM extent).
5. TWI computation uses simplified contributing area proxy, not true flow accumulation.
""".strip()

    (RESULTS_DIR / "h1_test_results.txt").write_text(stats_text, encoding="utf-8")
    print(f"\n  Full results: {RESULTS_DIR / 'h1_test_results.txt'}")

    # Map
    if len(grid) > 0:
        print("\nGenerating residual map...")
        plot_residual_map(grid, RESULTS_DIR / "map_residuals.html")

    print("\n" + "=" * 60)
    print("E005 complete.")
    print("NOTE: This is a PILOT result (Malang Raya only).")
    print("For Paper 1, re-run with full East Java DEM.")
    print("=" * 60)


if __name__ == "__main__":
    main()
