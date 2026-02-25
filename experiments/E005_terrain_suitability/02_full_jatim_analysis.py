"""
E005 / Step 2: Full Jawa Timur H1 terrain suitability test.

This is the production-level test using the full province DEM.
Requires:
  - data/processed/dem/jatim_dem.tif    (from E003 step 2)
  - data/processed/dem/jatim_slope.tif
  - data/processed/dem/jatim_twi.tif
  - data/processed/east_java_sites.geojson (from E001)

Run from repo root:
    python experiments/E005_terrain_suitability/02_full_jatim_analysis.py

Output:
  results/jatim_suitability.tif
  results/jatim_grid_analysis.csv
  results/jatim_h1_test.txt
  results/jatim_residual_map.html
  results/jatim_density_chart.png
"""

import sys
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

# Reuse E005 step 1 functions
step1_spec = importlib.util.spec_from_file_location(
    "step1", Path(__file__).parent / "01_terrain_suitability.py"
)
step1 = importlib.util.module_from_spec(step1_spec)
step1_spec.loader.exec_module(step1)

try:
    import geopandas as gpd
    import rasterio
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"
SITES_PATH = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
RESULTS_DIR = Path(__file__).parent / "results"

GRID_SIZE_M = 25000   # 25km grid
BIN_EDGES_KM = [0, 25, 50, 75, 100, 150, 200, 10000]
BIN_LABELS   = ["0-25", "25-50", "50-75", "75-100", "100-150", "150-200", "200+"]


def plot_summary(grid: gpd.GeoDataFrame, stats: dict, output_path: Path) -> None:
    """Two-panel figure: (left) observed vs predicted density by distance band,
       (right) scatter plot of residuals vs distance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "E005: Terrain Suitability-Controlled H1 Test — Jawa Timur\n"
        f"Spearman rho (residual vs dist) = {stats['spearman_rho']:.3f}, "
        f"p = {stats['p_value']:.4f}, n = {stats['n_cells']} grid cells",
        fontsize=12
    )

    # Panel 1: observed and predicted density by distance bin
    grid["dist_bin"] = pd.cut(
        grid["dist_to_volcano_km"],
        bins=BIN_EDGES_KM,
        labels=BIN_LABELS,
        right=False
    )
    bin_stats = grid.groupby("dist_bin", observed=True).agg(
        obs_total=("site_count", "sum"),
        pred_total=("predicted_count", "sum"),
        area_total=("cell_area_km2", "sum"),
    ).reset_index()
    bin_stats["obs_density"] = bin_stats["obs_total"] / bin_stats["area_total"] * 1000
    bin_stats["pred_density"] = bin_stats["pred_total"] / bin_stats["area_total"] * 1000

    x = range(len(bin_stats))
    w = 0.35
    ax1.bar([i - w/2 for i in x], bin_stats["obs_density"],  width=w, label="Observed", color="#2196F3", alpha=0.85)
    ax1.bar([i + w/2 for i in x], bin_stats["pred_density"], width=w, label="Terrain-predicted", color="#FF9800", alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{b} km" for b in bin_stats["dist_bin"]], rotation=30, ha="right")
    ax1.set_ylabel("Sites per 1,000 km²")
    ax1.set_xlabel("Distance from nearest active volcano")
    ax1.set_title("Observed vs Terrain-Predicted Site Density")
    ax1.legend()

    # Panel 2: residual density vs distance (scatter)
    valid = grid[grid["mean_suitability"] > 0.05]
    sc = ax2.scatter(
        valid["dist_to_volcano_km"],
        valid["residual_density"],
        c=valid["mean_suitability"],
        cmap="YlOrRd",
        alpha=0.7,
        edgecolors="none",
        s=40,
    )
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Distance to nearest volcano (km)")
    ax2.set_ylabel("Residual site density\n(observed - predicted, per 1000km²)")
    ax2.set_title("Residuals vs Volcanic Proximity\n(negative = fewer sites than expected)")
    plt.colorbar(sc, ax=ax2, label="Terrain suitability")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary chart saved: {output_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("E005 Step 2: Full Jawa Timur H1 Test")
    print("=" * 60)

    # Load DEM
    dem_path   = DEM_DIR / "jatim_dem.tif"
    slope_path = DEM_DIR / "jatim_slope.tif"
    twi_path   = DEM_DIR / "jatim_twi.tif"

    for p in [dem_path, slope_path, twi_path]:
        if not p.exists():
            print(f"ERROR: Missing {p.name}")
            print("Run E003 step 2 first: py experiments/E003_dem_acquisition/02_download_full_jatim_dem.py")
            sys.exit(1)

    print("\nLoading DEM derivatives...")
    dem_arr, profile = step1.load_raster_as_array(dem_path)
    slope_arr, _     = step1.load_raster_as_array(slope_path)
    twi_arr, _       = step1.load_raster_as_array(twi_path)
    print(f"  DEM shape: {dem_arr.shape}")

    # Suitability index
    print("\nBuilding suitability index...")
    suit_arr = step1.build_suitability_index(dem_arr, slope_arr, twi_arr, profile)
    valid = suit_arr[~np.isnan(suit_arr)]
    print(f"  Mean suitability: {valid.mean():.3f}")

    step1.save_suitability_raster(suit_arr, profile, RESULTS_DIR / "jatim_suitability.tif")

    # Load sites
    print("\nLoading sites...")
    sites = gpd.read_file(SITES_PATH)
    sites = sites[sites.geometry.notna() & ~sites.geometry.is_empty]
    sites = sites.to_crs(profile["crs"])
    print(f"  Total sites: {len(sites)}")

    # Build grid
    print(f"\nBuilding {GRID_SIZE_M/1000:.0f}km grid over Jawa Timur...")
    grid = step1.build_analysis_grid(profile, suit_arr, sites)
    print(f"  Grid cells: {len(grid)}")
    print(f"  Cells with sites: {(grid['site_count'] > 0).sum()}")

    # Predicted density + residuals
    grid = step1.compute_predicted_density(grid)

    # Volcano distances
    print("\nComputing volcano distances...")
    grid = step1.compute_grid_volcano_distance(grid)

    # Save grid
    grid.drop(columns="geometry").to_csv(RESULTS_DIR / "jatim_grid_analysis.csv", index=False)

    # H1 test
    print("\nRunning H1 residual test...")
    stats = step1.run_h1_residual_test(grid)
    print(f"  {stats['interpretation']}")

    # Write results
    results_text = f"""
E005 Step 2 — Full Jawa Timur H1 Test
======================================
Date: 2026-02-23
DEM: Copernicus GLO-30 (30m), full Jawa Timur
Grid size: {GRID_SIZE_M/1000:.0f}km x {GRID_SIZE_M/1000:.0f}km
Total grid cells analyzed: {stats['n_cells']}
Total sites: {len(sites)}

Spearman rho (residual density vs distance to volcano): {stats['spearman_rho']:.4f}
p-value: {stats['p_value']:.6f}

{stats['interpretation']}

H1 (Taphonomic Bias): {'SUPPORTED' if stats['h1_supported'] else 'NOT SUPPORTED / INCONCLUSIVE'}

Suitability model weights:
  Slope:     {step1.SUITABILITY_WEIGHTS['slope']}
  Elevation: {step1.SUITABILITY_WEIGHTS['elevation']}
  TWI:       {step1.SUITABILITY_WEIGHTS['twi']}
  River prox:{step1.SUITABILITY_WEIGHTS['river']}  (TWI proxy)

Notes:
- River proximity uses TWI as proxy (future: replace with actual river distance raster)
- Null model assumes site density proportional to terrain suitability (simplistic)
- Discovery bias (survey coverage) not controlled for
- Results should be interpreted alongside E004 findings
""".strip()

    (RESULTS_DIR / "jatim_h1_test.txt").write_text(results_text, encoding="utf-8")
    print(f"  Results: {RESULTS_DIR / 'jatim_h1_test.txt'}")

    # Plots
    print("\nGenerating charts...")
    plot_summary(grid, stats, RESULTS_DIR / "jatim_density_chart.png")

    print("\nGenerating residual map...")
    step1.plot_residual_map(grid, RESULTS_DIR / "jatim_residual_map.html")

    print("\n" + "=" * 60)
    print("E005 Step 2 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
