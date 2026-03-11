"""
E019: Site-Volcano Distance Analysis + Deep-Time Context Map.

Analysis 1: Compute haversine distance from each of 378 sites to nearest volcano.
             Compare with baseline grid distance distribution.
Analysis 3: Map 4 deep-time Java sites on zone overlay with volcano positions.

Run from repo root:
    python experiments/E019_spatial_distribution/01_site_volcano_distance.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed" / "dashboard"
DEEP_TIME_PATH = Path(__file__).parent / "data" / "deep_time_sites.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km. Accepts scalars or arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def nearest_volcano(lat, lon, volcanoes_df):
    """Return (distance_km, volcano_name) for each point."""
    dists = np.full(len(lat), np.inf)
    names = np.full(len(lat), "", dtype=object)
    for _, v in volcanoes_df.iterrows():
        d = haversine_km(lat, lon, v["lat"], v["lon"])
        closer = d < dists
        dists[closer] = d[closer]
        names[closer] = v["name"]
    return dists, names


def main():
    print("=" * 60)
    print("E019 Analysis 1: Site-Volcano Distance")
    print("=" * 60)

    # Load data
    sites = pd.read_csv(DATA_DIR / "sites.csv")
    volcanoes = pd.read_csv(DATA_DIR / "volcanoes.csv")
    grid = pd.read_csv(DATA_DIR / "grid_predictions.csv")
    deep_time = pd.read_csv(DEEP_TIME_PATH)

    print(f"  Sites: {len(sites)}")
    print(f"  Volcanoes: {len(volcanoes)}")
    print(f"  Grid cells: {len(grid)}")
    print(f"  Deep-time sites: {len(deep_time)}")

    # --- Analysis 1: Site distances ---
    print("\n[1/3] Computing site distances to nearest volcano...")
    site_dists, site_nearest = nearest_volcano(
        sites["lat"].values, sites["lon"].values, volcanoes
    )
    sites["dist_nearest_volcano_km"] = site_dists
    sites["nearest_volcano"] = site_nearest

    # Grid baseline distances (sample for efficiency if huge)
    grid_dists, _ = nearest_volcano(
        grid["lat"].values, grid["lon"].values, volcanoes
    )
    grid["dist_nearest_volcano_km"] = grid_dists

    # Stats
    print(f"\n  Site distances (km):")
    print(f"    Mean:   {site_dists.mean():.1f}")
    print(f"    Median: {np.median(site_dists):.1f}")
    print(f"    Std:    {site_dists.std():.1f}")
    print(f"    Min:    {site_dists.min():.1f}")
    print(f"    Max:    {site_dists.max():.1f}")

    print(f"\n  Grid baseline distances (km):")
    print(f"    Mean:   {grid_dists.mean():.1f}")
    print(f"    Median: {np.median(grid_dists):.1f}")
    print(f"    Std:    {grid_dists.std():.1f}")

    # Mann-Whitney U test: sites vs grid
    from scipy.stats import mannwhitneyu
    u_stat, p_val = mannwhitneyu(site_dists, grid_dists, alternative="two-sided")
    n1, n2 = len(site_dists), len(grid_dists)
    # Effect size r = Z / sqrt(N)
    z_score = (u_stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    r_effect = abs(z_score) / np.sqrt(n1 + n2)

    print(f"\n  Mann-Whitney U (sites vs grid baseline):")
    print(f"    U = {u_stat:.0f}, p = {p_val:.2e}")
    print(f"    Z = {z_score:.2f}, r = {r_effect:.3f}")
    if p_val < 0.05:
        direction = "farther" if np.median(site_dists) > np.median(grid_dists) else "closer"
        print(f"    Sites are significantly {direction} from volcanoes than geographic chance.")
    else:
        print(f"    No significant difference from geographic chance.")

    # Save site distances
    sites.to_csv(RESULTS_DIR / "site_volcano_distances.csv", index=False)
    print(f"\n  Saved: {RESULTS_DIR / 'site_volcano_distances.csv'}")

    # --- Deep-time site distances ---
    print("\n[2/3] Deep-time site distances...")
    dt_dists, dt_nearest = nearest_volcano(
        deep_time["lat"].values, deep_time["lon"].values, volcanoes
    )
    deep_time["dist_nearest_volcano_km"] = dt_dists
    deep_time["nearest_volcano"] = dt_nearest
    for _, row in deep_time.iterrows():
        print(f"    {row['name']}: {row['dist_nearest_volcano_km']:.1f} km "
              f"to {row['nearest_volcano']} ({row['context']})")

    # --- Figure 1: Distance histogram ---
    print("\n[3/3] Generating figures...")
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, 160, 5)
    ax.hist(grid_dists, bins=bins, density=True, alpha=0.4, color="gray",
            label=f"Grid baseline (n={len(grid_dists):,})")
    ax.hist(site_dists, bins=bins, density=True, alpha=0.6, color="steelblue",
            label=f"Known sites (n={len(site_dists)})")

    # Mark deep-time sites
    for _, row in deep_time.iterrows():
        ax.axvline(row["dist_nearest_volcano_km"], color="red", linestyle="--",
                   alpha=0.7, linewidth=1)
        ax.text(row["dist_nearest_volcano_km"] + 1, ax.get_ylim()[1] * 0.85,
                row["name"], rotation=90, fontsize=7, color="red", va="top")

    ax.set_xlabel("Distance to nearest volcano (km)")
    ax.set_ylabel("Density")
    ax.set_title("E019: Site Distance to Nearest Volcano vs Geographic Baseline")
    ax.legend(loc="upper right")
    stats_text = (f"Mann-Whitney U = {u_stat:.0f}\n"
                  f"p = {p_val:.2e}\n"
                  f"r = {r_effect:.3f}")
    ax.text(0.98, 0.65, stats_text, transform=ax.transAxes, fontsize=8,
            ha="right", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_distance_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'fig_distance_histogram.png'}")

    # --- Figure 3: Deep-time context map ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot grid cells colored by zone
    zone_colors = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c", "E": "#bdc3c7"}
    for zone, color in zone_colors.items():
        mask = grid["zone"] == zone
        if mask.sum() > 0:
            ax.scatter(grid.loc[mask, "lon"], grid.loc[mask, "lat"],
                       c=color, s=0.5, alpha=0.4, rasterized=True)

    # Volcanoes
    for _, v in volcanoes.iterrows():
        ax.scatter(v["lon"], v["lat"], c="red", s=100, marker="v",
                   edgecolors="black", linewidths=0.5, zorder=6)
        ax.annotate(v["name"], (v["lon"], v["lat"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", fontweight="bold")

    # Deep-time sites
    context_markers = {"cave": "s", "river_terrace": "D", "river_erosion": "D"}
    context_colors = {"cave": "#1a5276", "river_terrace": "#7d3c98", "river_erosion": "#7d3c98"}
    for _, row in deep_time.iterrows():
        marker = context_markers.get(row["context"], "o")
        color = context_colors.get(row["context"], "blue")
        ax.scatter(row["lon"], row["lat"], c=color, s=120, marker=marker,
                   edgecolors="white", linewidths=1.5, zorder=7)
        age_label = f"{row['age_bp']/1000:.0f} ka" if row["age_bp"] >= 1000 else f"{row['age_bp']} BP"
        ax.annotate(f"{row['name']}\n({age_label}, {row['context'].replace('_', ' ')})",
                    (row["lon"], row["lat"]), fontsize=7,
                    xytext=(8, -12), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9),
                    zorder=8)

    # Legend
    legend_handles = [
        Patch(facecolor="#2ecc71", alpha=0.6, label="Zone A (high suit., shallow burial)"),
        Patch(facecolor="#f39c12", alpha=0.6, label="Zone B (high suit., mod. burial, NO sites)"),
        Patch(facecolor="#e74c3c", alpha=0.6, label="Zone C (high suit., deep burial)"),
        Patch(facecolor="#bdc3c7", alpha=0.6, label="Zone E (low suitability)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="red",
               markersize=10, label="Volcano"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1a5276",
               markersize=10, label="Deep-time site (cave)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#7d3c98",
               markersize=10, label="Deep-time site (river)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=7, framealpha=0.9)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("E019: Deep-Time Java Sites on Zone Classification Map\n"
                 "All deep-time sites in karst caves or river terraces — none on volcanic plains")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_deep_time_context_map.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'fig_deep_time_context_map.png'}")

    print("\nAnalysis 1 + 3 COMPLETE.")


if __name__ == "__main__":
    main()
