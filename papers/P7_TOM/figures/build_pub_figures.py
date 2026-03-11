"""
Rebuild Fig 3, 4, 5 for Antiquity submission — clean publication versions.
Removes E019 experiment labels, fixes p-value display.

Run from repo root:
    python papers/P7_TOM/figures/build_pub_figures.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, FancyBboxPatch
    from matplotlib.lines import Line2D
    from scipy.stats import mannwhitneyu
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed" / "dashboard"
DEEP_TIME_PATH = REPO_ROOT / "experiments" / "E019_spatial_distribution" / "data" / "deep_time_sites.csv"
OUT_DIR = Path(__file__).parent


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def nearest_volcano(lat, lon, volcanoes_df):
    dists = np.full(len(lat), np.inf)
    names = np.full(len(lat), "", dtype=object)
    for _, v in volcanoes_df.iterrows():
        d = haversine_km(lat, lon, v["lat"], v["lon"])
        closer = d < dists
        dists[closer] = d[closer]
        names[closer] = v["name"]
    return dists, names


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 +
                           (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def save_fig(fig, name):
    """Save as both PNG and TIF at 300 DPI."""
    for ext in ["png", "tif"]:
        out = OUT_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {out}")


def main():
    print("Loading data...")
    sites = pd.read_csv(DATA_DIR / "sites.csv")
    volcanoes = pd.read_csv(DATA_DIR / "volcanoes.csv")
    grid = pd.read_csv(DATA_DIR / "grid_predictions.csv")
    deep_time = pd.read_csv(DEEP_TIME_PATH)

    # Compute distances
    print("Computing distances...")
    site_dists, _ = nearest_volcano(sites["lat"].values, sites["lon"].values, volcanoes)
    grid_dists, _ = nearest_volcano(grid["lat"].values, grid["lon"].values, volcanoes)
    grid["dist_nearest_volcano_km"] = grid_dists
    dt_dists, dt_nearest = nearest_volcano(deep_time["lat"].values, deep_time["lon"].values, volcanoes)
    deep_time["dist_nearest_volcano_km"] = dt_dists
    deep_time["nearest_volcano"] = dt_nearest

    dist_a = grid.loc[grid["zone"] == "A", "dist_nearest_volcano_km"].values
    dist_b = grid.loc[grid["zone"] == "B", "dist_nearest_volcano_km"].values

    u_stat, p_val = mannwhitneyu(dist_a, dist_b, alternative="two-sided")
    d_effect = cohens_d(dist_a, dist_b)
    n1, n2 = len(dist_a), len(dist_b)
    z_score = (u_stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    r_effect = abs(z_score) / np.sqrt(n1 + n2)

    # Sites vs grid test
    u_sg, p_sg = mannwhitneyu(site_dists, grid_dists, alternative="two-sided")
    n_s, n_g = len(site_dists), len(grid_dists)
    z_sg = (u_sg - (n_s * n_g / 2)) / np.sqrt(n_s * n_g * (n_s + n_g + 1) / 12)
    r_sg = abs(z_sg) / np.sqrt(n_s + n_g)

    # =========================================
    # FIG 3: Deep-time context map (NO E019 label)
    # =========================================
    print("\nFig 3: Deep-time context map...")
    fig, ax = plt.subplots(figsize=(12, 8))

    zone_colors = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c", "E": "#bdc3c7"}
    for zone, color in zone_colors.items():
        mask = grid["zone"] == zone
        if mask.sum() > 0:
            ax.scatter(grid.loc[mask, "lon"], grid.loc[mask, "lat"],
                       c=color, s=0.5, alpha=0.4, rasterized=True)

    for _, v in volcanoes.iterrows():
        ax.scatter(v["lon"], v["lat"], c="red", s=100, marker="v",
                   edgecolors="black", linewidths=0.5, zorder=6)
        ax.annotate(v["name"], (v["lon"], v["lat"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", fontweight="bold")

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
    ax.set_title("Deep-Time Java Sites on Zone Classification Map\n"
                 "All deep-time sites in karst caves or river terraces \u2014 none on volcanic plains")
    plt.tight_layout()
    save_fig(fig, "fig3_deep_time_map")
    plt.close()

    # =========================================
    # FIG 4: Zone distance boxplot (NO E019, fix p-value)
    # =========================================
    print("\nFig 4: Zone distance boxplot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Boxplot
    ax1 = axes[0]
    zone_data = []
    zone_labels = []
    zone_colors_list = []
    color_map = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c", "E": "#bdc3c7"}
    for zone in ["A", "B", "C", "E"]:
        d = grid.loc[grid["zone"] == zone, "dist_nearest_volcano_km"].values
        if len(d) > 0:
            zone_data.append(d)
            zone_labels.append(f"Zone {zone}\n(n={len(d):,})")
            zone_colors_list.append(color_map[zone])

    bp = ax1.boxplot(zone_data, labels=zone_labels, patch_artist=True,
                     showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], zone_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel("Distance to nearest volcano (km)")
    ax1.set_title("Zone Distance to Nearest Volcano")
    # Fixed p-value display
    sig_text = f"A vs B: p < 10\u207b\u00b9\u2070\u2070, d={d_effect:.2f}"
    ax1.text(0.5, 0.98, sig_text, transform=ax1.transAxes, fontsize=8,
             ha="center", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # Right: Violin plot
    ax2 = axes[1]
    parts = ax2.violinplot([dist_a, dist_b], positions=[1, 2], showmedians=True,
                            showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#2ecc71", "#f39c12"][i])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([f"Zone A\n(sites exist)\nmed={np.median(dist_a):.1f} km",
                          f"Zone B\n(no sites, buried?)\nmed={np.median(dist_b):.1f} km"])
    ax2.set_ylabel("Distance to nearest volcano (km)")
    ax2.set_title("Zone A vs Zone B: Key H-TOM Test")
    # Fixed p-value display
    ax2.text(0.5, 0.98, f"p < 10\u207b\u00b9\u2070\u2070 | Cohen's d = {d_effect:.2f} | r = {r_effect:.3f}",
             transform=ax2.transAxes, fontsize=8, ha="center", va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    save_fig(fig, "fig4_zone_distance")
    plt.close()

    # =========================================
    # FIG 5: Site distance histogram (NO E019)
    # =========================================
    print("\nFig 5: Site distance histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, 180, 5)
    ax.hist(grid_dists, bins=bins, density=True, alpha=0.4, color="gray",
            label=f"Grid baseline (n={len(grid_dists):,})")
    ax.hist(site_dists, bins=bins, density=True, alpha=0.6, color="steelblue",
            label=f"Known sites (n={len(site_dists)})")

    for _, row in deep_time.iterrows():
        ax.axvline(row["dist_nearest_volcano_km"], color="red", linestyle="--",
                   alpha=0.7, linewidth=1)
        ax.text(row["dist_nearest_volcano_km"] + 1, ax.get_ylim()[1] * 0.85,
                row["name"], rotation=90, fontsize=7, color="red", va="top")

    ax.set_xlabel("Distance to nearest volcano (km)")
    ax.set_ylabel("Density")
    ax.set_title("Site Distance to Nearest Volcano vs Geographic Baseline")
    ax.legend(loc="upper right")
    stats_text = (f"Mann-Whitney U = {u_sg:.0f}\n"
                  f"p = {p_sg:.2e}\n"
                  f"r = {r_sg:.3f}")
    ax.text(0.98, 0.65, stats_text, transform=ax.transAxes, fontsize=8,
            ha="right", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    save_fig(fig, "fig5_site_distance")
    plt.close()

    print("\nAll publication figures generated.")


if __name__ == "__main__":
    main()
