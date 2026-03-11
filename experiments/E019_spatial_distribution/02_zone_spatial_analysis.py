"""
E019: Zone Spatial Analysis (Analysis 2).

Key test: Are Zone B cells (high suitability, moderate burial, zero sites)
significantly closer to volcanoes than Zone A cells (where sites exist)?

If H-TOM is correct, Zone B clusters near the volcanic axis because
tephra burial increases with proximity — sites are buried, not absent.

Run from repo root:
    python experiments/E019_spatial_distribution/02_zone_spatial_analysis.py
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
    from scipy.stats import mannwhitneyu
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed" / "dashboard"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km. Accepts arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def nearest_volcano_dist(lat, lon, volcanoes_df):
    """Return distance (km) to nearest volcano for each point."""
    dists = np.full(len(lat), np.inf)
    for _, v in volcanoes_df.iterrows():
        d = haversine_km(lat, lon, v["lat"], v["lon"])
        dists = np.minimum(dists, d)
    return dists


def cohens_d(x, y):
    """Cohen's d effect size."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 +
                           (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def main():
    print("=" * 60)
    print("E019 Analysis 2: Zone Spatial Clustering")
    print("=" * 60)

    # Load data
    grid = pd.read_csv(DATA_DIR / "grid_predictions.csv")
    volcanoes = pd.read_csv(DATA_DIR / "volcanoes.csv")

    print(f"  Grid cells: {len(grid)}")
    print(f"  Volcanoes: {len(volcanoes)}")

    # Compute distance to nearest volcano for all grid cells
    print("\n[1/3] Computing distances for all grid cells...")
    grid["dist_nearest_volcano_km"] = nearest_volcano_dist(
        grid["lat"].values, grid["lon"].values, volcanoes
    )

    # Zone-level statistics
    print("\n[2/3] Zone distance statistics...")
    zone_stats = []
    for zone in ["A", "B", "C", "E"]:
        mask = grid["zone"] == zone
        d = grid.loc[mask, "dist_nearest_volcano_km"]
        stats = {
            "zone": zone,
            "n_cells": mask.sum(),
            "mean_dist_km": d.mean(),
            "median_dist_km": d.median(),
            "std_dist_km": d.std(),
            "min_dist_km": d.min(),
            "max_dist_km": d.max(),
            "p25_dist_km": d.quantile(0.25),
            "p75_dist_km": d.quantile(0.75),
        }
        zone_stats.append(stats)
        print(f"\n  Zone {zone} (n={stats['n_cells']:,}):")
        print(f"    Mean:   {stats['mean_dist_km']:.1f} km")
        print(f"    Median: {stats['median_dist_km']:.1f} km")
        print(f"    Std:    {stats['std_dist_km']:.1f} km")
        print(f"    IQR:    [{stats['p25_dist_km']:.1f}, {stats['p75_dist_km']:.1f}] km")

    zone_stats_df = pd.DataFrame(zone_stats)
    zone_stats_df.to_csv(RESULTS_DIR / "zone_distance_summary.csv", index=False)

    # --- KEY TEST: Zone A vs Zone B ---
    print("\n" + "=" * 60)
    print("KEY TEST: Zone A vs Zone B distance to nearest volcano")
    print("=" * 60)

    dist_a = grid.loc[grid["zone"] == "A", "dist_nearest_volcano_km"].values
    dist_b = grid.loc[grid["zone"] == "B", "dist_nearest_volcano_km"].values

    u_stat, p_val = mannwhitneyu(dist_a, dist_b, alternative="two-sided")
    d_effect = cohens_d(dist_a, dist_b)

    # r effect size for Mann-Whitney
    n1, n2 = len(dist_a), len(dist_b)
    z_score = (u_stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    r_effect = abs(z_score) / np.sqrt(n1 + n2)

    print(f"\n  Zone A: mean={np.mean(dist_a):.1f} km, median={np.median(dist_a):.1f} km (n={n1:,})")
    print(f"  Zone B: mean={np.mean(dist_b):.1f} km, median={np.median(dist_b):.1f} km (n={n2:,})")
    print(f"\n  Mann-Whitney U = {u_stat:.0f}")
    print(f"  p-value = {p_val:.2e}")
    print(f"  Z = {z_score:.2f}")
    print(f"  r (effect size) = {r_effect:.3f}")
    print(f"  Cohen's d = {d_effect:.3f}")

    if p_val < 0.05:
        if np.median(dist_b) < np.median(dist_a):
            verdict = "SUPPORTS H-TOM: Zone B is significantly CLOSER to volcanoes than Zone A."
        else:
            verdict = "COUNTER: Zone B is significantly FARTHER from volcanoes than Zone A."
    else:
        verdict = "NEUTRAL: No significant difference between Zone A and Zone B distances."
    print(f"\n  VERDICT: {verdict}")

    # Also test Zone A vs Zone E
    dist_e = grid.loc[grid["zone"] == "E", "dist_nearest_volcano_km"].values
    u_ae, p_ae = mannwhitneyu(dist_a, dist_e, alternative="two-sided")
    d_ae = cohens_d(dist_a, dist_e)
    print(f"\n  Supplementary: Zone A vs Zone E")
    print(f"    Zone E: mean={np.mean(dist_e):.1f} km, median={np.median(dist_e):.1f} km (n={len(dist_e):,})")
    print(f"    Mann-Whitney p = {p_ae:.2e}, Cohen's d = {d_ae:.3f}")

    # Zone B vs Zone E
    if len(dist_b) > 0 and len(dist_e) > 0:
        u_be, p_be = mannwhitneyu(dist_b, dist_e, alternative="two-sided")
        d_be = cohens_d(dist_b, dist_e)
        print(f"\n  Supplementary: Zone B vs Zone E")
        print(f"    Mann-Whitney p = {p_be:.2e}, Cohen's d = {d_be:.3f}")
    else:
        p_be, d_be = np.nan, np.nan

    # --- Figure: Zone distance boxplot ---
    print("\n[3/3] Generating figures...")
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
    ax1.set_title("E019: Zone Distance to Nearest Volcano")
    # Annotate significance
    if p_val < 0.05:
        sig_text = f"A vs B: p={p_val:.2e}, d={d_effect:.2f}"
    else:
        sig_text = f"A vs B: p={p_val:.2e} (n.s.), d={d_effect:.2f}"
    ax1.text(0.5, 0.98, sig_text, transform=ax1.transAxes, fontsize=8,
             ha="center", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # Right: Violin plot for Zone A vs B detail
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
    ax2.text(0.5, 0.98, f"p = {p_val:.2e} | Cohen's d = {d_effect:.2f} | r = {r_effect:.3f}",
             transform=ax2.transAxes, fontsize=8, ha="center", va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_zone_distance_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'fig_zone_distance_boxplot.png'}")

    # --- Spatial summary report ---
    summary = f"""E019: Spatial Distribution Test — Summary Report
{'=' * 55}
Date: 2026-03-05

ANALYSIS 2: Zone Spatial Clustering
------------------------------------
Question: Are Zone B cells (high suitability, buried, no sites) closer
to volcanoes than Zone A cells (where sites are found)?

Zone A: mean={np.mean(dist_a):.1f} km, median={np.median(dist_a):.1f} km (n={len(dist_a):,})
Zone B: mean={np.mean(dist_b):.1f} km, median={np.median(dist_b):.1f} km (n={len(dist_b):,})
Zone C: mean={zone_stats_df.loc[zone_stats_df['zone']=='C', 'mean_dist_km'].values[0]:.1f} km (n={zone_stats_df.loc[zone_stats_df['zone']=='C', 'n_cells'].values[0]:,})
Zone E: mean={np.mean(dist_e):.1f} km, median={np.median(dist_e):.1f} km (n={len(dist_e):,})

KEY TEST: Zone A vs Zone B
  Mann-Whitney U = {u_stat:.0f}
  p-value = {p_val:.2e}
  Z = {z_score:.2f}
  r (effect size) = {r_effect:.3f}
  Cohen's d = {d_effect:.3f}

VERDICT: {verdict}

Supplementary comparisons:
  Zone A vs Zone E: p={p_ae:.2e}, Cohen's d={d_ae:.3f}
  Zone B vs Zone E: p={p_be:.2e}, Cohen's d={d_be:.3f}

Interpretation:
{"- Zone B cells cluster closer to the volcanic axis, consistent with H-TOM." if p_val < 0.05 and np.median(dist_b) < np.median(dist_a) else "- Zone B cells do not show a clear proximity pattern relative to Zone A."}
- Burial depth increases with volcanic proximity (Pyle 1989 model),
  so zones closer to volcanoes have deeper burial and fewer surface sites.

Zone Distance Summary:
{zone_stats_df.to_string(index=False)}
"""
    with open(RESULTS_DIR / "spatial_summary.txt", "w") as f:
        f.write(summary)
    print(f"  Saved: {RESULTS_DIR / 'spatial_summary.txt'}")

    print(f"\nAnalysis 2 COMPLETE.")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
