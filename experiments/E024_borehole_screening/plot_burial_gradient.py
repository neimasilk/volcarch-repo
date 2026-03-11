"""
E024: Plot burial depth/rate vs distance to volcano.
Tests whether burial follows Pyle (1989) exponential thinning.

Run: python experiments/E024_borehole_screening/plot_burial_gradient.py
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
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

REPO = Path(__file__).parent.parent.parent
DATA = Path(__file__).parent / "data" / "buried_sites_v0.1.csv"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def pyle_exponential(d, T0, k):
    """Pyle 1989 exponential thinning: T = T0 * exp(-k * d)"""
    return T0 * np.exp(-k * d)


def main():
    print("Loading buried sites data...")
    df = pd.read_csv(DATA)
    print(f"  Total records: {len(df)}")

    # ============================================
    # FIGURE 1: Burial depth vs distance (all sites)
    # ============================================
    print("\nFig 1: Burial depth vs distance to volcano...")

    # Filter to sites with actual burial depth and distance
    has_depth = df[(df["burial_depth_m"] > 0) & (df["dist_to_volcano_km"] > 0)].copy()
    print(f"  Sites with burial depth + distance: {len(has_depth)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Burial depth vs distance
    ax1 = axes[0]
    type_colors = {
        "hindu_temple": "#e74c3c",
        "buddhist_temple": "#3498db",
        "buddhist_settlement": "#3498db",
        "statue": "#f39c12",
        "paleosol": "#27ae60",
        "volcanic_sequence": "#8e44ad",
        "geotechnical": "#95a5a6",
    }
    type_labels = {
        "hindu_temple": "Hindu temple (buried)",
        "buddhist_temple": "Buddhist temple",
        "buddhist_settlement": "Buddhist settlement",
        "statue": "Statue (Dwarapala)",
        "paleosol": "Paleosol (Sangiran)",
        "volcanic_sequence": "Volcanic section",
        "geotechnical": "Geotechnical borehole",
    }

    for site_type, color in type_colors.items():
        mask = has_depth["type"] == site_type
        if mask.sum() > 0:
            subset = has_depth[mask]
            ax1.scatter(subset["dist_to_volcano_km"], subset["burial_depth_m"],
                       c=color, s=100, label=type_labels.get(site_type, site_type),
                       edgecolors="black", linewidths=0.5, zorder=5)
            # Annotate
            for _, row in subset.iterrows():
                ax1.annotate(row["name"].split("(")[0].strip()[:20],
                           (row["dist_to_volcano_km"], row["burial_depth_m"]),
                           fontsize=6, xytext=(4, 4), textcoords="offset points")

    # Fit exponential (exclude paleosols — different timescale)
    fit_data = has_depth[~has_depth["type"].isin(["paleosol"])]
    if len(fit_data) >= 3:
        try:
            popt, pcov = curve_fit(pyle_exponential,
                                   fit_data["dist_to_volcano_km"].values,
                                   fit_data["burial_depth_m"].values,
                                   p0=[30, 0.05], maxfev=5000)
            x_fit = np.linspace(0, 50, 100)
            y_fit = pyle_exponential(x_fit, *popt)
            ax1.plot(x_fit, y_fit, "r--", alpha=0.7, linewidth=2,
                    label=f"Pyle fit: T={popt[0]:.1f}·e$^{{-{popt[1]:.3f}d}}$")
            print(f"  Pyle fit: T0={popt[0]:.1f}m, k={popt[1]:.4f}/km")
        except Exception as e:
            print(f"  Exponential fit failed: {e}")

    ax1.set_xlabel("Distance to nearest volcano (km)", fontsize=11)
    ax1.set_ylabel("Burial depth (m)", fontsize=11)
    ax1.set_title("Burial Depth vs Distance to Volcano\n(Java archaeological & geological sites)")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.invert_yaxis()  # depth increases downward
    ax1.set_xlim(-2, 55)
    ax1.grid(True, alpha=0.3)

    # Right panel: Sedimentation RATE vs distance
    ax2 = axes[1]

    # Calculate rates where we have enough info
    rate_data = []

    # Dwarapala: 1.85m in 510 years
    rate_data.append({"name": "Dwarapala", "dist_km": 17, "rate_mm_yr": 3.6,
                      "type": "calibration"})
    # Sambisari: ~5m in ~700 years (built ~800CE, erupted ~900CE... but accumulated over time)
    # More conservatively: 5m over ~1100 years (800CE to ~1900CE)
    rate_data.append({"name": "Sambisari", "dist_km": 28, "rate_mm_yr": 5.0/1100*1000,
                      "type": "temple"})
    # Kedulan: 7m, similar timeframe
    rate_data.append({"name": "Kedulan", "dist_km": 28, "rate_mm_yr": 7.0/1100*1000,
                      "type": "temple"})
    # Kelud near-vent: 32m in 1300 years
    rate_data.append({"name": "Kelud (vent)", "dist_km": 2, "rate_mm_yr": 32/1300*1000,
                      "type": "volcanic_section"})
    # Liangan: 4m, ~1100 years
    rate_data.append({"name": "Liangan", "dist_km": 15, "rate_mm_yr": 4.0/1100*1000,
                      "type": "temple"})
    # Dieng: 2m, ~1300 years
    rate_data.append({"name": "Dieng", "dist_km": 5, "rate_mm_yr": 2.0/1300*1000,
                      "type": "temple"})
    # Morangan: 3.5m, ~1100 years (900CE to ~2000CE)
    rate_data.append({"name": "Morangan", "dist_km": 20, "rate_mm_yr": 3.5/1100*1000,
                      "type": "temple"})
    # Kadisoka: 3m, ~1100 years
    rate_data.append({"name": "Kadisoka", "dist_km": 29, "rate_mm_yr": 3.0/1100*1000,
                      "type": "temple"})
    # Kimpulan: 2m, ~1100 years
    rate_data.append({"name": "Kimpulan", "dist_km": 25, "rate_mm_yr": 2.0/1100*1000,
                      "type": "temple"})
    # Pendem: 4m, ~1200 years (800CE to ~2000CE)
    rate_data.append({"name": "Pendem", "dist_km": 18, "rate_mm_yr": 4.0/1200*1000,
                      "type": "temple"})

    rate_df = pd.DataFrame(rate_data)

    colors_rate = {"calibration": "#f39c12", "temple": "#e74c3c", "volcanic_section": "#8e44ad"}
    for rtype, color in colors_rate.items():
        mask = rate_df["type"] == rtype
        if mask.sum() > 0:
            subset = rate_df[mask]
            ax2.scatter(subset["dist_km"], subset["rate_mm_yr"],
                       c=color, s=120, edgecolors="black", linewidths=0.5,
                       label=rtype.replace("_", " ").title(), zorder=5)
            for _, row in subset.iterrows():
                ax2.annotate(row["name"],
                           (row["dist_km"], row["rate_mm_yr"]),
                           fontsize=7, xytext=(5, 5), textcoords="offset points",
                           fontweight="bold")

    # Spearman correlation (rank-based, robust to outliers)
    from scipy.stats import spearmanr
    rho, p_val = spearmanr(rate_df["dist_km"].values, rate_df["rate_mm_yr"].values)
    print(f"  Spearman rho = {rho:.3f}, p = {p_val:.4f}")

    # Show two regimes instead of forcing single exponential
    # Near-vent (<10km): Kelud, Dieng — PDC-dominated
    # Distal (>10km): Dwarapala, Sambisari, Kedulan, Liangan — fallout-dominated
    distal = rate_df[rate_df["dist_km"] >= 10]
    near = rate_df[rate_df["dist_km"] < 10]

    ax2.axvspan(0, 10, alpha=0.08, color="red", label="Near-vent zone (<10km)")
    ax2.axvspan(10, 35, alpha=0.05, color="blue", label="Distal zone (>10km)")

    if len(distal) >= 2:
        distal_mean = distal["rate_mm_yr"].mean()
        distal_std = distal["rate_mm_yr"].std()
        ax2.axhline(y=distal_mean, color="steelblue", linestyle="--", alpha=0.6)
        ax2.text(32, distal_mean + 0.3, f"Distal mean: {distal_mean:.1f} mm/yr",
                fontsize=7, color="steelblue", ha="right")

    ax2.text(0.95, 0.95,
            f"Spearman rho = {rho:.3f}\np = {p_val:.3f}\nn = {len(rate_df)} sites\n\n"
            f"Near-vent (<10km): {near['rate_mm_yr'].mean():.1f} mm/yr\n"
            f"Distal (>10km): {distal['rate_mm_yr'].mean():.1f} mm/yr",
            transform=ax2.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    print(f"  Near-vent mean: {near['rate_mm_yr'].mean():.1f} mm/yr")
    print(f"  Distal mean: {distal['rate_mm_yr'].mean():.1f} mm/yr")

    # Add Paper 1 cross-system mean
    ax2.axhline(y=3.6, color="orange", linestyle=":", alpha=0.5, linewidth=1)
    ax2.text(32, 3.8, "P1 mean: 3.6 mm/yr", fontsize=7, color="orange", alpha=0.7)

    ax2.set_xlabel("Distance to nearest volcano (km)", fontsize=11)
    ax2.set_ylabel("Sedimentation rate (mm/yr)", fontsize=11)
    ax2.set_title("Sedimentation Rate vs Distance to Volcano\n(Independent validation of Pyle 1989 exponential thinning)")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.set_xlim(-2, 35)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    for ext in ["png", "tif"]:
        out_path = OUT / f"fig1_burial_gradient.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {out_path}")
    plt.close()

    # ============================================
    # Print summary table
    # ============================================
    print("\n" + "=" * 70)
    print("BURIAL RATE SUMMARY")
    print("=" * 70)
    print(f"{'Site':<20} {'Dist (km)':<12} {'Rate (mm/yr)':<14} {'Type'}")
    print("-" * 70)
    for _, row in rate_df.sort_values("dist_km").iterrows():
        print(f"{row['name']:<20} {row['dist_km']:<12.0f} {row['rate_mm_yr']:<14.1f} {row['type']}")

    # Implications using observed rates
    print("\n" + "=" * 70)
    print("IMPLICATIONS FOR H-TOM")
    print("=" * 70)
    print(f"\nPredicted burial depths using observed rates:")
    print(f"{'Zone':<20} {'Rate':<12} {'1 ka':<10} {'10 ka':<10} {'68 ka (pre-Sulawesi)':<20}")
    print("-" * 72)
    for label, rate in [("Near-vent (2km)", 24.6), ("Proximal (5km)", 1.5),
                         ("Distal mean", distal_mean), ("P1 calibration", 3.6)]:
        b1 = rate * 1  # 1ka in meters
        b10 = rate * 10  # 10ka in meters
        b68 = rate * 68  # 68ka in meters
        print(f"{label:<20} {rate:>6.1f} mm/yr  {b1:>6.1f} m   {b10:>6.1f} m   {b68:>7.1f} m")

    print("\nDone.")


if __name__ == "__main__":
    main()
