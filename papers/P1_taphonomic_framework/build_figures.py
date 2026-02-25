"""
Paper 1 Figure Generator — Taphonomic Framework
Generates publication-quality figures for JAS:Reports submission.

Run from repo root:
    py papers/P1_taphonomic_framework/build_figures.py
"""

import sys
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def fig1_dwarapala_timeline():
    """Fig 1: Dwarapala burial timeline showing sedimentation accumulation."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Timeline parameters
    years = [1268, 1803, 2026]
    labels = [
        "Construction\n~1268 CE\n(Singosari Kingdom)",
        "Discovery\n1803 CE\n(Nicolaus Engelhard)",
        "Present\n2026 CE",
    ]
    burial_cm = [0, 185, 185 + (2026 - 1803) * 0.36]  # 0.36 cm/yr

    # Draw timeline axis
    ax.axhline(y=0.5, xmin=0.05, xmax=0.95, color="#333", linewidth=2, zorder=1)

    # Position markers
    x_positions = [0.15, 0.55, 0.85]
    for i, (xp, yr, lbl, depth) in enumerate(zip(x_positions, years, labels, burial_cm)):
        # Vertical marker
        ax.plot([xp, xp], [0.45, 0.55], color="#333", linewidth=2, zorder=2,
                transform=ax.transAxes)
        # Circle marker
        color = ["#D32F2F", "#1565C0", "#2E7D32"][i]
        ax.plot(xp, 0.5, "o", color=color, markersize=14, zorder=3,
                transform=ax.transAxes)
        # Label above
        ax.text(xp, 0.68, lbl, ha="center", va="center", fontsize=9,
                fontweight="bold", transform=ax.transAxes)
        # Burial depth below
        if i > 0:
            ax.text(xp, 0.30, f"Burial: ~{depth:.0f} cm\n({depth/100:.1f} m)",
                    ha="center", va="center", fontsize=8.5, color="#555",
                    transform=ax.transAxes)

    # Arrows between points with duration
    ax.annotate("", xy=(0.55, 0.50), xytext=(0.15, 0.50),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.5),
                xycoords="axes fraction")
    ax.text(0.35, 0.56, "535 years\n3.6 mm/yr sedimentation",
            ha="center", va="bottom", fontsize=8, color="#666",
            transform=ax.transAxes)

    ax.annotate("", xy=(0.85, 0.50), xytext=(0.55, 0.50),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.5),
                xycoords="axes fraction")
    ax.text(0.70, 0.56, "223 years\ncontinuing burial",
            ha="center", va="bottom", fontsize=8, color="#666",
            transform=ax.transAxes)

    # Statue schematic at bottom
    # 1268: fully exposed
    rect1 = FancyBboxPatch((0.11, 0.05), 0.08, 0.18, boxstyle="round,pad=0.01",
                            facecolor="#D4A574", edgecolor="#8B6914", linewidth=1.5,
                            transform=ax.transAxes)
    ax.add_patch(rect1)
    ax.text(0.15, 0.14, "370 cm\nexposed", ha="center", va="center", fontsize=6.5,
            transform=ax.transAxes)

    # 1803: half buried
    rect2_top = FancyBboxPatch((0.51, 0.14), 0.08, 0.09, boxstyle="round,pad=0.01",
                                facecolor="#D4A574", edgecolor="#8B6914", linewidth=1.5,
                                transform=ax.transAxes)
    rect2_buried = FancyBboxPatch((0.51, 0.05), 0.08, 0.09, boxstyle="round,pad=0.01",
                                   facecolor="#A0522D", edgecolor="#8B6914", linewidth=1.5,
                                   transform=ax.transAxes, alpha=0.5)
    ax.add_patch(rect2_buried)
    ax.add_patch(rect2_top)
    ax.axhline(y=0.14, xmin=0.48, xmax=0.62, color="#654321", linewidth=1.5,
               linestyle="--", alpha=0.7)
    ax.text(0.55, 0.185, "185 cm\nexposed", ha="center", va="center", fontsize=6.5,
            transform=ax.transAxes)
    ax.text(0.55, 0.095, "185 cm\nburied", ha="center", va="center", fontsize=6.5,
            color="white", transform=ax.transAxes)

    # Ground level label
    ax.text(0.63, 0.14, "ground\nlevel", ha="left", va="center", fontsize=6,
            color="#654321", style="italic", transform=ax.transAxes)

    # Key finding box
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#F9A825",
                      linewidth=1.5)
    ax.text(0.50, -0.08,
            "Rate = 185 cm / 510 yr = 3.6 mm/yr  |  Kelud: ~20 eruptions in period  |  "
            "Cross-system mean: 4.4 ± 1.2 mm/yr",
            ha="center", va="center", fontsize=8, transform=ax.transAxes,
            bbox=bbox_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 0.85)
    ax.axis("off")
    ax.set_title("Dwarapala Singosari: Empirical Calibration of Volcanic Sedimentation Rate",
                 fontsize=12, fontweight="bold", pad=15)

    out = FIGURES_DIR / "fig1_dwarapala_timeline.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


def fig2_burial_depth_projections():
    """Fig 2: Burial depth projection by era with uncertainty range."""
    eras = [
        ("Late Majapahit\n~1400 CE", 1400),
        ("Singosari\n~1268 CE", 1268),
        ("Mataram (E. Java)\n~900 CE", 900),
        ("Kanjuruhan\n~760 CE", 760),
        ("Pre-Hindu\n~400 CE", 400),
    ]

    rates = {"Low (2.4 mm/yr)": 2.4, "Dwarapala (3.5 mm/yr)": 3.5,
             "Mean (4.4 mm/yr)": 4.4, "High (6.2 mm/yr)": 6.2}

    fig, ax = plt.subplots(figsize=(10, 6))

    era_labels = [e[0] for e in eras]
    era_years = [e[1] for e in eras]
    x = np.arange(len(eras))
    width = 0.18
    colors = ["#81C784", "#42A5F5", "#FFB74D", "#E57373"]

    for i, (rate_name, rate_val) in enumerate(rates.items()):
        depths = [(2026 - yr) * rate_val / 1000 for yr in era_years]  # meters
        bars = ax.bar(x + i * width - 1.5 * width, depths, width,
                      label=rate_name, color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, depth in zip(bars, depths):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{depth:.1f}", ha="center", va="bottom", fontsize=6.5, rotation=0)

    # GPR effective range
    ax.axhspan(0, 5, alpha=0.08, color="green", zorder=0)
    ax.text(len(eras) - 0.3, 2.5, "GPR effective\nrange (0-5 m)",
            ha="right", va="center", fontsize=7.5, color="#2E7D32", style="italic")

    ax.axhline(y=5, color="#2E7D32", linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(era_labels, fontsize=9)
    ax.set_ylabel("Estimated Burial Depth (meters)", fontsize=10)
    ax.set_title("Projected Burial Depth by Era\n(based on 4 calibration points across 2 volcanic systems)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_ylim(0, max([(2026 - 400) * 6.2 / 1000]) + 1.5)
    ax.invert_yaxis()  # Depth goes down
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIGURES_DIR / "fig2_burial_depth_projections.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


def fig3_calibration_points():
    """Fig 3: Multi-point calibration comparison across volcanic systems."""
    sites = [
        ("Dwarapala\nSingosari", 3.5, "Kelud", "#E53935"),
        ("Candi\nSambisari", 5.05, "Merapi", "#1E88E5"),  # midpoint of 4.4-5.7
        ("Candi\nKedulan", 5.75, "Merapi", "#1E88E5"),    # midpoint of 5.3-6.2
        ("Candi\nKimpulan", 3.45, "Merapi", "#1E88E5"),   # midpoint of 2.4-4.5
    ]
    errors = [
        (0, 0),          # Dwarapala: point estimate
        (0.65, 0.65),    # Sambisari: 4.4-5.7
        (0.45, 0.45),    # Kedulan: 5.3-6.2
        (1.05, 1.05),    # Kimpulan: 2.4-4.5
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(sites))
    for i, (name, rate, system, color) in enumerate(sites):
        err_low, err_high = errors[i]
        ax.bar(i, rate, color=color, edgecolor="white", width=0.6, alpha=0.85,
               label=f"{system} system" if i in [0, 1] else "")
        if err_low > 0:
            ax.errorbar(i, rate, yerr=[[err_low], [err_high]], fmt="none",
                        color="#333", capsize=5, linewidth=1.5)
        ax.text(i, rate + err_high + 0.15, f"{rate:.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    # Mean line
    mean_rate = 4.4
    ax.axhline(y=mean_rate, color="#FF9800", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(len(sites) - 0.5, mean_rate + 0.15,
            f"Cross-system mean: {mean_rate} ± 1.2 mm/yr",
            ha="right", va="bottom", fontsize=8, color="#E65100", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in sites], fontsize=9)
    ax.set_ylabel("Sedimentation Rate (mm/yr)", fontsize=10)
    ax.set_title("Empirical Sedimentation Rates: 4 Sites, 2 Volcanic Systems",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Custom legend
    handles = [
        mpatches.Patch(color="#E53935", label="Kelud system (East Java)"),
        mpatches.Patch(color="#1E88E5", label="Merapi system (Central Java)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5)

    out = FIGURES_DIR / "fig3_calibration_rates.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


def fig4_density_vs_distance():
    """Fig 4: Site density vs volcanic distance (E004 result)."""
    # Data from E004 results
    bands = ["0-25", "25-50", "50-75", "75-100", "100-150", "150-200", "200+"]
    densities = [12.96, 7.54, 2.00, 1.34, 1.44, 0.00, 0.00]
    sites = [147, 136, 37, 22, 41, 0, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: density bar chart
    colors = plt.cm.YlOrRd(np.linspace(0.8, 0.2, len(bands)))
    bars = ax1.bar(range(len(bands)), densities, color=colors, edgecolor="white", width=0.7)
    ax1.set_xticks(range(len(bands)))
    ax1.set_xticklabels(bands, fontsize=8, rotation=15)
    ax1.set_xlabel("Distance to Nearest Volcano (km)", fontsize=9)
    ax1.set_ylabel("Site Density (per 1,000 km²)", fontsize=9)
    ax1.set_title("Site Density by Volcanic Distance Band", fontsize=10, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for bar, d, n in zip(bars, densities, sites):
        if d > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"n={n}", ha="center", va="bottom", fontsize=7)

    ax1.text(0.95, 0.95, "Spearman ρ = −0.955\np = 0.0008",
             ha="right", va="top", transform=ax1.transAxes, fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#F9A825"))

    # Right: interpretation diagram
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("Why Sites Cluster Near Volcanoes", fontsize=10, fontweight="bold")

    explanations = [
        (0.5, 0.82, "NOT because ancient people preferred volcanic zones",
         "#E53935", "✗"),
        (0.5, 0.60, "Survey history: 200 years of BPCB focus on\n"
         "Singosari-Majapahit heartland (0-50 km from Kelud)",
         "#1565C0", "→"),
        (0.5, 0.38, "Survivorship bias: stone candis survive burial;\n"
         "wooden settlements (>99% of habitations) do not",
         "#2E7D32", "→"),
        (0.5, 0.16, "Blank zones (>150 km) = unsurveyed,\nnot uninhabited",
         "#FF8F00", "→"),
    ]
    for x, y, text, color, symbol in explanations:
        ax2.text(0.08, y, symbol, ha="center", va="center", fontsize=14,
                 color=color, fontweight="bold")
        ax2.text(0.15, y, text, ha="left", va="center", fontsize=8.5, color="#333",
                 wrap=True)

    plt.tight_layout()
    out = FIGURES_DIR / "fig4_density_vs_distance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating Paper 1 figures...")
    print()
    fig1_dwarapala_timeline()
    fig2_burial_depth_projections()
    fig3_calibration_points()
    fig4_density_vs_distance()
    print()
    print("All Paper 1 figures generated.")
