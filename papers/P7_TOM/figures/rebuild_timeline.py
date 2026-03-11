"""Rebuild fig2_dwarapala_timeline with fixed text overlap."""
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)

OUT_DIR = Path(__file__).parent


def build():
    fig, ax = plt.subplots(figsize=(10, 6.5))  # taller to avoid cramping

    # Timeline parameters
    years = [1268, 1803, 2026]
    labels = [
        "Construction\n~1268 CE\n(Singosari Kingdom)",
        "Discovery\n1803 CE\n(Nicolaus Engelhard)",
        "Present\n2026 CE",
    ]
    burial_cm = [0, 185, 185 + (2026 - 1803) * 0.36]

    # Draw timeline axis — moved up to y=0.55
    ax.axhline(y=0.55, xmin=0.05, xmax=0.95, color="#333", linewidth=2, zorder=1)

    # Position markers
    x_positions = [0.15, 0.55, 0.85]
    for i, (xp, yr, lbl, depth) in enumerate(zip(x_positions, years, labels, burial_cm)):
        ax.plot([xp, xp], [0.50, 0.60], color="#333", linewidth=2, zorder=2,
                transform=ax.transAxes)
        color = ["#D32F2F", "#1565C0", "#2E7D32"][i]
        ax.plot(xp, 0.55, "o", color=color, markersize=14, zorder=3,
                transform=ax.transAxes)
        # Label above
        ax.text(xp, 0.72, lbl, ha="center", va="center", fontsize=9,
                fontweight="bold", transform=ax.transAxes)
        # Burial depth below — moved down with more space
        if i > 0:
            ax.text(xp, 0.36, f"Burial: ~{depth:.0f} cm\n({depth/100:.1f} m)",
                    ha="center", va="center", fontsize=8.5, color="#555",
                    transform=ax.transAxes)

    # Arrows between points
    ax.annotate("", xy=(0.55, 0.55), xytext=(0.15, 0.55),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.5),
                xycoords="axes fraction")
    ax.text(0.35, 0.61, "535 years\n3.6 mm/yr sedimentation",
            ha="center", va="bottom", fontsize=8, color="#666",
            transform=ax.transAxes)

    ax.annotate("", xy=(0.85, 0.55), xytext=(0.55, 0.55),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.5),
                xycoords="axes fraction")
    ax.text(0.70, 0.61, "223 years\ncontinuing burial",
            ha="center", va="bottom", fontsize=8, color="#666",
            transform=ax.transAxes)

    # Statue schematics — moved down
    # 1268: fully exposed
    rect1 = FancyBboxPatch((0.11, 0.08), 0.08, 0.18, boxstyle="round,pad=0.01",
                            facecolor="#D4A574", edgecolor="#8B6914", linewidth=1.5,
                            transform=ax.transAxes)
    ax.add_patch(rect1)
    ax.text(0.15, 0.17, "370 cm\nexposed", ha="center", va="center", fontsize=6.5,
            transform=ax.transAxes)

    # 1803: half buried
    rect2_top = FancyBboxPatch((0.51, 0.17), 0.08, 0.09, boxstyle="round,pad=0.01",
                                facecolor="#D4A574", edgecolor="#8B6914", linewidth=1.5,
                                transform=ax.transAxes)
    rect2_buried = FancyBboxPatch((0.51, 0.08), 0.08, 0.09, boxstyle="round,pad=0.01",
                                   facecolor="#A0522D", edgecolor="#8B6914", linewidth=1.5,
                                   transform=ax.transAxes, alpha=0.5)
    ax.add_patch(rect2_buried)
    ax.add_patch(rect2_top)
    # Ground level dashed line
    ax.axhline(y=0.17, xmin=0.48, xmax=0.62, color="#654321", linewidth=1.5,
               linestyle="--", alpha=0.7)
    ax.text(0.55, 0.22, "185 cm\nexposed", ha="center", va="center", fontsize=6.5,
            transform=ax.transAxes)
    ax.text(0.55, 0.125, "185 cm\nburied", ha="center", va="center", fontsize=6.5,
            color="white", transform=ax.transAxes)

    # Ground level label — offset to right
    ax.text(0.63, 0.17, "ground\nlevel", ha="left", va="center", fontsize=6,
            color="#654321", style="italic", transform=ax.transAxes)

    # Key finding box at bottom
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#F9A825",
                      linewidth=1.5)
    ax.text(0.50, -0.05,
            "Rate = 185 cm / 510 yr = 3.6 mm/yr  |  Kelud: ~20 eruptions in period  |  "
            "Cross-system mean: 4.4 \u00b1 1.2 mm/yr",
            ha="center", va="center", fontsize=8, transform=ax.transAxes,
            bbox=bbox_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 0.90)
    ax.axis("off")
    ax.set_title("Dwarapala Singosari: Empirical Calibration of Volcanic Sedimentation Rate",
                 fontsize=12, fontweight="bold", pad=15)

    # Save PNG
    png_out = OUT_DIR / "fig2_dwarapala_timeline.png"
    plt.savefig(png_out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png_out}")

    # Save TIF
    tif_out = OUT_DIR / "fig2_dwarapala_timeline.tif"
    plt.savefig(tif_out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {tif_out}")

    plt.close()


if __name__ == "__main__":
    build()
