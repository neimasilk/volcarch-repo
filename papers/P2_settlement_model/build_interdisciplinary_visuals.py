"""
Build interdisciplinary illustration assets for Paper 2.

Run from repo root:
    py papers/P2_settlement_model/build_interdisciplinary_visuals.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO_ROOT = Path(__file__).parent.parent.parent
FIG_DIR = REPO_ROOT / "papers" / "P2_settlement_model" / "figures"


def draw_box(ax, xy, w, h, text, fc="#f5f5f5", ec="#333333", fontsize=10):
    rect = Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1f1f1f",
        wrap=True,
    )
    return rect


def draw_arrow(ax, p1, p2, color="#555555"):
    arr = FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=12, linewidth=1.2, color=color)
    ax.add_patch(arr)


def fig1_interdisciplinary_framework(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    draw_box(ax, (0.8, 5.3), 2.6, 1.5, "Computer Science\n(ML + Spatial CV)", fc="#d9edf7")
    draw_box(ax, (3.7, 6.1), 2.6, 1.5, "Archaeology\n(Survey + Discovery Bias)", fc="#fcf8e3")
    draw_box(ax, (6.6, 5.3), 2.6, 1.5, "Geology\n(Tephra + Burial Dynamics)", fc="#f2dede")

    draw_box(
        ax,
        (3.0, 3.1),
        4.0,
        1.5,
        "Observed Site Record\n(visible sites only)",
        fc="#eeeeee",
    )
    draw_box(
        ax,
        (3.0, 0.9),
        4.0,
        1.5,
        "Bias-Corrected Suitability Model\n(Tautology-free target)",
        fc="#dff0d8",
    )

    draw_arrow(ax, (2.1, 5.3), (4.0, 4.6))
    draw_arrow(ax, (5.0, 6.1), (5.0, 4.6))
    draw_arrow(ax, (7.9, 5.3), (6.0, 4.6))
    draw_arrow(ax, (5.0, 3.1), (5.0, 2.4), color="#2e7d32")

    ax.text(
        5.0,
        7.6,
        "Figure 1. Interdisciplinary Bias Framework",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        5.0,
        0.25,
        "Goal: align ML validation, survey logic, and depositional context in one pipeline",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig8_pipeline_overview(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.5, 2.2, 2.1, 1.5, "E007\nTerrain Baseline\nAUC 0.659", "#f7f7f7"),
        (2.9, 2.2, 2.1, 1.5, "E008\n+ River Distance\nAUC 0.695", "#e8f4fa"),
        (5.3, 2.2, 2.1, 1.5, "E009\n+ Soil Layers\nAUC 0.664", "#fdecea"),
        (7.7, 2.2, 2.1, 1.5, "E010-E012\nTGB Redesign\nAUC 0.711-0.730", "#fff8e1"),
        (10.1, 2.2, 1.4, 1.5, "E013\nHybrid\nAUC 0.768", "#dff0d8"),
    ]
    for x, y, w, h, txt, col in boxes:
        draw_box(ax, (x, y), w, h, txt, fc=col)

    for x1, x2 in [(2.6, 2.9), (5.0, 5.3), (7.4, 7.7), (9.8, 10.1)]:
        draw_arrow(ax, (x1, 2.95), (x2, 2.95))

    draw_box(
        ax,
        (3.1, 0.5),
        5.8,
        1.0,
        "Key lesson: pseudo-absence design changes transfer more than adding raw features",
        fc="#eeeeee",
        fontsize=10,
    )
    draw_arrow(ax, (6.0, 2.2), (6.0, 1.5), color="#666666")

    ax.text(
        6.0,
        5.5,
        "Figure 8. Experiment-to-Decision Pipeline (E007-E013)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig9_interpretation_bridge(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    draw_box(ax, (0.7, 4.7), 2.6, 1.4, "Computer Science:\nGeneralization\nunder spatial shift", fc="#d9edf7")
    draw_box(ax, (0.7, 2.8), 2.6, 1.4, "Archaeology:\nDiscovery process\nvs true settlement", fc="#fcf8e3")
    draw_box(ax, (0.7, 0.9), 2.6, 1.4, "Geology:\nBurial and visibility\nconstraints", fc="#f2dede")

    draw_box(
        ax,
        (4.0, 2.4),
        2.2,
        2.1,
        "Shared\nInference\nCore",
        fc="#e8e8e8",
        fontsize=11,
    )
    draw_box(
        ax,
        (6.9, 2.4),
        2.4,
        2.1,
        "Operational Output:\nPriority zones for\nfield verification",
        fc="#dff0d8",
        fontsize=10,
    )

    draw_arrow(ax, (3.3, 5.4), (4.0, 4.1))
    draw_arrow(ax, (3.3, 3.5), (4.0, 3.45))
    draw_arrow(ax, (3.3, 1.6), (4.0, 2.8))
    draw_arrow(ax, (6.2, 3.45), (6.9, 3.45), color="#2e7d32")

    ax.text(
        5.0,
        6.6,
        "Figure 9. Interpretation Bridge Across Disciplines",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    f1 = FIG_DIR / "fig1_interdisciplinary_framework.png"
    f8 = FIG_DIR / "fig8_pipeline_overview.png"
    f9 = FIG_DIR / "fig9_interpretation_bridge.png"

    fig1_interdisciplinary_framework(f1)
    fig8_pipeline_overview(f8)
    fig9_interpretation_bridge(f9)

    print("Saved:")
    print(f"  {f1}")
    print(f"  {f8}")
    print(f"  {f9}")


if __name__ == "__main__":
    main()
