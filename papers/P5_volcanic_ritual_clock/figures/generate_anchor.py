"""
Generate anchor figure for P5: The Volcanic Ritual Clock
Poses the question — shows the slametan sequence with Javanese body-state
beliefs, purely cultural data. No forensic overlay yet.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})


def fig_anchor():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.3, 4.5)
    ax.axis("off")

    # Color palette
    c_main = "#1A237E"      # deep indigo for text
    c_accent = "#C62828"    # deep red for the question
    c_line = "#5C6BC0"      # indigo line
    c_dot = "#283593"       # indigo dots
    c_belief = "#4E342E"    # brown for body-state beliefs
    c_bg = "#FAFAFA"

    fig.set_facecolor("white")

    # Title — the question
    ax.text(5, 4.2, "The Slametan Death Cycle: What Generated These Numbers?",
            ha="center", va="center", fontsize=13, fontweight="bold", color=c_main)

    # Subtitle
    ax.text(5, 3.8, "Javanese mortuary commemorations follow a fixed numerical sequence — "
            "unchanged across four religious conversions",
            ha="center", va="center", fontsize=8.5, color="#666", fontstyle="italic")

    # The 5 ceremonies as a horizontal timeline
    ceremonies = [
        {
            "day": 3, "name": "nelung dina", "x": 1.0,
            "belief": "Spirit remains\nin the house",
            "body": "Nafsu dissipate",
        },
        {
            "day": 7, "name": "mitung dina", "x": 3.0,
            "belief": "Spirit begins\nto depart",
            "body": "Skin and hair\nseparate",
        },
        {
            "day": 40, "name": "matang puluh", "x": 5.0,
            "belief": "Blood, flesh, marrow,\ninnards 'perfected'",
            "body": "darah, daging,\nsungsum, jeroan",
        },
        {
            "day": 100, "name": "nyatus", "x": 7.0,
            "belief": "Physical body\n'perfected'",
            "body": "badan wadhag\nsampurna",
        },
        {
            "day": 1000, "name": "nyewu", "x": 9.0,
            "belief": "Body 'fully one\nwith the earth'",
            "body": "akan tidak kembali\nke keluarga",
        },
    ]

    # Draw timeline line
    y_line = 2.3
    ax.plot([0.3, 9.7], [y_line, y_line], color=c_line, linewidth=2.5, alpha=0.4, zorder=1)

    for cer in ceremonies:
        x = cer["x"]

        # Vertical tick
        ax.plot([x, x], [y_line - 0.08, y_line + 0.08], color=c_line, linewidth=2, zorder=2)

        # Large circle with day number
        circle = plt.Circle((x, y_line), 0.28, facecolor=c_dot, edgecolor="white",
                            linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y_line, str(cer["day"]), ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=4)

        # Javanese name above
        ax.text(x, y_line + 0.55, cer["name"], ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=c_main)

        # "day X" label
        ax.text(x, y_line + 0.38, f"day {cer['day']}", ha="center", va="center",
                fontsize=7.5, color="#888")

        # Body-state belief below (italic) — the cultural claim
        ax.text(x, y_line - 0.55, cer["belief"], ha="center", va="top",
                fontsize=7.5, color=c_belief, fontstyle="italic", linespacing=1.2)

        # Javanese text in smaller font
        ax.text(x, y_line - 1.1, cer["body"], ha="center", va="top",
                fontsize=6.5, color="#999", fontstyle="italic", linespacing=1.2)

    # Arrow showing progression
    ax.annotate("", xy=(9.7, y_line), xytext=(9.3, y_line),
                arrowprops=dict(arrowstyle="-|>", color=c_line, lw=2))

    # Bottom: the puzzle statement
    ax.text(5, -0.05,
            "Not Hindu (no 40, 100, 1000)  ·  Not Buddhist (49, not 40)  ·  "
            "Not Islamic (bid'ah)  ·  Not Christian (adopted from substrate)",
            ha="center", va="center", fontsize=7.5, color="#888")

    # The question
    ax.text(5, 0.25,
            "These descriptions match forensic decomposition stages in volcanic soil with p = 0.008. Coincidence?",
            ha="center", va="center", fontsize=9, color=c_accent, fontweight="bold")

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "fig0_anchor.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig0_anchor.png", bbox_inches="tight", dpi=300)
    print("  Anchor figure saved.")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating anchor figure...")
    fig_anchor()
    print("Done.")
