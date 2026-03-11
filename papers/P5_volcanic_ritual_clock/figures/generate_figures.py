"""
Generate figures for P5: The Volcanic Ritual Clock
Three figures:
  Fig 1 — Slametan intervals mapped to decomposition stages (core argument)
  Fig 2 — Monte Carlo permutation results (statistical evidence)
  Fig 3 — H-TOM two-timescale diagram (synthesis)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# Consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})

COLORS = {
    "fresh": "#4CAF50",
    "bloat": "#FF9800",
    "decay": "#F44336",
    "skeleton": "#9C27B0",
    "bone": "#795548",
    "slametan": "#1565C0",
    "geo": "#E65100",
    "volc": "#D32F2F",
}


# ================================================================
# FIGURE 1: Slametan-Decomposition Mapping
# ================================================================
def fig1_mapping():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Decomposition stage ranges (days, log scale)
    stages = [
        ("Fresh\nends",        1,    5,    COLORS["fresh"]),
        ("Bloat\npeak",        3,   14,    COLORS["bloat"]),
        ("Advanced\ndecay",   20,   80,    COLORS["decay"]),
        ("Skeleton-\nisation", 60,  300,   COLORS["skeleton"]),
        ("Bone\ndissolution", 300, 2500,   COLORS["bone"]),
    ]

    # Slametan intervals
    slametan = [
        (3,    "nelung dina\n(3)",    "Spirit in house;\nautolysis"),
        (7,    "mitung dina\n(7)",    "Skin separates;\nputrefactive gases peak"),
        (40,   "matang puluh\n(40)",  "Soft tissue\n'perfected'"),
        (100,  "nyatus\n(100)",       "Physical body\n'perfected'"),
        (1000, "nyewu\n(1000)",       "Body 'fully one\nwith the earth'"),
    ]

    y_stage = 0.3
    bar_h = 0.18

    # Draw decomposition stage bars
    for i, (name, lo, hi, color) in enumerate(stages):
        ax.barh(y_stage, np.log10(hi) - np.log10(lo),
                left=np.log10(lo), height=bar_h,
                color=color, alpha=0.35, edgecolor=color, linewidth=1.2)
        # Label below bar
        mid = (np.log10(lo) + np.log10(hi)) / 2
        ax.text(mid, y_stage - 0.16, name,
                ha="center", va="top", fontsize=8, color=color, fontweight="bold")

    # Stage bar label
    ax.text(-0.15, y_stage, "Forensic\ndecomposition\nstages", ha="right", va="center",
            fontsize=9, fontstyle="italic", color="#555")

    # Draw slametan markers
    y_slam = 0.75
    for i, (day, name, belief) in enumerate(slametan):
        x = np.log10(day)
        # Vertical line from slametan to stage bar
        ax.plot([x, x], [y_stage + bar_h/2 + 0.02, y_slam - 0.03],
                color=COLORS["slametan"], linewidth=1.5, linestyle="-", alpha=0.6)
        # Diamond marker
        ax.plot(x, y_slam, marker="D", color=COLORS["slametan"],
                markersize=10, zorder=5)
        # Slametan name above
        ax.text(x, y_slam + 0.06, name,
                ha="center", va="bottom", fontsize=8.5,
                color=COLORS["slametan"], fontweight="bold")
        # Javanese belief text (italicized)
        ax.text(x, y_slam + 0.23, belief,
                ha="center", va="bottom", fontsize=7,
                color="#333", fontstyle="italic")

    # Slametan label
    ax.text(-0.15, y_slam, "Slametan\nintervals\n(days)", ha="right", va="center",
            fontsize=9, fontstyle="italic", color=COLORS["slametan"])

    # X-axis as log scale
    tick_days = [1, 3, 7, 10, 40, 100, 365, 1000, 2500]
    ax.set_xticks([np.log10(d) for d in tick_days])
    ax.set_xticklabels([str(d) for d in tick_days], fontsize=9)
    ax.set_xlabel("Days after death (log scale)", fontsize=10)

    ax.set_xlim(-0.2, np.log10(3500))
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_title("Slametan Mortuary Intervals Mapped to Forensic Decomposition Stages\n"
                 "(tropical burial, volcanic Andosol pH 4.5–5.5, 26–28°C)",
                 fontsize=11, fontweight="bold", pad=15)

    # Legend
    stage_patch = mpatches.Patch(color="#999", alpha=0.35, label="Decomposition stage range (forensic literature)")
    slam_line = plt.Line2D([0], [0], marker="D", color=COLORS["slametan"],
                           linestyle="None", markersize=8, label="Slametan interval")
    ax.legend(handles=[stage_patch, slam_line], loc="lower right", fontsize=8,
              framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_slametan_decomposition.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_slametan_decomposition.png", bbox_inches="tight", dpi=300)
    print("  Fig 1 saved.")
    plt.close(fig)


# ================================================================
# FIGURE 2: Monte Carlo / Permutation Test Results
# ================================================================
def fig2_monte_carlo():
    from itertools import permutations as perms

    # Reproduce permutation test
    SLAMETAN = [3, 7, 40, 100, 1000]
    ranges = [(1, 5), (3, 14), (20, 80), (60, 300), (300, 2500)]

    perm_matches = []
    for perm in perms(SLAMETAN):
        matches = sum(1 for val, (lo, hi) in zip(perm, ranges) if lo <= val <= hi)
        perm_matches.append(matches)

    # Monte Carlo uniform (quick version for figure)
    np.random.seed(42)
    N = 200_000
    mc_matches = []
    for _ in range(N):
        rand = sorted(np.random.randint(1, 1501, size=5))
        matches = sum(1 for val, (lo, hi) in zip(rand, ranges) if lo <= val <= hi)
        mc_matches.append(matches)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel A: Permutation test (120 permutations)
    bins = np.arange(-0.5, 6.5, 1)
    counts, _, bars = ax1.hist(perm_matches, bins=bins, color="#90CAF9",
                                edgecolor="#1565C0", linewidth=1.2, rwidth=0.8)

    # Highlight the 5/5 bar
    for bar, left_edge in zip(bars, bins[:-1]):
        if left_edge == 4.5:
            bar.set_color(COLORS["slametan"])
            bar.set_alpha(1.0)

    ax1.set_xlabel("Number of stages matched", fontsize=10)
    ax1.set_ylabel("Number of permutations", fontsize=10)
    ax1.set_title("(a) Exact Permutation Test\n(5! = 120 permutations)", fontsize=10, fontweight="bold")
    ax1.set_xticks(range(6))
    ax1.set_xlim(-0.5, 5.5)

    # Annotate the 5/5 result
    n_5 = perm_matches.count(5)
    ax1.annotate(f"Observed: {n_5}/120\np = {n_5/120:.3f}",
                 xy=(5, n_5), xytext=(3.5, max(counts)*0.7),
                 fontsize=9, fontweight="bold", color=COLORS["slametan"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["slametan"], lw=1.5),
                 ha="center")

    # Panel B: Monte Carlo uniform
    counts2, _, bars2 = ax2.hist(mc_matches, bins=bins, color="#FFCC80",
                                  edgecolor="#E65100", linewidth=1.2, rwidth=0.8)

    # Highlight >= 4
    for bar, left_edge in zip(bars2, bins[:-1]):
        if left_edge >= 3.5:
            bar.set_color(COLORS["geo"])
            bar.set_alpha(1.0)

    ax2.set_xlabel("Number of stages matched", fontsize=10)
    ax2.set_ylabel(f"Count (of {N:,} random sets)", fontsize=10)
    ax2.set_title(f"(b) Monte Carlo Simulation\n({N:,} random interval sets, uniform [1, 1500])",
                  fontsize=10, fontweight="bold")
    ax2.set_xticks(range(6))
    ax2.set_xlim(-0.5, 5.5)
    ax2.set_yscale("log")

    # Annotate
    n_5_mc = mc_matches.count(5)
    n_4_mc = mc_matches.count(4)
    ax2.annotate(f"5/5 match: {n_5_mc}\np < 0.001",
                 xy=(5, max(n_5_mc, 1)), xytext=(3.5, 1000),
                 fontsize=9, fontweight="bold", color=COLORS["geo"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["geo"], lw=1.5),
                 ha="center")

    fig.suptitle("Statistical Validation: Is the Slametan–Decomposition Correspondence Random?",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_monte_carlo.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_monte_carlo.png", bbox_inches="tight", dpi=300)
    print("  Fig 2 saved.")
    plt.close(fig)


# ================================================================
# FIGURE 3: H-TOM Two-Timescale Diagram
# ================================================================
def fig3_htom():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(5, 7.6, "H-TOM: Hypothetical Taphonomic Overlay Model",
            ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(5, 7.2, "Two timescales, one volcanic driver",
            ha="center", va="center", fontsize=10, fontstyle="italic", color="#555")

    # Central volcano box
    volc_box = FancyBboxPatch((3.5, 4.3), 3, 1.2, boxstyle="round,pad=0.15",
                               facecolor="#FFCDD2", edgecolor=COLORS["volc"], linewidth=2)
    ax.add_patch(volc_box)
    ax.text(5, 5.1, "VOLCANISM", ha="center", va="center",
            fontsize=12, fontweight="bold", color=COLORS["volc"])
    ax.text(5, 4.7, "Tephra production & weathering", ha="center", va="center",
            fontsize=8.5, color="#555")

    # Left box: Ritual timescale
    left_box = FancyBboxPatch((0.3, 1.5), 3.8, 2.2, boxstyle="round,pad=0.15",
                               facecolor="#E3F2FD", edgecolor=COLORS["slametan"], linewidth=1.8)
    ax.add_patch(left_box)
    ax.text(2.2, 3.35, "RITUAL TIMESCALE", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COLORS["slametan"])
    ax.text(2.2, 2.9, "~1,000 days (~2.7 years)", ha="center", va="center",
            fontsize=9.5, color=COLORS["slametan"])

    ritual_lines = [
        "Andosol pH 4.5–5.5",
        "Body → dust",
        "Slametan tracks decomposition",
        "3 → 7 → 40 → 100 → 1000 days",
    ]
    for i, line in enumerate(ritual_lines):
        ax.text(2.2, 2.45 - i * 0.25, line, ha="center", va="center",
                fontsize=8, color="#333")

    # Right box: Geological timescale
    right_box = FancyBboxPatch((5.9, 1.5), 3.8, 2.2, boxstyle="round,pad=0.15",
                                facecolor="#FFF3E0", edgecolor=COLORS["geo"], linewidth=1.8)
    ax.add_patch(right_box)
    ax.text(7.8, 3.35, "GEOLOGICAL TIMESCALE", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COLORS["geo"])
    ax.text(7.8, 2.9, "~1,000 years", ha="center", va="center",
            fontsize=9.5, color=COLORS["geo"])

    geo_lines = [
        "Sedimentation 3.5–7 mm/yr",
        "Site → buried >3.5 m",
        "Invisible to surface survey",
        "Sambisari: 6.5 m in ~1000 yr",
    ]
    for i, line in enumerate(geo_lines):
        ax.text(7.8, 2.45 - i * 0.25, line, ha="center", va="center",
                fontsize=8, color="#333")

    # Arrows from volcano to both boxes
    arrow_props = dict(arrowstyle="-|>", color=COLORS["volc"], lw=2.5,
                       connectionstyle="arc3,rad=0.15")
    ax.annotate("", xy=(2.2, 3.7), xytext=(4.2, 4.3), arrowprops=arrow_props)
    ax.annotate("", xy=(7.8, 3.7), xytext=(5.8, 4.3), arrowprops=arrow_props)

    # Left arrow label
    ax.text(2.6, 4.15, "Acidic soil\n(chemical weathering)",
            ha="center", va="center", fontsize=7.5, color=COLORS["volc"],
            fontstyle="italic", rotation=20)
    # Right arrow label
    ax.text(7.4, 4.15, "Tephra deposition\n(sedimentation)",
            ha="center", va="center", fontsize=7.5, color=COLORS["volc"],
            fontstyle="italic", rotation=-20)

    # Bottom: convergence statement
    conv_box = FancyBboxPatch((1.5, 0.3), 7, 0.9, boxstyle="round,pad=0.12",
                               facecolor="#F3E5F5", edgecolor="#7B1FA2", linewidth=1.5)
    ax.add_patch(conv_box)
    ax.text(5, 0.9, "The environment that calibrates the ritual clock",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#4A148C")
    ax.text(5, 0.55, "also destroys the material evidence of the people who kept it.",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#4A148C")

    # Arrows from both boxes to convergence
    ax.annotate("", xy=(3, 1.2), xytext=(2.2, 1.5),
                arrowprops=dict(arrowstyle="-|>", color="#7B1FA2", lw=1.5))
    ax.annotate("", xy=(7, 1.2), xytext=(7.8, 1.5),
                arrowprops=dict(arrowstyle="-|>", color="#7B1FA2", lw=1.5))

    fig.tight_layout()
    fig.savefig(OUT / "fig3_htom_synthesis.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_htom_synthesis.png", bbox_inches="tight", dpi=300)
    print("  Fig 3 saved.")
    plt.close(fig)


# ================================================================
if __name__ == "__main__":
    print("Generating P5 figures...")
    fig1_mapping()
    fig2_monte_carlo()
    fig3_htom()
    print("Done. All figures saved to:", OUT)
