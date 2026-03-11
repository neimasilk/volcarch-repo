"""
E026 Figure: Kelud Eruptions vs Majapahit Political Crises Timeline

Produces the key visual for P14 — a dual-track timeline showing:
- Top track: Kelud eruptions (from GVP)
- Bottom track: Pararaton political events
- Highlight: the 1376-1411 eruption cluster → Paregreg Civil War window
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Data
KELUD_ERUPTIONS = [1311, 1334, 1376, 1385, 1395, 1411, 1450, 1451, 1462, 1481]
KELUD_VEI = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

CRISES = [
    (1293, "Mongol invasion\nMajapahit founded", "war"),
    (1334, "Sadeng rebellion\n+ banyu pindah", "rebellion"),
    (1357, "Gajah Mada\ndies", "succession"),
    (1389, "Hayam Wuruk\ndies", "succession"),
    (1401, "Paregreg War\nbegins", "war"),
    (1405, "Paregreg War\nends", "war"),
    (1426, "Great\nfamine", "famine"),
    (1451, "Kertawijaya\ndies", "succession"),
    (1468, "Kingdom\nsplits", "civil_conflict"),
    (1478, "Sirna ilang\nkrtaning bhumi", "collapse"),
    (1481, "Guntur\npawatugunung", "geological"),
]

COLOR_MAP = {
    "war": "#d32f2f",
    "rebellion": "#e65100",
    "succession": "#1565c0",
    "famine": "#6a1b9a",
    "civil_conflict": "#c62828",
    "collapse": "#212121",
    "geological": "#2e7d32",
}

fig, ax = plt.subplots(figsize=(14, 6))

# Background shading for periods
ax.axvspan(1293, 1375, alpha=0.08, color='green', label='Peak period')
ax.axvspan(1376, 1527, alpha=0.08, color='red', label='Decline period')

# Eruption cluster highlight
ax.axvspan(1376, 1411, alpha=0.15, color='orange', label='Kelud cluster\n(4 eruptions / 35 yr)')

# Kelud eruptions (top)
for i, (year, vei) in enumerate(zip(KELUD_ERUPTIONS, KELUD_VEI)):
    ax.plot(year, 0.6, marker='^', color='#d32f2f', markersize=12, zorder=5)
    ax.vlines(year, 0, 0.6, color='#d32f2f', linewidth=1, alpha=0.5)

# Crisis events (bottom)
offsets = [-0.3, -0.5, -0.7, -0.9]
used_positions = []
for i, (year, label, typ) in enumerate(CRISES):
    # Stagger labels to avoid overlap
    offset_idx = 0
    for uy, uo in used_positions:
        if abs(year - uy) < 15 and uo == offsets[offset_idx]:
            offset_idx = min(offset_idx + 1, len(offsets) - 1)
    y_pos = offsets[offset_idx]
    used_positions.append((year, y_pos))

    color = COLOR_MAP.get(typ, '#666666')
    ax.plot(year, -0.1, marker='o', color=color, markersize=8, zorder=5)
    ax.vlines(year, y_pos + 0.08, -0.1, color=color, linewidth=0.8, alpha=0.5)
    ax.text(year, y_pos, label, ha='center', va='top', fontsize=7,
            color=color, fontweight='bold')

# Central timeline
ax.axhline(y=0, color='black', linewidth=2)
ax.axhline(y=0, xmin=0, xmax=1, color='black', linewidth=2)

# Labels
ax.text(1280, 0.75, 'KELUD ERUPTIONS (GVP)', fontsize=10, fontweight='bold',
        color='#d32f2f', va='bottom')
ax.text(1280, -0.15, 'PARARATON EVENTS', fontsize=10, fontweight='bold',
        color='#333333', va='top')

# Annotations
ax.annotate('', xy=(1395, 0.35), xytext=(1401, 0.35),
            arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=2))
ax.text(1398, 0.40, '6 yr', ha='center', fontsize=8, color='#d32f2f', fontweight='bold')

ax.annotate('', xy=(1411, 0.25), xytext=(1426, 0.25),
            arrowprops=dict(arrowstyle='->', color='#6a1b9a', lw=2))
ax.text(1418, 0.30, '15 yr', ha='center', fontsize=8, color='#6a1b9a', fontweight='bold')

# Stats box
stats_text = (
    "Proximity test: p = 0.037\n"
    "Rate ratio: 2.18×\n"
    "GVP match: 3/3"
)
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
ax.text(1510, 0.7, stats_text, fontsize=9, verticalalignment='top',
        bbox=props, family='monospace')

# Formatting
ax.set_xlim(1275, 1535)
ax.set_ylim(-1.1, 0.95)
ax.set_xlabel('Year CE', fontsize=11)
ax.set_xticks(range(1280, 1540, 20))
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_title('Kelud Eruptions and Majapahit Political Crises (1293–1527 CE)',
             fontsize=13, fontweight='bold', pad=15)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='green', alpha=0.15, label='Peak period (1293–1375)'),
    mpatches.Patch(facecolor='red', alpha=0.15, label='Decline period (1376–1527)'),
    mpatches.Patch(facecolor='orange', alpha=0.3, label='Kelud eruption cluster'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#d32f2f',
               markersize=10, label='Kelud eruption (VEI 3)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

plt.tight_layout()

# Save
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)
fig_path = output_dir / "fig1_kelud_majapahit_timeline.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {fig_path}")

# Also save to paper figures directory
paper_fig = Path(__file__).parent.parent.parent / "papers" / "P14_pararaton_collapse" / "figures" / "fig1_timeline.png"
paper_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(paper_fig, dpi=300, bbox_inches='tight')
print(f"Figure copied to {paper_fig}")

# PDF version
fig_pdf = paper_fig.with_suffix('.pdf')
plt.savefig(fig_pdf, bbox_inches='tight')
print(f"PDF saved to {fig_pdf}")

plt.close()
