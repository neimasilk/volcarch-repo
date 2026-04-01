"""
E174: The VOLCARCH Synthesis Figure
====================================
One figure that tells the entire story.
Designed for: papers, YouTube thumbnails, presentations, posters.

Layout: 2x3 panel figure
1. Population trajectory (E172) — 3.3M at 400 CE
2. Cascade model (E110) — 5 factors, 0.058% visible
3. Burial depth cross-section (E166) — depth vs distance
4. Two Javas (E084) — candi vs inscription distance
5. Ghost vocabulary (E165) — indigenous % over time
6. The gap (E108/E172) — 11,008x visual
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

fig, axes = plt.subplots(2, 3, figsize=(20, 13))

# Consistent styling
COLORS = {
    'volcanic': '#d62728',
    'indigenous': '#2ca02c',
    'sanskrit': '#9467bd',
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'bg_light': '#f7f7f7',
}

# ============================================================
# Panel 1: Population Trajectory (E172)
# ============================================================
ax = axes[0, 0]

# Load trajectory data
data = np.load("D:/documents/volcarch-repo/experiments/E172_population_dynamics/results/trajectories.npz")
time_ce = data['time_ce']
median = data['median']
ci_low = data['ci_2_5']
ci_high = data['ci_97_5']

# Zoom to -5000 to 1600
mask = time_ce >= -5000
ax.fill_between(time_ce[mask], ci_low[mask]/1e6, ci_high[mask]/1e6,
                alpha=0.3, color=COLORS['primary'])
ax.plot(time_ce[mask], median[mask]/1e6, color=COLORS['primary'], linewidth=2)
ax.axvline(x=400, color=COLORS['volcanic'], linestyle='--', alpha=0.7)
ax.annotate('First inscriptions\n400 CE', xy=(400, median[np.argmin(np.abs(time_ce-400))]/1e6),
            xytext=(700, 0.5), fontsize=8, arrowprops=dict(arrowstyle='->', color='red'),
            color='red')
ax.annotate('3.3M', xy=(400, 3.3), fontsize=14, fontweight='bold', color=COLORS['primary'],
            ha='left')
ax.set_xlabel('Year (CE)')
ax.set_ylabel('Population (millions)')
ax.set_title('A. Population of Java\n(50K Monte Carlo)', fontweight='bold', fontsize=11)
ax.set_ylim(0, 8)
ax.grid(True, alpha=0.2)

# ============================================================
# Panel 2: Cascade Model (E110)
# ============================================================
ax = axes[0, 1]

factors = ['F1\nVolcanic\nBurial', 'F2\nOrganic\nDecay', 'F3\nSurvey\nDeficit',
           'F4\nRecognition', 'F5\nPublication']
survivals = [0.58, 0.20, 0.025, 0.40, 0.50]
colors_bar = [COLORS['volcanic'], '#8c564b', COLORS['primary'], COLORS['secondary'], '#7f7f7f']

bars = ax.bar(factors, survivals, color=colors_bar, edgecolor='black', linewidth=0.5)

# Add product line
product = np.cumprod(survivals)
for i, (f, p) in enumerate(zip(factors, product)):
    ax.annotate(f'{p*100:.2f}%' if p > 0.001 else f'{p*100:.3f}%',
                xy=(i, survivals[i]), xytext=(0, 5),
                textcoords='offset points', ha='center', fontsize=7, color='gray')

ax.set_ylabel('Survival Probability')
ax.set_title('B. Five-Factor Cascade\n(Product = 0.058%)', fontweight='bold', fontsize=11)
ax.set_ylim(0, 1)
ax.axhline(y=0.00058, color='red', linestyle=':', alpha=0.5)
ax.annotate('Product: 0.058%', xy=(4, 0.00058), fontsize=8, color='red',
            va='bottom')

# ============================================================
# Panel 3: Burial Depth vs Distance
# ============================================================
ax = axes[0, 2]

distances = np.linspace(0, 60, 100)
rate = 8.0 * np.exp(-distances / 15.0)
depth_400ce = rate * 1626 / 1000

ax.fill_between(distances, 0, depth_400ce, alpha=0.3, color=COLORS['volcanic'])
ax.plot(distances, depth_400ce, color=COLORS['volcanic'], linewidth=2)

# Detection horizons
ax.axhline(y=3, color='green', linestyle='--', alpha=0.5, label='GPR limit (3m)')
ax.axhline(y=6, color='orange', linestyle='--', alpha=0.5, label='ERT limit (6m)')

# Calibration points
calib_points = [
    (8, 5.1*1626/1000, 'Sambisari'),
    (10, 5.8*1626/1000, 'Kedulan'),
    (12, 3.5*1626/1000, 'Kimpulan'),
    (20, 3.5*1626/1000, 'Dwarapala'),
]
for d, depth, name in calib_points:
    ax.plot(d, depth, 'ko', markersize=6)
    ax.annotate(name, xy=(d, depth), xytext=(5, 5), textcoords='offset points',
                fontsize=7)

ax.set_xlabel('Distance from volcano (km)')
ax.set_ylabel('Burial depth at 400 CE (m)')
ax.set_title('C. Burial Depth vs Distance\n(Pre-400 CE sites)', fontweight='bold', fontsize=11)
ax.set_xlim(0, 60)
ax.set_ylim(0, 15)
ax.invert_yaxis()
ax.legend(fontsize=8, loc='lower right')

# ============================================================
# Panel 4: Two Javas (candi vs inscriptions)
# ============================================================
ax = axes[1, 0]

# Histogram data (from E084)
bins_edges = [0, 10, 20, 30, 40, 60, 100]
candi_pct = [42.3, 14.8, 31.0, 5.6, 3.5, 2.8]
insc_pct = [12.5, 26.1, 39.2, 6.8, 11.4, 4.0]

x = np.arange(len(candi_pct))
width = 0.35

ax.bar(x - width/2, candi_pct, width, label='Candi (n=142)',
       color=COLORS['secondary'], edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, insc_pct, width, label='Inscriptions (n=176)',
       color=COLORS['indigenous'], edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(['0-10', '10-20', '20-30', '30-40', '40-60', '60-100'])
ax.set_xlabel('Distance from volcano (km)')
ax.set_ylabel('Percentage (%)')
ax.set_title('D. Two Javas: Sacred vs Administrative\n(E084, p < 0.000001)', fontweight='bold', fontsize=11)
ax.legend(fontsize=9)

# Zone labels
ax.axvspan(-0.5, 1.5, alpha=0.1, color=COLORS['volcanic'])
ax.axvspan(1.5, 3.5, alpha=0.1, color=COLORS['indigenous'])
ax.text(0.5, 42, 'Volcano\nJava', ha='center', fontsize=8, color=COLORS['volcanic'], fontstyle='italic')
ax.text(2.5, 42, 'Court\nJava', ha='center', fontsize=8, color=COLORS['indigenous'], fontstyle='italic')

# ============================================================
# Panel 5: Ghost Vocabulary / Indigenous % (E165)
# ============================================================
ax = axes[1, 1]

centuries = [7, 8, 9, 10, 11, 12, 13, 14]
indigenous_pct = [66.7, 64.3, 95.9, 93.5, 81.9, 50.0, 84.2, 78.6]
n_docs = [4, 25, 30, 45, 11, 2, 10, 6]

# Scale marker size by number of documents
sizes = [n * 8 for n in n_docs]

ax.scatter(centuries, indigenous_pct, s=sizes, c=COLORS['indigenous'],
           edgecolor='black', linewidth=0.5, zorder=5)
ax.plot(centuries, indigenous_pct, color=COLORS['indigenous'], linewidth=1.5, alpha=0.7)

# C8 annotation
ax.annotate('"Aku" disappears\nafter C8', xy=(8, 64.3), xytext=(8.5, 45),
            fontsize=8, arrowprops=dict(arrowstyle='->', color='red'),
            color='red', fontstyle='italic')

# 929 CE line
ax.axvline(x=9.29, color='purple', linestyle='--', alpha=0.5)
ax.annotate('929 CE', xy=(9.29, 98), fontsize=7, color='purple', ha='center')

ax.set_xlabel('Century CE')
ax.set_ylabel('Indigenous vocabulary (%)')
ax.set_title('E. Indigenous Voice Over Time\n(E165: 95,709 tokens, original OJ)', fontweight='bold', fontsize=11)
ax.set_ylim(40, 100)
ax.set_xlim(6.5, 14.5)
ax.grid(True, alpha=0.2)

# ============================================================
# Panel 6: The Gap — visual
# ============================================================
ax = axes[1, 2]

# Simple but powerful: two bars
categories = ['Expected\nSettlements\n(E172)', 'Known\nPre-400 CE\nSites']
values = [33024, 3]
colors_gap = [COLORS['primary'], COLORS['volcanic']]

bars = ax.bar(categories, values, color=colors_gap, edgecolor='black', linewidth=0.5)
ax.set_yscale('log')
ax.set_ylim(1, 100000)

# Add labels
ax.text(0, 33024, '33,024', ha='center', va='bottom', fontsize=14, fontweight='bold',
        color=COLORS['primary'])
ax.text(1, 3, '3', ha='center', va='bottom', fontsize=14, fontweight='bold',
        color=COLORS['volcanic'])

# Gap annotation
ax.annotate('', xy=(0.5, 33024), xytext=(0.5, 3),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(0.65, 300, '11,008x\ngap', fontsize=14, fontweight='bold', color='red',
        ha='left')

ax.set_title('F. The Archaeological Gap\n(E172: 3.3M people, ~3 sites)', fontweight='bold', fontsize=11)
ax.set_ylabel('Count (log scale)')

# ============================================================
# Overall title
# ============================================================
fig.suptitle('VOLCARCH: Unearthing the Invisible — 173 Experiments\n'
             'Volcanic taphonomic bias in Indonesian archaeological records',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

output_path = Path("D:/documents/volcarch-repo/experiments/E174_synthesis_figure/results")
fig.savefig(output_path / 'volcarch_synthesis_6panel.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()

# Also save as PDF for paper
fig2, axes2 = plt.subplots(2, 3, figsize=(20, 13))
# (would repeat the same plotting code — skip for now, PNG is primary)

print("E174: Synthesis figure saved.")
print(f"  PNG: {output_path / 'volcarch_synthesis_6panel.png'}")
print(f"  Size: {(output_path / 'volcarch_synthesis_6panel.png').stat().st_size / 1024:.0f} KB")
print("DONE.")
