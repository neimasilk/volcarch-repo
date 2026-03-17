"""P17 Figure Generation — Two Javas"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})
outdir = Path("papers/P17_two_javas/figures")
outdir.mkdir(exist_ok=True)

# Load data
candi = pd.read_csv("experiments/E031_candi_orientation/results/candi_volcano_pairs.csv")
geo = pd.read_csv("experiments/E082_inscription_georeferencing/results/geocoded_inscriptions.csv")
geo = geo[geo['lat'].notna()]

# === FIGURE 1: Dual histogram ===
fig, ax = plt.subplots(figsize=(9, 4.5))
bins = np.arange(0, 105, 5)
ax.hist(candi['distance_km'], bins=bins, alpha=0.6, color='#e65100', label=f'Candi (n={len(candi)})', density=True)
ax.hist(geo['volcano_dist_km'], bins=bins, alpha=0.6, color='#1565c0', label=f'Inscriptions (n={len(geo)})', density=True)
ax.axvline(14.6, color='#e65100', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(27.6, color='#1565c0', linestyle='--', linewidth=1, alpha=0.7)
ax.text(14.6, ax.get_ylim()[1]*0.9, 'Candi\nmedian\n14.6 km', ha='center', fontsize=7, color='#e65100')
ax.text(27.6, ax.get_ylim()[1]*0.75, 'Inscription\nmedian\n27.6 km', ha='center', fontsize=7, color='#1565c0')
# Zone shading
ax.axvspan(0, 15, alpha=0.08, color='red', label='Volcano zone')
ax.axvspan(15, 30, alpha=0.08, color='blue', label='Court zone')
ax.set_xlabel('Distance to Nearest Volcano (km)')
ax.set_ylabel('Density')
ax.set_title('Two Javas: Candi Cluster Near Volcanoes, Inscriptions in the Court Zone')
ax.legend(fontsize=8)
fig.savefig(outdir / 'fig1_dual_histogram.png')
fig.savefig(outdir / 'fig1_dual_histogram.pdf')
plt.close(fig)
print('fig1 done')

# === FIGURE 2: Elevation density ===
# Use precomputed results from E100
zones = ['Coastal\n(0-50m)', 'Lowland\n(50-200m)', 'Midslope\n(200-500m)', 'Highland\n(500-1000m)', 'Mountain\n(>1000m)']
densities = [1.96, 4.31, 4.99, 7.91, 18.61]
colors = ['#90caf9', '#64b5f6', '#42a5f5', '#1e88e5', '#0d47a1']

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(range(len(zones)), densities, color=colors, edgecolor='white', width=0.7)
for bar, d in zip(bars, densities):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{d:.1f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(len(zones)))
ax.set_xticklabels(zones, fontsize=9)
ax.set_ylabel('Sites per 1000 km$^2$')
ax.set_title('Archaeological Site Density by Elevation Zone (E. Java)')
ax.annotate('9.5x', xy=(4, 18.61), xytext=(3.3, 16), fontsize=11, fontweight='bold', color='#0d47a1',
            arrowprops=dict(arrowstyle='->', color='#0d47a1'))
fig.savefig(outdir / 'fig2_elevation_density.png')
fig.savefig(outdir / 'fig2_elevation_density.pdf')
plt.close(fig)
print('fig2 done')

# === FIGURE 3: Depth-binned vocabulary ===
depth_zones = ['Shallow\n(0-2m)', 'Medium\n(2-5m)', 'Deep\n(5-10m)']
indigenous_pct = [9.3, 53.8, 56.4]
sanskrit_pct = [90.7, 46.2, 43.6]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = range(len(depth_zones))
w = 0.35
ax.bar([i-w/2 for i in x], indigenous_pct, w, color='#4caf50', label='Indigenous', edgecolor='white')
ax.bar([i+w/2 for i in x], sanskrit_pct, w, color='#ff9800', label='Sanskrit', edgecolor='white')
for i, (ind, san) in enumerate(zip(indigenous_pct, sanskrit_pct)):
    ax.text(i-w/2, ind+1.5, f'{ind:.1f}%', ha='center', fontsize=9)
    ax.text(i+w/2, san+1.5, f'{san:.1f}%', ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(depth_zones)
ax.set_ylabel('Vocabulary Proportion (%)')
ax.set_title('Vocabulary Composition by Burial Depth Zone')
ax.legend()
ax.annotate('5.8x jump', xy=(0.5, 30), xytext=(0.8, 70), fontsize=10, fontweight='bold', color='#4caf50',
            arrowprops=dict(arrowstyle='->', color='#4caf50'))
fig.savefig(outdir / 'fig3_depth_vocabulary.png')
fig.savefig(outdir / 'fig3_depth_vocabulary.pdf')
plt.close(fig)
print('fig3 done')

# === FIGURE 4: Temporal trend by zone ===
dated = pd.read_csv("experiments/E030_prasasti_temporal_nlp/results/dated_inscriptions.csv")
merged = dated.merge(geo[['filename','volcano_dist_km']], on='filename', how='inner')
merged = merged[merged['pre_indic_ratio'].notna() & merged['year_ce'].notna()]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
zone_specs = [
    ('Volcano (<20km)', merged[merged['volcano_dist_km'] < 20], '#e65100'),
    ('Court (20-40km)', merged[(merged['volcano_dist_km'] >= 20) & (merged['volcano_dist_km'] < 40)], '#1565c0'),
    ('Periphery (>40km)', merged[merged['volcano_dist_km'] >= 40], '#2e7d32'),
]

for ax, (title, data, color) in zip(axes, zone_specs):
    if len(data) > 5:
        ax.scatter(data['year_ce'], data['pre_indic_ratio'], c=color, s=20, alpha=0.6)
        rho, p = stats.spearmanr(data['year_ce'], data['pre_indic_ratio'])
        z = np.polyfit(data['year_ce'], data['pre_indic_ratio'], 1)
        xs = np.linspace(data['year_ce'].min(), data['year_ce'].max(), 100)
        ax.plot(xs, np.polyval(z, xs), color=color, linewidth=2, alpha=0.7)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS'
        ax.set_title(f'{title}\n$\\rho$={rho:.3f} ({sig})', fontsize=10)
    else:
        ax.set_title(f'{title}\n(n={len(data)})')
    ax.set_xlabel('Year CE')
    ax.axvline(929, color='gray', linestyle=':', alpha=0.5)
    ax.text(929, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 0.5, '929', fontsize=7, ha='right', color='gray')

axes[0].set_ylabel('Pre-Indic Vocabulary Ratio')
plt.suptitle('Pre-Indic Vocabulary Trend by Volcanic Distance Zone', fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(outdir / 'fig4_temporal_by_zone.png')
fig.savefig(outdir / 'fig4_temporal_by_zone.pdf')
plt.close(fig)
print('fig4 done')

# === FIGURE 5: Pre/Post-929 zone × topic ===
merged['topic'] = pd.cut(merged['pre_indic_ratio'], bins=[-0.01, 0.05, 0.20, 1.0],
                          labels=['Sanskrit', 'Mixed', 'Indigenous'])
merged['era'] = merged['year_ce'].apply(lambda y: 'Pre-929' if y < 929 else 'Post-929')
merged['zone'] = merged['volcano_dist_km'].apply(lambda d: 'Volcano' if d < 15 else 'Court' if d < 30 else 'Periphery')

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, era in zip(axes, ['Pre-929', 'Post-929']):
    sub = merged[merged['era'] == era]
    ct = pd.crosstab(sub['zone'], sub['topic'])
    ct = ct.reindex(['Volcano', 'Court', 'Periphery'])
    ct = ct.reindex(columns=['Sanskrit', 'Mixed', 'Indigenous'])
    ct = ct.fillna(0)
    ct.plot(kind='bar', stacked=True, ax=ax, color=['#ff9800', '#9e9e9e', '#4caf50'], edgecolor='white')
    ax.set_title(era, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Inscriptions')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(fontsize=8)

plt.suptitle('Vocabulary Composition by Zone: Before and After 929 CE', fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(outdir / 'fig5_pre_post_929_zones.png')
fig.savefig(outdir / 'fig5_pre_post_929_zones.pdf')
plt.close(fig)
print('fig5 done')

print(f'\nAll figures saved to {outdir}')
