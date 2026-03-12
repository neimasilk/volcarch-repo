"""
P11 Figure Generation — Volcanic Informedness
Generates 5 publication-quality figures for the paper.
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE, "figures")
E031 = os.path.join(BASE, "..", "..", "experiments", "E031_candi_orientation", "results")
E032 = os.path.join(BASE, "..", "..", "experiments", "E032_pranata_mangsa", "results")
E039 = os.path.join(BASE, "..", "..", "experiments", "E039_vcs_crosscultural", "results")
E065 = os.path.join(BASE, "..", "..", "experiments", "E065_candi_elevation_analysis", "results")

# Common style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# ===========================================================
# FIGURE 1: Polar plot of candi-volcano bearings (n=142)
# ===========================================================
def fig1_polar_bearings():
    print("Generating Figure 1: Candi-volcano bearing polar plot...")
    df = pd.read_csv(os.path.join(E031, "candi_volcano_pairs.csv"))

    # azimuth_from_volcano = bearing FROM volcano TO candi
    azimuths = df['azimuth_from_volcano'].values
    distances = df['distance_km'].values

    # Convert to radians (matplotlib polar: 0=N, clockwise)
    theta = np.deg2rad(azimuths)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Set 0=North, clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Color by Penanggungan vs others
    is_penang = df['nearest_volcano'] == 'Penanggungan'

    ax.scatter(theta[is_penang], distances[is_penang],
               c='#D62728', alpha=0.6, s=40, label=f'Penanggungan (n={is_penang.sum()})', zorder=3)
    ax.scatter(theta[~is_penang], distances[~is_penang],
               c='#1F77B4', alpha=0.6, s=40, label=f'Other volcanoes (n={(~is_penang).sum()})', zorder=3)

    # Mean direction arrow
    mean_az = np.deg2rad(279)  # from E065 results
    ax.annotate('', xy=(mean_az, ax.get_rmax()*0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
    ax.text(mean_az, ax.get_rmax()*0.95, 'Mean: 279°\n(West)',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # Shade western quadrant
    theta_fill = np.linspace(np.deg2rad(225), np.deg2rad(315), 50)
    ax.fill_between(theta_fill, 0, ax.get_rmax(), alpha=0.08, color='green', label='Western quadrant (47.2%)')

    # Shade eastern quadrant (nearly empty)
    theta_fill_e = np.linspace(np.deg2rad(45), np.deg2rad(135), 50)
    ax.fill_between(theta_fill_e, 0, ax.get_rmax(), alpha=0.08, color='red', label='Eastern quadrant (3.5%)')

    ax.set_title('Bearing from Volcano to Candi (n=142)\nRayleigh p = 3.4×10⁻⁸', pad=20, fontsize=14)
    ax.set_rlabel_position(45)
    ax.set_ylabel('Distance (km)', labelpad=30)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.15), fontsize=9, framealpha=0.9)

    # Cardinal labels
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    out = os.path.join(FIGURES, "fig1_candi_polar_bearings.png")
    fig.savefig(out)
    fig.savefig(out.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================
# FIGURE 2: Penanggungan detail with west-clustering
# ===========================================================
def fig2_penanggungan_detail():
    print("Generating Figure 2: Penanggungan detail map...")
    df = pd.read_csv(os.path.join(E031, "candi_volcano_pairs.csv"))
    penang = df[df['nearest_volcano'] == 'Penanggungan'].copy()

    # Penanggungan peak coordinates (approximate)
    peak_lat, peak_lon = -7.6117, 112.6142

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Scatter plot of candi around Penanggungan
    ax1.scatter(penang['lon'], penang['lat'], c='#D62728', s=30, alpha=0.7, zorder=3, label='Candi')
    ax1.scatter(peak_lon, peak_lat, c='black', s=200, marker='^', zorder=4, label='Penanggungan peak')

    # Draw quadrant lines through peak
    xlim_range = 0.12
    ax1.axhline(y=peak_lat, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.axvline(x=peak_lon, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    # Shade west side
    ax1.axvspan(peak_lon - xlim_range, peak_lon, alpha=0.08, color='green', label='West side')
    ax1.axvspan(peak_lon, peak_lon + xlim_range, alpha=0.08, color='red', label='East side')

    # Count annotations
    west_count = (penang['lon'] < peak_lon).sum()
    east_count = (penang['lon'] >= peak_lon).sum()
    ax1.text(peak_lon - 0.06, peak_lat + 0.05, f'West: {west_count}\n({west_count/len(penang)*100:.0f}%)',
             ha='center', fontsize=12, fontweight='bold', color='darkgreen',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.text(peak_lon + 0.06, peak_lat + 0.05, f'East: {east_count}\n({east_count/len(penang)*100:.0f}%)',
             ha='center', fontsize=12, fontweight='bold', color='darkred',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'Penanggungan Candi Distribution (n={len(penang)})\nBinomial p < 10⁻⁶ for western preference')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_aspect('equal')

    # Wind arrow (SE monsoon → NW)
    ax1.annotate('SE Monsoon\n→ tephra NW',
                 xy=(peak_lon + 0.08, peak_lat - 0.04),
                 fontsize=9, ha='center', color='brown',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Right: Quadrant bar chart for all 142 candi
    with open(os.path.join(E065, "e065_results.json")) as f:
        e065 = json.load(f)

    quads = e065['azimuthal_stats']['quadrants']
    labels = ['North', 'East', 'South', 'West']
    counts = [quads[q] for q in labels]
    colors = ['#7FB3D8', '#FF9999', '#7FB3D8', '#77DD77']
    expected = 142 / 4

    bars = ax2.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.8)
    ax2.axhline(y=expected, color='black', linestyle='--', linewidth=1, label=f'Expected ({expected:.0f})')

    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{count}\n({count/142*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Number of Candi')
    ax2.set_title('Quadrant Distribution (all Java, n=142)\nχ² = 54.68, p < 0.0001')
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, max(counts) * 1.25)

    fig.suptitle('Figure 2: Candi West-Clustering Evidence', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    out = os.path.join(FIGURES, "fig2_penanggungan_westclustering.png")
    fig.savefig(out)
    fig.savefig(out.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================
# FIGURE 3: Eruption seasonality × Pranata Mangsa overlay
# ===========================================================
def fig3_eruption_seasonality():
    print("Generating Figure 3: Eruption seasonality polar plot...")

    with open(os.path.join(E032, "seasonality_summary.json")) as f:
        data = json.load(f)

    mangsa_order = ['Kasa', 'Karo', 'Katelu', 'Kapat', 'Kalima', 'Kanem',
                    'Kapitu', 'Kawolu', 'Kasanga', 'Kasepuluh', 'Desta', 'Saddha']

    # Approximate Gregorian correspondence
    gregorian = ['Jun-Jul', 'Jul-Aug', 'Aug-Sep', 'Sep', 'Sep-Oct', 'Oct-Nov',
                 'Nov-Jan', 'Jan-Feb', 'Feb-Mar', 'Mar', 'Mar-Apr', 'Apr-Jun']

    densities = [data['mangsa_density'][m]['density_per_30d'] for m in mangsa_order]
    seasons = [data['mangsa_density'][m]['season'] for m in mangsa_order]
    counts = [data['mangsa_density'][m]['count'] for m in mangsa_order]

    # Colors by season
    season_colors = {'dry': '#FFD700', 'wet': '#4682B4', 'transition': '#90EE90'}
    bar_colors = [season_colors[s] for s in seasons]

    # Polar bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                     subplot_kw={'projection': 'polar'})
    # Oops, need to create them separately since only ax1 should be polar
    plt.close(fig)

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection='polar')
    ax2 = fig.add_subplot(122)

    # Left: Polar plot
    n = len(mangsa_order)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    width = 2*np.pi / n

    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)

    bars = ax1.bar(theta, densities, width=width*0.85, color=bar_colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85)

    # Labels
    ax1.set_xticks(theta)
    ax1.set_xticklabels([f'{m}\n({g})' for m, g in zip(mangsa_order, gregorian)], fontsize=7)

    # Highlight Kapitu (peak)
    kapitu_idx = mangsa_order.index('Kapitu')
    bars[kapitu_idx].set_edgecolor('red')
    bars[kapitu_idx].set_linewidth(2.5)
    ax1.text(theta[kapitu_idx], densities[kapitu_idx] + 1.5, 'PEAK\n18.1/30d',
             ha='center', fontsize=9, fontweight='bold', color='red')

    # Highlight Kapat (lowest)
    kapat_idx = mangsa_order.index('Kapat')
    ax1.text(theta[kapat_idx], densities[kapat_idx] + 2.5, 'Lowest\n4.8/30d',
             ha='center', fontsize=8, color='gray')

    ax1.set_title('Eruption Density per Pranata Mangsa Month\n(events per 30-day equivalent)',
                  pad=20, fontsize=11)

    # Legend for seasons
    legend_patches = [mpatches.Patch(color=c, label=s.capitalize())
                      for s, c in season_colors.items()]
    ax1.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(-0.2, -0.15), fontsize=9)

    # Right: Bar chart with ratio annotation
    x = np.arange(n)
    bars2 = ax2.bar(x, densities, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(mangsa_order, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Eruption density (per 30-day equivalent)')
    ax2.set_title(f'Chi-squared p = 0.042 | Rayleigh p = 0.032\nKapitu/Kapat ratio = 3.8×')

    # Highlight peak and trough
    bars2[kapitu_idx].set_edgecolor('red')
    bars2[kapitu_idx].set_linewidth(2.5)
    bars2[kapat_idx].set_edgecolor('gray')
    bars2[kapat_idx].set_linewidth(2.5)
    bars2[kapat_idx].set_linestyle('--')

    # 3.8x annotation
    ax2.annotate('3.8×', xy=(kapitu_idx, densities[kapitu_idx]),
                xytext=(kapitu_idx + 1.5, densities[kapitu_idx] + 1),
                fontsize=14, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Expected uniform line
    mean_density = np.mean(densities)
    ax2.axhline(y=mean_density, color='black', linestyle='--', linewidth=1,
                label=f'Mean ({mean_density:.1f})')
    ax2.legend(fontsize=9)

    fig.suptitle('Figure 3: Pranata Mangsa Encodes Volcanic Hazard Seasonality',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    out = os.path.join(FIGURES, "fig3_eruption_seasonality.png")
    fig.savefig(out)
    fig.savefig(out.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================
# FIGURE 4: Volcanic distance vs ritual complexity (116 societies)
# ===========================================================
def fig4_crosscultural_scatter():
    print("Generating Figure 4: Cross-cultural scatterplot...")

    df = pd.read_csv(os.path.join(E039, "culture_volcano_distances.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Distance vs ritual complexity
    ax1.scatter(df['distance_km'], df['ritual_complexity'],
                alpha=0.5, s=40, c='#1F77B4', edgecolors='black', linewidth=0.3)

    # Linear fit for reference
    mask = df['distance_km'].notna() & df['ritual_complexity'].notna()
    x = df.loc[mask, 'distance_km'].values
    y = df.loc[mask, 'ritual_complexity'].values
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax1.plot(xline, np.polyval(z, xline), 'r--', alpha=0.5, linewidth=1.5,
                label=f'ρ = +0.145, p = 0.092\n(opposite to prediction)')

    ax1.set_xlabel('Distance to Nearest Volcano (km)')
    ax1.set_ylabel('Ritual Complexity Score')
    ax1.set_title('E039b: Distance vs Ritual Complexity\n(116 Austronesian Societies)')
    ax1.legend(fontsize=10, loc='upper right')

    # Annotate: "No global pattern"
    ax1.text(0.5, 0.05, 'NO GLOBAL VOLCANIC-CULTURAL CORRELATION',
             transform=ax1.transAxes, ha='center', fontsize=10, fontweight='bold',
             color='red', alpha=0.7,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Right: Boxplot volcanic vs non-volcanic
    ritual_scores = pd.read_csv(os.path.join(E039, "culture_ritual_scores.csv"))
    volcanic = ritual_scores[ritual_scores['is_volcanic'] == 1]['ritual_complexity'].dropna()
    non_volcanic = ritual_scores[ritual_scores['is_volcanic'] == 0]['ritual_complexity'].dropna()

    bp = ax2.boxplot([volcanic.values, non_volcanic.values],
                     labels=['Volcanic\nIslands', 'Non-Volcanic\nIslands'],
                     patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#FF9999')
    bp['boxes'][1].set_facecolor('#99CCFF')

    ax2.set_ylabel('Ritual Complexity Score')
    ax2.set_title('E039a: Volcanic vs Non-Volcanic Islands\nMann-Whitney p = 0.973')

    # Means
    for i, (data, label) in enumerate([(volcanic, 'Volcanic'), (non_volcanic, 'Non-volcanic')]):
        ax2.text(i+1, data.mean() + 0.02, f'μ={data.mean():.3f}',
                ha='center', fontsize=10, fontweight='bold')

    ax2.text(0.5, 0.05, 'NO SIGNIFICANT DIFFERENCE (p = 0.973)',
             transform=ax2.transAxes, ha='center', fontsize=10, fontweight='bold',
             color='red', alpha=0.7,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Figure 4: Volcanic Informedness Is Local, Not Global',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    out = os.path.join(FIGURES, "fig4_crosscultural_falsification.png")
    fig.savefig(out)
    fig.savefig(out.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================
# FIGURE 5: Volcanic Informedness Feedback Loop (conceptual)
# ===========================================================
def fig5_feedback_loop():
    print("Generating Figure 5: Feedback loop diagram...")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # 5-node circular layout
    nodes = [
        ('Volcanic\nActivity', 0, 3.5, '#FF6B6B'),           # top
        ('Organic Material\nBurial & Destruction', 3.3, 1.1, '#FFA07A'),  # top-right
        ('Archaeological\n"Blank"', 2.0, -2.8, '#FFD700'),     # bottom-right
        ('External Explanation\n(Indianization)', -2.0, -2.8, '#87CEEB'),  # bottom-left
        ('Endogenous Adaptation\nOverlooked', -3.3, 1.1, '#98D8C8'),    # top-left
    ]

    box_w, box_h = 2.6, 1.4

    for label, x, y, color in nodes:
        bbox = dict(boxstyle=f'round,pad=0.4', facecolor=color,
                    edgecolor='black', linewidth=1.5, alpha=0.85)
        ax.text(x, y, label, ha='center', va='center', fontsize=11,
                fontweight='bold', bbox=bbox, zorder=3)

    # Arrows between consecutive nodes (clockwise)
    arrow_params = dict(arrowstyle='->', color='#333333', lw=2,
                        connectionstyle='arc3,rad=0.15')

    # Node centers for arrow routing
    nc = [(0, 3.5), (3.3, 1.1), (2.0, -2.8), (-2.0, -2.8), (-3.3, 1.1)]

    arrow_labels = [
        'Tephra buries\norganic evidence',
        'Creates perceived\npre-Hindu void',
        'Scholars seek\nexternal origins',
        'Volcanic landscape\nadaptation ignored',
        'But volcanoes also\nSHAPED culture',
    ]

    for i in range(5):
        j = (i + 1) % 5
        x1, y1 = nc[i]
        x2, y2 = nc[j]

        # Shorten arrows to not overlap boxes
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        shrink = 1.2 / dist

        sx1 = x1 + dx * shrink
        sy1 = y1 + dy * shrink
        sx2 = x2 - dx * shrink
        sy2 = y2 - dy * shrink

        ax.annotate('', xy=(sx2, sy2), xytext=(sx1, sy1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=2.5,
                                    connectionstyle='arc3,rad=0.2'),
                    zorder=2)

        # Label on arrow midpoint
        mx = (sx1 + sx2) / 2
        my = (sy1 + sy2) / 2
        # Offset label outward from center
        cx, cy = 0, 0.3  # center of diagram
        ox = mx - cx
        oy = my - cy
        od = np.sqrt(ox**2 + oy**2)
        if od > 0:
            ox, oy = ox/od * 0.7, oy/od * 0.7

        ax.text(mx + ox, my + oy, arrow_labels[i], ha='center', va='center',
                fontsize=8, style='italic', color='#555555',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))

    # Central "P11 breaks the loop" annotation
    ax.text(0, 0, 'P11: Volcanic\nInformedness\n\nBreaks the cycle by\nshowing volcanoes\nSHAPED culture,\nnot just buried it',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F8E8',
                      edgecolor='darkgreen', linewidth=2, alpha=0.95),
            zorder=5)

    ax.set_title('Figure 5: The VOLCARCH Feedback Loop\nVolcanic Informedness Closes the Conceptual Circuit',
                 fontsize=14, fontweight='bold', pad=20)

    out = os.path.join(FIGURES, "fig5_feedback_loop.png")
    fig.savefig(out)
    fig.savefig(out.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================
# MAIN
# ===========================================================
if __name__ == '__main__':
    os.makedirs(FIGURES, exist_ok=True)

    fig1_polar_bearings()
    fig2_penanggungan_detail()
    fig3_eruption_seasonality()
    fig4_crosscultural_scatter()
    fig5_feedback_loop()

    print(f"\n=== All 5 figures generated in {FIGURES} ===")
    for f in sorted(os.listdir(FIGURES)):
        fpath = os.path.join(FIGURES, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")
