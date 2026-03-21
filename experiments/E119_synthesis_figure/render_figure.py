"""
E119: Render the VOLCARCH Synthesis Figure

Publication-quality figure showing burial depth vs. time,
with detection horizons and known archaeological sites.

Usage: python render_figure.py
Output: results/e119_synthesis_figure.png (300 DPI)
        results/e119_synthesis_figure.pdf (vector)
"""

import json
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "results"

def load_data():
    with open(OUT / "e119_figure_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

def render(data):
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Burial depth diagonal (4 mm/yr) ---
    rate = data["sedimentation_rate_mm_yr"]

    # Time range for the figure: 2000 BCE to 2026 CE
    t_min, t_max = -2000, 2026
    times = np.linspace(t_min, t_max, 500)
    depths = (2026 - times) * rate / 1000  # depth in meters

    ax.plot(times, depths, color='#2c3e50', linewidth=2.5, zorder=5,
            label=f'Burial depth at {rate:.0f} mm/yr')

    # --- Sedimentation rate uncertainty band (2.4 - 6.2 mm/yr) ---
    rate_lo, rate_hi = 2.4, 6.2
    depths_lo = (2026 - times) * rate_lo / 1000
    depths_hi = (2026 - times) * rate_hi / 1000
    ax.fill_between(times, depths_lo, depths_hi, alpha=0.12, color='#2c3e50',
                    zorder=2, label='Rate range (2.4\u20136.2 mm/yr)')

    # --- Detection horizons (horizontal lines) ---
    method_colors = {
        'Surface survey':      ('#27ae60', 0.5,  'Surface survey (0.5 m)'),
        'Shallow excavation':  ('#f39c12', 1.0,  'Shallow excavation (1 m)'),
        'Standard excavation': ('#e67e22', 2.0,  'Standard excavation (2 m)'),
        'GPR':                 ('#e74c3c', 5.0,  'GPR in volcanic ash (5 m)'),
        'Deep coring':         ('#8b0000', 10.0, 'Deep coring (10 m)'),
    }

    for method, (color, depth_m, label) in method_colors.items():
        ax.axhline(y=depth_m, color=color, linewidth=1.5, linestyle='--',
                   alpha=0.7, zorder=3)
        # Label on the right side
        ax.text(t_max + 30, depth_m, label, va='center', ha='left',
                fontsize=7.5, color=color, fontweight='bold')

    # --- Invisible zone shading ---
    # Below the burial diagonal = invisible
    ax.fill_between(times, depths, 12, alpha=0.08, color='#c0392b',
                    zorder=1, label='Invisible zone')

    # Detectable zone above diagonal
    ax.fill_between(times, 0, depths, alpha=0.05, color='#27ae60', zorder=1)

    # --- Known sites ---
    site_styles = {
        'cave':                  {'marker': 'o', 'color': '#3498db', 'size': 70,
                                  'label': 'Cave site (immune to burial)'},
        'coastal':               {'marker': 's', 'color': '#1abc9c', 'size': 80,
                                  'label': 'Coastal site (non-volcanic)'},
        'open_air_nonvolcanic':  {'marker': 'D', 'color': '#9b59b6', 'size': 60,
                                  'label': 'Open-air non-volcanic (karst)'},
        'temple_found':          {'marker': '^', 'color': '#e74c3c', 'size': 80,
                                  'label': 'Temple (found buried)'},
    }

    # Track which types we've already labeled
    labeled_types = set()

    for site in data['known_sites']:
        century = site['century_ce']
        depth = site['depth']
        stype = site['type']

        # Skip sites outside our time range
        if century < t_min - 200:
            continue

        style = site_styles.get(stype, {'marker': 'x', 'color': 'gray', 'size': 50, 'label': stype})
        label = style['label'] if stype not in labeled_types else None
        labeled_types.add(stype)

        ax.scatter(century, depth, marker=style['marker'], c=style['color'],
                   s=style['size'], zorder=10, edgecolors='black', linewidths=0.5,
                   label=label)

        # Annotate site name
        # Offset to avoid overlaps
        offset_y = -0.5  # default: below
        offset_x = 40
        ha = 'left'
        fontsize = 6.5

        # Special positioning to avoid overlaps
        if site['name'] == 'Sambisari':
            offset_x = 120
            offset_y = 0.8
        elif site['name'] == 'Kedulan':
            offset_x = 120
            offset_y = -0.3
        elif site['name'] == 'Prambanan':
            offset_x = -120
            ha = 'right'
            offset_y = 0.8
        elif site['name'] == 'Borobudur':
            offset_y = 0.8
            offset_x = -100
            ha = 'right'
        elif site['name'] == 'Dieng':
            offset_x = -120
            ha = 'right'
            offset_y = -0.3
        elif site['name'] == 'Trowulan/Majapahit':
            offset_x = 100
            offset_y = 0.5
        elif site['name'] == 'Singosari Dwarapala':
            offset_x = -120
            ha = 'right'
            offset_y = -0.5
        elif site['name'] == 'Buni Complex':
            offset_y = -0.7
            offset_x = -80
            ha = 'right'
        elif site['name'] == 'Batujaya':
            offset_y = 0.7
            offset_x = 80
        elif site['name'] == 'Song Keplek':
            offset_y = 0.7
            offset_x = 100
        elif site['name'] == 'Kendeng Lembu':
            offset_y = -0.7
            offset_x = -100
            ha = 'right'

        ax.annotate(site['name'],
                    xy=(century, depth),
                    xytext=(century + offset_x, depth + offset_y),
                    fontsize=fontsize, color='#2c3e50',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                    ha=ha, va='center', zorder=11)

    # --- "Pre-400 CE Volcanic Interior: NOTHING" annotation ---
    ax.annotate('Pre-400 CE volcanic interior Java:\nZERO open-air sites',
                xy=(-500, 7.5), fontsize=10, fontweight='bold',
                color='#c0392b', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fadbd8',
                          edgecolor='#c0392b', alpha=0.9),
                zorder=12)

    # --- Key intersection labels on diagonal ---
    intersections = [
        (1901, 0.5,  '~1900 CE', -250, -0.3),
        (1526, 2.0,  '~1526 CE', -300, 0.0),
        (776,  5.0,  '~776 CE',  -300, 0.0),
        (-474, 10.0, '~474 BCE', -300, 0.0),
    ]
    for yr, d, label, dx, dy in intersections:
        if t_min <= yr <= t_max:
            ax.plot(yr, d, 'ko', markersize=4, zorder=8)
            ax.annotate(label, xy=(yr, d),
                        xytext=(yr + dx, d + dy),
                        fontsize=7, fontstyle='italic', color='#555555',
                        arrowprops=dict(arrowstyle='->', color='#999999', lw=0.7),
                        zorder=8)

    # --- Formatting ---
    ax.set_xlim(t_min - 200, t_max + 50)
    ax.set_ylim(12, -0.8)  # Inverted: depth increases downward
    ax.set_xlabel('Time (CE / BCE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Burial Depth (m)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Why Pre-400 CE Open-Air Sites Are Invisible in Volcanic Java\n'
        'Burial Depth vs. Detection Horizons at 4 mm/yr Sedimentation',
        fontsize=13, fontweight='bold', pad=15
    )

    # X-axis: format centuries
    xticks = [-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000]
    xlabels = []
    for t in xticks:
        if t < 0:
            xlabels.append(f'{abs(t)} BCE')
        elif t == 0:
            xlabels.append('0')
        else:
            xlabels.append(f'{t} CE')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)

    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.tick_params(axis='y', labelsize=9)

    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.axvline(x=400, color='gray', linewidth=1, linestyle=':', alpha=0.5, zorder=2)
    ax.text(420, 11.5, '400 CE\n(Earliest Hindu\ninscriptions)', fontsize=7,
            color='gray', va='top', ha='left')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower left', fontsize=7.5, framealpha=0.9,
              ncol=2, borderaxespad=1)

    # Source note
    fig.text(0.02, 0.01,
             'VOLCARCH Project | Sedimentation rates: Dwarapala Singosari, Sambisari, Kedulan, Kimpulan (E083) | '
             'Detection limits: E092/E098 literature review',
             fontsize=6, color='gray', ha='left')

    plt.tight_layout(rect=[0, 0.03, 0.82, 1])  # Leave room for right-side labels

    # Save
    fig.savefig(OUT / 'e119_synthesis_figure.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    fig.savefig(OUT / 'e119_synthesis_figure.pdf', bbox_inches='tight',
                facecolor='white')
    plt.close()

    print(f"Figure saved:")
    print(f"  PNG: {OUT / 'e119_synthesis_figure.png'}")
    print(f"  PDF: {OUT / 'e119_synthesis_figure.pdf'}")


if __name__ == '__main__':
    render(load_data())
