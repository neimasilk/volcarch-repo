#!/usr/bin/env python3
"""
E216 Figure: Java palaeoecological core network + RSAP circles + heartland gap.
Produces figures/fig1_network_rsap_map.png
"""
import sys, math
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Circle
    import numpy as np
except ImportError:
    print("matplotlib not available — skipping figure generation")
    sys.exit(0)

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Core data (from e216_detection_function.py)
CORES = [
    dict(id='J1', name='Dieng', lat=-7.20, lon=109.90, rsap_km=8,  ctrl=True,  color='#2ca02c'),
    dict(id='J2', name='Rawa Danau', lat=-6.20, lon=105.90, rsap_km=25, ctrl=True, color='#2ca02c'),
    dict(id='J3', name='Teluk Banten', lat=-6.00, lon=106.10, rsap_km=200, ctrl=False, color='#1f77b4'),
    dict(id='J4', name='Bandung', lat=-6.90, lon=107.60, rsap_km=35, ctrl=False, color='#1f77b4'),
    dict(id='J5', name='Bayongbong', lat=-7.10, lon=107.80, rsap_km=7,  ctrl=False, color='#1f77b4'),
    dict(id='J6', name='Solo Marine', lat=-6.50, lon=112.00, rsap_km=400, ctrl=True, color='#ff7f0e'),
    dict(id='J7', name='Song Gupuh', lat=-8.00, lon=110.50, rsap_km=4,  ctrl=False, color='#1f77b4'),
]

# Heartlands
KEDU    = dict(lat=-7.50, lon=110.00, name='Kedu/Prambanan')
BRANTAS = dict(lat=-7.80, lon=112.00, name='Brantas/Kediri')

# Java island bounding box
JAVA_BOUNDS = dict(lon_min=105.0, lon_max=115.5, lat_min=-9.0, lat_max=-5.5)

# Degree to km conversion (approximate, for Java latitude)
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * math.cos(math.radians(-7.5))

fig, ax = plt.subplots(figsize=(14, 6))

# Java background
ax.set_facecolor('#e8f4f8')
ax.set_xlim(JAVA_BOUNDS['lon_min'], JAVA_BOUNDS['lon_max'])
ax.set_ylim(JAVA_BOUNDS['lat_min'], JAVA_BOUNDS['lat_max'])

# Draw RSAP circles (only terrestrial/small ones — marine too large to show meaningfully)
for c in CORES:
    if c['rsap_km'] > 50:  # skip very large marine RSAPs
        continue
    rsap_deg_lon = c['rsap_km'] / KM_PER_DEG_LON
    rsap_deg_lat = c['rsap_km'] / KM_PER_DEG_LAT
    # Use ellipse to account for lat/lon distortion
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(
        (c['lon'], c['lat']),
        width=rsap_deg_lon * 2,
        height=rsap_deg_lat * 2,
        fill=True,
        facecolor=c['color'],
        edgecolor=c['color'],
        alpha=0.15,
        linewidth=1.5
    )
    ax.add_patch(ellipse)
    ellipse_border = Ellipse(
        (c['lon'], c['lat']),
        width=rsap_deg_lon * 2,
        height=rsap_deg_lat * 2,
        fill=False,
        edgecolor=c['color'],
        alpha=0.4,
        linewidth=1.0,
        linestyle='--'
    )
    ax.add_patch(ellipse_border)

# Plot cores
for c in CORES:
    marker = '★' if c['ctrl'] else 'o'
    msize = 120 if c['ctrl'] else 60
    mshape = '*' if c['ctrl'] else 'o'
    ax.scatter(c['lon'], c['lat'], s=msize, c=c['color'], zorder=5,
               marker=mshape, edgecolors='black', linewidths=0.5)
    # Label offset
    offset_lon, offset_lat = 0.15, 0.08
    if c['id'] == 'J2': offset_lon, offset_lat = -1.8, -0.15
    if c['id'] == 'J3': offset_lon, offset_lat = -1.2, 0.1
    if c['id'] == 'J6': offset_lon, offset_lat = 0.1, 0.12
    ax.text(c['lon'] + offset_lon, c['lat'] + offset_lat,
            f"{c['id']}\n{c['name']}", fontsize=7, ha='left', va='bottom',
            color=c['color'], fontweight='bold' if c['ctrl'] else 'normal')

# Heartland markers
for h in [KEDU, BRANTAS]:
    ax.scatter(h['lon'], h['lat'], s=200, c='#d62728', zorder=6, marker='X',
               edgecolors='darkred', linewidths=1.0)
    ax.text(h['lon'] + 0.1, h['lat'] - 0.18, h['name'], fontsize=8,
            color='#d62728', fontweight='bold')

# Heartland gap annotation
ax.annotate(
    '← GAP: no core\nwithin RSAP of\nKedu/Brantas',
    xy=(110.5, -7.5), xytext=(112.5, -6.2),
    fontsize=8, color='#d62728',
    arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeeee', edgecolor='#d62728', alpha=0.8)
)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#2ca02c', alpha=0.4, label='Core with positive control (clearance recorded)'),
    mpatches.Patch(facecolor='#1f77b4', alpha=0.4, label='Core without positive control'),
    mpatches.Patch(facecolor='#ff7f0e', alpha=0.4, label='Marine core (hedged positive control)'),
    plt.scatter([], [], s=150, c='none', marker='*', edgecolors='#2ca02c', label='Core w/ +ctrl'),
    plt.scatter([], [], s=80, c='none', marker='o', edgecolors='#1f77b4', label='Core w/o +ctrl'),
    plt.scatter([], [], s=200, c='#d62728', marker='X', label='Inscription heartland (Kedu/Brantas)'),
]
ax.legend(handles=legend_elements[:], loc='lower left', fontsize=7, framealpha=0.9)

# Shaded circles indicate RSAP sizes
ax.text(106.5, -8.5, 'Circles = RSAP radius\n(pollen source area)', fontsize=7,
        style='italic', color='gray')

ax.set_xlabel('Longitude (°E)', fontsize=10)
ax.set_ylabel('Latitude (°S)', fontsize=10)
ax.set_title(
    'E216: Java Palaeoecological Core Network — RSAP Coverage Gap at Inscription Heartlands\n'
    'Instrument IS sensitive (Dieng +ctrl ~600 CE, Rawa Danau +ctrl ~AD 1770) '
    'but NO core has RSAP covering Kedu/Brantas → OUTCOME-3',
    fontsize=9, pad=10
)
ax.invert_yaxis()
ax.grid(True, alpha=0.2, linestyle=':')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_network_rsap_map.png', dpi=200, bbox_inches='tight')
print(f"Figure saved: {FIG_DIR / 'fig1_network_rsap_map.png'}")

# ── Figure 2: Detection probability vs population size ────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

import importlib, sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from e216_detection_function import run_detection_analysis, E196_FLOOR, E196_CENTRAL

pop_range = [100_000, 200_000, 400_000, 631_059, 1_000_000, 1_270_000, 2_000_000, 3_000_000]
p_net_A = []
p_net_B = []
p_kedu_core_A = []  # hypothetical core at Kedu

CONCENTRATION_FACTOR = 4.0
HEARTLAND_AREA_KM2 = math.pi * 50**2
JAVA_AREA_KM2 = 129_000

def p_at_kedu_hypothetical(N, mode='A'):
    """P(detect) for a hypothetical core at Kedu with RSAP 10 km."""
    from e216_detection_function import pop_to_cleared_km2, detect_prob
    _, total_mid, _ = pop_to_cleared_km2(N, mode)
    java_density = total_mid / JAVA_AREA_KM2
    heartland_density = min(java_density * CONCENTRATION_FACTOR, 1.0)
    RSAP_KM2 = math.pi * 10**2
    cleared_in_rsap = heartland_density * RSAP_KM2
    return detect_prob(cleared_in_rsap, RSAP_KM2)

for N in pop_range:
    _, p_a = run_detection_analysis(N, 'A')
    _, p_b = run_detection_analysis(N, 'B')
    p_net_A.append(p_a)
    p_net_B.append(p_b)
    p_kedu_core_A.append(p_at_kedu_hypothetical(N, 'A'))

ax2.semilogx(pop_range, p_net_A, 'b-o', label='Mode A (clearing), existing network', linewidth=2)
ax2.semilogx(pop_range, p_net_B, 'b--s', label='Mode B (dispersed), existing network', linewidth=1.5, alpha=0.7)
ax2.semilogx(pop_range, p_kedu_core_A, 'r-^', label='Mode A, hypothetical core AT Kedu', linewidth=2)

ax2.axhline(0.90, color='gray', linestyle=':', linewidth=1.5, label='Pre-registered threshold C=0.90')
ax2.axvline(E196_FLOOR,   color='darkgreen', linestyle='--', alpha=0.7, linewidth=1.2)
ax2.axvline(E196_CENTRAL, color='green',     linestyle='--', alpha=0.7, linewidth=1.2)
ax2.text(E196_FLOOR, 0.05, f'N_floor\n{E196_FLOOR//1000}k', fontsize=7, color='darkgreen', ha='center')
ax2.text(E196_CENTRAL, 0.05, f'N_central\n{E196_CENTRAL//1000}k', fontsize=7, color='green', ha='center')

ax2.set_xlabel('Population N at 400 CE', fontsize=10)
ax2.set_ylabel('P(network detects | N, mode)', fontsize=10)
ax2.set_title(
    'E216: Detection Power Surface\n'
    'Existing Java network: P≈0 at all plausible N. '
    'Hypothetical Kedu core: P≈1.0 at N≥200k.',
    fontsize=9
)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_detection_power.png', dpi=200, bbox_inches='tight')
print(f"Figure saved: {FIG_DIR / 'fig2_detection_power.png'}")
