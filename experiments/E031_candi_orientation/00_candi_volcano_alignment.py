#!/usr/bin/env python3
"""
E031 — Candi Orientation vs Volcanic Peak Alignment

Question: Are Javanese candi (Hindu-Buddhist temples) preferentially sited
          in specific directions and distances relative to volcanic peaks?
          Do entrance orientations (where known) correlate with volcano direction?

Analyses:
  A. Distance from each candi to nearest volcano
  B. Azimuth (bearing) from each candi to nearest volcano — rose diagram
  C. Compiled entrance orientations (~20 candi) vs volcano azimuth
  D. Penanggungan cluster analysis (densest candi concentration)

Data: sites.csv (135 candi), volcanoes.csv (7+extended), literature orientations
Author: VOLCARCH project (AI-assisted)
Date: 2026-03-10
"""

import sys
import io
import os
import json
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("E031 — Candi Orientation vs Volcanic Peak Alignment")
print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════════════════

print("\n[1] Loading data...")

# Sites (filter candi)
sites_csv = os.path.join(REPO, "data", "processed", "dashboard", "sites.csv")
df_all = pd.read_csv(sites_csv)
df_candi = df_all[df_all['name'].str.contains('Candi|candi', na=False)].copy()
print(f"  Total sites: {len(df_all)}, Candi: {len(df_candi)}")

# Volcanoes (extend with important peaks missing from dashboard)
volc_csv = os.path.join(REPO, "data", "processed", "dashboard", "volcanoes.csv")
df_volc = pd.read_csv(volc_csv)

# Add important Java volcanoes not in dashboard CSV
extra_volc = pd.DataFrame([
    {'name': 'Merapi', 'lat': -7.540, 'lon': 110.446},
    {'name': 'Merbabu', 'lat': -7.455, 'lon': 110.437},
    {'name': 'Lawu', 'lat': -7.625, 'lon': 111.192},
    {'name': 'Penanggungan', 'lat': -7.614, 'lon': 112.622},
    {'name': 'Wilis', 'lat': -7.808, 'lon': 111.758},
    {'name': 'Dieng', 'lat': -7.210, 'lon': 109.910},
    {'name': 'Sundoro', 'lat': -7.300, 'lon': 109.992},
    {'name': 'Sumbing', 'lat': -7.384, 'lon': 110.070},
    {'name': 'Ungaran', 'lat': -7.180, 'lon': 110.330},
])
df_volc = pd.concat([df_volc, extra_volc], ignore_index=True)
df_volc = df_volc.drop_duplicates(subset='name')
print(f"  Volcanoes: {len(df_volc)} ({', '.join(df_volc['name'].values)})")


# ═════════════════════════════════════════════════════════════════════════
# 2. GIS COMPUTATIONS
# ═════════════════════════════════════════════════════════════════════════

print("\n[2] Computing distances and azimuths...")


def _compass(deg):
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ix = round(deg / 45) % 8
    return dirs[ix]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2 (degrees, 0=N, 90=E)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# For each candi, find nearest volcano
results_list = []
for _, candi in df_candi.iterrows():
    best_dist = float('inf')
    best_volc = None
    best_az_to = None
    best_az_from = None

    for _, volc in df_volc.iterrows():
        d = haversine_km(candi['lat'], candi['lon'], volc['lat'], volc['lon'])
        if d < best_dist:
            best_dist = d
            best_volc = volc['name']
            # Azimuth FROM candi TO volcano
            best_az_to = bearing_deg(candi['lat'], candi['lon'], volc['lat'], volc['lon'])
            # Azimuth FROM volcano TO candi
            best_az_from = bearing_deg(volc['lat'], volc['lon'], candi['lat'], candi['lon'])

    results_list.append({
        'name': candi['name'],
        'lat': candi['lat'],
        'lon': candi['lon'],
        'zone': candi.get('zone', ''),
        'nearest_volcano': best_volc,
        'distance_km': best_dist,
        'azimuth_to_volcano': best_az_to,
        'azimuth_from_volcano': best_az_from,
    })

df_r = pd.DataFrame(results_list)
print(f"  Computed {len(df_r)} candi-volcano pairs")

# Distance stats
print(f"\n  Distance to nearest volcano:")
print(f"    Mean: {df_r['distance_km'].mean():.1f} km")
print(f"    Median: {df_r['distance_km'].median():.1f} km")
print(f"    Min: {df_r['distance_km'].min():.1f} km ({df_r.loc[df_r['distance_km'].idxmin(), 'name']})")
print(f"    Max: {df_r['distance_km'].max():.1f} km ({df_r.loc[df_r['distance_km'].idxmax(), 'name']})")
print(f"    <10 km: {(df_r['distance_km'] < 10).sum()} candi")
print(f"    <20 km: {(df_r['distance_km'] < 20).sum()} candi")
print(f"    <30 km: {(df_r['distance_km'] < 30).sum()} candi")

# Nearest volcano distribution
print(f"\n  Candi per nearest volcano:")
for v, count in df_r['nearest_volcano'].value_counts().items():
    median_d = df_r[df_r['nearest_volcano'] == v]['distance_km'].median()
    print(f"    {v}: {count} candi (median {median_d:.1f} km)")


# ═════════════════════════════════════════════════════════════════════════
# 3. ANALYSIS A: Azimuthal Distribution (from volcano perspective)
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[3] ANALYSIS A: Azimuthal Distribution")
print("=" * 70)

# Test: are candi uniformly distributed around volcanoes, or do they cluster?
# "From volcano to candi" — which direction are candi from the volcano?
az_from = df_r['azimuth_from_volcano'].values
angles_rad = np.radians(az_from)

C = np.mean(np.cos(angles_rad))
S = np.mean(np.sin(angles_rad))
R_bar = np.sqrt(C**2 + S**2)
mean_angle = np.degrees(np.arctan2(S, C)) % 360
Z = len(az_from) * R_bar**2
p_rayleigh = np.exp(-Z)

print(f"\n  All candi (n={len(az_from)}):")
print(f"    Mean direction (from volcano): {mean_angle:.1f}° "
      f"({_compass(mean_angle)})")
print(f"    R-bar: {R_bar:.4f}")
print(f"    Rayleigh Z={Z:.2f}, p={p_rayleigh:.4e}")

# Quadrant analysis
quadrants = {'N (315-45)': 0, 'E (45-135)': 0, 'S (135-225)': 0, 'W (225-315)': 0}
for az in az_from:
    if az >= 315 or az < 45:
        quadrants['N (315-45)'] += 1
    elif 45 <= az < 135:
        quadrants['E (45-135)'] += 1
    elif 135 <= az < 225:
        quadrants['S (135-225)'] += 1
    else:
        quadrants['W (225-315)'] += 1

print(f"\n  Quadrant distribution (candi direction FROM volcano):")
expected_q = len(az_from) / 4
for q, count in quadrants.items():
    ratio = count / expected_q
    print(f"    {q}: {count} ({ratio:.2f}x expected)")

# Chi-squared quadrant test
chi2_q, p_q = stats.chisquare(list(quadrants.values()))
print(f"\n  Quadrant chi-squared: chi2={chi2_q:.2f}, p={p_q:.4f}")

# Per-volcano azimuthal analysis (for major clusters)
print(f"\n  Per-volcano azimuthal patterns:")
for volc_name in df_r['nearest_volcano'].value_counts().head(5).index:
    sub = df_r[df_r['nearest_volcano'] == volc_name]
    if len(sub) < 3:
        continue
    az_sub = np.radians(sub['azimuth_from_volcano'].values)
    C_v = np.mean(np.cos(az_sub))
    S_v = np.mean(np.sin(az_sub))
    R_v = np.sqrt(C_v**2 + S_v**2)
    mean_v = np.degrees(np.arctan2(S_v, C_v)) % 360
    Z_v = len(sub) * R_v**2
    p_v = np.exp(-Z_v)
    print(f"    {volc_name} (n={len(sub)}): mean={mean_v:.0f}° ({_compass(mean_v)}), "
          f"R={R_v:.3f}, p={p_v:.4f}"
          f"{' ← SIGNIFICANT' if p_v < 0.05 else ''}")


# ═════════════════════════════════════════════════════════════════════════
# 4. ANALYSIS B: Known Entrance Orientations vs Volcano Azimuth
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[4] ANALYSIS B: Entrance Orientation vs Volcano Direction")
print("=" * 70)

# Compiled from archaeological literature (entrance facing direction)
# Sources: Dumarçay (1993), Soekmono (1995), Degroot (2009), BPCB reports
KNOWN_ORIENTATIONS = [
    # Central Java (Mataram period)
    {'name': 'Borobudur', 'lat': -7.608, 'lon': 110.204, 'entrance_az': 90,
     'note': 'Main entrance faces East (Hindu convention)'},
    {'name': 'Prambanan', 'lat': -7.752, 'lon': 110.491, 'entrance_az': 90,
     'note': 'Main entrance faces East'},
    {'name': 'Mendut', 'lat': -7.605, 'lon': 110.231, 'entrance_az': 270,
     'note': 'Faces West toward Borobudur'},
    {'name': 'Pawon', 'lat': -7.605, 'lon': 110.215, 'entrance_az': 270,
     'note': 'Faces West'},
    {'name': 'Kalasan', 'lat': -7.767, 'lon': 110.472, 'entrance_az': 90,
     'note': 'East entrance'},
    {'name': 'Sewu', 'lat': -7.745, 'lon': 110.492, 'entrance_az': 90,
     'note': 'Buddhist, East entrance'},
    {'name': 'Plaosan', 'lat': -7.740, 'lon': 110.508, 'entrance_az': 90,
     'note': 'East entrance'},
    {'name': 'Sambisari', 'lat': -7.752, 'lon': 110.436, 'entrance_az': 270,
     'note': 'Buried temple, faces West'},
    {'name': 'Banyunibo', 'lat': -7.768, 'lon': 110.528, 'entrance_az': 270,
     'note': 'Buddhist, faces West'},

    # East Java
    {'name': 'Candi Badut', 'lat': -7.958, 'lon': 112.599, 'entrance_az': 270,
     'note': 'Oldest E. Java, faces West toward Arjuno'},
    {'name': 'Candi Kidal', 'lat': -8.020, 'lon': 112.617, 'entrance_az': 270,
     'note': 'Singosari period, faces West'},
    {'name': 'Candi Singosari', 'lat': -7.889, 'lon': 112.641, 'entrance_az': 180,
     'note': 'Faces South (unusual)'},
    {'name': 'Candi Jawi', 'lat': -7.662, 'lon': 112.670, 'entrance_az': 270,
     'note': 'Faces West toward Penanggungan'},
    {'name': 'Candi Penataran', 'lat': -7.985, 'lon': 112.208, 'entrance_az': 270,
     'note': 'Largest E. Java, faces West toward Kelud'},
    {'name': 'Candi Surawana', 'lat': -7.769, 'lon': 112.009, 'entrance_az': 270,
     'note': 'Majapahit, faces West'},
    {'name': 'Candi Jabung', 'lat': -7.706, 'lon': 113.420, 'entrance_az': 270,
     'note': 'Majapahit, faces West'},
    {'name': 'Candi Jolotundo', 'lat': -7.610, 'lon': 112.596, 'entrance_az': 270,
     'note': 'Penanggungan slope, bathing place, faces West'},
    {'name': 'Candi Tikus', 'lat': -7.572, 'lon': 112.404, 'entrance_az': 0,
     'note': 'Trowulan, bathing place, entrance North'},
    {'name': 'Candi Bajang Ratu', 'lat': -7.568, 'lon': 112.399, 'entrance_az': 0,
     'note': 'Trowulan gate, faces North'},

    # Dieng Plateau
    {'name': 'Arjuna group (Dieng)', 'lat': -7.210, 'lon': 109.910, 'entrance_az': 270,
     'note': 'Most Dieng temples face West'},
]

df_orient = pd.DataFrame(KNOWN_ORIENTATIONS)
print(f"  Compiled orientations: {len(df_orient)} candi")

# For each, compute azimuth to nearest volcano
orient_results = []
for _, row in df_orient.iterrows():
    best_dist = float('inf')
    best_volc = None
    best_az = None

    for _, volc in df_volc.iterrows():
        d = haversine_km(row['lat'], row['lon'], volc['lat'], volc['lon'])
        if d < best_dist:
            best_dist = d
            best_volc = volc['name']
            best_az = bearing_deg(row['lat'], row['lon'], volc['lat'], volc['lon'])

    # Angular difference between entrance and volcano direction
    entrance = row['entrance_az']
    diff = abs(entrance - best_az)
    if diff > 180:
        diff = 360 - diff
    # Does entrance FACE the volcano (diff < 90) or AWAY (diff > 90)?
    faces_volcano = diff < 90

    orient_results.append({
        'name': row['name'],
        'entrance_az': entrance,
        'volcano_az': best_az,
        'nearest_volcano': best_volc,
        'distance_km': best_dist,
        'angular_diff': diff,
        'faces_volcano': faces_volcano,
        'note': row.get('note', '')
    })

df_ov = pd.DataFrame(orient_results)

print(f"\n  {'Name':<25} {'Entrance':>8} {'Volcano':>8} {'Diff':>6} {'Faces?':>8} {'Nearest':<15}")
print("  " + "-" * 80)
for _, row in df_ov.iterrows():
    facing = 'YES' if row['faces_volcano'] else 'no'
    print(f"  {row['name']:<25} {row['entrance_az']:>6.0f}° {row['volcano_az']:>6.0f}° "
          f"{row['angular_diff']:>5.0f}° {facing:>8} {row['nearest_volcano']:<15}")

n_faces = df_ov['faces_volcano'].sum()
n_total_o = len(df_ov)
pct_faces = n_faces / n_total_o * 100

print(f"\n  Faces volcano: {n_faces}/{n_total_o} ({pct_faces:.0f}%)")
print(f"  Expected by chance (uniform): 50%")

# Binomial test: is proportion > 50%?
btest = stats.binomtest(n_faces, n_total_o, 0.5, alternative='greater')
print(f"  Binomial test (>50%): p={btest.pvalue:.4f}")

# Mean angular difference
mean_diff = df_ov['angular_diff'].mean()
median_diff = df_ov['angular_diff'].median()
print(f"\n  Angular difference (entrance vs volcano):")
print(f"    Mean: {mean_diff:.1f}°")
print(f"    Median: {median_diff:.1f}°")
print(f"    Expected if random: 90°")

# One-sample test: is mean diff < 90?
t_stat, t_p = stats.ttest_1samp(df_ov['angular_diff'], 90)
print(f"    t-test vs 90°: t={t_stat:.2f}, p={t_p:.4f} (two-sided)")

# Separate Central vs East Java
cj = df_ov[df_ov['nearest_volcano'].isin(['Merapi', 'Merbabu', 'Dieng', 'Sundoro',
                                            'Sumbing', 'Ungaran'])]
ej = df_ov[~df_ov.index.isin(cj.index)]

print(f"\n  Central Java (n={len(cj)}):")
print(f"    Faces volcano: {cj['faces_volcano'].sum()}/{len(cj)} ({cj['faces_volcano'].mean()*100:.0f}%)")
print(f"    Mean angular diff: {cj['angular_diff'].mean():.1f}°")

print(f"\n  East Java (n={len(ej)}):")
print(f"    Faces volcano: {ej['faces_volcano'].sum()}/{len(ej)} ({ej['faces_volcano'].mean()*100:.0f}%)")
print(f"    Mean angular diff: {ej['angular_diff'].mean():.1f}°")


# ═════════════════════════════════════════════════════════════════════════
# 5. ANALYSIS C: Penanggungan Cluster
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[5] ANALYSIS C: Penanggungan Cluster")
print("=" * 70)

# Penanggungan is Java's most temple-rich mountain
penan = df_r[df_r['nearest_volcano'] == 'Penanggungan']
if len(penan) > 0:
    print(f"\n  Penanggungan candi: {len(penan)}")
    print(f"  Distance range: {penan['distance_km'].min():.1f}-{penan['distance_km'].max():.1f} km")

    # Azimuthal distribution around Penanggungan
    az_pen = penan['azimuth_from_volcano'].values
    angles_pen = np.radians(az_pen)
    C_p = np.mean(np.cos(angles_pen))
    S_p = np.mean(np.sin(angles_pen))
    R_p = np.sqrt(C_p**2 + S_p**2)
    mean_p = np.degrees(np.arctan2(S_p, C_p)) % 360
    Z_p = len(penan) * R_p**2
    p_p = np.exp(-Z_p)

    print(f"  Mean direction: {mean_p:.0f}° ({_compass(mean_p)})")
    print(f"  R-bar: {R_p:.4f}")
    print(f"  Rayleigh Z={Z_p:.2f}, p={p_p:.4e}")

    # Quadrants
    q_pen = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
    for az in az_pen:
        if az >= 315 or az < 45: q_pen['N'] += 1
        elif 45 <= az < 135: q_pen['E'] += 1
        elif 135 <= az < 225: q_pen['S'] += 1
        else: q_pen['W'] += 1
    for q, c in q_pen.items():
        print(f"    {q}: {c} candi")
else:
    print("  No candi assigned to Penanggungan")


# ═════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[6] Generating visualizations...")
print("=" * 70)

fig = plt.figure(figsize=(16, 10))
fig.suptitle('E031 — Candi Orientation vs Volcanic Peak Alignment\n'
             'Do Javanese Temples Face Their Volcanoes?',
             fontsize=14, fontweight='bold', y=0.98)

# ── Panel A: Map of candi and volcanoes ───────────────────────────────────
ax1 = fig.add_subplot(2, 2, 1)

# Plot candi
ax1.scatter(df_r['lon'], df_r['lat'], s=10, alpha=0.5, color='#2ecc71',
            label=f'Candi (n={len(df_r)})', zorder=2)
# Plot known-orientation candi
ax1.scatter(df_orient['lon'], df_orient['lat'], s=40, marker='D',
            edgecolors='#e74c3c', facecolors='none', linewidth=1.5,
            label=f'Known orientation (n={len(df_orient)})', zorder=3)
# Plot volcanoes
ax1.scatter(df_volc['lon'], df_volc['lat'], s=100, marker='^',
            color='red', zorder=4, label='Volcanoes')
for _, v in df_volc.iterrows():
    ax1.annotate(v['name'], (v['lon'], v['lat']), fontsize=6,
                 xytext=(3, 3), textcoords='offset points', color='red')

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('A. Candi and Volcano Locations', fontsize=11)
ax1.legend(fontsize=7, loc='lower left')

# ── Panel B: Distance histogram ───────────────────────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
ax2.hist(df_r['distance_km'], bins=30, color='#3498db', edgecolor='white',
         alpha=0.8)
ax2.axvline(df_r['distance_km'].median(), color='red', linestyle='--',
            label=f'Median: {df_r["distance_km"].median():.1f} km')
ax2.set_xlabel('Distance to Nearest Volcano (km)')
ax2.set_ylabel('Number of Candi')
ax2.set_title('B. Distance to Nearest Volcanic Peak', fontsize=11)
ax2.legend(fontsize=8)

# ── Panel C: Rose diagram (azimuth from volcano to candi) ────────────────
ax3 = fig.add_subplot(2, 2, 3, projection='polar')
theta = np.radians(az_from)
# 16-bin rose
n_bins = 16
bins = np.linspace(0, 2*np.pi, n_bins + 1)
counts_rose, _ = np.histogram(theta, bins=bins)
width_rose = 2 * np.pi / n_bins
centers = (bins[:-1] + bins[1:]) / 2
ax3.bar(centers, counts_rose, width=width_rose * 0.9, alpha=0.7,
        color='#2ecc71', edgecolor='white')
ax3.set_title(f'C. Candi Direction FROM Volcano\n(R={R_bar:.3f}, p={p_rayleigh:.3e})',
              fontsize=11, pad=20)
# Cardinal labels
ax3.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
ax3.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=8)

# ── Panel D: Entrance orientation vs volcano direction ────────────────────
ax4 = fig.add_subplot(2, 2, 4)

# Scatter: x = volcano azimuth, y = entrance azimuth
ax4.scatter(df_ov['volcano_az'], df_ov['entrance_az'], s=60,
            c=df_ov['faces_volcano'].map({True: '#2ecc71', False: '#e74c3c'}),
            edgecolors='gray', linewidth=0.5, zorder=3)

# Perfect alignment line
ax4.plot([0, 360], [0, 360], 'k--', alpha=0.3, label='Perfect alignment')
ax4.plot([0, 360], [180, 540], 'k:', alpha=0.2)  # opposite

# Annotate each point
for _, row in df_ov.iterrows():
    short_name = row['name'].replace('Candi ', '').replace(' group (Dieng)', '')[:10]
    ax4.annotate(short_name, (row['volcano_az'], row['entrance_az']),
                 fontsize=5, alpha=0.7, xytext=(2, 2), textcoords='offset points')

ax4.set_xlabel('Azimuth TO Nearest Volcano (°)', fontsize=10)
ax4.set_ylabel('Entrance Orientation (°)', fontsize=10)
ax4.set_title(f'D. Entrance vs Volcano Direction\n(green=faces volcano, '
              f'{n_faces}/{n_total_o}={pct_faces:.0f}%)', fontsize=11)
ax4.set_xlim(0, 360)
ax4.set_ylim(0, 360)
ax4.set_aspect('equal')
ax4.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(RESULTS_DIR, 'candi_volcano_4panel.png'), dpi=150,
            bbox_inches='tight')
print("  Saved: candi_volcano_4panel.png")

# ── Standalone headline figure ────────────────────────────────────────────
fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5),
                                   subplot_kw={'projection': None})

# Left: entrance vs volcano scatter
ax_a.scatter(df_ov['volcano_az'], df_ov['entrance_az'], s=80,
             c=df_ov['faces_volcano'].map({True: '#2ecc71', False: '#e74c3c'}),
             edgecolors='gray', linewidth=0.5, zorder=3)
ax_a.plot([0, 360], [0, 360], 'k--', alpha=0.3, label='Perfect alignment')
for _, row in df_ov.iterrows():
    sn = row['name'].replace('Candi ', '').replace(' group (Dieng)', '')
    ax_a.annotate(sn, (row['volcano_az'], row['entrance_az']),
                  fontsize=7, alpha=0.8, xytext=(3, 3), textcoords='offset points')
ax_a.set_xlabel('Azimuth to Nearest Volcano (°)', fontsize=11)
ax_a.set_ylabel('Entrance Orientation (°)', fontsize=11)
ax_a.set_title(f'Entrance Direction vs Volcano Direction\n'
               f'Green = faces volcano ({n_faces}/{n_total_o} = {pct_faces:.0f}%)',
               fontsize=12)
ax_a.set_xlim(0, 360)
ax_a.set_ylim(0, 360)
ax_a.set_aspect('equal')

# Right: angular difference histogram
ax_b.hist(df_ov['angular_diff'], bins=9, range=(0, 180),
          color='#3498db', edgecolor='white', alpha=0.8)
ax_b.axvline(90, color='red', linestyle='--', alpha=0.5, label='Random expectation (90°)')
ax_b.axvline(df_ov['angular_diff'].median(), color='#2ecc71', linestyle='-',
             linewidth=2, label=f'Median: {df_ov["angular_diff"].median():.0f}°')
ax_b.set_xlabel('Angular Difference: Entrance vs Volcano (°)', fontsize=11)
ax_b.set_ylabel('Number of Candi', fontsize=11)
ax_b.set_title('Distribution of Entrance-Volcano Alignment\n(0° = faces volcano, 180° = faces away)',
               fontsize=12)
ax_b.legend(fontsize=9)

fig2.suptitle('E031 — Do Javanese Temples Face Their Volcanoes?',
              fontsize=13, fontweight='bold', y=1.02)
fig2.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, 'candi_orientation_headline.png'), dpi=200,
             bbox_inches='tight')
print("  Saved: candi_orientation_headline.png")
plt.close('all')


# ═════════════════════════════════════════════════════════════════════════
# 7. STRUCTURED OUTPUT
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[7] Saving results...")
print("=" * 70)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

summary = {
    'experiment': 'E031_candi_orientation',
    'n_candi_geocoded': len(df_r),
    'n_candi_with_orientation': len(df_ov),
    'n_volcanoes': len(df_volc),

    'siting_analysis': {
        'distance_mean_km': round(df_r['distance_km'].mean(), 1),
        'distance_median_km': round(df_r['distance_km'].median(), 1),
        'rayleigh_R': round(R_bar, 4),
        'rayleigh_p': round(float(p_rayleigh), 6),
        'quadrant_chi2': round(chi2_q, 2),
        'quadrant_p': round(p_q, 4),
    },

    'orientation_analysis': {
        'faces_volcano_count': int(n_faces),
        'total': int(n_total_o),
        'pct_faces_volcano': round(pct_faces, 1),
        'binomial_p': round(btest.pvalue, 4),
        'mean_angular_diff': round(mean_diff, 1),
        'median_angular_diff': round(median_diff, 1),
    },
}

with open(os.path.join(RESULTS_DIR, 'alignment_summary.json'), 'w',
          encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

# Save detailed candi-volcano pairs
df_r.to_csv(os.path.join(RESULTS_DIR, 'candi_volcano_pairs.csv'), index=False)
df_ov.to_csv(os.path.join(RESULTS_DIR, 'orientation_vs_volcano.csv'), index=False)

print("  Saved: alignment_summary.json")
print("  Saved: candi_volcano_pairs.csv")
print("  Saved: orientation_vs_volcano.csv")


# ═════════════════════════════════════════════════════════════════════════
# 8. HEADLINE
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HEADLINE FINDING")
print("=" * 70)

print(f"""
  SITING: {len(df_r)} candi, median {df_r['distance_km'].median():.1f} km from nearest volcano
  Quadrant test: chi2={chi2_q:.2f}, p={p_q:.4f}
  Azimuthal clustering: R={R_bar:.4f}, Rayleigh p={p_rayleigh:.4e}

  ORIENTATION: {n_faces}/{n_total_o} ({pct_faces:.0f}%) candi entrances face their volcano
  Binomial test (>50%): p={btest.pvalue:.4f}
  Mean angular difference: {mean_diff:.1f}° (random = 90°)
  Median angular difference: {median_diff:.1f}°

  Central Java: {cj['faces_volcano'].sum()}/{len(cj)} face volcano (mostly East = Hindu convention)
  East Java: {ej['faces_volcano'].sum()}/{len(ej)} face volcano (mostly West = toward highlands)
""")

print("=" * 70)
print("E031 COMPLETE")
print("=" * 70)
