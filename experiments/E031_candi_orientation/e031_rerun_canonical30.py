#!/usr/bin/env python3
"""
E031 RE-RUN (2026-06-10) — canonical 30-volcano inventory.

Why: the original run used dashboard volcanoes.csv (7, eastern E. Java only)
plus 9 hardcoded peaks = 16. The P7/Antiquity rejection exposed that truncated
inventory as the source of the false "deep-time sites 90-170 km from volcanoes"
artifact. Per ME#18 integrity sweep, every volcano-distance experiment must be
re-pointed to data/processed/dashboard/volcanoes_java_full.csv (30 volcanoes).

This re-runs the numeric analyses of 00_candi_volcano_alignment.py (siting,
azimuthal distribution, entrance orientation) with the canonical inventory.
Outputs go to results/canonical30/ so the original results stay intact for
comparison. No figures (numbers are what feed papers).
"""

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "canonical30")
os.makedirs(OUT, exist_ok=True)

df_all = pd.read_csv(os.path.join(REPO, "data", "processed", "dashboard", "sites.csv"))
df_candi = df_all[df_all['name'].str.contains('Candi|candi', na=False)].copy()

df_volc = pd.read_csv(os.path.join(REPO, "data", "processed", "dashboard",
                                   "volcanoes_java_full.csv"))
print(f"Candi: {len(df_candi)}, Volcanoes (canonical): {len(df_volc)}")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def nearest(lat, lon):
    best = (float('inf'), None, None, None)
    for _, v in df_volc.iterrows():
        d = haversine_km(lat, lon, v['lat'], v['lon'])
        if d < best[0]:
            best = (d, v['name'],
                    bearing_deg(lat, lon, v['lat'], v['lon']),
                    bearing_deg(v['lat'], v['lon'], lat, lon))
    return best


rows = []
for _, c in df_candi.iterrows():
    d, vname, az_to, az_from = nearest(c['lat'], c['lon'])
    rows.append({'name': c['name'], 'lat': c['lat'], 'lon': c['lon'],
                 'zone': c.get('zone', ''), 'nearest_volcano': vname,
                 'distance_km': d, 'azimuth_to_volcano': az_to,
                 'azimuth_from_volcano': az_from})
df_r = pd.DataFrame(rows)

print(f"\nSITING (n={len(df_r)}):")
print(f"  mean {df_r['distance_km'].mean():.1f} km, median {df_r['distance_km'].median():.1f} km")
for cut in (10, 20, 30):
    print(f"  <{cut} km: {(df_r['distance_km'] < cut).sum()}")

az_from = df_r['azimuth_from_volcano'].values
ang = np.radians(az_from)
C, S = np.mean(np.cos(ang)), np.mean(np.sin(ang))
R_bar = np.sqrt(C**2 + S**2)
mean_angle = np.degrees(np.arctan2(S, C)) % 360
Z = len(az_from) * R_bar**2
p_rayleigh = np.exp(-Z)
print(f"\nAZIMUTH: mean dir {mean_angle:.1f} deg, R={R_bar:.4f}, Rayleigh p={p_rayleigh:.3e}")

quad = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
for a in az_from:
    if a >= 315 or a < 45: quad['N'] += 1
    elif a < 135: quad['E'] += 1
    elif a < 225: quad['S'] += 1
    else: quad['W'] += 1
chi2_q, p_q = stats.chisquare(list(quad.values()))
print(f"  quadrants {quad}, chi2={chi2_q:.2f}, p={p_q:.4f}")

# Entrance orientations: same literature compilation as the original run
KNOWN_ORIENTATIONS = [
    ('Borobudur', -7.608, 110.204, 90), ('Prambanan', -7.752, 110.491, 90),
    ('Mendut', -7.605, 110.231, 270), ('Pawon', -7.605, 110.215, 270),
    ('Kalasan', -7.767, 110.472, 90), ('Sewu', -7.745, 110.492, 90),
    ('Plaosan', -7.740, 110.508, 90), ('Sambisari', -7.752, 110.436, 270),
    ('Banyunibo', -7.768, 110.528, 270), ('Candi Badut', -7.958, 112.599, 270),
    ('Candi Kidal', -8.020, 112.617, 270), ('Candi Singosari', -7.889, 112.641, 180),
    ('Candi Jawi', -7.662, 112.670, 270), ('Candi Penataran', -7.985, 112.208, 270),
    ('Candi Surawana', -7.769, 112.009, 270), ('Candi Jabung', -7.706, 113.420, 270),
    ('Candi Jolotundo', -7.610, 112.596, 270), ('Candi Tikus', -7.572, 112.404, 0),
    ('Candi Bajang Ratu', -7.568, 112.399, 0), ('Arjuna group (Dieng)', -7.210, 109.910, 270),
]
orows = []
for name, lat, lon, entrance in KNOWN_ORIENTATIONS:
    d, vname, az_to, _ = nearest(lat, lon)
    diff = abs(entrance - az_to)
    if diff > 180:
        diff = 360 - diff
    orows.append({'name': name, 'entrance_az': entrance, 'volcano_az': az_to,
                  'nearest_volcano': vname, 'distance_km': d,
                  'angular_diff': diff, 'faces_volcano': diff < 90})
df_ov = pd.DataFrame(orows)
n_faces = int(df_ov['faces_volcano'].sum())
btest = stats.binomtest(n_faces, len(df_ov), 0.5, alternative='greater')
t_stat, t_p = stats.ttest_1samp(df_ov['angular_diff'], 90)
print(f"\nORIENTATION: faces volcano {n_faces}/{len(df_ov)} "
      f"({n_faces/len(df_ov)*100:.0f}%), binomial p={btest.pvalue:.4f}")
print(f"  angular diff mean {df_ov['angular_diff'].mean():.1f}, "
      f"median {df_ov['angular_diff'].median():.1f} (random=90), t p={t_p:.4f}")

summary = {
    'experiment': 'E031_candi_orientation (canonical-30 re-run, 2026-06-10)',
    'volcano_inventory': 'data/processed/dashboard/volcanoes_java_full.csv',
    'n_candi_geocoded': len(df_r),
    'n_candi_with_orientation': len(df_ov),
    'n_volcanoes': len(df_volc),
    'siting_analysis': {
        'distance_mean_km': round(df_r['distance_km'].mean(), 1),
        'distance_median_km': round(df_r['distance_km'].median(), 1),
        'pct_within_10km': round((df_r['distance_km'] < 10).mean() * 100, 1),
        'rayleigh_R': round(R_bar, 4),
        'rayleigh_p': float(p_rayleigh),
        'quadrant_counts': quad,
        'quadrant_chi2': round(chi2_q, 2),
        'quadrant_p': round(p_q, 6),
    },
    'orientation_analysis': {
        'faces_volcano_count': n_faces,
        'total': len(df_ov),
        'pct_faces_volcano': round(n_faces / len(df_ov) * 100, 1),
        'binomial_p': round(btest.pvalue, 4),
        'mean_angular_diff': round(df_ov['angular_diff'].mean(), 1),
        'median_angular_diff': round(df_ov['angular_diff'].median(), 1),
    },
}
with open(os.path.join(OUT, 'alignment_summary_canonical30.json'), 'w',
          encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
df_r.to_csv(os.path.join(OUT, 'candi_volcano_pairs_canonical30.csv'), index=False)
df_ov.to_csv(os.path.join(OUT, 'orientation_vs_volcano_canonical30.csv'), index=False)
print(f"\nSaved to {OUT}")
