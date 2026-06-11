#!/usr/bin/env python3
"""
E082 RE-RUN (2026-06-10) — canonical 30-volcano inventory.

Why: the original run used 20 hardcoded volcanoes (missing most of West and
Central Java's peaks, and including Krakatau, which pulled Sumatran
inscriptions into the "nearest volcano" stats). Per the ME#18 integrity sweep
every volcano-distance experiment is re-pointed to
data/processed/dashboard/volcanoes_java_full.csv (30 Java volcanoes).

Geocoding itself is unchanged: we reuse results/geocoded_inscriptions.csv
(lat/lon per inscription) and only recompute volcano distances. Agung and
Batur are kept as supplementary peaks for Bali inscriptions (inside the
Java/Bali analysis box but >100 km from any Java peak). Krakatau is dropped:
it is not on Java, and Sumatran inscriptions fall outside the analysis box
anyway.

The candi comparison uses E031's canonical-30 re-run output, so both sides of
the candi-vs-inscription gap (the "9.2 km" figure in P11) use the same
inventory. Outputs go to results/canonical30/.
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

df_volc = pd.read_csv(os.path.join(REPO, "data", "processed", "dashboard",
                                   "volcanoes_java_full.csv"))
BALI_SUPPLEMENT = pd.DataFrame([
    {'name': 'Agung', 'lat': -8.343, 'lon': 115.508},
    {'name': 'Batur', 'lat': -8.242, 'lon': 115.375},
])
df_volc = pd.concat([df_volc, BALI_SUPPLEMENT], ignore_index=True)
print(f"Volcanoes: {len(df_volc)} (canonical 30 Java + Agung/Batur for Bali)")

df_ins = pd.read_csv(os.path.join(HERE, "results", "geocoded_inscriptions.csv"))
print(f"Geocoded inscriptions (reused): {len(df_ins)}")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def nearest(lat, lon):
    d = haversine_km(lat, lon, df_volc['lat'].values, df_volc['lon'].values)
    i = int(np.argmin(d))
    return df_volc.iloc[i]['name'], float(d[i])


df_ins[['nearest_volcano_c30', 'volcano_dist_km_c30']] = df_ins.apply(
    lambda r: pd.Series(nearest(r['lat'], r['lon'])), axis=1)

# Same Java/Bali analysis box as the original run
java = df_ins[(df_ins['lat'] > -9.0) & (df_ins['lat'] < -6.0) &
              (df_ins['lon'] > 105.0) & (df_ins['lon'] < 116.0)].copy()
jd = java['volcano_dist_km_c30']
print(f"\nJava/Bali subset: {len(java)}")
print(f"  mean {jd.mean():.1f} km, median {jd.median():.1f} km")

zone_a = int((jd <= 10).sum())
zone_b = int(((jd > 10) & (jd <= 30)).sum())
zone_c = int((jd > 30).sum())
print(f"  Zone A (<=10): {zone_a}, Zone B (10-30): {zone_b}, Zone C (>30): {zone_c}")

print("\nNearest volcano distribution (Java/Bali):")
for v, n in java['nearest_volcano_c30'].value_counts().items():
    print(f"  {v}: {n}")

# Candi comparison — both sides on canonical inventory
candi_csv = os.path.join(REPO, "experiments", "E031_candi_orientation",
                         "results", "canonical30",
                         "candi_volcano_pairs_canonical30.csv")
df_candi = pd.read_csv(candi_csv)
cd = df_candi['distance_km']
gap_mean = jd.mean() - cd.mean()
gap_median = jd.median() - cd.median()
u, p_mw = stats.mannwhitneyu(jd, cd, alternative='two-sided')
print(f"\nCANDI vs INSCRIPTION volcano distance (canonical 30):")
print(f"  candi (n={len(cd)}): mean {cd.mean():.1f}, median {cd.median():.1f}")
print(f"  inscriptions (n={len(jd)}): mean {jd.mean():.1f}, median {jd.median():.1f}")
print(f"  mean gap: {gap_mean:.1f} km (P11 cited 9.2 km from the 20-volcano run)")
print(f"  Mann-Whitney U={u:.0f}, p={p_mw:.2e}")

# Bootstrap CI for the mean gap (mirrors P11's reported 95% CI)
rng = np.random.default_rng(42)
boots = [rng.choice(jd, len(jd)).mean() - rng.choice(cd, len(cd)).mean()
         for _ in range(10000)]
ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
print(f"  bootstrap 95% CI for mean gap: {ci_lo:.1f} to {ci_hi:.1f} km")

# Century trend (Spearman), as in the original run
cent = java.dropna(subset=['century'])
if len(cent) > 5:
    means = cent.groupby('century')['volcano_dist_km_c30'].mean()
    rho, p_s = stats.spearmanr(means.index, means.values)
    print(f"\nCentury vs mean distance: rho={rho:.3f}, p={p_s:.4f}")
    print({f"C{int(k)}": round(v, 1) for k, v in means.items()})

summary = {
    'experiment': 'E082_inscription_georeferencing (canonical-30 re-run, 2026-06-10)',
    'volcano_inventory': 'volcanoes_java_full.csv (30) + Agung/Batur (Bali); Krakatau dropped',
    'n_geocoded_reused': len(df_ins),
    'java_bali_subset': len(java),
    'distance_stats': {
        'java_mean': round(jd.mean(), 1),
        'java_median': round(jd.median(), 1),
    },
    'zone_distribution': {'zone_A_0_10km': zone_a, 'zone_B_10_30km': zone_b,
                          'zone_C_gt_30km': zone_c},
    'candi_comparison': {
        'candi_mean': round(cd.mean(), 1),
        'candi_median': round(cd.median(), 1),
        'inscription_mean': round(jd.mean(), 1),
        'inscription_median': round(jd.median(), 1),
        'mean_gap_km': round(gap_mean, 1),
        'median_gap_km': round(gap_median, 1),
        'mean_gap_bootstrap_ci95': [round(ci_lo, 1), round(ci_hi, 1)],
        'mannwhitney_p': float(p_mw),
        'p11_cited_value_old_inventory': 9.2,
    },
    'century_mean_distances': {f"C{int(k)}": round(v, 1) for k, v in means.items()},
    'century_spearman': {'rho': round(rho, 3), 'p': round(p_s, 4)},
}
with open(os.path.join(OUT, 'e082_results_canonical30.json'), 'w',
          encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
java.to_csv(os.path.join(OUT, 'geocoded_inscriptions_canonical30.csv'), index=False)
print(f"\nSaved to {OUT}")
