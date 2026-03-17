"""
E102 — Vocabulary Richness × Burial Depth Nexus
================================================
Tests whether inscriptions geographically near deeply-buried sites
have richer pre-Indic vocabulary. If L4 (cosmological overwrite) and
L1 (volcanic burial) interact: volcanic zones may preserve more
indigenous vocabulary because volcanic communities maintained
stronger pre-Hindu practices.

Experiment #103 in VOLCARCH series.
"""

import pandas as pd
import numpy as np
from scipy import stats
from math import radians, sin, cos, sqrt, atan2
import json

print("=" * 70)
print("E102 — VOCABULARY RICHNESS x BURIAL DEPTH NEXUS")
print("=" * 70)

# --- 1. Load data ---
print("\n[1/6] Loading data...")

# E074: inscription metadata (Sanskrit vs indigenous word counts)
meta = pd.read_csv("experiments/E074_dharma_deep_nlp/results/inscription_metadata.csv")

# E082: geocoded inscriptions
geo = pd.read_csv("experiments/E082_inscription_georeferencing/results/geocoded_inscriptions.csv")

# E030: dated inscriptions with pre-indic ratio
dated = pd.read_csv("experiments/E030_prasasti_temporal_nlp/results/dated_inscriptions.csv")

# E083: burial sites with depth
e083 = pd.read_csv("experiments/E083_tephra_archaeological_correlation/results/tephra_archaeological_correlation.csv")
burial_sites = e083[e083['burial_depth_m'].notna() & e083['site_lat'].notna()].copy()

print(f"  E074 metadata: {len(meta)} inscriptions")
print(f"  E082 geocoded: {len(geo)} inscriptions")
print(f"  E030 dated: {len(dated)} inscriptions")
print(f"  E083 burial sites with depth+coords: {len(burial_sites)}")

# --- 2. Merge inscription data ---
print("\n[2/6] Merging inscription data...")

# Merge E074 (vocab) with E082 (coords) on filename
merged = meta.merge(geo[['filename', 'lat', 'lon', 'nearest_volcano', 'volcano_dist_km']],
                    on='filename', how='inner')

# Also merge E030 pre-indic ratio
merged = merged.merge(dated[['filename', 'pre_indic_ratio', 'has_hyang']],
                      on='filename', how='left')

# Compute indigenous ratio
merged['indigenous_ratio'] = merged['n_indigenous'] / (merged['n_sanskrit'] + merged['n_indigenous'] + 0.1)
merged['vocab_richness'] = merged['n_indigenous'] + merged['n_sanskrit']

print(f"  Merged inscriptions with coords + vocab: {len(merged)}")
print(f"  With pre-indic ratio: {merged['pre_indic_ratio'].notna().sum()}")

# --- 3. Match inscriptions to nearest burial site ---
print("\n[3/6] Matching inscriptions to nearest burial sites...")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# For each inscription, find nearest burial site and its depth
nearest_depth = []
nearest_dist_to_burial = []
for _, insc in merged.iterrows():
    if pd.isna(insc['lat']) or pd.isna(insc['lon']):
        nearest_depth.append(np.nan)
        nearest_dist_to_burial.append(np.nan)
        continue

    min_dist = float('inf')
    min_depth = np.nan
    for _, bs in burial_sites.iterrows():
        d = haversine(insc['lat'], insc['lon'], bs['site_lat'], bs['site_lon'])
        if d < min_dist:
            min_dist = d
            min_depth = bs['burial_depth_m']

    nearest_depth.append(min_depth if min_dist < 100 else np.nan)  # 100km cutoff
    nearest_dist_to_burial.append(min_dist)

merged['nearest_burial_depth'] = nearest_depth
merged['dist_to_burial_site'] = nearest_dist_to_burial

valid = merged[merged['nearest_burial_depth'].notna()].copy()
print(f"  Inscriptions matched to burial sites (<100km): {len(valid)}")

# --- 4. Correlations ---
print("\n[4/6] Correlation analysis...")

print(f"\n  {'Feature pair':<45} {'rho':>8} {'p':>10} {'n':>5}")
print(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*5}")

pairs = [
    ('indigenous_ratio', 'nearest_burial_depth', 'Indigenous ratio × burial depth'),
    ('n_indigenous', 'nearest_burial_depth', 'Indigenous count × burial depth'),
    ('n_volcanic_terms', 'nearest_burial_depth', 'Volcanic terms × burial depth'),
    ('n_geo_terms', 'nearest_burial_depth', 'Geographic terms × burial depth'),
    ('indigenous_ratio', 'volcano_dist_km', 'Indigenous ratio × volcano distance'),
    ('n_indigenous', 'volcano_dist_km', 'Indigenous count × volcano distance'),
    ('pre_indic_ratio', 'nearest_burial_depth', 'Pre-indic ratio × burial depth'),
    ('pre_indic_ratio', 'volcano_dist_km', 'Pre-indic ratio × volcano distance'),
    ('n_admin_terms', 'nearest_burial_depth', 'Admin terms × burial depth'),
]

corr_results = {}
for feat_a, feat_b, label in pairs:
    data = valid[[feat_a, feat_b]].dropna()
    if len(data) > 5:
        rho, p = stats.spearmanr(data[feat_a], data[feat_b])
        corr_results[label] = {'rho': float(rho), 'p': float(p), 'n': len(data)}
        sig = " *" if p < 0.05 else " **" if p < 0.01 else ""
        print(f"  {label:<45} {rho:>8.4f} {p:>10.4f} {len(data):>5}{sig}")

# --- 5. Zone-based analysis ---
print("\n[5/6] Volcanic zone analysis...")

# Compare inscriptions near active volcanoes (<20km) vs far (>30km)
near_volcano = valid[valid['volcano_dist_km'] < 20]
far_volcano = valid[valid['volcano_dist_km'] > 30]

print(f"\n  Near volcano (<20km): {len(near_volcano)} inscriptions")
print(f"  Far from volcano (>30km): {len(far_volcano)} inscriptions")

if len(near_volcano) > 3 and len(far_volcano) > 3:
    # Compare indigenous ratio
    mw_ratio, p_ratio = stats.mannwhitneyu(
        near_volcano['indigenous_ratio'].dropna(),
        far_volcano['indigenous_ratio'].dropna(),
        alternative='two-sided'
    )
    print(f"\n  Indigenous ratio — near: {near_volcano['indigenous_ratio'].mean():.4f}, far: {far_volcano['indigenous_ratio'].mean():.4f}")
    print(f"  Mann-Whitney U={mw_ratio:.0f}, p={p_ratio:.4f}")

    # Compare vocab richness
    mw_rich, p_rich = stats.mannwhitneyu(
        near_volcano['vocab_richness'].dropna(),
        far_volcano['vocab_richness'].dropna(),
        alternative='two-sided'
    )
    print(f"\n  Vocab richness — near: {near_volcano['vocab_richness'].mean():.1f}, far: {far_volcano['vocab_richness'].mean():.1f}")
    print(f"  Mann-Whitney U={mw_rich:.0f}, p={p_rich:.4f}")

    # Compare volcanic terms
    mw_volc, p_volc = stats.mannwhitneyu(
        near_volcano['n_volcanic_terms'].dropna(),
        far_volcano['n_volcanic_terms'].dropna(),
        alternative='two-sided'
    )
    print(f"\n  Volcanic terms — near: {near_volcano['n_volcanic_terms'].mean():.2f}, far: {far_volcano['n_volcanic_terms'].mean():.2f}")
    print(f"  Mann-Whitney U={mw_volc:.0f}, p={p_volc:.4f}")

    # Compare geographic terms
    mw_geo, p_geo = stats.mannwhitneyu(
        near_volcano['n_geo_terms'].dropna(),
        far_volcano['n_geo_terms'].dropna(),
        alternative='two-sided'
    )
    print(f"\n  Geographic terms — near: {near_volcano['n_geo_terms'].mean():.2f}, far: {far_volcano['n_geo_terms'].mean():.2f}")
    print(f"  Mann-Whitney U={mw_geo:.0f}, p={p_geo:.4f}")

# --- 6. Depth-binned vocabulary profile ---
print("\n[6/6] Depth-binned vocabulary profile...")

# Bin inscriptions by nearest burial depth
depth_bins = [(0, 2, "Shallow (0-2m)"), (2, 5, "Medium (2-5m)"), (5, 10, "Deep (5-10m)")]
for lo, hi, label in depth_bins:
    subset = valid[(valid['nearest_burial_depth'] >= lo) & (valid['nearest_burial_depth'] < hi)]
    if len(subset) > 0:
        print(f"\n  {label}: {len(subset)} inscriptions")
        print(f"    Indigenous ratio: {subset['indigenous_ratio'].mean():.4f}")
        print(f"    Sanskrit count: {subset['n_sanskrit'].mean():.1f}")
        print(f"    Indigenous count: {subset['n_indigenous'].mean():.1f}")
        print(f"    Volcanic terms: {subset['n_volcanic_terms'].mean():.2f}")
        print(f"    Geographic terms: {subset['n_geo_terms'].mean():.2f}")

# --- Save ---
results = {
    'meta': {
        'experiment': 'E102',
        'date': '2026-03-17',
        'n_merged': len(merged),
        'n_matched': len(valid),
    },
    'correlations': corr_results,
    'zone_comparison': {
        'near_n': len(near_volcano),
        'far_n': len(far_volcano),
        'indigenous_ratio': {
            'near_mean': float(near_volcano['indigenous_ratio'].mean()),
            'far_mean': float(far_volcano['indigenous_ratio'].mean()),
            'p': float(p_ratio),
        } if len(near_volcano) > 3 and len(far_volcano) > 3 else {},
    },
}

with open("experiments/E102_vocabulary_preservation_nexus/results/e102_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("E102 SUMMARY")
print("=" * 70)
# Find strongest correlation
if corr_results:
    best = min(corr_results.items(), key=lambda x: x[1]['p'])
    print(f"  Strongest correlation: {best[0]}")
    print(f"    rho={best[1]['rho']:.4f}, p={best[1]['p']:.4f}")
print(f"  Inscriptions analyzed: {len(valid)}")
print("=" * 70)
