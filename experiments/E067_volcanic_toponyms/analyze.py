"""
E067: Volcanic Toponyms — Do volcanic place names cluster near volcanoes?

Tests whether villages with volcanic-related morphemes in their names
(gunung, kawah, gumuk, watu, api, lahar, pasir) are closer to active
volcanoes than average villages.

Hypothesis: If volcanic informedness extends to toponymy, kabupaten closer
to volcanoes should have higher proportions of volcanic place names.

Data: E051 village classifications (25,244 villages) + kabupaten summaries
Channel: Ch1 (Geology), Ch9 (Archaeoastronomy/landscape)
Papers served: P11 (volcanic informedness)
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

E051 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "E051_toponymic_substrate", "results")

# ===================================================================
# 1. Load data
# ===================================================================
print("Loading village classifications...")
villages = pd.read_csv(os.path.join(E051, "village_classifications.csv"))
kab_summary = pd.read_csv(os.path.join(E051, "kabupaten_summary.csv"))

print(f"  {len(villages)} villages loaded")
print(f"  {len(kab_summary)} kabupaten loaded")

# ===================================================================
# 2. Define volcanic morphemes
# ===================================================================
# Three tiers of volcanic terms:
# Tier 1: Directly volcanic (unambiguous)
TIER1 = {
    'kawah': 'crater',
    'lahar': 'volcanic mudflow',
    'gumuk': 'volcanic mound/hill (Javanese)',
}

# Tier 2: Volcanic landscape (strong association)
TIER2 = {
    'gunung': 'mountain/volcano',
    'watu': 'stone/rock (Javanese)',
    'batu': 'stone/rock (Malay/Indonesian)',
    'pasir': 'sand/volcanic sand',
    'wedhi': 'sand (Javanese)',
    'segara': 'sea/crater lake',
    'tlogo': 'lake/crater lake (Javanese)',
    'sendang': 'spring (volcanic)',
}

# Tier 3: Potentially volcanic (weaker association)
TIER3 = {
    'api': 'fire',
    'panas': 'hot (thermal)',
    'belerang': 'sulfur',
    'gede': 'great/large (often volcano names)',
    'merapi': 'Merapi (fire mountain)',
    'kelud': 'Kelud volcano',
    'bromo': 'Bromo volcano',
    'semeru': 'Semeru volcano',
    'agung': 'Agung (great, often volcano)',
}

ALL_MORPHEMES = {**TIER1, **TIER2, **TIER3}

# ===================================================================
# 3. Search for volcanic morphemes in village names
# ===================================================================
print("\nSearching for volcanic morphemes in village names...")

def has_morpheme(name, morpheme):
    """Check if a name contains a morpheme (case-insensitive, word boundary aware)."""
    name_lower = str(name).lower()
    morph_lower = morpheme.lower()
    # Check as standalone word or prefix/suffix
    words = name_lower.replace('-', ' ').replace('.', ' ').split()
    for word in words:
        if morph_lower in word:
            return True
    return False

# Check each village
volcanic_flags = {}
for morph in ALL_MORPHEMES:
    volcanic_flags[f'has_{morph}'] = villages['nama'].apply(lambda n: has_morpheme(n, morph))

# Any volcanic morpheme
villages['has_any_volcanic'] = pd.DataFrame(volcanic_flags).any(axis=1)
villages['has_tier1'] = pd.DataFrame({k: volcanic_flags[f'has_{k}'] for k in TIER1}).any(axis=1)
villages['has_tier2'] = pd.DataFrame({k: volcanic_flags[f'has_{k}'] for k in TIER2}).any(axis=1)
villages['has_tier3'] = pd.DataFrame({k: volcanic_flags[f'has_{k}'] for k in TIER3}).any(axis=1)

# Count per morpheme
print("\n=== Morpheme Frequency ===")
morph_counts = {}
for morph in ALL_MORPHEMES:
    count = volcanic_flags[f'has_{morph}'].sum()
    if count > 0:
        morph_counts[morph] = count
        tier = 'T1' if morph in TIER1 else ('T2' if morph in TIER2 else 'T3')
        print(f"  [{tier}] {morph} ({ALL_MORPHEMES[morph]}): {count} villages ({count/len(villages)*100:.2f}%)")

n_any = villages['has_any_volcanic'].sum()
n_t1 = villages['has_tier1'].sum()
n_t2 = villages['has_tier2'].sum()
n_t3 = villages['has_tier3'].sum()
print(f"\n  Total with ANY volcanic morpheme: {n_any} ({n_any/len(villages)*100:.1f}%)")
print(f"  Tier 1 (directly volcanic): {n_t1} ({n_t1/len(villages)*100:.2f}%)")
print(f"  Tier 2 (volcanic landscape): {n_t2} ({n_t2/len(villages)*100:.1f}%)")
print(f"  Tier 3 (potentially volcanic): {n_t3} ({n_t3/len(villages)*100:.1f}%)")

# ===================================================================
# 4. Aggregate to kabupaten level and correlate with volcanic distance
# ===================================================================
print("\n=== Kabupaten-Level Analysis ===")

# Extract kabupaten code from village code (first 5 chars: XX.XX)
villages['kab_code'] = villages['kode'].astype(str).str[:5]

# Aggregate volcanic toponym counts per kabupaten
kab_volcanic = villages.groupby('kab_code').agg(
    total=('nama', 'count'),
    n_volcanic=('has_any_volcanic', 'sum'),
    n_tier1=('has_tier1', 'sum'),
    n_tier2=('has_tier2', 'sum'),
).reset_index()

kab_volcanic['volcanic_ratio'] = kab_volcanic['n_volcanic'] / kab_volcanic['total']
kab_volcanic['tier2_ratio'] = kab_volcanic['n_tier2'] / kab_volcanic['total']

# Merge with kabupaten summary (for volcanic distance)
kab_summary['kab_code'] = kab_summary['kab_code'].astype(str)
kab_volcanic['kab_code'] = kab_volcanic['kab_code'].astype(str)
merged = kab_volcanic.merge(kab_summary[['kab_code', 'kab_name', 'province', 'dist_volcano_km']],
                            on='kab_code', how='inner')

print(f"  Merged: {len(merged)} kabupaten with both toponym counts and volcanic distances")

# Correlation tests
mask = merged['dist_volcano_km'].notna() & merged['volcanic_ratio'].notna()
x = merged.loc[mask, 'dist_volcano_km'].values
y = merged.loc[mask, 'volcanic_ratio'].values

rho_all, p_all = stats.spearmanr(x, y)
print(f"\n  Spearman (all volcanic morphemes vs distance): rho={rho_all:.3f}, p={p_all:.4f}")

# Tier 2 only (volcanic landscape terms)
y_t2 = merged.loc[mask, 'tier2_ratio'].values
rho_t2, p_t2 = stats.spearmanr(x, y_t2)
print(f"  Spearman (Tier 2 only vs distance): rho={rho_t2:.3f}, p={p_t2:.4f}")

# Binary comparison: close (<30km) vs far (>30km)
close = merged[merged['dist_volcano_km'] < 30]['volcanic_ratio']
far = merged[merged['dist_volcano_km'] >= 30]['volcanic_ratio']
if len(close) > 0 and len(far) > 0:
    u_stat, u_p = stats.mannwhitneyu(close, far, alternative='greater')
    print(f"\n  Close (<30km) mean volcanic ratio: {close.mean():.4f} (n={len(close)})")
    print(f"  Far (>=30km) mean volcanic ratio: {far.mean():.4f} (n={len(far)})")
    print(f"  Mann-Whitney U (one-tailed, close > far): p={u_p:.4f}")

# Zone analysis (matching P11 zones)
# Use kabupaten-appropriate zones (no kabupaten centers within 10km of volcano)
zone_near = merged[merged['dist_volcano_km'] < 25]['volcanic_ratio']
zone_mid = merged[(merged['dist_volcano_km'] >= 25) & (merged['dist_volcano_km'] < 50)]['volcanic_ratio']
zone_far = merged[merged['dist_volcano_km'] >= 50]['volcanic_ratio']
print(f"\n  Near (<25km): mean={zone_near.mean():.4f}, n={len(zone_near)}")
print(f"  Mid (25-50km): mean={zone_mid.mean():.4f}, n={len(zone_mid)}")
print(f"  Far (>50km): mean={zone_far.mean():.4f}, n={len(zone_far)}")

kw_stat, kw_p = 0, 1.0
if len(zone_near) > 0 and len(zone_mid) > 0 and len(zone_far) > 0:
    kw_stat, kw_p = stats.kruskal(zone_near, zone_mid, zone_far)
    print(f"  Kruskal-Wallis: H={kw_stat:.3f}, p={kw_p:.4f}")

# ===================================================================
# 5. Top volcanic-toponym kabupaten
# ===================================================================
print("\n=== Top 10 Kabupaten by Volcanic Toponym Ratio ===")
top10 = merged.nlargest(10, 'volcanic_ratio')
for _, row in top10.iterrows():
    print(f"  {row['kab_name']} ({row['province']}): "
          f"{row['volcanic_ratio']*100:.1f}% volcanic, "
          f"{row['dist_volcano_km']:.0f} km from volcano")

print("\n=== Bottom 10 (lowest volcanic toponym ratio) ===")
bot10 = merged.nsmallest(10, 'volcanic_ratio')
for _, row in bot10.iterrows():
    print(f"  {row['kab_name']} ({row['province']}): "
          f"{row['volcanic_ratio']*100:.1f}% volcanic, "
          f"{row['dist_volcano_km']:.0f} km from volcano")

# ===================================================================
# 6. Example volcanic village names
# ===================================================================
print("\n=== Example Volcanic Village Names ===")
volcanic_villages = villages[villages['has_any_volcanic']].head(30)
for morph in sorted(morph_counts, key=morph_counts.get, reverse=True)[:8]:
    examples = villages[volcanic_flags[f'has_{morph}']]['nama'].head(5).tolist()
    print(f"  {morph}: {', '.join(examples)}")

# ===================================================================
# 7. Generate figures
# ===================================================================
print("\n=== Generating Figures ===")

# Figure 1: Scatterplot — volcanic distance vs volcanic toponym ratio
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(merged['dist_volcano_km'], merged['volcanic_ratio'] * 100,
            alpha=0.5, s=30, c='#D62728', edgecolors='black', linewidth=0.3)

# Trendline
z = np.polyfit(x, y * 100, 1)
xline = np.linspace(x.min(), x.max(), 100)
ax1.plot(xline, np.polyval(z, xline), 'k--', linewidth=1.5, alpha=0.7)

ax1.set_xlabel('Distance to Nearest Volcano (km)')
ax1.set_ylabel('Volcanic Toponym Ratio (%)')
ax1.set_title(f'Distance vs Volcanic Toponyms (n={len(merged)})\n'
              f'Spearman rho={rho_all:.3f}, p={p_all:.4f}')

# Figure 2: Zone comparison boxplot
zone_data = [zone_near*100, zone_mid*100, zone_far*100]
zone_labels = [f'Near\n<25km\n(n={len(zone_near)})',
               f'Mid\n25-50km\n(n={len(zone_mid)})',
               f'Far\n>50km\n(n={len(zone_far)})']

bp = ax2.boxplot(zone_data, labels=zone_labels, patch_artist=True, widths=0.5)
colors = ['#FF6B6B', '#FFD700', '#87CEEB']
for box, color in zip(bp['boxes'], colors):
    box.set_facecolor(color)

ax2.set_ylabel('Volcanic Toponym Ratio (%)')
kw_label = f'KW p={kw_p:.4f}' if len(zone_near) > 0 and len(zone_mid) > 0 and len(zone_far) > 0 else ''
ax2.set_title(f'Volcanic Toponyms by Proximity Zone\n{kw_label}')

# Zone means
for i, data in enumerate(zone_data):
    ax2.text(i+1, data.mean() + 0.5, f'{data.mean():.1f}%',
            ha='center', fontsize=10, fontweight='bold')

fig.suptitle('E067: Volcanic Toponyms in Java Village Names',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "volcanic_toponyms.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(RESULTS, "volcanic_toponyms.pdf"), bbox_inches='tight')
plt.close(fig)
print("  Saved volcanic_toponyms.png/.pdf")

# Figure 3: Morpheme frequency bar chart
fig, ax = plt.subplots(figsize=(10, 6))
sorted_morphs = sorted(morph_counts.items(), key=lambda x: x[1], reverse=True)[:15]
labels = [f"{m}\n({ALL_MORPHEMES[m]})" for m, _ in sorted_morphs]
counts_sorted = [c for _, c in sorted_morphs]
tier_colors = []
for m, _ in sorted_morphs:
    if m in TIER1: tier_colors.append('#FF6B6B')
    elif m in TIER2: tier_colors.append('#FFD700')
    else: tier_colors.append('#87CEEB')

ax.barh(range(len(labels)), counts_sorted, color=tier_colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Number of Villages')
ax.set_title('Volcanic Morpheme Frequency in Java Village Names')
ax.invert_yaxis()

# Legend
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color='#FF6B6B', label='Tier 1: Directly volcanic'),
    mpatches.Patch(color='#FFD700', label='Tier 2: Volcanic landscape'),
    mpatches.Patch(color='#87CEEB', label='Tier 3: Potentially volcanic'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "morpheme_frequency.png"), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved morpheme_frequency.png")

# ===================================================================
# 8. Save results
# ===================================================================
results = {
    "experiment": "E067",
    "title": "Volcanic Toponyms in Java Village Names",
    "date": "2026-03-12",
    "n_villages": len(villages),
    "n_kabupaten": len(merged),
    "morpheme_counts": {k: int(v) for k, v in morph_counts.items()},
    "total_volcanic_villages": int(n_any),
    "volcanic_ratio": round(n_any / len(villages), 4),
    "tier_counts": {
        "tier1": int(n_t1),
        "tier2": int(n_t2),
        "tier3": int(n_t3),
    },
    "distance_correlation": {
        "spearman_rho_all": round(rho_all, 3),
        "spearman_p_all": float(p_all),
        "spearman_rho_tier2": round(rho_t2, 3),
        "spearman_p_tier2": float(p_t2),
    },
    "zone_analysis": {
        "near_mean": round(float(zone_near.mean()), 4) if len(zone_near) > 0 else None,
        "mid_mean": round(float(zone_mid.mean()), 4) if len(zone_mid) > 0 else None,
        "far_mean": round(float(zone_far.mean()), 4) if len(zone_far) > 0 else None,
        "kruskal_wallis_p": float(kw_p),
    },
    "binary_comparison": {
        "close_mean": round(close.mean(), 4) if len(close) > 0 else None,
        "far_mean": round(far.mean(), 4) if len(far) > 0 else None,
        "mann_whitney_p": float(u_p) if len(close) > 0 else None,
    },
    "papers_served": ["P11"],
    "channels": [1, 9],
}

# Determine status
if p_all < 0.05 and rho_all < 0:
    results["status"] = "SUCCESS"
    results["key_finding"] = f"Volcanic toponyms correlate with volcanic proximity (rho={rho_all:.3f}, p={p_all:.4f}). Direct evidence of volcanic informedness in place names."
elif p_all < 0.1:
    results["status"] = "CONDITIONAL"
    results["key_finding"] = f"Weak/marginal signal (rho={rho_all:.3f}, p={p_all:.4f}). Volcanic toponyms show trend but not conclusive at alpha=0.05."
else:
    results["status"] = "INFORMATIVE NEGATIVE"
    results["key_finding"] = f"No significant correlation between volcanic toponyms and proximity (rho={rho_all:.3f}, p={p_all:.4f}). Volcanic terminology in place names is distributed evenly across Java, not concentrated near volcanoes."

with open(os.path.join(RESULTS, "e067_results.json"), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved e067_results.json")

# Print summary
print("\n" + "="*60)
print(f"E067 SUMMARY: Volcanic Toponyms — {results['status']}")
print("="*60)
print(f"  {n_any}/{len(villages)} villages ({n_any/len(villages)*100:.1f}%) have volcanic morphemes")
print(f"  Distance correlation: rho={rho_all:.3f}, p={p_all:.4f}")
print(f"  Near: {zone_near.mean()*100:.1f}%, Mid: {zone_mid.mean()*100:.1f}%, Far: {zone_far.mean()*100:.1f}%")
print(f"  {results['key_finding']}")
print("="*60)
