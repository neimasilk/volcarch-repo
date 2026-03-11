#!/usr/bin/env python3
"""
E038 — Volcanic Vocabulary Semantic Drift Across Austronesian
==============================================================
Question: Do Austronesian languages in VOLCANIC regions show different
          cognacy patterns for volcanic-domain vocabulary (fire, ash, smoke,
          stone, earth) compared to NON-VOLCANIC regions?

Hypothesis: Languages near active volcanoes may:
  1. Have MORE lexical diversity for volcanic concepts (more unique terms)
  2. Show LOWER cognacy rates (local innovation for important concepts)
  3. Retain older (pre-Austronesian?) substrate terms for volcanic phenomena

Method:
  1. Extract ABVD forms for volcanic-domain concepts across all Austronesian languages
  2. Classify languages as VOLCANIC or NON-VOLCANIC (based on GVP proximity)
  3. Compare cognacy rates, form diversity, and phonological features
  4. Control comparison: non-volcanic concepts (body parts, kinship)

Data: ABVD (4472 languages, 210 concepts), GVP volcano locations.

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

warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ABVD_DIR = os.path.join(REPO, "experiments", "E022_linguistic_subtraction",
                         "data", "abvd", "cldf")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("E038 — Volcanic Vocabulary Semantic Drift")
print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════
# 1. DEFINE CONCEPT DOMAINS
# ═════════════════════════════════════════════════════════════════════════

# Volcanic-domain concepts (phenomena directly associated with volcanism)
VOLCANIC_CONCEPTS = [
    '143_fire', '144_toburn', '145_smoke', '146_ashes',
    '119_earthsoil', '120_stone', '121_sand', '11_dust',
    '134_thunder', '135_lightning', '138_warm',
]

# Control concepts (body parts — should NOT differ by volcanic proximity)
CONTROL_CONCEPTS = [
    '1_hand', '3_right', '4_legfoot', '5_towalk',
    '23_head', '24_neck', '27_nose', '30_tooth',
    '33_tongue', '72_toeat', '73_todrink',
]

# Environment concepts (non-volcanic geography)
ENVIRONMENT_CONCEPTS = [
    '122_water', '124_sea', '126_lake', '127_woodsforest',
    '128_sky', '131_cloud', '133_rain', '136_wind',
]

print(f"\n[1] Concept domains:")
print(f"  Volcanic: {len(VOLCANIC_CONCEPTS)} concepts")
print(f"  Control (body): {len(CONTROL_CONCEPTS)} concepts")
print(f"  Environment: {len(ENVIRONMENT_CONCEPTS)} concepts")


# ═════════════════════════════════════════════════════════════════════════
# 2. LOAD ABVD DATA
# ═════════════════════════════════════════════════════════════════════════

print("\n[2] Loading ABVD data...")

# Languages
df_lang = pd.read_csv(os.path.join(ABVD_DIR, "languages.csv"))
df_lang = df_lang[df_lang['Family'] == 'Austronesian'].copy()
df_lang = df_lang[df_lang['Latitude'].notna() & df_lang['Longitude'].notna()].copy()
print(f"  Austronesian languages with coordinates: {len(df_lang)}")

# Forms
df_forms = pd.read_csv(os.path.join(ABVD_DIR, "forms.csv"),
                        usecols=['ID', 'Language_ID', 'Parameter_ID', 'Form', 'Cognacy'])
print(f"  Total forms: {len(df_forms)}")

# Cognates
df_cog = pd.read_csv(os.path.join(ABVD_DIR, "cognates.csv"),
                      usecols=['Form_ID', 'Cognateset_ID'])
print(f"  Cognate judgments: {len(df_cog)}")

# Filter to Austronesian
austronesian_ids = set(df_lang['ID'].astype(str).values)
df_forms['lang_id_str'] = df_forms['Language_ID'].astype(str)
df_forms = df_forms[df_forms['lang_id_str'].isin(austronesian_ids)].copy()
print(f"  Austronesian forms: {len(df_forms)}")


# ═════════════════════════════════════════════════════════════════════════
# 3. CLASSIFY LANGUAGES BY VOLCANIC PROXIMITY
# ═════════════════════════════════════════════════════════════════════════

print("\n[3] Classifying languages by volcanic proximity...")

# Major Austronesian volcanic regions (simplified — using GVP-known active regions)
# Active volcano locations (representative points for major volcanic arcs)
VOLCANIC_REGIONS = [
    # Indonesia
    (-6.1, 106.2, 'Krakatau'), (-7.5, 110.4, 'Merapi'), (-8.0, 112.9, 'Semeru'),
    (-8.1, 112.3, 'Kelud'), (-7.6, 112.6, 'Arjuno'), (-7.9, 110.4, 'Merbabu'),
    (-8.3, 115.5, 'Agung'), (-8.4, 116.5, 'Rinjani'), (-8.7, 121.7, 'Kelimutu'),
    (-2.0, 125.5, 'Soputan'), (-1.3, 127.3, 'Ternate'), (1.3, 124.8, 'Lokon'),
    (-4.1, 145.0, 'Manam'), (-6.1, 155.2, 'Bagana'), # Papua/Solomons
    # Philippines
    (13.3, 123.7, 'Mayon'), (14.0, 120.9, 'Taal'), (15.1, 120.3, 'Pinatubo'),
    # Polynesia
    (-15.4, -175.6, 'Niuafoou'), (-20.5, -175.6, 'Tofua'),
    (-16.2, 168.1, 'Ambrym'), (-14.3, 167.8, 'Gaua'), # Vanuatu
    # Taiwan
    (25.2, 121.5, 'Tatun'),
    # New Zealand
    (-38.3, 176.3, 'Taupo'), (-39.3, 175.6, 'Ruapehu'),
    # Hawaii/Samoa
    (19.4, -155.3, 'Kilauea'), (-13.6, -172.5, 'Savaii'),
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


# For each language, find distance to nearest volcano
VOLCANIC_THRESHOLD_KM = 200  # Within 200km of active volcano = "volcanic"

lang_volcanic = {}
lang_min_dist = {}
for _, lang in df_lang.iterrows():
    min_dist = float('inf')
    for vlat, vlon, vname in VOLCANIC_REGIONS:
        d = haversine_km(lang['Latitude'], lang['Longitude'], vlat, vlon)
        if d < min_dist:
            min_dist = d
    lang_volcanic[str(lang['ID'])] = min_dist < VOLCANIC_THRESHOLD_KM
    lang_min_dist[str(lang['ID'])] = min_dist

df_lang['is_volcanic'] = df_lang['ID'].astype(str).map(lang_volcanic)
df_lang['min_volcano_dist_km'] = df_lang['ID'].astype(str).map(lang_min_dist)

n_volcanic = df_lang['is_volcanic'].sum()
n_nonvolcanic = (~df_lang['is_volcanic']).sum()
print(f"  Volcanic (<{VOLCANIC_THRESHOLD_KM}km): {n_volcanic} languages")
print(f"  Non-volcanic: {n_nonvolcanic} languages")

# Geographic distribution
print(f"\n  By Macroarea:")
for area, group in df_lang.groupby('Macroarea'):
    n_v = group['is_volcanic'].sum()
    n_total = len(group)
    print(f"    {area}: {n_total} total, {n_v} volcanic ({n_v/n_total*100:.0f}%)")


# ═════════════════════════════════════════════════════════════════════════
# 4. COMPUTE COGNACY METRICS PER CONCEPT × VOLCANIC STATUS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[4] Computing cognacy metrics...")
print("=" * 70)

# Add volcanic status to forms
df_forms['is_volcanic'] = df_forms['lang_id_str'].map(lang_volcanic)

# Merge cognate sets
df_merged = df_forms.merge(df_cog, left_on='ID', right_on='Form_ID', how='left')


def compute_concept_metrics(concept_id, forms_df, cognates_df):
    """Compute lexical diversity metrics for a concept."""
    concept_forms = forms_df[forms_df['Parameter_ID'] == concept_id]

    if len(concept_forms) == 0:
        return None

    # Split by volcanic status
    volc_forms = concept_forms[concept_forms['is_volcanic'] == True]
    nonv_forms = concept_forms[concept_forms['is_volcanic'] == False]

    # Cognacy: count unique cognate sets per group
    volc_cog = cognates_df[cognates_df['Form_ID'].isin(volc_forms['ID'])]
    nonv_cog = cognates_df[cognates_df['Form_ID'].isin(nonv_forms['ID'])]

    n_volc_sets = volc_cog['Cognateset_ID'].nunique() if len(volc_cog) > 0 else 0
    n_nonv_sets = nonv_cog['Cognateset_ID'].nunique() if len(nonv_cog) > 0 else 0

    # Normalized by number of languages
    n_volc_langs = volc_forms['lang_id_str'].nunique()
    n_nonv_langs = nonv_forms['lang_id_str'].nunique()

    diversity_volc = n_volc_sets / max(n_volc_langs, 1)
    diversity_nonv = n_nonv_sets / max(n_nonv_langs, 1)

    # Form length comparison
    volc_len = volc_forms['Form'].str.len().mean() if len(volc_forms) > 0 else 0
    nonv_len = nonv_forms['Form'].str.len().mean() if len(nonv_forms) > 0 else 0

    # Unique forms (lexical diversity)
    volc_unique = volc_forms['Form'].str.lower().nunique() / max(len(volc_forms), 1)
    nonv_unique = nonv_forms['Form'].str.lower().nunique() / max(len(nonv_forms), 1)

    return {
        'concept': concept_id,
        'n_volc_langs': n_volc_langs,
        'n_nonv_langs': n_nonv_langs,
        'n_volc_cogsets': n_volc_sets,
        'n_nonv_cogsets': n_nonv_sets,
        'diversity_volc': diversity_volc,
        'diversity_nonv': diversity_nonv,
        'diversity_diff': diversity_volc - diversity_nonv,
        'mean_len_volc': volc_len,
        'mean_len_nonv': nonv_len,
        'unique_ratio_volc': volc_unique,
        'unique_ratio_nonv': nonv_unique,
    }


# Compute for all concept domains
all_results = []
for domain_name, concepts in [('volcanic', VOLCANIC_CONCEPTS),
                               ('control', CONTROL_CONCEPTS),
                               ('environment', ENVIRONMENT_CONCEPTS)]:
    for concept in concepts:
        result = compute_concept_metrics(concept, df_forms, df_cog)
        if result:
            result['domain'] = domain_name
            all_results.append(result)

df_results = pd.DataFrame(all_results)

print(f"\n  Computed metrics for {len(df_results)} concepts")
print(f"\n  {'Concept':<25} {'Domain':<12} {'Div.Volc':>9} {'Div.NonV':>9} {'Diff':>8}")
print("  " + "-" * 70)
for _, row in df_results.sort_values('diversity_diff', ascending=False).iterrows():
    cname = row['concept'].split('_', 1)[1][:22]
    print(f"  {cname:<25} {row['domain']:<12} {row['diversity_volc']:>8.3f} "
          f"{row['diversity_nonv']:>8.3f} {row['diversity_diff']:>+7.3f}")


# ═════════════════════════════════════════════════════════════════════════
# 5. STATISTICAL TESTS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[5] Statistical Tests")
print("=" * 70)

# Test 1: Do volcanic concepts show more diversity in volcanic regions?
volc_domain = df_results[df_results['domain'] == 'volcanic']
ctrl_domain = df_results[df_results['domain'] == 'control']
env_domain = df_results[df_results['domain'] == 'environment']

# Diversity difference: volcanic vs non-volcanic regions
print(f"\n  Mean cognate diversity difference (volcanic - non-volcanic regions):")
for domain_name, domain_df in [('Volcanic concepts', volc_domain),
                                 ('Control (body)', ctrl_domain),
                                 ('Environment', env_domain)]:
    diff = domain_df['diversity_diff'].values
    mean_diff = np.mean(diff)
    t, p = stats.ttest_1samp(diff, 0) if len(diff) > 1 else (0, 1)
    print(f"    {domain_name:<25} mean diff={mean_diff:+.4f}  t={t:.3f}  p={p:.4f}"
          f"{'  *' if p < 0.05 else ''}")

# Test 2: Compare diversity differences between domains
if len(volc_domain) > 0 and len(ctrl_domain) > 0:
    diff_volc = volc_domain['diversity_diff'].values
    diff_ctrl = ctrl_domain['diversity_diff'].values
    u_stat, u_p = stats.mannwhitneyu(diff_volc, diff_ctrl, alternative='two-sided')
    print(f"\n  Volcanic vs Control domain diversity difference:")
    print(f"    Mann-Whitney U: U={u_stat:.1f}, p={u_p:.4f}")

# Test 3: Form length comparison
print(f"\n  Mean form length (volcanic vs non-volcanic regions):")
for domain_name, domain_df in [('Volcanic concepts', volc_domain),
                                 ('Control (body)', ctrl_domain),
                                 ('Environment', env_domain)]:
    len_volc = domain_df['mean_len_volc'].mean()
    len_nonv = domain_df['mean_len_nonv'].mean()
    print(f"    {domain_name:<25} volcanic={len_volc:.2f}  non-volcanic={len_nonv:.2f}"
          f"  diff={len_volc-len_nonv:+.2f}")

# Test 4: Unique form ratio
print(f"\n  Unique form ratio (volcanic vs non-volcanic regions):")
for domain_name, domain_df in [('Volcanic concepts', volc_domain),
                                 ('Control (body)', ctrl_domain),
                                 ('Environment', env_domain)]:
    u_volc = domain_df['unique_ratio_volc'].mean()
    u_nonv = domain_df['unique_ratio_nonv'].mean()
    print(f"    {domain_name:<25} volcanic={u_volc:.3f}  non-volcanic={u_nonv:.3f}"
          f"  diff={u_volc-u_nonv:+.3f}")


# ═════════════════════════════════════════════════════════════════════════
# 6. INDIVIDUAL CONCEPT DEEP DIVES
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[6] Deep Dives: Key Volcanic Concepts")
print("=" * 70)

KEY_CONCEPTS = ['143_fire', '146_ashes', '145_smoke', '120_stone', '119_earthsoil']

for concept_id in KEY_CONCEPTS:
    concept_forms = df_forms[df_forms['Parameter_ID'] == concept_id].copy()
    if len(concept_forms) == 0:
        continue

    concept_name = concept_id.split('_', 1)[1]
    volc_f = concept_forms[concept_forms['is_volcanic'] == True]
    nonv_f = concept_forms[concept_forms['is_volcanic'] == False]

    print(f"\n  --- {concept_name.upper()} ---")
    print(f"  Volcanic: {len(volc_f)} forms from {volc_f['lang_id_str'].nunique()} languages")
    print(f"  Non-volcanic: {len(nonv_f)} forms from {nonv_f['lang_id_str'].nunique()} languages")

    # Most common forms in each group
    if len(volc_f) > 0:
        top_volc = volc_f['Form'].str.lower().value_counts().head(5)
        print(f"  Top volcanic forms: {', '.join(f'{f}({c})' for f, c in top_volc.items())}")
    if len(nonv_f) > 0:
        top_nonv = nonv_f['Form'].str.lower().value_counts().head(5)
        print(f"  Top non-volcanic: {', '.join(f'{f}({c})' for f, c in top_nonv.items())}")


# ═════════════════════════════════════════════════════════════════════════
# 7. DISTANCE-BASED ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[7] Distance-based Analysis: Cognacy vs Volcano Distance")
print("=" * 70)

# For each language, compute "volcanic concept cognacy" — how many volcanic concepts
# share the most common cognate set (= high cognacy = conservative/inherited)

# First get the most common cognate set per concept
concept_dominant_cog = {}
for concept in VOLCANIC_CONCEPTS:
    concept_cogs = df_merged[df_merged['Parameter_ID'] == concept]
    if len(concept_cogs) > 0:
        dom = concept_cogs['Cognateset_ID'].value_counts()
        if len(dom) > 0:
            concept_dominant_cog[concept] = dom.index[0]

# Per language: fraction of volcanic concepts using dominant cognate set
lang_conservatism = {}
for lang_id in df_lang['ID'].astype(str).unique():
    lang_forms = df_merged[df_merged['lang_id_str'] == lang_id]
    n_match = 0
    n_total = 0
    for concept, dom_cog in concept_dominant_cog.items():
        concept_rows = lang_forms[lang_forms['Parameter_ID'] == concept]
        if len(concept_rows) > 0:
            n_total += 1
            if dom_cog in concept_rows['Cognateset_ID'].values:
                n_match += 1
    if n_total >= 3:  # Need at least 3 concepts
        lang_conservatism[lang_id] = n_match / n_total

# Correlate with volcano distance
conserv_df = pd.DataFrame([
    {'lang_id': k, 'conservatism': v,
     'min_volcano_dist': lang_min_dist.get(k, np.nan)}
    for k, v in lang_conservatism.items()
]).dropna()

if len(conserv_df) > 10:
    r, p = stats.spearmanr(conserv_df['min_volcano_dist'],
                            conserv_df['conservatism'])
    print(f"\n  Languages with 3+ volcanic concepts: {len(conserv_df)}")
    print(f"  Spearman correlation (volcano distance vs cognate conservatism):")
    print(f"    rho={r:+.3f}, p={p:.4f}"
          f"{'  *' if p < 0.05 else ''}")
    print(f"  Interpretation: {'Closer to volcanoes = MORE innovative' if r > 0 else 'Closer to volcanoes = MORE conservative' if r < 0 else 'No relationship'}")

    # Bin by distance for visualization
    conserv_df['dist_bin'] = pd.cut(conserv_df['min_volcano_dist'],
                                     bins=[0, 100, 200, 500, 1000, 5000],
                                     labels=['<100km', '100-200', '200-500',
                                             '500-1000', '>1000km'])
    print(f"\n  Conservatism by distance band:")
    for dist_bin, group in conserv_df.groupby('dist_bin', observed=True):
        print(f"    {dist_bin}: n={len(group)}, mean conservatism={group['conservatism'].mean():.3f}")


# ═════════════════════════════════════════════════════════════════════════
# 8. VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[8] Generating visualizations...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E038 — Volcanic Vocabulary Drift Across Austronesian\n'
             'Do Languages Near Volcanoes Innovate More for Fire/Ash/Smoke?',
             fontsize=14, fontweight='bold', y=0.98)

# Panel A: Diversity difference by domain
ax1 = axes[0, 0]
domains = ['volcanic', 'control', 'environment']
domain_labels = ['Volcanic\n(fire, ash, smoke...)', 'Control\n(body parts)',
                 'Environment\n(water, sky, rain...)']
domain_diffs = []
for d in domains:
    diffs = df_results[df_results['domain'] == d]['diversity_diff'].values
    domain_diffs.append(diffs)

bp = ax1.boxplot(domain_diffs, labels=domain_labels, patch_artist=True,
                  widths=0.6)
colors_bp = ['#e74c3c', '#3498db', '#27ae60']
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_ylabel('Diversity Difference\n(volcanic region - non-volcanic region)')
ax1.set_title('A. Cognate Diversity by Domain', fontsize=11)

# Panel B: Per-concept diversity
ax2 = axes[0, 1]
for domain_name, color, marker in [('volcanic', '#e74c3c', 'o'),
                                     ('control', '#3498db', 's'),
                                     ('environment', '#27ae60', '^')]:
    dom_data = df_results[df_results['domain'] == domain_name]
    ax2.scatter(dom_data['diversity_nonv'], dom_data['diversity_volc'],
               c=color, marker=marker, s=50, alpha=0.7, label=domain_name,
               edgecolors='gray', linewidth=0.3)

lims2 = [0, max(df_results['diversity_volc'].max(),
                df_results['diversity_nonv'].max()) * 1.1]
ax2.plot(lims2, lims2, 'k--', alpha=0.3, label='Equal diversity')
ax2.set_xlabel('Diversity (non-volcanic regions)')
ax2.set_ylabel('Diversity (volcanic regions)')
ax2.set_title('B. Per-Concept Diversity Comparison', fontsize=11)
ax2.legend(fontsize=8)

# Panel C: Distance vs conservatism scatter
ax3 = axes[1, 0]
if len(conserv_df) > 10:
    ax3.scatter(conserv_df['min_volcano_dist'], conserv_df['conservatism'],
               s=10, alpha=0.3, color='#e74c3c')
    # Trend line
    z = np.polyfit(conserv_df['min_volcano_dist'], conserv_df['conservatism'], 1)
    x_line = np.linspace(0, conserv_df['min_volcano_dist'].max(), 100)
    ax3.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5)
    ax3.set_xlabel('Distance to Nearest Active Volcano (km)')
    ax3.set_ylabel('Volcanic Concept Conservatism')
    ax3.set_title(f'C. Conservatism vs Volcano Distance\n(rho={r:+.3f}, p={p:.4f})',
                  fontsize=11)
else:
    ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')

# Panel D: Map of volcanic vs non-volcanic languages
ax4 = axes[1, 1]
nonv_langs = df_lang[~df_lang['is_volcanic']]
volc_langs = df_lang[df_lang['is_volcanic']]
ax4.scatter(nonv_langs['Longitude'], nonv_langs['Latitude'],
           s=3, alpha=0.2, color='#3498db', label=f'Non-volcanic ({len(nonv_langs)})')
ax4.scatter(volc_langs['Longitude'], volc_langs['Latitude'],
           s=5, alpha=0.4, color='#e74c3c', label=f'Volcanic ({len(volc_langs)})')
# Plot volcanoes
for vlat, vlon, vname in VOLCANIC_REGIONS:
    ax4.plot(vlon, vlat, '^', color='red', markersize=4, alpha=0.5)
ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
ax4.set_title('D. Austronesian Languages: Volcanic Proximity', fontsize=11)
ax4.legend(fontsize=8, loc='lower left')

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(RESULTS_DIR, 'volcanic_vocab_4panel.png'), dpi=150,
            bbox_inches='tight')
print("  Saved: volcanic_vocab_4panel.png")
plt.close('all')


# ═════════════════════════════════════════════════════════════════════════
# 9. STRUCTURED OUTPUT
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[9] Saving results...")
print("=" * 70)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

summary = {
    "experiment": "E038_volcanic_vocabulary",
    "n_languages": len(df_lang),
    "n_volcanic": int(n_volcanic),
    "n_nonvolcanic": int(n_nonvolcanic),
    "threshold_km": VOLCANIC_THRESHOLD_KM,
    "domains": {
        "volcanic_concepts": len(VOLCANIC_CONCEPTS),
        "control_concepts": len(CONTROL_CONCEPTS),
        "environment_concepts": len(ENVIRONMENT_CONCEPTS),
    },
    "diversity_diff_by_domain": {},
}

for domain_name in ['volcanic', 'control', 'environment']:
    dom = df_results[df_results['domain'] == domain_name]
    diff = dom['diversity_diff'].values
    t, p_val = stats.ttest_1samp(diff, 0) if len(diff) > 1 else (0, 1)
    summary["diversity_diff_by_domain"][domain_name] = {
        "mean_diff": round(float(np.mean(diff)), 4),
        "t_stat": round(float(t), 3),
        "p_value": round(float(p_val), 4),
    }

if len(conserv_df) > 10:
    summary["distance_conservatism"] = {
        "n_languages": len(conserv_df),
        "spearman_rho": round(float(r), 3),
        "spearman_p": round(float(p), 4),
    }

with open(os.path.join(RESULTS_DIR, 'vocabulary_drift_summary.json'), 'w',
          encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

df_results.to_csv(os.path.join(RESULTS_DIR, 'concept_metrics.csv'), index=False)

print("  Saved: vocabulary_drift_summary.json")
print("  Saved: concept_metrics.csv")


# ═════════════════════════════════════════════════════════════════════════
# 10. HEADLINE
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HEADLINE FINDING")
print("=" * 70)

print(f"""
  CORPUS: {len(df_lang)} Austronesian languages
  VOLCANIC: {n_volcanic} (<{VOLCANIC_THRESHOLD_KM}km from active volcano)
  NON-VOLCANIC: {n_nonvolcanic}

  COGNATE DIVERSITY DIFFERENCE (volcanic - non-volcanic regions):
""")

for domain_name in ['volcanic', 'control', 'environment']:
    dom = df_results[df_results['domain'] == domain_name]
    diff = dom['diversity_diff'].values
    mean_d = np.mean(diff)
    t, p_val = stats.ttest_1samp(diff, 0) if len(diff) > 1 else (0, 1)
    sig = " ← SIGNIFICANT" if p_val < 0.05 else ""
    print(f"  {domain_name:<15} mean={mean_d:+.4f}  p={p_val:.4f}{sig}")

if len(conserv_df) > 10:
    print(f"\n  DISTANCE CORRELATION:")
    print(f"  rho={r:+.3f}, p={p:.4f}")

print("\n" + "=" * 70)
print("E038 COMPLETE")
print("=" * 70)
