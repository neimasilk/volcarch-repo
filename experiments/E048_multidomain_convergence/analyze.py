"""E048: Multi-Domain Temporal Convergence Analysis.

Merges all DHARMA datasets and tests whether de-Sanskritization, pre-Indic
resurgence, organic material prominence, and botanical diversity move
together as a single temporal wave.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BASE, '..', '..'))
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

print('=' * 60)
print('E048: Multi-Domain Temporal Convergence Analysis')
print('=' * 60)

# ============================================================
# 1. LOAD ALL DATASETS
# ============================================================
print('\n--- Loading datasets ---')

# Master corpus (268 inscriptions)
df_master = pd.read_csv(os.path.join(REPO, 'experiments/E023_ritual_screening/results/full_corpus_classification.csv'))
print(f'  Master corpus: {len(df_master)} inscriptions')

# Material culture (268 inscriptions)
df_material = pd.read_csv(os.path.join(REPO, 'experiments/E040_bamboo_civilization/results/material_culture_inscriptions.csv'))
print(f'  Material culture: {len(df_material)} inscriptions')

# Botanical (249 inscriptions)
df_botanical = pd.read_csv(os.path.join(REPO, 'experiments/E035_prasasti_botanical/results/botanical_inscriptions.csv'))
print(f'  Botanical: {len(df_botanical)} inscriptions')

# Dated subset (166 inscriptions)
df_dated = pd.read_csv(os.path.join(REPO, 'experiments/E030_prasasti_temporal_nlp/results/dated_inscriptions.csv'))
print(f'  Dated subset: {len(df_dated)} inscriptions')

# ============================================================
# 2. MERGE INTO UNIFIED DATASET
# ============================================================
print('\n--- Merging datasets ---')

# Start with DATED inscriptions (they have year_ce + century)
df = df_dated.copy()
df['century_label'] = df['century'].apply(lambda c: f'C{c}')

# Merge material culture
material_cols = ['filename', 'n_organic', 'n_lithic', 'n_metal', 'materials', 'classes']
material_cols = [c for c in material_cols if c in df_material.columns]
df = df.merge(df_material[material_cols], on='filename', how='left')

# Merge botanical
bot_cols = ['filename']
for c in df_botanical.columns:
    if c not in df.columns:
        bot_cols.append(c)
df = df.merge(df_botanical[bot_cols], on='filename', how='left')

# Derived features
df['n_organic'] = df['n_organic'].fillna(0)
df['n_lithic'] = df['n_lithic'].fillna(0)
df['organic_ratio'] = df['n_organic'] / (df['n_organic'] + df['n_lithic']).replace(0, np.nan)
df['has_organic'] = df['n_organic'] > 0
df['has_lithic'] = df['n_lithic'] > 0
df['n_plants'] = df['n_plants'].fillna(0).astype(int) if 'n_plants' in df.columns else 0

# Alias for temporal analysis
df_t = df.copy()
print(f'  Merged dated dataset: {len(df_t)} inscriptions')
print(f'  Centuries represented: {sorted(df_t["century_label"].unique())}')

# ============================================================
# 3. CENTURY-BY-CENTURY MULTI-DOMAIN SUMMARY
# ============================================================
print('\n--- Century-by-century summary ---')

centuries_order = ['C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']

# Aggregate by century
agg = df_t.groupby('century_label').agg(
    n_inscriptions=('filename', 'count'),
    mean_pre_indic_ratio=('pre_indic_ratio', 'mean'),
    mean_indic=('indic', 'mean'),
    mean_pre_indic=('pre_indic', 'mean'),
    pct_has_hyang=('has_hyang', 'mean'),
    mean_organic=('n_organic', 'mean'),
    mean_lithic=('n_lithic', 'mean'),
    mean_organic_ratio=('organic_ratio', 'mean'),
    pct_has_organic=('has_organic', 'mean'),
    mean_n_plants=('n_plants', 'mean'),
    mean_word_count=('word_count', 'mean'),
).reindex([c for c in centuries_order if c in df_t['century_label'].unique()])

# Compute Indic ratio (inverse of pre-Indic)
agg['indic_ratio'] = 1 - agg['mean_pre_indic_ratio']

# Filter to centuries with N >= 2
agg = agg[agg['n_inscriptions'] >= 2]

print(agg.to_string())

# Save
agg.to_csv(os.path.join(RESULTS, 'century_summary.csv'))
print(f'\n  Saved: century_summary.csv')

# ============================================================
# 4. TEMPORAL CORRELATIONS
# ============================================================
print('\n--- Temporal correlations (century-level) ---')

# Convert century to numeric for correlation
agg['century_num'] = [int(c[1:]) for c in agg.index]
print(f'  Centuries with N>=2: {list(agg.index)}')

signals = {
    'Indic ratio': 'indic_ratio',
    'Pre-Indic ratio': 'mean_pre_indic_ratio',
    'Hyang presence': 'pct_has_hyang',
    'Organic mean': 'mean_organic',
    'Organic ratio': 'mean_organic_ratio',
    'Plant diversity': 'mean_n_plants',
    'Word count': 'mean_word_count',
}

print('\n  Signal vs Century (Spearman):')
for name, col in signals.items():
    valid = agg[[col, 'century_num']].dropna()
    if len(valid) >= 4:
        rho, p = stats.spearmanr(valid['century_num'], valid[col])
        direction = 'INCREASES' if rho > 0 else 'DECREASES'
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '(ns)'
        print(f'    {name:20s}: rho={rho:+.3f}, p={p:.4f} {sig} → {direction} over time')

# Cross-correlations between signals
print('\n  Cross-correlations (inscription-level, dated subset):')
signal_pairs = [
    ('pre_indic_ratio', 'n_organic', 'Pre-Indic ↔ Organic'),
    ('pre_indic_ratio', 'n_plants', 'Pre-Indic ↔ Botanical'),
    ('n_organic', 'n_plants', 'Organic ↔ Botanical'),
    ('pre_indic_ratio', 'word_count', 'Pre-Indic ↔ Length'),
    ('n_organic', 'word_count', 'Organic ↔ Length'),
]

for col1, col2, label in signal_pairs:
    if col1 in df_t.columns and col2 in df_t.columns:
        valid = df_t[[col1, col2]].dropna()
        if len(valid) >= 10:
            rho, p = stats.spearmanr(valid[col1], valid[col2])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '(ns)'
            print(f'    {label:30s}: rho={rho:+.3f}, p={p:.4f} {sig}')

# ============================================================
# 5. THE KEY TEST: Partial correlation (controlling for word count)
# ============================================================
print('\n--- Partial correlations (controlling for inscription length) ---')

def partial_corr(df, x, y, z):
    """Partial Spearman correlation of x,y controlling for z."""
    valid = df[[x, y, z]].dropna()
    if len(valid) < 10:
        return np.nan, np.nan
    # Rank-transform for Spearman
    for col in [x, y, z]:
        valid[f'{col}_rank'] = valid[col].rank()
    # Residualize x and y on z
    from numpy.polynomial.polynomial import polyfit
    # Simple linear residuals of ranks
    bx = np.polyfit(valid[f'{z}_rank'], valid[f'{x}_rank'], 1)
    by = np.polyfit(valid[f'{z}_rank'], valid[f'{y}_rank'], 1)
    res_x = valid[f'{x}_rank'] - np.polyval(bx, valid[f'{z}_rank'])
    res_y = valid[f'{y}_rank'] - np.polyval(by, valid[f'{z}_rank'])
    rho, p = stats.spearmanr(res_x, res_y)
    return rho, p

for col1, col2, label in signal_pairs[:3]:  # Main three
    if col1 in df_t.columns and col2 in df_t.columns:
        rho, p = partial_corr(df_t, col1, col2, 'word_count')
        if not np.isnan(rho):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '(ns)'
            print(f'    {label:30s}: partial rho={rho:+.3f}, p={p:.4f} {sig} (controlling for length)')

# ============================================================
# 6. GENRE TAPHONOMY TEST (L5)
# ============================================================
print('\n--- Genre taphonomy test (L5) ---')
# Split by inscription language/format (Old Javanese sima vs Sanskrit short)
# Proxy: word count as indicator of format (longer = sima, shorter = Sanskrit)
median_wc = df_t['word_count'].median()
df_t = df_t.copy()
df_t['format'] = df_t['word_count'].apply(lambda w: 'long_sima' if w >= median_wc else 'short_format')

for fmt in ['short_format', 'long_sima']:
    subset = df_t[df_t['format'] == fmt]
    org_pct = subset['has_organic'].mean() * 100
    lit_pct = subset['has_lithic'].mean() * 100
    pre_indic = subset['pre_indic_ratio'].mean()
    print(f'  {fmt:15s} (n={len(subset):3d}): organic={org_pct:.1f}%, lithic={lit_pct:.1f}%, pre_indic_ratio={pre_indic:.3f}')

# Test: do short inscriptions have lower organic mention rate?
short = df_t[df_t['format'] == 'short_format']['n_organic']
long = df_t[df_t['format'] == 'long_sima']['n_organic']
u_stat, u_p = stats.mannwhitneyu(short, long, alternative='less')
print(f'  Mann-Whitney U: short < long organic? U={u_stat:.0f}, p={u_p:.4f}')
print(f'  → {"YES: genre (format length) significantly affects organic mention rate" if u_p < 0.05 else "NO: genre effect not significant"}')

# ============================================================
# 7. VISUALIZATION — Multi-domain temporal convergence
# ============================================================
print('\n--- Generating figures ---')

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('E048: Multi-Domain Temporal Convergence in Old Javanese Inscriptions',
             fontsize=14, fontweight='bold')

century_nums = agg['century_num'].values
century_labels = agg.index.tolist()

# Plot 1: Indic vs Pre-Indic ratio
ax = axes[0, 0]
ax.plot(century_nums, agg['indic_ratio'], 'rs-', label='Indic ratio', linewidth=2)
ax.plot(century_nums, agg['mean_pre_indic_ratio'], 'bo-', label='Pre-Indic ratio', linewidth=2)
ax.set_xlabel('Century CE')
ax.set_ylabel('Ratio')
ax.set_title('Indianization Wave')
ax.legend(fontsize=8)
ax.set_xticks(century_nums)
ax.set_xticklabels(century_labels, fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Hyang persistence
ax = axes[0, 1]
ax.bar(century_nums, agg['pct_has_hyang'] * 100, color='purple', alpha=0.7)
ax.set_xlabel('Century CE')
ax.set_ylabel('% inscriptions')
ax.set_title('Hyang (PMP *qiang) Persistence')
ax.set_xticks(century_nums)
ax.set_xticklabels(century_labels, fontsize=8)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# Plot 3: Organic vs Lithic
ax = axes[0, 2]
ax.plot(century_nums, agg['mean_organic'], 'g^-', label='Organic mentions', linewidth=2)
ax.plot(century_nums, agg['mean_lithic'], 'kv-', label='Lithic mentions', linewidth=2)
ax.set_xlabel('Century CE')
ax.set_ylabel('Mean count per inscription')
ax.set_title('Material Culture Mentions')
ax.legend(fontsize=8)
ax.set_xticks(century_nums)
ax.set_xticklabels(century_labels, fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: Organic ratio over time
ax = axes[1, 0]
ax.plot(century_nums, agg['mean_organic_ratio'] * 100, 'g-o', linewidth=2)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
ax.set_xlabel('Century CE')
ax.set_ylabel('Organic / (Organic+Lithic) %')
ax.set_title('Organic Dominance Over Time')
ax.set_xticks(century_nums)
ax.set_xticklabels(century_labels, fontsize=8)
ax.set_ylim(0, 100)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 5: Botanical diversity
ax = axes[1, 1]
ax.plot(century_nums, agg['mean_n_plants'], 'gD-', linewidth=2, color='darkgreen')
ax.set_xlabel('Century CE')
ax.set_ylabel('Mean plant species per inscription')
ax.set_title('Botanical Diversity')
ax.set_xticks(century_nums)
ax.set_xticklabels(century_labels, fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 6: Inscription count (sample size context)
ax = axes[1, 2]
ax.bar(century_nums, agg['n_inscriptions'], color='steelblue', alpha=0.7)
ax.set_xlabel('Century CE')
ax.set_ylabel('Count')
ax.set_title('Sample Size by Century')
ax.set_xticks(century_nums)
ax.set_xticklabels(century_labels, fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS, 'multidomain_convergence.png'), dpi=300, bbox_inches='tight')
print(f'  Saved: multidomain_convergence.png')

# ============================================================
# 8. CORRELATION HEATMAP
# ============================================================
# Century-level correlation matrix
corr_cols = ['indic_ratio', 'mean_pre_indic_ratio', 'pct_has_hyang',
             'mean_organic', 'mean_organic_ratio', 'mean_n_plants']
corr_labels = ['Indic ratio', 'Pre-Indic ratio', 'Hyang %',
               'Organic count', 'Organic ratio', 'Plant diversity']

# Inscription-level correlations
inscr_cols = ['pre_indic_ratio', 'n_organic', 'n_lithic', 'n_plants', 'word_count']
inscr_labels = ['Pre-Indic ratio', 'Organic count', 'Lithic count', 'Plant count', 'Word count']

valid_cols = [c for c in inscr_cols if c in df_t.columns]
valid_labels = [inscr_labels[i] for i, c in enumerate(inscr_cols) if c in df_t.columns]

corr_matrix = df_t[valid_cols].corr(method='spearman')

fig2, ax2 = plt.subplots(figsize=(8, 6))
im = ax2.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax2.set_xticks(range(len(valid_labels)))
ax2.set_yticks(range(len(valid_labels)))
ax2.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(valid_labels, fontsize=9)

# Add correlation values
for i in range(len(valid_labels)):
    for j in range(len(valid_labels)):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)

plt.colorbar(im, ax=ax2, label='Spearman rho')
ax2.set_title('E048: Cross-Domain Correlation Matrix\n(166 dated inscriptions)', fontsize=12)
fig2.savefig(os.path.join(RESULTS, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
print(f'  Saved: correlation_heatmap.png')

# ============================================================
# 9. SYNTHESIS
# ============================================================
print('\n' + '=' * 60)
print('SYNTHESIS')
print('=' * 60)

# Key finding: does de-Sanskritization correlate with organic prominence?
valid = agg[['indic_ratio', 'mean_organic_ratio']].dropna()
if len(valid) >= 4:
    rho, p = stats.spearmanr(valid['indic_ratio'], valid['mean_organic_ratio'])
    print(f'\n  KEY TEST: Indic ratio ↔ Organic ratio')
    print(f'  Spearman rho = {rho:+.3f}, p = {p:.4f}')
    if rho < 0 and p < 0.1:
        print(f'  → CONVERGENCE: As Indianization declines, organic material prominence INCREASES')
        print(f'  → The "wave" is visible in BOTH linguistic AND material dimensions')
    elif rho < 0:
        print(f'  → Direction consistent but not significant (small N = {len(valid)} centuries)')
    else:
        print(f'  → No convergence detected')

# Summary statistics
print(f'\n  Total inscriptions analyzed: {len(df_t)} (dated)')
print(f'  Century range: C{df_t["century"].min()} to C{df_t["century"].max()}')
print(f'  Mean pre-Indic ratio: {df_t["pre_indic_ratio"].mean():.3f}')
print(f'  Mean organic mentions: {df_t["n_organic"].mean():.1f} per inscription')
if 'n_plants' in df_t.columns:
    print(f'  Mean plant species: {df_t["n_plants"].mean():.1f} per inscription')

print('\nDone!')
