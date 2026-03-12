"""
E062 — Temporal Synthesis: Multi-Dimensional Visibility Curve
=============================================================
Joins classifications from E023, E030, E035, E040 into a single
temporal model showing how indigenous visibility changes across centuries.

H1: All indigenous markers increase together (correlation matrix)
H2: A single "visibility" PC explains >50% variance
H3: Visibility curve peaks C10-C11, not C14
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BASE, '..', '..'))
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

E023 = os.path.join(REPO, 'experiments', 'E023_ritual_screening', 'results', 'full_corpus_classification.csv')
E030 = os.path.join(REPO, 'experiments', 'E030_prasasti_temporal_nlp', 'results', 'dated_inscriptions.csv')
E035 = os.path.join(REPO, 'experiments', 'E035_prasasti_botanical', 'results', 'botanical_inscriptions.csv')
E040 = os.path.join(REPO, 'experiments', 'E040_bamboo_civilization', 'results', 'material_culture_inscriptions.csv')

# ---------------------------------------------------------------------------
# 1. Load & Join
# ---------------------------------------------------------------------------
print('=' * 72)
print('E062 — TEMPORAL SYNTHESIS: MULTI-DIMENSIONAL VISIBILITY CURVE')
print('=' * 72)

df_e023 = pd.read_csv(E023)
df_e030 = pd.read_csv(E030)
df_e035 = pd.read_csv(E035)
df_e040 = pd.read_csv(E040)

print(f'\nLoaded datasets:')
print(f'  E023 (ritual screening):    {len(df_e023)} rows')
print(f'  E030 (dated inscriptions):  {len(df_e030)} rows')
print(f'  E035 (botanical):           {len(df_e035)} rows')
print(f'  E040 (material culture):    {len(df_e040)} rows')

# Use E030 as the base (166 dated inscriptions with century info)
# Join E023 columns (pre_indic_ratio, has_hyang, has_manhuri, has_wuku, word_count)
# Join E035 columns (n_plants, has_ritual_context)
# Join E040 columns (n_organic, n_lithic, n_metal)

# E030 already has pre_indic_ratio, has_hyang, has_manhuri, has_wuku, word_count
# But E023 has the full 268-row classification; E030 is a subset with dates.
# We'll use E030 as the base and supplement from E035 and E040.

# Rename to avoid collisions on join
e035_cols = df_e035[['filename', 'n_plants', 'has_ritual_context']].copy()
e040_cols = df_e040[['filename', 'n_organic', 'n_lithic', 'n_metal']].copy()

# Start from E030 (has century, year_ce, and the ritual columns)
df = df_e030.copy()

# Join botanical
df = df.merge(e035_cols, on='filename', how='left')

# Join material culture
df = df.merge(e040_cols, on='filename', how='left')

# Fill NaN for inscriptions not in E035 (if any)
df['n_plants'] = df['n_plants'].fillna(0).astype(int)
df['has_ritual_context'] = df['has_ritual_context'].fillna(False).astype(bool)
df['n_organic'] = df['n_organic'].fillna(0).astype(int)
df['n_lithic'] = df['n_lithic'].fillna(0).astype(int)
df['n_metal'] = df['n_metal'].fillna(0).astype(int)

# Convert boolean columns to int for analysis
df['has_hyang_int'] = df['has_hyang'].astype(int)
df['has_wuku_int'] = df['has_wuku'].astype(int)
df['has_manhuri_int'] = df['has_manhuri'].astype(int)
df['has_ritual_int'] = df['has_ritual_context'].astype(int)

print(f'\nJoined dataset: {len(df)} dated inscriptions')
print(f'  Columns: {list(df.columns)}')

# Check join completeness
n_with_plants = df['n_plants'].notna().sum()
n_with_organic = df['n_organic'].notna().sum()
print(f'  With botanical data: {n_with_plants}')
print(f'  With material data:  {n_with_organic}')

# Save joined dataset
df.to_csv(os.path.join(RESULTS, 'joined_dated_inscriptions.csv'), index=False)
print(f'\nSaved: results/joined_dated_inscriptions.csv')

# ---------------------------------------------------------------------------
# 2. Per-Century Averages
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('PER-CENTURY AVERAGES')
print('=' * 72)

dimensions = ['pre_indic_ratio', 'n_plants', 'n_organic', 'n_lithic',
              'has_hyang_int', 'has_wuku_int', 'has_manhuri_int',
              'word_count', 'has_ritual_int', 'n_metal']

century_stats = df.groupby('century').agg(
    n=('filename', 'count'),
    pre_indic_ratio=('pre_indic_ratio', 'mean'),
    pre_indic_ratio_std=('pre_indic_ratio', 'std'),
    n_plants=('n_plants', 'mean'),
    n_organic=('n_organic', 'mean'),
    n_lithic=('n_lithic', 'mean'),
    n_metal=('n_metal', 'mean'),
    has_hyang=('has_hyang_int', 'mean'),
    has_wuku=('has_wuku_int', 'mean'),
    has_manhuri=('has_manhuri_int', 'mean'),
    word_count=('word_count', 'mean'),
    has_ritual=('has_ritual_int', 'mean'),
).reset_index()

print(f'\nCentury distribution (n inscriptions):')
for _, row in century_stats.iterrows():
    print(f'  C{int(row.century):2d}: n={int(row.n):3d}  '
          f'pre_indic={row.pre_indic_ratio:.3f}  '
          f'plants={row.n_plants:.2f}  '
          f'organic={row.n_organic:.2f}  '
          f'lithic={row.n_lithic:.2f}  '
          f'hyang={row.has_hyang:.2f}  '
          f'wuku={row.has_wuku:.2f}  '
          f'words={row.word_count:.0f}')

century_stats.to_csv(os.path.join(RESULTS, 'century_averages.csv'), index=False)
print(f'\nSaved: results/century_averages.csv')

# ---------------------------------------------------------------------------
# 3. Correlation Matrix (H1)
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('H1: CORRELATION MATRIX — Do indigenous markers co-vary?')
print('=' * 72)

analysis_cols = ['pre_indic_ratio', 'n_plants', 'n_organic', 'n_lithic',
                 'has_hyang_int', 'has_wuku_int', 'word_count']
labels_short = ['Pre-Indic\nRatio', 'N Plants', 'N Organic', 'N Lithic',
                'Has Hyang', 'Has Wuku', 'Word Count']

corr_data = df[analysis_cols].copy()
corr_matrix = corr_data.corr(method='spearman')

print('\nSpearman correlation matrix:')
print(corr_matrix.round(3).to_string())

# Count significant positive correlations among indigenous markers
indigenous_cols = ['pre_indic_ratio', 'n_plants', 'n_organic', 'has_hyang_int', 'has_wuku_int']
n_sig_pos = 0
n_pairs = 0
print('\nPairwise tests (indigenous markers only):')
for i in range(len(indigenous_cols)):
    for j in range(i + 1, len(indigenous_cols)):
        rho, p = stats.spearmanr(df[indigenous_cols[i]], df[indigenous_cols[j]])
        sig = '*' if p < 0.05 else ' '
        direction = '+' if rho > 0 else '-'
        print(f'  {indigenous_cols[i]:20s} × {indigenous_cols[j]:20s}: rho={rho:+.3f}  p={p:.4f} {sig}')
        n_pairs += 1
        if p < 0.05 and rho > 0:
            n_sig_pos += 1

print(f'\nH1 result: {n_sig_pos}/{n_pairs} indigenous pairs show significant positive correlation')
h1_verdict = 'SUPPORTED' if n_sig_pos > n_pairs / 2 else ('PARTIAL' if n_sig_pos > 0 else 'REJECTED')
print(f'H1 verdict: {h1_verdict}')

# ---------------------------------------------------------------------------
# 4. PCA (H2)
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('H2: PCA — Is there a single visibility component (>50% variance)?')
print('=' * 72)

pca_cols = ['pre_indic_ratio', 'n_plants', 'n_organic', 'has_hyang_int', 'has_wuku_int', 'word_count']
pca_labels = ['Pre-Indic Ratio', 'N Plants', 'N Organic', 'Has Hyang', 'Has Wuku', 'Word Count']

X = df[pca_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

print('\nPCA explained variance ratios:')
cumvar = 0
for i, (ev, cumv) in enumerate(zip(pca.explained_variance_ratio_,
                                     np.cumsum(pca.explained_variance_ratio_))):
    cumvar = cumv
    print(f'  PC{i+1}: {ev:.3f} (cumulative: {cumv:.3f})')

pc1_var = pca.explained_variance_ratio_[0]
h2_verdict = 'SUPPORTED' if pc1_var > 0.50 else ('PARTIAL — PC1 largest but <50%' if pc1_var > 0.30 else 'REJECTED')
print(f'\nPC1 explains {pc1_var:.1%} of variance')
print(f'H2 verdict: {h2_verdict}')

print('\nPC1 loadings (visibility axis):')
loadings = pca.components_[0]
for name, loading in sorted(zip(pca_labels, loadings), key=lambda x: -abs(x[1])):
    direction = '+' if loading > 0 else '-'
    print(f'  {name:20s}: {loading:+.3f} ({direction}indigenous)')

# Compute PC1 scores for each inscription
df['pc1_score'] = X_scaled @ pca.components_[0]
# Also compute for century averages
century_pc1 = df.groupby('century')['pc1_score'].mean().reset_index()
century_pc1.columns = ['century', 'visibility_score']

# ---------------------------------------------------------------------------
# 5. Temporal Trends (H3)
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('H3: VISIBILITY CURVE — Does it peak C10-C11?')
print('=' * 72)

# Spearman correlation of each dimension with century
print('\nTemporal trends (Spearman rho with century):')
temporal_results = {}
for col, label in zip(pca_cols + ['pc1_score'], pca_labels + ['Visibility (PC1)']):
    rho, p = stats.spearmanr(df['century'], df[col])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f'  {label:20s}: rho={rho:+.3f}  p={p:.4f} {sig}')
    temporal_results[col] = (rho, p)

# Find peak century for visibility
peak_row = century_pc1.loc[century_pc1['visibility_score'].idxmax()]
peak_century = int(peak_row['century'])
peak_score = peak_row['visibility_score']
print(f'\nPeak visibility century: C{peak_century} (score={peak_score:.3f})')
print(f'\nVisibility scores by century:')
for _, row in century_pc1.sort_values('century').iterrows():
    bar = '#' * int(max(0, (row.visibility_score + 2) * 10))
    print(f'  C{int(row.century):2d}: {row.visibility_score:+.3f}  {bar}')

# Test: is peak in C10-C11?
h3_verdict = 'SUPPORTED' if peak_century in [10, 11] else f'PARTIAL — peak at C{peak_century}'
print(f'\nH3 verdict: {h3_verdict}')

# Merge visibility score into century_stats
century_stats = century_stats.merge(century_pc1, on='century', how='left')
century_stats.to_csv(os.path.join(RESULTS, 'century_averages.csv'), index=False)

# ---------------------------------------------------------------------------
# 6. Additional: Organic/Lithic ratio over time
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('SUPPLEMENTARY: ORGANIC vs LITHIC RATIO OVER TIME')
print('=' * 72)

df['organic_ratio'] = df['n_organic'] / (df['n_organic'] + df['n_lithic']).replace(0, np.nan)
century_organic = df.groupby('century')['organic_ratio'].mean()
print('\nOrganic ratio (organic / (organic+lithic)) by century:')
for c, r in century_organic.items():
    print(f'  C{int(c):2d}: {r:.3f}')

org_valid = df[['century', 'organic_ratio']].dropna()
rho_org, p_org = stats.spearmanr(org_valid['century'], org_valid['organic_ratio'])
print(f'\nOrganic ratio × century: rho={rho_org:+.3f}, p={p_org:.4f}')

# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('GENERATING FIGURES')
print('=' * 72)

# Filter to centuries with enough data (n >= 3)
cs = century_stats[century_stats['n'] >= 3].copy()

# --- Figure 1: Multi-panel temporal plot ---
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
fig.suptitle('E062 — Multi-Dimensional Temporal Profile of Old Javanese Inscriptions',
             fontsize=14, fontweight='bold', y=0.98)

panels = [
    ('pre_indic_ratio', 'Pre-Indic Keyword Ratio', 'tab:blue', axes[0, 0]),
    ('n_plants', 'Mean Plants per Inscription', 'tab:green', axes[0, 1]),
    ('n_organic', 'Mean Organic Materials', 'tab:orange', axes[1, 0]),
    ('n_lithic', 'Mean Lithic Materials', 'tab:gray', axes[1, 1]),
    ('has_hyang', 'Fraction with hyang', 'tab:red', axes[2, 0]),
    ('word_count', 'Mean Word Count', 'tab:purple', axes[2, 1]),
]

for col, ylabel, color, ax in panels:
    ax.bar(cs['century'], cs[col], color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(ylabel, fontsize=11, fontweight='bold')
    # Annotate n on top of each bar
    for _, row in cs.iterrows():
        val = row[col]
        if pd.notna(val):
            ax.text(row['century'], val, f'n={int(row["n"])}',
                    ha='center', va='bottom', fontsize=7, color='gray')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

axes[2, 0].set_xlabel('Century CE', fontsize=11)
axes[2, 1].set_xlabel('Century CE', fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(RESULTS, 'fig1_temporal_multipanel.png'), dpi=200, bbox_inches='tight')
print('Saved: results/fig1_temporal_multipanel.png')
plt.close(fig)

# --- Figure 2: Correlation heatmap ---
fig2, ax2 = plt.subplots(figsize=(9, 8))
corr_display = corr_data.rename(columns=dict(zip(analysis_cols, labels_short))).corr(method='spearman')
im = ax2.imshow(corr_display.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax2.set_xticks(range(len(labels_short)))
ax2.set_yticks(range(len(labels_short)))
ax2.set_xticklabels(labels_short, fontsize=9, rotation=45, ha='right')
ax2.set_yticklabels(labels_short, fontsize=9)
# Annotate cells
for i in range(len(labels_short)):
    for j in range(len(labels_short)):
        val = corr_display.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)
plt.colorbar(im, ax=ax2, shrink=0.8, label='Spearman rho')
ax2.set_title('E062 — Spearman Correlation Heatmap\n(166 dated inscriptions)', fontsize=13, fontweight='bold')
fig2.savefig(os.path.join(RESULTS, 'fig2_correlation_heatmap.png'), dpi=200, bbox_inches='tight')
print('Saved: results/fig2_correlation_heatmap.png')
plt.close(fig2)

# --- Figure 3: PCA biplot ---
fig3, ax3 = plt.subplots(figsize=(10, 8))
# Project inscriptions onto PC1-PC2
scores = X_scaled @ pca.components_[:2].T

# Color by century
centuries = df['century'].values
unique_centuries = sorted(df['century'].dropna().unique())
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(unique_centuries), vmax=max(unique_centuries))

sc = ax3.scatter(scores[:, 0], scores[:, 1], c=centuries, cmap=cmap, norm=norm,
                 alpha=0.6, s=30, edgecolors='gray', linewidths=0.3)
plt.colorbar(sc, ax=ax3, label='Century CE', shrink=0.8)

# Loading arrows
arrow_scale = 3
for i, label in enumerate(pca_labels):
    ax3.annotate('',
                 xy=(pca.components_[0, i] * arrow_scale, pca.components_[1, i] * arrow_scale),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax3.text(pca.components_[0, i] * arrow_scale * 1.15,
             pca.components_[1, i] * arrow_scale * 1.15,
             label, fontsize=9, color='red', ha='center', va='center',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

ax3.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax3.axvline(0, color='gray', linestyle='--', linewidth=0.5)
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax3.set_title('E062 — PCA Biplot: Indigenous Visibility Dimensions\n(166 dated inscriptions)',
              fontsize=13, fontweight='bold')
fig3.savefig(os.path.join(RESULTS, 'fig3_pca_biplot.png'), dpi=200, bbox_inches='tight')
print('Saved: results/fig3_pca_biplot.png')
plt.close(fig3)

# --- Figure 4: Visibility Curve (composite PC1 score by century) ---
fig4, ax4 = plt.subplots(figsize=(10, 6))
cs_vis = cs.sort_values('century')

bars = ax4.bar(cs_vis['century'], cs_vis['visibility_score'],
               color=['#d62728' if c == peak_century else '#1f77b4' for c in cs_vis['century']],
               edgecolor='black', linewidth=0.5, alpha=0.8, width=0.7)

# Add trend line
z = np.polyfit(cs_vis['century'], cs_vis['visibility_score'], 2)
p_fit = np.poly1d(z)
x_smooth = np.linspace(cs_vis['century'].min(), cs_vis['century'].max(), 100)
ax4.plot(x_smooth, p_fit(x_smooth), 'k--', linewidth=1.5, alpha=0.5, label='Quadratic fit')

# Annotate n and score
for _, row in cs_vis.iterrows():
    ypos = row['visibility_score']
    va = 'bottom' if ypos >= 0 else 'top'
    ax4.text(row['century'], ypos, f'{ypos:+.2f}\nn={int(row["n"])}',
             ha='center', va=va, fontsize=8, fontweight='bold')

ax4.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax4.set_xlabel('Century CE', fontsize=12)
ax4.set_ylabel('Visibility Score (PC1 mean)', fontsize=12)
ax4.set_title('E062 — The Visibility Curve: Composite Indigenous Visibility by Century',
              fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig4.savefig(os.path.join(RESULTS, 'fig4_visibility_curve.png'), dpi=200, bbox_inches='tight')
print('Saved: results/fig4_visibility_curve.png')
plt.close(fig4)

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('SUMMARY')
print('=' * 72)
print(f'''
Dataset: {len(df)} dated Old Javanese inscriptions (C{int(df.century.min())}-C{int(df.century.max())})
Dimensions joined: ritual (E023/E030), botanical (E035), material (E040)

H1 (co-variation): {h1_verdict}
  {n_sig_pos}/{n_pairs} indigenous pairs show significant positive Spearman correlation

H2 (single visibility component): {h2_verdict}
  PC1 explains {pc1_var:.1%} of total variance
  Top PC1 loadings: {", ".join(f"{n}={l:+.3f}" for n, l in sorted(zip(pca_labels, loadings), key=lambda x: -abs(x[1]))[:3])}

H3 (peak C10-C11): {h3_verdict}
  Peak century: C{peak_century} (visibility score = {peak_score:+.3f})

Visibility score by century:''')
for _, row in century_pc1.sort_values('century').iterrows():
    n_row = century_stats.loc[century_stats['century'] == row['century'], 'n']
    n_val = int(n_row.values[0]) if len(n_row) > 0 else '?'
    print(f'  C{int(row.century):2d} (n={n_val:>3}): {row.visibility_score:+.3f}')

print(f'''
Key insight: Indigenous content visibility is driven primarily by inscription
length (word_count) and the shift from short Sanskrit-style dedications to
long Old Javanese sima grants. The "visibility" is partly a genre effect.

Files saved:
  results/joined_dated_inscriptions.csv
  results/century_averages.csv
  results/fig1_temporal_multipanel.png
  results/fig2_correlation_heatmap.png
  results/fig3_pca_biplot.png
  results/fig4_visibility_curve.png
''')
