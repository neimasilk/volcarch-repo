#!/usr/bin/env python3
"""
E033 — The Indianization Curve: Sanskrit Vocabulary in Prasasti (550-1356 CE)

Question: What is the SHAPE of Indianization in the epigraphic record?
          Is it monotonic growth, peak-and-decline, or wave?

Analyses:
  A. Language proportion by century (Sanskrit vs Old Javanese vs Old Malay)
  B. Indic keyword ratio: indic / (indic + pre_indic) per century
  C. Indic keyword density per 100 words (length-controlled)
  D. Keyword-level temporal heatmap
  E. Borobudur sensitivity test (48 labels dominate 8th c.)

Extends: E030 (pre-Indic ratio increases, rho=+0.50). E033 inverts the lens.

Data: E023 DHARMA corpus classification (268 inscriptions, 166 dated)
Author: VOLCARCH project (AI-assisted)
Date: 2026-03-10
"""

import sys
import io
import os
import json
import warnings

# Windows cp1252 console fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E023_RESULTS = os.path.join(REPO, "experiments", "E023_ritual_screening", "results")
E030_RESULTS = os.path.join(REPO, "experiments", "E030_prasasti_temporal_nlp", "results")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("E033 — The Indianization Curve")
print("Sanskrit Vocabulary Penetration in Prasasti (550-1356 CE)")
print("=" * 70)

# ═════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════════════════

print("\n[1] Loading data...")

# E030 already extracted dates — use its dated_inscriptions.csv
dated_csv = os.path.join(E030_RESULTS, "dated_inscriptions.csv")
df = pd.read_csv(dated_csv)
print(f"  Dated inscriptions from E030: {len(df)}")
print(f"  Date range: {int(df['year_ce'].min())}-{int(df['year_ce'].max())} CE")

# Merge ritual_keywords from E023 inventory for keyword-level analysis
inventory_csv = os.path.join(E023_RESULTS, "dharma_corpus_inventory.csv")
df_inv = pd.read_csv(inventory_csv)
kw_map = dict(zip(df_inv['filename'], df_inv['ritual_keywords']))
df['ritual_keywords_str'] = df['filename'].map(kw_map)

# Keyword classification (from E030/E023 ontology)
PRE_INDIC_TERMS = {'hyang', 'hyaṁ', 'maṅhuri', 'kabuyutan', 'wuku',
                   'karāman', 'panumbas', 'gunung'}
INDIC_TERMS = {'homa', 'pūjā', 'puja', 'mantra', 'svarga', 'svargga',
               'kalpa', 'piṇḍa', 'pitr', 'pralaya', 'nakṣatra', 'tithi',
               'vāra', 'śrāddha', 'danu'}
AMBIGUOUS_TERMS = {'sīma', 'śapatha', 'sapatha', 'samudra', 'samgat',
                   'parvvata', 'saka', 'śaka', 'atīta'}


def parse_keywords(kw_str):
    if not isinstance(kw_str, str) or kw_str.strip() == '':
        return []
    return [k.strip() for k in kw_str.split('|') if k.strip()]


df['all_keywords'] = df['ritual_keywords_str'].apply(parse_keywords)

# Flag Borobudur labels (they distort 8th century statistics)
df['is_borobudur'] = df['filename'].str.contains('Borobudur', case=False, na=False)
n_boro = df['is_borobudur'].sum()
print(f"  Borobudur labels: {n_boro} (all 8th c., san-Latn, 1-6 words)")

# Century assignment (already in data, verify)
df['century'] = ((df['year_ce'] - 1) // 100 + 1).astype(int)

# Inscriptions per century
print(f"\n  Inscriptions per century:")
for c, count in sorted(df['century'].value_counts().items()):
    n_boro_c = df[(df['century'] == c) & df['is_borobudur']].shape[0]
    boro_note = f" ({n_boro_c} Borobudur)" if n_boro_c > 0 else ""
    print(f"    C{c}: {count}{boro_note}")


# ═════════════════════════════════════════════════════════════════════════
# 2. ANALYSIS A: Language Proportion by Century
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[2] ANALYSIS A: Language Proportion by Century")
print("=" * 70)

# Standardize language labels
lang_map = {
    'san-Latn': 'Sanskrit',
    'kaw-Latn': 'Old Javanese',
    'omy-Latn': 'Old Malay',
    'osn-Latn': 'Old Sundanese',
    'unknown': 'Unknown'
}
df['lang_label'] = df['lang'].map(lang_map).fillna('Unknown')

centuries = sorted(df['century'].unique())

# Count per language per century
lang_counts = {}
for c in centuries:
    sub = df[df['century'] == c]
    counts = sub['lang_label'].value_counts()
    total = len(sub)
    lang_counts[c] = {
        'total': total,
        'Sanskrit': counts.get('Sanskrit', 0),
        'Old Javanese': counts.get('Old Javanese', 0),
        'Old Malay': counts.get('Old Malay', 0),
        'Other': counts.get('Old Sundanese', 0) + counts.get('Unknown', 0),
        'Sanskrit_pct': counts.get('Sanskrit', 0) / total * 100 if total > 0 else 0,
        'OJ_pct': counts.get('Old Javanese', 0) / total * 100 if total > 0 else 0,
    }

print("\n  Language distribution:")
print(f"  {'Century':<10} {'Sanskrit':>10} {'Old Jav':>10} {'Old Malay':>10} {'Other':>8} {'Total':>8}")
print("  " + "-" * 58)
for c in centuries:
    lc = lang_counts[c]
    print(f"  C{c:<9} {lc['Sanskrit']:>10} {lc['Old Javanese']:>10} "
          f"{lc['Old Malay']:>10} {lc['Other']:>8} {lc['total']:>8}")

# WITHOUT Borobudur
print("\n  Language distribution (excluding Borobudur labels):")
df_nb = df[~df['is_borobudur']].copy()
for c in centuries:
    sub = df_nb[df_nb['century'] == c]
    if len(sub) == 0:
        continue
    san_pct = (sub['lang_label'] == 'Sanskrit').sum() / len(sub) * 100
    oj_pct = (sub['lang_label'] == 'Old Javanese').sum() / len(sub) * 100
    print(f"    C{c}: Sanskrit={san_pct:.0f}%, OJ={oj_pct:.0f}%, n={len(sub)}")


# ═════════════════════════════════════════════════════════════════════════
# 3. ANALYSIS B: Indic Keyword Ratio Per Century
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[3] ANALYSIS B: Indic Keyword Ratio (The Indianization Curve)")
print("=" * 70)

# indic_ratio = indic / (indic + pre_indic) for inscriptions with >= 1 keyword
# This is 1 - pre_indic_ratio but only for inscriptions WITH classified keywords

df['has_classified'] = (df['indic'] + df['pre_indic']) > 0
df_kw = df[df['has_classified']].copy()
df_kw['indic_ratio'] = df_kw['indic'] / (df_kw['indic'] + df_kw['pre_indic'])

print(f"\n  Inscriptions with >= 1 classified keyword: {len(df_kw)}/{len(df)}")

# Also without Borobudur
df_kw_nb = df_kw[~df_kw['is_borobudur']].copy()
print(f"  Without Borobudur: {len(df_kw_nb)}")

# Spearman correlation: indic_ratio vs year_ce
rho_all, p_all = stats.spearmanr(df_kw['year_ce'], df_kw['indic_ratio'])
rho_nb, p_nb = stats.spearmanr(df_kw_nb['year_ce'], df_kw_nb['indic_ratio'])

print(f"\n  Spearman (all): rho={rho_all:.3f}, p={p_all:.2e}, n={len(df_kw)}")
print(f"  Spearman (no Borobudur): rho={rho_nb:.3f}, p={p_nb:.2e}, n={len(df_kw_nb)}")

# Century-level statistics with bootstrap CI
def bootstrap_ci(data, n_boot=2000, ci=95):
    """Bootstrap confidence interval for the mean."""
    if len(data) < 2:
        m = np.mean(data)
        return m, m, m
    boot_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(data), lo, hi


np.random.seed(42)
century_stats = {}
century_stats_nb = {}

print(f"\n  Century breakdown (all inscriptions with keywords):")
print(f"  {'Century':<10} {'Mean':>8} {'95% CI':>18} {'n':>5}")
print("  " + "-" * 45)
for c in centuries:
    sub = df_kw[df_kw['century'] == c]
    if len(sub) == 0:
        continue
    vals = sub['indic_ratio'].values
    mean, lo, hi = bootstrap_ci(vals)
    century_stats[c] = {'mean': mean, 'lo': lo, 'hi': hi, 'n': len(sub)}
    print(f"  C{c:<9} {mean:>8.3f} [{lo:.3f}, {hi:.3f}]{len(sub):>5}")

    # Without Borobudur
    sub_nb = df_kw_nb[df_kw_nb['century'] == c]
    if len(sub_nb) > 0:
        vals_nb = sub_nb['indic_ratio'].values
        mean_nb, lo_nb, hi_nb = bootstrap_ci(vals_nb)
        century_stats_nb[c] = {'mean': mean_nb, 'lo': lo_nb, 'hi': hi_nb, 'n': len(sub_nb)}


# ═════════════════════════════════════════════════════════════════════════
# 4. ANALYSIS C: Indic Keyword Density (Length-Controlled)
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[4] ANALYSIS C: Indic Keyword Density (per 100 words)")
print("=" * 70)

# Only for inscriptions with word_count > 0
df_len = df[df['word_count'] > 10].copy()  # exclude very short labels
df_len['indic_density'] = df_len['indic'] / df_len['word_count'] * 100
df_len['pre_indic_density'] = df_len['pre_indic'] / df_len['word_count'] * 100

# Without Borobudur
df_len_nb = df_len[~df_len['is_borobudur']].copy()

rho_d, p_d = stats.spearmanr(df_len_nb['year_ce'], df_len_nb['indic_density'])
print(f"\n  Indic density vs year (no Borobudur, word_count>10):")
print(f"  Spearman: rho={rho_d:.3f}, p={p_d:.2e}, n={len(df_len_nb)}")

print(f"\n  Century means (no Borobudur, word_count > 10):")
for c in centuries:
    sub = df_len_nb[df_len_nb['century'] == c]
    if len(sub) == 0:
        continue
    print(f"    C{c}: indic_density={sub['indic_density'].mean():.2f}/100w, "
          f"pre_indic_density={sub['pre_indic_density'].mean():.2f}/100w, n={len(sub)}")


# ═════════════════════════════════════════════════════════════════════════
# 5. ANALYSIS D: Keyword-Level Temporal Profile
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[5] ANALYSIS D: Keyword-Level Temporal Profile")
print("=" * 70)

# For each keyword, count appearances per century
all_terms = sorted(INDIC_TERMS | PRE_INDIC_TERMS | AMBIGUOUS_TERMS)

# Build keyword × century matrix
kw_century = pd.DataFrame(0, index=all_terms, columns=centuries)

for _, row in df.iterrows():
    c = row['century']
    for kw in row['all_keywords']:
        kw_clean = kw.strip().lower()
        # Match against known terms
        for term in all_terms:
            if kw_clean == term.lower() or kw_clean == term:
                kw_century.loc[term, c] += 1
                break

# Print non-zero terms
print("\n  Keyword appearances per century (non-zero only):")
for term in all_terms:
    row_vals = kw_century.loc[term]
    if row_vals.sum() > 0:
        origin = 'PRE' if term in PRE_INDIC_TERMS else ('IND' if term in INDIC_TERMS else 'AMB')
        vals_str = '  '.join([f"C{c}:{v}" for c, v in row_vals.items() if v > 0])
        print(f"    [{origin}] {term:<15} total={row_vals.sum():>3}  {vals_str}")

# Compute unique Indic terms per century (term diversity)
print("\n  Unique Indic terms per century:")
for c in centuries:
    indic_in_c = [t for t in INDIC_TERMS if kw_century.loc[t, c] > 0]
    pre_in_c = [t for t in PRE_INDIC_TERMS if kw_century.loc[t, c] > 0]
    n_insc = len(df[df['century'] == c])
    print(f"    C{c}: {len(indic_in_c)} Indic terms, {len(pre_in_c)} pre-Indic terms "
          f"(from {n_insc} inscriptions)")


# ═════════════════════════════════════════════════════════════════════════
# 6. ANALYSIS E: Borobudur Sensitivity Test
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[6] ANALYSIS E: Borobudur Sensitivity Test")
print("=" * 70)

# How much does removing Borobudur change the 8th century picture?
c8_all = df[df['century'] == 8]
c8_nb = df_nb[df_nb['century'] == 8]
c8_boro = df[df['is_borobudur']]

print(f"\n  8th century with Borobudur: {len(c8_all)} inscriptions")
print(f"    Sanskrit: {(c8_all['lang_label']=='Sanskrit').sum()}, "
      f"OJ: {(c8_all['lang_label']=='Old Javanese').sum()}")
print(f"    Mean word count: {c8_all['word_count'].mean():.1f}")

print(f"\n  8th century WITHOUT Borobudur: {len(c8_nb)} inscriptions")
if len(c8_nb) > 0:
    print(f"    Sanskrit: {(c8_nb['lang_label']=='Sanskrit').sum()}, "
          f"OJ: {(c8_nb['lang_label']=='Old Javanese').sum()}")
    print(f"    Mean word count: {c8_nb['word_count'].mean():.1f}")

print(f"\n  Borobudur labels ({len(c8_boro)} total):")
print(f"    All Sanskrit: {(c8_boro['lang_label']=='Sanskrit').sum() == len(c8_boro)}")
print(f"    Word count range: {c8_boro['word_count'].min()}-{c8_boro['word_count'].max()}")
print(f"    Keywords: {c8_boro['total_keywords'].sum()} total")


# ═════════════════════════════════════════════════════════════════════════
# 7. POLITICAL ERA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[7] Political Era Analysis")
print("=" * 70)

# Define eras (Borobudur excluded from all)
eras = {
    'Srivijaya/Medang (550-929)': (550, 929),
    'East Java: Kahuripan-Kadiri (929-1222)': (929, 1222),
    'Singhasari-Majapahit (1222-1400)': (1222, 1400),
}

for era_name, (y0, y1) in eras.items():
    sub = df_kw_nb[(df_kw_nb['year_ce'] >= y0) & (df_kw_nb['year_ce'] < y1)]
    if len(sub) == 0:
        print(f"\n  {era_name}: no data")
        continue
    mean_ir = sub['indic_ratio'].mean()
    mean_pr = sub['pre_indic_ratio'].mean()
    print(f"\n  {era_name}:")
    print(f"    n={len(sub)}, mean indic_ratio={mean_ir:.3f}, mean pre_indic_ratio={mean_pr:.3f}")
    # What are the dominant Indic terms?
    all_kw_era = [kw for kws in sub['all_keywords'] for kw in kws]
    indic_kw = [kw for kw in all_kw_era if kw in INDIC_TERMS]
    pre_indic_kw = [kw for kw in all_kw_era if kw in PRE_INDIC_TERMS]
    if indic_kw:
        from collections import Counter
        top_indic = Counter(indic_kw).most_common(5)
        print(f"    Top Indic terms: {top_indic}")
    if pre_indic_kw:
        from collections import Counter
        top_pre = Counter(pre_indic_kw).most_common(5)
        print(f"    Top pre-Indic terms: {top_pre}")

# Mann-Whitney: early vs late Indic ratio
early = df_kw_nb[df_kw_nb['year_ce'] < 929]['indic_ratio']
late = df_kw_nb[df_kw_nb['year_ce'] >= 929]['indic_ratio']
if len(early) > 0 and len(late) > 0:
    u_stat, u_p = stats.mannwhitneyu(early, late, alternative='two-sided')
    print(f"\n  Mann-Whitney (929 CE split, no Borobudur):")
    print(f"    Early (<929): median={early.median():.3f}, n={len(early)}")
    print(f"    Late (>=929): median={late.median():.3f}, n={len(late)}")
    print(f"    U={u_stat:.0f}, p={u_p:.4f}")


# ═════════════════════════════════════════════════════════════════════════
# 8. VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[8] Generating visualizations...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E033 — The Indianization Curve\nSanskrit Vocabulary in Prasasti (550-1356 CE)',
             fontsize=14, fontweight='bold', y=0.98)

# ── Panel A: Language Distribution ────────────────────────────────────────
ax = axes[0, 0]
san_pcts = []
oj_pcts = []
om_pcts = []
other_pcts = []

# Use no-Borobudur data
for c in centuries:
    sub = df_nb[df_nb['century'] == c]
    total = len(sub) if len(sub) > 0 else 1
    san_pcts.append((sub['lang_label'] == 'Sanskrit').sum() / total * 100)
    oj_pcts.append((sub['lang_label'] == 'Old Javanese').sum() / total * 100)
    om_pcts.append((sub['lang_label'] == 'Old Malay').sum() / total * 100)
    other_pcts.append(100 - san_pcts[-1] - oj_pcts[-1] - om_pcts[-1])

x = np.arange(len(centuries))
width = 0.7
ax.bar(x, san_pcts, width, label='Sanskrit', color='#e74c3c', alpha=0.8)
ax.bar(x, oj_pcts, width, bottom=san_pcts, label='Old Javanese', color='#2ecc71', alpha=0.8)
bottom2 = [s + o for s, o in zip(san_pcts, oj_pcts)]
ax.bar(x, om_pcts, width, bottom=bottom2, label='Old Malay', color='#3498db', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'C{c}' for c in centuries], fontsize=9)
ax.set_ylabel('Proportion (%)')
ax.set_title('A. Language of Inscription\n(excl. Borobudur labels)', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 105)

# Add sample sizes
for i, c in enumerate(centuries):
    n = len(df_nb[df_nb['century'] == c])
    ax.text(i, 102, f'n={n}', ha='center', fontsize=7, color='gray')

# ── Panel B: Indic Keyword Ratio (The Main Curve) ────────────────────────
ax = axes[0, 1]

# Scatter individual inscriptions (no Borobudur)
ax.scatter(df_kw_nb['year_ce'], df_kw_nb['indic_ratio'],
           alpha=0.25, s=15, color='#e74c3c', zorder=2)

# Century means with CI
cs_nb = [c for c in centuries if c in century_stats_nb]
means_nb = [century_stats_nb[c]['mean'] for c in cs_nb]
lo_nb = [century_stats_nb[c]['lo'] for c in cs_nb]
hi_nb = [century_stats_nb[c]['hi'] for c in cs_nb]
century_midpoints = [c * 100 - 50 for c in cs_nb]

ax.plot(century_midpoints, means_nb, 's-', color='#c0392b', markersize=8,
        linewidth=2, zorder=4, label='Century mean')
ax.fill_between(century_midpoints, lo_nb, hi_nb, alpha=0.2, color='#e74c3c',
                zorder=3, label='95% bootstrap CI')

# Political era markers
for year, label in [(929, 'Court→E.Java'), (1222, 'Singhasari'), (1293, 'Majapahit')]:
    ax.axvline(year, color='gray', linestyle=':', alpha=0.5, zorder=1)
    ax.text(year + 5, 0.98, label, fontsize=7, color='gray', rotation=90,
            va='top', ha='left')

ax.set_xlabel('Year CE')
ax.set_ylabel('Indic Ratio (indic / [indic + pre-Indic])')
ax.set_title('B. The Indianization Curve\n(excl. Borobudur, keywords only)', fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)
ax.text(0.03, 0.05, f'rho={rho_nb:.3f}, p={p_nb:.1e}',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# ── Panel C: Keyword Heatmap ─────────────────────────────────────────────
ax = axes[1, 0]

# Select terms that appear at least once
active_terms = [t for t in all_terms if kw_century.loc[t].sum() > 0]

# Separate by origin for visual grouping
indic_active = [t for t in active_terms if t in INDIC_TERMS]
pre_indic_active = [t for t in active_terms if t in PRE_INDIC_TERMS]
ambig_active = [t for t in active_terms if t in AMBIGUOUS_TERMS]
ordered_terms = pre_indic_active + ['---'] + indic_active + ['---'] + ambig_active

# Build heatmap matrix
heatmap_data = np.zeros((len(ordered_terms), len(centuries)))
term_labels = []
for i, term in enumerate(ordered_terms):
    if term == '---':
        term_labels.append('─' * 8)
        continue
    term_labels.append(term)
    for j, c in enumerate(centuries):
        heatmap_data[i, j] = kw_century.loc[term, c]

im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
ax.set_xticks(range(len(centuries)))
ax.set_xticklabels([f'C{c}' for c in centuries], fontsize=8)
ax.set_yticks(range(len(term_labels)))
ax.set_yticklabels(term_labels, fontsize=7)
ax.set_title('C. Keyword Temporal Profile\n(frequency per century)', fontsize=11)

# Add count text on cells
for i in range(len(ordered_terms)):
    for j in range(len(centuries)):
        val = int(heatmap_data[i, j])
        if val > 0:
            color = 'white' if val > heatmap_data.max() * 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=7, color=color)

# Color bar for origin
for i, term in enumerate(ordered_terms):
    if term == '---' or term.startswith('─'):
        continue
    if term in PRE_INDIC_TERMS:
        ax.text(-0.6, i, '●', color='#2ecc71', fontsize=8, ha='center', va='center')
    elif term in INDIC_TERMS:
        ax.text(-0.6, i, '●', color='#e74c3c', fontsize=8, ha='center', va='center')
    else:
        ax.text(-0.6, i, '●', color='#f39c12', fontsize=8, ha='center', va='center')

# ── Panel D: Density Comparison (length-controlled) ──────────────────────
ax = axes[1, 1]

# Per-century indic and pre-indic density
density_data = {'century': [], 'indic_density': [], 'pre_indic_density': []}
for c in centuries:
    sub = df_len_nb[df_len_nb['century'] == c]
    if len(sub) < 2:
        continue
    density_data['century'].append(c)
    density_data['indic_density'].append(sub['indic_density'].mean())
    density_data['pre_indic_density'].append(sub['pre_indic_density'].mean())

if density_data['century']:
    x_d = np.arange(len(density_data['century']))
    width_d = 0.35
    ax.bar(x_d - width_d/2, density_data['indic_density'], width_d,
           label='Indic density', color='#e74c3c', alpha=0.8)
    ax.bar(x_d + width_d/2, density_data['pre_indic_density'], width_d,
           label='Pre-Indic density', color='#2ecc71', alpha=0.8)
    ax.set_xticks(x_d)
    ax.set_xticklabels([f'C{c}' for c in density_data['century']], fontsize=9)

ax.set_ylabel('Keywords per 100 words')
ax.set_title('D. Keyword Density (length-controlled)\n(excl. Borobudur, word_count > 10)', fontsize=11)
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(RESULTS_DIR, 'indianization_curve_4panel.png'), dpi=150,
            bbox_inches='tight')
print("  Saved: indianization_curve_4panel.png")

# ── Standalone headline figure (Panel B only, publication quality) ────────
fig2, ax2 = plt.subplots(figsize=(10, 5))

# Scatter individual
ax2.scatter(df_kw_nb['year_ce'], df_kw_nb['indic_ratio'],
            alpha=0.2, s=20, color='#e74c3c', zorder=2, label='Individual inscription')

# Century means
ax2.plot(century_midpoints, means_nb, 's-', color='#c0392b', markersize=10,
         linewidth=2.5, zorder=4, label='Century mean')
ax2.fill_between(century_midpoints, lo_nb, hi_nb, alpha=0.15, color='#e74c3c',
                 zorder=3, label='95% bootstrap CI')

# Annotations
for year, label, va in [(929, 'Court shift\nto East Java', 'top'),
                         (1222, 'Rise of\nSinghasari', 'top'),
                         (1293, 'Majapahit\nfounded', 'top')]:
    ax2.axvline(year, color='gray', linestyle='--', alpha=0.4, zorder=1)
    ax2.text(year + 8, 0.95 if va == 'top' else 0.05, label,
             fontsize=9, color='#555', va=va, ha='left')

ax2.set_xlabel('Year CE', fontsize=12)
ax2.set_ylabel('Indic Ratio\n(Indic keywords / [Indic + Pre-Indic])', fontsize=11)
ax2.set_title('The Indianization Curve: Sanskrit Vocabulary in Old Javanese Inscriptions\n'
              '(DHARMA corpus, N=166 dated prasasti, excl. Borobudur labels)',
              fontsize=12, fontweight='bold')
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlim(530, 1380)
ax2.legend(fontsize=9, loc='lower left')
ax2.text(0.97, 0.05, f'Spearman rho={rho_nb:.3f}\np={p_nb:.1e}',
         transform=ax2.transAxes, fontsize=10, ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig2.savefig(os.path.join(RESULTS_DIR, 'indianization_curve_headline.png'), dpi=200,
             bbox_inches='tight')
print("  Saved: indianization_curve_headline.png")

plt.close('all')


# ═════════════════════════════════════════════════════════════════════════
# 9. STRUCTURED OUTPUT
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[9] Saving structured results...")
print("=" * 70)

results = {
    'experiment': 'E033_sanskrit_temporal_curve',
    'title': 'The Indianization Curve',
    'date': '2026-03-10',
    'extends': 'E030_prasasti_temporal_nlp',
    'data_source': 'E023 DHARMA corpus (268 inscriptions, 166 dated)',
    'total_inscriptions': len(df),
    'borobudur_labels': int(n_boro),
    'date_range_ce': [int(df['year_ce'].min()), int(df['year_ce'].max())],

    'analysis_A_language_proportion': {
        'description': 'Language of inscription per century (excl. Borobudur)',
        'centuries': {str(c): lang_counts[c] for c in centuries}
    },

    'analysis_B_indianization_curve': {
        'description': 'Indic keyword ratio per century',
        'spearman_all': {'rho': round(rho_all, 4), 'p': float(f'{p_all:.4e}'), 'n': len(df_kw)},
        'spearman_no_borobudur': {'rho': round(rho_nb, 4), 'p': float(f'{p_nb:.4e}'), 'n': len(df_kw_nb)},
        'century_means_no_borobudur': {
            str(c): {'mean': round(v['mean'], 4), 'ci_lo': round(v['lo'], 4),
                     'ci_hi': round(v['hi'], 4), 'n': v['n']}
            for c, v in century_stats_nb.items()
        },
        'interpretation': ('Indianization curve to be determined by results. '
                           'Inverse of E030 pre-Indic ratio trend.')
    },

    'analysis_C_density': {
        'description': 'Indic keyword density per 100 words (length-controlled)',
        'spearman_no_borobudur': {'rho': round(rho_d, 4), 'p': float(f'{p_d:.4e}'),
                                  'n': len(df_len_nb)},
    },

    'analysis_D_keyword_profile': {
        'description': 'Individual keyword appearances per century',
        'n_active_terms': len(active_terms),
    },

    'analysis_E_borobudur_sensitivity': {
        'n_borobudur': int(n_boro),
        'c8_with': len(c8_all),
        'c8_without': len(c8_nb),
        'conclusion': 'Borobudur labels inflate 8th century Sanskrit proportion'
    },
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(os.path.join(RESULTS_DIR, 'indianization_summary.json'), 'w',
          encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
print("  Saved: indianization_summary.json")


# ═════════════════════════════════════════════════════════════════════════
# 10. HEADLINE FINDING
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HEADLINE FINDING")
print("=" * 70)

if rho_nb < 0:
    direction = "DECLINES"
    shape = "Indianization is a WAVE, not a permanent transformation"
elif rho_nb > 0:
    direction = "INCREASES"
    shape = "Indianization shows cumulative vocabulary adoption"
else:
    direction = "shows NO TREND"
    shape = "Indianization vocabulary is stable across centuries"

print(f"""
  Indic keyword ratio {direction} over time (rho={rho_nb:.3f}, p={p_nb:.1e})

  Shape: {shape}

  Key implication for VOLCARCH:
  - If declining: supports P5/P15 thesis that Sanskrit = overlay, not replacement
  - If stable: coexistence model — Indic and pre-Indic vocabularies parallel
  - If increasing: standard Indianization narrative, but E030 showed pre-Indic
    vocabulary ALSO increases — suggesting BOTH expand simultaneously

  Borobudur effect: {n_boro} labels inflate 8th century Sanskrit; removing them
  may change the curve shape significantly.
""")

print("=" * 70)
print("E033 COMPLETE")
print("=" * 70)
