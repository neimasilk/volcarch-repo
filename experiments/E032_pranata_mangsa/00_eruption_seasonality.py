#!/usr/bin/env python3
"""
E032 — Pranata Mangsa × Eruption Seasonality

Question: Does the traditional Javanese agricultural calendar (Pranata Mangsa)
          encode empirical knowledge of volcanic eruption seasonality?

Hypothesis: If eruptions cluster in specific months, and those months align
            with Pranata Mangsa season boundaries or "danger" periods, the
            calendar may encode volcanic hazard knowledge accumulated over
            centuries.

Method:
  1. Extract month from GVP eruption start_dates (where available)
  2. Test for non-uniform monthly distribution (Rayleigh test, circular stats)
  3. Map eruptions to Pranata Mangsa seasons (12 mangsa)
  4. Compare eruption density per mangsa
  5. Per-volcano seasonal patterns (Merapi, Kelud, Bromo, Semeru)

Data: GVP eruption_history.csv (168 records, Java volcanoes)
      Pranata Mangsa dates (traditional, well-documented)

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

# ── Paths ─────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ERUPTION_CSV = os.path.join(REPO, "data", "processed", "eruption_history.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("E032 — Pranata Mangsa x Eruption Seasonality")
print("Does the Javanese calendar encode volcanic hazard knowledge?")
print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════
# 1. PRANATA MANGSA DEFINITION
# ═════════════════════════════════════════════════════════════════════════

# Traditional Javanese agricultural calendar (12 mangsa)
# Dates are approximate; based on solar position, traditionally fixed.
# Source: Daldjoeni (1984), Ammarell (1988), Wisnubroto (1999)
PRANATA_MANGSA = [
    {'mangsa': 1,  'name': 'Kasa',      'start': (6, 22), 'end': (8, 1),
     'days': 41, 'meaning': 'Dry season begins, leaves fall, land burned',
     'season': 'dry'},
    {'mangsa': 2,  'name': 'Karo',      'start': (8, 2),  'end': (8, 24),
     'days': 23, 'meaning': 'Peak dry, land cleared for planting',
     'season': 'dry'},
    {'mangsa': 3,  'name': 'Katelu',    'start': (8, 25), 'end': (9, 17),
     'days': 24, 'meaning': 'Caterpillars emerge, soil prepared',
     'season': 'dry'},
    {'mangsa': 4,  'name': 'Kapat',     'start': (9, 18), 'end': (10, 12),
     'days': 25, 'meaning': 'Springs dry up, winds shift SE-NW',
     'season': 'transition'},
    {'mangsa': 5,  'name': 'Kalima',    'start': (10, 13), 'end': (11, 8),
     'days': 27, 'meaning': 'First rain expected, planting begins',
     'season': 'transition'},
    {'mangsa': 6,  'name': 'Kanem',     'start': (11, 9),  'end': (12, 21),
     'days': 43, 'meaning': 'Rainy season starts, rice seedlings planted',
     'season': 'wet'},
    {'mangsa': 7,  'name': 'Kapitu',    'start': (12, 22), 'end': (2, 2),
     'days': 43, 'meaning': 'Peak rain, flooding risk, storms',
     'season': 'wet'},
    {'mangsa': 8,  'name': 'Kawolu',    'start': (2, 3),   'end': (2, 28),
     'days': 26, 'meaning': 'Insects emerge, caterpillar pest, rice grows',
     'season': 'wet'},
    {'mangsa': 9,  'name': 'Kasanga',   'start': (3, 1),   'end': (3, 25),
     'days': 25, 'meaning': 'Fruit season begins, some harvest',
     'season': 'wet'},
    {'mangsa': 10, 'name': 'Kasepuluh', 'start': (3, 26),  'end': (4, 18),
     'days': 24, 'meaning': 'Main harvest period',
     'season': 'transition'},
    {'mangsa': 11, 'name': 'Desta',     'start': (4, 19),  'end': (5, 11),
     'days': 23, 'meaning': 'Harvest continues, some second planting',
     'season': 'transition'},
    {'mangsa': 12, 'name': 'Saddha',    'start': (5, 12),  'end': (6, 21),
     'days': 41, 'meaning': 'Dry season returns, cold nights',
     'season': 'dry'},
]


def month_day_to_doy(month, day):
    """Convert month/day to day-of-year (non-leap)."""
    import datetime
    return datetime.date(2001, month, day).timetuple().tm_yday


def date_to_mangsa(month, day=15):
    """Assign a month/day to its Pranata Mangsa period."""
    doy = month_day_to_doy(month, min(day, 28))
    for m in PRANATA_MANGSA:
        s_doy = month_day_to_doy(m['start'][0], m['start'][1])
        e_doy = month_day_to_doy(m['end'][0], m['end'][1])
        if s_doy <= e_doy:
            if s_doy <= doy <= e_doy:
                return m['mangsa'], m['name']
        else:  # wraps around year boundary (Kapitu: Dec 22 - Feb 2)
            if doy >= s_doy or doy <= e_doy:
                return m['mangsa'], m['name']
    return None, None


# ═════════════════════════════════════════════════════════════════════════
# 2. LOAD AND FILTER ERUPTION DATA
# ═════════════════════════════════════════════════════════════════════════

print("\n[1] Loading eruption data...")

df = pd.read_csv(ERUPTION_CSV)
print(f"  Total eruption records: {len(df)}")
print(f"  Volcanoes: {df['volcano'].nunique()} ({', '.join(sorted(df['volcano'].unique()))})")

# Extract month from start_date
def extract_month(date_str):
    """Extract month from start_date (format YYYY-MM-DD or YYYY-MM-00)."""
    if not isinstance(date_str, str):
        return None
    parts = date_str.split('-')
    if len(parts) >= 2:
        try:
            month = int(parts[1])
            if 1 <= month <= 12:
                return month
        except (ValueError, IndexError):
            pass
    return None


def extract_day(date_str):
    """Extract day from start_date."""
    if not isinstance(date_str, str):
        return None
    parts = date_str.split('-')
    if len(parts) >= 3:
        try:
            day = int(parts[2])
            if 1 <= day <= 31:
                return day
        except (ValueError, IndexError):
            pass
    return None


df['month'] = df['start_date'].apply(extract_month)
df['day'] = df['start_date'].apply(extract_day)

# Filter to eruptions with known month
df_m = df[df['month'].notna()].copy()
df_m['month'] = df_m['month'].astype(int)
df_m['day'] = df_m['day'].fillna(15).astype(int)

print(f"  Eruptions with month data: {len(df_m)}/{len(df)}")
print(f"  Year range (with month): {df_m['year'].min()} - {df_m['year'].max()}")

# Summary per volcano
print(f"\n  Per-volcano eruptions (with month data):")
for v, count in df_m['volcano'].value_counts().items():
    print(f"    {v}: {count}")


# ═════════════════════════════════════════════════════════════════════════
# 3. MONTHLY DISTRIBUTION (ALL JAVA VOLCANOES)
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[2] Monthly Distribution of Java Volcanic Eruptions")
print("=" * 70)

month_counts = df_m['month'].value_counts().sort_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print("\n  Monthly eruption frequency:")
for m in range(1, 13):
    count = month_counts.get(m, 0)
    bar = '#' * count
    print(f"    {month_names[m-1]:>3}: {count:>3} {bar}")

# Expected if uniform
n_total = len(df_m)
expected = n_total / 12
print(f"\n  Total eruptions: {n_total}")
print(f"  Expected per month (uniform): {expected:.1f}")

# Chi-squared test for uniformity
observed = [month_counts.get(m, 0) for m in range(1, 13)]
chi2, p_chi2 = stats.chisquare(observed)
print(f"\n  Chi-squared test for uniform distribution:")
print(f"    chi2={chi2:.2f}, p={p_chi2:.4f}")
if p_chi2 < 0.05:
    print(f"    → REJECT uniformity: eruptions cluster in specific months")
else:
    print(f"    → Cannot reject uniformity: no significant seasonal pattern")


# ═════════════════════════════════════════════════════════════════════════
# 4. RAYLEIGH TEST (Circular Statistics)
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[3] Circular Statistics (Rayleigh Test)")
print("=" * 70)

# Convert months to circular angles (month 1 = Jan = 0 radians, month 12 = Dec)
angles = df_m['month'].values * 2 * np.pi / 12  # 0 to 2pi

# Mean resultant length (R-bar)
C = np.mean(np.cos(angles))
S = np.mean(np.sin(angles))
R_bar = np.sqrt(C**2 + S**2)
mean_angle = np.arctan2(S, C) % (2 * np.pi)
mean_month = mean_angle * 12 / (2 * np.pi)

# Rayleigh test: Z = n * R_bar^2
n = len(angles)
Z = n * R_bar**2
# p-value approximation: p ≈ exp(-Z) for large n
p_rayleigh = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))
p_rayleigh = max(0, min(1, p_rayleigh))

print(f"\n  n = {n} eruptions")
print(f"  Mean resultant length (R-bar) = {R_bar:.4f}")
print(f"  Mean direction = {mean_month:.1f} (month index, 1-indexed)")
print(f"  Mean month ≈ {month_names[int(mean_month) % 12]}")
print(f"  Rayleigh Z = {Z:.2f}")
print(f"  Rayleigh p ≈ {p_rayleigh:.4f}")

if p_rayleigh < 0.05:
    print(f"  → SIGNIFICANT directionality: eruptions cluster around {month_names[int(mean_month) % 12]}")
else:
    print(f"  → No significant directionality")


# ═════════════════════════════════════════════════════════════════════════
# 5. PRANATA MANGSA MAPPING
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[4] Mapping Eruptions to Pranata Mangsa Seasons")
print("=" * 70)

# Assign each eruption to a mangsa
df_m['mangsa_num'], df_m['mangsa_name'] = zip(
    *df_m.apply(lambda r: date_to_mangsa(r['month'], r['day']), axis=1)
)

# Count per mangsa
mangsa_counts = {}
for m_info in PRANATA_MANGSA:
    num = m_info['mangsa']
    name = m_info['name']
    count = len(df_m[df_m['mangsa_num'] == num])
    days = m_info['days']
    # Density: eruptions per day of mangsa (normalizes for unequal period lengths)
    density = count / days * 30  # normalized to per-30-day-equivalent
    mangsa_counts[num] = {
        'name': name, 'count': count, 'days': days,
        'density': density, 'season': m_info['season'],
        'meaning': m_info['meaning']
    }

print(f"\n  {'Mangsa':<12} {'Name':<12} {'Season':<12} {'Count':>6} {'Days':>5} "
      f"{'Density':>8} {'Meaning'}")
print("  " + "-" * 95)
for num in range(1, 13):
    mc = mangsa_counts[num]
    print(f"  {num:<12} {mc['name']:<12} {mc['season']:<12} {mc['count']:>6} "
          f"{mc['days']:>5} {mc['density']:>8.2f} {mc['meaning'][:40]}")

# Which mangsa has the highest density?
max_mangsa = max(mangsa_counts.items(), key=lambda x: x[1]['density'])
min_mangsa = min(mangsa_counts.items(), key=lambda x: x[1]['density'])
print(f"\n  Highest density: Mangsa {max_mangsa[0]} ({max_mangsa[1]['name']}) "
      f"= {max_mangsa[1]['density']:.2f}/30d")
print(f"  Lowest density:  Mangsa {min_mangsa[0]} ({min_mangsa[1]['name']}) "
      f"= {min_mangsa[1]['density']:.2f}/30d")
print(f"  Ratio: {max_mangsa[1]['density']/max(min_mangsa[1]['density'], 0.01):.1f}x")

# Dry vs wet season comparison
dry_counts = sum(mc['count'] for mc in mangsa_counts.values() if mc['season'] == 'dry')
wet_counts = sum(mc['count'] for mc in mangsa_counts.values() if mc['season'] == 'wet')
trans_counts = sum(mc['count'] for mc in mangsa_counts.values() if mc['season'] == 'transition')
dry_days = sum(mc['days'] for mc in mangsa_counts.values() if mc['season'] == 'dry')
wet_days = sum(mc['days'] for mc in mangsa_counts.values() if mc['season'] == 'wet')
trans_days = sum(mc['days'] for mc in mangsa_counts.values() if mc['season'] == 'transition')

print(f"\n  Season summary:")
print(f"    Dry (Kasa+Karo+Katelu+Saddha):     {dry_counts} eruptions in {dry_days} days "
      f"({dry_counts/dry_days*365:.1f}/yr-equiv)")
print(f"    Wet (Kanem+Kapitu+Kawolu+Kasanga):   {wet_counts} eruptions in {wet_days} days "
      f"({wet_counts/wet_days*365:.1f}/yr-equiv)")
print(f"    Transition (Kapat+Kalima+Kasepuluh+Desta): {trans_counts} eruptions in {trans_days} days "
      f"({trans_counts/trans_days*365:.1f}/yr-equiv)")

# Chi-squared: dry vs wet vs transition (normalized by days)
obs_season = [dry_counts, wet_counts, trans_counts]
exp_season = [n_total * dry_days/365, n_total * wet_days/365, n_total * trans_days/365]
chi2_s, p_s = stats.chisquare(obs_season, f_exp=exp_season)
print(f"\n  Chi-squared (seasonal, day-normalized):")
print(f"    chi2={chi2_s:.2f}, p={p_s:.4f}")


# ═════════════════════════════════════════════════════════════════════════
# 6. PER-VOLCANO SEASONAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[5] Per-Volcano Seasonal Patterns")
print("=" * 70)

# Focus on volcanoes with >= 5 dated eruptions
for volcano in sorted(df_m['volcano'].unique()):
    sub = df_m[df_m['volcano'] == volcano]
    if len(sub) < 5:
        continue

    # Circular stats
    angles_v = sub['month'].values * 2 * np.pi / 12
    C_v = np.mean(np.cos(angles_v))
    S_v = np.mean(np.sin(angles_v))
    R_v = np.sqrt(C_v**2 + S_v**2)
    mean_v = np.arctan2(S_v, C_v) % (2 * np.pi) * 12 / (2 * np.pi)
    Z_v = len(sub) * R_v**2
    p_v = np.exp(-Z_v)  # simplified

    # Monthly distribution
    m_dist = sub['month'].value_counts().sort_index()
    peak_month = m_dist.idxmax()
    peak_count = m_dist.max()

    # Mangsa distribution
    mangsa_dist = sub['mangsa_name'].value_counts()
    peak_mangsa = mangsa_dist.idxmax() if not mangsa_dist.empty else 'N/A'

    print(f"\n  {volcano} (n={len(sub)}):")
    print(f"    Mean month: {month_names[int(mean_v) % 12]} (R={R_v:.3f})")
    print(f"    Peak month: {month_names[peak_month-1]} ({peak_count} eruptions)")
    print(f"    Peak mangsa: {peak_mangsa}")
    print(f"    Rayleigh Z={Z_v:.2f}, p≈{p_v:.4f}"
          f"{' ← SIGNIFICANT' if p_v < 0.05 else ''}")
    # Monthly bars
    for m in range(1, 13):
        c = m_dist.get(m, 0)
        if c > 0:
            print(f"      {month_names[m-1]:>3}: {'#' * c} ({c})")


# ═════════════════════════════════════════════════════════════════════════
# 7. KEY FINDING: TRANSITION SEASON ALIGNMENT
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[6] Key Finding: Transition Season Analysis")
print("=" * 70)

# Pranata Mangsa "dangerous" transitions:
# - Mangsa 4 (Kapat): "springs dry up, winds shift" — monsoon transition
# - Mangsa 5 (Kalima): "first rain expected" — dry-to-wet transition
# - Mangsa 6 (Kanem): "rainy season starts" — wet onset
# These transition months (Sep-Dec) coincide with monsoon shift
# which can interact with volcanic plume dispersal

# Do eruptions concentrate in the dry-to-wet transition (Kapat-Kalima-Kanem)?
transition_to_wet = sum(1 for _, r in df_m.iterrows()
                       if r['mangsa_num'] in [4, 5, 6])
transition_days_tw = sum(mc['days'] for num, mc in mangsa_counts.items() if num in [4, 5, 6])
other = n_total - transition_to_wet
other_days = 365 - transition_days_tw

expected_tw = n_total * transition_days_tw / 365
print(f"\n  Dry-to-wet transition (Kapat+Kalima+Kanem, Sep-Dec):")
print(f"    Observed: {transition_to_wet} eruptions in {transition_days_tw} days")
print(f"    Expected (uniform): {expected_tw:.1f}")
print(f"    Ratio: {transition_to_wet/max(expected_tw,1):.2f}x expected")

# Binomial test
btest = stats.binomtest(transition_to_wet, n_total, transition_days_tw/365,
                        alternative='greater')
p_binom = btest.pvalue
print(f"    Binomial test (one-sided, excess): p={p_binom:.4f}")


# ═════════════════════════════════════════════════════════════════════════
# 8. VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[7] Generating visualizations...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E032 — Pranata Mangsa × Eruption Seasonality\n'
             'Does the Javanese Calendar Encode Volcanic Hazard Knowledge?',
             fontsize=13, fontweight='bold', y=0.98)

# ── Panel A: Monthly eruption frequency ───────────────────────────────────
ax = axes[0, 0]
months = range(1, 13)
counts = [month_counts.get(m, 0) for m in months]

# Color by season
season_colors = {
    'dry': '#e67e22',       # orange
    'wet': '#3498db',       # blue
    'transition': '#2ecc71' # green
}
# Map each month to its primary mangsa season
month_season_map = {}
for m_info in PRANATA_MANGSA:
    start_m = m_info['start'][0]
    end_m = m_info['end'][0]
    # Simple: assign season to start month
    if start_m <= end_m:
        for mm in range(start_m, end_m + 1):
            month_season_map[mm] = m_info['season']
    else:  # wraps (Kapitu: Dec-Feb)
        for mm in range(start_m, 13):
            month_season_map[mm] = m_info['season']
        for mm in range(1, end_m + 1):
            month_season_map[mm] = m_info['season']

bar_colors = [season_colors.get(month_season_map.get(m, 'dry'), 'gray') for m in months]
ax.bar(months, counts, color=bar_colors, edgecolor='white', linewidth=0.5)
ax.axhline(expected, color='red', linestyle='--', alpha=0.5, label=f'Expected (uniform): {expected:.1f}')
ax.set_xticks(months)
ax.set_xticklabels(month_names, fontsize=9)
ax.set_ylabel('Number of Eruptions')
ax.set_title('A. Monthly Eruption Frequency\n(Java volcanoes, GVP data)', fontsize=11)
ax.legend(fontsize=8)

# Add season legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e67e22', label='Dry'),
                   Patch(facecolor='#2ecc71', label='Transition'),
                   Patch(facecolor='#3498db', label='Wet')]
ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color='red', linestyle='--',
          label=f'Expected: {expected:.1f}')], fontsize=7, loc='upper left')

# ── Panel B: Pranata Mangsa eruption density ──────────────────────────────
ax = axes[0, 1]
mangsa_nums = range(1, 13)
densities = [mangsa_counts[m]['density'] for m in mangsa_nums]
mangsa_names = [mangsa_counts[m]['name'] for m in mangsa_nums]
mangsa_seasons = [mangsa_counts[m]['season'] for m in mangsa_nums]
bar_colors_m = [season_colors[s] for s in mangsa_seasons]

ax.bar(mangsa_nums, densities, color=bar_colors_m, edgecolor='white', linewidth=0.5)
ax.set_xticks(mangsa_nums)
ax.set_xticklabels(mangsa_names, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Eruptions per 30-day equivalent')
ax.set_title('B. Eruption Density by Pranata Mangsa\n(normalized for unequal period lengths)',
             fontsize=11)

# Add raw counts
for i, num in enumerate(mangsa_nums):
    ax.text(num, densities[i] + 0.1, f'n={mangsa_counts[num]["count"]}',
            ha='center', fontsize=7, color='gray')

# ── Panel C: Circular/Rose plot ───────────────────────────────────────────
ax = axes[1, 0]
ax.remove()
ax = fig.add_subplot(2, 2, 3, projection='polar')

# Rose histogram (monthly)
theta = np.array([(m - 0.5) * 2 * np.pi / 12 for m in range(1, 13)])
width = 2 * np.pi / 12
radii = [month_counts.get(m, 0) for m in range(1, 13)]
bars = ax.bar(theta, radii, width=width * 0.9, alpha=0.7,
              color=[season_colors.get(month_season_map.get(m, 'dry'), 'gray')
                     for m in range(1, 13)],
              edgecolor='white', linewidth=0.5)

# Labels
ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
ax.set_xticklabels(month_names, fontsize=8)

# Mean direction arrow
ax.annotate('', xy=(mean_angle, max(radii) * 0.9),
            xytext=(mean_angle, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.set_title(f'C. Circular Distribution\n(mean≈{month_names[int(mean_month) % 12]}, '
             f'R={R_bar:.3f}, p={p_rayleigh:.3f})',
             fontsize=11, pad=20)

# ── Panel D: Per-volcano comparison ───────────────────────────────────────
ax = axes[1, 1]

# Volcanoes with enough data
major_volcanoes = [v for v in df_m['volcano'].unique()
                   if len(df_m[df_m['volcano'] == v]) >= 5]
major_volcanoes = sorted(major_volcanoes)

if major_volcanoes:
    volcano_monthly = {}
    for v in major_volcanoes:
        sub = df_m[df_m['volcano'] == v]
        monthly = [len(sub[sub['month'] == m]) for m in range(1, 13)]
        volcano_monthly[v] = monthly

    x = np.arange(12)
    n_vol = len(major_volcanoes)
    width_v = 0.8 / n_vol
    colors_v = plt.cm.Set2(np.linspace(0, 1, n_vol))

    for i, v in enumerate(major_volcanoes):
        ax.bar(x + i * width_v - 0.4 + width_v/2, volcano_monthly[v],
               width_v, label=v, color=colors_v[i], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(month_names, fontsize=8)
    ax.set_ylabel('Number of Eruptions')
    ax.set_title('D. Per-Volcano Monthly Patterns\n(volcanoes with ≥5 dated eruptions)',
                 fontsize=11)
    ax.legend(fontsize=7, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(RESULTS_DIR, 'pranata_mangsa_4panel.png'), dpi=150,
            bbox_inches='tight')
print("  Saved: pranata_mangsa_4panel.png")

# ── Standalone headline figure ────────────────────────────────────────────
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Monthly with Pranata Mangsa overlay
ax2a.bar(months, counts, color=bar_colors, edgecolor='white', linewidth=0.5, zorder=2)
ax2a.axhline(expected, color='red', linestyle='--', alpha=0.5, zorder=1)

# Annotate mangsa transitions
for m_info in PRANATA_MANGSA:
    if m_info['mangsa'] in [1, 4, 6, 7, 10]:  # key transitions
        start_m = m_info['start'][0]
        start_d = m_info['start'][1]
        x_pos = start_m + (start_d - 1) / 30 - 0.5
        ax2a.axvline(x_pos, color='purple', alpha=0.3, linestyle=':', zorder=1)
        ax2a.text(x_pos + 0.1, max(counts) * 0.95, m_info['name'],
                  fontsize=7, color='purple', rotation=90, va='top')

ax2a.set_xticks(months)
ax2a.set_xticklabels(month_names, fontsize=10)
ax2a.set_ylabel('Number of Eruptions', fontsize=11)
ax2a.set_xlabel('Month', fontsize=11)
ax2a.set_title('Monthly Eruption Frequency\nwith Pranata Mangsa Boundaries', fontsize=12)
ax2a.legend(handles=legend_elements, fontsize=8, loc='upper left')

# Right: Density by mangsa
ax2b.barh(range(12, 0, -1), densities, color=bar_colors_m,
          edgecolor='white', linewidth=0.5)
ax2b.set_yticks(range(12, 0, -1))
ax2b.set_yticklabels([f'{mangsa_counts[m]["name"]} ({mangsa_counts[m]["season"]})'
                       for m in range(1, 13)], fontsize=9)
ax2b.set_xlabel('Eruptions per 30-day equivalent', fontsize=11)
ax2b.set_title('Eruption Density by Pranata Mangsa\n(length-normalized)', fontsize=12)
ax2b.axvline(n_total / 12 * 30 / (365/12), color='red', linestyle='--', alpha=0.5)

fig2.suptitle('E032 — Pranata Mangsa × Java Volcanic Eruption Seasonality',
              fontsize=13, fontweight='bold', y=1.02)
fig2.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, 'pranata_mangsa_headline.png'), dpi=200,
             bbox_inches='tight')
print("  Saved: pranata_mangsa_headline.png")

plt.close('all')


# ═════════════════════════════════════════════════════════════════════════
# 9. STRUCTURED OUTPUT
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[8] Saving structured results...")
print("=" * 70)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

results = {
    'experiment': 'E032_pranata_mangsa',
    'title': 'Pranata Mangsa x Eruption Seasonality',
    'date': '2026-03-10',
    'n_eruptions_total': len(df),
    'n_eruptions_with_month': len(df_m),

    'monthly_distribution': {
        'counts': {month_names[m-1]: int(month_counts.get(m, 0)) for m in range(1, 13)},
        'chi2': round(chi2, 2),
        'p_chi2': round(p_chi2, 4),
    },

    'circular_statistics': {
        'R_bar': round(R_bar, 4),
        'mean_direction_month': round(float(mean_month), 2),
        'mean_month_name': month_names[int(mean_month) % 12],
        'rayleigh_Z': round(Z, 2),
        'rayleigh_p': round(p_rayleigh, 4),
    },

    'mangsa_density': {
        mangsa_counts[m]['name']: {
            'count': mangsa_counts[m]['count'],
            'days': mangsa_counts[m]['days'],
            'density_per_30d': round(mangsa_counts[m]['density'], 2),
            'season': mangsa_counts[m]['season']
        }
        for m in range(1, 13)
    },

    'season_comparison': {
        'dry': {'count': dry_counts, 'days': dry_days},
        'wet': {'count': wet_counts, 'days': wet_days},
        'transition': {'count': trans_counts, 'days': trans_days},
        'chi2': round(chi2_s, 2),
        'p': round(p_s, 4),
    },
}

with open(os.path.join(RESULTS_DIR, 'seasonality_summary.json'), 'w',
          encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
print("  Saved: seasonality_summary.json")


# ═════════════════════════════════════════════════════════════════════════
# 10. HEADLINE FINDING
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HEADLINE FINDING")
print("=" * 70)

print(f"""
  Monthly chi-squared test: chi2={chi2:.2f}, p={p_chi2:.4f}
  Rayleigh test: R={R_bar:.4f}, p={p_rayleigh:.4f}
  Mean eruption direction: ~{month_names[int(mean_month) % 12]}

  Season comparison (chi2={chi2_s:.2f}, p={p_s:.4f}):
    Dry season:     {dry_counts} eruptions ({dry_counts/n_total*100:.0f}%)
    Wet season:     {wet_counts} eruptions ({wet_counts/n_total*100:.0f}%)
    Transition:     {trans_counts} eruptions ({trans_counts/n_total*100:.0f}%)

  Highest density mangsa: {max_mangsa[1]['name']} ({max_mangsa[1]['density']:.2f}/30d)
  Lowest density mangsa:  {min_mangsa[1]['name']} ({min_mangsa[1]['density']:.2f}/30d)

  If seasonal: Pranata Mangsa may encode volcanic hazard knowledge
  If uniform: Calendar reflects agricultural/monsoon cycles, not volcanic
""")

print("=" * 70)
print("E032 COMPLETE")
print("=" * 70)
