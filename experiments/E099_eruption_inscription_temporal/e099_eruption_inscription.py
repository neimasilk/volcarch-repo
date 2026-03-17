"""
E099 — Eruption Frequency x Inscription Visibility Gradient
============================================================
Tests whether volcanic eruption frequency anti-correlates with
inscription production, demonstrating L6 periodicity is volcano-driven.

Uses: GVP eruption database + DHARMA dated inscriptions (E030).
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import json

print("=" * 70)
print("E099 — ERUPTION FREQUENCY x INSCRIPTION VISIBILITY")
print("=" * 70)

# --- Load data ---
print("\n[1/5] Loading data...")

eruptions = pd.read_csv("data/processed/eruption_history.csv")
inscriptions = pd.read_csv("experiments/E030_prasasti_temporal_nlp/results/dated_inscriptions.csv")

print(f"  Eruptions: {len(eruptions)} records")
print(f"  Inscriptions: {len(inscriptions)} records")

# Filter eruptions to historical period
eruptions_hist = eruptions[(eruptions['year'] >= 0) & (eruptions['year'] <= 1500)].copy()
eruptions_with_vei = eruptions_hist[eruptions_hist['vei'].notna()].copy()
print(f"  Eruptions 0-1500 CE: {len(eruptions_hist)} (with VEI: {len(eruptions_with_vei)})")

# Use ALL eruptions (with or without VEI) for count analysis
eruptions_dated = eruptions_hist.copy()
eruptions_dated['vei'] = eruptions_dated['vei'].fillna(2.0)  # Assign VEI 2 as default for unknowns
print(f"  Using all {len(eruptions_dated)} eruptions (VEI unknown assigned 2.0)")

# Filter inscriptions to dated ones
inscriptions_dated = inscriptions[inscriptions['year_ce'].notna()].copy()
inscriptions_dated = inscriptions_dated[(inscriptions_dated['year_ce'] >= 600) & (inscriptions_dated['year_ce'] <= 1500)]
print(f"  Inscriptions 600-1500 CE: {len(inscriptions_dated)}")

# --- Century-level analysis ---
print("\n[2/5] Century-level analysis...")

# Assign centuries
eruptions_dated['century'] = ((eruptions_dated['year'] - 1) // 100 + 1).astype(int)
inscriptions_dated['century_num'] = inscriptions_dated['century'].astype(str).str.replace('C', '').astype(int)

# Count per century
centuries = range(7, 16)  # C7 to C15
century_eruptions = {}
century_inscriptions = {}
century_vei_sum = {}

for c in centuries:
    mask_e = eruptions_dated['century'] == c
    century_eruptions[c] = mask_e.sum()
    century_vei_sum[c] = eruptions_dated.loc[mask_e, 'vei'].sum()

    mask_i = inscriptions_dated['century_num'] == c
    century_inscriptions[c] = mask_i.sum()

print(f"\n  {'Century':<10} {'Eruptions':>10} {'Sum VEI':>10} {'Inscriptions':>13}")
print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*13}")
for c in centuries:
    print(f"  C{c:<9} {century_eruptions[c]:>10} {century_vei_sum[c]:>10.0f} {century_inscriptions[c]:>13}")

# Correlation: eruption count vs inscription count
e_counts = [century_eruptions[c] for c in centuries]
i_counts = [century_inscriptions[c] for c in centuries]
v_sums = [century_vei_sum[c] for c in centuries]

rho_count, p_count = stats.spearmanr(e_counts, i_counts)
rho_vei, p_vei = stats.spearmanr(v_sums, i_counts)

print(f"\n  Spearman (eruption count vs inscriptions): rho={rho_count:.4f}, p={p_count:.4f}")
print(f"  Spearman (VEI sum vs inscriptions): rho={rho_vei:.4f}, p={p_vei:.4f}")

# --- 50-year bin analysis (finer resolution) ---
print("\n[3/5] 50-year bin analysis...")

bins_50yr = range(600, 1500, 50)
bin_e = {}
bin_i = {}

for b in bins_50yr:
    mask_e = (eruptions_dated['year'] >= b) & (eruptions_dated['year'] < b + 50)
    bin_e[b] = mask_e.sum()

    mask_i = (inscriptions_dated['year_ce'] >= b) & (inscriptions_dated['year_ce'] < b + 50)
    bin_i[b] = mask_i.sum()

bins = sorted(bin_e.keys())
e50 = [bin_e[b] for b in bins]
i50 = [bin_i[b] for b in bins]

rho_50, p_50 = stats.spearmanr(e50, i50)
print(f"  50-year bins: {len(bins)} bins")
print(f"  Spearman (eruptions vs inscriptions): rho={rho_50:.4f}, p={p_50:.4f}")

# --- Lag analysis ---
print("\n[4/5] Lag analysis (does eruption in decade D suppress inscriptions in D+1 to D+5?)...")

# Decade bins
decades = range(600, 1500, 10)
dec_e = {}
dec_i = {}

for d in decades:
    mask_e = (eruptions_dated['year'] >= d) & (eruptions_dated['year'] < d + 10)
    dec_e[d] = mask_e.sum()

    mask_i = (inscriptions_dated['year_ce'] >= d) & (inscriptions_dated['year_ce'] < d + 10)
    dec_i[d] = mask_i.sum()

dec_list = sorted(dec_e.keys())
e_dec = np.array([dec_e[d] for d in dec_list])
i_dec = np.array([dec_i[d] for d in dec_list])

# Test lags 0 to 5 decades
print(f"\n  {'Lag (decades)':<15} {'rho':>8} {'p-value':>10} {'Interpretation'}")
print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*20}")
lag_results = {}
for lag in range(6):
    if lag == 0:
        rho_l, p_l = stats.spearmanr(e_dec, i_dec)
    else:
        rho_l, p_l = stats.spearmanr(e_dec[:-lag], i_dec[lag:])
    lag_results[lag] = {'rho': float(rho_l), 'p': float(p_l)}
    interp = "ANTI-CORR" if rho_l < -0.15 and p_l < 0.1 else "positive" if rho_l > 0.15 else "none"
    print(f"  {lag:<15} {rho_l:>8.4f} {p_l:>10.4f} {interp}")

# --- Volcano-specific analysis ---
print("\n[5/5] Volcano-specific analysis...")

# Major Java volcanoes
major_volcanoes = ['Merapi', 'Kelud', 'Bromo', 'Semeru']
for volcano in major_volcanoes:
    v_mask = eruptions_dated['volcano'].str.contains(volcano, case=False, na=False)
    v_eruptions = eruptions_dated[v_mask]
    if len(v_eruptions) < 3:
        print(f"  {volcano}: too few eruptions ({len(v_eruptions)})")
        continue

    # Count per century
    v_per_century = Counter(v_eruptions['century'])
    v_counts = [v_per_century.get(c, 0) for c in centuries]

    rho_v, p_v = stats.spearmanr(v_counts, i_counts)
    print(f"  {volcano}: {len(v_eruptions)} eruptions, rho={rho_v:.4f}, p={p_v:.4f}")

# --- Eruption-free intervals vs inscription peaks ---
print("\n  Eruption-free intervals (>20yr gap) vs inscription peaks:")
eruptions_sorted = eruptions_dated.sort_values('year')
years = eruptions_sorted['year'].values
gaps = []
for i in range(1, len(years)):
    gap = years[i] - years[i-1]
    if gap > 20 and 600 <= years[i-1] <= 1400:
        # Count inscriptions in the quiet period
        quiet_start = years[i-1]
        quiet_end = years[i]
        n_insc = len(inscriptions_dated[(inscriptions_dated['year_ce'] >= quiet_start) &
                                         (inscriptions_dated['year_ce'] <= quiet_end)])
        rate = n_insc / (gap / 100)  # inscriptions per century-equivalent
        gaps.append({'start': int(quiet_start), 'end': int(quiet_end),
                    'gap_years': int(gap), 'inscriptions': int(n_insc),
                    'rate_per_century': float(rate)})

if gaps:
    gaps_df = pd.DataFrame(gaps)
    print(f"  Found {len(gaps)} quiet periods (>20yr)")
    for _, row in gaps_df.head(10).iterrows():
        print(f"    {row['start']}-{row['end']} ({row['gap_years']}yr): {row['inscriptions']} inscriptions, rate={row['rate_per_century']:.1f}/century")

    # Compare quiet period inscription rate vs active period rate
    total_quiet_years = gaps_df['gap_years'].sum()
    total_quiet_inscr = gaps_df['inscriptions'].sum()
    quiet_rate = total_quiet_inscr / (total_quiet_years / 100)

    total_years = 1400 - 600
    total_inscr = len(inscriptions_dated)
    active_years = total_years - total_quiet_years
    active_inscr = total_inscr - total_quiet_inscr
    active_rate = active_inscr / (active_years / 100) if active_years > 0 else 0

    print(f"\n  Quiet periods: {total_quiet_inscr} inscriptions in {total_quiet_years} years = {quiet_rate:.1f}/century")
    print(f"  Active periods: {active_inscr} inscriptions in {active_years} years = {active_rate:.1f}/century")
    ratio = quiet_rate / active_rate if active_rate > 0 else float('inf')
    print(f"  Ratio (quiet/active): {ratio:.2f}x")

# --- Save results ---
results = {
    'meta': {
        'experiment': 'E099',
        'date': '2026-03-17',
        'n_eruptions': len(eruptions_dated),
        'n_inscriptions': len(inscriptions_dated),
        'period': '600-1500 CE',
    },
    'century_level': {
        'spearman_eruption_count': {'rho': float(rho_count), 'p': float(p_count)},
        'spearman_vei_sum': {'rho': float(rho_vei), 'p': float(p_vei)},
        'centuries': {f'C{c}': {'eruptions': int(century_eruptions[c]), 'vei_sum': float(century_vei_sum[c]),
                                 'inscriptions': int(century_inscriptions[c])} for c in centuries},
    },
    'bin_50yr': {
        'spearman': {'rho': float(rho_50), 'p': float(p_50)},
        'n_bins': len(bins),
    },
    'lag_analysis': lag_results,
    'quiet_vs_active': {
        'quiet_rate_per_century': float(quiet_rate),
        'active_rate_per_century': float(active_rate),
        'ratio': float(ratio),
    },
    'quiet_periods': gaps if gaps else [],
}

with open("experiments/E099_eruption_inscription_temporal/results/e099_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("E099 SUMMARY")
print("=" * 70)
print(f"  Century-level: rho={rho_count:.4f} (count), rho={rho_vei:.4f} (VEI)")
print(f"  50-year bins:  rho={rho_50:.4f}")
print(f"  Best lag:      {min(lag_results.items(), key=lambda x: x[1]['p'])[0]} decades")
if rho_count < -0.3 and p_count < 0.1:
    print("  VERDICT: ANTI-CORRELATION DETECTED (eruptions suppress inscriptions)")
elif rho_count > 0.3:
    print("  VERDICT: POSITIVE CORRELATION (eruptions coincide with inscription activity)")
else:
    print("  VERDICT: WEAK/NO CORRELATION at century level")
print(f"  Quiet period rate: {quiet_rate:.1f}/century vs active: {active_rate:.1f}/century")
print("=" * 70)
