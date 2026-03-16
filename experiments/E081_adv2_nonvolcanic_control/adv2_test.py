"""
E081 ADV-2: Non-Volcanic Control Test
======================================
Critical adversarial test for VOLCARCH.
If non-volcanic regions (Kalimantan, Madagascar) show the SAME cave/enclosed bias
as volcanic regions, then the volcanic taphonomic explanation collapses.

Pass criterion: Fisher exact p < 0.05 showing volcanic regions have statistically
different site-type distribution from non-volcanic controls.
"""

import csv
import json
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

# --- Configuration ---
INPUT_CSV = Path("D:/documents/volcarch-repo/experiments/E020_mini_nusarc/data/mini_nusarc_v3.csv")
RESULTS_DIR = Path("D:/documents/volcarch-repo/experiments/E081_adv2_nonvolcanic_control/results")
RESULTS_DIR.mkdir(exist_ok=True)

VOLCANIC_REGIONS = {"Java", "Sumatra", "Nusa_Tenggara", "Philippines", "Sulawesi", "Maluku"}
NONVOLCANIC_REGIONS = {"Kalimantan", "Madagascar"}

ENCLOSED_TYPES = {"cave", "rockshelter"}
OPEN_TYPES = {"open_air", "river_terrace"}

# Approximate land areas in km^2
LAND_AREA_KM2 = {
    "Java": 129000,
    "Sumatra": 473000,
    "Kalimantan": 540000,
    "Sulawesi": 174600,
    "Nusa_Tenggara": 73000,
    "Philippines": 300000,
    "Maluku": 75000,
    "Madagascar": 587000,
}

# --- Load data ---
print("=" * 70)
print("E081 ADV-2: Non-Volcanic Control Test")
print("=" * 70)

sites = []
with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sites.append(row)

print(f"\nTotal sites loaded: {len(sites)}")

# --- Classify regions ---
volcanic_sites = [s for s in sites if s['region'] in VOLCANIC_REGIONS]
nonvolcanic_sites = [s for s in sites if s['region'] in NONVOLCANIC_REGIONS]

print(f"Volcanic region sites: {len(volcanic_sites)}")
print(f"Non-volcanic control sites: {len(nonvolcanic_sites)}")

# Check for unclassified
unclassified = [s for s in sites if s['region'] not in VOLCANIC_REGIONS and s['region'] not in NONVOLCANIC_REGIONS]
if unclassified:
    print(f"WARNING: {len(unclassified)} sites not classified: {set(s['region'] for s in unclassified)}")

# --- Site type counts per group ---
def count_site_types(site_list, label):
    """Count site types and return summary dict."""
    type_counts = Counter(s['site_type'] for s in site_list)
    total = len(site_list)
    enclosed = sum(type_counts.get(t, 0) for t in ENCLOSED_TYPES)
    open_count = sum(type_counts.get(t, 0) for t in OPEN_TYPES)

    print(f"\n--- {label} (N={total}) ---")
    for st in ['cave', 'rockshelter', 'open_air', 'river_terrace']:
        n = type_counts.get(st, 0)
        pct = (n / total * 100) if total > 0 else 0
        print(f"  {st}: {n} ({pct:.1f}%)")

    enclosed_pct = (enclosed / total * 100) if total > 0 else 0
    open_pct = (open_count / total * 100) if total > 0 else 0
    print(f"  ENCLOSED (cave+rockshelter): {enclosed} ({enclosed_pct:.1f}%)")
    print(f"  OPEN (open_air+river_terrace): {open_count} ({open_pct:.1f}%)")

    return {
        'total': total,
        'type_counts': dict(type_counts),
        'enclosed': enclosed,
        'open': open_count,
        'enclosed_pct': round(enclosed_pct, 1),
        'open_pct': round(open_pct, 1),
    }

volcanic_summary = count_site_types(volcanic_sites, "VOLCANIC REGIONS")
nonvolcanic_summary = count_site_types(nonvolcanic_sites, "NON-VOLCANIC CONTROLS")

# --- Per-region breakdown ---
print("\n--- Per-Region Breakdown ---")
region_summaries = {}
for region in sorted(VOLCANIC_REGIONS | NONVOLCANIC_REGIONS):
    rsites = [s for s in sites if s['region'] == region]
    if not rsites:
        continue
    tc = Counter(s['site_type'] for s in rsites)
    total = len(rsites)
    enclosed = sum(tc.get(t, 0) for t in ENCLOSED_TYPES)
    enclosed_pct = (enclosed / total * 100) if total > 0 else 0
    area = LAND_AREA_KM2.get(region, None)
    density = (total / area * 10000) if area else None  # sites per 10,000 km^2

    vol_label = "VOLCANIC" if region in VOLCANIC_REGIONS else "CONTROL"
    density_str = f"{density:.2f}" if density else "N/A"
    print(f"  {region:20s} [{vol_label:8s}]: N={total:3d}, enclosed={enclosed_pct:5.1f}%, "
          f"density={density_str} sites/10k km2")

    region_summaries[region] = {
        'n_sites': total,
        'volcanic': region in VOLCANIC_REGIONS,
        'type_counts': dict(tc),
        'enclosed_n': enclosed,
        'enclosed_pct': round(enclosed_pct, 1),
        'land_area_km2': area,
        'density_per_10k_km2': round(density, 3) if density else None,
    }

# --- Temporal distribution ---
print("\n--- Temporal Distribution (date_bp) ---")
def temporal_stats(site_list, label):
    """Compute mean/median date_bp."""
    dates = []
    for s in site_list:
        try:
            d = float(s['date_bp'])
            dates.append(d)
        except (ValueError, KeyError):
            pass

    if not dates:
        print(f"  {label}: No valid dates")
        return {'mean_bp': None, 'median_bp': None, 'n': 0}

    dates_sorted = sorted(dates)
    mean_bp = sum(dates_sorted) / len(dates_sorted)
    n = len(dates_sorted)
    if n % 2 == 0:
        median_bp = (dates_sorted[n//2 - 1] + dates_sorted[n//2]) / 2
    else:
        median_bp = dates_sorted[n//2]

    print(f"  {label}: N={n}, mean={mean_bp:.0f} BP, median={median_bp:.0f} BP, "
          f"range={min(dates_sorted):.0f}-{max(dates_sorted):.0f} BP")
    return {'mean_bp': round(mean_bp, 0), 'median_bp': round(median_bp, 0), 'n': n,
            'min_bp': round(min(dates_sorted), 0), 'max_bp': round(max(dates_sorted), 0)}

volcanic_temporal = temporal_stats(volcanic_sites, "Volcanic")
nonvolcanic_temporal = temporal_stats(nonvolcanic_sites, "Non-volcanic")

# --- Statistical Tests ---
print("\n" + "=" * 70)
print("STATISTICAL TESTS")
print("=" * 70)

# We need scipy for Fisher exact and chi-square
try:
    from scipy.stats import fisher_exact, chi2_contingency
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available. Computing manual Fisher exact test.")

# Fisher Exact Test: enclosed vs open in volcanic vs non-volcanic
# Contingency table:
#                Enclosed    Open
# Volcanic       a           b
# Non-volcanic   c           d

a = volcanic_summary['enclosed']
b = volcanic_summary['open']
c = nonvolcanic_summary['enclosed']
d = nonvolcanic_summary['open']

print(f"\nFisher Exact Test: Enclosed vs Open")
print(f"  Contingency table:")
print(f"                   Enclosed  Open   Total")
print(f"  Volcanic:        {a:6d}  {b:5d}   {a+b:5d}")
print(f"  Non-volcanic:    {c:6d}  {d:5d}   {c+d:5d}")

if HAS_SCIPY:
    table = [[a, b], [c, d]]
    odds_ratio, fisher_p = fisher_exact(table, alternative='two-sided')
    print(f"\n  Fisher exact p-value (two-sided): {fisher_p:.6f}")
    print(f"  Odds ratio: {odds_ratio:.4f}")

    # Interpretation
    if fisher_p < 0.05:
        fisher_interpretation = "SIGNIFICANT (p < 0.05): Volcanic and non-volcanic regions show DIFFERENT site-type distributions."
    else:
        fisher_interpretation = "NOT SIGNIFICANT (p >= 0.05): Cannot reject null hypothesis that distributions are the same."
    print(f"  Interpretation: {fisher_interpretation}")
else:
    # Manual calculation for when scipy is not available
    from math import comb, log10
    n_total = a + b + c + d
    # Hypergeometric probability
    # P = C(a+b,a) * C(c+d,c) / C(n_total, a+c)
    # Sum over all tables at least as extreme
    fisher_p = None
    odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float('inf')
    fisher_interpretation = "scipy unavailable; manual p-value not computed"
    print(f"  Odds ratio (manual): {odds_ratio:.4f}")
    print(f"  WARNING: Install scipy for exact p-value computation")

# Chi-square test on full site_type distribution (4 categories)
print(f"\nChi-Square Test: Full site-type distribution (4 categories)")
all_types = ['cave', 'rockshelter', 'open_air', 'river_terrace']
vol_row = [volcanic_summary['type_counts'].get(t, 0) for t in all_types]
nonvol_row = [nonvolcanic_summary['type_counts'].get(t, 0) for t in all_types]

print(f"  {'Type':15s} Volcanic  Non-volcanic")
for i, t in enumerate(all_types):
    print(f"  {t:15s} {vol_row[i]:8d}  {nonvol_row[i]:12d}")

if HAS_SCIPY:
    chi2_table = [vol_row, nonvol_row]
    # Check for cells with expected count < 5
    total_all = sum(vol_row) + sum(nonvol_row)
    row_totals = [sum(vol_row), sum(nonvol_row)]
    col_totals = [vol_row[i] + nonvol_row[i] for i in range(len(all_types))]
    expected_min = min(
        (row_totals[r] * col_totals[c]) / total_all
        for r in range(2) for c in range(len(all_types))
    )
    print(f"\n  Minimum expected cell count: {expected_min:.2f}")
    if expected_min < 5:
        print(f"  WARNING: Expected count < 5 in at least one cell. Chi-square may be unreliable.")
        print(f"  Fisher exact test is the primary test for this analysis.")

    # Remove zero columns to avoid issues
    nonzero_cols = [i for i in range(len(all_types)) if (vol_row[i] + nonvol_row[i]) > 0]
    chi2_table_clean = [[vol_row[i] for i in nonzero_cols], [nonvol_row[i] for i in nonzero_cols]]

    if len(nonzero_cols) >= 2:
        chi2_stat, chi2_p, chi2_dof, chi2_expected = chi2_contingency(chi2_table_clean)
        print(f"  Chi-square statistic: {chi2_stat:.4f}")
        print(f"  Chi-square p-value: {chi2_p:.6f}")
        print(f"  Degrees of freedom: {chi2_dof}")

        if chi2_p < 0.05:
            chi2_interpretation = "SIGNIFICANT: Full site-type distributions differ between groups."
        else:
            chi2_interpretation = "NOT SIGNIFICANT: Cannot reject null that distributions are the same."
        print(f"  Interpretation: {chi2_interpretation}")
    else:
        chi2_stat = chi2_p = chi2_dof = None
        chi2_interpretation = "Insufficient non-zero columns for chi-square test."
        print(f"  {chi2_interpretation}")
else:
    chi2_stat = chi2_p = chi2_dof = None
    chi2_interpretation = "scipy unavailable"

# --- Effect size: difference in enclosed proportion ---
print(f"\n--- Effect Size ---")
vol_enclosed_pct = volcanic_summary['enclosed_pct']
nonvol_enclosed_pct = nonvolcanic_summary['enclosed_pct']
diff_pct = vol_enclosed_pct - nonvol_enclosed_pct
print(f"  Volcanic enclosed %: {vol_enclosed_pct:.1f}%")
print(f"  Non-volcanic enclosed %: {nonvol_enclosed_pct:.1f}%")
print(f"  Difference: {diff_pct:+.1f} percentage points")
print(f"  (Positive = volcanic MORE enclosed than non-volcanic)")

# --- Sample size power consideration ---
print(f"\n--- Sample Size Consideration ---")
print(f"  Volcanic N = {volcanic_summary['total']}")
print(f"  Non-volcanic N = {nonvolcanic_summary['total']}")
total_n = volcanic_summary['total'] + nonvolcanic_summary['total']
print(f"  Total N = {total_n}")
if nonvolcanic_summary['total'] < 20:
    print(f"  WARNING: Non-volcanic control group has only {nonvolcanic_summary['total']} sites.")
    print(f"  This limits statistical power. Results may be INCONCLUSIVE due to small N.")
    small_control = True
else:
    small_control = False

# --- Final Verdict ---
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

# Determine outcome
if fisher_p is not None and fisher_p < 0.05:
    if diff_pct > 0:
        verdict = "L1_SUPPORTED"
        verdict_text = (
            "VOLCARCH L1 SUPPORTED: Volcanic regions show significantly MORE enclosed sites "
            f"than non-volcanic controls (Fisher p={fisher_p:.4f}, diff={diff_pct:+.1f}pp). "
            "The volcanic taphonomic explanation is consistent with this pattern."
        )
    else:
        verdict = "L1_CHALLENGED"
        verdict_text = (
            "VOLCARCH L1 CHALLENGED: Volcanic regions show significantly FEWER enclosed sites "
            f"than non-volcanic controls (Fisher p={fisher_p:.4f}, diff={diff_pct:+.1f}pp). "
            "This is the OPPOSITE of what cave-bias-is-universal would predict, but also "
            "opposite of simple volcanic burial destroying open-air sites."
        )
elif fisher_p is not None and fisher_p >= 0.05:
    if small_control:
        verdict = "INCONCLUSIVE"
        verdict_text = (
            f"INCONCLUSIVE: No significant difference detected (Fisher p={fisher_p:.4f}), "
            f"but non-volcanic control group (N={nonvolcanic_summary['total']}) is small. "
            "Cannot confidently distinguish between 'same pattern everywhere' and 'different but underpowered'. "
            f"Enclosed rate: volcanic={vol_enclosed_pct:.1f}%, non-volcanic={nonvol_enclosed_pct:.1f}% "
            f"(diff={diff_pct:+.1f}pp)."
        )
    else:
        verdict = "L1_FAILED"
        verdict_text = (
            f"VOLCARCH L1 FAILED (ADV-2): Cave bias is UNIVERSAL — non-volcanic regions show "
            f"the same pattern as volcanic regions (Fisher p={fisher_p:.4f}). "
            f"Enclosed rate: volcanic={vol_enclosed_pct:.1f}%, non-volcanic={nonvol_enclosed_pct:.1f}%. "
            "The volcanic taphonomic explanation does NOT account for the cave dominance."
        )
else:
    verdict = "ERROR"
    verdict_text = "Could not compute Fisher exact test. Install scipy."

print(f"\n  VERDICT: {verdict}")
print(f"  {verdict_text}")

# --- Additional nuance: Java-specific analysis ---
# Java is uniquely interesting because it has river_terrace H. erectus sites
print(f"\n--- Java-Specific Analysis (volcanic region with most open-air sites) ---")
java_sites = [s for s in sites if s['region'] == 'Java']
java_summary = count_site_types(java_sites, "Java only")

print(f"\n  Java vs Kalimantan comparison:")
kal_sites = [s for s in sites if s['region'] == 'Kalimantan']
kal_summary = count_site_types(kal_sites, "Kalimantan only")

if HAS_SCIPY and java_summary['total'] > 0 and kal_summary['total'] > 0:
    jk_table = [
        [java_summary['enclosed'], java_summary['open']],
        [kal_summary['enclosed'], kal_summary['open']]
    ]
    jk_or, jk_p = fisher_exact(jk_table, alternative='two-sided')
    print(f"\n  Java vs Kalimantan Fisher exact p: {jk_p:.6f}, OR={jk_or:.4f}")
    java_kal_result = {'fisher_p': round(jk_p, 6), 'odds_ratio': round(jk_or, 4)}
else:
    java_kal_result = None

# --- Save results ---
print(f"\n\nSaving results...")

# Text output
results_txt = RESULTS_DIR / "adv2_results.txt"
# Redirect output to file
import io

# Capture everything we've printed by re-running the summary
lines = []
lines.append("=" * 70)
lines.append("E081 ADV-2: Non-Volcanic Control Test — Results")
lines.append("=" * 70)
lines.append(f"")
lines.append(f"Total sites: {len(sites)}")
lines.append(f"Volcanic region sites: {len(volcanic_sites)}")
lines.append(f"Non-volcanic control sites: {len(nonvolcanic_sites)}")
lines.append(f"")
lines.append(f"VOLCANIC REGIONS: {', '.join(sorted(VOLCANIC_REGIONS))}")
lines.append(f"NON-VOLCANIC CONTROLS: {', '.join(sorted(NONVOLCANIC_REGIONS))}")
lines.append(f"")
lines.append(f"--- Site Type Distribution ---")
lines.append(f"{'':20s} {'Volcanic':>10s} {'Non-volcanic':>12s}")
for st in ['cave', 'rockshelter', 'open_air', 'river_terrace']:
    v = volcanic_summary['type_counts'].get(st, 0)
    nv = nonvolcanic_summary['type_counts'].get(st, 0)
    lines.append(f"  {st:18s} {v:10d} {nv:12d}")
lines.append(f"")
lines.append(f"  Enclosed (cave+rock): {volcanic_summary['enclosed']:4d} ({volcanic_summary['enclosed_pct']:.1f}%)  "
             f"vs  {nonvolcanic_summary['enclosed']:4d} ({nonvolcanic_summary['enclosed_pct']:.1f}%)")
lines.append(f"  Open (air+terrace):   {volcanic_summary['open']:4d} ({volcanic_summary['open_pct']:.1f}%)  "
             f"vs  {nonvolcanic_summary['open']:4d} ({nonvolcanic_summary['open_pct']:.1f}%)")
lines.append(f"")
lines.append(f"--- Temporal Distribution ---")
lines.append(f"  Volcanic: mean={volcanic_temporal['mean_bp']} BP, median={volcanic_temporal['median_bp']} BP")
lines.append(f"  Non-volcanic: mean={nonvolcanic_temporal['mean_bp']} BP, median={nonvolcanic_temporal['median_bp']} BP")
lines.append(f"")
lines.append(f"--- Fisher Exact Test (enclosed vs open) ---")
lines.append(f"  p-value: {fisher_p:.6f}" if fisher_p else "  p-value: N/A")
lines.append(f"  odds ratio: {odds_ratio:.4f}" if odds_ratio else "  odds ratio: N/A")
lines.append(f"  interpretation: {fisher_interpretation}")
lines.append(f"")
if chi2_stat is not None:
    lines.append(f"--- Chi-Square Test (4-category) ---")
    lines.append(f"  chi2: {chi2_stat:.4f}, p={chi2_p:.6f}, dof={chi2_dof}")
    lines.append(f"  interpretation: {chi2_interpretation}")
    lines.append(f"")
lines.append(f"--- Per-Region Summary ---")
for region, rs in sorted(region_summaries.items()):
    vol_label = "VOLCANIC" if rs['volcanic'] else "CONTROL"
    lines.append(f"  {region:20s} [{vol_label:8s}] N={rs['n_sites']:3d}, "
                 f"enclosed={rs['enclosed_pct']:5.1f}%, "
                 f"density={rs['density_per_10k_km2']:.3f} sites/10k km2" if rs['density_per_10k_km2'] else
                 f"  {region:20s} [{vol_label:8s}] N={rs['n_sites']:3d}, "
                 f"enclosed={rs['enclosed_pct']:5.1f}%, density=N/A")
lines.append(f"")
lines.append(f"--- VERDICT ---")
lines.append(f"  {verdict}: {verdict_text}")

with open(results_txt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"  Saved: {results_txt}")

# JSON output
results_json = RESULTS_DIR / "adv2_summary.json"
summary = {
    'experiment': 'E081_adv2_nonvolcanic_control',
    'description': 'ADV-2: Non-Volcanic Control Test for VOLCARCH',
    'total_sites': len(sites),
    'volcanic_sites': len(volcanic_sites),
    'nonvolcanic_sites': len(nonvolcanic_sites),
    'volcanic_regions': sorted(VOLCANIC_REGIONS),
    'nonvolcanic_regions': sorted(NONVOLCANIC_REGIONS),
    'volcanic_summary': volcanic_summary,
    'nonvolcanic_summary': nonvolcanic_summary,
    'temporal': {
        'volcanic': volcanic_temporal,
        'nonvolcanic': nonvolcanic_temporal,
    },
    'fisher_exact': {
        'p_value': round(fisher_p, 6) if fisher_p else None,
        'odds_ratio': round(odds_ratio, 4) if odds_ratio and odds_ratio != float('inf') else None,
        'contingency_table': {'volcanic': [a, b], 'nonvolcanic': [c, d]},
        'interpretation': fisher_interpretation,
    },
    'chi_square': {
        'statistic': round(chi2_stat, 4) if chi2_stat else None,
        'p_value': round(chi2_p, 6) if chi2_p else None,
        'dof': chi2_dof,
        'interpretation': chi2_interpretation,
    },
    'effect_size': {
        'volcanic_enclosed_pct': vol_enclosed_pct,
        'nonvolcanic_enclosed_pct': nonvol_enclosed_pct,
        'difference_pp': round(diff_pct, 1),
    },
    'region_summaries': region_summaries,
    'java_vs_kalimantan': java_kal_result,
    'pass_criterion': 'Fisher exact p < 0.05',
    'verdict': verdict,
    'verdict_text': verdict_text,
    'small_control_warning': small_control,
}

with open(results_json, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  Saved: {results_json}")

print(f"\nDone.")
