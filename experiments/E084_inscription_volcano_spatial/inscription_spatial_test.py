#!/usr/bin/env python3
"""
E084: Formal Inscription-Volcano Spatial Analysis

Tests whether inscriptions systematically avoid volcanic proximity zones
compared to candi, reflecting different taphonomic processes.

Inscriptions = administrative/royal documents placed in agricultural zones.
Candi = sacred structures deliberately built at volcano flanks.

Input:
  - E082 geocoded inscriptions (182 total, filtered to Java/Bali, confidence != 'low')
  - E031 candi-volcano pairs (142 pairs with lat/lon and distance)

Tests:
  1. Mann-Whitney U: inscription vs candi distance to nearest volcano
  2. KS test: full distribution comparison
  3. Bootstrap 95% CI for mean difference
  4. Zone analysis: chi-square on Zone A/B/C distributions
  5. Temporal analysis: century vs distance, pre/post 929 CE split
  6. Grid-cell density analysis: inscription vs candi density correlations with volcanic distance
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ============================================================
# SETUP
# ============================================================
BASE = Path('experiments/E084_inscription_volcano_spatial')
OUT = BASE / 'results'
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("E084: Formal Inscription-Volcano Spatial Analysis")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n--- LOADING DATA ---")

# Load inscriptions from E082
insc_raw = pd.read_csv('experiments/E082_inscription_georeferencing/results/geocoded_inscriptions.csv')
print(f"E082 raw inscriptions: {len(insc_raw)}")

# Filter: Java/Bali only, confidence != 'low'
insc = insc_raw[
    (insc_raw['lat'] >= -9.0) & (insc_raw['lat'] <= -5.5) &
    (insc_raw['lon'] >= 105.0) & (insc_raw['lon'] <= 116.0) &
    (insc_raw['confidence'] != 'low')
].copy()
print(f"After filtering (Java/Bali, confidence != 'low'): {len(insc)}")

# Load candi from E031
candi = pd.read_csv('experiments/E031_candi_orientation/results/candi_volcano_pairs.csv')
print(f"E031 candi-volcano pairs: {len(candi)}")

# Extract distance arrays
insc_dist = insc['volcano_dist_km'].values
candi_dist = candi['distance_km'].values

print(f"\nInscription distances: n={len(insc_dist)}, "
      f"mean={np.mean(insc_dist):.1f} km, median={np.median(insc_dist):.1f} km, "
      f"std={np.std(insc_dist):.1f} km")
print(f"Candi distances:      n={len(candi_dist)}, "
      f"mean={np.mean(candi_dist):.1f} km, median={np.median(candi_dist):.1f} km, "
      f"std={np.std(candi_dist):.1f} km")
print(f"Mean difference: {np.mean(insc_dist) - np.mean(candi_dist):.1f} km "
      f"(inscriptions {'farther' if np.mean(insc_dist) > np.mean(candi_dist) else 'closer'})")

# ============================================================
# 2. DISTRIBUTION TESTS
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: Mann-Whitney U (inscription vs candi volcano distance)")
print("=" * 70)

mw_stat, mw_p = stats.mannwhitneyu(insc_dist, candi_dist, alternative='two-sided')
# Effect size: rank-biserial correlation
n1, n2 = len(insc_dist), len(candi_dist)
rank_biserial = 1 - (2 * mw_stat) / (n1 * n2)
print(f"U = {mw_stat:.0f}, p = {mw_p:.6f}")
print(f"Rank-biserial r = {rank_biserial:.3f}")
if mw_p < 0.001:
    print("=> HIGHLY SIGNIFICANT: inscription and candi distances differ")
elif mw_p < 0.05:
    print("=> SIGNIFICANT: inscription and candi distances differ")
else:
    print("=> NOT SIGNIFICANT at alpha=0.05")

print("\n" + "=" * 70)
print("TEST 2: Kolmogorov-Smirnov (full distribution comparison)")
print("=" * 70)

ks_stat, ks_p = stats.ks_2samp(insc_dist, candi_dist)
print(f"KS statistic = {ks_stat:.4f}, p = {ks_p:.6f}")
if ks_p < 0.05:
    print("=> SIGNIFICANT: distributions are different")
else:
    print("=> NOT SIGNIFICANT: distributions are similar")

print("\n" + "=" * 70)
print("TEST 3: Bootstrap 95% CI for mean difference")
print("=" * 70)

np.random.seed(42)
n_boot = 10000
boot_diffs = np.zeros(n_boot)
for i in range(n_boot):
    boot_insc = np.random.choice(insc_dist, size=len(insc_dist), replace=True)
    boot_candi = np.random.choice(candi_dist, size=len(candi_dist), replace=True)
    boot_diffs[i] = np.mean(boot_insc) - np.mean(boot_candi)

ci_low = np.percentile(boot_diffs, 2.5)
ci_high = np.percentile(boot_diffs, 97.5)
observed_diff = np.mean(insc_dist) - np.mean(candi_dist)
print(f"Observed mean difference: {observed_diff:.2f} km")
print(f"Bootstrap 95% CI: [{ci_low:.2f}, {ci_high:.2f}] km")
print(f"Bootstrap SE: {np.std(boot_diffs):.2f} km")
if ci_low > 0:
    print("=> CI excludes zero: inscriptions are SIGNIFICANTLY FARTHER from volcanoes")
elif ci_high < 0:
    print("=> CI excludes zero: inscriptions are SIGNIFICANTLY CLOSER to volcanoes")
else:
    print("=> CI includes zero: difference not significant")

# ============================================================
# 3. ZONE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Zone Analysis (Chi-square)")
print("=" * 70)

# Classify into zones
def classify_zone(d):
    if d <= 10:
        return 'A (0-10km)'
    elif d <= 30:
        return 'B (10-30km)'
    else:
        return 'C (>30km)'

insc['zone'] = insc['volcano_dist_km'].apply(classify_zone)
candi['zone'] = candi['distance_km'].apply(classify_zone)

# Count per zone
zones = ['A (0-10km)', 'B (10-30km)', 'C (>30km)']
insc_zone_counts = [len(insc[insc['zone'] == z]) for z in zones]
candi_zone_counts = [len(candi[candi['zone'] == z]) for z in zones]

print("\nZone distribution:")
print(f"{'Zone':<15} {'Inscriptions':<15} {'%':<8} {'Candi':<10} {'%':<8}")
print("-" * 56)
for z, ic, cc in zip(zones, insc_zone_counts, candi_zone_counts):
    ip = 100 * ic / len(insc) if len(insc) > 0 else 0
    cp = 100 * cc / len(candi) if len(candi) > 0 else 0
    print(f"{z:<15} {ic:<15} {ip:<8.1f} {cc:<10} {cp:<8.1f}")

# Chi-square: compare zone distributions between inscriptions and candi
# Create contingency table
contingency = np.array([insc_zone_counts, candi_zone_counts])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square test (contingency): chi2={chi2:.2f}, df={dof}, p={chi_p:.6f}")

# Expected vs observed ratios
print("\nExpected vs Observed (using E065 Zone A 17.9x as context):")
for i, z in enumerate(zones):
    if expected[1][i] > 0:
        candi_ratio = candi_zone_counts[i] / expected[1][i]
    else:
        candi_ratio = float('inf')
    if expected[0][i] > 0:
        insc_ratio = insc_zone_counts[i] / expected[0][i]
    else:
        insc_ratio = float('inf')
    print(f"  {z}: Inscription O/E = {insc_ratio:.2f}x, Candi O/E = {candi_ratio:.2f}x")

# Direct comparison: what fraction in Zone A?
insc_zone_a_frac = insc_zone_counts[0] / len(insc) if len(insc) > 0 else 0
candi_zone_a_frac = candi_zone_counts[0] / len(candi) if len(candi) > 0 else 0
print(f"\nZone A fraction: Inscriptions = {insc_zone_a_frac:.1%}, Candi = {candi_zone_a_frac:.1%}")
if candi_zone_a_frac > 0:
    print(f"Candi are {candi_zone_a_frac / insc_zone_a_frac:.1f}x more concentrated in Zone A "
          f"than inscriptions" if insc_zone_a_frac > 0 else "No inscriptions in Zone A")

# Fisher's exact test for Zone A vs not-A (2x2)
zone_a_table = np.array([
    [insc_zone_counts[0], sum(insc_zone_counts[1:])],
    [candi_zone_counts[0], sum(candi_zone_counts[1:])]
])
fisher_or, fisher_p = stats.fisher_exact(zone_a_table)
print(f"\nFisher's exact test (Zone A vs rest): OR={fisher_or:.3f}, p={fisher_p:.6f}")
if fisher_p < 0.05:
    if fisher_or < 1:
        print("=> Inscriptions significantly UNDERREPRESENTED in Zone A relative to candi")
    else:
        print("=> Inscriptions significantly OVERREPRESENTED in Zone A relative to candi")
else:
    print("=> No significant difference in Zone A representation")

# ============================================================
# 4. TEMPORAL ANALYSIS (inscriptions only)
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: Temporal Analysis (inscriptions)")
print("=" * 70)

# Inscriptions with century data
insc_dated = insc.dropna(subset=['century']).copy()
print(f"Dated inscriptions: {len(insc_dated)} / {len(insc)}")

# Century vs mean volcanic distance
century_stats = insc_dated.groupby('century').agg(
    n=('volcano_dist_km', 'count'),
    mean_dist=('volcano_dist_km', 'mean'),
    median_dist=('volcano_dist_km', 'median'),
    std_dist=('volcano_dist_km', 'std')
).reset_index()

print("\nCentury-level distance statistics:")
print(f"{'Century':<10} {'n':<5} {'Mean dist':<12} {'Median dist':<12} {'Std':<10}")
print("-" * 49)
for _, row in century_stats.iterrows():
    std_str = f"{row['std_dist']:.1f}" if pd.notna(row['std_dist']) else "N/A"
    print(f"{int(row['century']):<10} {int(row['n']):<5} {row['mean_dist']:<12.1f} "
          f"{row['median_dist']:<12.1f} {std_str:<10}")

# Spearman: century vs distance (individual level)
rho_ind, p_ind = stats.spearmanr(insc_dated['century'], insc_dated['volcano_dist_km'])
print(f"\nSpearman (individual): rho={rho_ind:.3f}, p={p_ind:.6f}")

# Spearman: century vs mean distance (aggregate level, requires >2 centuries)
if len(century_stats) >= 3:
    rho_agg, p_agg = stats.spearmanr(century_stats['century'], century_stats['mean_dist'])
    print(f"Spearman (century means): rho={rho_agg:.3f}, p={p_agg:.6f}")
else:
    rho_agg, p_agg = np.nan, np.nan
    print("Too few centuries for aggregate correlation")

# Before/after 929 CE split (Mataram -> Kadiri transition)
print("\n--- 929 CE Split (Mataram -> Kadiri) ---")
insc_with_date = insc.dropna(subset=['date_ce']).copy()
pre_929 = insc_with_date[insc_with_date['date_ce'] <= 929]
post_929 = insc_with_date[insc_with_date['date_ce'] > 929]
print(f"Pre-929 CE:  n={len(pre_929)}, mean dist={pre_929['volcano_dist_km'].mean():.1f} km, "
      f"median={pre_929['volcano_dist_km'].median():.1f} km")
print(f"Post-929 CE: n={len(post_929)}, mean dist={post_929['volcano_dist_km'].mean():.1f} km, "
      f"median={post_929['volcano_dist_km'].median():.1f} km")

if len(pre_929) >= 3 and len(post_929) >= 3:
    mw_929, p_929 = stats.mannwhitneyu(pre_929['volcano_dist_km'], post_929['volcano_dist_km'],
                                        alternative='two-sided')
    print(f"Mann-Whitney U: U={mw_929:.0f}, p={p_929:.6f}")
    diff_929 = post_929['volcano_dist_km'].mean() - pre_929['volcano_dist_km'].mean()
    print(f"Shift: {diff_929:+.1f} km ({'farther' if diff_929 > 0 else 'closer'} after 929 CE)")
else:
    mw_929, p_929 = np.nan, np.nan
    print("Insufficient data for pre/post 929 test")

# Zone shift across the 929 divide
print("\nZone distribution shift:")
for period, subset in [("Pre-929", pre_929), ("Post-929", post_929)]:
    n = len(subset)
    if n > 0:
        za = len(subset[subset['volcano_dist_km'] <= 10])
        zb = len(subset[(subset['volcano_dist_km'] > 10) & (subset['volcano_dist_km'] <= 30)])
        zc = len(subset[subset['volcano_dist_km'] > 30])
        print(f"  {period} (n={n}): A={za} ({100*za/n:.0f}%), B={zb} ({100*zb/n:.0f}%), "
              f"C={zc} ({100*zc/n:.0f}%)")

# ============================================================
# 5. DENSITY ANALYSIS (0.25-degree grid cells)
# ============================================================
print("\n" + "=" * 70)
print("TEST 6: Grid-Cell Density Analysis (0.25 degree)")
print("=" * 70)

# Define grid covering Java (lon 105-116, lat -9 to -5.5)
grid_size = 0.25
lon_bins = np.arange(105, 116 + grid_size, grid_size)
lat_bins = np.arange(-9.0, -5.5 + grid_size, grid_size)

# Volcano locations from candi data (unique volcanoes with approximate coords)
# Extract from nearest volcano data — get approximate volcano coordinates
# For each volcano, the candi closest to it gives an approximation
# Better: use the candi coords and their distances/azimuths to back-calculate volcano position
# Simpler: use the candi with min distance to each volcano as proxy for volcano location
volcano_coords = {}
for vname in candi['nearest_volcano'].unique():
    v_candi = candi[candi['nearest_volcano'] == vname]
    closest = v_candi.loc[v_candi['distance_km'].idxmin()]
    # Approximate: the volcano is roughly at the closest candi location
    # (error is at most closest.distance_km which is <1km for some)
    # For better accuracy, use azimuth to back-calculate
    dist_m = closest['distance_km'] * 1000
    az_rad = np.radians(closest['azimuth_to_volcano'])
    # Approximate back-calculation (flat earth ok for <50km)
    dlat = dist_m * np.cos(az_rad) / 111320
    dlon = dist_m * np.sin(az_rad) / (111320 * np.cos(np.radians(closest['lat'])))
    vlat = closest['lat'] + dlat
    vlon = closest['lon'] + dlon
    volcano_coords[vname] = (vlat, vlon)

print(f"Estimated volcano coordinates ({len(volcano_coords)} volcanoes):")
for vn, (vla, vlo) in sorted(volcano_coords.items()):
    print(f"  {vn}: ({vla:.3f}, {vlo:.3f})")

# Calculate mean volcanic distance for each grid cell
def mean_volcano_distance(cell_lat, cell_lon, v_coords):
    """Min distance from cell center to any volcano (in km)."""
    min_d = float('inf')
    for vlat, vlon in v_coords.values():
        dlat = (cell_lat - vlat) * 111.32
        dlon = (cell_lon - vlon) * 111.32 * np.cos(np.radians((cell_lat + vlat) / 2))
        d = np.sqrt(dlat**2 + dlon**2)
        if d < min_d:
            min_d = d
    return min_d

# Assign inscriptions and candi to grid cells
insc['grid_lon'] = np.digitize(insc['lon'].values, lon_bins) - 1
insc['grid_lat'] = np.digitize(insc['lat'].values, lat_bins) - 1
candi['grid_lon'] = np.digitize(candi['lon'].values, lon_bins) - 1
candi['grid_lat'] = np.digitize(candi['lat'].values, lat_bins) - 1

# Build grid
grid_data = []
for i in range(len(lat_bins) - 1):
    for j in range(len(lon_bins) - 1):
        cell_lat = (lat_bins[i] + lat_bins[i+1]) / 2
        cell_lon = (lon_bins[j] + lon_bins[j+1]) / 2
        n_insc = len(insc[(insc['grid_lat'] == i) & (insc['grid_lon'] == j)])
        n_candi = len(candi[(candi['grid_lat'] == i) & (candi['grid_lon'] == j)])
        v_dist = mean_volcano_distance(cell_lat, cell_lon, volcano_coords)
        grid_data.append({
            'lat': cell_lat, 'lon': cell_lon,
            'n_inscriptions': n_insc, 'n_candi': n_candi,
            'volcano_dist_km': v_dist
        })

grid_df = pd.DataFrame(grid_data)

# Filter to cells that have at least one inscription OR candi
occupied = grid_df[(grid_df['n_inscriptions'] > 0) | (grid_df['n_candi'] > 0)]
print(f"\nGrid cells: {len(grid_df)} total, {len(occupied)} occupied")

# Spearman: inscription density vs volcanic distance (all occupied cells)
insc_cells = grid_df[grid_df['n_inscriptions'] > 0]
candi_cells = grid_df[grid_df['n_candi'] > 0]

print(f"Cells with inscriptions: {len(insc_cells)}")
print(f"Cells with candi: {len(candi_cells)}")

if len(insc_cells) >= 3:
    rho_insc_grid, p_insc_grid = stats.spearmanr(
        insc_cells['n_inscriptions'], insc_cells['volcano_dist_km'])
    print(f"\nSpearman (inscription count vs volcano distance): rho={rho_insc_grid:.3f}, p={p_insc_grid:.4f}")
else:
    rho_insc_grid, p_insc_grid = np.nan, np.nan
    print("Too few inscription cells for correlation")

if len(candi_cells) >= 3:
    rho_candi_grid, p_candi_grid = stats.spearmanr(
        candi_cells['n_candi'], candi_cells['volcano_dist_km'])
    print(f"Spearman (candi count vs volcano distance):        rho={rho_candi_grid:.3f}, p={p_candi_grid:.4f}")
else:
    rho_candi_grid, p_candi_grid = np.nan, np.nan
    print("Too few candi cells for correlation")

# Compare the two correlations: Fisher z-transform
if not (np.isnan(rho_insc_grid) or np.isnan(rho_candi_grid)):
    z_insc = np.arctanh(rho_insc_grid)
    z_candi = np.arctanh(rho_candi_grid)
    se_diff = np.sqrt(1/(len(insc_cells)-3) + 1/(len(candi_cells)-3))
    z_diff = (z_insc - z_candi) / se_diff
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
    print(f"\nFisher z-test (comparing correlations): z={z_diff:.3f}, p={p_diff:.4f}")
    if p_diff < 0.05:
        print("=> Inscription-volcano and candi-volcano density correlations SIGNIFICANTLY DIFFER")
    else:
        print("=> No significant difference in density-distance correlations")
else:
    z_diff, p_diff = np.nan, np.nan

# Additional: for all occupied cells, compare inscription vs candi presence
# with respect to volcanic distance
print("\n--- Volcano distance by cell occupation type ---")
both = occupied[(occupied['n_inscriptions'] > 0) & (occupied['n_candi'] > 0)]
insc_only = occupied[(occupied['n_inscriptions'] > 0) & (occupied['n_candi'] == 0)]
candi_only = occupied[(occupied['n_inscriptions'] == 0) & (occupied['n_candi'] > 0)]
print(f"Both:       n={len(both)}, mean volcano dist = {both['volcano_dist_km'].mean():.1f} km")
print(f"Insc only:  n={len(insc_only)}, mean volcano dist = {insc_only['volcano_dist_km'].mean():.1f} km")
print(f"Candi only: n={len(candi_only)}, mean volcano dist = {candi_only['volcano_dist_km'].mean():.1f} km")

# ============================================================
# 6. ADDITIONAL: Percentile comparison
# ============================================================
print("\n" + "=" * 70)
print("ADDITIONAL: Percentile Comparison")
print("=" * 70)

for p in [10, 25, 50, 75, 90]:
    ip = np.percentile(insc_dist, p)
    cp = np.percentile(candi_dist, p)
    print(f"  P{p}: Inscriptions = {ip:.1f} km, Candi = {cp:.1f} km, diff = {ip-cp:+.1f} km")

# ============================================================
# 7. SYNTHESIS
# ============================================================
print("\n" + "=" * 70)
print("SYNTHESIS")
print("=" * 70)

findings = []

# Mann-Whitney
findings.append(f"Mann-Whitney U: inscriptions vs candi volcano distance "
                f"U={mw_stat:.0f}, p={mw_p:.6f}, rank-biserial r={rank_biserial:.3f}")
if mw_p < 0.05:
    findings.append(f"  -> SIGNIFICANT: inscriptions are {abs(observed_diff):.1f} km "
                    f"{'farther from' if observed_diff > 0 else 'closer to'} volcanoes than candi")

# KS
findings.append(f"KS test: D={ks_stat:.4f}, p={ks_p:.6f} "
                f"({'SIGNIFICANT' if ks_p < 0.05 else 'not significant'})")

# Bootstrap
findings.append(f"Bootstrap 95% CI for mean difference: [{ci_low:.1f}, {ci_high:.1f}] km "
                f"({'excludes zero' if ci_low > 0 or ci_high < 0 else 'includes zero'})")

# Zone
findings.append(f"Zone chi-square: chi2={chi2:.2f}, p={chi_p:.6f} "
                f"({'SIGNIFICANT' if chi_p < 0.05 else 'not significant'})")
findings.append(f"  Zone A: inscriptions {insc_zone_a_frac:.1%} vs candi {candi_zone_a_frac:.1%}")
findings.append(f"  Fisher exact (Zone A): OR={fisher_or:.3f}, p={fisher_p:.6f}")

# Temporal
if not np.isnan(rho_ind):
    findings.append(f"Temporal: individual Spearman rho={rho_ind:.3f}, p={p_ind:.4f}")
if not np.isnan(p_929):
    findings.append(f"  929 CE split: pre mean={pre_929['volcano_dist_km'].mean():.1f} km, "
                    f"post mean={post_929['volcano_dist_km'].mean():.1f} km, p={p_929:.4f}")

# Grid
if not np.isnan(rho_insc_grid):
    findings.append(f"Grid density: inscription rho={rho_insc_grid:.3f} (p={p_insc_grid:.4f}), "
                    f"candi rho={rho_candi_grid:.3f} (p={p_candi_grid:.4f})")

for i, f in enumerate(findings, 1):
    print(f"  {i}. {f}")

# Determine overall status
overall_sig = sum([
    mw_p < 0.05,
    ks_p < 0.05,
    ci_low > 0 or ci_high < 0,
    chi_p < 0.05,
    fisher_p < 0.05
])

print(f"\n  Significant tests: {overall_sig}/5 core tests")
if overall_sig >= 3:
    status = "SUCCESS"
    conclusion = ("Inscriptions have a SIGNIFICANTLY DIFFERENT spatial distribution relative to "
                  "volcanoes compared to candi. Inscriptions are systematically farther from "
                  "volcanic centers, consistent with their function as administrative documents "
                  "placed in agricultural/lowland zones rather than sacred volcanic sites.")
elif overall_sig >= 1:
    status = "INCONCLUSIVE"
    conclusion = ("Some evidence of different spatial distributions, but not all tests converge. "
                  "The difference may be real but modest.")
else:
    status = "FAILED"
    conclusion = ("No significant difference in spatial distribution relative to volcanoes "
                  "between inscriptions and candi.")

print(f"\n  STATUS: {status}")
print(f"  CONCLUSION: {conclusion}")

# ============================================================
# 8. VOLCARCH IMPLICATIONS
# ============================================================
print("\n" + "=" * 70)
print("VOLCARCH IMPLICATIONS")
print("=" * 70)

implications = []
if mw_p < 0.05 and observed_diff > 0:
    implications.append(
        "Inscriptions being farther from volcanoes supports volcanic taphonomic bias: "
        "administrative activity occurred at greater distances, but the ARCHAEOLOGICAL RECORD "
        "(dominated by candi) clusters at volcano flanks. This means the spatial distribution "
        "of surviving archaeological evidence overrepresents volcanic proximity zones."
    )
    implications.append(
        f"The {abs(observed_diff):.0f} km mean difference between inscription and candi placement "
        "suggests that textual (inscriptional) evidence samples a DIFFERENT geographic zone than "
        "architectural evidence. Cross-referencing both provides more complete spatial coverage."
    )
if chi_p < 0.05:
    implications.append(
        f"Zone A (0-10 km) contains {candi_zone_a_frac:.0%} of candi but only "
        f"{insc_zone_a_frac:.0%} of inscriptions, confirming that candi are "
        "disproportionately built in high-burial-risk volcanic zones."
    )
if not np.isnan(p_929) and p_929 < 0.05:
    implications.append(
        "The pre/post 929 CE shift in inscription distance suggests the eastward migration "
        "of political centers (Mataram -> Kadiri/Singasari) changed the spatial sampling of "
        "the epigraphic record."
    )

for i, imp in enumerate(implications, 1):
    print(f"  {i}. {imp}")

if not implications:
    print("  No strong implications for VOLCARCH from this analysis.")

# ============================================================
# 9. SAVE RESULTS
# ============================================================
print("\n--- SAVING RESULTS ---")

# Save text results
with open(OUT / 'e084_results.txt', 'w', encoding='utf-8') as f:
    f.write("E084: Formal Inscription-Volcano Spatial Analysis\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Date: 2026-03-13\n")
    f.write(f"Status: {status}\n\n")

    f.write("DATA SUMMARY\n")
    f.write(f"  Inscriptions (Java/Bali, not low confidence): n={len(insc)}\n")
    f.write(f"  Candi (E031): n={len(candi)}\n")
    f.write(f"  Inscription mean distance: {np.mean(insc_dist):.1f} +/- {np.std(insc_dist):.1f} km\n")
    f.write(f"  Candi mean distance: {np.mean(candi_dist):.1f} +/- {np.std(candi_dist):.1f} km\n")
    f.write(f"  Difference: {observed_diff:.1f} km\n\n")

    f.write("CORE TESTS\n")
    f.write(f"  1. Mann-Whitney U: U={mw_stat:.0f}, p={mw_p:.6f}, r_rb={rank_biserial:.3f}\n")
    f.write(f"  2. KS test: D={ks_stat:.4f}, p={ks_p:.6f}\n")
    f.write(f"  3. Bootstrap 95% CI: [{ci_low:.2f}, {ci_high:.2f}] km\n")
    f.write(f"  4. Zone chi-square: chi2={chi2:.2f}, df={dof}, p={chi_p:.6f}\n")
    f.write(f"  5. Fisher exact (Zone A): OR={fisher_or:.3f}, p={fisher_p:.6f}\n\n")

    f.write("ZONE DISTRIBUTION\n")
    f.write(f"  {'Zone':<15} {'Inscriptions':<15} {'%':<8} {'Candi':<10} {'%':<8}\n")
    for z, ic, cc in zip(zones, insc_zone_counts, candi_zone_counts):
        ip = 100 * ic / len(insc)
        cp = 100 * cc / len(candi)
        f.write(f"  {z:<15} {ic:<15} {ip:<8.1f} {cc:<10} {cp:<8.1f}\n")

    f.write(f"\nTEMPORAL ANALYSIS\n")
    f.write(f"  Individual Spearman: rho={rho_ind:.3f}, p={p_ind:.4f}\n")
    if not np.isnan(p_929):
        f.write(f"  929 CE split: pre mean={pre_929['volcano_dist_km'].mean():.1f} km, "
                f"post mean={post_929['volcano_dist_km'].mean():.1f} km, MW p={p_929:.4f}\n")

    f.write(f"\nGRID DENSITY ANALYSIS\n")
    if not np.isnan(rho_insc_grid):
        f.write(f"  Inscription density vs volcano dist: rho={rho_insc_grid:.3f}, p={p_insc_grid:.4f}\n")
    if not np.isnan(rho_candi_grid):
        f.write(f"  Candi density vs volcano dist: rho={rho_candi_grid:.3f}, p={p_candi_grid:.4f}\n")
    if not np.isnan(z_diff):
        f.write(f"  Fisher z-test (comparing correlations): z={z_diff:.3f}, p={p_diff:.4f}\n")

    f.write(f"\nCONCLUSION\n  {conclusion}\n")

    if implications:
        f.write(f"\nVOLCARCH IMPLICATIONS\n")
        for i, imp in enumerate(implications, 1):
            f.write(f"  {i}. {imp}\n")

print(f"Saved: {OUT / 'e084_results.txt'}")

# Save JSON summary
summary = {
    "experiment": "E084",
    "title": "Formal Inscription-Volcano Spatial Analysis",
    "date": "2026-03-13",
    "status": status,
    "data": {
        "n_inscriptions": int(len(insc)),
        "n_candi": int(len(candi)),
        "inscription_mean_dist_km": float(np.mean(insc_dist)),
        "inscription_median_dist_km": float(np.median(insc_dist)),
        "inscription_std_dist_km": float(np.std(insc_dist)),
        "candi_mean_dist_km": float(np.mean(candi_dist)),
        "candi_median_dist_km": float(np.median(candi_dist)),
        "candi_std_dist_km": float(np.std(candi_dist)),
        "mean_difference_km": float(observed_diff)
    },
    "tests": {
        "mann_whitney": {
            "U": float(mw_stat),
            "p": float(mw_p),
            "rank_biserial_r": float(rank_biserial),
            "significant": bool(mw_p < 0.05)
        },
        "ks_test": {
            "D": float(ks_stat),
            "p": float(ks_p),
            "significant": bool(ks_p < 0.05)
        },
        "bootstrap": {
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "se": float(np.std(boot_diffs)),
            "excludes_zero": bool(ci_low > 0 or ci_high < 0)
        },
        "zone_chi_square": {
            "chi2": float(chi2),
            "df": int(dof),
            "p": float(chi_p),
            "significant": bool(chi_p < 0.05)
        },
        "fisher_exact_zone_a": {
            "odds_ratio": float(fisher_or),
            "p": float(fisher_p),
            "significant": bool(fisher_p < 0.05)
        }
    },
    "zone_distribution": {
        "inscriptions": {z: int(c) for z, c in zip(zones, insc_zone_counts)},
        "candi": {z: int(c) for z, c in zip(zones, candi_zone_counts)},
        "inscription_zone_a_fraction": float(insc_zone_a_frac),
        "candi_zone_a_fraction": float(candi_zone_a_frac)
    },
    "temporal": {
        "spearman_individual": {
            "rho": float(rho_ind) if not np.isnan(rho_ind) else None,
            "p": float(p_ind) if not np.isnan(p_ind) else None
        },
        "split_929ce": {
            "pre_n": int(len(pre_929)),
            "pre_mean_dist": float(pre_929['volcano_dist_km'].mean()) if len(pre_929) > 0 else None,
            "post_n": int(len(post_929)),
            "post_mean_dist": float(post_929['volcano_dist_km'].mean()) if len(post_929) > 0 else None,
            "mw_p": float(p_929) if not np.isnan(p_929) else None
        }
    },
    "grid_density": {
        "grid_size_deg": float(grid_size),
        "n_occupied_cells": int(len(occupied)),
        "inscription_density_rho": float(rho_insc_grid) if not np.isnan(rho_insc_grid) else None,
        "inscription_density_p": float(p_insc_grid) if not np.isnan(p_insc_grid) else None,
        "candi_density_rho": float(rho_candi_grid) if not np.isnan(rho_candi_grid) else None,
        "candi_density_p": float(p_candi_grid) if not np.isnan(p_candi_grid) else None,
        "fisher_z_diff": float(z_diff) if not np.isnan(z_diff) else None,
        "fisher_z_p": float(p_diff) if not np.isnan(p_diff) else None
    },
    "conclusion": conclusion,
    "volcarch_implications": implications,
    "findings": findings,
    "significant_tests": int(overall_sig),
    "total_core_tests": 5,
    "papers_served": ["P11"],
    "channels": [1, 4, 9]
}

with open(OUT / 'e084_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Saved: {OUT / 'e084_summary.json'}")
print("\nDone!")
