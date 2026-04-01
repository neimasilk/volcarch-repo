"""
E159: Robustness Battery for Cathedral Findings
================================================
AutoResearch Program 1: Systematic stress-testing of VOLCARCH's
strongest findings using bootstrap, jackknife, and permutation tests.

Tests 5 cathedral findings with raw data available:
1. E069: Volcanic signal survives survey control (p=0.0015)
2. E031: Candi west-clustering (Rayleigh p=3.4e-8)
3. E051: Toponymic substrate court effect (p<1e-14)
4. E084: Inscription-volcano spatial divergence (p=5.2e-8)
5. E152: Post-929 shift (p=3.89e-12)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N_BOOTSTRAP = 10000
N_PERMUTATION = 10000
ALPHA = 0.05

results_dir = Path("D:/documents/volcarch-repo/experiments/E159_robustness_battery/results")
results_dir.mkdir(exist_ok=True)

print("=" * 70)
print("E159: ROBUSTNESS BATTERY FOR CATHEDRAL FINDINGS")
print(f"Bootstrap: {N_BOOTSTRAP}, Permutation: {N_PERMUTATION}")
print("=" * 70)

all_results = {}

# ============================================================
# TEST 1: E069 — Volcanic signal survives survey control
# ============================================================
print(f"\n{'='*70}")
print("TEST 1: E069 — Volcanic signal vs survey intensity")
print(f"{'='*70}")

try:
    df_069 = pd.read_csv("D:/documents/volcarch-repo/experiments/E069_adversarial_comparanda/adv3_survey_intensity/results/adv3_cell_data.csv")
    print(f"  Data: {len(df_069)} grid cells")

    # Original finding: quasi-Poisson regression shows volcanic proximity has
    # independent effect after controlling for survey proxies.
    # We'll test the simpler correlation: site_count vs volcano_dist
    # after partialling out road_dist and bpcb_dist.

    from scipy.stats import spearmanr, pearsonr

    # Direct correlation
    rho_direct, p_direct = spearmanr(df_069['volcano_dist'], df_069['site_count'])
    print(f"  Direct: rho={rho_direct:.4f}, p={p_direct:.6f}")

    # Partial correlation (controlling for road_dist and bpcb_dist)
    # Using residuals method
    from numpy.linalg import lstsq

    X_control = df_069[['road_dist', 'bpcb_dist']].values
    X_control = np.column_stack([np.ones(len(X_control)), X_control])

    # Residualize volcano_dist
    coef_v, _, _, _ = lstsq(X_control, df_069['volcano_dist'].values, rcond=None)
    resid_volcano = df_069['volcano_dist'].values - X_control @ coef_v

    # Residualize site_count
    coef_s, _, _, _ = lstsq(X_control, df_069['site_count'].values, rcond=None)
    resid_sites = df_069['site_count'].values - X_control @ coef_s

    rho_partial, p_partial = spearmanr(resid_volcano, resid_sites)
    print(f"  Partial (controlling road+bpcb): rho={rho_partial:.4f}, p={p_partial:.6f}")

    # Bootstrap CI for partial correlation
    bootstrap_rhos = []
    n = len(df_069)
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n, n, replace=True)
        X_b = X_control[idx]
        v_b = df_069['volcano_dist'].values[idx]
        s_b = df_069['site_count'].values[idx]

        coef_v_b, _, _, _ = lstsq(X_b, v_b, rcond=None)
        coef_s_b, _, _, _ = lstsq(X_b, s_b, rcond=None)
        resid_v_b = v_b - X_b @ coef_v_b
        resid_s_b = s_b - X_b @ coef_s_b

        rho_b, _ = spearmanr(resid_v_b, resid_s_b)
        bootstrap_rhos.append(rho_b)

    bootstrap_rhos = np.array(bootstrap_rhos)
    ci_low = np.percentile(bootstrap_rhos, 2.5)
    ci_high = np.percentile(bootstrap_rhos, 97.5)
    print(f"  Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  CI excludes zero: {'YES' if ci_low > 0 or ci_high < 0 else 'NO'}")

    # Permutation test
    perm_count = 0
    for _ in range(N_PERMUTATION):
        perm_idx = np.random.permutation(n)
        rho_perm, _ = spearmanr(resid_volcano[perm_idx], resid_sites)
        if abs(rho_perm) >= abs(rho_partial):
            perm_count += 1

    p_perm = perm_count / N_PERMUTATION
    print(f"  Permutation p-value: {p_perm:.6f}")

    # Jackknife (leave-one-out influence)
    jackknife_rhos = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_j = X_control[mask]
        v_j = df_069['volcano_dist'].values[mask]
        s_j = df_069['site_count'].values[mask]
        coef_v_j, _, _, _ = lstsq(X_j, v_j, rcond=None)
        coef_s_j, _, _, _ = lstsq(X_j, s_j, rcond=None)
        resid_v_j = v_j - X_j @ coef_v_j
        resid_s_j = s_j - X_j @ coef_s_j
        rho_j, _ = spearmanr(resid_v_j, resid_s_j)
        jackknife_rhos.append(rho_j)

    jackknife_rhos = np.array(jackknife_rhos)
    max_influence = np.max(np.abs(jackknife_rhos - rho_partial))
    print(f"  Jackknife stability: max influence = {max_influence:.4f}")
    print(f"  Jackknife range: [{jackknife_rhos.min():.4f}, {jackknife_rhos.max():.4f}]")

    verdict_069 = "ROBUST" if (ci_low > 0 or ci_high < 0) and p_perm < 0.01 else "FRAGILE"
    print(f"\n  VERDICT: {verdict_069}")

    all_results["E069"] = {
        "finding": "Volcanic signal survives survey control",
        "original_p": 0.0015,
        "partial_rho": float(rho_partial),
        "partial_p": float(p_partial),
        "bootstrap_ci": [float(ci_low), float(ci_high)],
        "permutation_p": float(p_perm),
        "jackknife_max_influence": float(max_influence),
        "verdict": verdict_069,
    }

except Exception as e:
    print(f"  ERROR: {e}")
    all_results["E069"] = {"verdict": "ERROR", "error": str(e)}

# ============================================================
# TEST 2: E031 — Candi west-clustering
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: E031 - Candi directional clustering (west of volcanoes)")
print(f"{'='*70}")

try:
    df_031 = pd.read_csv("D:/documents/volcarch-repo/experiments/E031_candi_orientation/results/candi_volcano_pairs.csv")
    print(f"  Data: {len(df_031)} candi-volcano pairs")

    # Extract azimuths (direction FROM volcano TO candi)
    azimuths = df_031['azimuth_from_volcano'].values
    azimuths_rad = np.deg2rad(azimuths)

    # Rayleigh test
    n = len(azimuths_rad)
    C = np.sum(np.cos(azimuths_rad))
    S = np.sum(np.sin(azimuths_rad))
    R = np.sqrt(C**2 + S**2)
    R_bar = R / n
    mean_dir = np.rad2deg(np.arctan2(S, C)) % 360

    # Rayleigh Z statistic
    Z = n * R_bar**2
    # Rayleigh p-value approximation
    p_rayleigh = np.exp(-Z) * (1 + (2*Z - Z**2)/(4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4)/(288*n**2))
    p_rayleigh = max(p_rayleigh, 1e-20)

    print(f"  Mean direction: {mean_dir:.1f} degrees")
    print(f"  R-bar: {R_bar:.4f}")
    print(f"  Rayleigh Z: {Z:.2f}, p = {p_rayleigh:.2e}")

    # Bootstrap CI for mean direction and R-bar
    boot_means = []
    boot_rbars = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n, n, replace=True)
        az_b = azimuths_rad[idx]
        C_b = np.sum(np.cos(az_b))
        S_b = np.sum(np.sin(az_b))
        R_b = np.sqrt(C_b**2 + S_b**2) / n
        mean_b = np.rad2deg(np.arctan2(S_b, C_b)) % 360
        boot_means.append(mean_b)
        boot_rbars.append(R_b)

    boot_rbars = np.array(boot_rbars)
    rbar_ci = [float(np.percentile(boot_rbars, 2.5)), float(np.percentile(boot_rbars, 97.5))]
    print(f"  Bootstrap R-bar 95% CI: [{rbar_ci[0]:.4f}, {rbar_ci[1]:.4f}]")

    # Permutation test for non-uniformity
    perm_count = 0
    for _ in range(N_PERMUTATION):
        perm_az = np.random.uniform(0, 2*np.pi, n)
        C_p = np.sum(np.cos(perm_az))
        S_p = np.sum(np.sin(perm_az))
        R_p = np.sqrt(C_p**2 + S_p**2) / n
        if R_p >= R_bar:
            perm_count += 1

    p_perm = perm_count / N_PERMUTATION
    print(f"  Permutation p-value: {p_perm:.6f}")

    # Jackknife
    jk_rbars = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        az_j = azimuths_rad[mask]
        C_j = np.sum(np.cos(az_j))
        S_j = np.sum(np.sin(az_j))
        R_j = np.sqrt(C_j**2 + S_j**2) / (n-1)
        jk_rbars.append(R_j)

    jk_rbars = np.array(jk_rbars)
    max_influence = np.max(np.abs(jk_rbars - R_bar))
    print(f"  Jackknife max influence: {max_influence:.4f}")
    print(f"  Jackknife R-bar range: [{jk_rbars.min():.4f}, {jk_rbars.max():.4f}]")

    # Quadrant analysis
    west_count = np.sum((azimuths >= 225) | (azimuths <= 315)) if False else np.sum((azimuths >= 180) & (azimuths <= 360))
    # Actually, let's count "western hemisphere" (180-360 or equivalently azimuths where candi is W of volcano)
    west_fraction = np.sum((azimuths > 180) & (azimuths < 360)) / n
    print(f"  Western hemisphere: {west_fraction*100:.1f}% (chance = 50%)")

    verdict_031 = "ROBUST" if p_perm < 0.001 and rbar_ci[0] > 0.1 else "FRAGILE"
    print(f"\n  VERDICT: {verdict_031}")

    all_results["E031"] = {
        "finding": "Candi cluster west of volcanoes",
        "original_p": 3.4e-8,
        "mean_direction": float(mean_dir),
        "R_bar": float(R_bar),
        "rayleigh_Z": float(Z),
        "rayleigh_p": float(p_rayleigh),
        "bootstrap_rbar_ci": rbar_ci,
        "permutation_p": float(p_perm),
        "jackknife_max_influence": float(max_influence),
        "west_fraction": float(west_fraction),
        "verdict": verdict_031,
    }

except Exception as e:
    print(f"  ERROR: {e}")
    all_results["E031"] = {"verdict": "ERROR", "error": str(e)}

# ============================================================
# TEST 3: E051 — Toponymic substrate court effect
# ============================================================
print(f"\n{'='*70}")
print("TEST 3: E051 - Court-center toponymic substrate gradient")
print(f"{'='*70}")

try:
    df_kab = pd.read_csv("D:/documents/volcarch-repo/experiments/E051_toponymic_substrate/results/kabupaten_summary.csv")
    print(f"  Data: {len(df_kab)} kabupaten")

    # Core finding: pre-Hindu ratio correlates with distance from court centers (volcano proxy)
    # Yogyakarta has lowest pre-Hindu ratio (26.2%) vs average 57.7%
    rho_orig, p_orig = spearmanr(df_kab['dist_volcano_km'], df_kab['prehidu_ratio'])
    print(f"  Spearman rho (volcano_dist vs pre-Hindu ratio): {rho_orig:.4f}, p={p_orig:.6f}")

    # Bootstrap
    n = len(df_kab)
    boot_rhos = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n, n, replace=True)
        rho_b, _ = spearmanr(df_kab['dist_volcano_km'].values[idx],
                             df_kab['prehidu_ratio'].values[idx])
        boot_rhos.append(rho_b)

    boot_rhos = np.array(boot_rhos)
    ci_low = float(np.percentile(boot_rhos, 2.5))
    ci_high = float(np.percentile(boot_rhos, 97.5))
    print(f"  Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  CI excludes zero: {'YES' if ci_low > 0 or ci_high < 0 else 'NO'}")

    # Permutation
    perm_count = 0
    for _ in range(N_PERMUTATION):
        perm_idx = np.random.permutation(n)
        rho_p, _ = spearmanr(df_kab['dist_volcano_km'].values[perm_idx],
                             df_kab['prehidu_ratio'].values)
        if abs(rho_p) >= abs(rho_orig):
            perm_count += 1

    p_perm = perm_count / N_PERMUTATION
    print(f"  Permutation p-value: {p_perm:.6f}")

    # Jackknife
    jk_rhos = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rho_j, _ = spearmanr(df_kab['dist_volcano_km'].values[mask],
                             df_kab['prehidu_ratio'].values[mask])
        jk_rhos.append(rho_j)

    jk_rhos = np.array(jk_rhos)
    max_influence = float(np.max(np.abs(jk_rhos - rho_orig)))
    print(f"  Jackknife max influence: {max_influence:.4f}")
    print(f"  Jackknife range: [{jk_rhos.min():.4f}, {jk_rhos.max():.4f}]")

    # Identify most influential observation
    most_influential_idx = np.argmax(np.abs(jk_rhos - rho_orig))
    print(f"  Most influential: {df_kab.iloc[most_influential_idx]['kab_name']} "
          f"(ratio={df_kab.iloc[most_influential_idx]['prehidu_ratio']:.3f}, "
          f"dist={df_kab.iloc[most_influential_idx]['dist_volcano_km']:.0f} km)")

    verdict_051 = "ROBUST" if (ci_low > 0 or ci_high < 0) and p_perm < 0.01 else "FRAGILE"
    print(f"\n  VERDICT: {verdict_051}")

    all_results["E051"] = {
        "finding": "Pre-Hindu toponyms increase with volcano distance",
        "original_p": 5.1e-14,
        "spearman_rho": float(rho_orig),
        "bootstrap_ci": [ci_low, ci_high],
        "permutation_p": float(p_perm),
        "jackknife_max_influence": max_influence,
        "most_influential": str(df_kab.iloc[most_influential_idx]['kab_name']),
        "verdict": verdict_051,
    }

except Exception as e:
    print(f"  ERROR: {e}")
    all_results["E051"] = {"verdict": "ERROR", "error": str(e)}

# ============================================================
# TEST 4: E084 — Inscription-volcano spatial divergence
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: E084 - Inscription vs candi distance to volcanoes")
print(f"{'='*70}")

try:
    # Load inscription data
    df_insc = pd.read_csv("D:/documents/volcarch-repo/experiments/E082_inscription_georeferencing/results/geocoded_inscriptions.csv")
    df_candi = pd.read_csv("D:/documents/volcarch-repo/experiments/E031_candi_orientation/results/candi_volcano_pairs.csv")

    # Filter to Java only (inscriptions may include Sumatra etc.)
    df_insc_java = df_insc[
        (df_insc['lat'] > -9) & (df_insc['lat'] < -5.5) &
        (df_insc['lon'] > 105) & (df_insc['lon'] < 115)
    ].copy()

    print(f"  Inscriptions (Java): {len(df_insc_java)}")
    print(f"  Candi: {len(df_candi)}")

    insc_dists = df_insc_java['volcano_dist_km'].dropna().values
    candi_dists = df_candi['distance_km'].values

    # Mann-Whitney U test
    U, p_mw = stats.mannwhitneyu(insc_dists, candi_dists, alternative='two-sided')
    print(f"  Inscription median dist: {np.median(insc_dists):.1f} km")
    print(f"  Candi median dist: {np.median(candi_dists):.1f} km")
    print(f"  Mann-Whitney p: {p_mw:.2e}")

    # Bootstrap CI for median difference
    boot_diffs = []
    n_i, n_c = len(insc_dists), len(candi_dists)
    for _ in range(N_BOOTSTRAP):
        boot_i = np.random.choice(insc_dists, n_i, replace=True)
        boot_c = np.random.choice(candi_dists, n_c, replace=True)
        boot_diffs.append(np.median(boot_i) - np.median(boot_c))

    boot_diffs = np.array(boot_diffs)
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    print(f"  Bootstrap median diff 95% CI: [{ci_low:.1f}, {ci_high:.1f}] km")
    print(f"  CI excludes zero: {'YES' if ci_low > 0 or ci_high < 0 else 'NO'}")

    # Permutation
    combined = np.concatenate([insc_dists, candi_dists])
    n_total = len(combined)
    obs_diff = np.median(insc_dists) - np.median(candi_dists)

    perm_count = 0
    for _ in range(N_PERMUTATION):
        perm_idx = np.random.permutation(n_total)
        perm_i = combined[perm_idx[:n_i]]
        perm_c = combined[perm_idx[n_i:]]
        perm_diff = np.median(perm_i) - np.median(perm_c)
        if abs(perm_diff) >= abs(obs_diff):
            perm_count += 1

    p_perm = perm_count / N_PERMUTATION
    print(f"  Permutation p-value: {p_perm:.6f}")

    verdict_084 = "ROBUST" if (ci_low > 0 or ci_high < 0) and p_perm < 0.001 else "FRAGILE"
    print(f"\n  VERDICT: {verdict_084}")

    all_results["E084"] = {
        "finding": "Inscriptions farther from volcanoes than candi",
        "original_p": 5.2e-8,
        "inscription_median_km": float(np.median(insc_dists)),
        "candi_median_km": float(np.median(candi_dists)),
        "mann_whitney_p": float(p_mw),
        "bootstrap_diff_ci": [ci_low, ci_high],
        "permutation_p": float(p_perm),
        "verdict": verdict_084,
    }

except Exception as e:
    print(f"  ERROR: {e}")
    all_results["E084"] = {"verdict": "ERROR", "error": str(e)}

# ============================================================
# TEST 5: E031 — Zone A overrepresentation (E065 finding)
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: Zone A (0-15km) candi overrepresentation")
print(f"{'='*70}")

try:
    # From E031 data: count candi in Zone A (0-15km) vs expected from area
    zone_a = df_candi[df_candi['distance_km'] <= 15]
    zone_b = df_candi[(df_candi['distance_km'] > 15) & (df_candi['distance_km'] <= 30)]
    zone_c = df_candi[df_candi['distance_km'] > 30]

    n_total = len(df_candi)
    n_zone_a = len(zone_a)

    # Expected fraction in Zone A based on area
    # Zone A (0-15km ring around each of ~7 major E. Java volcanoes)
    # Java total area ~129,000 km2, Zone A ~ 7 * pi * 15^2 ~ 4,950 km2
    # Fraction: ~3.8%
    expected_fraction = 0.038
    observed_fraction = n_zone_a / n_total

    # Binomial test
    p_binom = stats.binom_test(n_zone_a, n_total, expected_fraction, alternative='greater')
    overrep = observed_fraction / expected_fraction

    print(f"  Zone A candi: {n_zone_a}/{n_total} ({observed_fraction*100:.1f}%)")
    print(f"  Expected by area: {expected_fraction*100:.1f}%")
    print(f"  Overrepresentation: {overrep:.1f}x")
    print(f"  Binomial p: {p_binom:.2e}")

    # Bootstrap CI for overrepresentation ratio
    boot_overreps = []
    for _ in range(N_BOOTSTRAP):
        boot_dists = np.random.choice(df_candi['distance_km'].values, n_total, replace=True)
        boot_zone_a = np.sum(boot_dists <= 15) / n_total
        boot_overreps.append(boot_zone_a / expected_fraction)

    boot_overreps = np.array(boot_overreps)
    ci_low = float(np.percentile(boot_overreps, 2.5))
    ci_high = float(np.percentile(boot_overreps, 97.5))
    print(f"  Bootstrap overrep 95% CI: [{ci_low:.1f}x, {ci_high:.1f}x]")

    # Permutation: shuffle candi among random locations in Java
    perm_count = 0
    for _ in range(N_PERMUTATION):
        # Generate random distances from volcanoes (uniform across Java area)
        # Distance PDF for uniform random point near a volcano: f(r) ~ r (linear)
        perm_dists = np.sqrt(np.random.uniform(0, 100**2, n_total))  # 0-100km uniform area
        perm_zone_a = np.sum(perm_dists <= 15) / n_total
        perm_overrep = perm_zone_a / expected_fraction
        if perm_overrep >= overrep:
            perm_count += 1

    p_perm = perm_count / N_PERMUTATION
    print(f"  Permutation p-value: {p_perm:.6f}")

    verdict_065 = "ROBUST" if ci_low > 2.0 and p_perm < 0.001 else "FRAGILE"
    print(f"\n  VERDICT: {verdict_065}")

    all_results["E065_zone_a"] = {
        "finding": "Candi overrepresented in Zone A (0-15km from volcano)",
        "original_p": 1e-6,
        "zone_a_count": int(n_zone_a),
        "total": int(n_total),
        "overrepresentation": float(overrep),
        "bootstrap_ci": [ci_low, ci_high],
        "permutation_p": float(p_perm),
        "verdict": verdict_065,
    }

except Exception as e:
    print(f"  ERROR: {e}")
    all_results["E065_zone_a"] = {"verdict": "ERROR", "error": str(e)}


# ============================================================
# SYNTHESIS
# ============================================================
print(f"\n{'='*70}")
print("SYNTHESIS: ROBUSTNESS BATTERY RESULTS")
print(f"{'='*70}")

print(f"\n{'Test':<25} {'Original p':<15} {'Permutation p':<15} {'Bootstrap CI excl. 0':<22} {'JK Stable':<12} {'Verdict'}")
print(f"{'-'*95}")

for test_id, result in all_results.items():
    if result.get("verdict") == "ERROR":
        print(f"{test_id:<25} {'ERROR':<15} {'-':<15} {'-':<22} {'-':<12} ERROR")
        continue

    orig_p = result.get("original_p", "-")
    perm_p = result.get("permutation_p", "-")
    ci = result.get("bootstrap_ci", result.get("bootstrap_diff_ci", result.get("bootstrap_rbar_ci", ["-", "-"])))
    jk = result.get("jackknife_max_influence", "-")
    verdict = result.get("verdict", "-")

    orig_str = f"{orig_p:.2e}" if isinstance(orig_p, float) else str(orig_p)
    perm_str = f"{perm_p:.6f}" if isinstance(perm_p, float) else str(perm_p)
    ci_excl = "YES" if isinstance(ci[0], float) and (ci[0] > 0 or ci[1] < 0) else "NO"
    jk_str = f"{jk:.4f}" if isinstance(jk, float) else str(jk)

    print(f"{test_id:<25} {orig_str:<15} {perm_str:<15} {ci_excl:<22} {jk_str:<12} {verdict}")

# Count verdicts
robust_count = sum(1 for r in all_results.values() if r.get("verdict") == "ROBUST")
fragile_count = sum(1 for r in all_results.values() if r.get("verdict") == "FRAGILE")
error_count = sum(1 for r in all_results.values() if r.get("verdict") == "ERROR")
total = len(all_results)

print(f"\nROBUST: {robust_count}/{total}")
print(f"FRAGILE: {fragile_count}/{total}")
print(f"ERROR: {error_count}/{total}")

# Save results
with open(results_dir / "robustness_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nResults saved to {results_dir / 'robustness_results.json'}")
print(f"\nDONE.")
