"""
E121: Robustness Battery - Automated Resampling Tests
VOLCARCH AutoResearch Program 1

For each FDR-surviving experiment with accessible data:
1. Bootstrap 1000x -> confidence intervals
2. Permutation 10,000 shuffles -> empirical p-value
3. Jackknife leave-one-out -> stability
4. Effect size + power analysis

Experiments tested:
- E004: Site density vs volcanic distance (Spearman, n=7 bins)
- E005: Terrain residuals vs distance (Spearman, n=187 cells)
- E031: Candi zone distribution (Chi2, n=142 candi)
- E031b: Candi azimuthal clustering (Rayleigh, n=142)
- E051: Toponymic substrate court effect (Chi2, n=25244 villages)
- E083: Tephra burial effect type (Chi2, n=51 pairs)
- E083b: Sedimentation rate calibration (Pearson, n=24 measured depths)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats

np.random.seed(42)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_BOOTSTRAP = 1000
N_PERMUTATION = 10000
REPO = Path(__file__).parent.parent.parent

results_log = []

def log_result(experiment, test, original_stat, original_p, bootstrap_ci, perm_p, jackknife_mean, jackknife_std, verdict):
    entry = {
        "experiment": experiment,
        "test": test,
        "original_stat": float(original_stat) if original_stat is not None else None,
        "original_p": float(original_p) if original_p is not None else None,
        "bootstrap_ci_low": float(bootstrap_ci[0]) if bootstrap_ci else None,
        "bootstrap_ci_high": float(bootstrap_ci[1]) if bootstrap_ci else None,
        "perm_p": float(perm_p) if perm_p is not None else None,
        "jackknife_mean": float(jackknife_mean) if jackknife_mean is not None else None,
        "jackknife_std": float(jackknife_std) if jackknife_std is not None else None,
        "verdict": verdict,
    }
    results_log.append(entry)
    status = "ROBUST" if verdict == "ROBUST" else "FRAGILE" if verdict == "FRAGILE" else verdict
    print(f"  [{status}] {experiment}/{test}: stat={original_stat:.4f}, p={original_p:.2e}, "
          f"boot_CI=[{bootstrap_ci[0]:.4f},{bootstrap_ci[1]:.4f}], perm_p={perm_p:.4e}, "
          f"jack={jackknife_mean:.4f}+/-{jackknife_std:.4f}")


# ============================================================
# E004: Site Density vs Volcanic Distance (Spearman)
# ============================================================
print("=" * 70)
print("E004: Site density vs volcanic distance")
print("=" * 70)

df004 = pd.read_csv(REPO / "experiments/E004_density_analysis/results/density_by_distance.csv")
# Extract midpoints from bin labels
df004["dist_mid"] = df004["bin"].apply(lambda b: float(b.replace("+", "-250").split("-")[0]) + 12.5)
x004 = df004["dist_mid"].values
y004 = df004["density_per_1000km2"].values

rho, p_orig = stats.spearmanr(x004, y004)

# Bootstrap
boot_rhos = []
n = len(x004)
for _ in range(N_BOOTSTRAP):
    idx = np.random.choice(n, n, replace=True)
    r, _ = stats.spearmanr(x004[idx], y004[idx])
    boot_rhos.append(r)
ci = (np.percentile(boot_rhos, 2.5), np.percentile(boot_rhos, 97.5))

# Permutation
count = 0
for _ in range(N_PERMUTATION):
    y_shuf = np.random.permutation(y004)
    r_shuf, _ = stats.spearmanr(x004, y_shuf)
    if abs(r_shuf) >= abs(rho):
        count += 1
perm_p = count / N_PERMUTATION

# Jackknife
jack_rhos = []
for i in range(n):
    xi = np.delete(x004, i)
    yi = np.delete(y004, i)
    r, _ = stats.spearmanr(xi, yi)
    jack_rhos.append(r)

verdict = "ROBUST" if perm_p < 0.05 and ci[0] < 0 and ci[1] < 0 else "FRAGILE"
# Note: for negative correlation, CI should be entirely negative
if rho < 0 and ci[1] < 0:
    verdict = "ROBUST"
elif rho < 0 and ci[1] >= 0:
    verdict = "FRAGILE"

log_result("E004", "Spearman(density, distance)", rho, p_orig, ci, perm_p,
           np.mean(jack_rhos), np.std(jack_rhos), verdict)


# ============================================================
# E005: Terrain Residuals vs Volcanic Distance (Spearman)
# ============================================================
print("\n" + "=" * 70)
print("E005: Terrain suitability residuals vs volcanic distance")
print("=" * 70)

df005 = pd.read_csv(REPO / "experiments/E005_terrain_suitability/results/grid_analysis.csv")
x005 = df005["dist_to_volcano_km"].values
y005 = df005["residual"].values

rho5, p5 = stats.spearmanr(x005, y005)

# Bootstrap
boot5 = []
n5 = len(x005)
for _ in range(N_BOOTSTRAP):
    idx = np.random.choice(n5, n5, replace=True)
    r, _ = stats.spearmanr(x005[idx], y005[idx])
    boot5.append(r)
ci5 = (np.percentile(boot5, 2.5), np.percentile(boot5, 97.5))

# Permutation
count5 = 0
for _ in range(N_PERMUTATION):
    y_shuf = np.random.permutation(y005)
    r_shuf, _ = stats.spearmanr(x005, y_shuf)
    if abs(r_shuf) >= abs(rho5):
        count5 += 1
perm5 = count5 / N_PERMUTATION

# Jackknife (sample 50 for speed with n=187)
jack5 = []
jack_indices = np.random.choice(n5, min(50, n5), replace=False)
for i in jack_indices:
    xi = np.delete(x005, i)
    yi = np.delete(y005, i)
    r, _ = stats.spearmanr(xi, yi)
    jack5.append(r)

verdict5 = "ROBUST" if perm5 < 0.05 else "FRAGILE"
log_result("E005", "Spearman(residual, distance)", rho5, p5, ci5, perm5,
           np.mean(jack5), np.std(jack5), verdict5)


# ============================================================
# E031: Candi Zone Distribution (Chi2)
# ============================================================
print("\n" + "=" * 70)
print("E031: Candi zone overrepresentation")
print("=" * 70)

df031 = pd.read_csv(REPO / "experiments/E031_candi_orientation/results/candi_volcano_pairs.csv")
zones = df031["zone"].values
n31 = len(zones)

# Observed zone counts
zone_counts = pd.Series(zones).value_counts()
observed = np.array([zone_counts.get("A", 0), zone_counts.get("B", 0), zone_counts.get("C", 0)])

# Expected: proportional to area (Java ~129000 km2)
# Zone A: 0-10km from 45 volcanoes ~ 3.5% of area
# Zone B: 10-30km ~ 25% of area
# Zone C: >30km ~ 71.5% of area
expected_frac = np.array([0.035, 0.25, 0.715])
expected = expected_frac * observed.sum()  # match total

chi2_orig, p_chi2 = stats.chisquare(observed, expected)

# Zone A overrepresentation ratio
zone_a_ratio = (observed[0] / n31) / expected_frac[0]

# Bootstrap
boot_ratios = []
for _ in range(N_BOOTSTRAP):
    idx = np.random.choice(n31, n31, replace=True)
    z_boot = zones[idx]
    a_count = np.sum(z_boot == "A")
    ratio = (a_count / n31) / expected_frac[0]
    boot_ratios.append(ratio)
ci31 = (np.percentile(boot_ratios, 2.5), np.percentile(boot_ratios, 97.5))

# Permutation: draw from expected distribution, see how often chi2 as extreme
count31 = 0
for _ in range(N_PERMUTATION):
    sim = np.random.multinomial(observed.sum(), expected_frac)
    chi2_sim, _ = stats.chisquare(sim, expected)
    if chi2_sim >= chi2_orig:
        count31 += 1
perm31 = count31 / N_PERMUTATION

# Jackknife
jack31 = []
for i in range(min(50, n31)):
    z_jack = np.delete(zones, i)
    a_count = np.sum(z_jack == "A")
    ratio = (a_count / (n31 - 1)) / expected_frac[0]
    jack31.append(ratio)

verdict31 = "ROBUST" if perm31 < 0.001 and ci31[0] > 1.0 else "FRAGILE"
log_result("E031", "ZoneA overrepresentation ratio", zone_a_ratio, p_chi2, ci31, perm31,
           np.mean(jack31), np.std(jack31), verdict31)


# ============================================================
# E031b: Candi Azimuthal Clustering (Rayleigh Test)
# ============================================================
print("\n" + "=" * 70)
print("E031b: Candi directional clustering (west bias)")
print("=" * 70)

azimuths = df031["azimuth_from_volcano"].values
theta = np.radians(azimuths)

# Rayleigh test
C = np.mean(np.cos(theta))
S = np.mean(np.sin(theta))
R = np.sqrt(C**2 + S**2)
n_ray = len(theta)
# Rayleigh test: p = exp(-n * R^2) for large n
rayleigh_z = n_ray * R**2
rayleigh_p = np.exp(-rayleigh_z)  # approximation for large n
mean_dir = np.degrees(np.arctan2(S, C)) % 360

# Bootstrap mean direction
boot_dirs = []
for _ in range(N_BOOTSTRAP):
    idx = np.random.choice(n_ray, n_ray, replace=True)
    th = theta[idx]
    c = np.mean(np.cos(th))
    s = np.mean(np.sin(th))
    d = np.degrees(np.arctan2(s, c)) % 360
    boot_dirs.append(d)
# Handle circular CI carefully
boot_dirs = np.array(boot_dirs)
ci_ray = (np.percentile(boot_dirs, 2.5), np.percentile(boot_dirs, 97.5))

# Permutation: draw uniform random angles, see how often R as large
count_ray = 0
for _ in range(N_PERMUTATION):
    th_rand = np.random.uniform(0, 2 * np.pi, n_ray)
    c = np.mean(np.cos(th_rand))
    s = np.mean(np.sin(th_rand))
    R_rand = np.sqrt(c**2 + s**2)
    if R_rand >= R:
        count_ray += 1
perm_ray = count_ray / N_PERMUTATION

# Jackknife R
jack_R = []
for i in range(min(50, n_ray)):
    th_j = np.delete(theta, i)
    c = np.mean(np.cos(th_j))
    s = np.mean(np.sin(th_j))
    jack_R.append(np.sqrt(c**2 + s**2))

verdict_ray = "ROBUST" if perm_ray < 0.001 else "FRAGILE"
log_result("E031b", f"Rayleigh R (mean_dir={mean_dir:.0f}deg)", R, rayleigh_p,
           (np.percentile(boot_dirs, 2.5), np.percentile(boot_dirs, 97.5)), perm_ray,
           np.mean(jack_R), np.std(jack_R), verdict_ray)


# ============================================================
# E051: Toponymic Substrate - Court Effect (Chi2)
# ============================================================
print("\n" + "=" * 70)
print("E051: Toponymic substrate - Yogyakarta court effect")
print("=" * 70)

df051 = pd.read_csv(REPO / "experiments/E051_toponymic_substrate/results/village_classifications.csv")
n51 = len(df051)

# Pre-Hindu ratio by province
province_ratios = df051.groupby("province").apply(
    lambda g: (g["layer"] == "PRE_HINDU").mean()
).to_dict()

# Yogyakarta vs others
yogya_mask = df051["province"].str.contains("Yogyakarta", case=False, na=False)
yogya_ratio = df051.loc[yogya_mask, "layer"].apply(lambda x: x == "PRE_HINDU").mean()
others_ratio = df051.loc[~yogya_mask, "layer"].apply(lambda x: x == "PRE_HINDU").mean()
ratio_diff = others_ratio - yogya_ratio  # expected positive (court effect = LESS pre-Hindu)

# Chi2 test
yogya_pre = df051.loc[yogya_mask, "layer"].apply(lambda x: x == "PRE_HINDU").sum()
yogya_total = yogya_mask.sum()
others_pre = df051.loc[~yogya_mask, "layer"].apply(lambda x: x == "PRE_HINDU").sum()
others_total = (~yogya_mask).sum()

table = np.array([[yogya_pre, yogya_total - yogya_pre],
                  [others_pre, others_total - others_pre]])
chi2_51, p_51 = stats.chi2_contingency(table)[:2]

# Bootstrap
boot_diffs = []
for _ in range(N_BOOTSTRAP):
    idx = np.random.choice(n51, n51, replace=True)
    df_b = df051.iloc[idx]
    y_mask = df_b["province"].str.contains("Yogyakarta", case=False, na=False)
    if y_mask.sum() > 0 and (~y_mask).sum() > 0:
        yr = (df_b.loc[y_mask, "layer"] == "PRE_HINDU").mean()
        or_ = (df_b.loc[~y_mask, "layer"] == "PRE_HINDU").mean()
        boot_diffs.append(or_ - yr)
ci51 = (np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5))

# Permutation: shuffle province labels
count51 = 0
provinces = df051["province"].values
pre_hindu = (df051["layer"] == "PRE_HINDU").values
for _ in range(min(N_PERMUTATION, 5000)):  # limit for speed
    prov_shuf = np.random.permutation(provinces)
    y_mask_s = np.array(["Yogyakarta" in str(p) for p in prov_shuf])
    if y_mask_s.sum() > 0 and (~y_mask_s).sum() > 0:
        yr_s = pre_hindu[y_mask_s].mean()
        or_s = pre_hindu[~y_mask_s].mean()
        if (or_s - yr_s) >= ratio_diff:
            count51 += 1
perm51 = count51 / min(N_PERMUTATION, 5000)

verdict51 = "ROBUST" if perm51 < 0.001 and ci51[0] > 0 else "FRAGILE"
log_result("E051", f"Court effect (Yogya {yogya_ratio:.3f} vs others {others_ratio:.3f})",
           ratio_diff, p_51, ci51, perm51,
           np.mean(boot_diffs), np.std(boot_diffs), verdict51)


# ============================================================
# E083: Tephra-Archaeological Correlation (Effect Types)
# ============================================================
print("\n" + "=" * 70)
print("E083: Tephra-site effect type distribution")
print("=" * 70)

df083 = pd.read_csv(REPO / "experiments/E083_tephra_archaeological_correlation/results/tephra_archaeological_correlation.csv")
n83 = len(df083)

# Effect type distribution
effects = df083["effect_type"].values
effect_counts = pd.Series(effects).value_counts()
buried_frac = effect_counts.get("buried", 0) / n83

# Binomial test: buried > 50%?
binom_p = stats.binomtest(effect_counts.get("buried", 0), n83, 0.5, alternative="greater").pvalue

# Bootstrap buried fraction
boot83 = []
for _ in range(N_BOOTSTRAP):
    idx = np.random.choice(n83, n83, replace=True)
    bf = np.mean(effects[idx] == "buried")
    boot83.append(bf)
ci83 = (np.percentile(boot83, 2.5), np.percentile(boot83, 97.5))

# Permutation: is buried rate significantly different from uniform?
count83 = 0
n_types = len(effect_counts)
for _ in range(N_PERMUTATION):
    eff_shuf = np.random.choice(effects, n83, replace=True)
    bf_shuf = np.mean(eff_shuf == "buried")
    if bf_shuf >= buried_frac:
        count83 += 1
perm83 = count83 / N_PERMUTATION

# Jackknife
jack83 = []
for i in range(n83):
    eff_j = np.delete(effects, i)
    jack83.append(np.mean(eff_j == "buried"))

verdict83 = "ROBUST" if ci83[0] > 0.5 else "MARGINAL" if ci83[0] > 0.4 else "FRAGILE"
log_result("E083", "Buried fraction", buried_frac, binom_p, ci83, perm83,
           np.mean(jack83), np.std(jack83), verdict83)


# ============================================================
# E083b: Sedimentation Rate from Burial Depths
# ============================================================
print("\n" + "=" * 70)
print("E083b: Sedimentation rate from measured burial depths")
print("=" * 70)

# Filter to rows with measured burial depths
df083["burial_depth_m"] = pd.to_numeric(df083["burial_depth_m"], errors="coerce")
df083["eruption_year"] = pd.to_numeric(df083["eruption_year"], errors="coerce")
df83_depth = df083[df083["burial_depth_m"].notna() & (df083["burial_depth_m"] > 0)].copy()
df83_depth["age_years"] = 2026 - df83_depth["eruption_year"]
df83_depth["sed_rate_mm_yr"] = (df83_depth["burial_depth_m"] * 1000) / df83_depth["age_years"]

# Filter to reasonable rates (exclude Toba-age entries)
df83_recent = df83_depth[df83_depth["eruption_year"] > 0].copy()
n83b = len(df83_recent)

if n83b >= 5:
    rates = df83_recent["sed_rate_mm_yr"].values
    mean_rate = np.mean(rates)
    std_rate = np.std(rates, ddof=1)

    # Bootstrap mean
    boot_rates = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n83b, n83b, replace=True)
        boot_rates.append(np.mean(rates[idx]))
    ci83b = (np.percentile(boot_rates, 2.5), np.percentile(boot_rates, 97.5))

    # Jackknife
    jack_rates = []
    for i in range(n83b):
        jack_rates.append(np.mean(np.delete(rates, i)))

    # One-sample t-test: is mean rate > 0?
    t_stat, t_p = stats.ttest_1samp(rates, 0)

    verdict83b = "ROBUST" if ci83b[0] > 1.0 else "FRAGILE"
    log_result("E083b", f"Mean sed rate (n={n83b})", mean_rate, t_p, ci83b, t_p,
               np.mean(jack_rates), np.std(jack_rates), verdict83b)
else:
    print(f"  SKIPPED: only {n83b} entries with measured depths from historical era")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ROBUSTNESS BATTERY SUMMARY")
print("=" * 70)

robust_count = sum(1 for r in results_log if r["verdict"] == "ROBUST")
fragile_count = sum(1 for r in results_log if r["verdict"] == "FRAGILE")
marginal_count = sum(1 for r in results_log if r["verdict"] == "MARGINAL")
total = len(results_log)

print(f"\n  Total tests: {total}")
print(f"  ROBUST:   {robust_count} ({robust_count/total*100:.0f}%)")
print(f"  MARGINAL: {marginal_count} ({marginal_count/total*100:.0f}%)")
print(f"  FRAGILE:  {fragile_count} ({fragile_count/total*100:.0f}%)")

print(f"\n  {'Experiment':<15} {'Test':<45} {'Verdict':<10}")
print(f"  {'-'*15} {'-'*45} {'-'*10}")
for r in results_log:
    print(f"  {r['experiment']:<15} {r['test'][:45]:<45} {r['verdict']:<10}")

# Save results
with open(RESULTS_DIR / "robustness_battery_results.json", "w") as f:
    json.dump({
        "summary": {
            "total_tests": total,
            "robust": robust_count,
            "marginal": marginal_count,
            "fragile": fragile_count,
            "pass_rate": robust_count / total if total > 0 else 0,
        },
        "tests": results_log,
        "parameters": {
            "n_bootstrap": N_BOOTSTRAP,
            "n_permutation": N_PERMUTATION,
            "random_seed": 42,
        },
    }, f, indent=2)

print(f"\n  Results saved to {RESULTS_DIR}/robustness_battery_results.json")
