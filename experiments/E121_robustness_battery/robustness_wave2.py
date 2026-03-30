"""
E121 Wave 2: ML Robustness + Inscription Spatial + Genre Taphonomy
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

np.random.seed(42)
RESULTS_DIR = Path(__file__).parent / "results"
REPO = Path(__file__).parent.parent.parent

results_log = []

def log_result(experiment, test, original_stat, original_p, bootstrap_ci, perm_p, jackknife_mean, jackknife_std, verdict):
    entry = {
        "experiment": experiment, "test": test,
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
    print(f"  [{verdict}] {experiment}/{test}: stat={original_stat:.4f}, "
          f"boot_CI=[{bootstrap_ci[0]:.4f},{bootstrap_ci[1]:.4f}], perm_p={perm_p:.4e}")


# ============================================================
# E027/E085: ML Substrate Detection Robustness
# ============================================================
print("=" * 70)
print("E027/E085: ML Substrate Detection - Cross-Validation Robustness")
print("=" * 70)

df027 = pd.read_csv(REPO / "experiments/E027_ml_substrate_detection/data/features_matrix.csv")

# Identify target and features
# Target column should indicate substrate vs cognate
target_cols = [c for c in df027.columns if c.lower() in ["label", "target", "class", "substrate", "is_substrate"]]
if not target_cols:
    # Try to find it by examining unique values
    for col in df027.columns:
        if df027[col].nunique() == 2:
            vals = set(df027[col].unique())
            if vals in [{"substrate", "cognate"}, {0, 1}, {"0", "1"}, {"austronesian", "substrate"}]:
                target_cols = [col]
                break

if target_cols:
    target_col = target_cols[0]
    y = df027[target_col].values
    # Convert to binary if needed
    if y.dtype == object:
        unique_vals = sorted(df027[target_col].unique())
        y = (y == unique_vals[1]).astype(int)

    feature_cols = [c for c in df027.columns if c != target_col and df027[c].dtype in [np.float64, np.int64, float, int]]
    X = df027[feature_cols].values

    print(f"  Features: {len(feature_cols)}, Samples: {len(y)}, Positive rate: {y.mean():.3f}")

    # Stratified 10-fold CV with RandomForest
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=skf, scoring="roc_auc")

    mean_auc = scores.mean()
    std_auc = scores.std()

    # Bootstrap across folds
    boot_aucs = []
    for _ in range(1000):
        idx = np.random.choice(10, 10, replace=True)
        boot_aucs.append(scores[idx].mean())
    ci_auc = (np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5))

    # Permutation: shuffle labels 100 times (each with 10-fold CV)
    perm_count = 0
    n_perm_ml = 100  # reduced for speed
    for i in range(n_perm_ml):
        y_shuf = np.random.permutation(y)
        scores_shuf = cross_val_score(rf, X, y_shuf, cv=skf, scoring="roc_auc")
        if scores_shuf.mean() >= mean_auc:
            perm_count += 1
        if (i + 1) % 20 == 0:
            print(f"    Permutation {i+1}/{n_perm_ml}...")
    perm_p_ml = perm_count / n_perm_ml

    # Feature ablation: remove top features one at a time
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_full.fit(X, y)
    importances = rf_full.feature_importances_
    top5_idx = np.argsort(importances)[-5:][::-1]

    print(f"\n  Top 5 features by importance:")
    ablation_results = []
    for fi in top5_idx:
        X_ablated = np.delete(X, fi, axis=1)
        scores_abl = cross_val_score(rf, X_ablated, y, cv=skf, scoring="roc_auc")
        drop = mean_auc - scores_abl.mean()
        print(f"    {feature_cols[fi]}: importance={importances[fi]:.4f}, AUC_drop={drop:.4f}")
        ablation_results.append({"feature": feature_cols[fi], "importance": importances[fi], "auc_drop": drop})

    verdict_ml = "ROBUST" if perm_p_ml < 0.05 and ci_auc[0] > 0.60 else "FRAGILE"
    log_result("E027/E085", f"RF 10-fold CV AUC (n={len(y)})", mean_auc, perm_p_ml,
               ci_auc, perm_p_ml, mean_auc, std_auc, verdict_ml)
else:
    print("  WARNING: Could not identify target column in features_matrix.csv")
    print(f"  Columns: {list(df027.columns)[:10]}...")


# ============================================================
# E084: Inscription-Candi Distance Contrast
# ============================================================
print("\n" + "=" * 70)
print("E084: Inscription vs Candi distance to nearest volcano")
print("=" * 70)

# Load E084 summary for statistics
e084_path = REPO / "experiments/E084_inscription_volcano_spatial/results/e084_summary.json"
if e084_path.exists():
    with open(e084_path) as f:
        e084 = json.load(f)
    print(f"  E084 summary loaded")

# Load raw candi data from E031
df_candi = pd.read_csv(REPO / "experiments/E031_candi_orientation/results/candi_volcano_pairs.csv")
candi_dist = df_candi["distance_km"].values

# Try to find inscription data
insc_paths = list((REPO / "experiments").glob("E082*/results/*.csv")) + \
             list((REPO / "experiments").glob("E084*/results/*.csv")) + \
             list((REPO / "experiments").glob("E084*/data/*.csv"))

insc_dist = None
for p in insc_paths:
    try:
        df_insc = pd.read_csv(p)
        dist_cols = [c for c in df_insc.columns if "dist" in c.lower() and "volcano" in c.lower()]
        if dist_cols:
            insc_dist = df_insc[dist_cols[0]].dropna().values
            print(f"  Loaded inscription distances from {p.name}: n={len(insc_dist)}")
            break
    except Exception:
        continue

if insc_dist is not None and len(insc_dist) > 10:
    # Mann-Whitney U test
    u_stat, mw_p = stats.mannwhitneyu(insc_dist, candi_dist, alternative="greater")
    mean_diff = np.mean(insc_dist) - np.mean(candi_dist)

    # Bootstrap mean difference
    boot_diffs = []
    n_insc, n_candi = len(insc_dist), len(candi_dist)
    for _ in range(1000):
        bi = np.random.choice(insc_dist, n_insc, replace=True)
        bc = np.random.choice(candi_dist, n_candi, replace=True)
        boot_diffs.append(np.mean(bi) - np.mean(bc))
    ci84 = (np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5))

    # Permutation
    combined = np.concatenate([insc_dist, candi_dist])
    count84 = 0
    for _ in range(10000):
        np.random.shuffle(combined)
        d_shuf = np.mean(combined[:n_insc]) - np.mean(combined[n_insc:])
        if d_shuf >= mean_diff:
            count84 += 1
    perm84 = count84 / 10000

    # Jackknife (sample 50 inscriptions)
    jack84 = []
    for i in range(min(50, n_insc)):
        di = np.delete(insc_dist, i)
        jack84.append(np.mean(di) - np.mean(candi_dist))

    verdict84 = "ROBUST" if perm84 < 0.001 and ci84[0] > 0 else "FRAGILE"
    log_result("E084", f"Insc-Candi distance diff ({mean_diff:.1f} km)", mean_diff, mw_p,
               ci84, perm84, np.mean(jack84), np.std(jack84), verdict84)
else:
    print("  Inscription distance data not found in accessible CSV. Using E084 summary stats.")
    # Synthetic test using reported values
    print(f"  Reported: inscriptions 25.7 km vs candi 16.5 km, MW p=5.2e-8")
    print(f"  Cannot run resampling without raw data. VERDICT: DEFERRED")


# ============================================================
# E070: Colonial Burial Depth Bootstrap
# ============================================================
print("\n" + "=" * 70)
print("E070: Colonial site register burial depths")
print("=" * 70)

e070_path = REPO / "experiments/E070_colonial_mining/results/colonial_site_register_v1.0.csv"
if e070_path.exists():
    df070 = pd.read_csv(e070_path)
    depth_cols = [c for c in df070.columns if "depth" in c.lower() or "burial" in c.lower()]
    if depth_cols:
        depths = pd.to_numeric(df070[depth_cols[0]], errors="coerce").dropna().values
        depths = depths[depths > 0]
        if len(depths) >= 5:
            mean_depth = np.mean(depths)
            # Bootstrap
            boot_depths = [np.mean(np.random.choice(depths, len(depths), replace=True)) for _ in range(1000)]
            ci070 = (np.percentile(boot_depths, 2.5), np.percentile(boot_depths, 97.5))
            # One-sample t-test: mean > 1m?
            t_stat, t_p = stats.ttest_1samp(depths, 1.0)
            verdict070 = "ROBUST" if ci070[0] > 1.0 else "MARGINAL" if ci070[0] > 0.5 else "FRAGILE"
            log_result("E070", f"Mean burial depth (n={len(depths)})", mean_depth, t_p,
                       ci070, t_p, np.mean(boot_depths), np.std(boot_depths), verdict070)
        else:
            print(f"  Only {len(depths)} depth values. SKIPPED.")
    else:
        print(f"  Columns: {list(df070.columns)}")
        print("  No depth column found. SKIPPED.")
else:
    print("  E070 CSV not found. SKIPPED.")


# ============================================================
# WAVE 2 SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("WAVE 2 SUMMARY")
print("=" * 70)

robust_count = sum(1 for r in results_log if r["verdict"] == "ROBUST")
fragile_count = sum(1 for r in results_log if r["verdict"] == "FRAGILE")
marginal_count = sum(1 for r in results_log if r["verdict"] == "MARGINAL")
total = len(results_log)

print(f"\n  Total tests: {total}")
if total > 0:
    print(f"  ROBUST:   {robust_count} ({robust_count/total*100:.0f}%)")
    print(f"  MARGINAL: {marginal_count} ({marginal_count/total*100:.0f}%)")
    print(f"  FRAGILE:  {fragile_count} ({fragile_count/total*100:.0f}%)")

    print(f"\n  {'Experiment':<15} {'Test':<50} {'Verdict':<10}")
    print(f"  {'-'*15} {'-'*50} {'-'*10}")
    for r in results_log:
        print(f"  {r['experiment']:<15} {r['test'][:50]:<50} {r['verdict']:<10}")

# Load wave 1 and merge
w1_path = RESULTS_DIR / "robustness_battery_results.json"
if w1_path.exists():
    with open(w1_path) as f:
        w1 = json.load(f)
    all_tests = w1["tests"] + results_log
else:
    all_tests = results_log

total_all = len(all_tests)
robust_all = sum(1 for r in all_tests if r["verdict"] == "ROBUST")
fragile_all = sum(1 for r in all_tests if r["verdict"] == "FRAGILE")
marginal_all = sum(1 for r in all_tests if r["verdict"] == "MARGINAL")

print(f"\n  COMBINED (Wave 1 + 2): {total_all} tests, {robust_all} ROBUST ({robust_all/total_all*100:.0f}%)")

# Save
with open(RESULTS_DIR / "robustness_wave2_results.json", "w") as f:
    json.dump({
        "wave2_summary": {"total": total, "robust": robust_count, "marginal": marginal_count, "fragile": fragile_count},
        "combined_summary": {"total": total_all, "robust": robust_all, "marginal": marginal_all, "fragile": fragile_all},
        "wave2_tests": results_log,
        "all_tests": all_tests,
    }, f, indent=2)

print(f"\n  Saved to {RESULTS_DIR}/robustness_wave2_results.json")
