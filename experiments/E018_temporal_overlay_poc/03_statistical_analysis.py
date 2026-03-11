"""
E018: Statistical Analysis — TOM vs TAP correlation.

Tests whether taphonomic pressure (TAP_index) correlates with temporal
gaps between independent settlement "clocks" (linguistic, genetic,
archaeological).

Tests:
  1. Spearman rho(TAP_index, max_gap)
  2. Kendall tau
  3. Permutation test (10K iterations)
  4. Sensitivity: date perturbation ±20%, alpha sweep, leave-one-out

Outputs:
  - results/correlation_results.txt
  - results/sensitivity_report.txt
  - results/fig_gap_vs_tap.png
  - results/fig_three_clocks.png
  - results/fig_sensitivity.png

Run from repo root:
    python experiments/E018_temporal_overlay_poc/03_statistical_analysis.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install matplotlib scipy")
    sys.exit(1)

# === Paths ===
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

# === Parameters ===
N_PERM = 10000
N_PERTURB = 10000
PERTURB_FRAC = 0.20  # ±20% date perturbation
ALPHA_SWEEP = np.arange(0.3, 0.91, 0.05)
RANDOM_SEED = 42


def permutation_test(x, y, n_perm, rng):
    """Two-sided permutation test for Spearman correlation."""
    observed_rho, _ = stats.spearmanr(x, y)
    count_extreme = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        rho_perm, _ = stats.spearmanr(x, y_perm)
        if abs(rho_perm) >= abs(observed_rho):
            count_extreme += 1
    p_perm = (count_extreme + 1) / (n_perm + 1)
    return observed_rho, p_perm


def normalize_01(x):
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min:
        return np.zeros_like(x, dtype=float)
    return (x - x_min) / (x_max - x_min)


def recompute_tap(tap_df, alpha):
    """Recompute TAP_index with a different alpha weight."""
    return alpha * tap_df["V_score"].values + (1 - alpha) * tap_df["C_score"].values


def main():
    print("=" * 60)
    print("E018 Step 3: Statistical Analysis — TOM vs TAP")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED)

    # --- Load data ---
    print("\n[1/5] Loading TOM table...")
    tom_path = RESULTS_DIR / "tom_table.csv"
    if not tom_path.exists():
        print("  ERROR: tom_table.csv not found. Run steps 01 and 02 first.")
        sys.exit(1)

    tom = pd.read_csv(tom_path)
    print(f"  Loaded {len(tom)} regions")

    required_cols = ["region", "L_age_bp", "G_age_bp", "A_age_bp",
                     "max_gap", "TAP_index", "V_score", "C_score"]
    missing_cols = [c for c in required_cols if c not in tom.columns]
    if missing_cols:
        print(f"  ERROR: Missing columns: {missing_cols}")
        print("  Run 02_compute_tap_index.py first.")
        sys.exit(1)

    # Drop rows with NaN in key columns
    valid = tom.dropna(subset=["max_gap", "TAP_index"])
    n = len(valid)
    print(f"  Valid rows for analysis: {n}")

    tap_idx = valid["TAP_index"].values
    max_gap = valid["max_gap"].values
    l_gap = valid["L_gap"].values
    g_gap = valid["G_gap"].values

    # --- Primary correlation tests ---
    print("\n[2/5] Computing correlations...")

    # Spearman
    rho_sp, p_sp = stats.spearmanr(tap_idx, max_gap)
    print(f"  Spearman rho(TAP, max_gap) = {rho_sp:.4f}  (p = {p_sp:.4f})")

    # Kendall
    tau_k, p_k = stats.kendalltau(tap_idx, max_gap)
    print(f"  Kendall  tau(TAP, max_gap) = {tau_k:.4f}  (p = {p_k:.4f})")

    # Permutation test
    print(f"\n  Running permutation test ({N_PERM} iterations)...")
    _, p_perm = permutation_test(tap_idx, max_gap, N_PERM, rng)
    print(f"  Permutation p-value: {p_perm:.4f}")

    # Separate gap correlations
    rho_l, p_l = stats.spearmanr(tap_idx, l_gap)
    rho_g, p_g = stats.spearmanr(tap_idx, g_gap)
    print(f"\n  Spearman rho(TAP, L_gap) = {rho_l:.4f}  (p = {p_l:.4f})")
    print(f"  Spearman rho(TAP, G_gap) = {rho_g:.4f}  (p = {p_g:.4f})")

    # --- Decision ---
    if rho_sp > 0.5:
        decision = "GO — Strong support for H-TOM"
    elif rho_sp > 0.3:
        decision = "CONDITIONAL GO — Moderate support, identify data gaps"
    elif rho_sp > 0:
        decision = "INCONCLUSIVE — Weak positive signal, need better data"
    else:
        decision = "KILL — Wrong direction, H-TOM not supported"

    # --- Write correlation results ---
    corr_report = f"""E018: Temporal Overlay Matrix — Correlation Results
{'=' * 55}
Date: 2026-03-05
Regions: {n}
Regions included: {', '.join(valid['region'].tolist())}

PRIMARY TEST:
  Spearman rho(TAP_index, max_gap) = {rho_sp:.4f}  (p = {p_sp:.4f})

SECONDARY TESTS:
  Kendall  tau(TAP_index, max_gap) = {tau_k:.4f}  (p = {p_k:.4f})
  Permutation test p-value (n={N_PERM}): {p_perm:.4f}

SEPARATE GAP CORRELATIONS:
  Spearman rho(TAP, L_gap) = {rho_l:.4f}  (p = {p_l:.4f})
  Spearman rho(TAP, G_gap) = {rho_g:.4f}  (p = {p_g:.4f})

DECISION: {decision}

Note: With n={n}, formal statistical significance (p<0.05) is unlikely.
We rely on effect size (rho magnitude), direction consistency, and
sensitivity robustness per the pre-registered criteria.

Data per region:
"""
    for _, row in valid.sort_values("TAP_index", ascending=False).iterrows():
        corr_report += (f"  {row['region']:16s}  TAP={row['TAP_index']:.3f}  "
                        f"max_gap={row['max_gap']:7.0f}  "
                        f"L_gap={row['L_gap']:7.0f}  G_gap={row['G_gap']:7.0f}\n")

    with open(RESULTS_DIR / "correlation_results.txt", "w") as f:
        f.write(corr_report)
    print(f"\n  Saved: {RESULTS_DIR / 'correlation_results.txt'}")

    # --- Sensitivity analyses ---
    print("\n[3/5] Running sensitivity analyses...")

    # 3a: Date perturbation (±20%)
    print(f"  Date perturbation ±{int(PERTURB_FRAC*100)}% ({N_PERTURB} iterations)...")
    perturb_rhos = []
    for _ in range(N_PERTURB):
        # Perturb each age independently
        l_pert = valid["L_age_bp"].values * (1 + rng.uniform(-PERTURB_FRAC, PERTURB_FRAC, n))
        g_pert = valid["G_age_bp"].values * (1 + rng.uniform(-PERTURB_FRAC, PERTURB_FRAC, n))
        a_pert = valid["A_age_bp"].values * (1 + rng.uniform(-PERTURB_FRAC, PERTURB_FRAC, n))
        l_gap_pert = l_pert - a_pert
        g_gap_pert = g_pert - a_pert
        max_gap_pert = np.maximum(l_gap_pert, g_gap_pert)
        rho_pert, _ = stats.spearmanr(tap_idx, max_gap_pert)
        perturb_rhos.append(rho_pert)

    perturb_rhos = np.array(perturb_rhos)
    perturb_pos_frac = (perturb_rhos > 0).mean()
    perturb_median = np.median(perturb_rhos)
    perturb_q05 = np.percentile(perturb_rhos, 5)
    perturb_q95 = np.percentile(perturb_rhos, 95)

    print(f"    Median rho: {perturb_median:.4f}")
    print(f"    90% CI: [{perturb_q05:.4f}, {perturb_q95:.4f}]")
    print(f"    Fraction positive: {perturb_pos_frac:.3f}")

    # 3b: Alpha weight sweep
    print(f"\n  Alpha sweep ({ALPHA_SWEEP[0]:.2f} to {ALPHA_SWEEP[-1]:.2f})...")
    alpha_rhos = []
    for alpha in ALPHA_SWEEP:
        tap_recomp = recompute_tap(valid, alpha)
        rho_a, _ = stats.spearmanr(tap_recomp, max_gap)
        alpha_rhos.append(rho_a)
        print(f"    alpha={alpha:.2f}  rho={rho_a:.4f}")
    alpha_rhos = np.array(alpha_rhos)

    # 3c: Leave-one-out
    print("\n  Leave-one-out analysis...")
    loo_rhos = []
    loo_labels = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rho_loo, _ = stats.spearmanr(tap_idx[mask], max_gap[mask])
        loo_rhos.append(rho_loo)
        loo_labels.append(valid.iloc[i]["region"])
        print(f"    Drop {valid.iloc[i]['region']:16s}  rho={rho_loo:.4f}")
    loo_rhos = np.array(loo_rhos)

    # Direction robustness: same sign as full-sample rho?
    direction_robust = (np.sign(loo_rhos) == np.sign(rho_sp)).all()

    # --- Write sensitivity report ---
    sens_report = f"""E018: Sensitivity Analysis Report
{'=' * 55}
Date: 2026-03-05
Full-sample Spearman rho: {rho_sp:.4f}

1. DATE PERTURBATION (±{int(PERTURB_FRAC*100)}%, n={N_PERTURB})
   Median rho: {perturb_median:.4f}
   90% CI: [{perturb_q05:.4f}, {perturb_q95:.4f}]
   Fraction positive: {perturb_pos_frac:.3f}
   Interpretation: {"Direction is ROBUST" if perturb_pos_frac > 0.8 else "Direction is NOT ROBUST" if perturb_pos_frac < 0.5 else "Direction is MARGINALLY ROBUST"}

2. ALPHA WEIGHT SWEEP (volcanic vs coastal weight)
"""
    for alpha, rho_a in zip(ALPHA_SWEEP, alpha_rhos):
        sens_report += f"   alpha={alpha:.2f}  rho={rho_a:.4f}\n"
    sens_report += f"""   Range: [{alpha_rhos.min():.4f}, {alpha_rhos.max():.4f}]
   Interpretation: {"Robust across alpha values" if alpha_rhos.min() > 0 and alpha_rhos.max() > 0 else "NOT robust — sign changes with alpha"}

3. LEAVE-ONE-OUT
"""
    for label, rho_l_val in zip(loo_labels, loo_rhos):
        sens_report += f"   Drop {label:16s}  rho={rho_l_val:.4f}\n"
    sens_report += f"""   Range: [{loo_rhos.min():.4f}, {loo_rhos.max():.4f}]
   Direction robust (all same sign): {direction_robust}
   Most influential region: {loo_labels[np.argmin(np.abs(loo_rhos))]} (lowest |rho| when dropped)

OVERALL ROBUSTNESS ASSESSMENT:
  Date perturbation: {"PASS" if perturb_pos_frac > 0.7 else "FAIL"}
  Alpha sweep: {"PASS" if (alpha_rhos > 0).all() else "FAIL"}
  Leave-one-out: {"PASS" if direction_robust else "FAIL"}
"""

    with open(RESULTS_DIR / "sensitivity_report.txt", "w") as f:
        f.write(sens_report)
    print(f"\n  Saved: {RESULTS_DIR / 'sensitivity_report.txt'}")

    # --- Generate figures ---
    print("\n[4/5] Generating figures...")

    # Figure 1: Gap vs TAP scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(tap_idx, max_gap, s=80, c="steelblue", edgecolors="black", zorder=5)
    for i, row in valid.iterrows():
        ax.annotate(row["region"], (row["TAP_index"], row["max_gap"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    # Fit line for visual reference
    if n > 2:
        z = np.polyfit(tap_idx, max_gap, 1)
        x_line = np.linspace(tap_idx.min() - 0.05, tap_idx.max() + 0.05, 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5)
    ax.set_xlabel("Taphonomic Pressure Index (TAP)", fontsize=12)
    ax.set_ylabel("Maximum Temporal Gap (years)", fontsize=12)
    ax.set_title(f"E018: TOM Gap vs Taphonomic Pressure\n"
                 f"Spearman ρ = {rho_sp:.3f}, n = {n}", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_gap_vs_tap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig_gap_vs_tap.png")

    # Figure 2: Three clocks bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n)
    width = 0.25
    regions_sorted = valid.sort_values("TAP_index", ascending=False)
    ax.bar(x - width, regions_sorted["L_age_bp"], width, label="Linguistic (L_age)",
           color="tab:blue", edgecolor="black", linewidth=0.5)
    ax.bar(x, regions_sorted["G_age_bp"], width, label="Genetic (G_age)",
           color="tab:orange", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, regions_sorted["A_age_bp"], width, label="Archaeological (A_age)",
           color="tab:green", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(regions_sorted["region"], rotation=45, ha="right")
    ax.set_ylabel("Age (years BP)", fontsize=12)
    ax.set_title("E018: Three Settlement Clocks by Region\n"
                 "(sorted by TAP_index, high → low)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    # Add TAP_index as secondary label
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"TAP={row['TAP_index']:.2f}" for _, row in regions_sorted.iterrows()],
                        rotation=45, ha="left", fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_three_clocks.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig_three_clocks.png")

    # Figure 3: Sensitivity — two subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 3a: Perturbation histogram
    axes[0].hist(perturb_rhos, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(rho_sp, color="red", linestyle="--", linewidth=2,
                    label=f"Observed ρ = {rho_sp:.3f}")
    axes[0].axvline(0, color="black", linestyle="-", linewidth=1)
    axes[0].set_xlabel("Spearman ρ")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Date Perturbation ±{int(PERTURB_FRAC*100)}%\n"
                      f"(n={N_PERTURB}, {perturb_pos_frac:.0%} positive)")
    axes[0].legend(fontsize=9)

    # 3b: Alpha sweep
    axes[1].plot(ALPHA_SWEEP, alpha_rhos, "o-", color="tab:orange", markersize=6)
    axes[1].axhline(0, color="black", linestyle="-", linewidth=1)
    axes[1].axhline(rho_sp, color="red", linestyle="--", alpha=0.5,
                    label=f"Default α=0.6")
    axes[1].set_xlabel("α (volcanic weight)")
    axes[1].set_ylabel("Spearman ρ")
    axes[1].set_title("Alpha Weight Sweep")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # 3c: Leave-one-out
    colors = ["red" if np.sign(r) != np.sign(rho_sp) else "steelblue" for r in loo_rhos]
    axes[2].barh(range(n), loo_rhos, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].set_yticks(range(n))
    axes[2].set_yticklabels(loo_labels, fontsize=9)
    axes[2].axvline(rho_sp, color="red", linestyle="--", linewidth=1.5,
                    label=f"Full ρ = {rho_sp:.3f}")
    axes[2].axvline(0, color="black", linestyle="-", linewidth=1)
    axes[2].set_xlabel("Spearman ρ (with region dropped)")
    axes[2].set_title("Leave-One-Out")
    axes[2].legend(fontsize=9)

    plt.suptitle("E018: Sensitivity Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig_sensitivity.png")

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("[5/5] SUMMARY")
    print("=" * 60)
    print(f"  Spearman rho:        {rho_sp:.4f}")
    print(f"  Kendall tau:         {tau_k:.4f}")
    print(f"  Permutation p:       {p_perm:.4f}")
    print(f"  Perturbation median: {perturb_median:.4f} ({perturb_pos_frac:.0%} positive)")
    print(f"  Alpha sweep range:   [{alpha_rhos.min():.4f}, {alpha_rhos.max():.4f}]")
    print(f"  LOO direction robust: {direction_robust}")
    print(f"\n  DECISION: {decision}")
    print("\nStep 3 COMPLETE.")


if __name__ == "__main__":
    main()
