"""
Paper 2 robustness checks for E013 best hybrid configuration.

Checks implemented:
1) Alternate-seed stability for pseudo-absence sampling.
2) Bootstrap 95% CI on seed-level mean AUC/TSS.

Run from repo root:
    py papers/P2_settlement_model/robustness_checks.py
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
PAPER_DIR = REPO_ROOT / "papers" / "P2_settlement_model"
FIG_DIR = PAPER_DIR / "figures"
SUPP_DIR = PAPER_DIR / "supplement"

E013_SCRIPT = REPO_ROOT / "experiments" / "E013_settlement_model_v7" / "01_settlement_model_v7.py"
E013_SWEEP_CSV = REPO_ROOT / "experiments" / "E013_settlement_model_v7" / "results" / "sweep_results.csv"

N_ALT_SEEDS = 20
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 20260224


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("e013_v7_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_mean_ci(values: np.ndarray, n_iter: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("bootstrap_mean_ci expects a non-empty 1D array")
    rng = np.random.default_rng(seed)
    n = arr.size
    samples = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        samples[i] = arr[idx].mean()
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def deterministic_seed_list(best_seed: int, n_total: int) -> list[int]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seeds = [best_seed]
    used = {best_seed}
    while len(seeds) < n_total:
        candidate = int(rng.integers(100, 50000))
        if candidate in used:
            continue
        used.add(candidate)
        seeds.append(candidate)
    return seeds


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_DIR.mkdir(parents=True, exist_ok=True)

    e013 = load_module(E013_SCRIPT)

    sweep_df = pd.read_csv(E013_SWEEP_CSV).sort_values("best_auc", ascending=False).reset_index(drop=True)
    if sweep_df.empty:
        raise RuntimeError(f"No rows in {E013_SWEEP_CSV}")

    best_cfg = sweep_df.iloc[0]
    best_seed = int(best_cfg["cfg_seed"])
    region_blend = float(best_cfg["region_blend"])
    hard_frac_target = float(best_cfg["hard_frac_target"])

    print("=" * 68)
    print("E013 robustness checks (alternate seeds + bootstrap CIs)")
    print("=" * 68)
    print(
        f"Best config fixed: region_blend={region_blend:.2f}, "
        f"hard_frac_target={hard_frac_target:.2f}, seed={best_seed}"
    )

    raster_files = {
        "elevation": e013.DEM_DIR / "jatim_dem.tif",
        "slope": e013.DEM_DIR / "jatim_slope.tif",
        "twi": e013.DEM_DIR / "jatim_twi.tif",
        "tri": e013.DEM_DIR / "jatim_tri.tif",
        "aspect": e013.DEM_DIR / "jatim_aspect.tif",
        "river_dist": e013.DEM_DIR / "jatim_river_dist.tif",
    }
    road_dist_path = e013.DEM_DIR / "jatim_road_dist_expanded.tif"

    for name, path in raster_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing raster for {name}: {path}")
    if not road_dist_path.exists():
        raise FileNotFoundError(f"Missing expanded road distance raster: {road_dist_path}")

    rasters = {}
    ref_bounds = None
    for name, path in raster_files.items():
        arr, transform, crs, bounds = e013.load_raster(path)
        rasters[name] = (arr, transform, crs, bounds)
        if ref_bounds is None:
            ref_bounds = bounds
    road_arr, road_transform, _, _ = e013.load_raster(road_dist_path)

    print("Loading sites and extracting features...")
    sites = e013.load_sites()
    sites_proj = sites.to_crs("EPSG:32749")
    rasters_simple = {k: (v[0], v[1]) for k, v in rasters.items()}
    site_feats = e013.extract_features_at_sites(sites, rasters_simple)
    site_feats = site_feats.dropna(subset=e013.FEAT_COLS)
    site_feats = site_feats[site_feats["elevation"] > 0].copy()
    n_pa_target = len(site_feats) * e013.PSEUDOABSENCE_RATIO
    print(f"  valid presences={len(site_feats)}, pseudo-absence target={n_pa_target}")

    if ref_bounds is None:
        raise RuntimeError("Raster bounds were not initialized.")
    bounds_utm = (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
    midx = (ref_bounds.left + ref_bounds.right) / 2.0
    midy = (ref_bounds.bottom + ref_bounds.top) / 2.0

    pres_mean = site_feats[e013.FEAT_COLS].mean().to_numpy(dtype=np.float64)
    pres_std = site_feats[e013.FEAT_COLS].std().replace(0, 1.0).to_numpy(dtype=np.float64)
    site_regions = e013.assign_regions(site_feats["x"].values, site_feats["y"].values, midx, midy)
    presence_region_prop = np.bincount(site_regions, minlength=4).astype(np.float64)
    presence_region_prop = presence_region_prop / presence_region_prop.sum()

    print("Building deterministic candidate pool...")
    pool_rng = np.random.default_rng(e013.RANDOM_SEED + 555)
    pool_df = e013.build_tgb_candidate_pool(
        sites_proj=sites_proj,
        bounds=bounds_utm,
        rasters=rasters_simple,
        road_arr=road_arr,
        road_transform=road_transform,
        n_target=n_pa_target * 16,
        decay_m=e013.BASE_DECAY_M,
        max_road_dist_m=e013.BASE_MAX_ROAD_DIST_M,
        min_prob=e013.MIN_ACCEPT_PROB,
        pres_mean=pres_mean,
        pres_std=pres_std,
        midx=midx,
        midy=midy,
        rng=pool_rng,
    )
    if len(pool_df) < n_pa_target:
        raise RuntimeError(f"Candidate pool too small ({len(pool_df)} < {n_pa_target}).")

    seeds = deterministic_seed_list(best_seed=best_seed, n_total=N_ALT_SEEDS)
    print(f"Evaluating alternate seeds: n={len(seeds)}")

    seed_rows = []
    fold_rows = []

    for run_idx, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        pa_df = e013.sample_hybrid_pseudo_absences(
            pool_df=pool_df,
            n_total=n_pa_target,
            hard_frac=hard_frac_target,
            region_blend=region_blend,
            presence_region_prop=presence_region_prop,
            rng=rng,
        )

        if len(pa_df) < int(0.9 * n_pa_target):
            print(f"  [{run_idx:02d}/{len(seeds)}] seed={seed}: skipped (shortfall {len(pa_df)})")
            continue

        site_df = site_feats[e013.FEAT_COLS + ["x", "y"]].copy()
        site_df["presence"] = 1
        pa_train = pa_df[e013.FEAT_COLS + ["x", "y"]].copy()
        pa_train["presence"] = 0
        train_df = pd.concat([site_df, pa_train], ignore_index=True).dropna(subset=e013.FEAT_COLS)

        metrics = e013.run_spatial_cv(train_df)
        seed_rows.append(
            {
                "seed": int(seed),
                "region_blend": region_blend,
                "hard_frac_target": hard_frac_target,
                "hard_frac_actual": float((pa_df["zdist"] >= e013.HARD_Z_MIN).mean()),
                "n_presences": int(site_df.shape[0]),
                "n_pseudoabsences": int(pa_train.shape[0]),
                "pa_road_mean_m": float(pa_df["road_dist_tgb"].mean()),
                "pa_zdist_mean": float(pa_df["zdist"].mean()),
                "xgb_mean_auc": float(metrics["xgb_mean_auc"]),
                "xgb_std_auc": float(metrics["xgb_std_auc"]),
                "xgb_mean_tss": float(metrics["xgb_mean_tss"]),
                "xgb_std_tss": float(metrics["xgb_std_tss"]),
                "rf_mean_auc": float(metrics["rf_mean_auc"]),
                "rf_std_auc": float(metrics["rf_std_auc"]),
                "rf_mean_tss": float(metrics["rf_mean_tss"]),
                "rf_std_tss": float(metrics["rf_std_tss"]),
                "xgb_mvr_pass": int(metrics["xgb_mean_auc"] >= 0.75),
                "rf_mvr_pass": int(metrics["rf_mean_auc"] >= 0.75),
            }
        )

        for fr in metrics["fold_rows"]:
            fold_rows.append(
                {
                    "seed": int(seed),
                    "fold": int(fr["fold"]),
                    "xgb_auc": float(fr["xgb_auc"]),
                    "xgb_tss": float(fr["xgb_tss"]),
                    "rf_auc": float(fr["rf_auc"]),
                    "rf_tss": float(fr["rf_tss"]),
                    "n_train": int(fr["n_train"]),
                    "n_test": int(fr["n_test"]),
                }
            )

        print(
            f"  [{run_idx:02d}/{len(seeds)}] seed={seed}: "
            f"XGB AUC={metrics['xgb_mean_auc']:.3f}, RF AUC={metrics['rf_mean_auc']:.3f}"
        )

    seed_df = pd.DataFrame(seed_rows).sort_values("xgb_mean_auc", ascending=False).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows).sort_values(["seed", "fold"]).reset_index(drop=True)

    if seed_df.empty:
        raise RuntimeError("No robustness runs completed.")

    seed_csv = SUPP_DIR / "e013_seed_stability.csv"
    fold_csv = SUPP_DIR / "e013_fold_metrics_by_seed.csv"
    seed_df.to_csv(seed_csv, index=False)
    fold_df.to_csv(fold_csv, index=False)

    xgb_auc_ci = bootstrap_mean_ci(seed_df["xgb_mean_auc"].values, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 1)
    xgb_tss_ci = bootstrap_mean_ci(seed_df["xgb_mean_tss"].values, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 2)
    rf_auc_ci = bootstrap_mean_ci(seed_df["rf_mean_auc"].values, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 3)
    rf_tss_ci = bootstrap_mean_ci(seed_df["rf_mean_tss"].values, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 4)

    summary = {
        "n_runs": int(len(seed_df)),
        "xgb_auc_mean": float(seed_df["xgb_mean_auc"].mean()),
        "xgb_auc_std": float(seed_df["xgb_mean_auc"].std(ddof=1)),
        "xgb_tss_mean": float(seed_df["xgb_mean_tss"].mean()),
        "xgb_tss_std": float(seed_df["xgb_mean_tss"].std(ddof=1)),
        "rf_auc_mean": float(seed_df["rf_mean_auc"].mean()),
        "rf_auc_std": float(seed_df["rf_mean_auc"].std(ddof=1)),
        "rf_tss_mean": float(seed_df["rf_mean_tss"].mean()),
        "rf_tss_std": float(seed_df["rf_mean_tss"].std(ddof=1)),
        "xgb_mvr_pass_rate": float(seed_df["xgb_mvr_pass"].mean()),
        "rf_mvr_pass_rate": float(seed_df["rf_mvr_pass"].mean()),
    }

    summary_txt = f"""E013 robustness checks (Paper 2 supplement)
=================================================
Date: {date.today().isoformat()}
Input experiment: experiments/E013_settlement_model_v7

Fixed hybrid parameters:
  region_blend={region_blend:.2f}
  hard_frac_target={hard_frac_target:.2f}
  base_best_seed={best_seed}

Alternate-seed runs:
  n_runs={summary['n_runs']}
  n_presences={int(seed_df['n_presences'].iloc[0])}
  n_pseudoabsences={int(seed_df['n_pseudoabsences'].iloc[0])}

XGBoost:
  mean AUC={summary['xgb_auc_mean']:.3f} +/- {summary['xgb_auc_std']:.3f}
  bootstrap 95% CI AUC=[{xgb_auc_ci[0]:.3f}, {xgb_auc_ci[1]:.3f}]
  mean TSS={summary['xgb_tss_mean']:.3f} +/- {summary['xgb_tss_std']:.3f}
  bootstrap 95% CI TSS=[{xgb_tss_ci[0]:.3f}, {xgb_tss_ci[1]:.3f}]
  pass-rate AUC>=0.75: {summary['xgb_mvr_pass_rate'] * 100:.1f}%

RandomForest:
  mean AUC={summary['rf_auc_mean']:.3f} +/- {summary['rf_auc_std']:.3f}
  bootstrap 95% CI AUC=[{rf_auc_ci[0]:.3f}, {rf_auc_ci[1]:.3f}]
  mean TSS={summary['rf_tss_mean']:.3f} +/- {summary['rf_tss_std']:.3f}
  bootstrap 95% CI TSS=[{rf_tss_ci[0]:.3f}, {rf_tss_ci[1]:.3f}]
  pass-rate AUC>=0.75: {summary['rf_mvr_pass_rate'] * 100:.1f}%

Observed ranges across alternate seeds:
  XGB AUC range=[{seed_df['xgb_mean_auc'].min():.3f}, {seed_df['xgb_mean_auc'].max():.3f}]
  RF AUC range=[{seed_df['rf_mean_auc'].min():.3f}, {seed_df['rf_mean_auc'].max():.3f}]
"""

    summary_path = SUPP_DIR / "e013_robustness_summary.txt"
    summary_path.write_text(summary_txt, encoding="utf-8")

    seed_plot = seed_df.sort_values("seed").reset_index(drop=True)
    idx = np.arange(1, len(seed_plot) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle("E013 Robustness: Alternate-Seed Stability", fontsize=12)

    axes[0].plot(idx, seed_plot["xgb_mean_auc"], marker="o", color="#E53935", label="XGBoost")
    axes[0].plot(idx, seed_plot["rf_mean_auc"], marker="o", color="#1E88E5", label="RandomForest")
    axes[0].axhline(0.75, color="green", linestyle="--", linewidth=1.2, label="MVR AUC (0.75)")
    axes[0].set_xlabel("Seed run index (sorted by seed)")
    axes[0].set_ylabel("Mean spatial AUC")
    axes[0].set_ylim(0.65, 0.82)
    axes[0].set_title("AUC Stability")
    axes[0].legend(fontsize=8)

    axes[1].plot(idx, seed_plot["xgb_mean_tss"], marker="o", color="#43A047", label="XGBoost")
    axes[1].plot(idx, seed_plot["rf_mean_tss"], marker="o", color="#6D4C41", label="RandomForest")
    axes[1].axhline(0.40, color="purple", linestyle="--", linewidth=1.2, label="TSS target (0.40)")
    axes[1].set_xlabel("Seed run index (sorted by seed)")
    axes[1].set_ylabel("Mean TSS")
    axes[1].set_ylim(0.30, 0.60)
    axes[1].set_title("TSS Stability")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig_path = FIG_DIR / "fig6_e013_seed_stability.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved robustness artifacts:")
    print(f"  {seed_csv}")
    print(f"  {fold_csv}")
    print(f"  {summary_path}")
    print(f"  {fig_path}")
    print("\nHeadline results:")
    print(
        f"  XGB mean AUC={summary['xgb_auc_mean']:.3f} "
        f"(95% CI {xgb_auc_ci[0]:.3f}-{xgb_auc_ci[1]:.3f}), "
        f"pass-rate={summary['xgb_mvr_pass_rate'] * 100:.1f}%"
    )
    print(
        f"  RF mean AUC={summary['rf_auc_mean']:.3f} "
        f"(95% CI {rf_auc_ci[0]:.3f}-{rf_auc_ci[1]:.3f}), "
        f"pass-rate={summary['rf_mvr_pass_rate'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
