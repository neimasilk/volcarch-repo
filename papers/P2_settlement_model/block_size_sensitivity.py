"""
Block-size sensitivity checks for Paper 2 (E013 best hybrid setup).

Tests three spatial block scales under fixed pseudo-absence settings:
- ~40 km equivalent
- ~50 km baseline (BLOCK_SIZE_DEG=0.45)
- ~60 km equivalent

Run from repo root:
    py papers/P2_settlement_model/block_size_sensitivity.py
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
BLOCK_SIZE_GRID = [
    ("40km", 40.0 / 111.0),
    ("50km_baseline", 0.45),
    ("60km", 60.0 / 111.0),
]


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("e013_v7_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def bootstrap_mean_ci(values: np.ndarray, n_iter: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("bootstrap_mean_ci expects a non-empty 1D array")
    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def assign_spatial_blocks(x: np.ndarray, y: np.ndarray, block_size_deg: float) -> np.ndarray:
    block_size_m = block_size_deg * 111000.0
    bx = (x / block_size_m).astype(int)
    by = (y / block_size_m).astype(int)
    return bx * 10000 + by


def spatial_cv_folds_deterministic(blocks: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_blocks = np.unique(blocks)
    unique_blocks.sort()
    block_splits = np.array_split(unique_blocks, n_folds)
    folds = []
    for test_blocks in block_splits:
        test_mask = np.isin(blocks, test_blocks)
        train_mask = ~test_mask
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return folds


def run_spatial_cv_blocksize(
    df: pd.DataFrame,
    feat_cols: list[str],
    block_size_deg: float,
    n_folds: int,
    random_seed: int,
    e013_module,
) -> dict:
    X = df[feat_cols].values
    y = df["presence"].values
    blocks = assign_spatial_blocks(df["x"].values, df["y"].values, block_size_deg)
    folds = spatial_cv_folds_deterministic(blocks, n_folds)

    xgb_aucs, xgb_tsss, rf_aucs, rf_tsss = [], [], [], []
    for train_idx, test_idx in folds:
        if len(test_idx) == 0 or y[test_idx].sum() == 0:
            continue
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scale_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb_model = e013_module.xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pw,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
            random_state=random_seed,
        )
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_aucs.append(float(e013_module.roc_auc_score(y_test, xgb_prob)))
        xgb_tsss.append(float(e013_module.compute_tss(y_test, xgb_prob)))

        rf_model = e013_module.RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        rf_aucs.append(float(e013_module.roc_auc_score(y_test, rf_prob)))
        rf_tsss.append(float(e013_module.compute_tss(y_test, rf_prob)))

    return {
        "xgb_mean_auc": float(np.mean(xgb_aucs)),
        "xgb_std_auc": float(np.std(xgb_aucs)),
        "xgb_mean_tss": float(np.mean(xgb_tsss)),
        "xgb_std_tss": float(np.std(xgb_tsss)),
        "rf_mean_auc": float(np.mean(rf_aucs)),
        "rf_std_auc": float(np.std(rf_aucs)),
        "rf_mean_tss": float(np.mean(rf_tsss)),
        "rf_std_tss": float(np.std(rf_tsss)),
    }


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

    print("=" * 70)
    print("E013 block-size sensitivity (~40km / ~50km / ~60km)")
    print("=" * 70)
    print(
        f"Fixed hybrid config: region_blend={region_blend:.2f}, "
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

    print("Loading sites and feature matrix...")
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
    print(f"Running seed set size={len(seeds)} for each block scale...")

    detail_rows = []
    for block_label, block_deg in BLOCK_SIZE_GRID:
        block_km = block_deg * 111.0
        print(f"\nBlock scale: {block_label} ({block_deg:.4f} deg, ~{block_km:.1f} km)")
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

            metrics = run_spatial_cv_blocksize(
                df=train_df,
                feat_cols=e013.FEAT_COLS,
                block_size_deg=block_deg,
                n_folds=e013.N_FOLDS,
                random_seed=e013.RANDOM_SEED,
                e013_module=e013,
            )
            detail_rows.append(
                {
                    "block_label": block_label,
                    "block_size_deg": block_deg,
                    "block_size_km": block_km,
                    "seed": int(seed),
                    "xgb_mean_auc": metrics["xgb_mean_auc"],
                    "xgb_std_auc": metrics["xgb_std_auc"],
                    "xgb_mean_tss": metrics["xgb_mean_tss"],
                    "xgb_std_tss": metrics["xgb_std_tss"],
                    "rf_mean_auc": metrics["rf_mean_auc"],
                    "rf_std_auc": metrics["rf_std_auc"],
                    "rf_mean_tss": metrics["rf_mean_tss"],
                    "rf_std_tss": metrics["rf_std_tss"],
                    "xgb_mvr_pass": int(metrics["xgb_mean_auc"] >= 0.75),
                    "rf_mvr_pass": int(metrics["rf_mean_auc"] >= 0.75),
                }
            )
            print(
                f"  [{run_idx:02d}/{len(seeds)}] seed={seed}: "
                f"XGB={metrics['xgb_mean_auc']:.3f}, RF={metrics['rf_mean_auc']:.3f}"
            )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        raise RuntimeError("No block-size sensitivity runs completed.")

    detail_path = SUPP_DIR / "e013_blocksize_seed_metrics.csv"
    detail_df.to_csv(detail_path, index=False)

    summary_rows = []
    for i, (block_label, block_deg) in enumerate(BLOCK_SIZE_GRID):
        subset = detail_df[detail_df["block_label"] == block_label].copy()
        if subset.empty:
            continue
        xgb_auc_ci = bootstrap_mean_ci(subset["xgb_mean_auc"].values, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 100 + i)
        xgb_tss_ci = bootstrap_mean_ci(subset["xgb_mean_tss"].values, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 200 + i)
        rf_auc_ci = bootstrap_mean_ci(subset["rf_mean_auc"].values, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 300 + i)
        rf_tss_ci = bootstrap_mean_ci(subset["rf_mean_tss"].values, BOOTSTRAP_ITERS, BOOTSTRAP_SEED + 400 + i)
        summary_rows.append(
            {
                "block_label": block_label,
                "block_size_deg": block_deg,
                "block_size_km": block_deg * 111.0,
                "n_runs": int(len(subset)),
                "xgb_auc_mean": float(subset["xgb_mean_auc"].mean()),
                "xgb_auc_std": float(subset["xgb_mean_auc"].std(ddof=1)),
                "xgb_auc_ci_low": xgb_auc_ci[0],
                "xgb_auc_ci_high": xgb_auc_ci[1],
                "xgb_tss_mean": float(subset["xgb_mean_tss"].mean()),
                "xgb_tss_std": float(subset["xgb_mean_tss"].std(ddof=1)),
                "xgb_tss_ci_low": xgb_tss_ci[0],
                "xgb_tss_ci_high": xgb_tss_ci[1],
                "xgb_mvr_pass_rate": float(subset["xgb_mvr_pass"].mean()),
                "rf_auc_mean": float(subset["rf_mean_auc"].mean()),
                "rf_auc_std": float(subset["rf_mean_auc"].std(ddof=1)),
                "rf_auc_ci_low": rf_auc_ci[0],
                "rf_auc_ci_high": rf_auc_ci[1],
                "rf_tss_mean": float(subset["rf_mean_tss"].mean()),
                "rf_tss_std": float(subset["rf_mean_tss"].std(ddof=1)),
                "rf_tss_ci_low": rf_tss_ci[0],
                "rf_tss_ci_high": rf_tss_ci[1],
                "rf_mvr_pass_rate": float(subset["rf_mvr_pass"].mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("block_size_km").reset_index(drop=True)
    if summary_df.empty:
        raise RuntimeError("No summary rows produced for block-size sensitivity.")

    baseline_row = summary_df[summary_df["block_label"] == "50km_baseline"]
    if not baseline_row.empty:
        base_xgb_auc = float(baseline_row["xgb_auc_mean"].iloc[0])
        base_rf_auc = float(baseline_row["rf_auc_mean"].iloc[0])
        summary_df["delta_xgb_auc_vs_50km"] = summary_df["xgb_auc_mean"] - base_xgb_auc
        summary_df["delta_rf_auc_vs_50km"] = summary_df["rf_auc_mean"] - base_rf_auc

    summary_path = SUPP_DIR / "e013_blocksize_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    x = np.arange(len(summary_df))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("E013 Block-Size Sensitivity (20 seeds per scale)", fontsize=12)

    axes[0].errorbar(
        x - 0.05,
        summary_df["xgb_auc_mean"],
        yerr=summary_df["xgb_auc_std"],
        fmt="o-",
        color="#E53935",
        label="XGBoost",
    )
    axes[0].errorbar(
        x + 0.05,
        summary_df["rf_auc_mean"],
        yerr=summary_df["rf_auc_std"],
        fmt="o-",
        color="#1E88E5",
        label="RandomForest",
    )
    axes[0].axhline(0.75, color="green", linestyle="--", linewidth=1.2, label="MVR AUC (0.75)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_df["block_label"])
    axes[0].set_ylabel("Mean AUC across seeds")
    axes[0].set_ylim(0.70, 0.78)
    axes[0].set_title("AUC vs Block Scale")
    axes[0].legend(fontsize=8)

    axes[1].errorbar(
        x - 0.05,
        summary_df["xgb_tss_mean"],
        yerr=summary_df["xgb_tss_std"],
        fmt="o-",
        color="#43A047",
        label="XGBoost",
    )
    axes[1].errorbar(
        x + 0.05,
        summary_df["rf_tss_mean"],
        yerr=summary_df["rf_tss_std"],
        fmt="o-",
        color="#6D4C41",
        label="RandomForest",
    )
    axes[1].axhline(0.40, color="purple", linestyle="--", linewidth=1.2, label="TSS target (0.40)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary_df["block_label"])
    axes[1].set_ylabel("Mean TSS across seeds")
    axes[1].set_ylim(0.42, 0.52)
    axes[1].set_title("TSS vs Block Scale")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig_path = FIG_DIR / "fig7_e013_blocksize_sensitivity.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    lines = [
        "E013 block-size sensitivity (Paper 2 supplement)",
        "================================================",
        f"Date: {date.today().isoformat()}",
        "Input experiment: experiments/E013_settlement_model_v7",
        "",
        "Fixed hybrid parameters:",
        f"  region_blend={region_blend:.2f}",
        f"  hard_frac_target={hard_frac_target:.2f}",
        f"  base_best_seed={best_seed}",
        "",
        "Summary by block scale (20 alternate seeds each):",
    ]
    for _, row in summary_df.iterrows():
        lines.extend(
            [
                (
                    f"  {row['block_label']} (~{row['block_size_km']:.1f} km): "
                    f"XGB AUC={row['xgb_auc_mean']:.3f} "
                    f"[{row['xgb_auc_ci_low']:.3f},{row['xgb_auc_ci_high']:.3f}], "
                    f"RF AUC={row['rf_auc_mean']:.3f} "
                    f"[{row['rf_auc_ci_low']:.3f},{row['rf_auc_ci_high']:.3f}]"
                ),
                (
                    f"    pass-rate AUC>=0.75: XGB={row['xgb_mvr_pass_rate'] * 100:.1f}%, "
                    f"RF={row['rf_mvr_pass_rate'] * 100:.1f}%"
                ),
            ]
        )
    summary_txt = "\n".join(lines) + "\n"
    summary_txt_path = SUPP_DIR / "e013_blocksize_summary.txt"
    summary_txt_path.write_text(summary_txt, encoding="utf-8")

    print("\nSaved block-size sensitivity artifacts:")
    print(f"  {detail_path}")
    print(f"  {summary_path}")
    print(f"  {summary_txt_path}")
    print(f"  {fig_path}")
    print("\nHeadline summary:")
    for _, row in summary_df.iterrows():
        print(
            f"  {row['block_label']}: "
            f"XGB AUC={row['xgb_auc_mean']:.3f} ({row['xgb_auc_ci_low']:.3f}-{row['xgb_auc_ci_high']:.3f}), "
            f"RF AUC={row['rf_auc_mean']:.3f} ({row['rf_auc_ci_low']:.3f}-{row['rf_auc_ci_high']:.3f})"
        )


if __name__ == "__main__":
    main()
