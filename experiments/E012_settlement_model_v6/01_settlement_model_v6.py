"""
E012: Settlement Suitability Model v6 - TGB proxy enrichment + sweep.

Builds on E011 (AUC=0.725). Keeps sweep protocol fixed but uses expanded road
classes for survey-accessibility proxy.

Run from repo root:
    py experiments/E012_settlement_model_v6/00_prepare_road_proxy_expanded.py
    py experiments/E012_settlement_model_v6/01_settlement_model_v6.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import rowcol
    from shapely.geometry import Point
    from scipy.stats import spearmanr
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import folium
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"
SITES_PATH = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
RESULTS_DIR = Path(__file__).parent / "results"

VOLCANOES = {
    "Kelud": (-7.9300, 112.3080),
    "Semeru": (-8.1080, 112.9220),
    "Arjuno-Welirang": (-7.7290, 112.5750),
    "Bromo": (-7.9420, 112.9500),
    "Lamongan": (-7.9770, 113.3430),
    "Raung": (-8.1250, 114.0420),
    "Ijen": (-8.0580, 114.2420),
}

BLOCK_SIZE_DEG = 0.45
N_FOLDS = 5
PSEUDOABSENCE_RATIO = 5
RANDOM_SEED = 42
TGB_MIN_ACCEPT_PROB = 0.03

DECAY_GRID = [8000.0, 12000.0, 16000.0, 20000.0]
MAX_ROAD_DIST_GRID = [20000.0, 40000.0, 60000.0]

FEAT_COLS = ["elevation", "slope", "twi", "tri", "aspect", "river_dist"]


def load_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, src.transform, src.crs, src.bounds


def extract_at_points(points_xy: np.ndarray, raster_arr: np.ndarray, transform) -> np.ndarray:
    rows, cols = rowcol(transform, points_xy[:, 0], points_xy[:, 1])
    rows = np.array(rows)
    cols = np.array(cols)
    h, w = raster_arr.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    values = np.full(len(points_xy), np.nan, dtype=np.float32)
    values[valid] = raster_arr[rows[valid], cols[valid]]
    return values


def load_sites() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SITES_PATH)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs("EPSG:4326")
    jatim = (-9.0, 111.0, -6.5, 115.0)
    gdf = gdf[(gdf.geometry.y >= jatim[0]) & (gdf.geometry.x >= jatim[1]) &
              (gdf.geometry.y <= jatim[2]) & (gdf.geometry.x <= jatim[3])]
    print(f"  Sites loaded: {len(gdf)}")
    return gdf


def extract_features_at_sites(sites: gpd.GeoDataFrame, rasters: dict) -> pd.DataFrame:
    sites_proj = sites.to_crs("EPSG:32749")
    xy = np.column_stack([sites_proj.geometry.x, sites_proj.geometry.y])
    out = {}
    for name, (arr, transform, *_) in rasters.items():
        out[name] = extract_at_points(xy, arr, transform)
    df = pd.DataFrame(out)
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]
    return df


def build_feature_grid(rasters: dict, feat_cols: list) -> gpd.GeoDataFrame:
    step = 10
    ref_arr, ref_transform, *_ = list(rasters.values())[0]
    h, w = ref_arr.shape
    rows_idx = np.arange(0, h, step)
    cols_idx = np.arange(0, w, step)
    rr, cc = np.meshgrid(rows_idx, cols_idx, indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()
    xs, ys = rasterio.transform.xy(ref_transform, rr, cc)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    data = {"x": xs, "y": ys}
    for name, (arr, transform, *_) in rasters.items():
        data[name] = arr[rr, cc]

    df = pd.DataFrame(data)
    mask = df[feat_cols].notna().all(axis=1) & (df["elevation"] > 0)
    df = df[mask].reset_index(drop=True)
    geom = gpd.GeoSeries.from_xy(df["x"], df["y"], crs="EPSG:32749")
    return gpd.GeoDataFrame(df, geometry=geom)


def tgb_accept_prob(road_dist_m: float, decay_m: float, min_prob: float) -> float:
    p = float(np.exp(-road_dist_m / decay_m))
    return max(min_prob, min(1.0, p))


def generate_tgb_pseudo_absences(
    sites_proj: gpd.GeoDataFrame,
    bounds,
    n: int,
    rasters: dict,
    road_arr: np.ndarray,
    road_transform,
    decay_m: float,
    max_road_dist_m: float,
    min_prob: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    from shapely.ops import unary_union

    site_buffer = unary_union(sites_proj.buffer(2000))
    minx, miny, maxx, maxy = bounds
    candidates = []
    max_tries = n * 500
    tries = 0

    while len(candidates) < n and tries < max_tries:
        tries += 1
        px = rng.uniform(minx, maxx)
        py = rng.uniform(miny, maxy)
        pt = Point(px, py)
        if site_buffer.contains(pt):
            continue

        xy = np.array([[px, py]])
        road_val = extract_at_points(xy, road_arr, road_transform)[0]
        if not np.isfinite(road_val):
            continue
        if road_val > max_road_dist_m:
            continue

        p_accept = tgb_accept_prob(float(road_val), decay_m, min_prob)
        if rng.random() > p_accept:
            continue

        feats = {}
        valid = True
        for name, (arr, transform, *_) in rasters.items():
            val = extract_at_points(xy, arr, transform)[0]
            if not np.isfinite(val) or (name == "elevation" and val <= 0):
                valid = False
                break
            feats[name] = val

        if valid:
            feats["x"] = px
            feats["y"] = py
            feats["road_dist_tgb"] = float(road_val)
            feats["accept_prob_tgb"] = p_accept
            candidates.append(feats)

    if len(candidates) < n:
        print(f"  WARNING: Only generated {len(candidates)}/{n} pseudo-absences")
    return pd.DataFrame(candidates)


def assign_spatial_blocks(x: np.ndarray, y: np.ndarray, block_size_deg: float) -> np.ndarray:
    bx = (x / (block_size_deg * 111000)).astype(int)
    by = (y / (block_size_deg * 111000)).astype(int)
    return bx * 10000 + by


def spatial_cv_folds_deterministic(blocks: np.ndarray, n_folds: int) -> list:
    unique_blocks = np.unique(blocks)
    unique_blocks.sort()
    block_splits = np.array_split(unique_blocks, n_folds)
    folds = []
    for test_blocks in block_splits:
        test_mask = np.isin(blocks, test_blocks)
        train_mask = ~test_mask
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return folds


def compute_tss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def run_spatial_cv(df: pd.DataFrame) -> dict:
    X = df[FEAT_COLS].values
    y = df["presence"].values
    blocks = assign_spatial_blocks(df["x"].values, df["y"].values, BLOCK_SIZE_DEG)
    folds = spatial_cv_folds_deterministic(blocks, N_FOLDS)

    xgb_aucs, xgb_tsss, rf_aucs, rf_tsss = [], [], [], []
    fold_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
        if len(test_idx) == 0 or y[test_idx].sum() == 0:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scale_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pw,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
            random_state=RANDOM_SEED,
        )
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_prob)
        xgb_tss = compute_tss(y_test, xgb_prob)
        xgb_aucs.append(xgb_auc)
        xgb_tsss.append(xgb_tss)

        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_prob)
        rf_tss = compute_tss(y_test, rf_prob)
        rf_aucs.append(rf_auc)
        rf_tsss.append(rf_tss)

        fold_rows.append(
            {
                "fold": fold_i,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "xgb_auc": float(xgb_auc),
                "xgb_tss": float(xgb_tss),
                "rf_auc": float(rf_auc),
                "rf_tss": float(rf_tss),
            }
        )

    return {
        "xgb_mean_auc": float(np.mean(xgb_aucs)),
        "xgb_std_auc": float(np.std(xgb_aucs)),
        "xgb_mean_tss": float(np.mean(xgb_tsss)),
        "xgb_std_tss": float(np.std(xgb_tsss)),
        "rf_mean_auc": float(np.mean(rf_aucs)),
        "rf_std_auc": float(np.std(rf_aucs)),
        "rf_mean_tss": float(np.mean(rf_tsss)),
        "rf_std_tss": float(np.std(rf_tsss)),
        "xgb_aucs": xgb_aucs,
        "rf_aucs": rf_aucs,
        "fold_rows": fold_rows,
    }


def min_volcano_distance_km(x_utm: np.ndarray, y_utm: np.ndarray) -> np.ndarray:
    import pyproj

    transformer = pyproj.Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(x_utm, y_utm)
    geod = pyproj.Geod(ellps="WGS84")
    min_dists = np.full(len(lons), np.inf)
    for _, (vlat, vlon) in VOLCANOES.items():
        _, _, dists = geod.inv(
            np.full(len(lons), vlon), np.full(len(lons), vlat), lons, lats
        )
        min_dists = np.minimum(min_dists, dists / 1000)
    return min_dists


def verdict_from_auc(best_auc: float) -> str:
    if best_auc >= 0.85:
        return "EXCELLENT (>0.85)"
    if best_auc >= 0.75:
        return "GOOD - MVR MET (>0.75)"
    if best_auc >= 0.65:
        return "REVISIT (0.65-0.75)"
    return "KILL SIGNAL (<0.65)"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("E012: Settlement Suitability Model v6 (TGB proxy enrichment + sweep)")
    print("=" * 60)

    raster_files = {
        "elevation": DEM_DIR / "jatim_dem.tif",
        "slope": DEM_DIR / "jatim_slope.tif",
        "twi": DEM_DIR / "jatim_twi.tif",
        "tri": DEM_DIR / "jatim_tri.tif",
        "aspect": DEM_DIR / "jatim_aspect.tif",
        "river_dist": DEM_DIR / "jatim_river_dist.tif",
    }
    road_dist_path = DEM_DIR / "jatim_road_dist_expanded.tif"

    print("\nLoading rasters...")
    for name, path in raster_files.items():
        if not path.exists():
            print(f"ERROR: Missing {path}")
            sys.exit(1)
    if not road_dist_path.exists():
        print(f"ERROR: Missing {road_dist_path}. Run:")
        print("  py experiments/E012_settlement_model_v6/00_prepare_road_proxy_expanded.py")
        sys.exit(1)

    rasters = {}
    ref_bounds = None
    for name, path in raster_files.items():
        arr, transform, crs, bounds = load_raster(path)
        rasters[name] = (arr, transform, crs, bounds)
        if ref_bounds is None:
            ref_bounds = bounds
        print(f"  {name}: {arr.shape}, range [{np.nanmin(arr):.1f}, {np.nanmax(arr):.1f}]")
    road_arr, road_transform, _, _ = load_raster(road_dist_path)
    print(f"  road_dist_expanded proxy: range [{np.nanmin(road_arr):.1f}, {np.nanmax(road_arr):.1f}]")

    print("\nLoading sites and site features...")
    sites = load_sites()
    sites_proj = sites.to_crs("EPSG:32749")
    rasters_simple = {k: (v[0], v[1]) for k, v in rasters.items()}
    site_feats = extract_features_at_sites(sites, rasters_simple)
    site_feats = site_feats.dropna(subset=FEAT_COLS)
    site_feats = site_feats[site_feats["elevation"] > 0].copy()
    n_pa_target = len(site_feats) * PSEUDOABSENCE_RATIO
    print(f"  Sites with valid features: {len(site_feats)}")
    print(f"  Pseudo-absence target per config: {n_pa_target}")

    bounds_utm = (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
    sweep_rows = []
    print("\nRunning TGB parameter sweep...")
    total_cfg = len(DECAY_GRID) * len(MAX_ROAD_DIST_GRID)
    cfg_idx = 0
    for decay_m in DECAY_GRID:
        for max_road_dist_m in MAX_ROAD_DIST_GRID:
            cfg_idx += 1
            cfg_seed = RANDOM_SEED + cfg_idx * 101
            rng = np.random.default_rng(cfg_seed)
            pa_feats = generate_tgb_pseudo_absences(
                sites_proj=sites_proj,
                bounds=bounds_utm,
                n=n_pa_target,
                rasters=rasters_simple,
                road_arr=road_arr,
                road_transform=road_transform,
                decay_m=decay_m,
                max_road_dist_m=max_road_dist_m,
                min_prob=TGB_MIN_ACCEPT_PROB,
                rng=rng,
            )
            if len(pa_feats) < int(0.9 * n_pa_target):
                print(f"  [{cfg_idx}/{total_cfg}] decay={decay_m:.0f}, max={max_road_dist_m:.0f} -> "
                      f"insufficient pseudo-absences ({len(pa_feats)})")
                continue

            site_df = site_feats[FEAT_COLS + ["x", "y"]].copy()
            site_df["presence"] = 1
            pa_df = pa_feats[FEAT_COLS + ["x", "y"]].copy()
            pa_df["presence"] = 0
            train_df = pd.concat([site_df, pa_df], ignore_index=True)
            train_df = train_df.dropna(subset=FEAT_COLS)

            metrics = run_spatial_cv(train_df)
            best_auc = max(metrics["xgb_mean_auc"], metrics["rf_mean_auc"])
            row = {
                "cfg_seed": cfg_seed,
                "decay_m": decay_m,
                "max_road_dist_m": max_road_dist_m,
                "min_accept_prob": TGB_MIN_ACCEPT_PROB,
                "n_presences": int(site_df.shape[0]),
                "n_pseudoabsences": int(pa_df.shape[0]),
                "pa_road_mean_m": float(pa_feats["road_dist_tgb"].mean()),
                "pa_road_median_m": float(pa_feats["road_dist_tgb"].median()),
                "pa_accept_mean": float(pa_feats["accept_prob_tgb"].mean()),
                "xgb_mean_auc": metrics["xgb_mean_auc"],
                "xgb_std_auc": metrics["xgb_std_auc"],
                "xgb_mean_tss": metrics["xgb_mean_tss"],
                "rf_mean_auc": metrics["rf_mean_auc"],
                "rf_std_auc": metrics["rf_std_auc"],
                "rf_mean_tss": metrics["rf_mean_tss"],
                "best_auc": float(best_auc),
            }
            sweep_rows.append(row)
            print(f"  [{cfg_idx}/{total_cfg}] decay={decay_m:.0f} max={max_road_dist_m:.0f} "
                  f"| XGB AUC={metrics['xgb_mean_auc']:.3f} | RF AUC={metrics['rf_mean_auc']:.3f} "
                  f"| BEST={best_auc:.3f}")

    if not sweep_rows:
        print("ERROR: Sweep produced no valid configurations.")
        sys.exit(1)

    sweep_df = pd.DataFrame(sweep_rows).sort_values("best_auc", ascending=False).reset_index(drop=True)
    sweep_csv = RESULTS_DIR / "sweep_results.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"\nSaved sweep table: {sweep_csv}")

    best_cfg = sweep_df.iloc[0]
    print("\nBest configuration:")
    print(f"  decay={best_cfg['decay_m']:.0f} m, max_road_dist={best_cfg['max_road_dist_m']:.0f} m, "
          f"seed={int(best_cfg['cfg_seed'])}")
    print(f"  XGB AUC={best_cfg['xgb_mean_auc']:.3f}, RF AUC={best_cfg['rf_mean_auc']:.3f}, "
          f"best={best_cfg['best_auc']:.3f}")

    print("\nRebuilding dataset with best configuration...")
    best_rng = np.random.default_rng(int(best_cfg["cfg_seed"]))
    pa_best = generate_tgb_pseudo_absences(
        sites_proj=sites_proj,
        bounds=bounds_utm,
        n=n_pa_target,
        rasters=rasters_simple,
        road_arr=road_arr,
        road_transform=road_transform,
        decay_m=float(best_cfg["decay_m"]),
        max_road_dist_m=float(best_cfg["max_road_dist_m"]),
        min_prob=float(best_cfg["min_accept_prob"]),
        rng=best_rng,
    )
    site_df = site_feats[FEAT_COLS + ["x", "y"]].copy()
    site_df["presence"] = 1
    pa_df = pa_best[FEAT_COLS + ["x", "y"]].copy()
    pa_df["presence"] = 0
    train_df = pd.concat([site_df, pa_df], ignore_index=True).dropna(subset=FEAT_COLS)

    best_metrics = run_spatial_cv(train_df)
    best_auc = max(best_metrics["xgb_mean_auc"], best_metrics["rf_mean_auc"])
    verdict = verdict_from_auc(best_auc)
    print(f"  Final CV summary for best config: XGB={best_metrics['xgb_mean_auc']:.3f}, "
          f"RF={best_metrics['rf_mean_auc']:.3f}, verdict={verdict}")

    X = train_df[FEAT_COLS].values
    y = train_df["presence"].values
    scale_pw_all = (y == 0).sum() / max((y == 1).sum(), 1)
    final_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pw_all,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        verbosity=0,
        random_state=RANDOM_SEED,
    )
    final_xgb.fit(X, y)
    importance = dict(zip(FEAT_COLS, final_xgb.feature_importances_))

    print("\nGenerating suitability map + Challenge 1...")
    grid_gdf = build_feature_grid(rasters, FEAT_COLS)
    grid_prob = final_xgb.predict_proba(grid_gdf[FEAT_COLS].values)[:, 1]
    grid_gdf["suitability"] = grid_prob
    grid_wgs = grid_gdf.to_crs("EPSG:4326")
    grid_gdf["lon"] = grid_wgs.geometry.x
    grid_gdf["lat"] = grid_wgs.geometry.y
    grid_gdf["volcano_dist_km"] = min_volcano_distance_km(
        grid_gdf["x"].values, grid_gdf["y"].values
    )
    rho_taut, p_taut = spearmanr(grid_gdf["volcano_dist_km"], grid_gdf["suitability"])
    high_suit = grid_gdf[grid_gdf["suitability"] >= grid_gdf["suitability"].quantile(0.75)]
    near_volc_high = float((high_suit["volcano_dist_km"] <= 50).mean())
    if rho_taut > 0.3:
        ch1 = "TAUTOLOGY RISK"
    elif rho_taut > 0:
        ch1 = "MILD TAUTOLOGY"
    else:
        ch1 = "TAUTOLOGY-FREE"

    top5_lines = []
    for i, row in sweep_df.head(5).iterrows():
        top5_lines.append(
            f"  {i+1}. decay={row['decay_m']:.0f}, max={row['max_road_dist_m']:.0f}, "
            f"XGB={row['xgb_mean_auc']:.3f}, RF={row['rf_mean_auc']:.3f}, BEST={row['best_auc']:.3f}"
        )

    results_txt = f"""E012 - Settlement Suitability Model v6 (TGB expanded-proxy sweep) Results
===============================================
Date: 2026-02-24
Feature set: {FEAT_COLS}
Pseudo-absence strategy: Target-Group Background (expanded road-distance weighted)
Sweep grid: decay={DECAY_GRID}, max_road_dist={MAX_ROAD_DIST_GRID}, min_prob={TGB_MIN_ACCEPT_PROB}
N configs evaluated: {len(sweep_df)}

Best configuration:
  decay_m={best_cfg['decay_m']:.0f}
  max_road_dist_m={best_cfg['max_road_dist_m']:.0f}
  seed={int(best_cfg['cfg_seed'])}
  pa_road_mean_m={best_cfg['pa_road_mean_m']:.0f}
  pa_accept_mean={best_cfg['pa_accept_mean']:.3f}

Best config CV metrics:
  XGBoost AUC: {best_metrics['xgb_mean_auc']:.3f} +/- {best_metrics['xgb_std_auc']:.3f}
  XGBoost TSS: {best_metrics['xgb_mean_tss']:.3f} +/- {best_metrics['xgb_std_tss']:.3f}
  RF AUC:      {best_metrics['rf_mean_auc']:.3f} +/- {best_metrics['rf_std_auc']:.3f}
  RF TSS:      {best_metrics['rf_mean_tss']:.3f} +/- {best_metrics['rf_std_tss']:.3f}
  Verdict: {verdict}

AUC progression:
  E007: 0.659
  E008: 0.695
  E009: 0.664
  E010: 0.711
  E011(best): 0.725
  E012(best): {best_metrics['xgb_mean_auc']:.3f}

Top 5 configs by best AUC:
{chr(10).join(top5_lines)}

Challenge 1:
  rho(suitability, volcano_dist)={rho_taut:.3f} (p={p_taut:.4f})
  High-suitability cells within 50km volcano radius={near_volc_high*100:.1f}%
  Result: {ch1}
"""
    (RESULTS_DIR / "model_results.txt").write_text(results_txt, encoding="utf-8")
    print(f"Saved: {RESULTS_DIR / 'model_results.txt'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"E012 TGB Sweep - Best config AUC={best_metrics['xgb_mean_auc']:.3f} ({verdict})",
        fontsize=11,
    )
    x = np.arange(len(best_metrics["xgb_aucs"]))
    w = 0.35
    axes[0].bar(x - w / 2, best_metrics["xgb_aucs"], width=w, label="XGBoost", color="#E53935", alpha=0.8)
    axes[0].bar(x + w / 2, best_metrics["rf_aucs"], width=w, label="RandomForest", color="#1E88E5", alpha=0.8)
    axes[0].axhline(0.75, color="green", linestyle="--", label="MVR (0.75)")
    axes[0].axhline(0.695, color="orange", linestyle="-.", alpha=0.7, label="E008 baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Fold {i + 1}" for i in range(len(best_metrics["xgb_aucs"]))])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Spatial AUC")
    axes[0].set_title("Best Config Fold AUC")
    axes[0].legend(fontsize=7)

    feats_sorted = sorted(importance.items(), key=lambda kv: kv[1])
    axes[1].barh([k for k, _ in feats_sorted], [v for _, v in feats_sorted], color="#43A047", alpha=0.85)
    axes[1].set_xlabel("Importance")
    axes[1].set_title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_cv_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'model_cv_results.png'}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    pivot = sweep_df.pivot(index="decay_m", columns="max_road_dist_m", values="best_auc")
    im = ax2.imshow(pivot.values, cmap="YlOrRd", vmin=0.6, vmax=0.8)
    ax2.set_xticks(np.arange(len(pivot.columns)))
    ax2.set_xticklabels([f"{int(v/1000)}k" for v in pivot.columns])
    ax2.set_yticks(np.arange(len(pivot.index)))
    ax2.set_yticklabels([f"{int(v/1000)}k" for v in pivot.index])
    ax2.set_xlabel("max_road_dist (m)")
    ax2.set_ylabel("decay (m)")
    ax2.set_title("TGB Sweep Heatmap (best AUC)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=8)
    fig2.colorbar(im, ax=ax2, label="Best AUC")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sweep_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'sweep_heatmap.png'}")

    m = folium.Map(
        location=[float(grid_gdf["lat"].mean()), float(grid_gdf["lon"].mean())],
        zoom_start=8,
        tiles="CartoDB positron",
    )
    map_sample = grid_gdf.iloc[::5].copy()
    colormap = plt.cm.YlOrRd
    for _, row in map_sample.iterrows():
        color = mcolors.to_hex(colormap(row["suitability"]))
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=0,
            tooltip=(f"Suit: {row['suitability']:.2f} | "
                     f"RivDist: {row['river_dist']:.0f}m | "
                     f"{row['volcano_dist_km']:.0f}km from volcano"),
        ).add_to(m)
    sites_wgs = sites.to_crs("EPSG:4326")
    for _, row in sites_wgs.iterrows():
        if row.geometry is None:
            continue
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.9,
            weight=1,
            tooltip=f"Site: {row.get('name', '?')}",
        ).add_to(m)
    m.save(str(RESULTS_DIR / "suitability_map.html"))
    print(f"Saved: {RESULTS_DIR / 'suitability_map.html'}")

    print("\n" + "=" * 60)
    print(f"E012 complete. Best AUC={best_auc:.3f} | {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
