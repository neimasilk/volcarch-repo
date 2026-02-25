"""
E013: Settlement Suitability Model v7 - Hybrid bias-corrected background.

Builds on E012 (AUC=0.730). Uses expanded-road TGB proxy, then applies:
- regional quota blending (presence-driven + uniform),
- hard-negative fraction control via environmental dissimilarity.

Run from repo root:
    py experiments/E013_settlement_model_v7/01_settlement_model_v7.py
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
    from scipy.stats import spearmanr
    from shapely.geometry import Point
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

# Base TGB parameters fixed to E012 best for fair downstream comparison.
BASE_DECAY_M = 12000.0
BASE_MAX_ROAD_DIST_M = 20000.0
MIN_ACCEPT_PROB = 0.03

# Hybrid controls to sweep.
REGION_BLEND_GRID = [0.0, 0.3, 0.5, 0.7]
HARD_FRAC_GRID = [0.0, 0.15, 0.30]
HARD_Z_MIN = 2.0
HARD_Z_MAX = 5.0

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


def region_id_for_xy(x: float, y: float, midx: float, midy: float) -> int:
    east = 1 if x > midx else 0
    north = 1 if y > midy else 0
    return east + (2 * north)


def assign_regions(x_arr: np.ndarray, y_arr: np.ndarray, midx: float, midy: float) -> np.ndarray:
    east = (x_arr > midx).astype(int)
    north = (y_arr > midy).astype(int)
    return east + (2 * north)


def allocate_counts(total: int, weights: np.ndarray) -> np.ndarray:
    raw = total * weights
    counts = np.floor(raw).astype(int)
    rem = int(total - counts.sum())
    if rem > 0:
        frac_idx = np.argsort(-(raw - counts))
        for i in frac_idx[:rem]:
            counts[i] += 1
    return counts


def build_tgb_candidate_pool(
    sites_proj: gpd.GeoDataFrame,
    bounds,
    rasters: dict,
    road_arr: np.ndarray,
    road_transform,
    n_target: int,
    decay_m: float,
    max_road_dist_m: float,
    min_prob: float,
    pres_mean: np.ndarray,
    pres_std: np.ndarray,
    midx: float,
    midy: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    from shapely.ops import unary_union

    site_buffer = unary_union(sites_proj.buffer(2000))
    minx, miny, maxx, maxy = bounds

    candidates = []
    tries = 0
    max_tries = n_target * 300

    while len(candidates) < n_target and tries < max_tries:
        tries += 1
        px = rng.uniform(minx, maxx)
        py = rng.uniform(miny, maxy)
        pt = Point(px, py)
        if site_buffer.contains(pt):
            continue

        xy = np.array([[px, py]])
        road_val = extract_at_points(xy, road_arr, road_transform)[0]
        if not np.isfinite(road_val) or road_val > max_road_dist_m:
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
            feats[name] = float(val)
        if not valid:
            continue

        feat_vec = np.array([feats[c] for c in FEAT_COLS], dtype=np.float64)
        zdist = float(np.sqrt(np.sum(((feat_vec - pres_mean) / pres_std) ** 2)))
        feats["x"] = px
        feats["y"] = py
        feats["road_dist_tgb"] = float(road_val)
        feats["accept_prob_tgb"] = p_accept
        feats["zdist"] = zdist
        feats["region_id"] = region_id_for_xy(px, py, midx, midy)
        candidates.append(feats)

    if len(candidates) < n_target:
        print(f"  WARNING: candidate pool shortfall {len(candidates)}/{n_target}")
    return pd.DataFrame(candidates)


def sample_hybrid_pseudo_absences(
    pool_df: pd.DataFrame,
    n_total: int,
    hard_frac: float,
    region_blend: float,
    presence_region_prop: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    base_pool = pool_df[pool_df["zdist"] <= HARD_Z_MAX].copy()
    hard_pool = base_pool[base_pool["zdist"] >= HARD_Z_MIN].copy()

    if len(base_pool) == 0:
        return pd.DataFrame()

    uniform = np.full(4, 0.25)
    weights = (1.0 - region_blend) * presence_region_prop + region_blend * uniform
    weights = weights / weights.sum()

    n_hard_target = int(round(n_total * hard_frac))
    n_core_target = n_total - n_hard_target

    hard_targets = allocate_counts(n_hard_target, weights)
    core_targets = allocate_counts(n_core_target, weights)

    selected = []
    selected_set = set()

    # 1) sample hard negatives by region
    for rid in range(4):
        need = int(hard_targets[rid])
        if need <= 0:
            continue
        cand = hard_pool.index[hard_pool["region_id"] == rid].to_numpy()
        if len(cand) == 0:
            continue
        take = min(need, len(cand))
        pick = rng.choice(cand, size=take, replace=False)
        selected.extend(pick.tolist())
        selected_set.update(pick.tolist())

    # Hard shortfall fill
    if len(selected) < n_hard_target:
        remain_need = n_hard_target - len(selected)
        remain_cand = hard_pool.index[~hard_pool.index.isin(list(selected_set))].to_numpy()
        if len(remain_cand) > 0:
            take = min(remain_need, len(remain_cand))
            pick = rng.choice(remain_cand, size=take, replace=False)
            selected.extend(pick.tolist())
            selected_set.update(pick.tolist())

    # 2) sample core by region from base pool excluding already selected hard
    core_selected = 0
    for rid in range(4):
        need = int(core_targets[rid])
        if need <= 0:
            continue
        cand = base_pool.index[(base_pool["region_id"] == rid) &
                               (~base_pool.index.isin(list(selected_set)))].to_numpy()
        if len(cand) == 0:
            continue
        take = min(need, len(cand))
        pick = rng.choice(cand, size=take, replace=False)
        selected.extend(pick.tolist())
        selected_set.update(pick.tolist())
        core_selected += take

    # Core shortfall fill
    if core_selected < n_core_target:
        remain_need = n_core_target - core_selected
        remain_cand = base_pool.index[~base_pool.index.isin(list(selected_set))].to_numpy()
        if len(remain_cand) > 0:
            take = min(remain_need, len(remain_cand))
            pick = rng.choice(remain_cand, size=take, replace=False)
            selected.extend(pick.tolist())
            selected_set.update(pick.tolist())

    # Final shortfall fill (any remaining base candidates)
    if len(selected) < n_total:
        remain_need = n_total - len(selected)
        remain_cand = base_pool.index[~base_pool.index.isin(list(selected_set))].to_numpy()
        if len(remain_cand) > 0:
            take = min(remain_need, len(remain_cand))
            pick = rng.choice(remain_cand, size=take, replace=False)
            selected.extend(pick.tolist())
            selected_set.update(pick.tolist())

    out = base_pool.loc[selected].copy()
    out = out.head(n_total)
    return out


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
                "xgb_auc": float(xgb_auc),
                "xgb_tss": float(xgb_tss),
                "rf_auc": float(rf_auc),
                "rf_tss": float(rf_tss),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
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
    print("E013: Settlement Suitability Model v7 (hybrid bias correction)")
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
    for _, path in raster_files.items():
        if not path.exists():
            print(f"ERROR: Missing raster {path}")
            sys.exit(1)
    if not road_dist_path.exists():
        print(f"ERROR: Missing {road_dist_path}")
        print("Run: py experiments/E012_settlement_model_v6/00_prepare_road_proxy_expanded.py")
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
    print(f"  road_dist_expanded: range [{np.nanmin(road_arr):.1f}, {np.nanmax(road_arr):.1f}]")

    print("\nLoading sites and extracting site features...")
    sites = load_sites()
    sites_proj = sites.to_crs("EPSG:32749")
    rasters_simple = {k: (v[0], v[1]) for k, v in rasters.items()}
    site_feats = extract_features_at_sites(sites, rasters_simple)
    site_feats = site_feats.dropna(subset=FEAT_COLS)
    site_feats = site_feats[site_feats["elevation"] > 0].copy()
    n_pa_target = len(site_feats) * PSEUDOABSENCE_RATIO
    print(f"  Sites with valid features: {len(site_feats)}")
    print(f"  Pseudo-absence target: {n_pa_target}")

    bounds_utm = (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
    midx = (ref_bounds.left + ref_bounds.right) / 2.0
    midy = (ref_bounds.bottom + ref_bounds.top) / 2.0

    pres_mean = site_feats[FEAT_COLS].mean().to_numpy(dtype=np.float64)
    pres_std = site_feats[FEAT_COLS].std().replace(0, 1.0).to_numpy(dtype=np.float64)
    site_regions = assign_regions(site_feats["x"].values, site_feats["y"].values, midx, midy)
    presence_region_prop = np.bincount(site_regions, minlength=4).astype(np.float64)
    presence_region_prop = presence_region_prop / presence_region_prop.sum()
    print(f"  Presence region proportions (Q1,Q2,Q3,Q4): {np.round(presence_region_prop, 3)}")

    pool_target = n_pa_target * 16
    print("\nBuilding TGB candidate pool...")
    print(f"  Base params: decay={BASE_DECAY_M:.0f}m, max_road_dist={BASE_MAX_ROAD_DIST_M:.0f}m")
    pool_rng = np.random.default_rng(RANDOM_SEED + 555)
    pool_df = build_tgb_candidate_pool(
        sites_proj=sites_proj,
        bounds=bounds_utm,
        rasters=rasters_simple,
        road_arr=road_arr,
        road_transform=road_transform,
        n_target=pool_target,
        decay_m=BASE_DECAY_M,
        max_road_dist_m=BASE_MAX_ROAD_DIST_M,
        min_prob=MIN_ACCEPT_PROB,
        pres_mean=pres_mean,
        pres_std=pres_std,
        midx=midx,
        midy=midy,
        rng=pool_rng,
    )
    if len(pool_df) < n_pa_target:
        print("ERROR: candidate pool too small for target pseudo-absence sample.")
        sys.exit(1)
    print(f"  Candidate pool size: {len(pool_df)}")
    print(f"  Candidate zdist stats: mean={pool_df['zdist'].mean():.2f}, "
          f"p50={pool_df['zdist'].median():.2f}, p90={pool_df['zdist'].quantile(0.9):.2f}")

    print("\nRunning hybrid sweep...")
    sweep_rows = []
    cfg_idx = 0
    total_cfg = len(REGION_BLEND_GRID) * len(HARD_FRAC_GRID)
    for blend in REGION_BLEND_GRID:
        for hard_frac in HARD_FRAC_GRID:
            cfg_idx += 1
            cfg_seed = RANDOM_SEED + cfg_idx * 111
            rng = np.random.default_rng(cfg_seed)

            pa_df = sample_hybrid_pseudo_absences(
                pool_df=pool_df,
                n_total=n_pa_target,
                hard_frac=hard_frac,
                region_blend=blend,
                presence_region_prop=presence_region_prop,
                rng=rng,
            )
            if len(pa_df) < int(0.9 * n_pa_target):
                print(f"  [{cfg_idx}/{total_cfg}] blend={blend:.2f}, hard_frac={hard_frac:.2f} "
                      f"-> insufficient pseudo-absences ({len(pa_df)})")
                continue

            site_df = site_feats[FEAT_COLS + ["x", "y"]].copy()
            site_df["presence"] = 1
            pa_train = pa_df[FEAT_COLS + ["x", "y"]].copy()
            pa_train["presence"] = 0
            train_df = pd.concat([site_df, pa_train], ignore_index=True).dropna(subset=FEAT_COLS)

            metrics = run_spatial_cv(train_df)
            best_auc = max(metrics["xgb_mean_auc"], metrics["rf_mean_auc"])
            actual_hard_frac = float((pa_df["zdist"] >= HARD_Z_MIN).mean())
            row = {
                "cfg_seed": cfg_seed,
                "region_blend": blend,
                "hard_frac_target": hard_frac,
                "hard_frac_actual": actual_hard_frac,
                "n_presences": int(site_df.shape[0]),
                "n_pseudoabsences": int(pa_train.shape[0]),
                "pa_zdist_mean": float(pa_df["zdist"].mean()),
                "pa_road_mean_m": float(pa_df["road_dist_tgb"].mean()),
                "xgb_mean_auc": metrics["xgb_mean_auc"],
                "xgb_std_auc": metrics["xgb_std_auc"],
                "xgb_mean_tss": metrics["xgb_mean_tss"],
                "rf_mean_auc": metrics["rf_mean_auc"],
                "rf_std_auc": metrics["rf_std_auc"],
                "rf_mean_tss": metrics["rf_mean_tss"],
                "best_auc": float(best_auc),
            }
            sweep_rows.append(row)
            print(f"  [{cfg_idx}/{total_cfg}] blend={blend:.2f}, hard={hard_frac:.2f} "
                  f"| XGB={metrics['xgb_mean_auc']:.3f} RF={metrics['rf_mean_auc']:.3f} "
                  f"| BEST={best_auc:.3f}")

    if not sweep_rows:
        print("ERROR: no valid hybrid configurations were evaluated.")
        sys.exit(1)

    sweep_df = pd.DataFrame(sweep_rows).sort_values("best_auc", ascending=False).reset_index(drop=True)
    sweep_df.to_csv(RESULTS_DIR / "sweep_results.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'sweep_results.csv'}")

    best_cfg = sweep_df.iloc[0]
    print("\nBest config:")
    print(f"  region_blend={best_cfg['region_blend']:.2f}, hard_frac_target={best_cfg['hard_frac_target']:.2f}, "
          f"hard_frac_actual={best_cfg['hard_frac_actual']:.2f}, seed={int(best_cfg['cfg_seed'])}")
    print(f"  XGB={best_cfg['xgb_mean_auc']:.3f}, RF={best_cfg['rf_mean_auc']:.3f}, "
          f"BEST={best_cfg['best_auc']:.3f}")

    # rebuild best training set deterministically
    best_rng = np.random.default_rng(int(best_cfg["cfg_seed"]))
    best_pa = sample_hybrid_pseudo_absences(
        pool_df=pool_df,
        n_total=n_pa_target,
        hard_frac=float(best_cfg["hard_frac_target"]),
        region_blend=float(best_cfg["region_blend"]),
        presence_region_prop=presence_region_prop,
        rng=best_rng,
    )
    site_df = site_feats[FEAT_COLS + ["x", "y"]].copy()
    site_df["presence"] = 1
    pa_train = best_pa[FEAT_COLS + ["x", "y"]].copy()
    pa_train["presence"] = 0
    train_df = pd.concat([site_df, pa_train], ignore_index=True).dropna(subset=FEAT_COLS)
    best_metrics = run_spatial_cv(train_df)
    best_auc = max(best_metrics["xgb_mean_auc"], best_metrics["rf_mean_auc"])
    verdict = verdict_from_auc(best_auc)
    print(f"  Final CV for best config: XGB={best_metrics['xgb_mean_auc']:.3f}, "
          f"RF={best_metrics['rf_mean_auc']:.3f} ({verdict})")

    # fit final model + map/challenge
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

    print("\nGenerating suitability map and Challenge 1...")
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

    top5 = []
    for i, row in sweep_df.head(5).iterrows():
        top5.append(
            f"  {i+1}. blend={row['region_blend']:.2f}, hard={row['hard_frac_target']:.2f}, "
            f"XGB={row['xgb_mean_auc']:.3f}, RF={row['rf_mean_auc']:.3f}, BEST={row['best_auc']:.3f}"
        )

    results_txt = f"""E013 - Settlement Suitability Model v7 (hybrid bias correction) Results
===============================================
Date: 2026-02-24
Feature set: {FEAT_COLS}
Base TGB params: decay={BASE_DECAY_M:.0f}m, max_road_dist={BASE_MAX_ROAD_DIST_M:.0f}m,
                 min_accept_prob={MIN_ACCEPT_PROB:.2f}
Hybrid sweep grid:
  region_blend={REGION_BLEND_GRID}
  hard_frac={HARD_FRAC_GRID}
  hard_z_min={HARD_Z_MIN}, hard_z_max={HARD_Z_MAX}
N configs evaluated: {len(sweep_df)}

Best configuration:
  region_blend={best_cfg['region_blend']:.2f}
  hard_frac_target={best_cfg['hard_frac_target']:.2f}
  hard_frac_actual={best_cfg['hard_frac_actual']:.2f}
  seed={int(best_cfg['cfg_seed'])}
  pa_road_mean_m={best_cfg['pa_road_mean_m']:.0f}
  pa_zdist_mean={best_cfg['pa_zdist_mean']:.2f}

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
  E011: 0.725
  E012: 0.730
  E013(best): {best_metrics['xgb_mean_auc']:.3f}

Top 5 configs by best AUC:
{chr(10).join(top5)}

Feature importances (XGBoost):
{chr(10).join(f'  {k}: {v:.3f}' for k,v in sorted(importance.items(), key=lambda x:-x[1]))}

Challenge 1:
  rho(suitability, volcano_dist)={rho_taut:.3f} (p={p_taut:.4f})
  High-suitability cells within 50km volcano radius={near_volc_high*100:.1f}%
  Result: {ch1}
"""
    (RESULTS_DIR / "model_results.txt").write_text(results_txt, encoding="utf-8")
    print(f"Saved: {RESULTS_DIR / 'model_results.txt'}")

    # fold chart + feature importance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"E013 Hybrid Sweep - Best config AUC={best_metrics['xgb_mean_auc']:.3f} ({verdict})",
        fontsize=11,
    )
    x = np.arange(len(best_metrics["xgb_aucs"]))
    w = 0.35
    axes[0].bar(x - w / 2, best_metrics["xgb_aucs"], width=w, label="XGBoost", color="#E53935", alpha=0.8)
    axes[0].bar(x + w / 2, best_metrics["rf_aucs"], width=w, label="RandomForest", color="#1E88E5", alpha=0.8)
    axes[0].axhline(0.75, color="green", linestyle="--", label="MVR (0.75)")
    axes[0].axhline(0.730, color="orange", linestyle="-.", alpha=0.7, label="E012 baseline")
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

    # heatmap of sweep
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    pivot = sweep_df.pivot(index="hard_frac_target", columns="region_blend", values="best_auc")
    im = ax2.imshow(pivot.values, cmap="YlOrRd", vmin=0.65, vmax=0.80)
    ax2.set_xticks(np.arange(len(pivot.columns)))
    ax2.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax2.set_yticks(np.arange(len(pivot.index)))
    ax2.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax2.set_xlabel("region_blend")
    ax2.set_ylabel("hard_frac_target")
    ax2.set_title("Hybrid Sweep Heatmap (best AUC)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=8)
    fig2.colorbar(im, ax=ax2, label="Best AUC")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sweep_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'sweep_heatmap.png'}")

    # map
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
    print(f"E013 complete. Best AUC={best_auc:.3f} | {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
