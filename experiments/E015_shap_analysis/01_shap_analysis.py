"""
E015: SHAP Analysis for E013 Best XGBoost Model.

Rebuilds E013's best training set (region_blend=0.00, hard_frac=0.30, seed=375),
retrains the identical XGBoost model, then computes TreeSHAP values.

Outputs:
  - results/shap_beeswarm.png       (manuscript Figure 13)
  - results/shap_bar.png            (mean |SHAP| bar chart)
  - results/shap_summary.csv        (feature-level SHAP statistics)
  - results/shap_analysis_report.txt

Run from repo root:
    python experiments/E015_shap_analysis/01_shap_analysis.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import rowcol
    from shapely.geometry import Point
    import xgboost as xgb
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install shap xgboost geopandas rasterio matplotlib")
    sys.exit(1)

# === Paths ===
REPO_ROOT = Path(__file__).parent.parent.parent
DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"
SITES_PATH = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# === E013 best config (deterministic reproduction) ===
FEAT_COLS = ["elevation", "slope", "twi", "tri", "aspect", "river_dist"]
FEAT_LABELS = {
    "elevation": "Elevation (m)",
    "slope": "Slope (degrees)",
    "twi": "TWI",
    "tri": "TRI",
    "aspect": "Aspect (degrees)",
    "river_dist": "River distance (m)",
}

# E013 best configuration
BASE_DECAY_M = 12000.0
BASE_MAX_ROAD_DIST_M = 20000.0
MIN_ACCEPT_PROB = 0.03
BEST_REGION_BLEND = 0.0
BEST_HARD_FRAC = 0.30
HARD_Z_MIN = 2.0
HARD_Z_MAX = 5.0
PSEUDOABSENCE_RATIO = 5
RANDOM_SEED = 42
CONFIG_SEED = 375

BLOCK_SIZE_DEG = 0.45


def load_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, src.transform, src.crs, src.bounds


def extract_at_points(points_xy, raster_arr, transform):
    rows, cols = rowcol(transform, points_xy[:, 0], points_xy[:, 1])
    rows = np.array(rows)
    cols = np.array(cols)
    h, w = raster_arr.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    values = np.full(len(points_xy), np.nan, dtype=np.float32)
    values[valid] = raster_arr[rows[valid], cols[valid]]
    return values


def load_sites():
    gdf = gpd.read_file(SITES_PATH)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs("EPSG:4326")
    jatim = (-9.0, 111.0, -6.5, 115.0)
    gdf = gdf[(gdf.geometry.y >= jatim[0]) & (gdf.geometry.x >= jatim[1]) &
              (gdf.geometry.y <= jatim[2]) & (gdf.geometry.x <= jatim[3])]
    return gdf


def extract_features_at_sites(sites, rasters):
    sites_proj = sites.to_crs("EPSG:32749")
    xy = np.column_stack([sites_proj.geometry.x, sites_proj.geometry.y])
    out = {}
    for name, (arr, transform, *_) in rasters.items():
        out[name] = extract_at_points(xy, arr, transform)
    df = pd.DataFrame(out)
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]
    return df


def region_id_for_xy(x, y, midx, midy):
    east = 1 if x > midx else 0
    north = 1 if y > midy else 0
    return east + (2 * north)


def allocate_counts(total, weights):
    raw = total * weights
    counts = np.floor(raw).astype(int)
    rem = int(total - counts.sum())
    if rem > 0:
        frac_idx = np.argsort(-(raw - counts))
        for i in frac_idx[:rem]:
            counts[i] += 1
    return counts


def build_tgb_candidate_pool(sites_proj, bounds, rasters, road_arr, road_transform,
                              n_target, pres_mean, pres_std, midx, midy, rng):
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
        if not np.isfinite(road_val) or road_val > BASE_MAX_ROAD_DIST_M:
            continue

        p_accept = max(MIN_ACCEPT_PROB, min(1.0, float(np.exp(-road_val / BASE_DECAY_M))))
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
        feats["zdist"] = zdist
        feats["region_id"] = region_id_for_xy(px, py, midx, midy)
        candidates.append(feats)

    return pd.DataFrame(candidates)


def sample_hybrid_pseudo_absences(pool_df, n_total, hard_frac, region_blend,
                                   presence_region_prop, rng):
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

    if len(selected) < n_hard_target:
        remain_need = n_hard_target - len(selected)
        remain_cand = hard_pool.index[~hard_pool.index.isin(list(selected_set))].to_numpy()
        if len(remain_cand) > 0:
            take = min(remain_need, len(remain_cand))
            pick = rng.choice(remain_cand, size=take, replace=False)
            selected.extend(pick.tolist())
            selected_set.update(pick.tolist())

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

    return pool_df.loc[selected].reset_index(drop=True)


def main():
    print("=" * 60)
    print("E015: SHAP Analysis for E013 Best XGBoost Model")
    print("=" * 60)

    # --- Load rasters ---
    print("\n[1/6] Loading rasters...")
    raster_files = {
        "elevation": "jatim_dem.tif",
        "slope": "jatim_slope.tif",
        "twi": "jatim_twi.tif",
        "tri": "jatim_tri.tif",
        "aspect": "jatim_aspect.tif",
        "river_dist": "jatim_river_dist.tif",
    }
    rasters = {}
    bounds = None
    for name, fname in raster_files.items():
        path = DEM_DIR / fname
        if not path.exists():
            print(f"  ERROR: {path} not found")
            sys.exit(1)
        arr, transform, crs, bnds = load_raster(path)
        rasters[name] = (arr, transform, crs, bnds)
        if bounds is None:
            bounds = bnds
        print(f"  {name}: {arr.shape}")

    # Load road distance for TGB
    road_path = DEM_DIR / "jatim_road_dist_expanded.tif"
    road_arr, road_transform, _, _ = load_raster(road_path)
    print(f"  road_dist_expanded: {road_arr.shape}")

    # --- Load sites ---
    print("\n[2/6] Loading sites and extracting features...")
    sites = load_sites()
    print(f"  Sites in Jatim bbox: {len(sites)}")
    site_feats = extract_features_at_sites(sites, rasters)
    valid_mask = site_feats[FEAT_COLS].notna().all(axis=1) & (site_feats["elevation"] > 0)
    site_feats = site_feats[valid_mask].reset_index(drop=True)
    n_sites = len(site_feats)
    print(f"  Valid sites with features: {n_sites}")

    # --- Build TGB candidate pool (same as E013) ---
    print("\n[3/6] Building TGB candidate pool (E013 best config)...")
    sites_proj = sites.to_crs("EPSG:32749")
    pres_vals = site_feats[FEAT_COLS].values
    pres_mean = np.nanmean(pres_vals, axis=0)
    pres_std = np.nanstd(pres_vals, axis=0)
    pres_std[pres_std < 1e-6] = 1.0

    midx = (bounds.left + bounds.right) / 2
    midy = (bounds.bottom + bounds.top) / 2

    n_pa_target = n_sites * PSEUDOABSENCE_RATIO
    pool_size = n_pa_target * 16

    pool_rng = np.random.default_rng(RANDOM_SEED)
    pool_df = build_tgb_candidate_pool(
        sites_proj=sites_proj, bounds=bounds, rasters=rasters,
        road_arr=road_arr, road_transform=road_transform,
        n_target=pool_size, pres_mean=pres_mean, pres_std=pres_std,
        midx=midx, midy=midy, rng=pool_rng,
    )
    print(f"  Candidate pool: {len(pool_df)}")

    # Presence region proportions
    site_regions = np.array([region_id_for_xy(x, y, midx, midy)
                             for x, y in zip(site_feats["x"], site_feats["y"])])
    presence_region_prop = np.array([(site_regions == r).sum() / len(site_regions) for r in range(4)])

    # --- Sample pseudo-absences (best config) ---
    print("\n[4/6] Sampling hybrid pseudo-absences...")
    best_rng = np.random.default_rng(CONFIG_SEED)
    best_pa = sample_hybrid_pseudo_absences(
        pool_df=pool_df, n_total=n_pa_target,
        hard_frac=BEST_HARD_FRAC, region_blend=BEST_REGION_BLEND,
        presence_region_prop=presence_region_prop, rng=best_rng,
    )
    print(f"  Pseudo-absences sampled: {len(best_pa)}")

    # Build training set
    site_df = site_feats[FEAT_COLS + ["x", "y"]].copy()
    site_df["presence"] = 1
    pa_train = best_pa[FEAT_COLS + ["x", "y"]].copy()
    pa_train["presence"] = 0
    train_df = pd.concat([site_df, pa_train], ignore_index=True).dropna(subset=FEAT_COLS)
    print(f"  Training set: {len(train_df)} ({(train_df['presence']==1).sum()} pos, "
          f"{(train_df['presence']==0).sum()} neg)")

    # --- Train XGBoost (identical to E013) ---
    print("\n[5/6] Training XGBoost model...")
    X = train_df[FEAT_COLS].values
    y = train_df["presence"].values
    scale_pw = (y == 0).sum() / max((y == 1).sum(), 1)

    model = xgb.XGBClassifier(
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
    model.fit(X, y)

    # Gain-based importance for comparison
    gain_importance = dict(zip(FEAT_COLS, model.feature_importances_))
    print("  Gain-based importance:")
    for f in sorted(gain_importance, key=gain_importance.get, reverse=True):
        print(f"    {f}: {gain_importance[f]:.3f}")

    # --- Compute SHAP values ---
    print("\n[6/6] Computing SHAP values (TreeSHAP)...")
    explainer = shap.TreeExplainer(model)
    X_df = pd.DataFrame(X, columns=[FEAT_LABELS.get(c, c) for c in FEAT_COLS])
    shap_values = explainer(X_df)

    # Mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_summary = pd.DataFrame({
        "feature": FEAT_COLS,
        "feature_label": [FEAT_LABELS.get(c, c) for c in FEAT_COLS],
        "mean_abs_shap": mean_abs_shap,
        "gain_importance": [gain_importance[c] for c in FEAT_COLS],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_summary["shap_rank"] = range(1, len(shap_summary) + 1)
    shap_summary["gain_rank"] = shap_summary["gain_importance"].rank(ascending=False).astype(int)
    shap_summary.to_csv(RESULTS_DIR / "shap_summary.csv", index=False)
    print("\n  SHAP vs Gain ranking:")
    print(shap_summary[["feature_label", "mean_abs_shap", "shap_rank",
                         "gain_importance", "gain_rank"]].to_string(index=False))

    # --- Generate plots ---
    print("\nGenerating SHAP plots...")

    # Beeswarm plot (main manuscript figure)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.beeswarm(shap_values, show=False, max_display=len(FEAT_COLS))
    plt.title("E015: SHAP Feature Importance (E013 Best XGBoost)", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'shap_beeswarm.png'}")

    # Bar plot (supplementary)
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.bar(shap_values, show=False, max_display=len(FEAT_COLS))
    plt.title("E015: Mean |SHAP| per Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'shap_bar.png'}")

    # Dependence plot for top feature
    top_feat_label = shap_summary.iloc[0]["feature_label"]
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.scatter(shap_values[:, top_feat_label], show=False)
    plt.title(f"SHAP Dependence: {top_feat_label}", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_dependence_top.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'shap_dependence_top.png'}")

    # --- Rank consistency check ---
    rank_corr = shap_summary[["shap_rank", "gain_rank"]].corr(method="spearman").iloc[0, 1]
    consistency = "CONSISTENT" if rank_corr > 0.7 else "DIVERGENT" if rank_corr < 0.3 else "MODERATE"

    # --- Write report ---
    report = f"""E015: SHAP Analysis for E013 Best XGBoost Model
{'=' * 55}
Date: 2026-03-03
Model: XGBoost (n_estimators=300, max_depth=4, lr=0.05)
Training set: {len(train_df)} samples ({(y==1).sum()} presences, {(y==0).sum()} pseudo-absences)
E013 config: region_blend={BEST_REGION_BLEND}, hard_frac={BEST_HARD_FRAC}, seed={CONFIG_SEED}

SHAP Feature Ranking (mean |SHAP|):
{shap_summary[['feature_label', 'mean_abs_shap', 'shap_rank']].to_string(index=False)}

Gain-Based Feature Ranking:
{shap_summary.sort_values('gain_rank')[['feature_label', 'gain_importance', 'gain_rank']].to_string(index=False)}

Rank Consistency (Spearman rho): {rank_corr:.3f} ({consistency})

Interpretation:
- SHAP provides instance-level feature attribution, complementing gain-based importance.
- TreeSHAP is exact for tree models (no sampling approximation needed).
- {"Rankings are broadly consistent, confirming gain-based importance." if consistency == "CONSISTENT" else "Some ranking differences between SHAP and gain may reflect feature interaction effects captured by SHAP but not by gain-based importance."}

Output files:
  - shap_beeswarm.png  (manuscript Figure 13)
  - shap_bar.png        (supplementary)
  - shap_dependence_top.png (top feature dependence)
  - shap_summary.csv    (numerical summary)
"""
    with open(RESULTS_DIR / "shap_analysis_report.txt", "w") as f:
        f.write(report)
    print(f"\n  Report: {RESULTS_DIR / 'shap_analysis_report.txt'}")
    print(f"\nRank consistency: {rank_corr:.3f} ({consistency})")
    print("\nE015 COMPLETE.")


if __name__ == "__main__":
    main()
