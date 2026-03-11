"""
Precompute dashboard data from E013/E015/E016 pipelines.

Retrains E013 best model (identical pipeline), generates suitability grid
at ~900m spacing, computes Pyle (1989) burial depth with Dwarapala
calibration, assigns zones (A/B/C/E), and computes TreeSHAP values.

Outputs to data/processed/dashboard/:
  - grid_predictions.csv  (~65k rows: lon, lat, suitability, burial_depth_cm, zone)
  - model_xgb.json        (serialized XGBoost model)
  - shap_values.npz       (SHAP matrix + feature names)
  - shap_beeswarm.png     (beeswarm plot)
  - shap_bar.png          (mean |SHAP| bar chart)
  - shap_summary.csv      (feature-level SHAP statistics)
  - sites.csv             (archaeological sites with predictions)
  - volcanoes.csv         (volcano locations)
  - zone_statistics.csv   (zone-level statistics)
  - metadata.json         (model stats, thresholds, validation results)

Run from repo root:
    python tools/precompute_dashboard_data.py
"""

import sys
import json
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
    from shapely.ops import unary_union
    import xgboost as xgb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install geopandas rasterio xgboost matplotlib shap")
    sys.exit(1)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed, SHAP analysis will be skipped")
    print("Install with: pip install shap")

# ── Paths ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"
SITES_PATH = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
ERUPTION_PATH = REPO_ROOT / "data" / "processed" / "eruption_history.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "dashboard"

# ── Features ──────────────────────────────────────────────────────────────
FEAT_COLS = ["elevation", "slope", "twi", "tri", "aspect", "river_dist"]
FEAT_LABELS = {
    "elevation": "Elevation (m)",
    "slope": "Slope (degrees)",
    "twi": "TWI",
    "tri": "TRI",
    "aspect": "Aspect (degrees)",
    "river_dist": "River distance (m)",
}

# ── Volcanoes (EPSG:4326) ────────────────────────────────────────────────
VOLCANOES = {
    "Kelud":           {"lat": -7.9300, "lon": 112.3080, "gvp_id": 263280},
    "Semeru":          {"lat": -8.1080, "lon": 112.9220, "gvp_id": 263300},
    "Arjuno-Welirang": {"lat": -7.7290, "lon": 112.5750, "gvp_id": 263260},
    "Bromo":           {"lat": -7.9420, "lon": 112.9500, "gvp_id": 263310},
    "Lamongan":        {"lat": -7.9770, "lon": 113.3430, "gvp_id": 263320},
    "Raung":           {"lat": -8.1250, "lon": 114.0420, "gvp_id": 263340},
    "Ijen":            {"lat": -8.0580, "lon": 114.2420, "gvp_id": 263350},
}

# ── Pyle (1989) exponential thinning parameters ─────────────────────────
PYLE_PARAMS = {
    0: {"T0": 0.1, "k": 0.15},
    1: {"T0": 0.5, "k": 0.12},
    2: {"T0": 3.0, "k": 0.08},
    3: {"T0": 15.0, "k": 0.06},
    4: {"T0": 80.0, "k": 0.05},
    5: {"T0": 500.0, "k": 0.04},
}

# ── E013 best configuration ─────────────────────────────────────────────
BASE_DECAY_M = 12000.0
BASE_MAX_ROAD_DIST_M = 20000.0
MIN_ACCEPT_PROB = 0.03
PSEUDOABSENCE_RATIO = 5
RANDOM_SEED = 42
CONFIG_SEED = 375
BEST_REGION_BLEND = 0.0
BEST_HARD_FRAC = 0.30
HARD_Z_MIN = 2.0
HARD_Z_MAX = 5.0

# ── Dwarapala validation ────────────────────────────────────────────────
DWARAPALA = {"lat": -7.973, "lon": 112.435, "actual_depth_cm": 185, "year_built": 1268}

# ── Zone thresholds ─────────────────────────────────────────────────────
BURIAL_SHALLOW = 100   # cm
BURIAL_MODERATE = 300  # cm

# ── Grid spacing ────────────────────────────────────────────────────────
GRID_STEP = 30  # ~900m at 30m raster resolution


def load_raster(path):
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


def compute_burial_depth_grid(grid_lon, grid_lat, eruptions_since_1268):
    """Cumulative burial depth (cm) from Pyle (1989) exponential thinning."""
    n = len(grid_lon)
    depth = np.zeros(n, dtype=np.float64)
    for _, erup in eruptions_since_1268.iterrows():
        vei = erup["vei"]
        if pd.isna(vei) or int(vei) not in PYLE_PARAMS:
            continue
        volcano_name = erup["volcano"]
        if volcano_name not in VOLCANOES:
            continue
        vlat = VOLCANOES[volcano_name]["lat"]
        vlon = VOLCANOES[volcano_name]["lon"]
        dlat = np.radians(grid_lat - vlat)
        dlon = np.radians(grid_lon - vlon)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(vlat)) * np.cos(np.radians(grid_lat))
             * np.sin(dlon / 2) ** 2)
        dist_km = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        p = PYLE_PARAMS[int(vei)]
        thickness = p["T0"] * np.exp(-p["k"] * dist_km)
        thickness[thickness < 0.01] = 0.0
        depth += thickness
    return depth


def allocate_counts(total, weights):
    raw = total * weights
    counts = np.floor(raw).astype(int)
    rem = int(total - counts.sum())
    if rem > 0:
        frac_idx = np.argsort(-(raw - counts))
        for i in frac_idx[:rem]:
            counts[i] += 1
    return counts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Precompute Dashboard Data")
    print("=" * 60)

    # ─── 1. Load eruption history ────────────────────────────────────────
    print("\n[1/8] Loading eruption history...")
    eruptions = pd.read_csv(ERUPTION_PATH)
    eruptions_since_1268 = eruptions[eruptions["year"] >= 1268].copy()
    print(f"  Eruptions since 1268 CE: {len(eruptions_since_1268)}")

    # ─── 2. Load rasters ────────────────────────────────────────────────
    print("\n[2/8] Loading rasters...")
    raster_files = {
        "elevation": "jatim_dem.tif",
        "slope": "jatim_slope.tif",
        "twi": "jatim_twi.tif",
        "tri": "jatim_tri.tif",
        "aspect": "jatim_aspect.tif",
        "river_dist": "jatim_river_dist.tif",
    }
    rasters = {}
    ref_bounds = None
    ref_transform = None
    ref_shape = None
    for name, fname in raster_files.items():
        path = DEM_DIR / fname
        if not path.exists():
            print(f"  ERROR: Missing raster {path}")
            sys.exit(1)
        arr, transform, crs, bnds = load_raster(path)
        rasters[name] = (arr, transform, crs, bnds)
        if ref_bounds is None:
            ref_bounds = bnds
            ref_transform = transform
            ref_shape = arr.shape
        print(f"  {name}: {arr.shape}")

    road_path = DEM_DIR / "jatim_road_dist_expanded.tif"
    if not road_path.exists():
        print(f"  ERROR: Missing {road_path}")
        sys.exit(1)
    road_arr, road_transform, _, _ = load_raster(road_path)
    print(f"  road_dist_expanded: {road_arr.shape}")

    # ─── 3. Load sites + extract features ────────────────────────────────
    print("\n[3/8] Loading sites and extracting features...")
    sites = gpd.read_file(SITES_PATH)
    sites = sites[sites.geometry.notna() & ~sites.geometry.is_empty].to_crs("EPSG:4326")
    jatim = (-9.0, 111.0, -6.5, 115.0)
    sites = sites[
        (sites.geometry.y >= jatim[0]) & (sites.geometry.x >= jatim[1])
        & (sites.geometry.y <= jatim[2]) & (sites.geometry.x <= jatim[3])
    ]

    sites_proj = sites.to_crs("EPSG:32749")
    xy = np.column_stack([sites_proj.geometry.x, sites_proj.geometry.y])
    site_feats = {}
    for name, (arr, transform, *_) in rasters.items():
        site_feats[name] = extract_at_points(xy, arr, transform)
    site_df = pd.DataFrame(site_feats)
    site_df["x"] = xy[:, 0]
    site_df["y"] = xy[:, 1]
    valid_mask = site_df[FEAT_COLS].notna().all(axis=1) & (site_df["elevation"] > 0)
    site_df = site_df[valid_mask].reset_index(drop=True)
    sites_valid = sites.iloc[valid_mask.values[: len(sites)]].reset_index(drop=True)
    n_sites = len(site_df)
    print(f"  Valid sites: {n_sites}")

    # ─── 4. Build TGB pool + sample PAs (matches E016/E015) ─────────────
    print("\n[4/8] Building pseudo-absence candidate pool...")
    pres_vals = site_df[FEAT_COLS].values
    pres_mean = np.nanmean(pres_vals, axis=0)
    pres_std = np.nanstd(pres_vals, axis=0)
    pres_std[pres_std < 1e-6] = 1.0

    midx = (ref_bounds.left + ref_bounds.right) / 2
    midy = (ref_bounds.bottom + ref_bounds.top) / 2

    n_pa = n_sites * PSEUDOABSENCE_RATIO
    sites_proj_valid = sites_valid.to_crs("EPSG:32749")
    site_buffer = unary_union(sites_proj_valid.buffer(2000))
    pool_rng = np.random.default_rng(RANDOM_SEED)
    candidates = []
    tries = 0
    pool_target = n_pa * 16
    max_tries = pool_target * 300

    while len(candidates) < pool_target and tries < max_tries:
        tries += 1
        px = pool_rng.uniform(ref_bounds.left, ref_bounds.right)
        py = pool_rng.uniform(ref_bounds.bottom, ref_bounds.top)
        pt = Point(px, py)
        if site_buffer.contains(pt):
            continue
        xy_pt = np.array([[px, py]])
        road_val = extract_at_points(xy_pt, road_arr, road_transform)[0]
        if not np.isfinite(road_val) or road_val > BASE_MAX_ROAD_DIST_M:
            continue
        p_accept = max(MIN_ACCEPT_PROB, min(1.0, float(np.exp(-road_val / BASE_DECAY_M))))
        if pool_rng.random() > p_accept:
            continue
        feats = {}
        valid = True
        for fname, (arr, transform, *_) in rasters.items():
            val = extract_at_points(xy_pt, arr, transform)[0]
            if not np.isfinite(val) or (fname == "elevation" and val <= 0):
                valid = False
                break
            feats[fname] = float(val)
        if not valid:
            continue
        feat_vec = np.array([feats[c] for c in FEAT_COLS], dtype=np.float64)
        zdist = float(np.sqrt(np.sum(((feat_vec - pres_mean) / pres_std) ** 2)))
        feats["x"] = px
        feats["y"] = py
        feats["zdist"] = zdist
        east = 1 if px > midx else 0
        north = 1 if py > midy else 0
        feats["region_id"] = east + (2 * north)
        candidates.append(feats)

    pool_df = pd.DataFrame(candidates)
    print(f"  Candidate pool: {len(pool_df)}")

    # Sample hybrid pseudo-absences (identical to E016)
    print("  Sampling pseudo-absences...")
    site_regions = (
        np.array([1 if x > midx else 0 for x in site_df["x"]])
        + np.array([2 if y > midy else 0 for y in site_df["y"]])
    )
    presence_region_prop = np.array(
        [(site_regions == r).sum() / len(site_regions) for r in range(4)]
    )

    best_rng = np.random.default_rng(CONFIG_SEED)
    base_pool = pool_df[pool_df["zdist"] <= HARD_Z_MAX].copy()
    hard_pool = base_pool[base_pool["zdist"] >= HARD_Z_MIN].copy()

    uniform = np.full(4, 0.25)
    weights = (1.0 - BEST_REGION_BLEND) * presence_region_prop + BEST_REGION_BLEND * uniform
    weights = weights / weights.sum()

    n_hard_target = int(round(n_pa * BEST_HARD_FRAC))
    n_core_target = n_pa - n_hard_target

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
        pick = best_rng.choice(cand, size=take, replace=False)
        selected.extend(pick.tolist())
        selected_set.update(pick.tolist())

    if len(selected) < n_hard_target:
        remain_cand = hard_pool.index[~hard_pool.index.isin(list(selected_set))].to_numpy()
        if len(remain_cand) > 0:
            take = min(n_hard_target - len(selected), len(remain_cand))
            pick = best_rng.choice(remain_cand, size=take, replace=False)
            selected.extend(pick.tolist())
            selected_set.update(pick.tolist())

    for rid in range(4):
        need = int(core_targets[rid])
        if need <= 0:
            continue
        cand = base_pool.index[
            (base_pool["region_id"] == rid)
            & (~base_pool.index.isin(list(selected_set)))
        ].to_numpy()
        if len(cand) == 0:
            continue
        take = min(need, len(cand))
        pick = best_rng.choice(cand, size=take, replace=False)
        selected.extend(pick.tolist())
        selected_set.update(pick.tolist())

    pa_df = pool_df.loc[selected].reset_index(drop=True)
    print(f"  Pseudo-absences: {len(pa_df)}")

    # ─── 5. Train XGBoost model ──────────────────────────────────────────
    print("\n[5/8] Training XGBoost model...")
    train_pos = site_df[FEAT_COLS + ["x", "y"]].copy()
    train_pos["presence"] = 1
    train_neg = pa_df[FEAT_COLS + ["x", "y"]].copy()
    train_neg["presence"] = 0
    train_all = pd.concat([train_pos, train_neg], ignore_index=True).dropna(subset=FEAT_COLS)

    X = train_all[FEAT_COLS].values
    y = train_all["presence"].values
    scale_pw = (y == 0).sum() / max((y == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pw, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0, random_state=RANDOM_SEED,
    )
    model.fit(X, y)
    importance = dict(zip(FEAT_COLS, [float(v) for v in model.feature_importances_]))
    print("  Model trained.")
    print("  Feature importances:")
    for f in sorted(importance, key=importance.get, reverse=True):
        print(f"    {f}: {importance[f]:.3f}")

    # ─── 6. Build prediction grid (~900m spacing) ────────────────────────
    print("\n[6/8] Building prediction grid...")
    ref_arr = rasters["elevation"][0]
    h, w = ref_arr.shape
    rows_idx = np.arange(0, h, GRID_STEP)
    cols_idx = np.arange(0, w, GRID_STEP)
    rr, cc = np.meshgrid(rows_idx, cols_idx, indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()
    xs, ys = rasterio.transform.xy(ref_transform, rr, cc)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    data = {"x": xs, "y": ys}
    for name, (arr, transform, *_) in rasters.items():
        data[name] = arr[rr, cc]
    grid_df = pd.DataFrame(data)
    mask = grid_df[FEAT_COLS].notna().all(axis=1) & (grid_df["elevation"] > 0)
    grid_df = grid_df[mask].reset_index(drop=True)
    print(f"  Grid points: {len(grid_df)}")

    # Predict suitability
    grid_probs = model.predict_proba(grid_df[FEAT_COLS].values)[:, 1]
    grid_df["suitability"] = grid_probs

    # Convert grid to WGS84
    grid_gdf = gpd.GeoDataFrame(
        grid_df,
        geometry=gpd.GeoSeries.from_xy(grid_df["x"], grid_df["y"], crs="EPSG:32749"),
    )
    grid_wgs = grid_gdf.to_crs("EPSG:4326")
    grid_df["lon"] = grid_wgs.geometry.x.values
    grid_df["lat"] = grid_wgs.geometry.y.values

    # ─── 7. Compute burial depth + calibrate ─────────────────────────────
    print("\n[7/8] Computing burial depth (Pyle 1989 + Dwarapala calibration)...")
    burial_raw = compute_burial_depth_grid(
        grid_df["lon"].values, grid_df["lat"].values, eruptions_since_1268
    )

    # Calibrate to Dwarapala ground truth
    dw_raw = compute_burial_depth_grid(
        np.array([DWARAPALA["lon"]]), np.array([DWARAPALA["lat"]]),
        eruptions_since_1268,
    )[0]
    if dw_raw > 0:
        loss_factor = DWARAPALA["actual_depth_cm"] / dw_raw
    else:
        loss_factor = 1.0
        print("  WARNING: Dwarapala raw prediction is 0, skipping calibration")
    burial_depth = burial_raw * loss_factor
    grid_df["burial_depth_cm"] = burial_depth
    print(f"  Loss factor: {loss_factor:.3f} ({loss_factor * 100:.1f}% retention)")
    print(f"  Dwarapala: raw={dw_raw:.1f} cm, calibrated={dw_raw * loss_factor:.1f} cm, "
          f"actual={DWARAPALA['actual_depth_cm']} cm")

    # Assign zones
    suit_threshold = float(np.percentile(grid_df["suitability"], 75))
    high_suit = grid_df["suitability"] >= suit_threshold
    grid_df["zone"] = "E"
    grid_df.loc[high_suit & (grid_df["burial_depth_cm"] < BURIAL_SHALLOW), "zone"] = "A"
    grid_df.loc[
        high_suit
        & (grid_df["burial_depth_cm"] >= BURIAL_SHALLOW)
        & (grid_df["burial_depth_cm"] < BURIAL_MODERATE),
        "zone",
    ] = "B"
    grid_df.loc[high_suit & (grid_df["burial_depth_cm"] >= BURIAL_MODERATE), "zone"] = "C"

    zone_counts = grid_df["zone"].value_counts().to_dict()
    print(f"  Zones: A={zone_counts.get('A', 0)}, B={zone_counts.get('B', 0)}, "
          f"C={zone_counts.get('C', 0)}, E={zone_counts.get('E', 0)}")
    print(f"  Suitability P75 threshold: {suit_threshold:.3f}")

    # ─── 8. Compute SHAP + save everything ───────────────────────────────
    print("\n[8/8] Computing SHAP values and saving outputs...")

    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        X_labeled = pd.DataFrame(X, columns=[FEAT_LABELS.get(c, c) for c in FEAT_COLS])
        shap_values = explainer(X_labeled)

        # Save SHAP data
        np.savez(
            OUT_DIR / "shap_values.npz",
            values=shap_values.values,
            data=shap_values.data,
            base_values=shap_values.base_values,
            feature_names=np.array([FEAT_LABELS[c] for c in FEAT_COLS]),
        )

        # Beeswarm plot
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.beeswarm(shap_values, show=False, max_display=len(FEAT_COLS))
        plt.title("SHAP Feature Importance (TreeSHAP)", fontsize=12)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
        plt.close()

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.bar(shap_values, show=False, max_display=len(FEAT_COLS))
        plt.title("Mean |SHAP| per Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_bar.png", dpi=200, bbox_inches="tight")
        plt.close()

        # SHAP summary CSV
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        shap_summary = pd.DataFrame({
            "feature": FEAT_COLS,
            "feature_label": [FEAT_LABELS[c] for c in FEAT_COLS],
            "mean_abs_shap": mean_abs_shap,
            "gain_importance": [importance[c] for c in FEAT_COLS],
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_summary.to_csv(OUT_DIR / "shap_summary.csv", index=False)
        print("  SHAP computed and saved.")
    else:
        print("  SHAP skipped (shap not installed).")

    # Save grid predictions
    grid_out = grid_df[["lon", "lat", "suitability", "burial_depth_cm", "zone"]].copy()
    grid_out.to_csv(OUT_DIR / "grid_predictions.csv", index=False)
    print(f"  grid_predictions.csv: {len(grid_out)} rows")

    # Save model
    model.save_model(str(OUT_DIR / "model_xgb.json"))
    print("  model_xgb.json saved")

    # Save sites with predictions
    sites_wgs = sites_valid.to_crs("EPSG:4326")
    site_probs = model.predict_proba(site_df[FEAT_COLS].values)[:, 1]
    site_lons = sites_wgs.geometry.x.values
    site_lats = sites_wgs.geometry.y.values
    site_burial_raw = compute_burial_depth_grid(
        site_lons, site_lats, eruptions_since_1268
    )
    site_burial = site_burial_raw * loss_factor
    site_zones = np.where(
        site_probs >= suit_threshold,
        np.where(site_burial < BURIAL_SHALLOW, "A",
                 np.where(site_burial < BURIAL_MODERATE, "B", "C")),
        "E",
    )

    site_names = sites_valid["name"].values if "name" in sites_valid.columns else [""] * len(sites_valid)
    sites_out = pd.DataFrame({
        "name": site_names,
        "lat": site_lats,
        "lon": site_lons,
        "suitability": site_probs,
        "burial_depth_cm": site_burial,
        "zone": site_zones,
    })
    sites_out.to_csv(OUT_DIR / "sites.csv", index=False)
    print(f"  sites.csv: {len(sites_out)} rows")

    # Save volcanoes
    volc_rows = [{"name": k, "lat": v["lat"], "lon": v["lon"]} for k, v in VOLCANOES.items()]
    pd.DataFrame(volc_rows).to_csv(OUT_DIR / "volcanoes.csv", index=False)
    print("  volcanoes.csv saved")

    # Save zone statistics
    zone_stats = grid_df.groupby("zone").agg(
        count=("zone", "size"),
        mean_suitability=("suitability", "mean"),
        mean_burial_cm=("burial_depth_cm", "mean"),
        max_burial_cm=("burial_depth_cm", "max"),
    ).round(2)
    zone_stats.to_csv(OUT_DIR / "zone_statistics.csv")
    print("  zone_statistics.csv saved")

    # Save metadata (reference metrics from E013/E014 experiment results)
    meta = {
        "xgb_auc": 0.768,
        "xgb_auc_std": 0.069,
        "xgb_tss": 0.507,
        "xgb_tss_std": 0.167,
        "tautology_rho": -0.229,
        "tautology_verdict": "TAUTOLOGY-FREE",
        "n_sites": n_sites,
        "n_grid_points": len(grid_df),
        "suitability_p75": suit_threshold,
        "loss_factor": float(loss_factor),
        "dwarapala_raw_cm": float(dw_raw),
        "dwarapala_predicted_cm": float(dw_raw * loss_factor),
        "dwarapala_actual_cm": DWARAPALA["actual_depth_cm"],
        "zone_counts": {k: int(v) for k, v in zone_counts.items()},
        "feature_importances": importance,
        "auc_progression": {
            "E007": 0.659,
            "E008": 0.695,
            "E009": 0.664,
            "E010": 0.711,
            "E011": 0.725,
            "E012": 0.730,
            "E013": 0.768,
        },
        "temporal_validation": {
            "xgb_auc": 0.755,
            "rf_auc": 0.654,
            "spatial_cv_xgb_auc": 0.785,
            "split_method": "accessibility_proxy",
            "verdict": "PASS",
        },
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("  metadata.json saved")

    print("\n" + "=" * 60)
    print("Dashboard data precomputed successfully!")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
