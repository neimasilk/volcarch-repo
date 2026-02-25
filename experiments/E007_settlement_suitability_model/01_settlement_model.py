"""
E007: Settlement Suitability Model — First test of H3.

Trains an XGBoost classifier on ENVIRONMENTAL FEATURES ONLY (no volcanic proximity)
to predict archaeological settlement suitability. Validates with spatial block CV.
Also runs Challenge 1 (tautology test): does the model predict high suitability in
high-burial zones where few sites are currently known?

Run from repo root:
    python experiments/E007_settlement_suitability_model/01_settlement_model.py
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
    from shapely.geometry import Point, box
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
DEM_DIR   = REPO_ROOT / "data" / "processed" / "dem"
SITES_PATH = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
RESULTS_DIR = Path(__file__).parent / "results"

# Volcanoes for Challenge 1 tautology test (distance computed AFTER model, not used as feature)
VOLCANOES = {
    "Kelud":           (-7.9300, 112.3080),
    "Semeru":          (-8.1080, 112.9220),
    "Arjuno-Welirang": (-7.7290, 112.5750),
    "Bromo":           (-7.9420, 112.9500),
    "Lamongan":        (-7.9770, 113.3430),
    "Raung":           (-8.1250, 114.0420),
    "Ijen":            (-8.0580, 114.2420),
}

# Spatial CV block size in degrees (~50 km)
BLOCK_SIZE_DEG = 0.45
N_FOLDS = 5
PSEUDOABSENCE_RATIO = 5   # 5 pseudo-absences per presence
RANDOM_SEED = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raster(path: Path):
    """Load raster, return (array, transform, crs, nodata)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, src.transform, src.crs, src.bounds


def extract_at_points(points_xy: np.ndarray, raster_arr: np.ndarray,
                      transform) -> np.ndarray:
    """
    Sample raster values at (x, y) coordinates.
    Returns array of values; NaN where outside raster or nodata.
    """
    rows, cols = rowcol(transform, points_xy[:, 0], points_xy[:, 1])
    rows = np.array(rows)
    cols = np.array(cols)
    h, w = raster_arr.shape

    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    values = np.full(len(points_xy), np.nan, dtype=np.float32)
    values[valid] = raster_arr[rows[valid], cols[valid]]
    return values


def load_sites() -> gpd.GeoDataFrame:
    """Load geocoded sites, filter to East Java bounds."""
    gdf = gpd.read_file(SITES_PATH)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs("EPSG:4326")
    jatim = (-9.0, 111.0, -6.5, 115.0)  # south, west, north, east
    gdf = gdf[(gdf.geometry.y >= jatim[0]) & (gdf.geometry.x >= jatim[1]) &
              (gdf.geometry.y <= jatim[2]) & (gdf.geometry.x <= jatim[3])]
    print(f"  Sites loaded: {len(gdf)}")
    return gdf


# ── Feature extraction ────────────────────────────────────────────────────────

def build_feature_grid(dem_arr, slope_arr, twi_arr, tri_arr, aspect_arr,
                       transform) -> gpd.GeoDataFrame:
    """
    Build a raster grid of feature values for map prediction.
    Samples every Nth pixel to keep memory reasonable.
    Returns GeoDataFrame with features + centroid geometry.
    """
    step = 10   # every 10 pixels ~300m at 30m resolution
    h, w = dem_arr.shape
    rows_idx = np.arange(0, h, step)
    cols_idx = np.arange(0, w, step)

    rr, cc = np.meshgrid(rows_idx, cols_idx, indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()

    # Pixel centroid coordinates
    xs, ys = rasterio.transform.xy(transform, rr, cc)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    elev  = dem_arr[rr, cc]
    slope = slope_arr[rr, cc]
    twi   = twi_arr[rr, cc]
    tri   = tri_arr[rr, cc]
    asp   = aspect_arr[rr, cc]

    mask = (np.isfinite(elev) & np.isfinite(slope) & np.isfinite(twi) &
            np.isfinite(tri) & np.isfinite(asp) & (elev > 0))

    df = pd.DataFrame({
        "x": xs[mask], "y": ys[mask],
        "elevation": elev[mask], "slope": slope[mask],
        "twi": twi[mask], "tri": tri[mask], "aspect": asp[mask],
    })
    geom = gpd.GeoSeries.from_xy(df["x"], df["y"], crs="EPSG:32749")
    gdf = gpd.GeoDataFrame(df, geometry=geom)
    return gdf


def extract_features_at_sites(sites: gpd.GeoDataFrame,
                               rasters: dict) -> pd.DataFrame:
    """Extract raster features at site locations."""
    # Project sites to UTM 49S to match rasters
    sites_proj = sites.to_crs("EPSG:32749")
    xy = np.column_stack([sites_proj.geometry.x, sites_proj.geometry.y])

    ref_transform = list(rasters.values())[0][1]
    rows_out = {}
    for name, (arr, transform, *_) in rasters.items():
        rows_out[name] = extract_at_points(xy, arr, transform)

    df = pd.DataFrame(rows_out)
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]
    return df


# ── Pseudo-absence generation ─────────────────────────────────────────────────

def generate_pseudo_absences(sites_proj: gpd.GeoDataFrame,
                              bounds, n: int,
                              rasters: dict,
                              rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate pseudo-absences:
    - Random points within raster extent
    - Exclude cells within 2km of any known site
    - Must have valid (non-NaN) feature values
    """
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    # Buffer all known sites by 2km
    site_buffer = unary_union(sites_proj.buffer(2000))

    minx, miny, maxx, maxy = bounds
    ref_arr, ref_transform, *_ = list(rasters.values())[0]

    candidates = []
    max_tries = n * 50
    tries = 0

    while len(candidates) < n and tries < max_tries:
        tries += 1
        # Random point in bounds
        px = rng.uniform(minx, maxx)
        py = rng.uniform(miny, maxy)
        pt = Point(px, py)

        if site_buffer.contains(pt):
            continue

        # Check valid features
        xy = np.array([[px, py]])
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
            candidates.append(feats)

    if len(candidates) < n:
        print(f"  WARNING: Only generated {len(candidates)}/{n} pseudo-absences")

    return pd.DataFrame(candidates)


# ── Spatial block cross-validation ───────────────────────────────────────────

def assign_spatial_blocks(x: np.ndarray, y: np.ndarray,
                           block_size: float) -> np.ndarray:
    """Assign each point to a spatial block ID."""
    # Convert UTM coords to block indices
    bx = (x / (block_size * 111000)).astype(int)   # ~111km per degree-equivalent in meters
    by = (y / (block_size * 111000)).astype(int)
    # Encode as single int
    return bx * 10000 + by


def spatial_cv_folds(blocks: np.ndarray, n_folds: int,
                     rng: np.random.Generator) -> list:
    """Split unique block IDs into n_folds groups, return list of (train_idx, test_idx)."""
    unique_blocks = np.unique(blocks)
    rng.shuffle(unique_blocks)
    block_splits = np.array_split(unique_blocks, n_folds)

    folds = []
    for i, test_blocks in enumerate(block_splits):
        test_mask  = np.isin(blocks, test_blocks)
        train_mask = ~test_mask
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return folds


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_tss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """True Skill Statistic at optimal threshold."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    tss_vals = tpr - fpr
    return float(np.max(tss_vals))


# ── Volcano distance for Challenge 1 ─────────────────────────────────────────

def min_volcano_distance_km(x_utm: np.ndarray, y_utm: np.ndarray) -> np.ndarray:
    """Compute min distance to nearest volcano (km) for UTM 49S coords."""
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)

    lons, lats = transformer.transform(x_utm, y_utm)

    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    min_dists = np.full(len(lons), np.inf)

    for vname, (vlat, vlon) in VOLCANOES.items():
        _, _, dists = geod.inv(
            np.full(len(lons), vlon), np.full(len(lons), vlat),
            lons, lats
        )
        min_dists = np.minimum(min_dists, dists / 1000)  # km

    return min_dists


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 60)
    print("E007: Settlement Suitability Model")
    print("=" * 60)

    # 1. Load rasters
    print("\nLoading DEM derivatives (full Jawa Timur)...")
    raster_files = {
        "elevation": DEM_DIR / "jatim_dem.tif",
        "slope":     DEM_DIR / "jatim_slope.tif",
        "twi":       DEM_DIR / "jatim_twi.tif",
        "tri":       DEM_DIR / "jatim_tri.tif",
        "aspect":    DEM_DIR / "jatim_aspect.tif",
    }
    for name, path in raster_files.items():
        if not path.exists():
            print(f"ERROR: Missing {path.name} — run E003 first")
            sys.exit(1)

    rasters = {}
    ref_bounds = None
    for name, path in raster_files.items():
        arr, transform, crs, bounds = load_raster(path)
        rasters[name] = (arr, transform, crs, bounds)
        if ref_bounds is None:
            ref_bounds = bounds
        print(f"  {name}: {arr.shape}, range [{np.nanmin(arr):.1f}, {np.nanmax(arr):.1f}]")

    # 2. Load sites
    print("\nLoading archaeological sites...")
    sites = load_sites()
    sites_proj = sites.to_crs("EPSG:32749")

    # 3. Extract features at site locations
    print("\nExtracting features at site locations...")
    rasters_simple = {k: (v[0], v[1]) for k, v in rasters.items()}
    site_feats = extract_features_at_sites(sites, rasters_simple)

    # Drop rows with NaN features
    feat_cols = ["elevation", "slope", "twi", "tri", "aspect"]
    site_feats = site_feats.dropna(subset=feat_cols)
    site_feats = site_feats[site_feats["elevation"] > 0]
    print(f"  Sites with valid features: {len(site_feats)}")

    # 4. Generate pseudo-absences
    n_pa = len(site_feats) * PSEUDOABSENCE_RATIO
    print(f"\nGenerating {n_pa} pseudo-absences ({PSEUDOABSENCE_RATIO}x sites)...")
    bounds_utm = (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
    pa_feats = generate_pseudo_absences(sites_proj, bounds_utm, n_pa, rasters_simple, rng)
    print(f"  Generated: {len(pa_feats)}")

    # 5. Combine into training dataset
    print("\nBuilding training dataset...")
    site_feats["presence"] = 1
    pa_feats["presence"]   = 0

    df = pd.concat([site_feats[feat_cols + ["x", "y", "presence"]],
                    pa_feats[feat_cols  + ["x", "y", "presence"]]], ignore_index=True)
    df = df.dropna(subset=feat_cols)

    X = df[feat_cols].values
    y = df["presence"].values
    coords_x = df["x"].values
    coords_y = df["y"].values

    print(f"  Total samples: {len(df)} ({y.sum()} presences, {(1-y).sum()} absences)")

    # 6. Spatial block CV
    print(f"\nSpatial block cross-validation ({N_FOLDS} folds, ~{BLOCK_SIZE_DEG*111:.0f}km blocks)...")
    blocks = assign_spatial_blocks(coords_x, coords_y, BLOCK_SIZE_DEG)
    folds  = spatial_cv_folds(blocks, N_FOLDS, rng)

    xgb_aucs, xgb_tsss = [], []
    rf_aucs,  rf_tsss  = [], []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        if len(test_idx) == 0 or y[test_idx].sum() == 0:
            print(f"  Fold {fold_i+1}: no positives in test set, skipping")
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # XGBoost
        scale_pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pw, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0, random_state=RANDOM_SEED
        )
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc  = roc_auc_score(y_test, xgb_prob)
        xgb_tss  = compute_tss(y_test, xgb_prob)
        xgb_aucs.append(xgb_auc)
        xgb_tsss.append(xgb_tss)

        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        rf_auc  = roc_auc_score(y_test, rf_prob)
        rf_tss  = compute_tss(y_test, rf_prob)
        rf_aucs.append(rf_auc)
        rf_tsss.append(rf_tss)

        print(f"  Fold {fold_i+1}: n_train={len(train_idx)}, n_test={len(test_idx)} "
              f"| XGB AUC={xgb_auc:.3f} TSS={xgb_tss:.3f} "
              f"| RF  AUC={rf_auc:.3f} TSS={rf_tss:.3f}")

    # 7. Summary metrics
    print()
    xgb_mean_auc = np.mean(xgb_aucs)
    xgb_mean_tss = np.mean(xgb_tsss)
    rf_mean_auc  = np.mean(rf_aucs)
    rf_mean_tss  = np.mean(rf_tsss)

    print(f"XGBoost — Spatial AUC: {xgb_mean_auc:.3f} ± {np.std(xgb_aucs):.3f}  |  TSS: {xgb_mean_tss:.3f} ± {np.std(xgb_tsss):.3f}")
    print(f"Rand.F. — Spatial AUC: {rf_mean_auc:.3f} ± {np.std(rf_aucs):.3f}  |  TSS: {rf_mean_tss:.3f} ± {np.std(rf_tsss):.3f}")

    # EVAL.md thresholds
    best_auc = max(xgb_mean_auc, rf_mean_auc)
    print()
    if best_auc >= 0.85:
        verdict = "EXCELLENT (>0.85)"
    elif best_auc >= 0.75:
        verdict = "GOOD — MVR MET (>0.75)"
    elif best_auc >= 0.65:
        verdict = "BELOW MVR — tune features"
    else:
        verdict = "KILL SIGNAL (<0.65) — H3 may be falsified"
    print(f"Verdict: {verdict}")

    # 8. Train final model on all data for map generation
    print("\nTraining final model on all data for probability map...")
    best_model_name = "XGBoost" if xgb_mean_auc >= rf_mean_auc else "RandomForest"
    scale_pw_all = (y == 0).sum() / max((y == 1).sum(), 1)

    final_xgb = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pw_all, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0, random_state=RANDOM_SEED
    )
    final_xgb.fit(X, y)

    # Feature importance
    importance = dict(zip(feat_cols, final_xgb.feature_importances_))
    print("  Feature importances (XGBoost):")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"    {feat:12s}: {imp:.3f}")

    # 9. Generate probability map (sampled grid)
    print("\nGenerating suitability probability map...")
    ref_arr, ref_transform = rasters["elevation"][0], rasters["elevation"][1]
    grid_gdf = build_feature_grid(
        rasters["elevation"][0], rasters["slope"][0],
        rasters["twi"][0],      rasters["tri"][0],
        rasters["aspect"][0],   ref_transform
    )

    grid_X = grid_gdf[feat_cols].values
    grid_prob = final_xgb.predict_proba(grid_X)[:, 1]
    grid_gdf["suitability"] = grid_prob

    # 10. Challenge 1 — Tautology Test
    print("\nChallenge 1: Tautology Test...")
    grid_wgs = grid_gdf.to_crs("EPSG:4326")
    grid_gdf["lon"] = grid_wgs.geometry.x
    grid_gdf["lat"] = grid_wgs.geometry.y

    grid_gdf["volcano_dist_km"] = min_volcano_distance_km(
        grid_gdf["x"].values, grid_gdf["y"].values
    )

    # Correlation: suitability vs volcano distance
    rho_taut, p_taut = spearmanr(grid_gdf["volcano_dist_km"], grid_gdf["suitability"])

    # High suitability (top quartile) in high-burial zones (0-50km from volcano)
    high_suit = grid_gdf[grid_gdf["suitability"] >= grid_gdf["suitability"].quantile(0.75)]
    near_volc_high = (high_suit["volcano_dist_km"] <= 50).mean()

    print(f"  Spearman rho (suitability vs volcano dist): {rho_taut:.3f}, p={p_taut:.4f}")
    print(f"  High-suitability cells within 50km of volcano: {near_volc_high*100:.1f}%")

    if rho_taut > 0.3:
        ch1 = "TAUTOLOGY RISK: Model strongly prefers areas far from volcanoes (where sites are known)"
    elif rho_taut > 0:
        ch1 = "MILD TAUTOLOGY: Slight preference for areas away from volcanoes"
    else:
        ch1 = "TAUTOLOGY-FREE: Model predicts suitability independently of volcanic proximity"
    print(f"  Challenge 1: {ch1}")

    # 11. Save results
    print("\nSaving results...")
    results_txt = f"""E007 — Settlement Suitability Model Results
============================================
Date: 2026-02-24
Features: {feat_cols}
Positive samples (sites): {int(y.sum())}
Pseudo-absences: {int((1-y).sum())}
Spatial CV: {N_FOLDS} folds, ~{BLOCK_SIZE_DEG*111:.0f}km blocks

XGBoost  Spatial AUC: {xgb_mean_auc:.3f} ± {np.std(xgb_aucs):.3f}
XGBoost  TSS:         {xgb_mean_tss:.3f} ± {np.std(xgb_tsss):.3f}
Rand.For Spatial AUC: {rf_mean_auc:.3f} ± {np.std(rf_aucs):.3f}
Rand.For TSS:         {rf_mean_tss:.3f} ± {np.std(rf_tsss):.3f}

MVR (AUC > 0.75): {'MET' if best_auc >= 0.75 else 'NOT MET'}
Verdict: {verdict}

Feature Importances (XGBoost):
{chr(10).join(f'  {k}: {v:.3f}' for k,v in sorted(importance.items(), key=lambda x:-x[1]))}

Challenge 1 (Tautology Test):
  Spearman rho (suitability vs volcano dist): {rho_taut:.3f} (p={p_taut:.4f})
  High-suitability cells within 50km of volcano: {near_volc_high*100:.1f}%
  Result: {ch1}
"""
    (RESULTS_DIR / "model_results.txt").write_text(results_txt, encoding="utf-8")
    print(f"  Results: {RESULTS_DIR / 'model_results.txt'}")

    # 12. Probability map (Folium)
    print("  Building interactive map...")
    map_center_lat = float(grid_gdf["lat"].mean())
    map_center_lon = float(grid_gdf["lon"].mean())
    m = folium.Map(location=[map_center_lat, map_center_lon],
                   zoom_start=8, tiles="CartoDB positron")

    # Sample for map (every 5th point to keep filesize manageable)
    map_sample = grid_gdf.iloc[::5].copy()
    colormap = plt.cm.YlOrRd

    for _, row in map_sample.iterrows():
        rgba = colormap(row["suitability"])
        color = mcolors.to_hex(rgba)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3, color=color, fill=True,
            fill_color=color, fill_opacity=0.6, weight=0,
            tooltip=f"Suit: {row['suitability']:.2f} | {row['volcano_dist_km']:.0f}km from volcano"
        ).add_to(m)

    # Add known sites
    sites_wgs = sites.to_crs("EPSG:4326")
    for _, row in sites_wgs.iterrows():
        if row.geometry is None:
            continue
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4, color="blue", fill=True,
            fill_color="blue", fill_opacity=0.9, weight=1,
            tooltip=f"Site: {row.get('name','?')}"
        ).add_to(m)

    m.save(str(RESULTS_DIR / "suitability_map.html"))
    print(f"  Map: {RESULTS_DIR / 'suitability_map.html'}")

    # 13. Bar chart — AUC by fold
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"E007: Settlement Suitability Model — Spatial CV Results\n"
                 f"XGB AUC={xgb_mean_auc:.3f}  RF AUC={rf_mean_auc:.3f}  |  {verdict}",
                 fontsize=11)

    folds_labels = [f"Fold {i+1}" for i in range(len(xgb_aucs))]
    x = np.arange(len(xgb_aucs))
    w = 0.35
    axes[0].bar(x - w/2, xgb_aucs, width=w, label="XGBoost", color="#E53935", alpha=0.8)
    axes[0].bar(x + w/2, rf_aucs,  width=w, label="Random Forest", color="#1E88E5", alpha=0.8)
    axes[0].axhline(0.75, color="green", linestyle="--", label="MVR (0.75)")
    axes[0].axhline(0.65, color="red",   linestyle=":", label="Kill signal (0.65)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(folds_labels)
    axes[0].set_ylabel("Spatial AUC-ROC"); axes[0].set_ylim(0, 1)
    axes[0].set_title("AUC by Fold"); axes[0].legend(fontsize=8)

    feats_sorted = sorted(importance.items(), key=lambda x: x[1])
    axes[1].barh([f for f, _ in feats_sorted], [v for _, v in feats_sorted],
                 color="#43A047", alpha=0.85)
    axes[1].set_xlabel("Feature Importance"); axes[1].set_title("XGBoost Feature Importance")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_cv_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart: {RESULTS_DIR / 'model_cv_results.png'}")

    print()
    print("=" * 60)
    print("E007 complete.")
    print(f"Best spatial AUC: {best_auc:.3f} — {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
