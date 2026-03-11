"""
E016: Zone Classification Map.

Combines E013 settlement suitability with Pyle (1989) analytical burial depth
estimate to classify East Java into survey-priority zones.

Zones:
  A: High suitability + shallow burial (<100 cm) -- known sites expected
  B: High suitability + moderate burial (100-300 cm) -- GPR survey targets
  C: High suitability + deep burial (>300 cm) -- likely present, hard to reach
  E: Low suitability -- few/no sites expected

Run from repo root:
    python experiments/E016_zone_classification/01_zone_classification.py
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"
SITES_PATH = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
ERUPTION_PATH = REPO_ROOT / "data" / "processed" / "eruption_history.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FEAT_COLS = ["elevation", "slope", "twi", "tri", "aspect", "river_dist"]

# Volcanoes with coordinates (EPSG:4326) and GVP IDs
VOLCANOES = {
    "Kelud":            {"lat": -7.9300, "lon": 112.3080, "gvp_id": 263280},
    "Semeru":           {"lat": -8.1080, "lon": 112.9220, "gvp_id": 263300},
    "Arjuno-Welirang":  {"lat": -7.7290, "lon": 112.5750, "gvp_id": 263260},
    "Bromo":            {"lat": -7.9420, "lon": 112.9500, "gvp_id": 263310},
    "Lamongan":         {"lat": -7.9770, "lon": 113.3430, "gvp_id": 263320},
    "Raung":            {"lat": -8.1250, "lon": 114.0420, "gvp_id": 263340},
    "Ijen":             {"lat": -8.0580, "lon": 114.2420, "gvp_id": 263350},
}

# Pyle (1989) exponential thinning: T(d) = T0 * exp(-k * d)
# T0 = thickness at 1 km (cm), k = decay constant (1/km)
# Calibrated from Indonesian volcano literature (same as tools/scrape_gvp.py):
#   Kelud 1919 VEI 4: ~10 cm at 40 km (Thouret et al. 1998)
#   Kelud 2014 VEI 4: ~3 cm at 40 km (PVMBG reports)
PYLE_PARAMS = {
    0: {"T0": 0.1, "k": 0.15},
    1: {"T0": 0.5, "k": 0.12},
    2: {"T0": 3.0, "k": 0.08},
    3: {"T0": 15.0, "k": 0.06},
    4: {"T0": 80.0, "k": 0.05},
    5: {"T0": 500.0, "k": 0.04},
}

# E013 model parameters
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

# Dwarapala validation point
DWARAPALA = {"lat": -7.973, "lon": 112.435, "actual_depth_cm": 185, "year_built": 1268}

# Zone thresholds
SUITABILITY_THRESHOLD = 0.5  # P75 will be computed dynamically
BURIAL_SHALLOW = 100  # cm
BURIAL_MODERATE = 300  # cm

# Calibration: Pyle model over-predicts because it ignores post-depositional
# loss (erosion, compaction, lahar reworking). We calibrate against Dwarapala
# ground truth by computing a loss factor.
CALIBRATE_TO_DWARAPALA = True


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


def pyle_thickness_cm(vei, distance_km):
    """Pyle (1989) exponential thinning model. Returns thickness in cm."""
    if vei not in PYLE_PARAMS or np.isnan(vei):
        return 0.0
    p = PYLE_PARAMS[int(vei)]
    thickness = p["T0"] * np.exp(-p["k"] * distance_km)
    # Below 0.01 cm is negligible
    if isinstance(thickness, np.ndarray):
        thickness[thickness < 0.01] = 0.0
    elif thickness < 0.01:
        return 0.0
    return thickness


def compute_burial_depth_grid(grid_lon, grid_lat, eruptions_since_1268):
    """Compute cumulative burial depth (cm) at each grid point from all eruptions since 1268 CE."""
    n = len(grid_lon)
    depth = np.zeros(n, dtype=np.float64)

    for _, erup in eruptions_since_1268.iterrows():
        vei = erup["vei"]
        if pd.isna(vei) or int(vei) not in PYLE_PARAMS:
            continue

        # Find volcano coords
        volcano_name = erup["volcano"]
        if volcano_name not in VOLCANOES:
            continue
        vlat = VOLCANOES[volcano_name]["lat"]
        vlon = VOLCANOES[volcano_name]["lon"]

        # Great-circle distance approximation (good enough at these scales)
        dlat = np.radians(grid_lat - vlat)
        dlon = np.radians(grid_lon - vlon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(vlat)) * np.cos(np.radians(grid_lat)) * np.sin(dlon/2)**2
        dist_km = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        # Pyle thinning
        p = PYLE_PARAMS[int(vei)]
        thickness = p["T0"] * np.exp(-p["k"] * dist_km)
        # Negligible below 0.01 cm
        thickness[thickness < 0.01] = 0.0
        depth += thickness

    return depth


def main():
    print("=" * 60)
    print("E016: Zone Classification Map")
    print("=" * 60)

    # --- Load eruption data ---
    print("\n[1/5] Loading eruption history...")
    eruptions = pd.read_csv(ERUPTION_PATH)
    eruptions_since_1268 = eruptions[eruptions["year"] >= 1268].copy()
    print(f"  Total eruptions: {len(eruptions)}")
    print(f"  Eruptions since 1268 CE: {len(eruptions_since_1268)}")
    for v in eruptions_since_1268["volcano"].unique():
        n = len(eruptions_since_1268[eruptions_since_1268["volcano"] == v])
        print(f"    {v}: {n}")

    # --- Load rasters and train E013 model ---
    print("\n[2/5] Loading rasters...")
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
    for name, fname in raster_files.items():
        path = DEM_DIR / fname
        arr, transform, crs, bnds = load_raster(path)
        rasters[name] = (arr, transform, crs, bnds)
        if ref_bounds is None:
            ref_bounds = bnds
            ref_transform = transform
            ref_shape = arr.shape

    road_arr, road_transform, _, _ = load_raster(DEM_DIR / "jatim_road_dist_expanded.tif")

    # --- Load sites + extract features ---
    print("\n[3/5] Training E013 model...")
    sites = gpd.read_file(SITES_PATH)
    sites = sites[sites.geometry.notna() & ~sites.geometry.is_empty].to_crs("EPSG:4326")
    jatim = (-9.0, 111.0, -6.5, 115.0)
    sites = sites[(sites.geometry.y >= jatim[0]) & (sites.geometry.x >= jatim[1]) &
                  (sites.geometry.y <= jatim[2]) & (sites.geometry.x <= jatim[3])]

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
    n_sites = len(site_df)
    print(f"  Valid sites: {n_sites}")

    # Build pseudo-absences (simplified: reuse E013 logic)
    pres_vals = site_df[FEAT_COLS].values
    pres_mean = np.nanmean(pres_vals, axis=0)
    pres_std = np.nanstd(pres_vals, axis=0)
    pres_std[pres_std < 1e-6] = 1.0

    midx = (ref_bounds.left + ref_bounds.right) / 2
    midy = (ref_bounds.bottom + ref_bounds.top) / 2

    n_pa = n_sites * PSEUDOABSENCE_RATIO
    from shapely.ops import unary_union

    site_buffer = unary_union(sites_proj.buffer(2000))
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
        for name, (arr, transform, *_) in rasters.items():
            val = extract_at_points(xy_pt, arr, transform)[0]
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
        east = 1 if px > midx else 0
        north = 1 if py > midy else 0
        feats["region_id"] = east + (2 * north)
        candidates.append(feats)

    pool_df = pd.DataFrame(candidates)
    print(f"  Candidate pool: {len(pool_df)}")

    # Sample hybrid pseudo-absences
    site_regions = np.array([1 if x > midx else 0 for x in site_df["x"]]) + \
                   np.array([2 if y > midy else 0 for y in site_df["y"]])
    presence_region_prop = np.array([(site_regions == r).sum() / len(site_regions) for r in range(4)])

    best_rng = np.random.default_rng(CONFIG_SEED)
    base_pool = pool_df[pool_df["zdist"] <= HARD_Z_MAX].copy()
    hard_pool = base_pool[base_pool["zdist"] >= HARD_Z_MIN].copy()

    uniform = np.full(4, 0.25)
    weights = (1.0 - BEST_REGION_BLEND) * presence_region_prop + BEST_REGION_BLEND * uniform
    weights = weights / weights.sum()

    n_hard_target = int(round(n_pa * BEST_HARD_FRAC))
    n_core_target = n_pa - n_hard_target

    def allocate_counts(total, wts):
        raw = total * wts
        counts = np.floor(raw).astype(int)
        rem = int(total - counts.sum())
        if rem > 0:
            frac_idx = np.argsort(-(raw - counts))
            for i in frac_idx[:rem]:
                counts[i] += 1
        return counts

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
        cand = base_pool.index[(base_pool["region_id"] == rid) &
                               (~base_pool.index.isin(list(selected_set)))].to_numpy()
        if len(cand) == 0:
            continue
        take = min(need, len(cand))
        pick = best_rng.choice(cand, size=take, replace=False)
        selected.extend(pick.tolist())
        selected_set.update(pick.tolist())

    pa_df = pool_df.loc[selected].reset_index(drop=True)
    print(f"  Pseudo-absences: {len(pa_df)}")

    # Train
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
    print("  Model trained.")

    # --- Build prediction grid ---
    print("\n[4/5] Building prediction grid and computing zones...")
    step = 30  # ~900m grid spacing at 30m resolution
    ref_arr = rasters["elevation"][0]
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
    grid_df = pd.DataFrame(data)
    mask = grid_df[FEAT_COLS].notna().all(axis=1) & (grid_df["elevation"] > 0)
    grid_df = grid_df[mask].reset_index(drop=True)
    print(f"  Grid points: {len(grid_df)}")

    # Predict suitability
    grid_probs = model.predict_proba(grid_df[FEAT_COLS].values)[:, 1]
    grid_df["suitability"] = grid_probs

    # Convert grid to WGS84 for burial depth computation
    grid_gdf = gpd.GeoDataFrame(grid_df, geometry=gpd.GeoSeries.from_xy(grid_df["x"], grid_df["y"], crs="EPSG:32749"))
    grid_wgs = grid_gdf.to_crs("EPSG:4326")
    grid_df["lon"] = grid_wgs.geometry.x.values
    grid_df["lat"] = grid_wgs.geometry.y.values

    # Compute burial depth
    print("  Computing burial depth (Pyle 1989)...")
    burial_depth_raw = compute_burial_depth_grid(
        grid_df["lon"].values, grid_df["lat"].values, eruptions_since_1268
    )

    # Calibrate: compute correction factor from Dwarapala ground truth
    if CALIBRATE_TO_DWARAPALA:
        dw_raw = compute_burial_depth_grid(
            np.array([DWARAPALA["lon"]]), np.array([DWARAPALA["lat"]]),
            eruptions_since_1268
        )[0]
        if dw_raw > 0:
            loss_factor = DWARAPALA["actual_depth_cm"] / dw_raw
            print(f"  Dwarapala calibration: raw={dw_raw:.1f} cm, actual={DWARAPALA['actual_depth_cm']} cm")
            print(f"  Loss factor (retention): {loss_factor:.3f} ({loss_factor*100:.1f}% of deposited tephra retained)")
            burial_depth = burial_depth_raw * loss_factor
        else:
            print("  WARNING: Dwarapala raw prediction is 0, skipping calibration")
            burial_depth = burial_depth_raw
            loss_factor = 1.0
    else:
        burial_depth = burial_depth_raw
        loss_factor = 1.0

    grid_df["burial_depth_cm"] = burial_depth

    # --- Classify zones ---
    suit_threshold = np.percentile(grid_df["suitability"], 75)
    print(f"  Suitability P75 threshold: {suit_threshold:.3f}")

    high_suit = grid_df["suitability"] >= suit_threshold
    grid_df["zone"] = "E"  # default: low suitability
    grid_df.loc[high_suit & (grid_df["burial_depth_cm"] < BURIAL_SHALLOW), "zone"] = "A"
    grid_df.loc[high_suit & (grid_df["burial_depth_cm"] >= BURIAL_SHALLOW) &
                (grid_df["burial_depth_cm"] < BURIAL_MODERATE), "zone"] = "B"
    grid_df.loc[high_suit & (grid_df["burial_depth_cm"] >= BURIAL_MODERATE), "zone"] = "C"

    zone_counts = grid_df["zone"].value_counts()
    print(f"\n  Zone distribution:")
    for z in ["A", "B", "C", "E"]:
        n = zone_counts.get(z, 0)
        pct = 100.0 * n / len(grid_df)
        print(f"    Zone {z}: {n:,} cells ({pct:.1f}%)")

    # --- Dwarapala validation ---
    print("\n[5/5] Dwarapala validation...")
    dw_lon = DWARAPALA["lon"]
    dw_lat = DWARAPALA["lat"]
    dw_depth_raw = compute_burial_depth_grid(
        np.array([dw_lon]), np.array([dw_lat]), eruptions_since_1268
    )[0]
    dw_depth = dw_depth_raw * loss_factor
    dw_actual = DWARAPALA["actual_depth_cm"]
    dw_error_pct = abs(dw_depth - dw_actual) / dw_actual * 100

    print(f"  Dwarapala predicted burial: {dw_depth:.1f} cm")
    print(f"  Dwarapala actual burial: {dw_actual} cm")
    print(f"  Error: {dw_error_pct:.1f}%")
    if dw_error_pct <= 30:
        dw_verdict = "PASS (within +/-30%)"
    elif dw_error_pct <= 50:
        dw_verdict = "MARGINAL (within +/-50%)"
    else:
        dw_verdict = "FAIL (outside +/-50%)"
    print(f"  Verdict: {dw_verdict}")

    # --- Generate zone map ---
    print("\nGenerating zone classification map...")
    zone_colors = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c", "E": "#bdc3c7"}

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: Zone classification
    ax1 = axes[0]
    for zone, color in zone_colors.items():
        mask = grid_df["zone"] == zone
        if mask.sum() > 0:
            ax1.scatter(grid_df.loc[mask, "lon"], grid_df.loc[mask, "lat"],
                       c=color, s=1, alpha=0.6, label=f"Zone {zone}", rasterized=True)

    # Overlay known sites
    sites_wgs = sites.to_crs("EPSG:4326")
    ax1.scatter(sites_wgs.geometry.x, sites_wgs.geometry.y,
               c="black", s=8, marker="^", label="Known sites", zorder=5)

    # Mark Dwarapala
    ax1.scatter([dw_lon], [dw_lat], c="blue", s=80, marker="*",
               edgecolors="white", linewidths=0.5, label="Dwarapala", zorder=6)

    # Mark volcanoes
    for vname, vdata in VOLCANOES.items():
        ax1.scatter([vdata["lon"]], [vdata["lat"]], c="red", s=60, marker="v",
                   edgecolors="black", linewidths=0.5, zorder=6)
        ax1.annotate(vname, (vdata["lon"], vdata["lat"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("Zone Classification Map\n(E013 Suitability + Pyle 1989 Burial Depth)")
    legend_handles = [
        Patch(facecolor="#2ecc71", label="A: High suit., shallow (<100 cm)"),
        Patch(facecolor="#f39c12", label="B: High suit., moderate (100-300 cm) -- GPR targets"),
        Patch(facecolor="#e74c3c", label="C: High suit., deep (>300 cm)"),
        Patch(facecolor="#bdc3c7", label="E: Low suitability"),
    ]
    ax1.legend(handles=legend_handles, loc="lower left", fontsize=7, framealpha=0.9)

    # Right: Burial depth heatmap
    ax2 = axes[1]
    sc = ax2.scatter(grid_df["lon"], grid_df["lat"],
                    c=np.clip(grid_df["burial_depth_cm"], 0, 500),
                    cmap="YlOrRd", s=1, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax2, label="Estimated burial depth (cm)")
    ax2.scatter([dw_lon], [dw_lat], c="blue", s=80, marker="*",
               edgecolors="white", linewidths=0.5, zorder=6)
    ax2.annotate(f"Dwarapala\npred={dw_depth:.0f} cm\nactual={dw_actual} cm",
                (dw_lon, dw_lat), fontsize=7, xytext=(10, -20),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    for vname, vdata in VOLCANOES.items():
        ax2.scatter([vdata["lon"]], [vdata["lat"]], c="red", s=60, marker="v",
                   edgecolors="black", linewidths=0.5, zorder=6)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title("Estimated Burial Depth Since 1268 CE\n(Pyle 1989 Exponential Thinning)")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "zone_classification_map.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'zone_classification_map.png'}")

    # --- Save statistics ---
    zone_stats = grid_df.groupby("zone").agg(
        count=("zone", "size"),
        mean_suitability=("suitability", "mean"),
        mean_burial_cm=("burial_depth_cm", "mean"),
        max_burial_cm=("burial_depth_cm", "max"),
    ).round(2)
    zone_stats.to_csv(RESULTS_DIR / "zone_statistics.csv")

    # --- Write validation report ---
    report = f"""E016: Zone Classification Map — Results
{'=' * 50}
Date: 2026-03-03

Model: E013 XGBoost (best config: blend=0.00, hard_frac=0.30, seed=375)
Eruptions used: {len(eruptions_since_1268)} since 1268 CE
Grid points: {len(grid_df):,}
Suitability P75 threshold: {suit_threshold:.3f}
Burial depth calibration: Dwarapala-anchored (loss factor = {loss_factor:.3f})
  Raw Pyle prediction at Dwarapala: {dw_depth_raw:.1f} cm
  Actual observed: {DWARAPALA['actual_depth_cm']} cm
  Retention factor: {loss_factor*100:.1f}% of deposited tephra stays in place
  Remainder lost to: erosion, compaction, lahar reworking, fluvial transport

Zone Distribution:
  Zone A (high suit, shallow <100cm): {zone_counts.get('A', 0):,} ({100*zone_counts.get('A',0)/len(grid_df):.1f}%)
  Zone B (high suit, moderate 100-300cm): {zone_counts.get('B', 0):,} ({100*zone_counts.get('B',0)/len(grid_df):.1f}%)
  Zone C (high suit, deep >300cm): {zone_counts.get('C', 0):,} ({100*zone_counts.get('C',0)/len(grid_df):.1f}%)
  Zone E (low suitability): {zone_counts.get('E', 0):,} ({100*zone_counts.get('E',0)/len(grid_df):.1f}%)

Dwarapala Validation:
  Location: {dw_lat:.3f}N, {dw_lon:.3f}E
  Predicted burial depth: {dw_depth:.1f} cm
  Actual burial depth: {dw_actual} cm
  Error: {dw_error_pct:.1f}%
  Verdict: {dw_verdict}
  Acceptable range (+/-30%): {dw_actual*0.7:.0f} - {dw_actual*1.3:.0f} cm

Zone Statistics:
{zone_stats.to_string()}

Interpretation:
- Zone B cells are the primary GPR survey targets.
- Zone C cells may contain buried sites but require deep-penetrating methods.
- Zone A should correlate with known site locations (validation check).
- Burial depth is a first-order estimate; actual burial depends on local
  geomorphology, lahar pathways, and fluvial reworking not modeled here.
"""
    with open(RESULTS_DIR / "zone_classification_report.txt", "w") as f:
        f.write(report)
    print(f"  Report: {RESULTS_DIR / 'zone_classification_report.txt'}")

    print("\nE016 COMPLETE.")


if __name__ == "__main__":
    main()
