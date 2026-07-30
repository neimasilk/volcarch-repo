"""
E217: MaxEnt benchmark across the pseudo-absence ladder (JCAA R1 revision, Reviewer 1).

Reviewer 1 (JCAA #280): "as the main topic is around the presence-only modeling, the reason for
not using, comparing to, or at least relating to the similar approach through the Maximum Entropy
functions is problematic... why not use Maxent to evaluate the results?"

This script answers that head-on. It runs a factorial benchmark under a SINGLE set of
deterministic spatial-block CV folds:

    3 background designs  x  2 feature sets  x  3 algorithms  x  N seeds

Background designs replicate the paper's own ladder:
  - random : uniform background (the E007/E008 design)
  - tgb    : target-group background, road-decay acceptance (the E010-E012 design)
  - hybrid : TGB + regional quota + hard-negative fraction (the E013 design)

The scientific question is NOT "which algorithm wins". It is whether the paper's central claim --
that pseudo-absence realism dominates feature accumulation -- is an artefact of using boosted trees.
If the same monotonic gain appears in MaxEnt, the claim becomes algorithm-independent.

Pre-registered before running (see README.md): if MaxEnt matches or beats XGBoost at the hybrid
design, that is reported plainly and the algorithm choice is reframed as an interpretability
decision, not a performance claim.

DELIBERATE DEVIATION FROM E013, DOCUMENTED: backgrounds here are drawn from a 10x-decimated
raster lattice (~300 m spacing) rather than by continuous-coordinate rejection sampling. This makes
all designs share one sampling frame and keeps the benchmark tractable. Absolute AUCs may therefore
differ slightly from the published E007-E013 values; what this experiment measures is the WITHIN-E217
contrast across backgrounds and algorithms, which is the quantity Reviewer 1 asked about.

Run from repo root:
    py experiments/E217_maxent_benchmark/01_maxent_benchmark.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import geopandas as gpd
    import rasterio
    from rasterio.enums import Resampling
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, roc_curve
    import xgboost as xgb
    from elapid import MaxentModel
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install elapid xgboost scikit-learn geopandas rasterio")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"
SITES_PATH = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
RESULTS_DIR = Path(__file__).parent / "results"

# --- Parameters held identical to E013 so the comparison is fair -----------------
BLOCK_SIZE_DEG = 0.45
N_FOLDS = 5
PSEUDOABSENCE_RATIO = 5
RANDOM_SEED = 42
DECIMATE = 10                 # sampling-frame decimation (~300 m lattice)
SITE_BUFFER_M = 2000.0        # exclusion radius around presences (as in E013 TGB pool)

BASE_DECAY_M = 12000.0        # E012/E013 TGB acceptance decay
BASE_MAX_ROAD_DIST_M = 20000.0
MIN_ACCEPT_PROB = 0.03
HARD_Z_MIN, HARD_Z_MAX = 2.0, 5.0
HYBRID_HARD_FRAC = 0.30       # E013 best configuration
HYBRID_REGION_BLEND = 0.0     # E013 best configuration

N_SEEDS = 5

TERRAIN_COLS = ["elevation", "slope", "twi", "tri", "aspect"]
FULL_COLS = TERRAIN_COLS + ["river_dist"]
FEATURE_SETS = {"terrain": TERRAIN_COLS, "terrain_river": FULL_COLS}
BACKGROUNDS = ["random", "tgb", "hybrid"]
ALGORITHMS = ["maxent", "xgboost", "randomforest"]

RASTER_FILES = {
    "elevation": "jatim_dem.tif",
    "slope": "jatim_slope.tif",
    "twi": "jatim_twi.tif",
    "tri": "jatim_tri.tif",
    "aspect": "jatim_aspect.tif",
    "river_dist": "jatim_river_dist.tif",
}
ROAD_FILE = "jatim_road_dist_expanded.tif"


# --------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------

def read_decimated(path: Path, factor: int):
    """Read a raster at 1/factor resolution (nearest), returning array + transform."""
    with rasterio.open(path) as src:
        h, w = src.height // factor, src.width // factor
        arr = src.read(
            1, out_shape=(h, w), resampling=Resampling.nearest
        ).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        transform = src.transform * src.transform.scale(
            src.width / arr.shape[1], src.height / arr.shape[0]
        )
        return arr, transform


def sample_at_points(path: Path, xy: np.ndarray) -> np.ndarray:
    """Exact full-resolution raster values at point coordinates."""
    with rasterio.open(path) as src:
        vals = np.array(
            [v[0] for v in src.sample(xy)], dtype=np.float32
        )
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
        return vals


def load_sites() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SITES_PATH)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs("EPSG:4326")
    # Study-area clip identical to E013.
    jatim = (-9.0, 111.0, -6.5, 115.0)
    gdf = gdf[(gdf.geometry.y >= jatim[0]) & (gdf.geometry.x >= jatim[1]) &
              (gdf.geometry.y <= jatim[2]) & (gdf.geometry.x <= jatim[3])]
    return gdf.to_crs("EPSG:32749")


def build_frame() -> pd.DataFrame:
    """Decimated raster lattice used as the common background sampling frame."""
    cols = {}
    ref_transform = None
    for name, fname in RASTER_FILES.items():
        arr, transform = read_decimated(DEM_DIR / fname, DECIMATE)
        if ref_transform is None:
            ref_transform, ref_shape = transform, arr.shape
            rr, cc = np.meshgrid(
                np.arange(ref_shape[0]), np.arange(ref_shape[1]), indexing="ij"
            )
            xs, ys = rasterio.transform.xy(ref_transform, rr.ravel(), cc.ravel())
            cols["x"] = np.array(xs, dtype=np.float64)
            cols["y"] = np.array(ys, dtype=np.float64)
        if arr.shape != ref_shape:
            print(f"ERROR: raster {name} shape {arr.shape} != reference {ref_shape}")
            sys.exit(1)
        cols[name] = arr.ravel()
        del arr

    road_arr, _ = read_decimated(DEM_DIR / ROAD_FILE, DECIMATE)
    cols["road_dist"] = road_arr.ravel()
    del road_arr

    df = pd.DataFrame(cols)
    keep = df[FULL_COLS + ["road_dist"]].notna().all(axis=1) & (df["elevation"] > 0)
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------------
# Background designs
# --------------------------------------------------------------------------------

def assign_regions(x, y, midx, midy):
    return (x > midx).astype(int) + 2 * (y > midy).astype(int)


def allocate_counts(total: int, weights: np.ndarray) -> np.ndarray:
    raw = total * weights
    counts = np.floor(raw).astype(int)
    rem = int(total - counts.sum())
    if rem > 0:
        for i in np.argsort(-(raw - counts))[:rem]:
            counts[i] += 1
    return counts


def draw_random(frame: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    idx = rng.choice(frame.index.to_numpy(), size=n, replace=False)
    return frame.loc[idx]


def tgb_eligible(frame: pd.DataFrame, rng) -> np.ndarray:
    """Vectorised target-group acceptance: p = max(min_prob, exp(-road/decay))."""
    ok = frame["road_dist"].to_numpy() <= BASE_MAX_ROAD_DIST_M
    p = np.exp(-frame["road_dist"].to_numpy() / BASE_DECAY_M)
    p = np.clip(p, MIN_ACCEPT_PROB, 1.0)
    return ok & (rng.random(len(frame)) <= p)


def draw_tgb(frame: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    pool = frame[tgb_eligible(frame, rng)]
    if len(pool) < n:
        print(f"  WARNING: TGB pool {len(pool)} < target {n}")
        return pool
    idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
    return pool.loc[idx]


def draw_hybrid(frame: pd.DataFrame, n: int, presence_region_prop, rng) -> pd.DataFrame:
    """E013 design: TGB pool + regional quota blend + hard-negative fraction."""
    pool = frame[tgb_eligible(frame, rng)].copy()
    base = pool[pool["zdist"] <= HARD_Z_MAX]
    hard = base[base["zdist"] >= HARD_Z_MIN]

    uniform = np.full(4, 0.25)
    w = (1.0 - HYBRID_REGION_BLEND) * presence_region_prop + HYBRID_REGION_BLEND * uniform
    w = w / w.sum()

    n_hard = int(round(n * HYBRID_HARD_FRAC))
    picks: list = []

    for source, targets in ((hard, allocate_counts(n_hard, w)),
                            (base, allocate_counts(n - n_hard, w))):
        for rid in range(4):
            need = int(targets[rid])
            if need <= 0:
                continue
            cand = source.index[(source["region_id"] == rid) &
                                (~source.index.isin(picks))].to_numpy()
            if len(cand) == 0:
                continue
            picks.extend(rng.choice(cand, size=min(need, len(cand)), replace=False).tolist())

    if len(picks) < n:  # shortfall fill from any remaining base candidate
        remain = base.index[~base.index.isin(picks)].to_numpy()
        if len(remain) > 0:
            picks.extend(
                rng.choice(remain, size=min(n - len(picks), len(remain)), replace=False).tolist()
            )
    return base.loc[picks[:n]]


# --------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------

def assign_blocks(x, y, block_deg=BLOCK_SIZE_DEG):
    return (x / (block_deg * 111000)).astype(int) * 10000 + (y / (block_deg * 111000)).astype(int)


def deterministic_folds(blocks, n_folds=N_FOLDS):
    uniq = np.unique(blocks)
    uniq.sort()
    folds = []
    for test_blocks in np.array_split(uniq, n_folds):
        test = np.isin(blocks, test_blocks)
        folds.append((np.where(~test)[0], np.where(test)[0]))
    return folds


def tss(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def fit_predict(algo: str, X_tr, y_tr, X_te):
    if algo == "maxent":
        m = MaxentModel(
            feature_types=["linear", "hinge", "product"],
            beta_multiplier=1.5,
            transform="cloglog",
        )
        m.fit(X_tr, y_tr)
        return np.asarray(m.predict(X_te)).ravel()
    if algo == "xgboost":
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=spw,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            verbosity=0, random_state=RANDOM_SEED,
        )
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]
    m = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def run_cv(df: pd.DataFrame, feat_cols: list, algo: str) -> tuple:
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["presence"].to_numpy()
    folds = deterministic_folds(assign_blocks(df["x"].to_numpy(), df["y"].to_numpy()))
    aucs, tsss = [], []
    for tr, te in folds:
        if len(te) == 0 or y[te].sum() == 0 or y[te].sum() == len(te):
            continue
        p = fit_predict(algo, X[tr], y[tr], X[te])
        aucs.append(roc_auc_score(y[te], p))
        tsss.append(tss(y[te], p))
    return float(np.mean(aucs)), float(np.mean(tsss))


# --------------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 68)
    print("E217: MaxEnt benchmark across the pseudo-absence ladder")
    print("=" * 68)

    print("\nBuilding decimated sampling frame...")
    frame = build_frame()
    print(f"  Frame cells with complete covariates: {len(frame):,}")

    print("\nLoading presences...")
    sites = load_sites()
    xy = np.column_stack([sites.geometry.x, sites.geometry.y])
    pres = {c: sample_at_points(DEM_DIR / RASTER_FILES[c], xy) for c in FULL_COLS}
    pres = pd.DataFrame(pres)
    pres["x"], pres["y"] = xy[:, 0], xy[:, 1]
    pres = pres.dropna(subset=FULL_COLS)
    pres = pres[pres["elevation"] > 0].reset_index(drop=True)
    print(f"  Presences with valid features: {len(pres)}")

    n_pa = len(pres) * PSEUDOABSENCE_RATIO
    print(f"  Pseudo-absence target: {n_pa}")

    # Exclude frame cells within SITE_BUFFER_M of any presence (as E013's TGB pool does).
    from scipy.spatial import cKDTree
    tree = cKDTree(pres[["x", "y"]].to_numpy())
    d, _ = tree.query(frame[["x", "y"]].to_numpy(), k=1)
    frame = frame[d > SITE_BUFFER_M].reset_index(drop=True)
    print(f"  Frame after {SITE_BUFFER_M:.0f} m site-buffer exclusion: {len(frame):,}")

    # Environmental dissimilarity (zdist) + region id, needed by the hybrid design.
    mu = pres[FULL_COLS].mean().to_numpy(dtype=np.float64)
    sd = pres[FULL_COLS].std().replace(0, 1.0).to_numpy(dtype=np.float64)
    z = (frame[FULL_COLS].to_numpy(dtype=np.float64) - mu) / sd
    frame["zdist"] = np.sqrt((z ** 2).sum(axis=1))
    midx = 0.5 * (frame["x"].min() + frame["x"].max())
    midy = 0.5 * (frame["y"].min() + frame["y"].max())
    frame["region_id"] = assign_regions(frame["x"].to_numpy(), frame["y"].to_numpy(), midx, midy)

    pres_regions = assign_regions(pres["x"].to_numpy(), pres["y"].to_numpy(), midx, midy)
    prop = np.bincount(pres_regions, minlength=4).astype(float)
    prop = prop / prop.sum()
    print(f"  Presence region proportions: {np.round(prop, 3)}")

    rows = []
    for seed_i in range(N_SEEDS):
        rng = np.random.default_rng(RANDOM_SEED + 1000 * seed_i)
        print(f"\n--- seed {seed_i + 1}/{N_SEEDS} ---")

        draws = {
            "random": draw_random(frame, n_pa, rng),
            "tgb": draw_tgb(frame, n_pa, rng),
            "hybrid": draw_hybrid(frame, n_pa, prop, rng),
        }

        for bg in BACKGROUNDS:
            bg_df = draws[bg]
            frac_hard = float((bg_df["zdist"] >= HARD_Z_MIN).mean())
            data = pd.concat([
                pres[FULL_COLS + ["x", "y"]].assign(presence=1),
                bg_df[FULL_COLS + ["x", "y"]].assign(presence=0),
            ], ignore_index=True)

            for fs_name, feat_cols in FEATURE_SETS.items():
                for algo in ALGORITHMS:
                    auc, t = run_cv(data, feat_cols, algo)
                    rows.append({
                        "seed": seed_i, "background": bg, "feature_set": fs_name,
                        "algorithm": algo, "auc": auc, "tss": t,
                        "n_background": len(bg_df), "frac_zdist_ge2": frac_hard,
                    })
                    print(f"  {bg:<7} {fs_name:<13} {algo:<13} AUC={auc:.3f}  TSS={t:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "e217_raw_results.csv", index=False)

    summary = (res.groupby(["background", "feature_set", "algorithm"])
                  .agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                       tss_mean=("tss", "mean"), tss_sd=("tss", "std"))
                  .reset_index())
    summary.to_csv(RESULTS_DIR / "e217_summary.csv", index=False)

    print("\n" + "=" * 68)
    print("SUMMARY — mean AUC over seeds (rows: background x feature set)")
    print("=" * 68)
    pivot = summary.pivot_table(index=["background", "feature_set"],
                                columns="algorithm", values="auc_mean")
    pivot = pivot.reindex(BACKGROUNDS, level=0)
    print(pivot.round(3).to_string())
    pivot.round(4).to_csv(RESULTS_DIR / "e217_auc_matrix.csv")

    # --- Pre-registered readings ------------------------------------------------
    full = summary[summary["feature_set"] == "terrain_river"]
    verdict = {}
    for algo in ALGORITHMS:
        a = full[full["algorithm"] == algo].set_index("background")["auc_mean"]
        verdict[algo] = {
            "random": round(float(a["random"]), 4),
            "tgb": round(float(a["tgb"]), 4),
            "hybrid": round(float(a["hybrid"]), 4),
            "gain_random_to_hybrid": round(float(a["hybrid"] - a["random"]), 4),
            "monotonic": bool(a["random"] <= a["tgb"] <= a["hybrid"]),
        }

    # Background effect vs feature effect, measured on the same folds.
    tr = summary[summary["feature_set"] == "terrain"].set_index(
        ["background", "algorithm"])["auc_mean"]
    fr = summary[summary["feature_set"] == "terrain_river"].set_index(
        ["background", "algorithm"])["auc_mean"]
    feature_gain = float((fr - tr).mean())
    background_gain = float(np.mean([verdict[a]["gain_random_to_hybrid"] for a in ALGORITHMS]))

    out = {
        "experiment": "E217",
        "question": "Does the pseudo-absence effect replicate under MaxEnt? (JCAA R1, Reviewer 1)",
        "n_presences": int(len(pres)),
        "n_background_target": int(n_pa),
        "n_seeds": N_SEEDS,
        "cv": f"{N_FOLDS}-fold deterministic spatial block CV, {BLOCK_SIZE_DEG} deg blocks",
        "per_algorithm": verdict,
        "mean_gain_from_background_redesign": round(background_gain, 4),
        "mean_gain_from_adding_river_feature": round(feature_gain, 4),
        "background_effect_exceeds_feature_effect": bool(background_gain > feature_gain),
        "maxent_replicates_monotonic_ladder": verdict["maxent"]["monotonic"],
        "maxent_vs_xgboost_at_hybrid": round(
            verdict["maxent"]["hybrid"] - verdict["xgboost"]["hybrid"], 4),
    }
    with open(RESULTS_DIR / "e217_outcome.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 68)
    print("PRE-REGISTERED READINGS")
    print("=" * 68)
    for algo in ALGORITHMS:
        v = verdict[algo]
        print(f"  {algo:<13} random={v['random']:.3f} -> tgb={v['tgb']:.3f} -> "
              f"hybrid={v['hybrid']:.3f}  (gain {v['gain_random_to_hybrid']:+.3f}, "
              f"monotonic={v['monotonic']})")
    print(f"\n  Mean gain from BACKGROUND redesign : {background_gain:+.4f}")
    print(f"  Mean gain from ADDING river feature: {feature_gain:+.4f}")
    print(f"  Background effect > feature effect : {out['background_effect_exceeds_feature_effect']}")
    print(f"  MaxEnt replicates monotonic ladder : {out['maxent_replicates_monotonic_ladder']}")
    print(f"  MaxEnt - XGBoost at hybrid design  : {out['maxent_vs_xgboost_at_hybrid']:+.4f}")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
