"""
E219: Does background design change the MAP even when it does not change the score?

E217 + E218 established that P2's reported AUC ladder is an evaluation artefact, and that under an
artefact-immune metric the hybrid design shows no reliable discrimination benefit. That is a demolition.
A demolition alone is not a paper, and it makes Reviewer 2's objection WORSE, not better -- R2 already
asked what makes this specifically archaeological rather than a spatial-statistics exercise. Going more
methodological amplifies exactly that.

This experiment tests the constructive hypothesis. Target-group background was never justified by AUC in
the first place: its rationale (Phillips et al. 2009) is correcting sampling bias in the presences, which
is a claim about WHERE a model predicts, not how well it discriminates.

  H: background design materially changes the predicted suitability surface -- which cells a fieldworker
     would be sent to -- even when it does not change any discrimination metric.

CRITICAL CONTROL: maps are also compared BETWEEN SEEDS OF THE SAME DESIGN. If two draws of the same
design disagree as much as two different designs do, the "design effect" is just sampling noise. Without
this control the headline would be unfalsifiable.

Three parts:
  A. Map divergence      -- between-design vs within-design agreement (Spearman + top-decile Jaccard).
  B. Where they disagree -- is the difference organised by road distance, i.e. does TGB/hybrid actually
                            do the bias correction it claims to? Regressed against terrain as a control.
  C. Terrain-matched volcanic vs non-volcanic uplands (answers Reviewer 2's R2-F directly), using the
     CANONICAL 30-volcano inventory rather than the 7 hardcoded in the submitted code (fixes INT-1).

Run from repo root (long; use a background run):
    py experiments/E219_map_divergence/01_map_divergence.py
"""

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "e217base", HERE.parent / "E217_maxent_benchmark" / "01_maxent_benchmark.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

import pyproj                                    # noqa: E402
from scipy.spatial import cKDTree                # noqa: E402
from scipy.stats import spearmanr                # noqa: E402

RESULTS_DIR = HERE / "results"
FEAT = base.FULL_COLS
MAP_ALGOS = ["xgboost", "randomforest", "maxent"]
DESIGNS = ["random", "tgb", "hybrid"]
N_SEEDS = 5
TOP_FRAC = 0.10          # "survey priority" tier = top 10% of predicted suitability
PRED_CHUNK = 200_000     # MaxEnt hinge features are wide; predict in chunks

CANON_VOLCANOES = (base.REPO_ROOT / "data" / "processed" / "dashboard" / "volcanoes_java_full.csv")
# The 7 hardcoded in enhanced_tautology_tests.py and E013 (INT-1) -- kept to quantify the error.
LEGACY_VOLCANOES = {
    "Kelud": (-7.9300, 112.3080), "Semeru": (-8.1080, 112.9220),
    "Arjuno-Welirang": (-7.7290, 112.5750), "Bromo": (-7.9420, 112.9500),
    "Lamongan": (-7.9770, 113.3430), "Raung": (-8.1250, 114.0420),
    "Ijen": (-8.0580, 114.2420),
}

UPLAND_MIN_ELEV_M = 200.0
VOLCANIC_MAX_KM = 20.0
NONVOLCANIC_MIN_KM = 40.0
N_MATCH_BINS = 5


# ------------------------------------------------------------------ helpers

def to_utm(lat, lon):
    tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
    return tr.transform(lon, lat)


def volcano_distance_km(x, y, volc_latlon):
    """Euclidean distance in EPSG:32749 to the nearest volcano.

    Projected-plane distance rather than geodesic: within this UTM zone and at the <100 km ranges that
    matter here the difference is well under 1%, and it is ~1000x faster over a million cells.
    """
    vx, vy = [], []
    for lat, lon in volc_latlon:
        px, py = to_utm(lat, lon)
        vx.append(px)
        vy.append(py)
    tree = cKDTree(np.column_stack([vx, vy]))
    d, _ = tree.query(np.column_stack([x, y]), k=1)
    return d / 1000.0


def load_canonical_volcanoes(bounds_lon=(111.0, 115.0)):
    df = pd.read_csv(CANON_VOLCANOES)
    inb = df[(df["lon"] >= bounds_lon[0]) & (df["lon"] <= bounds_lon[1])]
    return list(zip(inb["lat"].to_numpy(), inb["lon"].to_numpy())), inb["name"].tolist()


def fit_full(algo, X, y):
    """Fit on all data (map production, not validation) and return a predict function."""
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from elapid import MaxentModel
    if algo == "maxent":
        m = MaxentModel(feature_types=["linear", "hinge", "product"],
                        beta_multiplier=1.5, transform="cloglog")
        m.fit(X, y)
        return lambda Z: np.asarray(m.predict(Z)).ravel()
    if algo == "xgboost":
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", verbosity=0, random_state=base.RANDOM_SEED)
        m.fit(X, y)
        return lambda Z: m.predict_proba(Z)[:, 1]
    m = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced",
                               random_state=base.RANDOM_SEED, n_jobs=-1)
    m.fit(X, y)
    return lambda Z: m.predict_proba(Z)[:, 1]


def predict_chunked(pred_fn, Z):
    out = np.empty(len(Z), dtype=np.float32)
    for i in range(0, len(Z), PRED_CHUNK):
        out[i:i + PRED_CHUNK] = pred_fn(Z[i:i + PRED_CHUNK])
    return out


def jaccard_top(a, b, frac=TOP_FRAC):
    k = int(len(a) * frac)
    ta = set(np.argpartition(-a, k)[:k])
    tb = set(np.argpartition(-b, k)[:k])
    return len(ta & tb) / len(ta | tb)


# ------------------------------------------------------------------ main

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("E219: does background design change the map, not just the score?")
    print("=" * 74)

    frame = base.build_frame()
    sites = base.load_sites()
    xy = np.column_stack([sites.geometry.x, sites.geometry.y])
    pres = pd.DataFrame({c: base.sample_at_points(base.DEM_DIR / base.RASTER_FILES[c], xy)
                         for c in FEAT})
    pres["x"], pres["y"] = xy[:, 0], xy[:, 1]
    pres = pres.dropna(subset=FEAT)
    pres = pres[pres["elevation"] > 0].reset_index(drop=True)

    tree = cKDTree(pres[["x", "y"]].to_numpy())
    d_site, _ = tree.query(frame[["x", "y"]].to_numpy(), k=1)
    frame_buf = frame[d_site > base.SITE_BUFFER_M].reset_index(drop=True)

    mu = pres[FEAT].mean().to_numpy(dtype=np.float64)
    sd = pres[FEAT].std().replace(0, 1.0).to_numpy(dtype=np.float64)
    midx = 0.5 * (frame["x"].min() + frame["x"].max())
    midy = 0.5 * (frame["y"].min() + frame["y"].max())

    def decorate(df):
        df = df.copy()
        z = (df[FEAT].to_numpy(dtype=np.float64) - mu) / sd
        df["zdist"] = np.sqrt((z ** 2).sum(axis=1))
        df["region_id"] = base.assign_regions(df["x"].to_numpy(), df["y"].to_numpy(), midx, midy)
        return df

    frame, frame_buf = decorate(frame), decorate(frame_buf)
    pres_reg = base.assign_regions(pres["x"].to_numpy(), pres["y"].to_numpy(), midx, midy)
    prop = np.bincount(pres_reg, minlength=4).astype(float)
    prop /= prop.sum()
    n_pa = len(pres) * base.PSEUDOABSENCE_RATIO
    print(f"  presences={len(pres)}  frame={len(frame):,}  background target={n_pa}")

    designs = {
        "random": lambda rng: base.draw_random(frame_buf, n_pa, rng),
        "tgb": lambda rng: base.draw_tgb(frame_buf, n_pa, rng),
        "hybrid": lambda rng: base.draw_hybrid(frame_buf, n_pa, prop, rng),
    }

    Z = frame[FEAT].to_numpy(dtype=np.float64)

    # ---------------- Part A: produce maps ----------------
    print("\nPart A — producing prediction surfaces...")
    maps = {}            # (algo, design, seed) -> ranks over the frame
    for s in range(N_SEEDS):
        rng = np.random.default_rng(base.RANDOM_SEED + 1000 * s)
        bgs = {d: f(rng) for d, f in designs.items()}
        for d in DESIGNS:
            X = np.vstack([pres[FEAT].to_numpy(dtype=np.float64),
                           bgs[d][FEAT].to_numpy(dtype=np.float64)])
            y = np.r_[np.ones(len(pres)), np.zeros(len(bgs[d]))]
            for algo in MAP_ALGOS:
                p = predict_chunked(fit_full(algo, X, y), Z)
                maps[(algo, d, s)] = p
        print(f"  seed {s + 1}/{N_SEEDS} done")

    rows = []
    for algo in MAP_ALGOS:
        # within-design (same design, different seeds) = the noise floor
        for d in DESIGNS:
            for i in range(N_SEEDS):
                for j in range(i + 1, N_SEEDS):
                    a, b = maps[(algo, d, i)], maps[(algo, d, j)]
                    rows.append({"algorithm": algo, "comparison": "within_design",
                                 "pair": f"{d}|{d}", "seed_i": i, "seed_j": j,
                                 "spearman": spearmanr(a, b).statistic,
                                 "jaccard_top10": jaccard_top(a, b)})
        # between-design, same seed = the design effect
        for i, d1 in enumerate(DESIGNS):
            for d2 in DESIGNS[i + 1:]:
                for s in range(N_SEEDS):
                    a, b = maps[(algo, d1, s)], maps[(algo, d2, s)]
                    rows.append({"algorithm": algo, "comparison": "between_design",
                                 "pair": f"{d1}|{d2}", "seed_i": s, "seed_j": s,
                                 "spearman": spearmanr(a, b).statistic,
                                 "jaccard_top10": jaccard_top(a, b)})
    div = pd.DataFrame(rows)
    div.to_csv(RESULTS_DIR / "e219_map_divergence.csv", index=False)

    print("\n" + "=" * 74)
    print("PART A — map agreement: between designs vs within design (noise floor)")
    print("=" * 74)
    ag = div.groupby(["algorithm", "comparison"])[["spearman", "jaccard_top10"]].mean().round(4)
    print(ag.to_string())
    pair_ag = div[div.comparison == "between_design"].groupby(
        ["algorithm", "pair"])[["spearman", "jaccard_top10"]].mean().round(4)
    print("\nBy design pair:")
    print(pair_ag.to_string())
    ag.to_csv(RESULTS_DIR / "e219_agreement_summary.csv")

    # ---------------- Part B: where do they disagree? ----------------
    print("\n" + "=" * 74)
    print("PART B — is the disagreement organised by road distance (the bias-correction rationale)?")
    print("=" * 74)
    road = frame["road_dist"].to_numpy()
    elev = frame["elevation"].to_numpy()
    road_q = pd.qcut(road, 5, labels=False, duplicates="drop")
    brows = []
    for algo in MAP_ALGOS:
        for d in ["tgb", "hybrid"]:
            diff = np.mean([
                pd.Series(maps[(algo, d, s)]).rank(pct=True).to_numpy() -
                pd.Series(maps[(algo, "random", s)]).rank(pct=True).to_numpy()
                for s in range(N_SEEDS)], axis=0)
            r_road = spearmanr(road, diff)
            r_elev = spearmanr(elev, diff)
            for q in range(int(np.nanmax(road_q)) + 1):
                m = road_q == q
                brows.append({"algorithm": algo, "design": d, "road_quintile": q + 1,
                              "median_road_m": float(np.median(road[m])),
                              "mean_rank_shift": float(diff[m].mean())})
            print(f"  {algo:<13} {d:<7} Spearman(road_dist, rank shift vs random) = "
                  f"{r_road.statistic:+.3f}   (elevation control: {r_elev.statistic:+.3f})")
    bdf = pd.DataFrame(brows)
    bdf.to_csv(RESULTS_DIR / "e219_disagreement_by_road.csv", index=False)
    print("\nMean percentile-rank shift vs random background, by road-distance quintile:")
    print(bdf.pivot_table(index=["algorithm", "design"], columns="road_quintile",
                          values="mean_rank_shift").round(4).to_string())

    # ---------------- Part C: terrain-matched volcanic vs non-volcanic ----------------
    print("\n" + "=" * 74)
    print("PART C — terrain-matched volcanic vs non-volcanic uplands (Reviewer 2, R2-F) + INT-1 fix")
    print("=" * 74)
    canon_latlon, canon_names = load_canonical_volcanoes()
    print(f"  Canonical volcanoes inside 111-115E: {len(canon_latlon)} — {', '.join(canon_names)}")
    print(f"  Legacy hardcoded set in submitted code: {len(LEGACY_VOLCANOES)}")

    fx, fy = frame["x"].to_numpy(), frame["y"].to_numpy()
    d_canon = volcano_distance_km(fx, fy, canon_latlon)
    d_legacy = volcano_distance_km(fx, fy, list(LEGACY_VOLCANOES.values()))
    frame["volc_km_canon"], frame["volc_km_legacy"] = d_canon, d_legacy

    # INT-1: the paper's Test 1 correlation, recomputed on the canonical inventory.
    hybrid_map = np.mean([maps[("xgboost", "hybrid", s)] for s in range(N_SEEDS)], axis=0)
    rho_legacy = spearmanr(hybrid_map, d_legacy).statistic
    rho_canon = spearmanr(hybrid_map, d_canon).statistic
    print(f"\n  INT-1 — Test 1 tautology correlation (suitability vs volcano distance):")
    print(f"    with 7 legacy volcanoes  : rho = {rho_legacy:+.3f}   (paper reports -0.163)")
    print(f"    with {len(canon_latlon)} canonical volcanoes: rho = {rho_canon:+.3f}")

    upland = frame["elevation"].to_numpy() >= UPLAND_MIN_ELEV_M
    volc = upland & (d_canon <= VOLCANIC_MAX_KM)
    nonvolc = upland & (d_canon >= NONVOLCANIC_MIN_KM)
    print(f"\n  Upland cells (>= {UPLAND_MIN_ELEV_M:.0f} m): {upland.sum():,}"
          f"   volcanic (<= {VOLCANIC_MAX_KM:.0f} km): {volc.sum():,}"
          f"   non-volcanic (>= {NONVOLCANIC_MIN_KM:.0f} km): {nonvolc.sum():,}")

    # Coarsened exact matching on the four terrain covariates.
    match_cols = ["elevation", "slope", "tri", "twi"]
    keys = np.zeros(len(frame), dtype=np.int64)
    for i, c in enumerate(match_cols):
        v = frame[c].to_numpy()
        edges = np.quantile(v[upland], np.linspace(0, 1, N_MATCH_BINS + 1)[1:-1])
        keys = keys * N_MATCH_BINS + np.digitize(v, edges)
    strata_v = pd.Series(keys[volc]).value_counts()
    strata_n = pd.Series(keys[nonvolc]).value_counts()
    common = sorted(set(strata_v.index) & set(strata_n.index))
    print(f"  Terrain strata present in BOTH groups: {len(common)} of "
          f"{len(set(strata_v.index) | set(strata_n.index))}")

    site_keys = np.zeros(len(pres), dtype=np.int64)
    for i, c in enumerate(match_cols):
        v = pres[c].to_numpy()
        edges = np.quantile(frame[c].to_numpy()[upland], np.linspace(0, 1, N_MATCH_BINS + 1)[1:-1])
        site_keys = site_keys * N_MATCH_BINS + np.digitize(v, edges)
    site_volc_km = volcano_distance_km(pres["x"].to_numpy(), pres["y"].to_numpy(), canon_latlon)
    site_upland = pres["elevation"].to_numpy() >= UPLAND_MIN_ELEV_M

    cell_km2 = (base.DECIMATE * 30.663793749005784 / 1000.0) ** 2
    crows = []
    for k in common:
        mv, mn = (keys == k) & volc, (keys == k) & nonvolc
        w = min(mv.sum(), mn.sum())          # matched weight = smaller arm
        sv = int(((site_keys == k) & site_upland & (site_volc_km <= VOLCANIC_MAX_KM)).sum())
        sn = int(((site_keys == k) & site_upland & (site_volc_km >= NONVOLCANIC_MIN_KM)).sum())
        crows.append({"stratum": int(k), "n_cells_volcanic": int(mv.sum()),
                      "n_cells_nonvolcanic": int(mn.sum()), "match_weight": int(w),
                      "suit_volcanic": float(hybrid_map[mv].mean()),
                      "suit_nonvolcanic": float(hybrid_map[mn].mean()),
                      "sites_volcanic": sv, "sites_nonvolcanic": sn,
                      "area_km2_volcanic": float(mv.sum() * cell_km2),
                      "area_km2_nonvolcanic": float(mn.sum() * cell_km2)})
    cdf = pd.DataFrame(crows)
    cdf.to_csv(RESULTS_DIR / "e219_terrain_matched.csv", index=False)

    W = cdf["match_weight"].to_numpy().astype(float)
    if W.sum() > 0:
        suit_v = float(np.average(cdf["suit_volcanic"], weights=W))
        suit_n = float(np.average(cdf["suit_nonvolcanic"], weights=W))
        dens_v = cdf["sites_volcanic"].sum() / max(cdf["area_km2_volcanic"].sum(), 1e-9)
        dens_n = cdf["sites_nonvolcanic"].sum() / max(cdf["area_km2_nonvolcanic"].sum(), 1e-9)
        print(f"\n  Matched-weighted mean predicted suitability:")
        print(f"    volcanic uplands     : {suit_v:.4f}")
        print(f"    non-volcanic uplands : {suit_n:.4f}   (difference {suit_v - suit_n:+.4f})")
        print(f"  Observed site density within matched strata (sites / km2):")
        print(f"    volcanic uplands     : {dens_v:.5f}  ({int(cdf['sites_volcanic'].sum())} sites / "
              f"{cdf['area_km2_volcanic'].sum():.0f} km2)")
        print(f"    non-volcanic uplands : {dens_n:.5f}  ({int(cdf['sites_nonvolcanic'].sum())} sites / "
              f"{cdf['area_km2_nonvolcanic'].sum():.0f} km2)")
    else:
        suit_v = suit_n = dens_v = dens_n = float("nan")

    # ---------------- verdicts ----------------
    out = {
        "experiment": "E219", "n_seeds": N_SEEDS, "n_presences": int(len(pres)),
        "frame_cells": int(len(frame)),
        "partA": {
            algo: {
                "within_design_spearman": float(
                    div[(div.algorithm == algo) & (div.comparison == "within_design")].spearman.mean()),
                "between_design_spearman": float(
                    div[(div.algorithm == algo) & (div.comparison == "between_design")].spearman.mean()),
                "within_design_jaccard": float(
                    div[(div.algorithm == algo) & (div.comparison == "within_design")].jaccard_top10.mean()),
                "between_design_jaccard": float(
                    div[(div.algorithm == algo) & (div.comparison == "between_design")].jaccard_top10.mean()),
            } for algo in MAP_ALGOS
        },
        "int1_test1_rho_legacy_7_volcanoes": round(float(rho_legacy), 4),
        "int1_test1_rho_canonical_volcanoes": round(float(rho_canon), 4),
        "int1_n_canonical_in_bounds": len(canon_latlon),
        "partC": {
            "suitability_volcanic_matched": suit_v,
            "suitability_nonvolcanic_matched": suit_n,
            "site_density_volcanic": float(dens_v),
            "site_density_nonvolcanic": float(dens_n),
            "n_matched_strata": len(common),
        },
    }
    for algo in MAP_ALGOS:
        a = out["partA"][algo]
        a["design_effect_exceeds_noise_floor"] = bool(
            a["between_design_jaccard"] < a["within_design_jaccard"] - 0.05)
    out["maps_differ_beyond_noise"] = bool(
        sum(out["partA"][a]["design_effect_exceeds_noise_floor"] for a in MAP_ALGOS) >= 2)

    with open(RESULTS_DIR / "e219_outcome.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 74)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 74)
    for algo in MAP_ALGOS:
        a = out["partA"][algo]
        # Plain ASCII: this script is run with a cp1252 console on Windows.
        print(f"  {algo:<13} top-10% overlap  within-design {a['within_design_jaccard']:.3f}  vs  "
              f"between-design {a['between_design_jaccard']:.3f}   "
              f"-> design effect > noise: {a['design_effect_exceeds_noise_floor']}")
    print(f"\n  MAPS DIFFER BEYOND SAMPLING NOISE (>=2 of 3 algorithms): "
          f"{out['maps_differ_beyond_noise']}")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
