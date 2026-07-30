"""
E222: does the wrong-direction selection replicate against GROUND TRUTH?

E217-E220 measured "generalisation" against the same 378 survey-biased presences. The harsh-review
hole: the honest ruler cannot see bias-correction either. This experiment builds synthetic worlds on
the real East Java lattice -- known intensity surface, deliberately applied survey bias, identical
pipeline code (E217 base draw functions, same CV, same algorithms) -- and scores every configuration
against truth: a held-out unbiased presence sample, and the intensity surface itself.
Pre-registration: DESIGN.md.

Two surfaces: A fully observed (4 terrain drivers), B misspecified (A + clay, a real raster withheld
from the feature set). 10 worlds each. Configs: random, tgb, hybrid hard_frac in {0.0, 0.3, 0.7, 1.0}.
Algorithms: maxent, xgboost, randomforest (E217 hyperparameters).

Run from repo root (long; use a background run):
    py experiments/E222_synthetic_ground_truth/01_synthetic_truth.py
"""

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
E217 = HERE.parent / "E217_maxent_benchmark"
spec = importlib.util.spec_from_file_location("e217base", E217 / "01_maxent_benchmark.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

E219 = HERE.parent / "E219_map_divergence"
spec219 = importlib.util.spec_from_file_location("e219main", E219 / "01_map_divergence.py")
e219 = importlib.util.module_from_spec(spec219)
spec219.loader.exec_module(e219)

from scipy.spatial import cKDTree                     # noqa: E402
from scipy.stats import spearmanr                     # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402

RESULTS_DIR = HERE / "results"
FEAT = base.FULL_COLS
ALGOS = base.ALGORITHMS
CONFIGS = [("random", None), ("tgb", None), ("hybrid", 0.0), ("hybrid", 0.3),
           ("hybrid", 0.7), ("hybrid", 1.0)]
N_WORLDS = 10
N_EVAL_PRES = 400
EVAL_RATIO = 5
TOP_FRAC = 0.10
# intensity coefficients (z-scored covariates); clay is the withheld driver in surface B
BETA_A = {"elevation": -1.0, "slope": -0.8, "river_dist": -1.2, "twi": 0.6}
BETA_CLAY = 0.8
SURVEY_DECAY_M = 12000.0
SURVEY_MIN_P = 0.03


def boyce_simple(pred_presence, pred_available):
    """Continuous Boyce at the E218/E220 defaults (width range/10, 101 windows)."""
    lo = float(min(pred_presence.min(), pred_available.min()))
    hi = float(max(pred_presence.max(), pred_available.max()))
    if hi <= lo:
        return float("nan")
    width = (hi - lo) / 10.0
    mids, ratios = [], []
    for s in np.linspace(lo, hi - width, 101):
        e = s + width
        n_a = int(((pred_available >= s) & (pred_available < e)).sum())
        if n_a == 0:
            continue
        n_p = int(((pred_presence >= s) & (pred_presence < e)).sum())
        mids.append(0.5 * (s + e))
        ratios.append((n_p / len(pred_presence)) / (n_a / len(pred_available)))
    if len(mids) < 5 or np.allclose(ratios, ratios[0]):
        return float("nan")
    return float(spearmanr(mids, ratios).statistic)


def build_world_frame(frame, obs_idx):
    """Frame decorated with zdist/region_id computed from THIS world's observed presences."""
    df = frame.copy()
    mu = df[FEAT].to_numpy(dtype=np.float64)[obs_idx].mean(axis=0)
    sd = df[FEAT].to_numpy(dtype=np.float64)[obs_idx].std(axis=0)
    sd[sd == 0] = 1.0
    z = (df[FEAT].to_numpy(dtype=np.float64) - mu) / sd
    df["zdist"] = np.sqrt((z ** 2).sum(axis=1))
    midx = 0.5 * (df["x"].min() + df["x"].max())
    midy = 0.5 * (df["y"].min() + df["y"].max())
    df["region_id"] = base.assign_regions(df["x"].to_numpy(), df["y"].to_numpy(), midx, midy)
    pres_reg = base.assign_regions(df["x"].to_numpy()[obs_idx],
                                   df["y"].to_numpy()[obs_idx], midx, midy)
    prop = np.bincount(pres_reg, minlength=4).astype(float)
    prop /= prop.sum()
    return df, prop


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("E222: synthetic ground-truth validation of the wrong-direction selection")
    print("=" * 74)

    # --- real lattice + real covariates; only the archaeology is synthetic
    frame = base.build_frame()
    clay = base.sample_at_points(base.DEM_DIR / "jatim_clay.tif",
                                 frame[["x", "y"]].to_numpy())
    keep = np.isfinite(clay)
    frame = frame[keep].reset_index(drop=True)
    clay = clay[keep]
    print(f"  frame cells (clay-complete): {len(frame):,}")

    Xf = frame[FEAT].to_numpy(dtype=np.float64)
    xyf = frame[["x", "y"]].to_numpy()
    road = frame["road_dist"].to_numpy()
    zf = (Xf - Xf.mean(axis=0)) / Xf.std(axis=0)
    zc = (clay - clay.mean()) / clay.std()
    remote_mask = road >= np.quantile(road, 0.8)
    survey_p = np.clip(np.exp(-road / SURVEY_DECAY_M), SURVEY_MIN_P, 1.0)

    loglam = {}
    for surf, use_clay in (("A_observed", False), ("B_misspecified", True)):
        ll = sum(BETA_A[c] * zf[:, FEAT.index(c)] for c in BETA_A)
        if use_clay:
            ll = ll + BETA_CLAY * zc
        ll = ll - ll.max()
        lam = np.exp(ll)
        loglam[surf] = lam / lam.sum()          # sampling probabilities
    top_true = {s: set(np.argpartition(-loglam[s], int(len(loglam[s]) * TOP_FRAC))[
                        :int(len(loglam[s]) * TOP_FRAC)]) for s in loglam}

    rows = []
    for surf in ("A_observed", "B_misspecified"):
        p_lam = loglam[surf]
        for w in range(N_WORLDS):
            wrng = np.random.default_rng(10_000 + 997 * w)
            # adaptive N so observed count matches the real dataset's power (250-800)
            obs = np.array([], dtype=int)
            n_true = 1100
            for _attempt in range(4):
                true_idx = wrng.choice(len(frame), size=n_true, replace=False, p=p_lam)
                obs = true_idx[wrng.random(n_true) < survey_p[true_idx]]
                if 250 <= len(obs) <= 800:
                    break
                n_true = int(n_true * (500 / max(len(obs), 50)))
            eval_idx = wrng.choice(len(frame), size=N_EVAL_PRES, replace=False, p=p_lam)
            avail_idx = wrng.choice(len(frame), size=N_EVAL_PRES * EVAL_RATIO, replace=False)

            wdf, prop = build_world_frame(frame, obs)
            tree = cKDTree(xyf[obs])
            d_site, _ = tree.query(xyf, k=1)
            buf_mask = d_site > base.SITE_BUFFER_M
            frame_buf = wdf[buf_mask]
            n_pa = len(obs) * base.PSEUDOABSENCE_RATIO

            pres_df = pd.DataFrame(Xf[obs], columns=FEAT)
            pres_df["x"], pres_df["y"] = xyf[obs, 0], xyf[obs, 1]
            eval_df = pd.DataFrame(Xf[eval_idx], columns=FEAT)
            eval_df["x"], eval_df["y"] = xyf[eval_idx, 0], xyf[eval_idx, 1]
            avail_df = pd.DataFrame(Xf[avail_idx], columns=FEAT)
            avail_df["x"], avail_df["y"] = xyf[avail_idx, 0], xyf[avail_idx, 1]

            for ci, (cname, hf) in enumerate(CONFIGS):
                crng = np.random.default_rng(77_777 + 1000 * w + ci)
                if cname == "random":
                    bg = base.draw_random(frame_buf, n_pa, crng)
                elif cname == "tgb":
                    bg = base.draw_tgb(frame_buf, n_pa, crng)
                else:
                    old = base.HYBRID_HARD_FRAC
                    base.HYBRID_HARD_FRAC = hf
                    try:
                        bg = base.draw_hybrid(frame_buf, n_pa, prop, crng)
                    finally:
                        base.HYBRID_HARD_FRAC = old
                if len(bg) < int(0.9 * n_pa):
                    continue

                tr = pd.concat([pres_df.assign(presence=1), bg.assign(presence=0)],
                               ignore_index=True)[FEAT + ["x", "y", "presence"]]
                te_true = pd.concat([eval_df.assign(presence=1), avail_df.assign(presence=0)],
                                    ignore_index=True)
                b_tr = base.assign_blocks(tr["x"].to_numpy(), tr["y"].to_numpy())
                b_te = base.assign_blocks(te_true["x"].to_numpy(), te_true["y"].to_numpy())
                uniq = np.unique(np.concatenate([b_tr, b_te]))
                uniq.sort()

                for algo in ALGOS:
                    auc_own_l, auc_true_l = [], []
                    for test_blocks in np.array_split(uniq, base.N_FOLDS):
                        m_tr = ~np.isin(b_tr, test_blocks)
                        m_own = np.isin(b_tr, test_blocks)
                        m_te = np.isin(b_te, test_blocks)
                        y_tr = tr["presence"].to_numpy()[m_tr]
                        y_own = tr["presence"].to_numpy()[m_own]
                        y_te = te_true["presence"].to_numpy()[m_te]
                        if (y_tr.sum() == 0 or y_own.sum() in (0, len(y_own))
                                or y_te.sum() in (0, len(y_te))):
                            continue
                        X_tr = tr.loc[m_tr, FEAT].to_numpy(dtype=np.float64)
                        X_own = tr.loc[m_own, FEAT].to_numpy(dtype=np.float64)
                        X_te = te_true.loc[m_te, FEAT].to_numpy(dtype=np.float64)
                        p = np.asarray(base.fit_predict(algo, X_tr, y_tr,
                                                        np.vstack([X_own, X_te]))).ravel()
                        auc_own_l.append(roc_auc_score(y_own, p[:len(X_own)]))
                        auc_true_l.append(roc_auc_score(y_te, p[len(X_own):]))
                    # full-data fit -> full-frame prediction -> recovery + Boyce
                    X_all = tr[FEAT].to_numpy(dtype=np.float64)
                    y_all = tr["presence"].to_numpy()
                    pred = e219.predict_chunked(e219.fit_full(algo, X_all, y_all), Xf)
                    k = int(len(pred) * TOP_FRAC)
                    top_pred = set(np.argpartition(-pred, k)[:k])
                    jac = len(top_pred & top_true[surf]) / len(top_pred | top_true[surf])
                    sp_full = spearmanr(pred, p_lam).statistic
                    sp_rem = spearmanr(pred[remote_mask], p_lam[remote_mask]).statistic
                    boy = boyce_simple(pred[obs], pred[avail_idx])
                    rows.append({"surface": surf, "world": w, "config": cname,
                                 "hard_frac": hf if hf is not None else -1,
                                 "algorithm": algo, "n_observed": len(obs),
                                 "auc_own": float(np.mean(auc_own_l)),
                                 "auc_true": float(np.mean(auc_true_l)),
                                 "boyce": boy, "map_jaccard": jac,
                                 "spearman_full": float(sp_full),
                                 "spearman_remote": float(sp_rem)})
            print(f"  {surf} world {w + 1}/{N_WORLDS} done (observed={len(obs)})")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e222_runs.csv", index=False)

    # ---------------- selection analysis ----------------
    srows = []
    for (surf, w, algo), g in df.groupby(["surface", "world", "algorithm"]):
        g = g.reset_index(drop=True)
        i_own = int(np.argmax(g["auc_own"]))
        i_true = int(np.argmax(g["auc_true"]))
        i_map = int(np.argmax(g["map_jaccard"]))
        srows.append({"surface": surf, "world": w, "algorithm": algo,
                      "Rown_pick": g.loc[i_own, "config"] + f"({g.loc[i_own, 'hard_frac']})",
                      "Rown_picks_hybrid1.0": bool(
                          g.loc[i_own, "config"] == "hybrid" and g.loc[i_own, "hard_frac"] == 1.0),
                      "cost_true": float(g.loc[i_true, "auc_true"] - g.loc[i_own, "auc_true"]),
                      "cost_map": float(g.loc[i_map, "map_jaccard"] - g.loc[i_own, "map_jaccard"]),
                      "boyce_vs_map_spearman": float(spearmanr(g["boyce"], g["map_jaccard"]).statistic),
                      "boyce_vs_auctrue_spearman": float(spearmanr(g["boyce"], g["auc_true"]).statistic)})
    sel = pd.DataFrame(srows)
    sel.to_csv(RESULTS_DIR / "e222_selection.csv", index=False)

    # tgb vs random recovery (P3)
    pv = df.pivot_table(index=["surface", "world", "algorithm"], columns="config",
                        values=["map_jaccard", "spearman_remote", "auc_true"])
    tgb_gain = (pv[("map_jaccard", "tgb")] - pv[("map_jaccard", "random")]).to_numpy()
    tgb_gain_rem = (pv[("spearman_remote", "tgb")] - pv[("spearman_remote", "random")]).to_numpy()

    # P1: inflation vs hard_frac among hybrid configs
    hyb = df[df.config == "hybrid"]
    r_inf = spearmanr(hyb["hard_frac"], hyb["auc_own"] - hyb["auc_true"])

    out = {
        "experiment": "E222", "n_worlds": N_WORLDS, "n_runs": len(df),
        "P1_inflation_spearman_hybrid": round(float(r_inf.statistic), 4),
        "P1_p_inflation": float(r_inf.pvalue),
        "P1_frac_Rown_picks_hybrid1.0": round(float(sel["Rown_picks_hybrid1.0"].mean()), 4),
        "P2_cost_true_median": round(float(sel["cost_true"].median()), 4),
        "P2_frac_cost_true_positive": round(float((sel["cost_true"] > 0).mean()), 4),
        "P2_cost_map_median": round(float(sel["cost_map"].median()), 4),
        "P3_tgb_minus_random_mapjaccard_mean": round(float(np.mean(tgb_gain)), 4),
        "P3_frac_positive": round(float((tgb_gain > 0).mean()), 4),
        "P3_remote_spearman_gain_mean": round(float(np.mean(tgb_gain_rem)), 4),
        "P3_remote_frac_positive": round(float((tgb_gain_rem > 0).mean()), 4),
        "P4_boyce_vs_map_spearman_median": round(float(sel["boyce_vs_map_spearman"].median()), 4),
        "P4_boyce_vs_auctrue_spearman_median": round(
            float(sel["boyce_vs_auctrue_spearman"].median()), 4),
    }
    out["P1_supported"] = bool(out["P1_inflation_spearman_hybrid"] > 0.5
                               and out["P1_frac_Rown_picks_hybrid1.0"] >= 0.6)
    out["P2_supported"] = bool(out["P2_cost_true_median"] >= 0.02
                               and out["P2_frac_cost_true_positive"] >= 0.7)
    out["P3_tgb_helps_map"] = bool(out["P3_frac_positive"] >= 0.6)
    out["P4_boyce_tracks_truth"] = bool(out["P4_boyce_vs_map_spearman_median"] >= 0.5)
    # P5 descriptive: surface A vs B differences in design effects
    for surf in ("A_observed", "B_misspecified"):
        s = df[df.surface == surf]
        out[f"P5_{surf}"] = {
            "mean_auc_true_random": round(float(s[s.config == "random"]["auc_true"].mean()), 4),
            "mean_map_jaccard_random": round(float(s[s.config == "random"]["map_jaccard"].mean()), 4),
            "mean_map_jaccard_tgb": round(float(s[s.config == "tgb"]["map_jaccard"].mean()), 4),
            "cost_true_median": round(float(sel[sel.surface == surf]["cost_true"].median()), 4),
        }
    with open(RESULTS_DIR / "e222_outcome.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 74)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 74)
    for key in ("P1_supported", "P2_supported", "P3_tgb_helps_map", "P4_boyce_tracks_truth"):
        print(f"  {key:<28}: {out[key]}")
    print(f"  inflation Spearman (hybrid)   : {out['P1_inflation_spearman_hybrid']:+.3f}")
    print(f"  R-own picks hybrid(1.0)       : {out['P1_frac_Rown_picks_hybrid1.0']:.0%}")
    print(f"  cost_true median / frac pos   : {out['P2_cost_true_median']:+.4f} / "
          f"{out['P2_frac_cost_true_positive']:.0%}")
    print(f"  tgb-random map Jaccard        : {out['P3_tgb_minus_random_mapjaccard_mean']:+.4f} "
          f"(frac pos {out['P3_frac_positive']:.0%}); remote {out['P3_remote_spearman_gain_mean']:+.4f}")
    print(f"  Boyce-truth agreement         : map {out['P4_boyce_vs_map_spearman_median']:+.3f}, "
          f"auc_true {out['P4_boyce_vs_auctrue_spearman_median']:+.3f}")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
