"""
E222 World D: region-BALANCED truth + regional survey bias — the quota's friendliest regime.

World C gave the quota regional survey bias but kept the concentrated intensity of surface A; the
quota's contamination failure there is itself regime-contingent. World D removes that confound: the
intensity surface is REBALANCED so every quadrant carries equal total intensity (within-region
environmental logic intact), while survey effort stays regionally uneven [1.0, 0.4, 0.15, 0.05].
The observed record then concentrates in region 0 through survey effort ALONE -- exactly the bias the
E013 regional quota was designed to correct. If the quota cannot win here, it has no home regime in
the suite.

Pre-registered fork (locked before running):
  - If hybrid(hf=0.0) beats random on truth-anchored recovery (auc_true, map_jaccard) in >= 60% of the
    30 world x algorithm cases: the quota has a DEMONSTRATED regime; the manuscript's message becomes
    "each design has a regime where its rationale holds, and the reported AUC cannot tell you which
    regime you are in".
  - If not: across four synthetic regimes, no background design ever beats uniform on truth, while the
    reported AUC always prefers the most extreme one. The manuscript says exactly that.

Run from repo root (~20 min):
    py experiments/E222_synthetic_ground_truth/03_world_d_balanced_truth.py
"""

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
spec01 = importlib.util.spec_from_file_location("e222main", HERE / "01_synthetic_truth.py")
m01 = importlib.util.module_from_spec(spec01)
spec01.loader.exec_module(m01)

base = m01.base
e219 = m01.e219
from scipy.spatial import cKDTree                     # noqa: E402
from scipy.stats import spearmanr                     # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402

RESULTS_DIR = HERE / "results"
FEAT = m01.FEAT
ALGOS = m01.ALGOS
CONFIGS = m01.CONFIGS
N_WORLDS = m01.N_WORLDS
N_EVAL_PRES = m01.N_EVAL_PRES
EVAL_RATIO = m01.EVAL_RATIO
TOP_FRAC = m01.TOP_FRAC
REGION_FACTOR = np.array([1.0, 0.4, 0.15, 0.05])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("E222 World D: region-balanced truth + regional survey bias")
    print("=" * 74)

    frame = base.build_frame()
    clay = base.sample_at_points(base.DEM_DIR / "jatim_clay.tif",
                                 frame[["x", "y"]].to_numpy())
    keep = np.isfinite(clay)
    frame = frame[keep].reset_index(drop=True)

    Xf = frame[FEAT].to_numpy(dtype=np.float64)
    xyf = frame[["x", "y"]].to_numpy()
    road = frame["road_dist"].to_numpy()
    zf = (Xf - Xf.mean(axis=0)) / Xf.std(axis=0)
    remote_mask = road >= np.quantile(road, 0.8)
    midx = 0.5 * (frame["x"].min() + frame["x"].max())
    midy = 0.5 * (frame["y"].min() + frame["y"].max())
    region = base.assign_regions(frame["x"].to_numpy(), frame["y"].to_numpy(), midx, midy)
    survey_p = (np.clip(np.exp(-road / m01.SURVEY_DECAY_M), m01.SURVEY_MIN_P, 1.0)
                * REGION_FACTOR[region])

    ll = sum(m01.BETA_A[c] * zf[:, FEAT.index(c)] for c in m01.BETA_A)
    lam = np.exp(ll - ll.max())
    # World D: rebalance so every quadrant carries equal TOTAL intensity (within-region logic intact)
    for r in range(4):
        m_r = region == r
        lam[m_r] *= 0.25 / lam[m_r].sum()
    p_lam = lam / lam.sum()
    k_top = int(len(frame) * TOP_FRAC)
    top_true = set(np.argpartition(-p_lam, k_top)[:k_top])

    rows = []
    for w in range(N_WORLDS):
        wrng = np.random.default_rng(30_000 + 997 * w)
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

        wdf, prop = m01.build_world_frame(frame, obs)
        tree = cKDTree(xyf[obs])
        d_site, _ = tree.query(xyf, k=1)
        frame_buf = wdf[d_site > base.SITE_BUFFER_M]
        n_pa = len(obs) * base.PSEUDOABSENCE_RATIO
        obs_region_share = np.bincount(region[obs], minlength=4) / len(obs)

        pres_df = pd.DataFrame(Xf[obs], columns=FEAT)
        pres_df["x"], pres_df["y"] = xyf[obs, 0], xyf[obs, 1]
        eval_df = pd.DataFrame(Xf[eval_idx], columns=FEAT)
        eval_df["x"], eval_df["y"] = xyf[eval_idx, 0], xyf[eval_idx, 1]
        avail_df = pd.DataFrame(Xf[avail_idx], columns=FEAT)
        avail_df["x"], avail_df["y"] = xyf[avail_idx, 0], xyf[avail_idx, 1]

        for ci, (cname, hf) in enumerate(CONFIGS):
            crng = np.random.default_rng(88_888 + 1000 * w + ci)
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
                X_all = tr[FEAT].to_numpy(dtype=np.float64)
                y_all = tr["presence"].to_numpy()
                pred = e219.predict_chunked(e219.fit_full(algo, X_all, y_all), Xf)
                top_pred = set(np.argpartition(-pred, k_top)[:k_top])
                jac = len(top_pred & top_true) / len(top_pred | top_true)
                rows.append({"world": w, "config": cname,
                             "hard_frac": hf if hf is not None else -1,
                             "algorithm": algo, "n_observed": len(obs),
                             "obs_region0_share": float(obs_region_share[0]),
                             "auc_own": float(np.mean(auc_own_l)),
                             "auc_true": float(np.mean(auc_true_l)),
                             "map_jaccard": jac,
                             "spearman_full": float(spearmanr(pred, p_lam).statistic),
                             "spearman_remote": float(
                                 spearmanr(pred[remote_mask], p_lam[remote_mask]).statistic)})
        print(f"  world {w + 1}/{N_WORLDS} done (observed={len(obs)}, "
              f"region0 share={obs_region_share[0]:.2f})")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e222d_runs.csv", index=False)

    pv = df.pivot_table(index=["world", "algorithm"], columns="config",
                        values=["auc_true", "map_jaccard"])
    h0_auc = (pv[("auc_true", "hybrid")] if ("auc_true", "hybrid") in pv else None)
    # hybrid column mixes hard_fracs; isolate hf=0.0 explicitly
    h0 = df[df.hard_frac == 0.0].set_index(["world", "algorithm"])
    rnd = df[df.config == "random"].set_index(["world", "algorithm"])
    tgb = df[df.config == "tgb"].set_index(["world", "algorithm"])
    d_auc = (h0["auc_true"] - rnd["auc_true"]).dropna()
    d_jac = (h0["map_jaccard"] - rnd["map_jaccard"]).dropna()
    d_auc_tgb = (tgb["auc_true"] - rnd["auc_true"]).dropna()

    out = {
        "experiment": "E222 World D", "n_worlds": N_WORLDS,
        "mean_obs_region0_share": round(float(df.groupby("world")["obs_region0_share"].first()
                                              .mean()), 4),
        "quota_vs_random_auc_true": {"mean": round(float(d_auc.mean()), 4),
                                     "frac_positive": round(float((d_auc > 0).mean()), 4)},
        "quota_vs_random_map_jaccard": {"mean": round(float(d_jac.mean()), 4),
                                        "frac_positive": round(float((d_jac > 0).mean()), 4)},
        "tgb_vs_random_auc_true": {"mean": round(float(d_auc_tgb.mean()), 4),
                                   "frac_positive": round(float((d_auc_tgb > 0).mean()), 4)},
        "mean_auc_true_by_config": {f"{c}|{h}": v for (c, h), v in
                                    df.groupby(["config", "hard_frac"])["auc_true"]
                                      .mean().round(4).to_dict().items()},
        "mean_map_jaccard_by_config": {f"{c}|{h}": v for (c, h), v in
                                       df.groupby(["config", "hard_frac"])["map_jaccard"]
                                         .mean().round(4).to_dict().items()},
    }
    out["quota_validated_in_balanced_truth"] = bool(
        out["quota_vs_random_auc_true"]["frac_positive"] >= 0.6
        or out["quota_vs_random_map_jaccard"]["frac_positive"] >= 0.6)
    with open(RESULTS_DIR / "e222d_outcome.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 74)
    print("PRE-REGISTERED FORK VERDICT")
    print("=" * 74)
    print(f"  observed region-0 share (survey-driven): {out['mean_obs_region0_share']:.2f}")
    print(f"  quota(hf=0.0) - random, auc_true   : {out['quota_vs_random_auc_true']}")
    print(f"  quota(hf=0.0) - random, map_jaccard: {out['quota_vs_random_map_jaccard']}")
    print(f"  QUOTA VALIDATED IN BALANCED TRUTH  : {out['quota_validated_in_balanced_truth']}")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
