"""
E224: does target-group background work once the bias variable is IN the feature set?

E222 found TGB no better than a random background against synthetic ground truth, even though the
simulated survey bias was exactly TGB's home condition. Correction K4 proposed the reason: the model's
features are terrain + river distance only, so the bias factor s(x) is not representable in feature
space and TGB has nothing to cancel.

This experiment repeats E222's world A with two arms that differ in ONE thing -- whether `road_dist`
is a model feature. Everything else (worlds, seeds, background draws, CV, algorithms, metrics) is
E222's. The `no_road` arm must reproduce E222 or the run is void.

Pre-registration: DESIGN.md (committed before this file was executed).

Run from repo root:
    py experiments/E224_road_feature_tgb/01_road_feature_tgb.py [--worlds 10]
"""

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
E217 = HERE.parent / "E217_maxent_benchmark"
E219 = HERE.parent / "E219_map_divergence"
E222 = HERE.parent / "E222_synthetic_ground_truth"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load("e217base", E217 / "01_maxent_benchmark.py")
e219 = _load("e219main", E219 / "01_map_divergence.py")

from scipy.spatial import cKDTree                     # noqa: E402
from scipy.stats import spearmanr                     # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402

RESULTS_DIR = HERE / "results"

# --- E222 constants, reproduced verbatim so the control arm is a true replicate -----
FEAT_NO_ROAD = base.FULL_COLS                       # elevation, slope, twi, tri, aspect, river_dist
FEAT_WITH_ROAD = base.FULL_COLS + ["road_dist"]     # the single manipulation
ARMS = {"no_road": FEAT_NO_ROAD, "with_road": FEAT_WITH_ROAD}
ALGOS = base.ALGORITHMS
# ci indices must match E222's CONFIGS list so the config RNG streams are identical
CONFIGS = [(0, "random", None), (1, "tgb", None)]
N_WORLDS = 10
N_EVAL_PRES = 400
EVAL_RATIO = 5
TOP_FRAC = 0.10
BETA_A = {"elevation": -1.0, "slope": -0.8, "river_dist": -1.2, "twi": 0.6}
SURVEY_DECAY_M = 12000.0
SURVEY_MIN_P = 0.03


def build_world_frame(frame, obs_idx, feat_for_zdist):
    """E222's helper. zdist is always built on E222's feature set so that the background
    draws are byte-identical between arms -- only the MODEL's features change."""
    df = frame.copy()
    X = df[feat_for_zdist].to_numpy(dtype=np.float64)
    mu, sd = X[obs_idx].mean(axis=0), X[obs_idx].std(axis=0)
    sd[sd == 0] = 1.0
    df["zdist"] = np.sqrt((((X - mu) / sd) ** 2).sum(axis=1))
    midx = 0.5 * (df["x"].min() + df["x"].max())
    midy = 0.5 * (df["y"].min() + df["y"].max())
    df["region_id"] = base.assign_regions(df["x"].to_numpy(), df["y"].to_numpy(), midx, midy)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=N_WORLDS)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("E224: TGB with and without the bias variable in the feature set (K4 confirmation)")
    print("=" * 78)

    # --- real lattice, real covariates; only the archaeology is synthetic (E222 §A) ---
    frame = base.build_frame()
    clay = base.sample_at_points(base.DEM_DIR / "jatim_clay.tif", frame[["x", "y"]].to_numpy())
    keep = np.isfinite(clay)                 # kept for exact frame parity with E222 world A
    frame = frame[keep].reset_index(drop=True)
    print(f"  frame cells (clay-complete, as E222): {len(frame):,}")

    xyf = frame[["x", "y"]].to_numpy()
    road = frame["road_dist"].to_numpy()
    Xf_no = frame[FEAT_NO_ROAD].to_numpy(dtype=np.float64)
    Xf_with = frame[FEAT_WITH_ROAD].to_numpy(dtype=np.float64)
    zf = (Xf_no - Xf_no.mean(axis=0)) / Xf_no.std(axis=0)
    remote_mask = road >= np.quantile(road, 0.8)
    survey_p = np.clip(np.exp(-road / SURVEY_DECAY_M), SURVEY_MIN_P, 1.0)

    # DESIGN §5: is the manipulation clean, or is road_dist a proxy for an existing feature?
    coll = {c: float(np.corrcoef(road, frame[c].to_numpy(dtype=np.float64))[0, 1])
            for c in FEAT_NO_ROAD}
    print("  road_dist correlation with existing features: "
          + ", ".join(f"{c} {v:+.3f}" for c, v in coll.items()))

    ll = sum(BETA_A[c] * zf[:, FEAT_NO_ROAD.index(c)] for c in BETA_A)
    lam = np.exp(ll - ll.max())
    p_lam = lam / lam.sum()
    k_true = int(len(p_lam) * TOP_FRAC)
    top_true = set(np.argpartition(-p_lam, k_true)[:k_true])

    rows = []
    for w in range(args.worlds):
        wrng = np.random.default_rng(10_000 + 997 * w)          # E222 world seed
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

        wdf = build_world_frame(frame, obs, FEAT_NO_ROAD)
        d_site, _ = cKDTree(xyf[obs]).query(xyf, k=1)
        frame_buf = wdf[d_site > base.SITE_BUFFER_M]
        n_pa = len(obs) * base.PSEUDOABSENCE_RATIO

        for ci, cname, _hf in CONFIGS:
            crng = np.random.default_rng(77_777 + 1000 * w + ci)   # E222 config seed
            bg = base.draw_random(frame_buf, n_pa, crng) if cname == "random" \
                else base.draw_tgb(frame_buf, n_pa, crng)
            if len(bg) < int(0.9 * n_pa):
                print(f"  world {w}: {cname} background short ({len(bg)}/{n_pa}) - skipped")
                continue
            bg_idx = bg.index.to_numpy()

            for arm, feat in ARMS.items():
                Xf = Xf_with if arm == "with_road" else Xf_no
                pres_df = pd.DataFrame(Xf[obs], columns=feat)
                pres_df["x"], pres_df["y"] = xyf[obs, 0], xyf[obs, 1]
                bg_df = pd.DataFrame(Xf[bg_idx], columns=feat)
                bg_df["x"], bg_df["y"] = xyf[bg_idx, 0], xyf[bg_idx, 1]
                eval_df = pd.DataFrame(Xf[eval_idx], columns=feat)
                eval_df["x"], eval_df["y"] = xyf[eval_idx, 0], xyf[eval_idx, 1]
                avail_df = pd.DataFrame(Xf[avail_idx], columns=feat)
                avail_df["x"], avail_df["y"] = xyf[avail_idx, 0], xyf[avail_idx, 1]

                tr = pd.concat([pres_df.assign(presence=1), bg_df.assign(presence=0)],
                               ignore_index=True)[feat + ["x", "y", "presence"]]
                te = pd.concat([eval_df.assign(presence=1), avail_df.assign(presence=0)],
                               ignore_index=True)
                b_tr = base.assign_blocks(tr["x"].to_numpy(), tr["y"].to_numpy())
                b_te = base.assign_blocks(te["x"].to_numpy(), te["y"].to_numpy())
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
                        y_te = te["presence"].to_numpy()[m_te]
                        if (y_tr.sum() == 0 or y_own.sum() in (0, len(y_own))
                                or y_te.sum() in (0, len(y_te))):
                            continue
                        X_tr = tr.loc[m_tr, feat].to_numpy(dtype=np.float64)
                        X_own = tr.loc[m_own, feat].to_numpy(dtype=np.float64)
                        X_te = te.loc[m_te, feat].to_numpy(dtype=np.float64)
                        p = np.asarray(base.fit_predict(algo, X_tr, y_tr,
                                                        np.vstack([X_own, X_te]))).ravel()
                        auc_own_l.append(roc_auc_score(y_own, p[:len(X_own)]))
                        auc_true_l.append(roc_auc_score(y_te, p[len(X_own):]))

                    X_all = tr[feat].to_numpy(dtype=np.float64)
                    pred = e219.predict_chunked(
                        e219.fit_full(algo, X_all, tr["presence"].to_numpy()), Xf)
                    k = int(len(pred) * TOP_FRAC)
                    top_pred = set(np.argpartition(-pred, k)[:k])
                    rows.append({
                        "arm": arm, "world": w, "config": cname, "algorithm": algo,
                        "n_observed": len(obs),
                        "auc_own": float(np.mean(auc_own_l)),
                        "auc_true": float(np.mean(auc_true_l)),
                        "map_jaccard": len(top_pred & top_true) / len(top_pred | top_true),
                        "spearman_full": float(spearmanr(pred, p_lam).statistic),
                        "spearman_remote": float(
                            spearmanr(pred[remote_mask], p_lam[remote_mask]).statistic),
                    })
        print(f"  world {w + 1}/{args.worlds} done (observed={len(obs)})")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e224_runs.csv", index=False)

    # ---------------- pre-registered analysis (DESIGN §4) ----------------
    out = {"experiment": "E224", "n_worlds": args.worlds,
           "road_dist_collinearity": {k: round(v, 4) for k, v in coll.items()},
           "arms": {}}
    keys = ["world", "algorithm"]
    for arm in ARMS:
        a = df[df.arm == arm]
        t = a[a.config == "tgb"].set_index(keys)
        r = a[a.config == "random"].set_index(keys)
        arm_out = {}
        for metric in ("map_jaccard", "auc_true", "spearman_remote", "auc_own"):
            d = (t[metric] - r[metric]).dropna()
            arm_out[metric] = {"n_pairs": int(len(d)),
                               "mean": round(float(d.mean()), 4),
                               "median": round(float(d.median()), 4),
                               "frac_positive": round(float((d > 0).mean()), 4)}
        arm_out["levels"] = {m: {c: round(float(a[a.config == c][m].mean()), 4)
                                 for c in ("random", "tgb")}
                             for m in ("map_jaccard", "auc_true", "auc_own")}
        out["arms"][arm] = arm_out

    prim = out["arms"]["with_road"]["map_jaccard"]
    ctrl = out["arms"]["no_road"]["map_jaccard"]
    out["H_supported"] = bool(prim["mean"] > 0 and prim["frac_positive"] >= 0.60)
    out["control_arm_null_as_in_E222"] = bool(abs(ctrl["mean"]) < 0.02)

    # control-arm replication check (DESIGN §3): compare to E222 world A
    e222 = pd.read_csv(E222 / "results" / "e222_runs.csv")
    e222 = e222[(e222.surface == "A_observed") & (e222.config.isin(["random", "tgb"]))]
    mine = df[df.arm == "no_road"]
    m = e222.merge(mine, on=["world", "config", "algorithm"], suffixes=("_e222", "_e224"))
    rep = {}
    for metric in ("auc_true", "map_jaccard", "auc_own"):
        d = (m[f"{metric}_e224"] - m[f"{metric}_e222"]).abs()
        rep[metric] = {"n": int(len(d)), "max_abs_diff": round(float(d.max()), 6),
                       "mean_abs_diff": round(float(d.mean()), 6)}
    out["control_arm_replicates_E222"] = rep
    out["run_void"] = bool(rep["auc_true"]["max_abs_diff"] > 0.02
                           or rep["map_jaccard"]["max_abs_diff"] > 0.02)

    (RESULTS_DIR / "e224_outcome.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("PRE-REGISTERED READOUT")
    print("=" * 78)
    for arm in ARMS:
        p = out["arms"][arm]["map_jaccard"]
        q = out["arms"][arm]["auc_true"]
        print(f"  {arm:10s} TGB - random   map_jaccard {p['mean']:+.4f} "
              f"({p['frac_positive']:.0%} positive, n={p['n_pairs']})   "
              f"auc_true {q['mean']:+.4f} ({q['frac_positive']:.0%})")
    print(f"  control arm reproduces E222 : {not out['run_void']}  {rep}")
    print(f"  H SUPPORTED                 : {out['H_supported']}")
    print(f"\n  -> {RESULTS_DIR / 'e224_outcome.json'}")


if __name__ == "__main__":
    main()
