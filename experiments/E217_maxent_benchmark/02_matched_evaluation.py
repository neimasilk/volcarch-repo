"""
E217b: Corrected comparison — common evaluation background + site-buffer ablation.

WHY THIS SECOND SCRIPT EXISTS
-----------------------------
Run 01 produced a result that contradicts P2's headline claim: the gain from redesigning the
background (+0.022 AUC) was SMALLER than the gain from adding one feature (+0.045), and no algorithm
reproduced a monotonic random -> tgb -> hybrid ladder. Before that is reported as a refutation, two
confounds have to be ruled out, because either one would make run 01 (and arguably the original
E007-E013 ladder) measure the wrong thing:

  C1. NON-COMPARABLE AUCs. Each background design supplies its own negatives to the TEST fold, so a
      design that samples negatives from environments unlike the presences gets an easier test set and
      a higher AUC -- without the model being any better. AUC is not comparable across different
      background samples (Lobo et al. 2008, already cited in the manuscript as a caution and then not
      applied to the paper's own ladder). FIX: train on each design's own background, but evaluate
      every model against ONE common evaluation background, held fixed across designs.

  C2. SITE-BUFFER EXCLUSION. Run 01 excluded frame cells within 2 km of a presence from ALL designs,
      including `random`. E013's TGB pool does this; the earlier random-background experiments did not.
      That is a data-cleaning step, not a background-realism step, and it could account for a large
      part of the published E007 -> E013 gain. FIX: ablate it explicitly.

The corrected contrast is the one that belongs in the revision either way -- if the claim survives it is
now properly supported, and if it does not, the claim gets downgraded rather than reworded.

Run from repo root (after 01):
    py experiments/E217_maxent_benchmark/02_matched_evaluation.py
"""

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("e217base", HERE / "01_maxent_benchmark.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

from scipy.spatial import cKDTree  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

RESULTS_DIR = HERE / "results"
N_SEEDS = 5
EVAL_RATIO = 5          # common evaluation background size = EVAL_RATIO x n_presences
SITE_BUFFER_M = base.SITE_BUFFER_M


def tss(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def run_cv_matched(pres, train_bg, eval_bg, feat_cols, algo):
    """Train on presences + design background; evaluate on presences + COMMON background.

    Folds are defined on spatial blocks, so a test fold contains the presences and the common
    evaluation cells falling in those blocks, regardless of which design supplied training negatives.
    """
    tr_pool = pd.concat([pres.assign(presence=1), train_bg.assign(presence=0)], ignore_index=True)
    te_pool = pd.concat([pres.assign(presence=1), eval_bg.assign(presence=0)], ignore_index=True)

    tr_blocks = base.assign_blocks(tr_pool["x"].to_numpy(), tr_pool["y"].to_numpy())
    te_blocks = base.assign_blocks(te_pool["x"].to_numpy(), te_pool["y"].to_numpy())

    uniq = np.unique(np.concatenate([tr_blocks, te_blocks]))
    uniq.sort()

    aucs, tsss = [], []
    for test_blocks in np.array_split(uniq, base.N_FOLDS):
        tr_mask = ~np.isin(tr_blocks, test_blocks)
        te_mask = np.isin(te_blocks, test_blocks)
        y_te = te_pool["presence"].to_numpy()[te_mask]
        y_tr = tr_pool["presence"].to_numpy()[tr_mask]
        if y_te.sum() == 0 or y_te.sum() == len(y_te) or y_tr.sum() == 0:
            continue
        X_tr = tr_pool.loc[tr_mask, feat_cols].to_numpy(dtype=np.float64)
        X_te = te_pool.loc[te_mask, feat_cols].to_numpy(dtype=np.float64)
        p = base.fit_predict(algo, X_tr, y_tr, X_te)
        aucs.append(roc_auc_score(y_te, p))
        tsss.append(tss(y_te, p))
    return float(np.mean(aucs)), float(np.mean(tsss))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("E217b: common-evaluation-background comparison + site-buffer ablation")
    print("=" * 72)

    print("\nBuilding sampling frame...")
    frame_all = base.build_frame()

    sites = base.load_sites()
    xy = np.column_stack([sites.geometry.x, sites.geometry.y])
    pres = pd.DataFrame({c: base.sample_at_points(base.DEM_DIR / base.RASTER_FILES[c], xy)
                         for c in base.FULL_COLS})
    pres["x"], pres["y"] = xy[:, 0], xy[:, 1]
    pres = pres.dropna(subset=base.FULL_COLS)
    pres = pres[pres["elevation"] > 0].reset_index(drop=True)
    n_pa = len(pres) * base.PSEUDOABSENCE_RATIO
    print(f"  Presences: {len(pres)}   background target: {n_pa}   frame: {len(frame_all):,}")

    # Distance from every frame cell to nearest presence -> lets us build both ablation variants.
    tree = cKDTree(pres[["x", "y"]].to_numpy())
    dist_to_site, _ = tree.query(frame_all[["x", "y"]].to_numpy(), k=1)
    frame_all = frame_all.assign(dist_to_site=dist_to_site)
    frame_buf = frame_all[frame_all["dist_to_site"] > SITE_BUFFER_M].reset_index(drop=True)
    print(f"  Frame after {SITE_BUFFER_M:.0f} m site-buffer exclusion: {len(frame_buf):,}")

    mu = pres[base.FULL_COLS].mean().to_numpy(dtype=np.float64)
    sd = pres[base.FULL_COLS].std().replace(0, 1.0).to_numpy(dtype=np.float64)
    midx = 0.5 * (frame_all["x"].min() + frame_all["x"].max())
    midy = 0.5 * (frame_all["y"].min() + frame_all["y"].max())

    def decorate(df):
        df = df.copy()
        z = (df[base.FULL_COLS].to_numpy(dtype=np.float64) - mu) / sd
        df["zdist"] = np.sqrt((z ** 2).sum(axis=1))
        df["region_id"] = base.assign_regions(df["x"].to_numpy(), df["y"].to_numpy(), midx, midy)
        return df

    frame_all, frame_buf = decorate(frame_all), decorate(frame_buf)

    pres_regions = base.assign_regions(pres["x"].to_numpy(), pres["y"].to_numpy(), midx, midy)
    prop = np.bincount(pres_regions, minlength=4).astype(float)
    prop /= prop.sum()

    # Four training-background designs; the ablation pair differs ONLY in the site buffer.
    designs = {
        "random_nobuffer": lambda rng: base.draw_random(frame_all, n_pa, rng),
        "random_buffered": lambda rng: base.draw_random(frame_buf, n_pa, rng),
        "tgb":             lambda rng: base.draw_tgb(frame_buf, n_pa, rng),
        "hybrid":          lambda rng: base.draw_hybrid(frame_buf, n_pa, prop, rng),
    }

    rows = []
    for seed_i in range(N_SEEDS):
        rng = np.random.default_rng(base.RANDOM_SEED + 1000 * seed_i)
        # ONE common evaluation background per seed, uniform over the whole frame, drawn BEFORE the
        # training designs so it cannot be influenced by them.
        eval_bg = base.draw_random(frame_all, len(pres) * EVAL_RATIO, rng)
        print(f"\n--- seed {seed_i + 1}/{N_SEEDS} (common eval background n={len(eval_bg)}) ---")

        for dname, draw in designs.items():
            train_bg = draw(rng)
            for fs_name, feat_cols in base.FEATURE_SETS.items():
                for algo in base.ALGORITHMS:
                    auc_m, tss_m = run_cv_matched(
                        pres[base.FULL_COLS + ["x", "y"]], train_bg, eval_bg, feat_cols, algo)
                    auc_o, tss_o = base.run_cv(
                        pd.concat([pres[base.FULL_COLS + ["x", "y"]].assign(presence=1),
                                   train_bg[base.FULL_COLS + ["x", "y"]].assign(presence=0)],
                                  ignore_index=True), feat_cols, algo)
                    rows.append({
                        "seed": seed_i, "design": dname, "feature_set": fs_name, "algorithm": algo,
                        "auc_common_eval": auc_m, "tss_common_eval": tss_m,
                        "auc_own_background": auc_o, "tss_own_background": tss_o,
                    })
                    print(f"  {dname:<16} {fs_name:<13} {algo:<13} "
                          f"AUC(common)={auc_m:.3f}  AUC(own)={auc_o:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "e217b_raw_results.csv", index=False)

    summ = (res.groupby(["design", "feature_set", "algorithm"])
               .agg(auc_common=("auc_common_eval", "mean"), auc_common_sd=("auc_common_eval", "std"),
                    auc_own=("auc_own_background", "mean"), auc_own_sd=("auc_own_background", "std"),
                    tss_common=("tss_common_eval", "mean"))
               .reset_index())
    summ.to_csv(RESULTS_DIR / "e217b_summary.csv", index=False)

    order = ["random_nobuffer", "random_buffered", "tgb", "hybrid"]
    print("\n" + "=" * 72)
    print("AUC on COMMON evaluation background (comparable across designs)")
    print("=" * 72)
    pc = summ.pivot_table(index=["design", "feature_set"], columns="algorithm",
                          values="auc_common").reindex(order, level=0)
    print(pc.round(3).to_string())
    print("\n" + "=" * 72)
    print("AUC on OWN background (what the submitted paper reports — NOT comparable)")
    print("=" * 72)
    po = summ.pivot_table(index=["design", "feature_set"], columns="algorithm",
                          values="auc_own").reindex(order, level=0)
    print(po.round(3).to_string())
    pc.round(4).to_csv(RESULTS_DIR / "e217b_auc_common.csv")
    po.round(4).to_csv(RESULTS_DIR / "e217b_auc_own.csv")

    # --- Decomposition of the reported ladder gain, on the full feature set -----------------
    full = summ[summ["feature_set"] == "terrain_river"]
    out = {"experiment": "E217b", "n_seeds": N_SEEDS, "n_presences": int(len(pres)),
           "eval_background_n": int(len(pres) * EVAL_RATIO), "per_algorithm": {}}

    for algo in base.ALGORITHMS:
        a = full[full["algorithm"] == algo].set_index("design")
        com, own = a["auc_common"], a["auc_own"]
        out["per_algorithm"][algo] = {
            "common_eval": {d: round(float(com[d]), 4) for d in order},
            "own_background": {d: round(float(own[d]), 4) for d in order},
            "gain_site_buffer_only": round(float(com["random_buffered"] - com["random_nobuffer"]), 4),
            "gain_tgb_over_buffered_random": round(float(com["tgb"] - com["random_buffered"]), 4),
            "gain_hybrid_over_tgb": round(float(com["hybrid"] - com["tgb"]), 4),
            "gain_total_common": round(float(com["hybrid"] - com["random_nobuffer"]), 4),
            "gain_total_own_background": round(float(own["hybrid"] - own["random_nobuffer"]), 4),
            "monotonic_common": bool(com["random_nobuffer"] <= com["tgb"] <= com["hybrid"]),
        }

    tr = summ[summ["feature_set"] == "terrain"].set_index(["design", "algorithm"])["auc_common"]
    fr = summ[summ["feature_set"] == "terrain_river"].set_index(["design", "algorithm"])["auc_common"]
    out["mean_gain_from_adding_river_feature_common"] = round(float((fr - tr).mean()), 4)
    out["mean_gain_from_background_redesign_common"] = round(
        float(np.mean([out["per_algorithm"][a]["gain_total_common"] for a in base.ALGORITHMS])), 4)
    out["background_effect_exceeds_feature_effect"] = bool(
        out["mean_gain_from_background_redesign_common"] >
        out["mean_gain_from_adding_river_feature_common"])
    out["inflation_from_own_background_evaluation"] = round(
        float(np.mean([out["per_algorithm"][a]["gain_total_own_background"] -
                       out["per_algorithm"][a]["gain_total_common"] for a in base.ALGORITHMS])), 4)

    with open(RESULTS_DIR / "e217b_outcome.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 72)
    print("DECOMPOSITION OF THE LADDER GAIN (common evaluation, terrain+river)")
    print("=" * 72)
    for algo in base.ALGORITHMS:
        v = out["per_algorithm"][algo]
        print(f"\n  {algo}")
        print(f"    site-buffer exclusion alone : {v['gain_site_buffer_only']:+.4f}")
        print(f"    TGB over buffered random    : {v['gain_tgb_over_buffered_random']:+.4f}")
        print(f"    hybrid over TGB             : {v['gain_hybrid_over_tgb']:+.4f}")
        print(f"    TOTAL (common eval)         : {v['gain_total_common']:+.4f}")
        print(f"    TOTAL (own background)      : {v['gain_total_own_background']:+.4f}")
    print(f"\n  Mean background gain (common) : "
          f"{out['mean_gain_from_background_redesign_common']:+.4f}")
    print(f"  Mean feature gain (common)    : "
          f"{out['mean_gain_from_adding_river_feature_common']:+.4f}")
    print(f"  Background > feature          : {out['background_effect_exceeds_feature_effect']}")
    print(f"  Inflation from own-background evaluation: "
          f"{out['inflation_from_own_background_evaluation']:+.4f} AUC")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
