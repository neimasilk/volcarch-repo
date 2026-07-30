"""
E218: Is the evaluation artefact found in E217 real, and what drives it?

E217 concluded that P2's reported AUC ladder is an artefact of scoring each background design against
negatives it selected for itself. That is a demolition of the paper's central claim, so before it is
carried into a manuscript -- or an email to the editor -- it has to survive the same adversarial
treatment the paper got. Pre-registration: DESIGN.md (written before this ran).

Four stages, each targeting one threat to the E217 conclusion:

  A (T1,T2,T3) DECISIVE: repeat the design comparison under FOUR fixed evaluation backgrounds
               (uniform / tgb-drawn / hybrid-drawn / spatially stratified), 20 seeds, and three
               metrics (AUC, TSS, continuous Boyce index).
               Prediction if E217 is right: the hybrid design wins ONLY when evaluated against
               hybrid-like negatives. A design that wins only on its own turf is not a better model.
               Prediction if the PAPER is right: hybrid wins under all four. That outcome is
               pre-registered as "E217 was wrong".

  B (T4)       Block-size sensitivity: ~40 / ~50 / ~60 km, matching the paper's own protocol.

  C (T6)       Mechanism: sweep backgrounds over CONTROLLED environmental dissimilarity and test
               whether AUC inflation is a function of it. Turns an asserted mechanism into a measured one.

  D (T5)       Decimation check: repeat one configuration on a ~150 m lattice instead of ~300 m,
               confirming the sampling frame is not doing the work.

Run from repo root (long; use a background run):
    py experiments/E218_evaluation_artefact/01_artefact_robustness.py
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

from scipy.spatial import cKDTree            # noqa: E402
from scipy.stats import spearmanr            # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

RESULTS_DIR = HERE / "results"
FEAT = base.FULL_COLS
ALGOS = base.ALGORITHMS
TRAIN_DESIGNS = ["random", "tgb", "hybrid"]
EVAL_KINDS = ["uniform", "tgb", "hybrid", "stratified"]

N_SEEDS_MAIN = 20          # matches the paper's own robustness protocol
N_SEEDS_AUX = 5
EVAL_RATIO = 5
BLOCK_SIZES_DEG = {"~40km": 0.36, "~50km": 0.45, "~60km": 0.54}
ZDIST_TARGETS = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25]


# ---------------------------------------------------------------- metrics

def tss_score(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def boyce_index(pred_presence: np.ndarray, pred_available: np.ndarray, n_windows: int = 101) -> float:
    """Continuous Boyce index (Hirzel et al. 2006).

    Presence-only: measures whether predicted suitability ranks environments in proportion to how much
    more often presences occur there than availability alone would predict. Included because AUC is the
    instrument under dispute -- condemning AUC using only AUC would be circular.
    """
    lo = float(min(pred_presence.min(), pred_available.min()))
    hi = float(max(pred_presence.max(), pred_available.max()))
    if hi <= lo:
        return float("nan")
    width = (hi - lo) / 10.0
    starts = np.linspace(lo, hi - width, n_windows)
    mids, ratios = [], []
    for s in starts:
        e = s + width
        n_p = int(((pred_presence >= s) & (pred_presence < e)).sum())
        n_a = int(((pred_available >= s) & (pred_available < e)).sum())
        if n_a == 0:
            continue
        f = n_p / len(pred_presence)
        expected = n_a / len(pred_available)
        mids.append(0.5 * (s + e))
        ratios.append(f / expected)
    if len(mids) < 5 or np.allclose(ratios, ratios[0]):
        return float("nan")
    return float(spearmanr(mids, ratios).statistic)


# ---------------------------------------------------------------- evaluation

def evaluate(pres, train_bg, eval_bg, availability_bg, algo, block_deg):
    """Train on presences + design background; score on presences + FIXED evaluation background.

    Boyce always uses `availability_bg` (the uniform draw) as the availability sample, so it is
    comparable across evaluation-background choices by construction.
    """
    tr = pd.concat([pres.assign(presence=1), train_bg.assign(presence=0)], ignore_index=True)
    te = pd.concat([pres.assign(presence=1), eval_bg.assign(presence=0)], ignore_index=True)

    tr_b = base.assign_blocks(tr["x"].to_numpy(), tr["y"].to_numpy(), block_deg)
    te_b = base.assign_blocks(te["x"].to_numpy(), te["y"].to_numpy(), block_deg)
    av_b = base.assign_blocks(availability_bg["x"].to_numpy(), availability_bg["y"].to_numpy(), block_deg)

    uniq = np.unique(np.concatenate([tr_b, te_b]))
    uniq.sort()

    aucs, tsss, boyces = [], [], []
    for test_blocks in np.array_split(uniq, base.N_FOLDS):
        tr_m = ~np.isin(tr_b, test_blocks)
        te_m = np.isin(te_b, test_blocks)
        av_m = np.isin(av_b, test_blocks)
        y_tr = tr["presence"].to_numpy()[tr_m]
        y_te = te["presence"].to_numpy()[te_m]
        if y_tr.sum() == 0 or y_te.sum() == 0 or y_te.sum() == len(y_te):
            continue

        X_tr = tr.loc[tr_m, FEAT].to_numpy(dtype=np.float64)
        X_te = te.loc[te_m, FEAT].to_numpy(dtype=np.float64)

        # Fit once, predict on the test fold and the availability sample together, then split --
        # fitting twice per fold would double the cost of Stage A for no gain.
        want_boyce = av_m.sum() > 50
        X_av = (availability_bg.loc[av_m, FEAT].to_numpy(dtype=np.float64)
                if want_boyce else np.empty((0, len(FEAT))))
        p_all = np.asarray(base.fit_predict(algo, X_tr, y_tr, np.vstack([X_te, X_av]))).ravel()
        p_te, p_av = p_all[:len(X_te)], p_all[len(X_te):]

        aucs.append(roc_auc_score(y_te, p_te))
        tsss.append(tss_score(y_te, p_te))
        if want_boyce:
            b = boyce_index(p_te[y_te == 1], p_av)
            if np.isfinite(b):
                boyces.append(b)

    return (float(np.mean(aucs)) if aucs else np.nan,
            float(np.mean(tsss)) if tsss else np.nan,
            float(np.mean(boyces)) if boyces else np.nan)


# ---------------------------------------------------------------- setup

def prepare(decimate: int):
    """Build frame + presences + design-drawing closures at a given lattice resolution."""
    old = base.DECIMATE
    base.DECIMATE = decimate
    frame_all = base.build_frame()
    base.DECIMATE = old

    sites = base.load_sites()
    xy = np.column_stack([sites.geometry.x, sites.geometry.y])
    pres = pd.DataFrame({c: base.sample_at_points(base.DEM_DIR / base.RASTER_FILES[c], xy)
                         for c in FEAT})
    pres["x"], pres["y"] = xy[:, 0], xy[:, 1]
    pres = pres.dropna(subset=FEAT)
    pres = pres[pres["elevation"] > 0].reset_index(drop=True)

    tree = cKDTree(pres[["x", "y"]].to_numpy())
    d, _ = tree.query(frame_all[["x", "y"]].to_numpy(), k=1)
    frame_buf = frame_all[d > base.SITE_BUFFER_M].reset_index(drop=True)

    mu = pres[FEAT].mean().to_numpy(dtype=np.float64)
    sd = pres[FEAT].std().replace(0, 1.0).to_numpy(dtype=np.float64)
    midx = 0.5 * (frame_all["x"].min() + frame_all["x"].max())
    midy = 0.5 * (frame_all["y"].min() + frame_all["y"].max())

    def decorate(df):
        df = df.copy()
        z = (df[FEAT].to_numpy(dtype=np.float64) - mu) / sd
        df["zdist"] = np.sqrt((z ** 2).sum(axis=1))
        df["region_id"] = base.assign_regions(df["x"].to_numpy(), df["y"].to_numpy(), midx, midy)
        return df

    frame_all, frame_buf = decorate(frame_all), decorate(frame_buf)
    pres_reg = base.assign_regions(pres["x"].to_numpy(), pres["y"].to_numpy(), midx, midy)
    prop = np.bincount(pres_reg, minlength=4).astype(float)
    prop /= prop.sum()

    n_pa = len(pres) * base.PSEUDOABSENCE_RATIO
    designs = {
        "random": lambda rng: base.draw_random(frame_buf, n_pa, rng),
        "tgb": lambda rng: base.draw_tgb(frame_buf, n_pa, rng),
        "hybrid": lambda rng: base.draw_hybrid(frame_buf, n_pa, prop, rng),
    }
    return pres[FEAT + ["x", "y"]], frame_all, frame_buf, designs, prop, n_pa


def draw_stratified(frame, n, rng, cell_m=5000.0):
    """Systematic sample: one random cell per ~5 km grid square, then thin to n."""
    order = rng.permutation(len(frame))
    x, y = frame["x"].to_numpy()[order], frame["y"].to_numpy()[order]
    key = (x // cell_m).astype(np.int64) * 100_000 + (y // cell_m).astype(np.int64)
    first = ~pd.Series(key).duplicated().to_numpy()   # one cell per ~5 km square
    picks = order[first]
    if len(picks) > n:
        picks = rng.choice(picks, size=n, replace=False)
    return frame.iloc[picks]


def draw_zdist_band(frame, n, target, rng, half_width=0.25):
    band = frame[(frame["zdist"] >= target - half_width) & (frame["zdist"] <= target + half_width)]
    if len(band) < n:
        return band if len(band) > 200 else None
    return band.iloc[rng.choice(len(band), size=n, replace=False)]


# ---------------------------------------------------------------- stages

def stage_a(pres, frame_all, frame_buf, designs, prop, n_pa):
    print("\n" + "=" * 74)
    print("STAGE A (DECISIVE) — 3 training designs x 4 fixed evaluation backgrounds x 20 seeds")
    print("=" * 74)
    n_eval = len(pres) * EVAL_RATIO
    rows = []
    for s in range(N_SEEDS_MAIN):
        rng = np.random.default_rng(base.RANDOM_SEED + 1000 * s)
        evals = {
            "uniform": base.draw_random(frame_all, n_eval, rng),
            "tgb": base.draw_tgb(frame_all, n_eval, rng),
            "hybrid": base.draw_hybrid(frame_all, n_eval, prop, rng),
            "stratified": draw_stratified(frame_all, n_eval, rng),
        }
        availability = evals["uniform"]
        trains = {d: f(rng) for d, f in designs.items()}
        for d in TRAIN_DESIGNS:
            for ek in EVAL_KINDS:
                for algo in ALGOS:
                    auc, tss_, boyce = evaluate(pres, trains[d], evals[ek], availability,
                                                algo, base.BLOCK_SIZE_DEG)
                    rows.append({"seed": s, "train_design": d, "eval_background": ek,
                                 "algorithm": algo, "auc": auc, "tss": tss_, "boyce": boyce})
        print(f"  seed {s + 1}/{N_SEEDS_MAIN} done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e218_stageA_raw.csv", index=False)
    return df


def stage_b(pres, frame_all, frame_buf, designs, n_pa):
    print("\n" + "=" * 74)
    print("STAGE B — block-size sensitivity (~40 / ~50 / ~60 km), uniform evaluation background")
    print("=" * 74)
    n_eval = len(pres) * EVAL_RATIO
    rows = []
    for s in range(N_SEEDS_AUX):
        rng = np.random.default_rng(base.RANDOM_SEED + 1000 * s)
        ev = base.draw_random(frame_all, n_eval, rng)
        trains = {d: f(rng) for d, f in designs.items()}
        for label, bdeg in BLOCK_SIZES_DEG.items():
            for d in TRAIN_DESIGNS:
                for algo in ALGOS:
                    auc, tss_, boyce = evaluate(pres, trains[d], ev, ev, algo, bdeg)
                    rows.append({"seed": s, "block": label, "train_design": d,
                                 "algorithm": algo, "auc": auc, "tss": tss_, "boyce": boyce})
        print(f"  seed {s + 1}/{N_SEEDS_AUX} done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e218_stageB_blocksize.csv", index=False)
    return df


def stage_c(pres, frame_all, frame_buf):
    print("\n" + "=" * 74)
    print("STAGE C — mechanism: does AUC inflation track background environmental dissimilarity?")
    print("=" * 74)
    n_pa = len(pres) * base.PSEUDOABSENCE_RATIO
    n_eval = len(pres) * EVAL_RATIO
    rows = []
    for s in range(N_SEEDS_AUX):
        rng = np.random.default_rng(base.RANDOM_SEED + 1000 * s)
        ev = base.draw_random(frame_all, n_eval, rng)
        for target in ZDIST_TARGETS:
            bg = draw_zdist_band(frame_buf, n_pa, target, rng)
            if bg is None:
                print(f"  seed {s}: zdist target {target} — insufficient cells, skipped")
                continue
            realized = float(bg["zdist"].mean())
            for algo in ALGOS:
                auc_own, _, _ = evaluate(pres, bg, bg, ev, algo, base.BLOCK_SIZE_DEG)
                auc_com, _, _ = evaluate(pres, bg, ev, ev, algo, base.BLOCK_SIZE_DEG)
                rows.append({"seed": s, "zdist_target": target, "zdist_realized": realized,
                             "algorithm": algo, "auc_own": auc_own, "auc_common": auc_com,
                             "inflation": auc_own - auc_com, "n_bg": len(bg)})
        print(f"  seed {s + 1}/{N_SEEDS_AUX} done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e218_stageC_dissimilarity.csv", index=False)
    return df


def stage_d():
    print("\n" + "=" * 74)
    print("STAGE D — decimation check (~150 m lattice instead of ~300 m)")
    print("=" * 74)
    pres, frame_all, frame_buf, designs, prop, n_pa = prepare(decimate=5)
    n_eval = len(pres) * EVAL_RATIO
    rows = []
    for s in range(N_SEEDS_AUX):
        rng = np.random.default_rng(base.RANDOM_SEED + 1000 * s)
        ev = base.draw_random(frame_all, n_eval, rng)
        trains = {d: f(rng) for d, f in designs.items()}
        for d in TRAIN_DESIGNS:
            auc_own, _, _ = evaluate(pres, trains[d], trains[d], ev, "xgboost", base.BLOCK_SIZE_DEG)
            auc_com, _, _ = evaluate(pres, trains[d], ev, ev, "xgboost", base.BLOCK_SIZE_DEG)
            rows.append({"seed": s, "train_design": d, "auc_own": auc_own, "auc_common": auc_com})
        print(f"  seed {s + 1}/{N_SEEDS_AUX} done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e218_stageD_decimation.csv", index=False)
    return df


# ---------------------------------------------------------------- main

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("E218: robustness and mechanism of the evaluation artefact (E217 follow-up)")
    print("=" * 74)

    pres, frame_all, frame_buf, designs, prop, n_pa = prepare(decimate=base.DECIMATE)
    print(f"  presences={len(pres)}  frame={len(frame_all):,}  "
          f"frame_buffered={len(frame_buf):,}  background target={n_pa}")

    A = stage_a(pres, frame_all, frame_buf, designs, prop, n_pa)
    B = stage_b(pres, frame_all, frame_buf, designs, n_pa)
    C = stage_c(pres, frame_all, frame_buf)
    D = stage_d()

    out = {"experiment": "E218", "n_seeds_main": N_SEEDS_MAIN, "n_presences": int(len(pres))}

    # ---- Stage A verdict: does hybrid win only on its own turf? ----
    print("\n" + "=" * 74)
    print("STAGE A RESULT — mean AUC by training design x evaluation background")
    print("=" * 74)
    pa = A.pivot_table(index=["algorithm", "train_design"], columns="eval_background", values="auc")
    pa = pa[EVAL_KINDS]
    print(pa.round(3).to_string())
    pa.round(4).to_csv(RESULTS_DIR / "e218_stageA_auc_matrix.csv")

    for metric in ("auc", "tss", "boyce"):
        piv = A.pivot_table(index=["algorithm", "train_design"], columns="eval_background", values=metric)
        piv = piv[EVAL_KINDS]
        piv.round(4).to_csv(RESULTS_DIR / f"e218_stageA_{metric}_matrix.csv")
        out.setdefault("stageA", {})[metric] = {}
        for ek in EVAL_KINDS:
            col = piv[ek]
            wins = {}
            for algo in ALGOS:
                sub = col.loc[algo]
                wins[algo] = {
                    "random": round(float(sub["random"]), 4),
                    "tgb": round(float(sub["tgb"]), 4),
                    "hybrid": round(float(sub["hybrid"]), 4),
                    "hybrid_minus_random": round(float(sub["hybrid"] - sub["random"]), 4),
                    "hybrid_best": bool(sub["hybrid"] == sub.max()),
                }
            out["stageA"][metric][ek] = wins

    hybrid_wins = {
        ek: sum(out["stageA"]["auc"][ek][a]["hybrid_best"] for a in ALGOS) for ek in EVAL_KINDS
    }
    out["stageA_hybrid_wins_by_eval_background"] = hybrid_wins
    out["artefact_confirmed"] = bool(
        hybrid_wins["hybrid"] > hybrid_wins["uniform"] and
        hybrid_wins["hybrid"] > hybrid_wins["stratified"]
    )
    out["paper_claim_survives"] = bool(all(v == len(ALGOS) for v in hybrid_wins.values()))

    # ---- Stage B ----
    print("\n" + "=" * 74)
    print("STAGE B RESULT — hybrid minus random (common eval) by block size")
    print("=" * 74)
    pb = B.pivot_table(index=["block", "algorithm"], columns="train_design", values="auc")
    gap_b = (pb["hybrid"] - pb["random"]).round(4)
    print(gap_b.to_string())
    out["stageB_hybrid_minus_random_by_block"] = {
        f"{k[0]}|{k[1]}": float(v) for k, v in gap_b.items()
    }

    # ---- Stage C ----
    print("\n" + "=" * 74)
    print("STAGE C RESULT — inflation vs background dissimilarity")
    print("=" * 74)
    cg = C.groupby("zdist_target").agg(
        zdist_realized=("zdist_realized", "mean"),
        auc_own=("auc_own", "mean"),
        auc_common=("auc_common", "mean"),
        inflation=("inflation", "mean")).round(4)
    print(cg.to_string())
    cg.to_csv(RESULTS_DIR / "e218_stageC_summary.csv")
    if len(C) > 5:
        r_inf = spearmanr(C["zdist_realized"], C["inflation"])
        r_com = spearmanr(C["zdist_realized"], C["auc_common"])
        r_own = spearmanr(C["zdist_realized"], C["auc_own"])
        out["stageC"] = {
            "spearman_dissimilarity_vs_inflation": round(float(r_inf.statistic), 4),
            "p_inflation": float(r_inf.pvalue),
            "spearman_dissimilarity_vs_auc_own": round(float(r_own.statistic), 4),
            "spearman_dissimilarity_vs_auc_common": round(float(r_com.statistic), 4),
            "p_common": float(r_com.pvalue),
        }
        out["mechanism_confirmed"] = bool(r_inf.statistic > 0.5 and r_inf.pvalue < 0.01)
        print(f"\n  Spearman(dissimilarity, inflation)  = {r_inf.statistic:+.3f} (p={r_inf.pvalue:.2e})")
        print(f"  Spearman(dissimilarity, auc_own)    = {r_own.statistic:+.3f}")
        print(f"  Spearman(dissimilarity, auc_common) = {r_com.statistic:+.3f} (p={r_com.pvalue:.2e})")

    # ---- Stage D ----
    print("\n" + "=" * 74)
    print("STAGE D RESULT — ~150 m lattice")
    print("=" * 74)
    pd_ = D.groupby("train_design")[["auc_own", "auc_common"]].mean().round(4)
    print(pd_.to_string())
    out["stageD"] = {d: {"auc_own": float(r["auc_own"]), "auc_common": float(r["auc_common"])}
                     for d, r in pd_.iterrows()}
    out["stageD_ladder_persists_on_own_bg"] = bool(
        pd_.loc["hybrid", "auc_own"] > pd_.loc["random", "auc_own"])
    out["stageD_ladder_flat_on_common_bg"] = bool(
        pd_.loc["hybrid", "auc_common"] - pd_.loc["random", "auc_common"] < 0.01)

    with open(RESULTS_DIR / "e218_outcome.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 74)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 74)
    print(f"  Hybrid ranked best, by evaluation background: {hybrid_wins}")
    print(f"    (out of {len(ALGOS)} algorithms each)")
    print(f"  ARTEFACT CONFIRMED (hybrid wins mainly on its own turf) : {out['artefact_confirmed']}")
    print(f"  PAPER'S CLAIM SURVIVES (hybrid wins under all evals)    : {out['paper_claim_survives']}")
    print(f"  MECHANISM CONFIRMED (inflation tracks dissimilarity)    : "
          f"{out.get('mechanism_confirmed')}")
    print(f"  Decimation check — ladder on own bg persists            : "
          f"{out['stageD_ladder_persists_on_own_bg']}")
    print(f"  Decimation check — ladder on common bg flat             : "
          f"{out['stageD_ladder_flat_on_common_bg']}")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
