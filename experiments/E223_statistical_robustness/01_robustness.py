"""
E223: statistical robustness package for the revision's headline claims.

Pre-registration: DESIGN.md. Four components:
  A  Equivalence testing -- 95% CI on (hybrid - random, common eval, 20 seeds) vs the submitted
     manuscript's ladder gain (+0.092). Pure analysis of E218 Stage A.
  B  Spatial block bootstrap -- resample presence BLOCKS (30 replicates), fit in-bag, score out-of-bag
     against a fixed uniform evaluation background. Seeds measure pipeline noise; this measures
     sensitivity to the archaeological record itself.
  C  MaxEnt regularisation sensitivity -- beta_multiplier in {0.5,1.0,1.5,2.5,4.0} x 3 designs x
     10 seeds, common evaluation background.
  D  k* threshold sensitivity -- recompute E221's stabilisation at J >= 0.85 / 0.90 / 0.95.

Run from repo root (~25 min):
    py experiments/E223_statistical_robustness/01_robustness.py
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

E218 = HERE.parent / "E218_evaluation_artefact"
spec218 = importlib.util.spec_from_file_location("e218main", E218 / "01_artefact_robustness.py")
e218 = importlib.util.module_from_spec(spec218)
spec218.loader.exec_module(e218)

from scipy import stats                              # noqa: E402
from sklearn.metrics import roc_auc_score            # noqa: E402

RESULTS_DIR = HERE / "results"
FEAT = base.FULL_COLS
ALGOS = base.ALGORITHMS
PUBLISHED_LADDER_GAIN = 0.092       # 0.659 -> 0.751, the submitted manuscript's headline
N_BOOT = 30
BETAS = [0.5, 1.0, 1.5, 2.5, 4.0]
N_SEEDS_C = 10


def part_a():
    print("\nPART A — equivalence: 95% CI on hybrid - random (common evals) vs published +0.092")
    A = pd.read_csv(E218 / "results" / "e218_stageA_raw.csv")
    pv = A.pivot_table(index=["seed", "algorithm", "eval_background"], columns="train_design",
                       values="auc")
    pv["diff"] = pv["hybrid"] - pv["random"]
    rows = []
    for (algo, ek), g in pv.reset_index().groupby(["algorithm", "eval_background"]):
        d = g["diff"].to_numpy()
        m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        lo, hi = stats.t.interval(0.95, len(d) - 1, loc=m, scale=se)
        rows.append({"algorithm": algo, "eval_background": ek, "n": len(d),
                     "mean": round(m, 4), "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                     "excludes_published_+0.092": bool(hi < PUBLISHED_LADDER_GAIN)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e223a_equivalence_ci.csv", index=False)
    print(df[df.eval_background == "uniform"].to_string(index=False))
    return df


def part_b(pres, frame_all, frame_buf, designs, prop, n_pa):
    print(f"\nPART B — spatial block bootstrap, {N_BOOT} replicates x 3 designs x 3 algorithms")
    pb = base.assign_blocks(pres["x"].to_numpy(), pres["y"].to_numpy())
    uniq_blocks = np.unique(pb)
    rows = []
    rng = np.random.default_rng(202_607)
    n_eval = len(pres) * e218.EVAL_RATIO
    for b in range(N_BOOT):
        picked = rng.choice(uniq_blocks, size=len(uniq_blocks), replace=True)
        # keep multiplicity: a block picked twice contributes its sites twice (standard bootstrap)
        in_idx = np.concatenate([np.where(pb == blk)[0] for blk in picked])
        oob_mask_blocks = ~np.isin(pb, np.unique(picked))
        inbag = pres.iloc[in_idx].reset_index(drop=True)
        oob = pres[oob_mask_blocks].reset_index(drop=True)
        if len(oob) < 50 or len(inbag) < 100:
            continue
        ev = base.draw_random(frame_all, n_eval, rng)
        te = pd.concat([oob.assign(presence=1), ev.assign(presence=0)], ignore_index=True)
        y_te = te["presence"].to_numpy()
        X_te = te[FEAT].to_numpy(dtype=np.float64)
        trains = {d: f(rng) for d, f in designs.items()}
        for d in ["random", "tgb", "hybrid"]:
            tr = pd.concat([inbag.assign(presence=1), trains[d].assign(presence=0)],
                           ignore_index=True)
            X_tr = tr[FEAT].to_numpy(dtype=np.float64)
            y_tr = tr["presence"].to_numpy()
            for algo in ALGOS:
                p = np.asarray(base.fit_predict(algo, X_tr, y_tr, X_te)).ravel()
                rows.append({"replicate": b, "design": d, "algorithm": algo,
                             "n_oob": len(oob), "auc": float(roc_auc_score(y_te, p))})
        print(f"  replicate {b + 1}/{N_BOOT} done (in-bag={len(inbag)}, oob={len(oob)})")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e223b_block_bootstrap.csv", index=False)
    piv = df.pivot_table(index=["replicate", "algorithm"], columns="design", values="auc")
    piv["diff"] = piv["hybrid"] - piv["random"]
    summ = []
    for algo in ALGOS:
        d = piv["diff"].xs(algo, level="algorithm").to_numpy()
        lo, hi = np.percentile(d, [2.5, 97.5])
        summ.append({"algorithm": algo, "n_replicates": len(d), "mean": round(float(d.mean()), 4),
                     "ci_lo_pct": round(float(lo), 4), "ci_hi_pct": round(float(hi), 4),
                     "excludes_published_+0.092": bool(hi < PUBLISHED_LADDER_GAIN)})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(RESULTS_DIR / "e223b_bootstrap_summary.csv", index=False)
    print(sdf.to_string(index=False))
    return sdf


def _evaluate_maxent_beta(pres, train_bg, eval_bg, beta):
    """E218-style spatial-block CV on the common evaluation background, MaxEnt with beta_multiplier."""
    from elapid import MaxentModel
    tr = pd.concat([pres.assign(presence=1), train_bg.assign(presence=0)], ignore_index=True)
    te = pd.concat([pres.assign(presence=1), eval_bg.assign(presence=0)], ignore_index=True)
    b_tr = base.assign_blocks(tr["x"].to_numpy(), tr["y"].to_numpy())
    b_te = base.assign_blocks(te["x"].to_numpy(), te["y"].to_numpy())
    uniq = np.unique(np.concatenate([b_tr, b_te]))
    uniq.sort()
    aucs = []
    for test_blocks in np.array_split(uniq, base.N_FOLDS):
        m_tr, m_te = ~np.isin(b_tr, test_blocks), np.isin(b_te, test_blocks)
        y_tr, y_te = tr["presence"].to_numpy()[m_tr], te["presence"].to_numpy()[m_te]
        if y_tr.sum() == 0 or y_te.sum() in (0, len(y_te)):
            continue
        m = MaxentModel(feature_types=["linear", "hinge", "product"],
                        beta_multiplier=beta, transform="cloglog")
        m.fit(tr.loc[m_tr, FEAT].to_numpy(dtype=np.float64), y_tr)
        p = np.asarray(m.predict(te.loc[m_te, FEAT].to_numpy(dtype=np.float64))).ravel()
        aucs.append(roc_auc_score(y_te, p))
    return float(np.mean(aucs)) if aucs else float("nan")


def part_c(pres, frame_all, frame_buf, designs, n_pa):
    print(f"\nPART C — MaxEnt beta_multiplier sweep x 3 designs x {N_SEEDS_C} seeds (common eval)")
    rows = []
    n_eval = len(pres) * e218.EVAL_RATIO
    for s in range(N_SEEDS_C):
        rng = np.random.default_rng(base.RANDOM_SEED + 1000 * s)
        ev = base.draw_random(frame_all, n_eval, rng)
        trains = {d: f(rng) for d, f in designs.items()}
        for beta in BETAS:
            for d in ["random", "tgb", "hybrid"]:
                auc = _evaluate_maxent_beta(pres, trains[d], ev, beta)
                rows.append({"seed": s, "beta": beta, "design": d, "auc": auc})
        print(f"  seed {s + 1}/{N_SEEDS_C} done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e223c_maxent_beta.csv", index=False)
    piv = df.pivot_table(index=["beta", "seed"], columns="design", values="auc")
    piv["diff"] = piv["hybrid"] - piv["random"]
    summ = (piv["diff"].groupby("beta").agg(n="size", mean="mean",
                                            frac_positive=lambda s: float((s > 0).mean()))
            .reset_index())
    summ.to_csv(RESULTS_DIR / "e223c_beta_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))
    return summ


def part_d():
    print("\nPART D — k* at J >= 0.85 / 0.90 / 0.95")
    curve = pd.read_csv(HERE.parent / "E221_seed_ensemble_stability" / "results"
                        / "e221_stabilisation_curve.csv")
    rows = []
    for thr in (0.85, 0.90, 0.95):
        for (algo, d), g in curve.groupby(["algorithm", "design"]):
            hit = g[g.jaccard_mean >= thr]["k"]
            rows.append({"threshold": thr, "algorithm": algo, "design": d,
                         "kstar": int(hit.min()) if len(hit) else None})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e223d_kstar_thresholds.csv", index=False)
    print(df.pivot_table(index=["algorithm", "design"], columns="threshold", values="kstar",
                         aggfunc="first").to_string())
    return df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("E223: statistical robustness package (pre-registration: DESIGN.md)")
    print("=" * 74)

    pres, frame_all, frame_buf, designs, prop, n_pa = e218.prepare(decimate=base.DECIMATE)
    print(f"  presences={len(pres)}  frame={len(frame_all):,}  buffered={len(frame_buf):,}")

    a = part_a()
    b = part_b(pres, frame_all, frame_buf, designs, prop, n_pa)
    c = part_c(pres, frame_all, frame_buf, designs, n_pa)
    d = part_d()

    out = {
        "experiment": "E223", "published_ladder_gain": PUBLISHED_LADDER_GAIN,
        "A_all_cells_exclude_published": bool(a["excludes_published_+0.092"].all()),
        "A_uniform_rows": a[a.eval_background == "uniform"].to_dict("records"),
        "B_all_algorithms_exclude_published": bool(b["excludes_published_+0.092"].all()),
        "B_summary": b.to_dict("records"),
        "C_maxent_beta_diff_max": round(float(c["mean"].max()), 4),
        "C_all_betas_nonpositive": bool((c["mean"] <= 0).all()),
        "D_kstar_by_threshold": d.to_dict("records"),
    }
    with open(RESULTS_DIR / "e223_outcome.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "=" * 74)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 74)
    print(f"  A equivalence (all 12 algo x eval cells exclude +0.092): "
          f"{out['A_all_cells_exclude_published']}")
    print(f"  B bootstrap excludes +0.092 (all algorithms)           : "
          f"{out['B_all_algorithms_exclude_published']}")
    print(f"  C MaxEnt beta sweep, max hybrid-random mean diff       : "
          f"{out['C_maxent_beta_diff_max']:+.4f} (all non-positive: {out['C_all_betas_nonpositive']})")
    print(f"  D k* table written")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
