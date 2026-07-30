"""
E220: Selection on the reported metric walks backwards (+ two robustness closures).

Co-author commissioned (review_package_20260727/05, Q3). Pre-registration: DESIGN.md.

  Part 1  hard_frac sweep, 20 seeds x 3 algorithms, one CV pass per configuration (fit once per fold,
          score BOTH the design's own background and the fixed common background on identical folds --
          a control improvement over E218b's two-pass design). Then simulate the selection rules a
          practitioner actually uses: argmax auc_own (the submitted manuscript's rule), argmax
          auc_common (honest rule, cross-fitted across seed halves), argmax Boyce (is the presence-only
          metric an honest selector? -- pre-registered fork P4).

  Part 2  Common evaluation background drawn from the BUFFERED frame (>2 km from any presence), closing
          the co-author probe that E217b/E218 drew evaluation negatives from the unbuffered frame.

  Part 3  Boyce recomputed at 3 window widths x 3 window counts on identical predictions, closing the
          probe that our "artefact-immune" metric has unexamined knobs of its own.

  Part 4  Wilcoxon signed-rank tests on every headline paired contrast (no new fits).

Run from repo root (long; use a background run):
    py experiments/E220_wrong_direction_selection/01_selection_and_robustness.py
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

from scipy.stats import spearmanr, wilcoxon                # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve      # noqa: E402

RESULTS_DIR = HERE / "results"
FEAT = base.FULL_COLS
ALGOS = base.ALGORITHMS
HARD_FRACS = [round(0.1 * i, 1) for i in range(11)]
N_SEEDS = 20
EVAL_RATIO = e218.EVAL_RATIO
E013_SUBGRID = [0.0, 0.15, 0.30]


def tss_score(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def boyce_param(pred_presence, pred_available, width_frac=0.1, n_windows=101):
    """Continuous Boyce index with explicit knobs (Part 3 sweeps them)."""
    lo = float(min(pred_presence.min(), pred_available.min()))
    hi = float(max(pred_presence.max(), pred_available.max()))
    if hi <= lo:
        return float("nan")
    width = (hi - lo) * width_frac
    starts = np.linspace(lo, hi - width, n_windows)
    mids, ratios = [], []
    for s in starts:
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


# ------------------------------------------------------------ Part 1 machinery

def evaluate_multi(pres, train_bg, eval_bg, algo, block_deg):
    """One CV pass: fit once per fold; score own-background AND common-background test sets.

    Folds are defined on the union of training and common-eval blocks, so own and common scores share
    identical partitions (E218b's two-pass version used marginally different ones).
    """
    tr = pd.concat([pres.assign(presence=1), train_bg.assign(presence=0)], ignore_index=True)
    te = pd.concat([pres.assign(presence=1), eval_bg.assign(presence=0)], ignore_index=True)

    b_tr = base.assign_blocks(tr["x"].to_numpy(), tr["y"].to_numpy(), block_deg)
    b_te = base.assign_blocks(te["x"].to_numpy(), te["y"].to_numpy(), block_deg)
    uniq = np.unique(np.concatenate([b_tr, b_te]))
    uniq.sort()

    auc_own, auc_com, tss_com, boyces = [], [], [], []
    for test_blocks in np.array_split(uniq, base.N_FOLDS):
        m_tr = ~np.isin(b_tr, test_blocks)
        m_own = np.isin(b_tr, test_blocks)
        m_te = np.isin(b_te, test_blocks)
        y_tr = tr["presence"].to_numpy()[m_tr]
        y_own = tr["presence"].to_numpy()[m_own]
        y_te = te["presence"].to_numpy()[m_te]
        if (y_tr.sum() == 0 or y_own.sum() in (0, len(y_own)) or y_te.sum() in (0, len(y_te))):
            continue
        X_tr = tr.loc[m_tr, FEAT].to_numpy(dtype=np.float64)
        X_own = tr.loc[m_own, FEAT].to_numpy(dtype=np.float64)
        X_te = te.loc[m_te, FEAT].to_numpy(dtype=np.float64)
        p = np.asarray(base.fit_predict(algo, X_tr, y_tr, np.vstack([X_own, X_te]))).ravel()
        p_own, p_te = p[:len(X_own)], p[len(X_own):]
        auc_own.append(roc_auc_score(y_own, p_own))
        auc_com.append(roc_auc_score(y_te, p_te))
        tss_com.append(tss_score(y_te, p_te))
        # Boyce: presences of the common test fold vs the eval background as availability
        # (p_te rows align with te[m_te], so the availability mask is y_te == 0)
        if (y_te == 0).sum() > 50:
            b = boyce_param(p_te[y_te == 1], p_te[y_te == 0])
            if np.isfinite(b):
                boyces.append(b)
    return (float(np.mean(auc_own)), float(np.mean(auc_com)),
            float(np.mean(tss_com)), float(np.mean(boyces)) if boyces else float("nan"))


def part1(pres, frame_all, frame_buf, prop, n_pa):
    print("\n" + "=" * 74)
    print("PART 1 — hard_frac sweep, 20 seeds x 3 algorithms, dual scoring per fold")
    print("=" * 74)
    n_eval = len(pres) * EVAL_RATIO
    rows = []
    original = base.HYBRID_HARD_FRAC
    try:
        for s in range(N_SEEDS):
            rng = np.random.default_rng(base.RANDOM_SEED + 1000 * s)
            ev = base.draw_random(frame_all, n_eval, rng)
            for hf in HARD_FRACS:
                base.HYBRID_HARD_FRAC = hf
                bg = base.draw_hybrid(frame_buf, n_pa, prop, rng)
                if len(bg) < int(0.9 * n_pa):
                    continue
                zmean = float(bg["zdist"].mean())
                for algo in ALGOS:
                    ao, ac, ts, bo = evaluate_multi(pres, bg, ev, algo, base.BLOCK_SIZE_DEG)
                    rows.append({"seed": s, "hard_frac": hf, "zdist_mean": zmean, "algorithm": algo,
                                 "auc_own": ao, "auc_common": ac, "tss_common": ts, "boyce": bo})
            print(f"  seed {s + 1}/{N_SEEDS} done")
    finally:
        base.HYBRID_HARD_FRAC = original
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e220_sweep.csv", index=False)
    return df


def analyse_selection(df):
    """Simulate the three selection rules per (seed, algorithm)."""
    rows = []
    for (s, algo), g in df.groupby(["seed", "algorithm"]):
        g = g.sort_values("hard_frac")
        hf = g["hard_frac"].to_numpy()
        own, com, bo = g["auc_own"].to_numpy(), g["auc_common"].to_numpy(), g["boyce"].to_numpy()
        pick_own = hf[int(np.argmax(own))]
        pick_com = hf[int(np.argmax(com))]
        pick_bo = hf[int(np.nanargmax(bo))] if np.isfinite(bo).any() else float("nan")
        com_at = lambda h: float(com[list(hf).index(h)])
        rows.append({"seed": s, "algorithm": algo,
                     "pick_own": pick_own, "pick_common_ins": pick_com, "pick_boyce": pick_bo,
                     "auc_common_at_pick_own": com_at(pick_own),
                     "auc_common_at_pick_common_ins": com_at(pick_com),
                     "auc_common_at_pick_boyce": com_at(pick_bo) if np.isfinite(pick_bo) else np.nan,
                     "auc_common_min": float(com.min()), "auc_common_max": float(com.max())})
    sel = pd.DataFrame(rows)
    sel.to_csv(RESULTS_DIR / "e220_selection_by_seed.csv", index=False)

    # Cross-fitted honest evaluation of R-own and R-common: select on one seed half, score on the other.
    cf_rows = []
    for algo in ALGOS:
        a = df[df.algorithm == algo]
        for half_a, half_b in ((range(0, 10), range(10, 20)), (range(10, 20), range(0, 10))):
            ga = a[a.seed.isin(half_a)].groupby("hard_frac")[["auc_own", "auc_common"]].mean()
            gb = a[a.seed.isin(half_b)].groupby("hard_frac")["auc_common"].mean()
            hf_own = ga["auc_own"].idxmax()
            hf_com = ga["auc_common"].idxmax()
            cf_rows.append({"algorithm": algo, "hf_picked_own": hf_own, "hf_picked_common": hf_com,
                            "auc_common_pick_own": float(gb[hf_own]),
                            "auc_common_pick_common": float(gb[hf_com]),
                            "cost": float(gb[hf_com] - gb[hf_own])})
    cf = pd.DataFrame(cf_rows)
    cf.to_csv(RESULTS_DIR / "e220_selection_crossfitted.csv", index=False)
    return sel, cf


# ------------------------------------------------------------ Part 2 / 3

def part2(pres, frame_buf, designs, n_pa):
    print("\n" + "=" * 74)
    print("PART 2 — common evaluation background drawn from the BUFFERED frame")
    print("=" * 74)
    n_eval = len(pres) * EVAL_RATIO
    rows = []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(base.RANDOM_SEED + 1000 * s)
        ev = base.draw_random(frame_buf, n_eval, rng)          # <-- the only change vs Stage A
        trains = {d: f(rng) for d, f in designs.items()}
        for d in ["random", "tgb", "hybrid"]:
            for algo in ALGOS:
                auc, tss_, bo = e218.evaluate(pres, trains[d], ev, ev, algo, base.BLOCK_SIZE_DEG)
                rows.append({"seed": s, "train_design": d, "algorithm": algo,
                             "auc": auc, "tss": tss_, "boyce": bo})
        print(f"  seed {s + 1}/{N_SEEDS} done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e220_buffered_eval.csv", index=False)
    return df


def part3(pres, frame_all, frame_buf, designs, n_pa):
    print("\n" + "=" * 74)
    print("PART 3 — Boyce window sensitivity (same predictions, 9 window configs)")
    print("=" * 74)
    n_eval = len(pres) * EVAL_RATIO
    rows = []
    for s in range(5):
        rng = np.random.default_rng(base.RANDOM_SEED + 1000 * s)
        ev = base.draw_random(frame_all, n_eval, rng)
        trains = {d: f(rng) for d, f in designs.items()}
        for d in ["random", "tgb", "hybrid"]:
            tr = pd.concat([pres.assign(presence=1), trains[d].assign(presence=0)], ignore_index=True)
            te = pd.concat([pres.assign(presence=1), ev.assign(presence=0)], ignore_index=True)
            b_tr = base.assign_blocks(tr["x"].to_numpy(), tr["y"].to_numpy())
            b_te = base.assign_blocks(te["x"].to_numpy(), te["y"].to_numpy())
            uniq = np.unique(np.concatenate([b_tr, b_te]))
            uniq.sort()
            for algo in ALGOS:
                p_pres_all, p_av_all = [], []
                for test_blocks in np.array_split(uniq, base.N_FOLDS):
                    m_tr = ~np.isin(b_tr, test_blocks)
                    m_te = np.isin(b_te, test_blocks)
                    y_tr = tr["presence"].to_numpy()[m_tr]
                    y_te = te["presence"].to_numpy()[m_te]
                    if y_tr.sum() == 0 or y_te.sum() in (0, len(y_te)):
                        continue
                    p = base.fit_predict(algo, tr.loc[m_tr, FEAT].to_numpy(dtype=np.float64), y_tr,
                                         te.loc[m_te, FEAT].to_numpy(dtype=np.float64))
                    p = np.asarray(p).ravel()
                    p_pres_all.append(p[y_te == 1])
                    p_av_all.append(p[y_te == 0])
                pp, pa = np.concatenate(p_pres_all), np.concatenate(p_av_all)
                for wf in (0.05, 0.10, 0.20):
                    for nw in (51, 101, 201):
                        rows.append({"seed": s, "train_design": d, "algorithm": algo,
                                     "width_frac": wf, "n_windows": nw,
                                     "boyce": boyce_param(pp, pa, wf, nw)})
        print(f"  seed {s + 1}/5 done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e220_boyce_windows.csv", index=False)
    return df


# ------------------------------------------------------------ Part 4 (analysis only)

def part4_wilcoxon(sweep, sel):
    print("\n" + "=" * 74)
    print("PART 4 — Wilcoxon signed-rank on headline paired contrasts")
    print("=" * 74)
    rows = []

    def add(name, diffs):
        diffs = np.asarray(diffs, dtype=float)
        diffs = diffs[np.isfinite(diffs)]
        if len(diffs) < 5 or np.all(diffs == 0):
            return
        w = wilcoxon(diffs)
        rows.append({"contrast": name, "n_pairs": len(diffs), "median_diff": float(np.median(diffs)),
                     "mean_diff": float(diffs.mean()), "frac_positive": float((diffs > 0).mean()),
                     "wilcoxon_stat": float(w.statistic), "p_value": float(w.pvalue)})
        print(f"  {name:<62} n={len(diffs):<3} mean={diffs.mean():+.4f} p={w.pvalue:.2e}")

    # E218 Stage A: hybrid - random per algorithm x eval background (20 seeds)
    A = pd.read_csv(E218 / "results" / "e218_stageA_raw.csv")
    pv = A.pivot_table(index=["seed", "algorithm", "eval_background"], columns="train_design", values="auc")
    pv["diff"] = pv["hybrid"] - pv["random"]
    for (algo, ek), g in pv.reset_index().groupby(["algorithm", "eval_background"]):
        add(f"E218A hybrid-random AUC | {algo} | eval={ek}", g["diff"])

    # E220 sweep: auc_common at hard_frac 1.0 vs 0.0, paired per seed
    for algo in ALGOS:
        a = sweep[sweep.algorithm == algo].pivot_table(index="seed", columns="hard_frac",
                                                       values="auc_common")
        add(f"E220 auc_common hard_frac 1.0 - 0.0 | {algo}", a[1.0] - a[0.0])
        add(f"E220 auc_common hard_frac 0.3 - 0.0 (E013's pick) | {algo}", a[0.3] - a[0.0])

    # Selection cost per seed (in-sample variant; cross-fitted magnitude reported separately)
    add("E220 selection cost per seed (auc_common best - at R-own pick)",
        sel["auc_common_max"] - sel["auc_common_at_pick_own"])

    # E217b: feature gain (60 pairs) and evaluation inflation (15 pairs)
    r = pd.read_csv(E217 / "results" / "e217b_raw_results.csv")
    p = r.pivot_table(index=["seed", "design", "algorithm"], columns="feature_set",
                      values="auc_common_eval")
    add("E217b feature gain terrain_river - terrain (common eval)", p["terrain_river"] - p["terrain"])
    h = r[r.feature_set == "terrain_river"].pivot_table(
        index=["seed", "algorithm"], columns="design",
        values=["auc_common_eval", "auc_own_background"])
    infl = ((h[("auc_own_background", "hybrid")] - h[("auc_own_background", "random_nobuffer")])
            - (h[("auc_common_eval", "hybrid")] - h[("auc_common_eval", "random_nobuffer")]))
    add("E217b own-background inflation (hybrid-vs-random_nobuffer)", infl)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e220_wilcoxon.csv", index=False)
    return df


# ------------------------------------------------------------ main

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("E220: wrong-direction selection + robustness closures (pre-registration: DESIGN.md)")
    print("=" * 74)

    pres, frame_all, frame_buf, designs, prop, n_pa = e218.prepare(decimate=base.DECIMATE)
    print(f"  presences={len(pres)}  frame={len(frame_all):,}  buffered={len(frame_buf):,}  n_bg={n_pa}")

    sweep = part1(pres, frame_all, frame_buf, prop, n_pa)
    sel, cf = analyse_selection(sweep)
    buffered = part2(pres, frame_buf, designs, n_pa)
    windows = part3(pres, frame_all, frame_buf, designs, n_pa)
    part4_wilcoxon(sweep, sel)

    out = {"experiment": "E220", "n_seeds": N_SEEDS, "n_presences": int(len(pres))}

    # --- dose-response re-estimated at 20 seeds (E218b was 5)
    r_inf = spearmanr(sweep["zdist_mean"], sweep["auc_own"] - sweep["auc_common"])
    r_com = spearmanr(sweep["zdist_mean"], sweep["auc_common"])
    out["dose_response_20seed"] = {
        "spearman_dissimilarity_vs_inflation": round(float(r_inf.statistic), 4),
        "p_inflation": float(r_inf.pvalue),
        "spearman_dissimilarity_vs_auc_common": round(float(r_com.statistic), 4),
        "p_common": float(r_com.pvalue),
    }

    # --- selection verdicts (pre-registered P1-P4)
    n_cases = len(sel)
    p1 = float((sel["pick_own"] >= 0.7).mean())
    p2 = float(((sel["auc_common_at_pick_own"] - sel["auc_common_min"]) <= 0.01).mean())
    cost_cf = float(cf["cost"].mean())
    out["selection"] = {
        "frac_Rown_picks_hard_frac_ge_0.7": round(p1, 4),
        "frac_Rown_pick_worst_or_near_worst": round(p2, 4),
        "crossfitted_cost_mean": round(cost_cf, 4),
        "crossfitted_cost_by_algorithm": {
            a: round(float(cf[cf.algorithm == a]["cost"].mean()), 4) for a in ALGOS},
        "Rown_pick_distribution": sel["pick_own"].value_counts().sort_index().to_dict(),
        "Rboyce_pick_median": float(sel["pick_boyce"].median()),
        "Rcommon_insample_pick_median": float(sel["pick_common_ins"].median()),
        "P1_supported": bool(p1 >= 0.60),
        "P2_supported": bool(p2 >= 0.50),
        "P3_supported": bool(cost_cf >= 0.05),
        "P4_boyce_tracks_common": bool(abs(sel["pick_boyce"].median() - sel["pick_common_ins"].median())
                                        <= 0.2),
    }
    # E013 sub-grid: what the restricted sweep hid
    sub = sweep[sweep.hard_frac.isin(E013_SUBGRID)]
    full_own = sub.groupby(["seed", "algorithm"])["auc_own"].max()  # best within E013's offered grid
    out["e013_subgrid_note"] = {
        "own_auc_gain_within_subgrid": round(float(
            (sweep[sweep.hard_frac == 0.3]["auc_own"].mean())
            - (sweep[sweep.hard_frac == 0.0]["auc_own"].mean())), 4),
        "own_auc_gain_full_dial": round(float(
            (sweep[sweep.hard_frac == 1.0]["auc_own"].mean())
            - (sweep[sweep.hard_frac == 0.0]["auc_own"].mean())), 4),
    }

    # --- Part 2 verdict
    piv = buffered.pivot_table(index=["algorithm", "train_design"], values="auc")
    piv = piv.reset_index().pivot(index="algorithm", columns="train_design", values="auc")
    out["part2_buffered_eval"] = {a: {"hybrid_minus_random": round(float(piv.loc[a, "hybrid"]
                                                                        - piv.loc[a, "random"]), 4)}
                                  for a in ALGOS}
    out["part2_ranking_unchanged"] = bool(all(
        out["part2_buffered_eval"][a]["hybrid_minus_random"] <= 0 for a in ALGOS))

    # --- Part 3 verdict
    w = windows.pivot_table(index=["algorithm", "train_design", "width_frac", "n_windows"],
                            values="boyce")
    w = w.reset_index().pivot_table(index=["algorithm", "width_frac", "n_windows"],
                                    columns="train_design", values="boyce")
    w["diff"] = w["hybrid"] - w["random"]
    stab = w.reset_index().groupby("algorithm")["diff"].agg(
        n="size", frac_negative=lambda s: float((s < 0).mean()), mean="mean")
    out["part3_boyce_windows"] = {a: {"n_configs": int(stab.loc[a, "n"]),
                                      "frac_configs_negative": round(stab.loc[a, "frac_negative"], 4),
                                      "mean_diff": round(float(stab.loc[a, "mean"]), 4)}
                                  for a in ALGOS}
    out["part3_sign_stable"] = {a: bool(stab.loc[a, "frac_negative"] >= 8 / 9 or
                                        stab.loc[a, "frac_negative"] <= 1 / 9) for a in ALGOS}

    with open(RESULTS_DIR / "e220_outcome.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 74)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 74)
    s = out["selection"]
    print(f"  20-seed dose-response: inflation rho={out['dose_response_20seed']['spearman_dissimilarity_vs_inflation']:+.3f}"
          f"  common rho={out['dose_response_20seed']['spearman_dissimilarity_vs_auc_common']:+.3f}")
    print(f"  P1 R-own picks hard_frac>=0.7 : {s['frac_Rown_picks_hard_frac_ge_0.7']:.0%}  (need >=60%) -> {s['P1_supported']}")
    print(f"  P2 R-own pick (near-)worst    : {s['frac_Rown_pick_worst_or_near_worst']:.0%}  (need >=50%) -> {s['P2_supported']}")
    print(f"  P3 cross-fitted cost          : {s['crossfitted_cost_mean']:+.4f}    (need >=+0.05) -> {s['P3_supported']}")
    print(f"  P4 Boyce tracks common rule   : {s['P4_boyce_tracks_common']}  (median picks: "
          f"boyce={s['Rboyce_pick_median']}, common={s['Rcommon_insample_pick_median']})")
    print(f"  Part2 buffered eval hybrid-random: {out['part2_buffered_eval']}  unchanged={out['part2_ranking_unchanged']}")
    print(f"  Part3 Boyce sign stable       : {out['part3_sign_stable']}")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
