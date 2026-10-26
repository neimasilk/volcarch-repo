"""E236: the decision quantities of DESIGN.md section 5, computed the same way for the original results ("full",
378 presences, read from the original experiment folders) and for the re-runs ("clean", 161 presences).

The definitions are those of the G1 script (papers/P2_settlement_model/revision_ammo/verify_headline_numbers.py),
so "full" must reproduce the manuscript's numbers; that is checked before "clean" is read.

    python 03_summarise.py full
    python 03_summarise.py clean
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ALGOS = ["maxent", "xgboost", "randomforest"]


def paths(which: str) -> dict:
    if which == "full":
        return {"A": EXP / "E217_maxent_benchmark/results/e217b_raw_results.csv",
                "B": EXP / "E218_evaluation_artefact/results/e218_stageA_raw.csv",
                "C": EXP / "E218_evaluation_artefact/results/e218b_hardfrac_sweep.csv",
                "D_sets": EXP / "E221_seed_ensemble_stability/results/e221_priority_sets.csv",
                "D_turn": EXP / "E221_seed_ensemble_stability/results/e221_turnover_pairs.csv",
                "E": EXP / "E235_elevation_vs_volcanic_control/results/e235_outcome.json",
                "F": HERE / "results/F_disjoint_full.csv"}
    r = HERE / "results"
    return {"A": r / "A_e217b/e217b_raw_results.csv", "B": r / "B_e218a/e218_stageA_raw.csv",
            "C": r / "C_e218b/e218b_hardfrac_sweep.csv", "D_sets": r / "D_e221/e221_priority_sets.csv",
            "D_turn": r / "D_e221/e221_turnover_pairs.csv", "E": r / "E_e235/e235_outcome.json",
            "F": r / "F_disjoint_clean.csv"}


def verdict(holds: bool, fails: bool) -> str:
    return "HOLDS" if holds else ("FAILS" if fails else "MIXED")


def part_a(p):
    r = pd.read_csv(p)
    m = r.groupby(["design", "feature_set", "algorithm"])[["auc_common_eval", "auc_own_background"]].mean()
    algos = sorted(r.algorithm.unique())
    redesign = float(np.mean([m.loc[("hybrid", "terrain_river", a), "auc_common_eval"]
                              - m.loc[("random_nobuffer", "terrain_river", a), "auc_common_eval"] for a in algos]))
    redesign_own = float(np.mean([m.loc[("hybrid", "terrain_river", a), "auc_own_background"]
                                  - m.loc[("random_nobuffer", "terrain_river", a), "auc_own_background"] for a in algos]))
    feat = [m.loc[(d, "terrain_river", a), "auc_common_eval"] - m.loc[(d, "terrain", a), "auc_common_eval"]
            for d in r.design.unique() for a in algos]
    tgb_own = float(np.mean([m.loc[("tgb", "terrain_river", a), "auc_own_background"]
                             - m.loc[("random_nobuffer", "terrain_river", a), "auc_own_background"] for a in algos]))
    tgb_com = float(np.mean([m.loc[("tgb", "terrain_river", a), "auc_common_eval"]
                             - m.loc[("random_nobuffer", "terrain_river", a), "auc_common_eval"] for a in algos]))
    river, npos = float(np.mean(feat)), int(np.sum(np.array(feat) > 0))
    return {"redesign_common": redesign, "redesign_own": redesign_own, "river_common": river,
            "river_positive": f"{npos}/{len(feat)}", "tgb_minus_random_own": tgb_own,
            "tgb_minus_random_common": tgb_com,
            "verdict": verdict(redesign <= 0.010 and river > redesign and npos >= 9, redesign >= 0.020)}


def part_b(p):
    raw = pd.read_csv(p)
    wins = {}
    for bg, g in raw.groupby("eval_background"):
        mm = g.groupby(["train_design", "algorithm"])["auc"].mean().unstack(0)
        wins[bg] = int((mm["hybrid"] > mm["random"]).sum())
    piv = raw[raw.train_design == "hybrid"].pivot_table(index=["seed", "algorithm"], columns="eval_background",
                                                        values="auc")
    infl = (piv["hybrid"] - piv["uniform"]).dropna()
    pt = raw[raw.train_design == "tgb"].pivot_table(index=["seed", "algorithm"], columns="eval_background", values="auc")
    infl_t = (pt["tgb"] - pt["uniform"]).dropna()
    xh = raw[(raw.train_design == "hybrid") & (raw.algorithm == "xgboost") & (raw.eval_background == "uniform")]
    npos, mean = int((infl > 0).sum()), float(infl.mean())
    others_ok = all(wins[b] <= 1 for b in wins if b != "hybrid")
    return {"wins_hybrid_over_random": wins, "hybrid_inflation_positive": f"{npos}/{len(infl)}",
            "hybrid_inflation_mean": mean, "hybrid_inflation_range": [float(infl.min()), float(infl.max())],
            "tgb_inflation_mean": float(infl_t.mean()), "tgb_inflation_positive": f"{int((infl_t > 0).sum())}/{len(infl_t)}",
            "xgb_hybrid_on_uniform": float(xh["auc"].mean()),
            "verdict": verdict(npos >= 0.9 * len(infl) and mean >= 0.015 and others_ok,
                               npos < (40 / 60) * len(infl) or mean < 0.005)}


def part_c(p):
    sw = pd.read_csv(p)
    g = sw.groupby("hard_frac")[["auc_own", "auc_common"]].mean()
    arg = float(g["auc_own"].idxmax())
    d_own = float(g.loc[1.0, "auc_own"] - g.loc[0.0, "auc_own"])
    d_com = float(g.loc[1.0, "auc_common"] - g.loc[0.0, "auc_common"])
    return {"argmax_own": arg, "own_0_to_1": d_own, "common_0_to_1": d_com,
            "own_by_hardfrac": [round(float(x), 4) for x in g["auc_own"]],
            "common_by_hardfrac": [round(float(x), 4) for x in g["auc_common"]],
            "verdict": verdict(arg >= 0.9 and d_own > 0 and d_com < 0, arg <= 0.7)}


def part_d(p_sets, p_turn):
    s = pd.read_csv(p_sets)
    piv = s.pivot_table(index="algorithm", columns="set", values="sites_per_1000km2")
    ratio = (piv["robust"] / piv["contingent"]).to_dict()
    fringe_elev = s[s.set == "contingent"].set_index("algorithm")["elevation_median_m"].to_dict()
    t = pd.read_csv(p_turn)
    turn = 1 - t.groupby(["algorithm", "design"])["jaccard"].mean()
    return {"density_robust": piv["robust"].round(2).to_dict(), "density_contingent": piv["contingent"].round(2).to_dict(),
            "ratio": {k: round(v, 2) for k, v in ratio.items()}, "fringe_elevation_median_m": fringe_elev,
            "turnover_range": [float(turn.min()), float(turn.max())],
            "verdict": verdict(all(v > 1.5 for v in ratio.values()), any(v <= 1.0 for v in ratio.values()))}


def part_e(p):
    o = json.loads(Path(p).read_text(encoding="utf-8"))
    out = {}
    for dname, d in o["definitions"].items():
        x = d["algorithms"]["xgboost"]
        out[dname] = {"E_star_m": d["M1"]["E_star_m"],
                      "share_above_V": d["M1"].get("share_above_E_star_V"),
                      "share_above_N": d["M1"].get("share_above_E_star_N"),
                      "M2_summary": x.get("M2_summary"),
                      **{m: {"estimate": x["measures"][m]["estimate"], "ci95": x["measures"][m]["ci95"],
                             "verdict": x["measures"][m]["verdict"]}
                         for m in ["suit_random", "suit_tgb", "suit_hybrid"] if m in x["measures"]}}
    return out


def part_f(p):
    if not Path(p).exists():
        return {"missing": str(p)}
    f = pd.read_csv(p)
    j = f.groupby(["algorithm", "k"])["jaccard_mean"].mean().unstack()
    gain = (j[5] - j[1]).to_dict()
    return {"J_k1": j[1].round(3).to_dict(), "J_k5": j[5].round(3).to_dict(),
            "J_k5_by_design": f[f.k == 5].set_index(["algorithm", "design"])["jaccard_mean"].round(3).to_dict(),
            "verdict_averaging_helps": verdict(all(v >= 0.10 for v in gain.values()), False)}


def main(which):
    p = paths(which)
    s = {"presence_set": which, "A_e217b": part_a(p["A"]), "B_e218a": part_b(p["B"]), "C_e218b": part_c(p["C"]),
         "D_e221": part_d(p["D_sets"], p["D_turn"]), "E_e235": part_e(p["E"]), "F_disjoint": part_f(p["F"])}
    s["F_disjoint"]["J_k5_by_design"] = {f"{a}|{d}": v for (a, d), v in s["F_disjoint"].get("J_k5_by_design", {}).items()}
    out = HERE / "results" / f"summary_{which}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=1, default=float), encoding="utf-8")
    print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
