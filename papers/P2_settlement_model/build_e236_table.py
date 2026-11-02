"""E236 result table for the P2 supplement (round-2 revision, decision K1). 2026-11-02.

Data only, generated from experiments/E236_p2_presence_sensitivity/results/summary_{full,clean}.json.
Under decision K1 (PI, 2026-11-02) the paper reports both presence sets; this is the table behind that.
Output: supplement/e236_table_v03.tex (one table, \\input-ready; becomes Table S9).
    python papers/P2_settlement_model/build_e236_table.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "experiments" / "E236_p2_presence_sensitivity" / "results"
OUT = HERE / "supplement" / "e236_table_v03.tex"

S = {k: json.loads((RES / f"summary_{k}.json").read_text(encoding="utf-8")) for k in ("full", "clean")}


def extension_row():
    """Claim A on ten seeds, cleaned set (E236/EXTENSION_SEEDS_20261102.md). Computed, not retyped."""
    df = pd.read_csv(RES / "A_seeds10" / "e217b_raw_results.csv")
    df = df[df.feature_set == "terrain_river"]
    piv = df.pivot_table(index=["seed", "algorithm"], columns="design",
                         values="auc_common_eval", aggfunc="mean").reset_index()
    piv["delta"] = piv["hybrid"] - piv["random_nobuffer"]
    per_seed = piv.groupby("seed")["delta"].mean()
    seeds = list(per_seed.index)
    rng = np.random.default_rng(20261102)
    boot = [per_seed.loc[rng.choice(seeds, size=len(seeds), replace=True)].mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    first, second = per_seed.loc[sorted(seeds)[:5]], per_seed.loc[sorted(seeds)[5:]]
    return (f"{per_seed.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]; {(piv['delta'] > 0).sum()}/{len(piv)} cells; "
            f"disjoint five-seed halves {first.mean():+.3f} / {second.mean():+.3f} "
            f"({(first > 0).sum()}/{len(first)} and {(second > 0).sum()}/{len(second)} positive)")


def cell(lo, hi, spec):
    """Format the same quantity from both summaries; spec(block) -> str."""
    return spec(lo), spec(hi)


def f3(x, sign=True):
    return (f"{x:+.3f}" if sign else f"{x:.3f}")


def pct(x):
    return f"{100 * x:.1f}\\%"


rows = []


def add(quantity, full, clean, note=""):
    rows.append(" & ".join([quantity, full, clean]) + r" \\")


A = {k: S[k]["A_e217b"] for k in S}
B = {k: S[k]["B_e218a"] for k in S}
C = {k: S[k]["C_e218b"] for k in S}
D = {k: S[k]["D_e221"] for k in S}
F = {k: S[k]["F_disjoint"] for k in S}

add(r"\textit{Presences}", "378 records, 335 distinct coordinates", "161 sites (155 OSM, 6 Wikidata)")
add(r"\textit{Claim A.} Redesign, hybrid $-$ random, common background",
    f3(A["full"]["redesign_common"]), f3(A["clean"]["redesign_common"]),
    "14/15 seed$\\times$algorithm cells positive (cleaned)")
add(r"\textit{Claim A.} Same designs, own backgrounds",
    f3(A["full"]["redesign_own"]), f3(A["clean"]["redesign_own"]))
add(r"\textit{Claim A, ten seeds} (declared extension, 2026-11-02; seeds 0--9, 0--4 pre-registered)",
    "not run", extension_row())
add(r"\textit{Claim A.} River feature added, common background",
    f3(A["full"]["river_common"]) + " (12/12)", f3(A["clean"]["river_common"]) + " (12/12)")
add(r"\textit{Claim A.} TGB $-$ random, own / common background",
    f3(A["full"]["tgb_minus_random_own"]) + " / " + f3(A["full"]["tgb_minus_random_common"]),
    f3(A["clean"]["tgb_minus_random_own"]) + " / " + f3(A["clean"]["tgb_minus_random_common"]))
add(r"\textit{Claim B.} Hybrid own-background inflation (hybrid $-$ uniform evaluation)",
    B["full"]["hybrid_inflation_positive"] + ", mean " + f3(B["full"]["hybrid_inflation_mean"]),
    B["clean"]["hybrid_inflation_positive"] + ", mean " + f3(B["clean"]["hybrid_inflation_mean"]))
add(r"\textit{Claim B.} Hybrid beats random, by evaluation background: uniform / TGB / hybrid / stratified",
    " / ".join(str(B["full"]["wins_hybrid_over_random"][k]) for k in ("uniform", "tgb", "hybrid", "stratified")),
    " / ".join(str(B["clean"]["wins_hybrid_over_random"][k]) for k in ("uniform", "tgb", "hybrid", "stratified")))
add(r"\textit{Claim B.} TGB inflation",
    f3(B["full"]["tgb_inflation_mean"]) + " (" + B["full"]["tgb_inflation_positive"] + ")",
    f3(B["clean"]["tgb_inflation_mean"]) + " (" + B["clean"]["tgb_inflation_positive"] + ")")


def curve(c, key):
    v = c[key]
    return f"{v[0]:.3f} $\\to$ {v[3]:.3f} $\\to$ {v[-1]:.3f}"


add(r"\textit{Claim C.} Own-background criterion, hard-negative share 0.0 $\to$ 0.3 $\to$ 1.0",
    curve(C["full"], "own_by_hardfrac"), curve(C["clean"], "own_by_hardfrac"))
add(r"\textit{Claim C.} Common background, same three settings",
    curve(C["full"], "common_by_hardfrac"), curve(C["clean"], "common_by_hardfrac"))
add(r"\textit{Claim D.} Site-density ratio, robust / contingent (RF, XGB, MaxEnt)",
    ", ".join(f"{D['full']['ratio'][k]:.2f}" for k in ("randomforest", "xgboost", "maxent")),
    ", ".join(f"{D['clean']['ratio'][k]:.2f}" for k in ("randomforest", "xgboost", "maxent")))
add(r"\textit{Claim D.} Seed turnover of the top decile",
    f"{pct(D['full']['turnover_range'][0])}--{pct(D['full']['turnover_range'][1])}",
    f"{pct(D['clean']['turnover_range'][0])}--{pct(D['clean']['turnover_range'][1])}")
add(r"\textit{Claim F.} Disjoint $k$-seed ensembles, Jaccard at $k=1 \to k=5$ (XGB, RF, MaxEnt)",
    "; ".join(f"{F['full']['J_k1'][k]:.3f} $\\to$ {F['full']['J_k5'][k]:.3f}" for k in ("xgboost", "randomforest", "maxent")),
    "; ".join(f"{F['clean']['J_k1'][k]:.3f} $\\to$ {F['clean']['J_k5'][k]:.3f}" for k in ("xgboost", "randomforest", "maxent")))
Ef, Ec = S["full"]["E_e235"]["primary_20_40"], S["clean"]["E_e235"]["primary_20_40"]
add(r"\textit{Claim E.} Below 1,000~m, near $-$ far suitability, 20/40~km (random / TGB / hybrid)",
    " / ".join(f3(Ef["suit_" + k]["estimate"]) for k in ("random", "tgb", "hybrid")),
    " / ".join(f3(Ec["suit_" + k]["estimate"]) for k in ("random", "tgb", "hybrid")))

table = [
    r"\begin{table}[ht]\centering\small",
    r"\caption{The quantities behind decision K1, on the compiled presence records and on the cleaned subset "
    r"(E236, pre-registered 2026-10-26). Cleaning drops rows geocoded from their names, modern monuments, "
    r"temples of other regions, tourist objects and modern institutions, then keeps one row per coordinate. "
    r"Every original analysis script was re-run unchanged on the cleaned presences, with only the site file "
    r"redirected. AUC contrasts are means over algorithms and seeds; own-background inflation is the hybrid "
    r"design scored on its own background minus the same model scored on a uniform evaluation background.}",
    r"\label{tab:e236compare}",
    r"\begin{tabular}{p{7.6cm}p{4.6cm}p{4.6cm}}",
    r"\toprule",
    r"Quantity (E236) & Compiled records (378) & Cleaned sites (161) \\",
    r"\midrule",
    *rows,
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(
    "% Generated by build_e236_table.py from E236 results -- do not edit numbers by hand.\n"
    + "\n".join(table) + "\n",
    encoding="utf-8",
)
print(f"wrote {OUT} ({len(rows)} rows)")
