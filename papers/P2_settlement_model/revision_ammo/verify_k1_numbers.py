"""G1 check for the numbers added by decision K1 (session 6, 2026-11-02).

Every K1 number in submission_jcaa_v0.3.tex is re-derived here from the stored result files and
searched for in the text. Nothing is compared with a previously printed value.
    python papers/P2_settlement_model/revision_ammo/verify_k1_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
TEX = (HERE / "submission_jcaa_v0.3.tex").read_text(encoding="utf-8")
RES = ROOT / "experiments" / "E236_p2_presence_sensitivity" / "results"

full = json.loads((RES / "summary_full.json").read_text(encoding="utf-8"))
clean = json.loads((RES / "summary_clean.json").read_text(encoding="utf-8"))
raw = pd.read_csv(RES / "A_seeds10" / "e217b_raw_results.csv")
raw = raw[raw.feature_set == "terrain_river"]
piv = raw.pivot_table(index=["seed", "algorithm"], columns="design", aggfunc="mean",
                      values=["auc_common_eval", "auc_own_background"]).reset_index()
piv["common"] = piv[("auc_common_eval", "hybrid")] - piv[("auc_common_eval", "random_nobuffer")]
piv["own"] = piv[("auc_own_background", "hybrid")] - piv[("auc_own_background", "random_nobuffer")]
per_seed = piv.groupby("seed")["common"].mean()
rng = np.random.default_rng(20261102)
seeds = list(per_seed.index)
boot = [per_seed.loc[rng.choice(seeds, size=len(seeds), replace=True)].mean() for _ in range(10000)]
lo, hi = np.percentile(boot, [2.5, 97.5])

# the compiled-set interval, as printed in Section 3.2, comes from the 5-seed E217b file
fraw = pd.read_csv(ROOT / "experiments" / "E217_maxent_benchmark" / "results" / "e217b_raw_results.csv")
fraw = fraw[fraw.feature_set == "terrain_river"]
fpiv = fraw.pivot_table(index=["seed", "algorithm"], columns="design",
                        values="auc_common_eval", aggfunc="mean").reset_index()
fpiv["d"] = fpiv["hybrid"] - fpiv["random_nobuffer"]
fp = fpiv.groupby("seed")["d"].mean()
fboot = [fp.loc[rng.choice(list(fp.index), size=len(fp), replace=True)].mean() for _ in range(10000)]
flo, fhi = np.percentile(fboot, [2.5, 97.5])

def pct(x):
    return f"{100 * x:.1f}"

checks = [
    # compiled set, Section 3.2
    ("3.2 redesign (mean)", "-0.0142", float(f"{fpiv.d.mean():.4f}") == -0.0142),
    ("3.2 redesign interval lo", f"{flo:+.3f}".lstrip("+"), abs(round(flo, 3)) == 0.021),
    ("3.2 redesign cells positive", "3 of 15", int((fpiv.d > 0).sum()) == 3),
    ("3.2 river feature", "+0.0424", round(full["A_e217b"]["river_common"], 4) == 0.0424),
    # cleaned set, Section 3.3
    ("3.3 five seeds", "+0.0507", round(clean["A_e217b"]["redesign_common"], 4) == 0.0507),
    ("3.3 ten seeds", "+0.0510", round(per_seed.mean(), 4) == 0.0510),
    ("3.3 ten-seed CI lo", "+0.039", abs(round(lo, 3) - 0.039) < 1e-9),
    ("3.3 ten-seed CI hi", "+0.063", abs(round(hi, 3) - 0.063) < 1e-9),
    ("3.3 cells positive", "29 of 30", int((piv["common"] > 0).sum()) == 29),
    ("3.3 disjoint halves", "+0.0505", round(per_seed.loc[0:4].mean(), 4) == 0.0505),
    ("3.3 disjoint halves 2", "+0.0516", round(per_seed.loc[5:9].mean(), 4) == 0.0516),
    ("3.3 own background", "+0.098", round(piv["own"].mean(), 3) == 0.098),
    ("3.3 inflation", "+0.047", round(piv["own"].mean() - per_seed.mean(), 3) == 0.047),
    ("3.3 XGBoost", "+0.038", round(piv[piv.algorithm == "xgboost"]["common"].mean(), 3) == 0.038),
    ("3.3 RandomForest", "+0.044", round(piv[piv.algorithm == "randomforest"]["common"].mean(), 3) == 0.044),
    ("3.3 MaxEnt", "+0.072", round(piv[piv.algorithm == "maxent"]["common"].mean(), 3) == 0.072),
    # inflation, claim B
    ("B cleaned cells", "54 of 60", clean["B_e218a"]["hybrid_inflation_positive"] == "54/60"),
    ("B cleaned mean", "+0.041", round(clean["B_e218a"]["hybrid_inflation_mean"], 3) == 0.041),
    ("B full mean", "+0.037", round(full["B_e218a"]["hybrid_inflation_mean"], 3) == 0.037),
    ("B cleaned wins", "3/3 algorithms", all(v == 3 for v in clean["B_e218a"]["wins_hybrid_over_random"].values())),
    ("B TGB cleaned own", "-0.005", round(clean["A_e217b"]["tgb_minus_random_own"], 3) == -0.005),
    ("B TGB cleaned common", "+0.001", round(clean["A_e217b"]["tgb_minus_random_common"], 3) == 0.001),
    # criterion, claim C
    ("C own 0.0", "0.697", round(clean["C_e218b"]["own_by_hardfrac"][0], 3) == 0.697),
    ("C own 1.0", "0.866", round(clean["C_e218b"]["own_by_hardfrac"][-1], 3) == 0.866),
    ("C common 0.0", "0.680", round(clean["C_e218b"]["common_by_hardfrac"][0], 3) == 0.680),
    ("C common 0.3", "0.713", round(clean["C_e218b"]["common_by_hardfrac"][3], 3) == 0.713),
    ("C common 1.0", "0.656", round(clean["C_e218b"]["common_by_hardfrac"][-1], 3) == 0.656),
    # densities and maps, claim D/F
    ("D ratios clean", "2.54", round(clean["D_e221"]["ratio"]["randomforest"], 2) == 2.54),
    ("D ratios clean 2", "8.51", round(clean["D_e221"]["ratio"]["xgboost"], 2) == 8.51),
    ("D ratios clean 3", "5.30", round(clean["D_e221"]["ratio"]["maxent"], 2) == 5.30),
    ("D turnover lo", "41.6", pct(clean["D_e221"]["turnover_range"][0]) == "41.6"),
    ("D turnover hi", "54.8", pct(clean["D_e221"]["turnover_range"][1]) == "54.8"),
    ("F clean k5 lo", "0.70", f"{min(clean['F_disjoint']['J_k5'].values()):.2f}" == "0.70"),
    ("F clean k5 hi", "0.77", f"{max(clean['F_disjoint']['J_k5'].values()):.2f}" == "0.77"),
    ("F full k5 lo", "0.76", f"{min(full['F_disjoint']['J_k5'].values()):.2f}" == "0.76"),
    ("F full k5 hi", "0.86", f"{max(full['F_disjoint']['J_k5'].values()):.2f}" == "0.86"),
    # elevation, claim E
    ("E clean random", "+0.046", round(clean["E_e235"]["primary_20_40"]["suit_random"]["estimate"], 3) == 0.046),
    ("E clean hybrid", "+0.060", round(clean["E_e235"]["primary_20_40"]["suit_hybrid"]["estimate"], 3) == 0.060),
    ("E full random", "+0.049", round(full["E_e235"]["primary_20_40"]["suit_random"]["estimate"], 3) == 0.049),
]

bad = 0
for label, needle, derived_ok in checks:
    in_text = needle in TEX
    status = "ok " if (in_text and derived_ok) else "FAIL"
    if status == "FAIL":
        bad += 1
    print(f"[{status}] {label:28s} {needle:16s} in_text={in_text} derived={derived_ok}")

print(f"\n{len(checks)} checks, {bad} failed")
sys.exit(1 if bad else 0)
