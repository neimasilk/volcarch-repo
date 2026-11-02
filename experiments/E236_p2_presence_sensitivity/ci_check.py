"""Uncertainty for E236 claim A (hybrid - random, common background), both presence sets.

The paper reports AUC contrasts with intervals; decision K1 (PI, 2026-11-02) puts +0.051 on the cleaned
set into the abstract, so it needs one too. Naive t-interval over the seed x algorithm cells, plus a
seed-level cluster bootstrap (seeds are the shared unit; the three algorithms are not independent).
Reads only the stored E217b / E236 outputs; no model is re-fitted.
    python experiments/E236_p2_presence_sensitivity/ci_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FULL = HERE.parents[0] / "E217_maxent_benchmark" / "results" / "e217b_raw_results.csv"
CLEAN = HERE / "results" / "A_e217b" / "e217b_raw_results.csv"
FEATURE = "terrain_river"
rng = np.random.default_rng(20261102)


def contrast(path):
    raw = pd.read_csv(path)
    raw = raw[raw.feature_set == FEATURE]
    piv = raw.pivot_table(index=["seed", "algorithm"], columns="design",
                          values="auc_common_eval", aggfunc="mean").reset_index()
    piv["delta"] = piv["hybrid"] - piv["random_nobuffer"]
    seeds = sorted(piv.seed.unique())
    per_seed = piv.groupby("seed")["delta"].mean()
    boot = [per_seed.loc[rng.choice(seeds, size=len(seeds), replace=True)].mean() for _ in range(10000)]
    return piv, per_seed, np.percentile(boot, [2.5, 97.5])


EXT = HERE / "results" / "A_seeds10" / "e217b_raw_results.csv"

for label, path in (("compiled 378", FULL), ("cleaned 161 (pre-registered, seeds 0-4)", CLEAN),
                    ("cleaned 161 (extension, seeds 0-9)", EXT)):
    piv, per_seed, (lo, hi) = contrast(path)
    c = piv["delta"].to_numpy()
    n, mean, sd = len(c), c.mean(), c.std(ddof=1)
    print(f"{label}: cells n={n}, mean={mean:+.4f}, sd={sd:.4f}, positive={(c > 0).sum()}/{n}")
    print(f"  naive t95 [{mean - 2.145 * sd / np.sqrt(n):+.4f}, {mean + 2.145 * sd / np.sqrt(n):+.4f}]")
    print(f"  seed-cluster bootstrap 95% ({per_seed.size} seeds): [{lo:+.4f}, {hi:+.4f}]")
    print(f"  per algorithm: {piv.groupby('algorithm')['delta'].mean().round(4).to_dict()}")
    print(f"  per seed: {per_seed.round(4).to_dict()}")
    if label.startswith("cleaned") and "0-9" in label:
        early, late = per_seed.loc[0:4], per_seed.loc[5:9]
        print(f"  DISJOINT HALVES: seeds 0-4 mean={early.mean():+.4f} (positive "
              f"{(piv[piv.seed <= 4].groupby('seed')['delta'].mean() > 0).sum()}/5); "
              f"seeds 5-9 mean={late.mean():+.4f} (positive "
              f"{(piv[piv.seed >= 5].groupby('seed')['delta'].mean() > 0).sum()}/5)")
