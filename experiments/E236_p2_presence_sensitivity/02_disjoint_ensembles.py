"""E236 part F (ledger C083): agreement between two DISJOINT k-seed ensembles, k = 1..5.

E221's curve compared a k-seed ensemble with the 10-seed ensemble that contains those k seeds, so part of its
agreement is shared membership. Here the two ensembles never share a seed. Ten stored seeds allow k <= 5.
Top decile and Jaccard are computed exactly as in E221 (`top_idx`, `jaccard_sets`).

    python 02_disjoint_ensembles.py full     # E221's stored maps (378 presences)
    python 02_disjoint_ensembles.py clean    # part D's maps (161 presences)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAPS = {"full": HERE.parent / "E221_seed_ensemble_stability" / "results" / "maps",
        "clean": HERE / "results" / "D_e221" / "maps"}
ALGOS = ["xgboost", "randomforest", "maxent"]
DESIGNS = ["random", "tgb", "hybrid"]
N_SEEDS, TOP_FRAC, N_PAIRS, RNG_SEED = 10, 0.10, 50, 236


def top_idx(a, frac=TOP_FRAC):
    k = int(len(a) * frac)
    return np.argpartition(-a, k)[:k]


def jaccard_sets(a, b):
    ta, tb = set(a), set(b)
    return len(ta & tb) / len(ta | tb)


def main(which: str) -> None:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for a in ALGOS:
        for d in DESIGNS:
            maps = [np.load(MAPS[which] / f"{a}_{d}_seed{s}.npy") for s in range(N_SEEDS)]
            for k in range(1, 6):
                js = []
                for _ in range(N_PAIRS):
                    perm = rng.permutation(N_SEEDS)
                    e1 = np.mean([maps[s] for s in perm[:k]], axis=0)
                    e2 = np.mean([maps[s] for s in perm[k:2 * k]], axis=0)
                    js.append(jaccard_sets(top_idx(e1), top_idx(e2)))
                rows.append({"presence_set": which, "algorithm": a, "design": d, "k": k,
                             "jaccard_mean": float(np.mean(js)), "jaccard_p05": float(np.percentile(js, 5)),
                             "jaccard_p95": float(np.percentile(js, 95))})
            print(f"  {which} {a} {d} done")
    df = pd.DataFrame(rows)
    out = HERE / "results" / f"F_disjoint_{which}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby(["algorithm", "k"])["jaccard_mean"].mean().unstack().round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
