"""
E221b: split-half ensemble control — the matched reference for ensemble-level design divergence.

Script 01's `between_design_survives_ensembling` verdict compared between-design ENSEMBLE Jaccards
against a noise floor built from SINGLE-RUN pairs. That reference is mismatched: ensemble maps are
smoother than single-run maps, so any two ensembles agree more than any two single runs, whatever the
design effect. The matched control is two independent same-design ensembles (seeds 0-4 vs seeds 5-9):
if the between-design ensemble gap does not beat THAT floor, the design effect is seed noise amplified.

Computed from the stored surfaces (results/maps/*.npy); no new fits. Supersedes the one verdict in
e221_outcome.json["between_design_survives_ensembling"]; all other script-01 outputs stand.

Run from repo root:
    py experiments/E221_seed_ensemble_stability/02_split_half_control.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
MAPS_DIR = RESULTS_DIR / "maps"
ALGOS = ["maxent", "randomforest", "xgboost"]
DESIGNS = ["random", "tgb", "hybrid"]
TOP_FRAC = 0.10


def top_set(a, frac=TOP_FRAC):
    k = int(len(a) * frac)
    return set(np.argpartition(-a, k)[:k])


def main():
    maps = {(a, d, s): np.load(MAPS_DIR / f"{a}_{d}_seed{s}.npy")
            for a in ALGOS for d in DESIGNS for s in range(10)}

    rows = []
    for a in ALGOS:
        # within-design: two independent same-design ensembles (the matched noise floor)
        floors = []
        for d in DESIGNS:
            eA = np.mean([maps[(a, d, s)] for s in range(5)], axis=0)
            eB = np.mean([maps[(a, d, s)] for s in range(5, 10)], axis=0)
            floors.append(len(top_set(eA) & top_set(eB)) / len(top_set(eA) | top_set(eB)))
        floor = float(np.mean(floors))
        for i, d1 in enumerate(DESIGNS):
            for d2 in DESIGNS[i + 1:]:
                js = []
                for half in (range(5), range(5, 10)):
                    e1 = np.mean([maps[(a, d1, s)] for s in half], axis=0)
                    e2 = np.mean([maps[(a, d2, s)] for s in half], axis=0)
                    js.append(len(top_set(e1) & top_set(e2)) / len(top_set(e1) | top_set(e2)))
                j = float(np.mean(js))
                rows.append({"algorithm": a, "pair": f"{d1}|{d2}",
                             "ensemble_jaccard": round(j, 4),
                             "split_half_floor": round(floor, 4),
                             "below_floor": bool(j < floor - 0.05)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e221_split_half_control.csv", index=False)
    print(df.to_string(index=False))

    survives = {a: bool(df[(df.algorithm == a) & (df.pair.str.contains("hybrid"))]["below_floor"].any())
                for a in ALGOS}
    out = {"note": "matched split-half (5+5) ensemble control; supersedes "
                   "e221_outcome.json[between_design_survives_ensembling]",
           "design_gap_survives_ensembling": survives}
    with open(RESULTS_DIR / "e221_outcome_split_half.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  design gap survives ensembling (matched control): {survives}")


if __name__ == "__main__":
    main()
