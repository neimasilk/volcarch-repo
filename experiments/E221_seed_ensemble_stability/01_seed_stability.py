"""
E221: Seed-ensemble stabilisation + robust/contingent priority sets.

E219 showed the survey-priority map moves between designs and between seeds, but stored no maps and used
5 seeds. This experiment re-produces the surfaces at 10 seeds x 3 designs x 3 algorithms, stores them,
and answers the two questions a heritage practitioner asks next: how many seeds until the priority tier
stops moving (Part A), and which priorities survive arbitrary analytic choices at all (Part B).
Pre-registration: DESIGN.md.

Run from repo root (long; use a background run):
    py experiments/E221_seed_ensemble_stability/01_seed_stability.py
"""

import importlib.util
import itertools
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

E219 = HERE.parent / "E219_map_divergence"
spec219 = importlib.util.spec_from_file_location("e219main", E219 / "01_map_divergence.py")
e219 = importlib.util.module_from_spec(spec219)
spec219.loader.exec_module(e219)

from scipy.spatial import cKDTree                    # noqa: E402

RESULTS_DIR = HERE / "results"
MAPS_DIR = RESULTS_DIR / "maps"
FEAT = base.FULL_COLS
ALGOS = e219.MAP_ALGOS
DESIGNS = ["random", "tgb", "hybrid"]
N_SEEDS = 10
TOP_FRAC = 0.10
N_SUBSETS = 50


def top_idx(a, frac=TOP_FRAC):
    k = int(len(a) * frac)
    return np.argpartition(-a, k)[:k]


def jaccard_sets(a, b):
    ta, tb = set(a), set(b)
    return len(ta & tb) / len(ta | tb)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("E221: seed-ensemble stabilisation + robust/contingent priority sets")
    print("=" * 74)

    pres, frame, frame_buf, designs, prop, n_pa = e218.prepare(decimate=base.DECIMATE)
    print(f"  presences={len(pres)}  frame={len(frame):,}  background target={n_pa}")
    Z = frame[FEAT].to_numpy(dtype=np.float64)

    # ---------------- produce + store 90 surfaces ----------------
    print("\nProducing prediction surfaces (10 seeds x 3 designs x 3 algorithms)...")
    maps = {}
    for s in range(N_SEEDS):
        rng = np.random.default_rng(base.RANDOM_SEED + 1000 * s)
        bgs = {d: f(rng) for d, f in designs.items()}
        for d in DESIGNS:
            X = np.vstack([pres[FEAT].to_numpy(dtype=np.float64),
                           bgs[d][FEAT].to_numpy(dtype=np.float64)])
            y = np.r_[np.ones(len(pres)), np.zeros(len(bgs[d]))]
            for algo in ALGOS:
                p = e219.predict_chunked(e219.fit_full(algo, X, y), Z).astype(np.float32)
                maps[(algo, d, s)] = p
                np.save(MAPS_DIR / f"{algo}_{d}_seed{s}.npy", p)
        print(f"  seed {s + 1}/{N_SEEDS} done")

    ens = {(a, d): np.mean([maps[(a, d, s)] for s in range(N_SEEDS)], axis=0)
           for a in ALGOS for d in DESIGNS}

    # ---------------- Part A: stabilisation curve ----------------
    print("\n" + "=" * 74)
    print("PART A — ensemble-of-k vs ensemble-of-10 (top-decile Jaccard)")
    print("=" * 74)
    rngA = np.random.default_rng(7)
    rows = []
    for algo in ALGOS:
        for d in DESIGNS:
            full = top_idx(ens[(algo, d)])
            singles = [jaccard_sets(top_idx(maps[(algo, d, s)]), full) for s in range(N_SEEDS)]
            for k in range(1, N_SEEDS):
                combos = list(itertools.combinations(range(N_SEEDS), k))
                if len(combos) > N_SUBSETS:
                    combos = [tuple(rngA.choice(N_SEEDS, size=k, replace=False))
                              for _ in range(N_SUBSETS)]
                js = []
                for c in combos:
                    sub = np.mean([maps[(algo, d, s)] for s in c], axis=0)
                    js.append(jaccard_sets(top_idx(sub), full))
                rows.append({"algorithm": algo, "design": d, "k": k,
                             "jaccard_mean": float(np.mean(js)), "jaccard_sd": float(np.std(js)),
                             "single_run_mean": float(np.mean(singles)),
                             "single_run_min": float(np.min(singles))})
    curve = pd.DataFrame(rows)
    curve.to_csv(RESULTS_DIR / "e221_stabilisation_curve.csv", index=False)
    kstar = (curve[curve.jaccard_mean >= 0.90]
             .groupby(["algorithm", "design"])["k"].min())
    print(kstar.to_string())
    singles_summary = (curve[curve.k == 1]
                       .set_index(["algorithm", "design"])[["single_run_mean", "single_run_min"]])
    print("\nSingle run vs 10-seed ensemble (k=1):")
    print(singles_summary.round(3).to_string())

    # Between-design divergence at ensemble level (falsification branch)
    ens_rows = []
    for algo in ALGOS:
        for d1, d2 in itertools.combinations(DESIGNS, 2):
            ens_rows.append({"algorithm": algo, "pair": f"{d1}|{d2}",
                             "jaccard_top10": jaccard_sets(top_idx(ens[(algo, d1)]),
                                                           top_idx(ens[(algo, d2)]))})
    ens_div = pd.DataFrame(ens_rows)
    ens_div.to_csv(RESULTS_DIR / "e221_ensemble_between_design.csv", index=False)
    print("\nBetween-design ENSEMBLE divergence (cf. E219 single-seed values):")
    print(ens_div.pivot_table(index="algorithm", columns="pair", values="jaccard_top10")
            .round(3).to_string())

    # ---------------- Part B: robust vs contingent priority sets ----------------
    print("\n" + "=" * 74)
    print("PART B — robust vs design-contingent priority sets")
    print("=" * 74)
    canon_latlon, canon_names = e219.load_canonical_volcanoes()
    volc_km = e219.volcano_distance_km(frame["x"].to_numpy(), frame["y"].to_numpy(), canon_latlon)
    site_tree = cKDTree(pres[["x", "y"]].to_numpy())
    dist_site, site_nn = site_tree.query(frame[["x", "y"]].to_numpy(), k=1)
    # map each known site to its nearest frame cell (for enrichment counts)
    frame_tree = cKDTree(frame[["x", "y"]].to_numpy())
    _, site_cell = frame_tree.query(pres[["x", "y"]].to_numpy(), k=1)

    brows = []
    for algo in ALGOS:
        tops = {d: np.zeros(len(frame), dtype=bool) for d in DESIGNS}
        for d in DESIGNS:
            tops[d][top_idx(ens[(algo, d)])] = True
        n_flag = sum(tops[d].astype(np.int8) for d in DESIGNS)
        robust = n_flag == 3
        contingent = n_flag == 1
        np.savez_compressed(
            RESULTS_DIR / f"e221_priority_sets_{algo}.npz",
            x=frame["x"].to_numpy(), y=frame["y"].to_numpy(),
            suit_random=ens[(algo, "random")], suit_tgb=ens[(algo, "tgb")],
            suit_hybrid=ens[(algo, "hybrid")], n_designs_top10=n_flag)

        cell_km2 = (base.DECIMATE * 30.663793749005784 / 1000.0) ** 2
        sets = {"robust": robust, "contingent": contingent, "rest": ~(robust | contingent)}
        for d in DESIGNS:      # split the contingent fringe by owning design
            sets[f"contingent_{d}"] = contingent & tops[d]
        for name, m in sets.items():
            cells = np.where(m)[0]
            if len(cells) == 0:
                continue
            in_sites = int(np.isin(site_cell, cells).sum())
            brows.append({"algorithm": algo, "set": name, "n_cells": len(cells),
                          "area_km2": len(cells) * cell_km2,
                          "frame_share": float(m.mean()),
                          "known_sites_inside": in_sites,
                          "sites_per_1000km2": in_sites / max(len(cells) * cell_km2, 1e-9) * 1000,
                          "road_dist_median_m": float(np.median(frame["road_dist"].to_numpy()[m])),
                          "elevation_median_m": float(np.median(frame["elevation"].to_numpy()[m])),
                          "volc_km_median": float(np.median(volc_km[m])),
                          "dist_to_site_median_m": float(np.median(dist_site[m]))})
    bdf = pd.DataFrame(brows)
    bdf.to_csv(RESULTS_DIR / "e221_priority_sets.csv", index=False)
    show = bdf[bdf.set.isin(["robust", "contingent", "rest"])]
    print(show.round(3).to_string(index=False))

    # ---------------- Part C: turnover, both definitions ----------------
    print("\n" + "=" * 74)
    print("PART C — within-design seed turnover, both definitions")
    print("=" * 74)
    crows = []
    for algo in ALGOS:
        for d in DESIGNS:
            for i, j in itertools.combinations(range(N_SEEDS), 2):
                J = jaccard_sets(top_idx(maps[(algo, d, i)]), top_idx(maps[(algo, d, j)]))
                crows.append({"algorithm": algo, "design": d, "seed_i": i, "seed_j": j,
                              "jaccard": J, "footprint_single_run_share": 1 - J,
                              "replaced_share": (1 - J) / (1 + J)})
    cdf = pd.DataFrame(crows)
    cdf.to_csv(RESULTS_DIR / "e221_turnover_pairs.csv", index=False)
    cs = cdf.groupby(["algorithm", "design"])[["jaccard", "footprint_single_run_share",
                                               "replaced_share"]].mean().round(4)
    print(cs.to_string())

    # ---------------- verdicts ----------------
    out = {
        "experiment": "E221", "n_seeds": N_SEEDS, "n_presences": int(len(pres)),
        "kstar_jac0.9": {f"{a}|{d}": (int(kstar[(a, d)]) if (a, d) in kstar.index else None)
                         for a in ALGOS for d in DESIGNS},
        "single_run_vs_ensemble": {f"{a}|{d}": round(float(singles_summary.loc[(a, d),
                                                                              "single_run_mean"]), 4)
                                   for a in ALGOS for d in DESIGNS},
        "ensemble_between_design": {f"{r['algorithm']}|{r['pair']}": round(r["jaccard_top10"], 4)
                                    for _, r in ens_div.iterrows()},
        "turnover_by_algo_design": {f"{a}|{d}": {
            "jaccard": float(cs.loc[(a, d), "jaccard"]),
            "one_minus_jaccard": float(cs.loc[(a, d), "footprint_single_run_share"]),
            "replaced_share": float(cs.loc[(a, d), "replaced_share"])}
            for a in ALGOS for d in DESIGNS},
    }
    kmax = max(v for v in out["kstar_jac0.9"].values() if v is not None)
    out["protocol_recommendation_min_seeds"] = kmax
    out["all_combos_stabilise_by_6"] = bool(all(
        v is not None and v <= 6 for v in out["kstar_jac0.9"].values()))
    # falsification branch: does ensembling erase the between-design gap?
    within_floor = cdf.groupby(["algorithm"])["jaccard"].mean()
    out["between_design_survives_ensembling"] = {
        a: bool(np.mean([r["jaccard_top10"] for _, r in ens_div[ens_div.algorithm == a].iterrows()])
                < within_floor[a] - 0.05) for a in ALGOS}
    # robust vs contingent site density direction (per algorithm)
    out["robust_vs_contingent_site_density"] = {}
    for algo in ALGOS:
        a = bdf[bdf.algorithm == algo].set_index("set")
        if "robust" in a.index and "contingent" in a.index:
            out["robust_vs_contingent_site_density"][algo] = {
                "robust": round(float(a.loc["robust", "sites_per_1000km2"]), 3),
                "contingent": round(float(a.loc["contingent", "sites_per_1000km2"]), 3),
                "robust_higher": bool(a.loc["robust", "sites_per_1000km2"]
                                      > a.loc["contingent", "sites_per_1000km2"])}

    with open(RESULTS_DIR / "e221_outcome.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 74)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 74)
    print(f"  k* (J>=0.9) per combo          : {out['kstar_jac0.9']}")
    print(f"  all combos stabilise by k=6    : {out['all_combos_stabilise_by_6']}")
    print(f"  protocol floor (max k*)        : {out['protocol_recommendation_min_seeds']} seeds")
    print(f"  between-design gap survives ensembling: {out['between_design_survives_ensembling']}")
    print(f"  robust > contingent site density      : {out['robust_vs_contingent_site_density']}")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
