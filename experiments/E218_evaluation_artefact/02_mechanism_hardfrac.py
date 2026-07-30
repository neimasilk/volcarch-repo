"""
E218b: Mechanism, redesigned — does AUC inflation track background dissimilarity after all?

WHY A SECOND ATTEMPT
--------------------
E218 Stage C tried to test whether AUC inflation rises with how far the background sits from the
presences in environmental space. It returned Spearman = -0.077 (p = 0.41), but the test was not
trustworthy: backgrounds were built by sampling a NARROW zdist band, which produces a background
concentrated in a thin environmental shell. Such a background is trivially separable from the presence
cloud no matter how far away the shell sits -- `auc_own` reached 0.98 at the NEAREST band, the opposite
of the hypothesis -- while `auc_common` collapsed to ~0.55 because a shell trains a model that does not
generalise. The construction confounded DISTANCE with CONCENTRATION and never tested the intended
quantity. A null from a broken instrument is not evidence.

THE FIX
-------
Sweep the paper's own `hard_frac` knob (0.0 -> 1.0) instead of a band. `hard_frac` is the proportion of
background points deliberately drawn from the environmentally dissimilar tail (zdist >= 2), with the
remainder drawn from the natural target-group pool. This:
  - varies mean background dissimilarity smoothly,
  - keeps every background a plausible draw from the actual landscape (no thin shells),
  - is interpretable in the manuscript's own terms, because E013 already tunes this exact parameter.

WHY IT MATTERS BEYOND CORRECTNESS
---------------------------------
Reviewer 1 already warned that the paper's finding is "not entirely novel... well established in adjacent
fields such as ecological niche modeling". The bare artefact result is close to Lobo et al. (2008). A
QUANTIFIED relationship -- inflation as a measurable function of a knob practitioners actually turn --
would be a contribution beyond that caution rather than a restatement of it. If this sweep comes back
null too, that novelty argument is not available and the manuscript must be scoped accordingly.

Pre-registered reading (locked before running):
  - Monotonic rise of inflation with realised dissimilarity, Spearman > 0.5, p < 0.01
      -> mechanism established; report as a quantified relationship.
  - Inflation high but flat across hard_frac
      -> the artefact does not depend on dissimilarity; some other property of own-background scoring
         drives it, and the manuscript says exactly that and stops.
  - auc_common also rises with hard_frac
      -> hard negatives genuinely help generalisation, which would PARTLY rehabilitate the paper's
         original intuition. Report it; do not bury it.

Run from repo root:
    py experiments/E218_evaluation_artefact/02_mechanism_hardfrac.py
"""

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "e217base", HERE.parent / "E217_maxent_benchmark" / "01_maxent_benchmark.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

main_spec = importlib.util.spec_from_file_location("e218main", HERE / "01_artefact_robustness.py")
e218 = importlib.util.module_from_spec(main_spec)
main_spec.loader.exec_module(e218)

from scipy.stats import spearmanr  # noqa: E402

RESULTS_DIR = HERE / "results"
HARD_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_SEEDS = 5


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("E218b: mechanism redesigned — inflation vs hard_frac (the paper's own knob)")
    print("=" * 74)

    pres, frame_all, frame_buf, _, prop, n_pa = e218.prepare(decimate=base.DECIMATE)
    n_eval = len(pres) * e218.EVAL_RATIO
    print(f"  presences={len(pres)}  frame={len(frame_all):,}  background target={n_pa}")

    original = base.HYBRID_HARD_FRAC
    rows = []
    try:
        for s in range(N_SEEDS):
            rng = np.random.default_rng(base.RANDOM_SEED + 1000 * s)
            ev = base.draw_random(frame_all, n_eval, rng)
            for hf in HARD_FRACS:
                base.HYBRID_HARD_FRAC = hf          # the swept knob
                bg = base.draw_hybrid(frame_buf, n_pa, prop, rng)
                if len(bg) < int(0.9 * n_pa):
                    print(f"  seed {s} hard_frac {hf}: pool shortfall ({len(bg)}), skipped")
                    continue
                realised = float(bg["zdist"].mean())
                frac_ge2 = float((bg["zdist"] >= 2.0).mean())
                for algo in base.ALGORITHMS:
                    auc_own, _, _ = e218.evaluate(pres, bg, bg, ev, algo, base.BLOCK_SIZE_DEG)
                    auc_com, _, _ = e218.evaluate(pres, bg, ev, ev, algo, base.BLOCK_SIZE_DEG)
                    rows.append({"seed": s, "hard_frac": hf, "zdist_mean": realised,
                                 "frac_zdist_ge2": frac_ge2, "algorithm": algo,
                                 "auc_own": auc_own, "auc_common": auc_com,
                                 "inflation": auc_own - auc_com, "n_bg": len(bg)})
            print(f"  seed {s + 1}/{N_SEEDS} done")
    finally:
        base.HYBRID_HARD_FRAC = original

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e218b_hardfrac_sweep.csv", index=False)

    summ = df.groupby("hard_frac").agg(
        zdist_mean=("zdist_mean", "mean"), frac_ge2=("frac_zdist_ge2", "mean"),
        auc_own=("auc_own", "mean"), auc_common=("auc_common", "mean"),
        inflation=("inflation", "mean")).round(4)
    summ.to_csv(RESULTS_DIR / "e218b_summary.csv")
    print("\n" + "=" * 74)
    print("SWEEP RESULT")
    print("=" * 74)
    print(summ.to_string())

    r_inf = spearmanr(df["zdist_mean"], df["inflation"])
    r_own = spearmanr(df["zdist_mean"], df["auc_own"])
    r_com = spearmanr(df["zdist_mean"], df["auc_common"])
    out = {
        "experiment": "E218b", "n_seeds": N_SEEDS,
        "spearman_dissimilarity_vs_inflation": round(float(r_inf.statistic), 4),
        "p_inflation": float(r_inf.pvalue),
        "spearman_dissimilarity_vs_auc_own": round(float(r_own.statistic), 4),
        "p_auc_own": float(r_own.pvalue),
        "spearman_dissimilarity_vs_auc_common": round(float(r_com.statistic), 4),
        "p_auc_common": float(r_com.pvalue),
        "mechanism_established": bool(r_inf.statistic > 0.5 and r_inf.pvalue < 0.01),
        "hard_negatives_help_generalisation": bool(r_com.statistic > 0.5 and r_com.pvalue < 0.01),
    }
    with open(RESULTS_DIR / "e218b_outcome.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 74)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 74)
    print(f"  Spearman(dissimilarity, inflation)  = {r_inf.statistic:+.3f} (p={r_inf.pvalue:.2e})")
    print(f"  Spearman(dissimilarity, auc_own)    = {r_own.statistic:+.3f} (p={r_own.pvalue:.2e})")
    print(f"  Spearman(dissimilarity, auc_common) = {r_com.statistic:+.3f} (p={r_com.pvalue:.2e})")
    print(f"\n  MECHANISM ESTABLISHED                        : {out['mechanism_established']}")
    print(f"  HARD NEGATIVES HELP GENERALISATION (rescue?) : "
          f"{out['hard_negatives_help_generalisation']}")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
