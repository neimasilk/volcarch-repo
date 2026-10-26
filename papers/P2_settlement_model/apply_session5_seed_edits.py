"""P2 round 2, session 5 (2026-10-26), part 2a: the seed-stability claims (ledger C083), rewritten from E236 part F
under its pre-registered rule ("averaging helps" HOLDS; quote the disjoint agreement actually measured).

Values: experiments/E236_p2_presence_sensitivity/results/F_disjoint_full.csv (mean over designs):
J(k=1) XGBoost 0.548, RandomForest 0.702, MaxEnt 0.699; J(k=5) XGBoost 0.759, RandomForest 0.861, MaxEnt 0.849.

usage: python apply_session5_seed_edits.py [--apply]
"""
import sys
from pathlib import Path

TEX = Path(__file__).parent / "submission_jcaa_v0.3.tex"

EDITS = [
    ("F1", "Priority maps from a single random seed are unstable; averaging at least seven seeds gives a stable map of "
           "the top 10\\% of cells.",
     "Priority maps from a single random seed are unstable; averaging seeds helps, but two independent five-seed "
     "averages still agree at a Jaccard overlap of only 0.76--0.86 on the top 10\\% of cells."),
    ("F2", "Seed ensemble & A combined map obtained by averaging the predictions of several random seeds; in this "
           "study, the map of the cells in the top 10\\% of the ranking becomes stable from seven seeds onwards, with a "
           "Jaccard overlap of at least 0.90 against the 10-seed ensemble. \\\\",
     "Seed ensemble & A combined map obtained by averaging the predictions of several random seeds. In this study, two "
     "ensembles of five seeds that share no seed agree at a Jaccard overlap of 0.76--0.86 on the cells in the top "
     "10\\% of the ranking, against 0.55--0.70 for two single seeds. \\\\"),
    ("F3", "Across random seeds, how much of the top-decile priority map turns over, and how many seeds must be "
           "ensembled to stabilise it?",
     "Across random seeds, how much of the top-decile priority map turns over, and how much does averaging seeds "
     "reduce it?"),
    ("F4", "\\subsection{Priority maps are unstable under seed, and the protocol that fixes them}",
     "\\subsection{Priority maps change with the seed; averaging seeds reduces this}"),
    ("F5", "The number of seeds required to stabilise the map is $k^* = 2$--$5$ for Jaccard $\\geq 0.85$, \\textbf{$4$--$7$ "
           "for $\\geq 0.90$}, and $7$--$9$ for $\\geq 0.95$; we recommend $k \\geq 7$ (Figure~\\ref{fig:stabilisation}).",
     "Measured against the 10-seed ensemble, which contains the seeds being averaged, the overlap passes 0.90 after "
     "$4$--$7$ seeds, but that comparison shares seeds with its reference and so flatters the result. Two ensembles "
     "that share no seed agree less: Jaccard $0.55$--$0.70$ for single seeds, and $0.76$ (XGBoost), $0.85$ (MaxEnt) "
     "and $0.86$ (RandomForest) for five seeds each (Figure~\\ref{fig:stabilisation}; E236). Averaging clearly helps, "
     "so we recommend at least seven seeds as a floor, together with reporting the agreement between ensembles that "
     "share no seed, rather than assuming that a given number of seeds makes the map stable."),
    ("F6", "\\caption{Agreement of an ensemble of $k$ seeds with the full 10-seed ensemble on the top-decile priority map, "
           "one line per learner. A single seed agrees at Jaccard $\\approx 0.65$--$0.80$; $k^* = 4$--$7$ suffices for "
           "Jaccard $\\geq 0.90$, and $k \\geq 7$ is the protocol floor (shaded).}",
     "\\caption{Agreement of top-decile priority maps as seeds are averaged, one colour per learner. Thin lines: an "
     "ensemble of $k$ seeds against the full 10-seed ensemble, which contains those seeds (E221). Thick lines: two "
     "ensembles of $k$ seeds that share no seed (E236; at most five seeds each from ten). The thick lines are the "
     "fair measure of stability.}"),
    ("F7", "the seed-ensemble protocol ($k \\geq 7$) is a precondition for operational use.",
     "the seed-ensemble protocol (at least seven seeds, with the agreement between disjoint ensembles reported) is a "
     "precondition for operational use."),
    ("F8", "\\item \\textbf{Seed-ensemble priority maps} ($k \\geq 7$); report a robust core and contingent fringe, never "
           "a single-seed map.",
     "\\item \\textbf{Seed-ensemble priority maps} (at least seven seeds) and report the agreement between ensembles "
     "that share no seed; report a robust core and contingent fringe, never a single-seed map."),
    ("F9", "and priority maps should be averaged over at least seven seeds.",
     "and priority maps should be averaged over at least seven seeds, with the agreement between ensembles that share "
     "no seed reported."),
]


def main():
    text = TEX.read_text(encoding="utf-8")
    bad = [(i, text.count(o)) for i, o, _ in EDITS if text.count(o) != 1]
    for i, old, new in EDITS:
        print(f"[{i}] {old.strip()[:100]!r}\n   -> {new.strip()[:100]!r}")
    if bad:
        print("NOT FOUND EXACTLY ONCE:", bad)
        sys.exit(1)
    if "--apply" in sys.argv:
        for _, old, new in EDITS:
            text = text.replace(old, new)
        TEX.write_text(text, encoding="utf-8")
        print(f"applied {len(EDITS)} edits")
    else:
        print(f"dry run: {len(EDITS)} edits, all match once")


if __name__ == "__main__":
    main()
