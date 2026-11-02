"""P2 round 2, session 6 (2026-11-02): decision K1 — report both presence sets.

The PI asked Claude to write the K1 text (2 Nov): "kamu saja yg menuliskan". Every edit below is
logged in PI_PROSE_LOG_R2.md ("Session 6") and in JOURNAL 2026-11-02. Numbers come from
experiments/E236_p2_presence_sensitivity/ (summary_{full,clean}.json, results/A_seeds10, ci_check.py).

usage: python apply_session6_k1.py          # dry run: every old -> new, each old string must occur once
       python apply_session6_k1.py --apply  # write submission_jcaa_v0.3.tex
"""
import sys
from pathlib import Path

TEX = Path(__file__).parent / "submission_jcaa_v0.3.tex"

ABSTRACT_OLD = r"""Changing how background points are selected can raise the cross-validated AUC of a presence--background model, but such an increase does not necessarily mean that the model has become better.
This study tests how the background design and the evaluation background affect model evaluation, using 378 archaeological sites in East Java, three algorithms (XGBoost, RandomForest and Maximum Entropy) and spatial block cross-validation.
In an initial sequence of seven experiments, the reported AUC rose overall from 0.659 to 0.751 (the latter a mean over seeds), and the rise was attributed to increasingly realistic background designs.
When all designs are evaluated on a common background, changing the background design contributes $-0.014$ AUC, whereas adding distance to rivers as a feature contributes $+0.042$.
The gain of the final, hybrid design is therefore an evaluation artefact, and the gains of the earlier target-group designs are not reproduced.
In synthetic data with a known truth, the hybrid design evaluated on its own background is reported above its true performance in all 240 of its runs.
The AUC on a design's own background, the criterion used to choose a design, reaches its maximum at the end of the range tested, so it gives no indication of when to stop adding hard negatives (background points selected far from the typical site environment).
Priority maps from a single random seed are unstable; averaging seeds helps, but two independent five-seed averages still agree at a Jaccard overlap of only 0.76--0.86 on the top 10\% of cells.
Above about 1,000~m, elevation and proximity to a Quaternary volcanic centre cannot be adequately separated in the study area.
We propose a protocol: hold the evaluation background fixed when comparing designs, state the availability domain of every metric, and average priority maps over seeds.
The contribution is this diagnosis and protocol, not a new prediction of site locations.
"""

ABSTRACT_NEW = r"""% Drafted by Claude from the PI's passages 4, 6, 8A and 2 at the PI's request (2026-10-22, session 4 part 2),
% and rewritten for decision K1 at the PI's request on 2026-11-02 (session 6). 281 words (limit 300).
Changing how background points are selected can raise cross-validated AUC, but not necessarily because the model has become better.
This study tests how background design and evaluation background affect model evaluation, using East Javanese presence records, three algorithms and spatial block cross-validation.
About a third of the compiled records are modern monuments or misplaced, so every real-data contrast is reported on the 378 records and on 161 cleaned sites.
In a sequence of seven experiments the reported AUC rose from 0.659 to 0.751, attributed to increasingly realistic background designs.
A common evaluation background removes that rise on the compiled records (redesign contributes $-0.014$ AUC); on the cleaned sites the same redesign contributes $+0.051$ [$+0.039$, $+0.063$] over ten seeds.
What holds in both is the inflation: own-background scoring overstates the gain by about $+0.04$ to $+0.05$ AUC, and only a common background shows how much real gain is left.
In synthetic data with a known truth, the hybrid design scored on its own background is reported above its true performance in all 240 runs.
The criterion used to choose a design, AUC on its own background, is maximal at the end of the range tested, so it never indicates when to stop.
Priority maps from a single seed are unstable; averaging helps, but two independent five-seed averages agree at a Jaccard overlap of only 0.70--0.86.
Above about 1,000~m, elevation and volcanic proximity cannot be separated in the study area.
We propose a protocol: hold the evaluation background fixed, state the availability domain of every metric, and average priority maps over seeds.
The contribution is this diagnosis and protocol, not a new prediction of site locations.
"""

PRESENCE = r"""\subsection{The presence records and a cleaned subset}
\label{sec:presence}

The records on which every real-data number in this section rests are compiled from public sources rather than taken from a site catalogue (Section~\ref{sec:covariates}): 378 records on 335 distinct coordinates, of which about a third are modern monuments, statues or memorials, or are misplaced.
A presence set of that composition cannot by itself establish a property of background design, so the same analyses were re-run on a cleaned subset of 161 sites (E236), with the cleaning rule given in Section~\ref{sec:covariates} and with every analysis script unchanged except for the file of presences it reads.
Table~\ref{tab:e236compare} reports every quantity below for both sets.

On the cleaned sites the contrast that is null on the compiled records is positive.
Redesigning the background, with a common evaluation background held fixed, contributes $+0.0507$ AUC over five seeds and $+0.0510$ [$+0.039$, $+0.063$] over ten, positive in 29 of 30 seed$\times$algorithm cells, and positive for each algorithm separately (XGBoost $+0.038$, RandomForest $+0.044$, MaxEnt $+0.072$).
Two ensembles of five seeds that share no seed give $+0.0505$ and $+0.0516$.
Scored on their own backgrounds instead, the same two designs differ by $+0.098$ over ten seeds, so own-background evaluation overstates the gain on this set by $+0.047$, close to the size of the gain itself.

The inflation is the quantity that does not depend on the presence data.
The hybrid design is reported higher on its own background than on a uniform one in 54 of 60 seed$\times$algorithm cells (mean $+0.041$), against 60/60 (mean $+0.037$) on the compiled records.
Two differences must be stated rather than smoothed.
First, on the cleaned sites the hybrid design also beats the random design on every evaluation background (3/3 algorithms on each of the uniform, target-group, hybrid and stratified backgrounds), whereas on the compiled records it led only when scored on its own background.
Second, the earlier target-group steps remain absent in both sets: TGB minus random is $-0.005$ on own backgrounds and $+0.001$ on the common one.

The selection criterion behaves as it does on the compiled records.
On the cleaned sites, AUC on a design's own background rises from $0.697$ at a hard-negative share of $0.0$ to $0.866$ at $1.0$, its maximum at the end of the range tested; the common-background AUC at $0.0$, $0.3$ and $1.0$ is $0.680$, $0.713$ and $0.656$.
The criterion therefore recommends the setting at which the common-background score is lowest, so on this set it moves a modeller away from the best setting rather than merely failing to stop.

The priority-map results hold with somewhat less stability.
Robust-core to contingent-fringe density ratios are $2.54$, $8.51$ and $5.30$ (RandomForest, XGBoost and MaxEnt), the top decile turns over by $41.6$--$54.8$~\% between seed pairs, and two disjoint five-seed ensembles agree at a Jaccard overlap of $0.70$--$0.77$, against $0.76$--$0.86$ on the compiled records (Section~\ref{sec:prioritymaps}).
The contingent fringe holds only 5, 15 and 5 sites across the three algorithms, so the ratios are read as a direction rather than as precise magnitudes.

The elevation control points the same way on both sets.
Below 1,000~m the near $-$ far suitability difference is $+0.046$ for the random and target-group backgrounds and $+0.060$ for the hybrid, against $+0.049$, $+0.049$ and $+0.060$ on the compiled records, and the pre-registered verdicts remain inconclusive or not specific (Section~\ref{sec:elevation}).
No design shows a volcano-specific effect in either set.

"""

EDITS = [
    # ---- 1. Abstract (K1) ----
    ("K1-abstract", ABSTRACT_OLD, ABSTRACT_NEW),

    # ---- 2. Methods 2.1: what the 378 records are, and the cleaning rule ----
    ("K1-methods",
     "Usable presences after covariate filtering were 378 (111--115$^\\circ$E); the soil-coverage subset used once in E009 was 259.\n",
     "Usable presences after covariate filtering were 378 (111--115$^\\circ$E); the soil-coverage subset used once in E009 was 259.\n"
     "These 378 are compiled records rather than a site catalogue.\n"
     "They come from OpenStreetMap, Wikidata and Wikipedia Indonesia, they hold 335 distinct coordinates, and about a third are not archaeological sites: 124 are modern monuments, statues or memorials, and 13 are misplaced or are not sites at all (Central Javanese temples carrying East Javanese coordinates, a Balinese temple placed near Surabaya, a waterfall).\n"
     "Because that composition bears on every model fitted to the records, we define a cleaned subset and report the real-data contrasts on both.\n"
     "The cleaning drops rows geocoded from their names, modern monuments, temples of other regions, tourist objects and modern institutions, and then keeps one row per coordinate; 161 sites remain (155 from OpenStreetMap, 6 from Wikidata).\n"
     "The original ladder (E007--E013, Section~\\ref{sec:ladder}) is the object under examination and stays trained on the 378 records; the contrasts of Section~\\ref{sec:artefact} and Section~\\ref{sec:presence} are reported on both sets.\n"),

    # ---- 3. Section 3.2: the contrast is about the compiled records ----
    ("K1-3.2a",
     "The part of the rise attributed to background redesign does not survive a common evaluation background.\n"
     "On real data, redesigning the background while holding a common evaluation background fixed contributes \\textbf{$-0.0142$ AUC} averaged over the three algorithms, whereas",
     "On the compiled records, the part of the rise attributed to background redesign does not survive a common evaluation background.\n"
     "Redesigning the background while holding a common evaluation background fixed contributes \\textbf{$-0.0142$ AUC} averaged over the three algorithms (seed-cluster interval $-0.021$ to $-0.005$; 3 of 15 seed$\\times$algorithm cells positive), whereas"),

    ("K1-3.2b",
     "(the redesign contribution is $+0.0054$ on the terrain-only feature set and $-0.0142$ on the full set; we state the feature set so a reader does not encounter the difference unannounced).\n",
     "(the redesign contribution is $+0.0054$ on the terrain-only feature set and $-0.0142$ on the full set; we state the feature set so a reader does not encounter the difference unannounced).\n"
     "The contrast is a property of these records and not of the design: on a cleaned subset of the sites the same comparison is positive, and Section~\\ref{sec:presence} reports both.\n"),

    ("K1-3.2c",
     "the inflation from own-background evaluation is positive in \\textbf{60/60} seed$\\times$algorithm cells (mean $+0.037$",
     "the inflation from own-background evaluation is positive in \\textbf{60/60} seed$\\times$algorithm cells on the compiled records (mean $+0.037$"),

    # ---- 4. New Results subsection, after the Figure 3 block ----
    ("K1-new-section",
     "\\subsection{The selection criterion has no interior optimum}\n\\label{sec:nooptimum}",
     PRESENCE + "\\subsection{The selection criterion has no interior optimum}\n\\label{sec:nooptimum}"),

    # ---- 5. Introduction: the ladder statement, scoped ----
    ("K1-intro",
     "However, the results in Section~\\ref{sec:artefact} show that the part of this rise attributed to increasingly realistic background designs is an evaluation artefact, not evidence that such designs produce better models.",
     "However, the results in Section~\\ref{sec:artefact} show that, on the records on which the ladder was run, the part of this rise attributed to increasingly realistic background designs is an evaluation artefact rather than evidence that such designs produce better models; whether they do where the records are cleaner is a separate question, answered in Section~\\ref{sec:presence}."),

    # ---- 6. Discussion 4.1: the two-set statement ----
    ("K1-4.1a",
     "The central methodological finding is that a background-design comparison is not identified unless the evaluation background is held fixed across the designs being compared, and that the artefact this produces is large enough to masquerade as a result.",
     "The central methodological finding is that a background-design comparison is not identified unless the evaluation background is held fixed across the designs being compared, and that the artefact this produces is large enough to masquerade as a result.\n"
     "How much real gain survives the correction depends on the presence data: none on the compiled records, about $+0.05$ AUC on the cleaned sites (Section~\\ref{sec:presence}).\n"
     "The inflation that a common background removes does not depend on the data: it is $+0.037$ on the compiled records (60/60 cells) and $+0.041$ on the cleaned sites (54/60)."),

    ("K1-4.1b",
     "and a direct quantification of the inflation it produces (for the hybrid design, all 240 synthetic runs and 60/60 real seed$\\times$algorithm cells).",
     "and a direct quantification of the inflation it produces (for the hybrid design, all 240 synthetic runs and 60/60 and 54/60 real seed$\\times$algorithm cells on the two presence sets)."),

    # ---- 7. Conclusions ----
    ("K1-conclusions",
     "When all designs are evaluated on a common background, changing the background design contributes $-0.014$ AUC and adding distance to rivers contributes $+0.042$.\n"
     "On a common background the change of design adds nothing. The last design, the hybrid, scores higher only when it is evaluated on its own background points, which is an evaluation artefact, and the gains of the earlier target-group designs are not reproduced.\n",
     "Scored on a common evaluation background, that rise is not reproduced on the 378 compiled records: redesigning the background contributes $-0.014$ AUC and adding distance to rivers contributes $+0.042$, and the last design, the hybrid, scores higher only when it is evaluated on its own background points.\n"
     "Whether a real gain remains depends on the presence data. On 161 cleaned sites the same redesign contributes $+0.051$ AUC [$+0.039$, $+0.063$] over ten seeds, positive in 29 of 30 cells, and two disjoint halves of five seeds agree to within 0.001; own-background evaluation still overstates that gain by about $+0.05$. The earlier target-group designs are not reproduced on either set.\n"
     "Only a common evaluation background shows how much of the reported gain is real.\n"),

    # ---- 8. Limitations ----
    ("K1-lim1",
     "  \\item \\textbf{A detection floor of $\\sim +0.03$} at $n=378$: smaller effects are indeterminate, not absent (Section~\\ref{sec:ci}).",
     "  \\item \\textbf{A detection floor of $\\sim +0.03$} at $n=378$: smaller effects are indeterminate, not absent (Section~\\ref{sec:ci}). The cleaned subset has 161 sites, so its floor is larger; the interval reported for its redesign contrast is over seeds, not over sites."),

    ("K1-lim2",
     "  \\item \\textbf{External transfer beyond East Java is untested.}\n\\end{enumerate}",
     "  \\item \\textbf{External transfer beyond East Java is untested.}\n"
     "  \\item \\textbf{The cleaned presence set is defined by our own rule} (Section~\\ref{sec:presence}); a different rule could retain more or fewer records, and the compiled set remains the one on which the original ladder was run.\n"
     "  \\item \\textbf{The cleaned set's contingent fringe holds 5, 15 and 5 sites} for the three algorithms, so its density ratios are reported as a direction, not as precise magnitudes.\n"
     "\\end{enumerate}"),

    # ---- 9. AI declaration: the K1 passages were drafted with Claude (PI's request) ----
    ("K1-ai-decl",
     "In this version, new passages were written by the first author in Indonesian and translated into English with Claude; the abstract, the conclusions and a number of corrections were drafted with Claude at the first author's request.",
     "In this version, new passages were written by the first author in Indonesian and translated into English with Claude, and the restatements of the abstract, the conclusions and the affected parts of the Results and Discussion were drafted with Claude at the first author's request and then checked and approved by the author."),

    # ---- 10. Front/back matter (mechanical) ----
    ("K1-s9",
     "  \\item \\textbf{Table S8:} E235 pooled differences --- by zone definition and algorithm, with the pre-registered verdicts.",
     "  \\item \\textbf{Table S8:} E235 pooled differences --- by zone definition and algorithm, with the pre-registered verdicts.\n"
     "  \\item \\textbf{Table S9:} E236 --- the same real-data quantities on the compiled records and on the cleaned presences (Section~\\ref{sec:presence})."),

    ("K1-code",
     "all source code, including E217--E224, E235 and the verification script",
     "all source code, including E217--E224, E235, E236 and the verification script"),
]


def main():
    text = TEX.read_text(encoding="utf-8")
    ok = True
    for name, old, new in EDITS:
        n = text.count(old)
        if n != 1:
            ok = False
            print(f"[FAIL] {name}: old string occurs {n} times")
            continue
        print(f"[ok]   {name}: {len(old.split())} -> {len(new.split())} words")
    if not ok:
        print("\nDRY RUN FAILED; nothing written.")
        sys.exit(1)
    if "--apply" in sys.argv:
        for _, old, new in EDITS:
            text = text.replace(old, new, 1)
        TEX.write_text(text, encoding="utf-8")
        print(f"\nAPPLIED {len(EDITS)} edits to {TEX.name}")
    else:
        print(f"\nDRY RUN OK: {len(EDITS)} edits would apply.")


if __name__ == "__main__":
    main()
