"""P2 round 2, session 5 (2026-10-26): corrections after the independent adversarial read, applied at the PI's
"ok, perbaiki" (26 Oct). Every edit is listed in PI_PROSE_LOG_R2.md ("Session 5"); ledger C082-C089.

Part 1 (this list) needs no E236 result. Part 2 (E236-dependent: presence data, seed stability, densities) is a
separate list, applied after E236 has run.

usage: python apply_session5_edits.py          # dry run: every old -> new, each old string must occur once
       python apply_session5_edits.py --apply  # write submission_jcaa_v0.3.tex
"""
import re
import sys
from pathlib import Path

TEX = Path(__file__).parent / "submission_jcaa_v0.3.tex"

EDITS = [
    # ---- C084: the TGB part of the "artefact"; "the rise is not real"; road_dist ----
    ("R5a", "The part of the rise attributed to background design is therefore an evaluation artefact.\n",
     "The gain of the final, hybrid design is therefore an evaluation artefact, and the gains of the earlier "
     "target-group designs are not reproduced.\n"),
    ("R5b", "The rise is not real.\n",
     "The part of the rise attributed to background redesign does not survive a common evaluation background.\n"),
    ("R5c", "The TGB steps of the ladder (E010--E012) are therefore invalidated not by own-background inflation but by "
            "the common-background comparison itself, which contributed approximately zero once identified.",
     "The TGB steps of the ladder (E010--E012) are not explained by own-background inflation either: in the benchmark "
     "TGB scores almost exactly as random does even on its own background ($+0.003$, E217), so their original gain is "
     "not reproduced "
     "rather than explained."),
    ("R5d", "The part of the rise attributed to background design is therefore an evaluation artefact: on a common "
            "background the change of design adds nothing, and the last design, the hybrid, scores higher only when it "
            "is evaluated on its own background points.",
     "On a common background the change of design adds nothing. The last design, the hybrid, scores higher only when "
     "it is evaluated on its own background points, which is an evaluation artefact, and the gains of the earlier "
     "target-group designs are not reproduced."),
    ("R5e", "After the benchmark reported in Section~3, it is the paper's actual subject, because the variable that "
            "defines the background (accessibility, via \\texttt{road\\_dist}) is the variable that turned out to drive "
            "the reported number.",
     "It is central to this paper: the variable that defines the target-group background (accessibility, through "
     "\\texttt{road\\_dist}) is kept out of the features, so accessibility can affect the reported numbers only "
     "through the background design, which is what Section~3 examines."),
    # ---- C087: 343/360 pooled across designs ----
    ("R9a", "In synthetic data with a known truth, a design evaluated on its own background is reported above its true "
            "performance in 343 of 360 runs.",
     "In synthetic data with a known truth, the hybrid design evaluated on its own background is reported above its "
     "true performance in all 240 of its runs."),
    ("R9b", "Aggregated over the synthetic worlds, a model scored on its own background is reported higher than its "
            "truth in \\textbf{343/360 runs (95.3\\%; median $+0.187$, minimum $-0.031$)} (Figure~\\ref{fig:artefact}).",
     "In the synthetic worlds, a model of the hybrid design scored on its own background is reported higher than its "
     "truth in \\textbf{all 240 runs (median $+0.238$)}. For the random and TGB designs the gap is smaller and not always "
     "positive (52/60 and 51/60 runs; medians $+0.020$ and $+0.014$), and part of it comes from the difference between "
     "the observed and the true presence samples rather than from the background. Over all 360 runs of the two "
     "synthetic surfaces the count is 343/360 (Figure~\\ref{fig:artefact})."),
    ("R9c", "Right: own-background inflation across 360 synthetic runs of known ground truth (343/360 positive, median "
            "$+0.187$); the reported $+0.092$ ladder gain is marked.}",
     "Right: reported minus true AUC in 360 synthetic runs of known ground truth, by design (hybrid 240/240 runs "
     "positive; random 52/60; TGB 51/60).}"),
    ("R9d", "(95.3\\% of synthetic runs; for the hybrid design, 60/60 real seed$\\times$algorithm cells)",
     "(for the hybrid design, all 240 synthetic runs and 60/60 real seed$\\times$algorithm cells)"),
    # ---- C085: E235 sensitivity runs; Southern Mountains only for the uplands ----
    ("R6a", "Additional post hoc analyses show that this difference ranges from $+0.01$ to $+0.07$, depending on the "
            "zone definitions and the matching procedure; when distance to rivers is also matched, the difference falls "
            "to $+0.028$.",
     "The two pre-registered sensitivity definitions give smaller differences: $-0.002$ (95\\% CI $-0.047$ to $+0.046$; "
     "NOT SPECIFIC) with near land within 10~km and far land beyond 30~km, and $+0.016$ (INCONCLUSIVE) with 15 and "
     "30~km. Post hoc variants give $+0.005$ to $+0.07$, depending on the zone definitions and the matching procedure; "
     "when distance to rivers is also matched, the difference falls to $+0.028$. No direction is therefore claimed "
     "below 1,000~m."),
    ("R6b", "The far comparison land lies mostly in the Southern Mountains,",
     "In the uplands (400--1,000~m), the far comparison land lies mostly in the Southern Mountains,"),
    ("R6c", "In addition, the MaxEnt contingent fringe in Figure~\\ref{fig:priority} has a median elevation of 1,107~m,",
     "In addition, the contingent fringe of the MaxEnt maps (Section~\\ref{sec:prioritymaps}) has a median elevation "
     "of 1,107~m,"),
    # ---- C082: "pre-registered" ----
    ("R8", "The seven experiments that overturn the ladder were pre-registered (decision rules written before the runs) "
           "and are the paper's centrepiece.",
     "The eight experiments E217--E224 are the paper's centrepiece. E218 and E220--E224 were run from written designs "
     "with decision rules; only for E224 was the design committed to the public repository before the run (for the "
     "others, design and results were committed together), and E217 and E219 have no written design."),
    # ---- C086/C088: review history, internal targets, idioms in the text ----
    ("H1", "They are no longer the argument of the paper; they are the object the argument is about, so we summarise "
           "them in one table and two paragraphs and move the per-iteration detail to the supplement.",
     "They are the object of the analysis, not its result, so we summarise them in one table and two paragraphs and "
     "move the per-iteration detail to the supplement."),
    ("H2", "read from the analysis scripts rather than from any earlier prose.", "read from the analysis scripts."),
    ("H3", "Our minimum viable result (MVR) criterion is spatial AUC $> 0.75$ under block cross-validation.",
     "The original pipeline set itself a target of spatial AUC $> 0.75$ under block cross-validation."),
    ("H4", "Because the decimation is ours and not the paper's, E218 re-ran",
     "Because the lattice is a departure from the original pipeline, E218 re-ran"),
    ("H5", "\\paragraph{E224 -- a diagnostic we pre-registered and then withdrew (see Section~\\ref{sec:tgbnull}).}",
     "\\paragraph{E224 -- testing one explanation of the TGB null (Section~\\ref{sec:tgbnull}).}"),
    ("H6", "This is the performance we reported, and on its face it supports the conclusion we drew.",
     "Taken at face value, this sequence supports the reading that more realistic background designs improved the "
     "model."),
    ("H7", "The practical consequence is mild at the operating point our manuscript used.",
     "The practical consequence is mild at the largest share tried in E013 (0.30)."),
    ("H8", "We stopped at \\texttt{hard\\_frac} $= 0.30$ because our grid stopped there, not because the criterion said "
           "to;",
     "E013 stopped at \\texttt{hard\\_frac} $= 0.30$ because its grid stopped there, not because the criterion said to;"),
    ("H9", "and on the manuscript's own grid the criterion does not pick", "and on that grid the criterion does not pick"),
    ("H10", "The dotted vertical line marks the manuscript's stop at \\texttt{hard\\_frac} $= 0.30$; the range ends at "
            "$1.0$.}",
     "The dotted vertical line marks 0.30, the largest share tried in E013; the range tried here ends at $1.0$.}"),
    ("H11", "the true performance (dashed) does not, so the criterion has no interior optimum. Left: real data --- the "
            "two series move in opposite directions. Right: synthetic ground truth --- reported performance moves about "
            "twice as fast as truth.",
     "the comparison series (dashed) falls on real data and rises only about half as fast on synthetic data, so the "
     "criterion has no interior optimum. Left: real data, scored on a common background; the two series move in "
     "opposite directions. Right: synthetic ground truth; reported performance moves about twice as fast as true "
     "performance."),
    ("H12", "\\item \\textbf{Calibration gap between synthetic and real.} At the manuscript's operating point the "
            "selection criterion",
     "\\item \\textbf{Calibration gap between synthetic and real.} At a hard-negative share of 0.30 (the largest tried "
     "in E013) the selection criterion"),
    ("H13", " This limits transferring the E222 conclusions back to the real case, and is stated here rather than hidden "
            "in a supplement.",
     " This limits transferring the E222 conclusions back to the real case."),
    ("H14", "Its optimum is uncalibrated to truth (median $+0.50/+0.54$, borderline); we use it only to confirm "
            "direction, and we have aligned the audit flag with this narrative so the two do not contradict.",
     "It tracks the truth only loosely (median rank correlation $+0.50$ with map agreement and $+0.54$ with true AUC, "
     "E222), so we use it only to confirm direction."),
    ("H15", "\\item \\textbf{Treat the selection criterion as unbraked}: because it has no interior optimum,",
     "\\item \\textbf{Do not let the selection criterion decide where to stop}: because it has no interior optimum,"),
    ("H16", "we are not aware of a comparable dose-response measurement of reported against true performance",
     "we are not aware of a comparable measurement of reported against true performance"),
    ("H17", "\\subsection{Corrections to our own diagnostics: the volcano inventory and a correlation that is not "
            "consistent across seeds}",
     "\\subsection{Corrections to two volcanic diagnostics}"),
    ("H18", "Two corrections to our own diagnostics.\n", "Two diagnostics of the original pipeline are corrected here.\n"),
    ("H19", "This difference shows that the diagnostic result can change between seeds, so its interpretation must take "
            "into account the instability within a single analysis configuration and the importance of the "
            "seed-ensemble protocol.",
     "The re-run also uses the lattice-based background sampling of Section~\\ref{sec:refutation}, so the difference "
     "may come from the seeds or from the change of sampling; the two cannot be separated here. Either way, a single "
     "model instance is not a sound basis for the diagnostic."),
    ("H20", "The interpretation that low predicted suitability near volcanoes is evidence of buried settlement therefore "
            "cannot be sustained, because the model comparison on which it rests does not hold up under evaluation on "
            "a common background (Section~\\ref{sec:artefact}).",
     "Low predicted suitability near a volcano therefore cannot, by itself, be read as evidence of buried settlement."),
    # ---- C088: precision items ----
    ("P1", "(30 replicates, fitted in-bag and scored out-of-bag)",
     "(30 replicates, of which one was skipped because fewer than 50 out-of-bag sites remained, so 29 were used; "
     "fitted in-bag and scored out-of-bag)"),
    ("P2", "Thirty replicates is a small bootstrap;", "Twenty-nine replicates is a small bootstrap;"),
    ("P3", "In regional-bias regimes TGB is neutral rather than harmful (World C $-0.0010$,",
     "In regional-bias regimes TGB is neutral rather than harmful (true AUC, TGB minus random: World C $-0.0010$,"),
    ("P4", "TGB scored above random in 27/60 runs)", "in the two road-bias and misspecified surfaces TGB scored above "
     "random in 27/60 runs)"),
    ("P5", "(seed-averaged 0.751; best run 0.768;", "(seed-averaged 0.751; original run 0.768, highest of 20 further "
     "seeds 0.774;"),
    ("P6", "The core concentrates on the accessible lowlands that hold the known record; the contingent fringe is the "
           "remote upland on which background assumptions disagree (median 1.0~km from roads and 1,107~m elevation in "
           "the MaxEnt fringe).}",
     "The core concentrates on the accessible lowlands that hold most of the known records; the contingent fringe, "
     "where the background designs disagree, lies higher (median elevation 491~m for XGBoost and 1,107~m for "
     "MaxEnt).}"),
    ("P7", "We note what the criterion does \\emph{not} do: the design it selects (hybrid(1.0)) is never the truth-worst "
           "design (0/60 cases); the truth-worst is hybrid(0.0) in 50/60, hybrid(0.3) in 8 and hybrid(0.7) in 2 of 60 "
           "cases.",
     "We note what the criterion does \\emph{not} do. In the synthetic worlds, the design it selects (hybrid(1.0), "
     "the hybrid design with a hard-negative share of 1.0) never has the lowest true AUC of the designs compared "
     "(0/60 cases); the lowest is hybrid(0.0) in 50/60, hybrid(0.3) in 8 and hybrid(0.7) in 2 of 60 cases."),
    ("P8", "against the truth-best configuration on the same grid,",
     "against the configuration with the highest true AUC on the same grid,"),
    # ---- R11: terms a non-specialist cannot follow ----
    ("C1", "In Test 1, the correlation between predicted suitability and the distances to volcanoes, to roads and to the "
           "nearest site is examined;",
     "The original pipeline included diagnostic tests, named Test 1 and Test 3 in its code. In Test 1, the correlation "
     "between predicted suitability and the distances to volcanoes, to roads and to the nearest site is examined;"),
    ("C2", "Test 3 evaluates model performance in the least-surveyed road-distance quartile, and the volcano-related "
           "component of Test 1 is reported in Section~\\ref{sec:diagnostics}.",
     "Test 3 compared model performance across road-distance quartiles; it was scored on the design's own background, "
     "so its result is subject to the artefact this paper describes and is not used here. The volcano-related "
     "component of Test 1 is reported in Section~\\ref{sec:diagnostics}."),
    ("C3", "BPCB Jawa Timur surveys inform permitting and rescue prioritisation.",
     "surveys by the regional heritage preservation office (Balai Pelestarian Cagar Budaya, BPCB) inform permitting "
     "and rescue prioritisation."),
    # ---- style swaps S1-S10 (agenda, PI "ok") ----
    ("S1", "\\subsection{What presence--background models are, and why their evaluation is delicate}",
     "\\subsection{What presence--background models are, and why evaluating them is difficult}"),
    ("S2", "Three things here are ours, and none of them is ``background design matters''.",
     "This study makes three contributions."),
    ("S4", "We proposed an explanation for this uncomfortable null and nearly wrote it into the manuscript: because",
     "One possible explanation was that, because"),
    ("S5", "so TGB has nothing to cancel --- a tidy, ``predicted'' null.", "so TGB has nothing to cancel."),
    ("S6", "which is a stronger statement and needs no magnification.", "which is a stronger result."),
    ("S7", "turn over between seed pairs --- a single-seed map is not a defensible basis for a survey decision.",
     "turn over between seed pairs, so a single-seed map is not a sound basis for a survey decision."),
    ("S8", "\\subsection{The lesson: hold the evaluation background fixed}",
     "\\subsection{Hold the evaluation background fixed}"),
    ("S9", "That is the lesson of this line of work, and we learned it about our own model.",
     "We learned this from our own model."),
    # S10: em-dash asides in prose -> commas, parentheses, colons
    ("S10a", "we characterise the criterion a modeller actually optimises --- AUC on the design's own background --- and "
             "show",
     "we characterise the criterion a modeller actually optimises (AUC on the design's own background) and show"),
    ("S10b", "report as a result --- which is why this is a substantive hazard", "report as a result, which is why this "
     "is a substantive hazard"),
    ("S10c", "it is in the background design --- a claim about where a survey", "it is in the background design: a claim "
     "about where a survey"),
    ("S10d", "\\emph{within-benchmark contrasts} --- design against design, and own-background against "
             "common-background, under identical folds --- and not the absolute levels.",
     "\\emph{within-benchmark contrasts} (design against design, and own-background against common-background, under "
     "identical folds), not the absolute levels."),
    ("S10e", "For the hybrid design --- the last step of the ladder --- the inflation",
     "For the hybrid design, the last step of the ladder, the inflation"),
    ("S10f", "Held to a common (uniform) background --- the sampling on which the nulls were scored --- the E013 design "
             "family",
     "Held to a common (uniform) background, the sampling on which the nulls were scored, the E013 design family"),
    ("S10g", "not measured under identical sampling --- $0.751$ comes from the original pipeline and $0.706$ from the "
             "lattice-based benchmark (Section~\\ref{sec:refutation}) --- so the shrinkage",
     "not measured under identical sampling ($0.751$ comes from the original pipeline and $0.706$ from the "
     "lattice-based benchmark, Section~\\ref{sec:refutation}), so the shrinkage"),
    ("S10h", "The qualitative level claim --- that the environmental features capture structure beyond nearest-site "
             "interpolation --- survives",
     "The qualitative level claim, that the environmental features capture structure beyond nearest-site "
     "interpolation, survives"),
    ("S10i", "The selection criterion --- AUC computed on a design's own background --- has no interior optimum:",
     "The selection criterion (AUC computed on a design's own background) has no interior optimum:"),
    ("S10j", "the condition its theory requires --- a known, survey-biased observation process --- and TGB did not help.",
     "the condition its theory requires (a known, survey-biased observation process), and TGB did not help."),
    ("S10k", "per-run median ratio $2.00\\times$ --- we name the estimator because the three definitions bracket "
             "``about twice'').",
     "per-run median ratio $2.00\\times$; the three estimators bracket `about twice')."),
    ("S10l", "\\textbf{The truth-slope sign is regime-contingent} --- negative on real data,",
     "\\textbf{The truth-slope sign is regime-contingent}: negative on real data,"),
    # ---- T2 competing interests (JCAA's own sentence) ----
    ("T2", "The authors declare no competing interests.", "The authors have no competing interests to declare."),
    # ---- C089: authors' contributions and the AI declaration ----
    ("A1", "M.A.\\ conceived the research design, developed the methodology, implemented software, conducted formal "
           "analysis, curated data, created visualisations, and wrote the original draft.",
     "M.A.\\ conceived the study and its design, directed and checked the analyses (implemented with AI assistance, as "
     "declared below), curated the data, and prepared the manuscript."),
]

AI_OLD_START = "\\subsection*{AI Disclosure}\n"
AI_OLD_END = "All results, code and text were reviewed and verified by the authors, who take full responsibility for the content.\n"
AI_NEW = r"""\subsection*{Declaration of AI Use}

We used Claude (Anthropic) throughout this study.
It wrote and debugged the analysis code, produced the figures, drafted the designs of the experiments described in Section~\ref{sec:refutation}, checked the references against Crossref and the cited papers, and re-derived every reported number from the raw result files.
It also helped with the text.
The earlier drafts of the Methods, Results and Discussion were generated with Claude from our analyses and notes, and were then revised by us.
In this version, new passages were written by the first author in Indonesian and translated into English with Claude; the abstract, the conclusions and a number of corrections were drafted with Claude at the first author's request.
The research questions, the decisions about what to test and what to claim, and the interpretation are ours.
We checked every statement and number against the results and take full responsibility for the content.
"""

ALT_ANCHOR = "\\input{references_jcaa.tex}\n"
ALT_BLOCK = r"""
\section*{Figure Alt Text}
\begin{itemize}
  \item Figure 1: Relief map of East Java with sites and volcanoes.
  \item Figure 2: Bar charts of AUC and TSS across seven experiments.
  \item Figure 3: Two panels comparing own-background and common-background model scores.
  \item Figure 4: Reported versus true performance as hard negatives increase.
  \item Figure 5: Map of East Java coloured by agreement between designs.
  \item Figure 6: Map agreement rising with the number of seeds averaged.
  \item Figure 7: Land area by elevation band and matched suitability differences.
\end{itemize}
"""


def single_quotes(text):
    """T1: JCAA asks for single quotation marks: ``x'' -> `x' (no nested quotes occur in this file)."""
    return re.sub(r"``([^`']*?)''", r"`\1'", text)


def main():
    text = TEX.read_text(encoding="utf-8")
    bad = [(i, text.count(o)) for i, o, _ in EDITS if text.count(o) != 1]
    for i, old, new in EDITS:
        print(f"[{i}] {old.strip()[:110]!r}\n   -> {new.strip()[:110]!r}")
    i0, i1 = text.find(AI_OLD_START), text.find(AI_OLD_END)
    if i0 < 0 or i1 < 0 or text.count(AI_OLD_START) != 1:
        bad.append(("AI", "start/end not found once"))
    if text.count(ALT_ANCHOR) != 1:
        bad.append(("ALT", text.count(ALT_ANCHOR)))
    if bad:
        print("NOT FOUND EXACTLY ONCE:", bad)
        sys.exit(1)
    for _, old, new in EDITS:
        text = text.replace(old, new)
    i0, i1 = text.find(AI_OLD_START), text.find(AI_OLD_END)
    text = text[:i0] + AI_NEW + text[i1 + len(AI_OLD_END):]
    text = text.replace(ALT_ANCHOR, ALT_ANCHOR + ALT_BLOCK)
    n_q = len(re.findall(r"``[^`']*?''", text))
    text = single_quotes(text)
    print(f"T1: {n_q} double-quoted spans -> single quotes; AI declaration replaced; alt-text list added")
    if "--apply" in sys.argv:
        TEX.write_text(text, encoding="utf-8")
        print(f"applied {len(EDITS)} edits + AI + ALT + T1")
    else:
        print(f"dry run: {len(EDITS)} edits, all match once")


if __name__ == "__main__":
    main()
