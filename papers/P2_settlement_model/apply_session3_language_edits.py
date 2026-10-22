"""P2 round 2, writing session 3 (2026-10-22): language edits approved by the PI as one list.

R1-1 asks that the paper stand alone (no talk of reviewers, revisions or earlier versions); R1-4 names idioms
("dial", "judging at home", "Jaccard said no"). The sentences that need new content are written by the PI
(PI_PROSE_LOG_R2.md); this script applies only deletions of review history, "published" -> "original"/"reported"
(P2 was never published) and the idiom substitutions, each listed so the log can cite it.

usage: python apply_session3_language_edits.py          # dry run: show every old -> new, check each matches once
       python apply_session3_language_edits.py --apply  # write submission_jcaa_v0.3.tex
"""
import sys
from pathlib import Path

TEX = Path(__file__).parent / "submission_jcaa_v0.3.tex"

# (id, old, new). Every old string must occur exactly once in the file.
EDITS = [
    # --- review history: deletions and word swaps (R1-1) ---
    ("H1", "Following a reviewer's taxonomy, a site may be absent", "A site may be absent"),
    ("H2", "We state this restraint up front because the earlier version of this manuscript overreached it, and the retraction of that overreach is reported in Section~4.\n", ""),
    ("H3", " because a reviewer of the previous version observed --- correctly --- that this family", " because this family"),
    ("H4", "The honest shape of this revision is worth stating.\nA title can be softened to match weaker evidence, but when the evidence supports a different proposition, softening is not enough; the proposition has to be replaced.\nRetitling from a bias-correction claim to an evaluation-artefact finding is that kind of replacement, and we believe it is what the reviewer's request required.\n\n", ""),
    ("H5", r"elith2006} added at a reviewer's request.", r"elith2006}."),
    ("H6", "that we originally reported as evidence", "that we initially read as evidence"),
    ("H7", "This is also the honest answer to the question of what makes the approach specifically archaeological:", "This is also what makes the approach specifically archaeological:"),
    ("H8", "benchmark (answers the Maximum Entropy request).}", "benchmark.}"),
    ("H9", "; the submitted version quoted these gaps against the best run (0.768), which overstated each by 0.017.", "."),
    ("H10", r"We note what the criterion does \emph{not} do, correcting an earlier internal claim:", r"We note what the criterion does \emph{not} do:"),
    ("H11", "% TODO-PI (history passage, writing pack section 1): this sentence still mentions the reviewers. \"Three\" -> \"Two\"\n% after the terrain-matched control moved to Section sec:elevation (mechanical, 2026-10-21).\nTwo corrections to our own diagnostics, none prompted by the reviewers.", "Two corrections to our own diagnostics."),
    ("H12", r"in the reviewer's own terms (Section~\ref{sec:taxonomy}: mechanism 3 cannot be read off a suitability surface).", r"in the terms of Section~\ref{sec:taxonomy}: mechanism 3 cannot be read off a suitability surface."),
    ("H13", "With the site-prediction claim withdrawn, the output is", "The output is"),
    ("H14", "This restraint is now stated in the manuscript in these words, and it is a constraint the revised paper takes seriously rather than a caveat appended to an overclaim.\n", ""),
    ("H15", "two conditions this revision makes explicit", "two conditions this study makes explicit"),
    ("H16", "the constructive contribution of the revision:", "the constructive contribution of this study:"),
    ("H17", "which requires the site-prediction claim we have withdrawn,", "which requires site-level prediction, which this design does not support,"),
    # --- "published": P2 was never published ---
    ("P1", "One deviation from the published pipeline", "One deviation from the original pipeline"),
    ("P2", "comparable to the published E007--E013 values", "comparable to the original E007--E013 values"),
    ("P3", "experiments of the published pipeline (E007--E013", "experiments of the original pipeline (E007--E013"),
    ("P4", "We reject the published ladder gain", "We reject the reported ladder gain"),
    ("P5", "comes from the published pipeline", "comes from the original pipeline"),
    ("P6", "the published $+0.092$ ladder gain is marked", "the reported $+0.092$ ladder gain is marked"),
    ("P7", "is the original published pipeline", "is the original pipeline"),
    # --- idioms (R1-4) ---
    ("I1", r"along a background-design dial (Section~\ref{sec:nooptimum})", r"as the share of hard negatives in the background is varied (Section~\ref{sec:nooptimum})"),
    ("I2", "We sweep the hard-negative dial and ask whether the AUC-on-own-background criterion ever tells a researcher to stop short of the dial's edge.", "We vary the share of hard negatives in the background from 0 to 1 and ask whether the AUC-on-own-background criterion ever tells a researcher to stop before the end of that range."),
    ("I3", "its maximum always sits at the edge of whatever dial is swept", "its maximum always sits at the end of whatever range is swept"),
    ("I4", "rises monotonically along the hard-negative dial", "rises monotonically with the hard-negative share"),
    ("I5", "if the dial were extended, the cost is large", "if the hard-negative share were pushed further, the cost is large"),
    ("I6", "keeps rising to the edge of the dial;", "keeps rising to the end of the hard-negative range;"),
    ("I7", r"as fast as the truth along the dial (", r"as fast as the truth as the hard-negative share rises ("),
    ("I8", "justify any stop on the design dial externally, not by the criterion's own maximum.", "justify where to stop varying a background-design parameter on external grounds, not by the criterion's own maximum."),
    ("I9", "comes from being judged at home.", "comes from being scored on its own background."),
    ("I10", "The advantage evaporates the moment it is judged anywhere but at home.", "The advantage disappears as soon as a design is scored on any background other than its own."),
    ("I11", "How large is judging at home?", "How large is own-background inflation?"),
    ("I12", "the top rung of the ladder", "the last step of the ladder"),
    ("I13", "This home-court inflation is specific to that design", "This own-background inflation is specific to that design"),
    ("I14", "The TGB rungs of the ladder (E010--E012) are therefore invalidated not by home-court inflation", "The TGB steps of the ladder (E010--E012) are therefore invalidated not by own-background inflation"),
    ("I15", "Only the hybrid design scores higher when judged at home.", "Only the hybrid design scores higher on its own background."),
    ("I16", "and map Jaccard said no.", "and it did not support that reading."),
    ("I17", "tested in its home regime (regional survey bias)", "tested in the regime it was built for (regional survey bias)"),
]


def main():
    text = TEX.read_text(encoding="utf-8")
    bad = [(i, text.count(o)) for i, o, _ in EDITS if text.count(o) != 1]
    for i, old, new in EDITS:
        print(f"[{i}] {old.strip()[:150]!r}\n   -> {new.strip()[:150]!r}")
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
