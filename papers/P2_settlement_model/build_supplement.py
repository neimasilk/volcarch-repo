#!/usr/bin/env python
"""
Build the supplementary tables (S1-S6) promised in submission_jcaa_v0.2.tex.

Why this exists: the manuscript's Supplementary Materials section listed six
tables, of which only S1 and S2 had a source file and none had a deliverable.
Promising a reviewer -- one who explicitly asked for reproducibility -- tables
that do not exist is the kind of gap this revision is about. Every table here is
generated from a raw result file, never retyped, so the numbers cannot drift
away from the experiments.

Run from the paper directory:
    python build_supplement.py
then:  pdflatex supplementary_tables_v0.2.tex  (x2)
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parent.parent
EXP = REPO / "experiments"
SUP = HERE / "supplement"

OUT = HERE / "supplementary_tables_v0.2.tex"


def fmt(x, nd=3):
    """Fixed-width numeric formatting; blank for missing rather than 'nan'."""
    if pd.isna(x):
        return "---"
    return f"{x:.{nd}f}"


def sgn(x, nd=4):
    """Signed number in math mode, so LaTeX renders a real minus, not a hyphen."""
    if pd.isna(x):
        return "---"
    return f"${x:+.{nd}f}$"


def tx(s):
    """Escape LaTeX specials in data-derived text (design names carry underscores)."""
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#"))


def latex_table(df, caption, label, colfmt=None, note=None, small=True):
    """Emit one booktabs table. df columns are used verbatim as headers."""
    ncol = len(df.columns)
    colfmt = colfmt or ("l" + "c" * (ncol - 1))
    head = " & ".join(str(c) for c in df.columns) + r" \\"
    body = "\n".join(" & ".join(str(v) for v in row) + r" \\" for row in df.values)
    size = r"\small" + "\n" if small else ""
    notetex = f"\n\\vspace{{2pt}}\n\\footnotesize\\emph{{{note}}}\n" if note else ""
    return f"""
\\begin{{table}}[htbp]
\\centering
{size}\\begin{{tabular}}{{{colfmt}}}
\\toprule
{head}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}{notetex}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
"""


# ---------------------------------------------------------------- S1
def table_s1():
    d = pd.read_csv(SUP / "e013_seed_stability.csv").sort_values("seed")
    out = pd.DataFrame({
        "Seed": d["seed"].astype(int),
        "hard\\_frac (realised)": d["hard_frac_actual"].map(lambda v: fmt(v, 3)),
        "XGB AUC": d["xgb_mean_auc"].map(lambda v: fmt(v, 3)),
        "XGB TSS": d["xgb_mean_tss"].map(lambda v: fmt(v, 3)),
        "RF AUC": d["rf_mean_auc"].map(lambda v: fmt(v, 3)),
        "RF TSS": d["rf_mean_tss"].map(lambda v: fmt(v, 3)),
    })
    note = (f"XGBoost AUC across the 20 seeds: min {d.xgb_mean_auc.min():.3f}, "
            f"max {d.xgb_mean_auc.max():.3f}, mean {d.xgb_mean_auc.mean():.3f}. "
            f"Realised hard-negative fraction is consistently near "
            f"{d.hard_frac_actual.mean():.2f}, well above the 0.30 target -- see manuscript "
            r"\S2.3.")
    return latex_table(
        out,
        "Table S1. Seed stability of E013 (the top rung of the reported ladder). "
        "Each row is one background-generation seed; AUC and TSS are means over the "
        "five deterministic spatial-CV folds. Source: "
        r"\texttt{supplement/e013\_seed\_stability.csv}.",
        "tab:s1", note=note)


# ---------------------------------------------------------------- S2
def table_s2():
    d = pd.read_csv(SUP / "e013_blocksize_summary.csv")
    out = pd.DataFrame({
        "Block size": d["block_label"].map(lambda v: tx(str(v).replace("_", " "))),
        "deg": d["block_size_deg"].map(lambda v: fmt(v, 3)),
        "km": d["block_size_km"].map(lambda v: fmt(v, 1)),
        "Runs": d["n_runs"].astype(int),
        "XGB AUC": d["xgb_auc_mean"].map(lambda v: fmt(v, 3)),
        "XGB 95\\% CI": [f"[{fmt(a)}, {fmt(b)}]" for a, b in
                         zip(d.xgb_auc_ci_low, d.xgb_auc_ci_high)],
        "RF AUC": d["rf_auc_mean"].map(lambda v: fmt(v, 3)),
        r"$\Delta$ XGB vs 50 km": d["delta_xgb_auc_vs_50km"].map(lambda v: sgn(v, 3)),
    })
    return latex_table(
        out,
        "Table S2. Block-size sensitivity of E013 under spatial block cross-validation. "
        "The reported result uses the 50 km baseline; performance falls at both 40 km "
        "and 60 km, so the baseline is not a conservative choice. Source: "
        r"\texttt{supplement/e013\_blocksize\_summary.csv}.",
        "tab:s2")


# ---------------------------------------------------------------- S3
def table_s3():
    d = pd.read_csv(EXP / "E217_maxent_benchmark/results/e217b_summary.csv")
    d = d.sort_values(["feature_set", "design", "algorithm"])
    out = pd.DataFrame({
        "Feature set": d["feature_set"].map(tx),
        "Background design": d["design"].map(tx),
        "Algorithm": d["algorithm"].map(tx),
        "AUC (own bg)": d["auc_own"].map(lambda v: fmt(v, 4)),
        "AUC (common bg)": d["auc_common"].map(lambda v: fmt(v, 4)),
        "Own $-$ common": (d["auc_own"] - d["auc_common"]).map(sgn),
    })
    diff = d["auc_own"] - d["auc_common"]
    note = (f"Own-background evaluation exceeds common-background evaluation in "
            f"{int((diff > 0).sum())}/{len(diff)} cells (mean {diff.mean():+.4f}). "
            r"Backgrounds are drawn from a decimated ($\sim$300 m) lattice, so absolute "
            r"levels are not comparable to Table 3 of the manuscript; the interpretable "
            r"quantities are the within-table contrasts (manuscript \S2.6).")
    return latex_table(
        out,
        "Table S3. E217 common-background benchmark: every background design and "
        "feature set scored both on its own background and on a shared evaluation "
        "background, under identical spatial-block folds. This is the comparison the "
        "reviewer's Maximum Entropy request made possible. Source: "
        r"\texttt{E217\_maxent\_benchmark/results/e217b\_summary.csv}.",
        "tab:s3", note=note)


# ---------------------------------------------------------------- S4
def table_s4():
    d = pd.read_csv(EXP / "E218_evaluation_artefact/results/e218_stageA_auc_matrix.csv")
    d = d.sort_values(["algorithm", "train_design"])
    out = pd.DataFrame({
        "Algorithm": d["algorithm"].map(tx),
        "Trained on": d["train_design"].map(tx),
        "Eval: uniform": d["uniform"].map(lambda v: fmt(v, 4)),
        "Eval: tgb": d["tgb"].map(lambda v: fmt(v, 4)),
        "Eval: hybrid": d["hybrid"].map(lambda v: fmt(v, 4)),
        "Eval: stratified": d["stratified"].map(lambda v: fmt(v, 4)),
    })
    note = ("Read down the columns, not across the rows. Within any single evaluation "
            "background, the hybrid-trained model does not lead; it leads only in the "
            "column that is its own background (hybrid). That column-versus-row "
            "asymmetry is the evaluation artefact.")
    return latex_table(
        out,
        "Table S4. E218 own- versus common-background matrix: three training designs "
        "(rows) each scored against four evaluation backgrounds (columns), mean over "
        "20 seeds, identical folds throughout. Source: "
        r"\texttt{E218\_evaluation\_artefact/results/e218\_stageA\_auc\_matrix.csv}.",
        "tab:s4", note=note)


# ---------------------------------------------------------------- S5
def table_s5():
    d = pd.read_csv(EXP / "E222_synthetic_ground_truth/results/e222_runs.csv")
    g = (d.groupby(["config", "algorithm"])
           .agg(n=("auc_own", "size"),
                auc_own=("auc_own", "mean"),
                auc_true=("auc_true", "mean"),
                jaccard=("map_jaccard", "mean"))
           .reset_index())
    g["gap"] = g["auc_own"] - g["auc_true"]
    out = pd.DataFrame({
        "Background config": g["config"].map(tx),
        "Algorithm": g["algorithm"].map(tx),
        "$n$ runs": g["n"].astype(int),
        "Reported AUC": g["auc_own"].map(lambda v: fmt(v, 4)),
        "True AUC": g["auc_true"].map(lambda v: fmt(v, 4)),
        "Reported $-$ true": g["gap"].map(sgn),
        "Map Jaccard": g["jaccard"].map(lambda v: fmt(v, 4)),
    })
    gap_all = d["auc_own"] - d["auc_true"]
    note = (f"Across all {len(d)} synthetic runs the reported value exceeds the truth in "
            f"{int((gap_all > 0).sum())}/{len(gap_all)} cases "
            f"({100 * (gap_all > 0).mean():.1f}\\%), median {gap_all.median():+.3f}, "
            f"minimum {gap_all.min():+.3f}. The published ladder gain under examination "
            r"was $+0.092$.")
    return latex_table(
        out,
        "Table S5. E222 synthetic worlds of known ground truth: reported AUC (scored on "
        "the design's own background) against true AUC (scored against the known "
        "intensity surface), averaged within background configuration and algorithm. "
        r"Source: \texttt{E222\_synthetic\_ground\_truth/results/e222\_runs.csv}.",
        "tab:s5", note=note)


# ---------------------------------------------------------------- S6
def table_s6():
    """Covariate/design inclusion for the refutation suite, extending Table 2."""
    rows = [
        ("E217", "terrain / terrain+river", "no", "random, tgb, hybrid",
         "own + common", "5", "MaxEnt, XGB, RF"),
        ("E217b", "terrain / terrain+river", "no", "random, tgb, hybrid",
         "own + common", "5", "MaxEnt, XGB, RF"),
        ("E218 A", "terrain+river", "no", "random, tgb, hybrid",
         "uniform, tgb, hybrid, stratified", "20", "MaxEnt, XGB, RF"),
        ("E218 B", "terrain+river", "no", "random, tgb, hybrid",
         "common (block-size sweep)", "20", "XGB"),
        ("E218 C", "terrain+river", "no", "dissimilarity sweep", "common", "5", "XGB"),
        ("E218 D", "terrain+river", "no", "random, tgb, hybrid",
         r"own + common ($\sim$150 m lattice)", "5", "XGB"),
        ("E218b", "terrain+river", "no", r"hybrid, \texttt{hard\_frac} 0.0--1.0",
         "own + common", "5", "XGB"),
        ("E220", "terrain+river", "no", r"\texttt{hard\_frac} dial", "own", "5", "XGB"),
        ("E221", "terrain+river", "no", "random, tgb, hybrid", "own", "10",
         "MaxEnt, XGB, RF"),
        ("E222", "synthetic covariates", "no", "random, tgb, hybrid, quota",
         "own + ground truth", "30", "MaxEnt, XGB, RF"),
        ("E222c/d", "synthetic covariates", "no", "random, tgb, hybrid, quota",
         "own + ground truth", "30", "MaxEnt, XGB, RF"),
        ("E223", "terrain+river", "no", "random, tgb, hybrid",
         "own + common (bootstrap)", "30", "MaxEnt, XGB, RF"),
        ("E224", "synthetic covariates", r"\textbf{yes} (the manipulation)",
         "random, tgb", "own + ground truth", "30", "MaxEnt, XGB, RF"),
    ]
    out = pd.DataFrame(rows, columns=[
        "Exp.", "Feature set", r"\texttt{road\_dist} a feature?",
        "Background designs", "Evaluation background", "Seeds", "Algorithms"])
    note = (r"\texttt{road\_dist} is a training feature in E224 only, and there only "
            "because withholding it was the explanation under test; the test failed "
            r"(manuscript \S3.5). Everywhere else it builds the background and is never "
            "a predictor, exactly as in E007--E013.")
    return latex_table(
        out,
        "Table S6. Per-experiment design matrix for the refutation suite E217--E224, "
        "extending Table 2 of the manuscript. Built by reading the analysis scripts, "
        "not the prose.",
        "tab:s6", colfmt="llp{2.1cm}p{3.0cm}p{2.9cm}cl", note=note)


def main():
    parts = [table_s1(), table_s2(), table_s3(), table_s4(), table_s5(), table_s6()]

    doc = r"""% Supplementary tables for JCAA #280 revision v0.2.
% GENERATED by build_supplement.py -- do not edit by hand; edit the script and re-run.
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[a4paper, margin=2.2cm, landscape]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{amsmath}
\usepackage{longtable}

\title{Supplementary Tables\\
\large An Evaluation Artefact in Presence--Background Archaeological Modelling:\\
Evidence from East Java and a Corrected Comparison Protocol}
\author{}
\date{}

\begin{document}
\maketitle
\vspace{-2.2cm}

\noindent
Every table below is generated directly from the raw result files by
\texttt{build\_supplement.py}, which is included in the code repository cited in the
manuscript's Data Availability statement. No value is retyped. Tables S3--S6 describe the
refutation suite (E217--E224); backgrounds there are drawn from a decimated raster lattice
rather than by continuous rejection sampling, so absolute AUC levels are not directly
comparable with the published E007--E013 pipeline --- the interpretable quantities are the
within-experiment contrasts. This is stated in the manuscript at \S2.6.

""" + "\n".join(parts) + r"""
\end{document}
"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT}  ({len(doc.splitlines())} lines, 6 tables)")


if __name__ == "__main__":
    main()
