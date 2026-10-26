#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v0.3 versions of manuscript Figures 2, 3, 4 and 6 (P2 / JCAA #280, round 2), 2026-10-26.

The v0.2 builders (build_v02_figures.py) still printed the wording Reviewer 1 objected to ("judged at home",
"dial", "published", "manuscript stop") and two internal targets ("MVR threshold", "TSS target"); ledger C086.
This script redraws the same data with plain labels and writes new files, so the v0.2 figures stay reproducible:
  fig02_progression_v03.png   (was fig3_auc_tss_progression.png)
  fig03_artefact_v03.png      (was fig14_artefact_two_panel.png; right panel now split by design, C087)
  fig04_criterion_v03.png     (was fig15_dose_response.png)
  fig06_seed_v03.png          (was fig17_seed_stabilisation.png; adds the disjoint-ensemble curve, C083/E236 F)
Every number is recomputed from the raw files named in each function. Run from repo root:
    python papers/P2_settlement_model/build_v03_figures_r2.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v02", HERE / "build_v02_figures.py")
v02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v02)          # palette, rcParams and helpers only; its main() is not run
plt = v02.plt
EXP, FIG, PAPER = v02.EXP, v02.FIG, v02.PAPER
BLUE, ORANGE, AQUA, INK, MUTED = v02.BLUE, v02.ORANGE, v02.AQUA, v02.INK, v02.MUTED
GREY = "#9a988f"
NAMES = {"xgboost": "XGBoost", "randomforest": "RandomForest", "maxent": "MaxEnt"}


def fig_progression() -> Path:
    m = pd.read_csv(PAPER / "tables_experiment_progression.csv")
    x = np.arange(len(m))
    fig, (axA, axT) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    w = 0.37
    for ax, col_x, col_r, lab, ylim in [(axA, "xgb_auc", "rf_auc", "Spatial AUC", (0.55, 0.82)),
                                        (axT, "xgb_tss", "rf_tss", "TSS", (0.25, 0.56))]:
        ax.bar(x - w / 2, m[col_x], width=w, color=BLUE, alpha=0.9, label="XGBoost")
        ax.bar(x + w / 2, m[col_r], width=w, color=ORANGE, alpha=0.9, label="RandomForest")
        ax.set_ylabel(lab)
        ax.set_ylim(*ylim)
        ax.grid(axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels(m["experiment"].tolist())
        ax.legend(fontsize=8, loc="upper left", frameon=False)
    axA.set_title("Reported AUC, E007$-$E013")
    axT.set_title("Reported TSS, E007$-$E013")
    fig.suptitle("Reported performance across the seven experiments of the original pipeline", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG / "fig02_progression_v03.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_artefact() -> Path:
    a = pd.read_csv(EXP / "E218_evaluation_artefact/results/e218_stageA_raw.csv")
    r = pd.read_csv(EXP / "E222_synthetic_ground_truth/results/e222_runs.csv")
    own_name = {"random": "uniform", "tgb": "tgb", "hybrid": "hybrid"}
    designs = ["random", "tgb", "hybrid"]
    rep, com = [], []
    for d in designs:
        dsub = a[a.train_design == d]
        rep.append(v02._algo_band(dsub[dsub.eval_background == own_name[d]], "auc"))
        com.append(v02._algo_band(dsub[dsub.eval_background == "uniform"], "auc"))
    rep, com = np.array(rep), np.array(com)

    infl = r["auc_own"] - r["auc_true"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.08, 1.0]})
    xs = np.arange(len(designs))
    axL.errorbar(xs, rep[:, 0], yerr=[rep[:, 0] - rep[:, 1], rep[:, 2] - rep[:, 0]], fmt="o-", color=BLUE, lw=2,
                 ms=6, capsize=3, label="Scored on its own background")
    axL.errorbar(xs, com[:, 0], yerr=[com[:, 0] - com[:, 1], com[:, 2] - com[:, 0]], fmt="s--", color=ORANGE,
                 lw=1.8, ms=6, capsize=3, label="Scored on a common background (uniform)")
    axL.set_xticks(xs)
    axL.set_xticklabels(["random\n(E007$-$E009)", "TGB\n(E010$-$E012)", "hybrid\n(E013)"])
    axL.set_ylim(0.66, 0.75)
    axL.set_ylabel("Spatial AUC (mean across algorithms)")
    axL.set_title("Real data (E218): only the hybrid design scores\nhigher on its own background", fontsize=10.5)
    axL.grid(axis="y")
    axL.legend(fontsize=8.5, loc="lower right", frameon=False)

    bins = np.histogram_bin_edges(infl, bins=40)
    hyb = infl[r.config == "hybrid"]
    oth = infl[r.config != "hybrid"]
    axR.hist([oth, hyb], bins=bins, stacked=True, color=[GREY, BLUE], alpha=0.85, edgecolor="white", lw=0.4,
             label=["random and TGB designs", "hybrid design"])
    axR.axvline(0, color=INK, lw=0.9)
    counts = {c: (int((infl[r.config == c] > 0).sum()), int((r.config == c).sum())) for c in ["hybrid", "random", "tgb"]}
    axR.text(0.99, 0.62, "runs above their true AUC:\n"
             f"hybrid {counts['hybrid'][0]}/{counts['hybrid'][1]}\n"
             f"random {counts['random'][0]}/{counts['random'][1]}\n"
             f"TGB {counts['tgb'][0]}/{counts['tgb'][1]}",
             transform=axR.transAxes, fontsize=8.5, va="top", ha="right", color=INK)
    axR.set_xlabel("Reported minus true AUC  (AUC$_{own}$ $-$ AUC$_{true}$)")
    axR.set_ylabel("Synthetic runs")
    axR.set_title(f"Synthetic worlds with known truth (E222, {len(r)} runs)", fontsize=10.5)
    axR.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.tight_layout()
    out = FIG / "fig03_artefact_v03.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_criterion() -> Path:
    real = pd.read_csv(EXP / "E218_evaluation_artefact/results/e218b_hardfrac_sweep.csv")
    syn = pd.read_csv(EXP / "E222_synthetic_ground_truth/results/e222_runs.csv")
    syn = syn[syn.config == "hybrid"]
    fig, (axR, axS) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    STOP = 0.30   # the largest share tried in E013

    def band(df, col):
        xs = sorted(df.hard_frac.unique())
        g = [df[df.hard_frac == h].groupby("algorithm")[col].mean() for h in xs]
        return xs, [x.mean() for x in g], [x.min() for x in g], [x.max() for x in g]

    for ax, df, c2, lab2, title in [
            (axR, real, "auc_common", "Scored on a common background (uniform)",
             "Real data (E218b): the two series\nmove in opposite directions"),
            (axS, syn, "auc_true", "True AUC (known ground truth)",
             "Synthetic worlds (E222): reported AUC rises\nabout twice as fast as the true AUC")]:
        xs, m, lo, hi = band(df, "auc_own")
        ax.plot(xs, m, "o-", color=BLUE, lw=2, ms=5, label="Scored on its own background")
        ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.15)
        xs, m, lo, hi = band(df, c2)
        ax.plot(xs, m, "s--", color=ORANGE, lw=1.8, ms=5, label=lab2)
        ax.fill_between(xs, lo, hi, color=ORANGE, alpha=0.15)
        ax.axvline(STOP, color=INK, lw=1.2, ls=":")
        ax.set_xlabel("Share of hard negatives in the background")
        ax.set_ylabel("AUC (mean across algorithms)")
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y")
    axR.text(STOP, axR.get_ylim()[1], "  largest share\n  tried in E013 (0.3)", fontsize=8, va="top", color=MUTED)
    axR.legend(fontsize=8.5, loc="lower left", frameon=False)
    axS.legend(fontsize=8.5, loc="upper left", frameon=False)
    fig.tight_layout()
    out = FIG / "fig04_criterion_v03.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_seed() -> Path:
    """Agreement with the 10-seed ensemble (E221, contains the k seeds) and between two disjoint k-seed ensembles
    (E236 part F, shares no seed)."""
    c = pd.read_csv(EXP / "E221_seed_ensemble_stability/results/e221_stabilisation_curve.csv")
    f = pd.read_csv(EXP / "E236_p2_presence_sensitivity/results/F_disjoint_full.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for algo, col in [("xgboost", BLUE), ("randomforest", ORANGE), ("maxent", AQUA)]:
        g = c[c.algorithm == algo].groupby("k")["jaccard_mean"].mean()
        ax.plot(g.index, g.values, "o-", color=col, lw=1.4, ms=4, alpha=0.55)
        h = f[f.algorithm == algo].groupby("k")["jaccard_mean"].mean()
        ax.plot(h.index, h.values, "s-", color=col, lw=2.2, ms=5, label=NAMES[algo])
    ax.plot([], [], "o-", color=MUTED, lw=1.4, alpha=0.55, label="k seeds vs the 10-seed ensemble (overlapping)")
    ax.plot([], [], "s-", color=MUTED, lw=2.2, label="two disjoint k-seed ensembles")
    ax.set_xlabel("Number of seeds averaged (k)")
    ax.set_ylabel("Jaccard overlap of top-decile maps")
    ax.set_title("Averaging seeds makes the priority map more stable", fontsize=11)
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    out = FIG / "fig06_seed_v03.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    import sys
    jobs = {"2": fig_progression, "3": fig_artefact, "4": fig_criterion, "6": fig_seed}
    for k in (sys.argv[1:] or jobs):
        p = jobs[k]()
        print(f"  Figure {k}: {p.name} ({p.stat().st_size / 1024:.0f} KB)")
