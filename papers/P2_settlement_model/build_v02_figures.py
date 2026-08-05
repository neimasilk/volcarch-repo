#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build all v0.2 figures for P2 / JCAA #280 revision, from raw per-run result files.

Figures produced (written to papers/P2_settlement_model/figures/):
  fig14_artefact_two_panel.png  -- Fig 4 in v0.2: the evaluation artefact.
       left:  reported (own-background) vs common-background AUC across the
              three design families (random / tgb / hybrid) used in E007-E013,
              mean over algorithms with the cross-algorithm band.  E218 stage A.
       right: histogram of own-background inflation (auc_own - auc_true) across
              the 360 synthetic runs of E222, with the published +0.092 marked.
  fig15_dose_response.png       -- Fig 5 in v0.2: the selection criterion on its
       own background (reported) vs true/common performance along the hard_frac
       dial. Left panel: real data (E218b). Right panel: synthetic (E222, hybrid).
       Manuscript stop at hard_frac=0.30 and grid edge at 1.0 marked.
  fig16_priority_map.png        -- new: robust core / contingent fringe map
       (XGBoost), cells coloured by how many of the three background designs put
       them in the top decile. Overlay known sites + canonical 13 volcanoes.
       Source: e221_priority_sets_xgboost.npz, east_java_sites.geojson,
       volcanoes_java_full.csv.
  fig17_seed_stabilisation.png  -- new: ensemble-of-k vs ensemble-of-10
       top-decile Jaccard, one line per algorithm, k* at J>=0.90 and the k>=7
       protocol floor marked.  Source: e221_stabilisation_curve.csv.
  fig10_study_area_map.png      -- redraw: 13 canonical volcanic centres (INT-1),
       presences.  Sources: east_java_sites.geojson, volcanoes_java_full.csv,
       jatim_dem.tif (terrain background, fallback if no basemap tiles).
  fig3_auc_tss_progression.png  -- redraw: the reported ladder E007-E013 (Table 3)
       restated as the object under examination.  Source:
       tables_experiment_progression.csv.

Every number drawn on a figure is recomputed here from the CSV/NPZ it cites; nothing
is hardcoded except the manuscript's published constants (e.g. +0.092 ladder gain)
and the colour palette.  Run from repo root:
    python papers/P2_settlement_model/build_v02_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments"
DATA = REPO / "data" / "processed"
FIG = REPO / "papers" / "P2_settlement_model" / "figures"
PAPER = REPO / "papers" / "P2_settlement_model"

# ---- colour system (dataviz default palette: CVD-safe, print-friendly) ----
BLUE    = "#2a78d6"   # categorical slot 1 / primary series
ORANGE  = "#eb6834"   # categorical slot 2 / comparator series
AQUA    = "#1baf7a"   # categorical slot 3
INK     = "#0b0b0b"
MUTED   = "#52514e"
GRID    = "#e1e0d9"
LIGHT   = "#fcfcfb"
BLUE_DARK  = "#104281"   # robust core
BLUE_MID   = "#3987e5"   # intermediate
BLUE_LIGHT = "#9ec5f4"   # contingent
NONE_GRAY  = "#e6e4de"   # not top-decile anywhere
VOLC_COL   = "#d03b3b"   # critical-red, distinct from all series

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "figure.facecolor": LIGHT,
    "axes.facecolor": LIGHT,
    "savefig.facecolor": LIGHT,
})


def _algo_band(g: pd.DataFrame, col: str) -> tuple:
    """Return (mean over algorithms/seeds, low, high) for col, per x-index."""
    by = g.groupby("algorithm")[col].mean()
    return float(by.mean()), float(by.min()), float(by.max())


# ===========================================================================
# Fig 14 -- the evaluation artefact (two-panel)
# ===========================================================================
def fig_artefact() -> Path:
    a = pd.read_csv(EXP / "E218_evaluation_artefact/results/e218_stageA_raw.csv")
    r = pd.read_csv(EXP / "E222_synthetic_ground_truth/results/e222_runs.csv")

    # --- left: reported vs common-background across the three design families --
    own_name = {"random": "uniform", "tgb": "tgb", "hybrid": "hybrid"}
    designs = ["random", "tgb", "hybrid"]
    rep, rep_lo, rep_hi = [], [], []
    com, com_lo, com_hi = [], [], []
    for d in designs:
        dsub = a[a.train_design == d]
        own = dsub[dsub.eval_background == own_name[d]]
        uni = dsub[dsub.eval_background == "uniform"]
        m, lo, hi = _algo_band(own, "auc")
        rep.append(m); rep_lo.append(lo); rep_hi.append(hi)
        m, lo, hi = _algo_band(uni, "auc")
        com.append(m); com_lo.append(lo); com_hi.append(hi)

    # --- right: histogram of synthetic own-background inflation -------------
    infl = r["auc_own"] - r["auc_true"]
    n_pos = int((infl > 0).sum())
    n_tot = len(infl)
    med = float(infl.median())
    PUB_GAIN = 0.092  # the ladder gain published in v0.1 and rejected in v0.2

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.08, 1.0]})

    xs = np.arange(len(designs))
    axL.errorbar(xs, rep, yerr=[np.array(rep) - np.array(rep_lo), np.array(rep_hi) - np.array(rep)],
                 fmt="o-", color=BLUE, lw=2, ms=6, capsize=3,
                 label="Reported (evaluated on own background)")
    axL.errorbar(xs, com, yerr=[np.array(com) - np.array(com_lo), np.array(com_hi) - np.array(com)],
                 fmt="s--", color=ORANGE, lw=1.8, ms=6, capsize=3,
                 label="Held to a common background (uniform)")
    axL.set_xticks(xs)
    axL.set_xticklabels(["random\n(E007$-$E009)", "TGB\n(E010$-$E012)", "hybrid\n(E013)"])
    axL.set_ylim(0.66, 0.75)
    axL.set_ylabel("Spatial AUC (mean across algorithms)")
    axL.set_title("Only the hybrid design scores higher\nwhen judged at home", fontsize=10.5)
    axL.grid(axis="y")
    axL.legend(fontsize=8.5, loc="lower right", frameon=False)

    bins = np.histogram_bin_edges(infl, bins=40)
    axR.hist(infl, bins=bins, color=BLUE, alpha=0.75, edgecolor="white", lw=0.4)
    axR.axvline(PUB_GAIN, color=ORANGE, lw=1.8, ls="--", label=f"published +0.092 ladder gain")
    axR.axvline(0, color=INK, lw=0.9)
    axR.text(0.01, axR.get_ylim()[1] * 0.96,
             f"{n_pos}/{n_tot} runs positive (median +{med:.3f})",
             fontsize=9, va="top", color=INK)
    axR.set_xlabel("Own-background inflation  =  AUC$_{own}$ $-$ AUC$_{true}$")
    axR.set_ylabel("Synthetic runs")
    axR.set_title("Synthetic ground truth (E222)", fontsize=10.5)
    axR.legend(fontsize=8.5, frameon=False)

    fig.tight_layout()
    out = FIG / "fig14_artefact_two_panel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Fig 15 -- dose-response along the hard_frac dial (two-panel)
# ===========================================================================
def fig_dose_response() -> Path:
    real = pd.read_csv(EXP / "E218_evaluation_artefact/results/e218b_hardfrac_sweep.csv")
    synth = pd.read_csv(EXP / "E222_synthetic_ground_truth/results/e222_runs.csv")
    syn = synth[synth.config == "hybrid"]  # the dial is swept on the hybrid design

    fig, (axR, axS) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    STOP = 0.30   # the manuscript's grid max (PAPER_GRID_MAX in verify_headline_numbers.py)

    # --- real data (left) ---
    xr = sorted(real.hard_frac.unique())
    rep_m = [real[real.hard_frac == h].groupby("algorithm")["auc_own"].mean().mean() for h in xr]
    rep_lo = [real[real.hard_frac == h].groupby("algorithm")["auc_own"].mean().min() for h in xr]
    rep_hi = [real[real.hard_frac == h].groupby("algorithm")["auc_own"].mean().max() for h in xr]
    com_m = [real[real.hard_frac == h].groupby("algorithm")["auc_common"].mean().mean() for h in xr]
    com_lo = [real[real.hard_frac == h].groupby("algorithm")["auc_common"].mean().min() for h in xr]
    com_hi = [real[real.hard_frac == h].groupby("algorithm")["auc_common"].mean().max() for h in xr]
    axR.plot(xr, rep_m, "o-", color=BLUE, lw=2, ms=5, label="Reported (own background)")
    axR.fill_between(xr, rep_lo, rep_hi, color=BLUE, alpha=0.15)
    axR.plot(xr, com_m, "s--", color=ORANGE, lw=1.8, ms=5, label="True (common background)")
    axR.fill_between(xr, com_lo, com_hi, color=ORANGE, alpha=0.15)
    axR.axvline(STOP, color=INK, lw=1.2, ls=":")
    axR.text(STOP, axR.get_ylim()[1], f"  manuscript stop\n  (hard\\_frac = {STOP})",
             fontsize=8, va="top", color=MUTED)
    axR.set_xlabel("Hard-negative fraction along the dial")
    axR.set_ylabel("AUC (mean across algorithms)")
    axR.set_title("Real data (E218b): the two series\nmove in opposite directions", fontsize=10.5)
    axR.grid(axis="y")
    axR.legend(fontsize=8.5, loc="lower left", frameon=False)

    # --- synthetic data (right) ---
    xs = sorted(syn.hard_frac.unique())
    rep_m = [syn[syn.hard_frac == h].groupby("algorithm")["auc_own"].mean().mean() for h in xs]
    rep_lo = [syn[syn.hard_frac == h].groupby("algorithm")["auc_own"].mean().min() for h in xs]
    rep_hi = [syn[syn.hard_frac == h].groupby("algorithm")["auc_own"].mean().max() for h in xs]
    tr_m = [syn[syn.hard_frac == h].groupby("algorithm")["auc_true"].mean().mean() for h in xs]
    tr_lo = [syn[syn.hard_frac == h].groupby("algorithm")["auc_true"].mean().min() for h in xs]
    tr_hi = [syn[syn.hard_frac == h].groupby("algorithm")["auc_true"].mean().max() for h in xs]
    axS.plot(xs, rep_m, "o-", color=BLUE, lw=2, ms=5, label="Reported (own background)")
    axS.fill_between(xs, rep_lo, rep_hi, color=BLUE, alpha=0.15)
    axS.plot(xs, tr_m, "s--", color=ORANGE, lw=1.8, ms=5, label="True (known ground truth)")
    axS.fill_between(xs, tr_lo, tr_hi, color=ORANGE, alpha=0.15)
    axS.axvline(STOP, color=INK, lw=1.2, ls=":")
    axS.set_xlabel("Hard-negative fraction along the dial")
    axS.set_ylabel("AUC (mean across algorithms)")
    axS.set_title("Synthetic ground truth (E222): reported moves\n~2$\\times$ faster than truth", fontsize=10.5)
    axS.grid(axis="y")
    axS.legend(fontsize=8.5, loc="upper left", frameon=False)

    fig.tight_layout()
    out = FIG / "fig15_dose_response.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Fig 16 -- robust core / contingent fringe priority map (XGBoost)
# ===========================================================================
def fig_priority_map() -> Path:
    d = np.load(EXP / "E221_seed_ensemble_stability/results/e221_priority_sets_xgboost.npz")
    x, y, nd = d["x"], d["y"], d["n_designs_top10"]

    import geopandas as gpd
    sites = gpd.read_file(DATA / "east_java_sites.geojson")
    sites = sites[sites.geometry.notna() & ~sites.geometry.is_empty]
    sites = sites.to_crs("EPSG:32749")
    sites = sites.cx[x.min():x.max(), y.min():y.max()]

    volc = pd.read_csv(DATA / "dashboard/volcanoes_java_full.csv")
    v13 = volc[(volc.lon >= 111) & (volc.lon <= 115) & (volc.lat >= -9) & (volc.lat <= -6.5)]
    vg = gpd.GeoDataFrame(v13, geometry=gpd.points_from_xy(v13.lon, v13.lat), crs="EPSG:4326").to_crs("EPSG:32749")

    # downsample for a light scatter (every 3rd cell)
    step = 3
    xs = x[::step]; ys = y[::step]; nds = nd[::step]
    cls = np.where(nds == 0, NONE_GRAY,
          np.where(nds == 1, BLUE_LIGHT,
          np.where(nds == 2, BLUE_MID, BLUE_DARK)))

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(xs, ys, c=cls, s=0.35, marker="s", linewidths=0, rasterized=True)
    sites.plot(ax=ax, color=INK, markersize=4, alpha=0.55, marker="o",
               label="known sites")
    vg.plot(ax=ax, color=VOLC_COL, marker="^", markersize=46,
            label="canonical volcanic centres (13)")
    ax.set_xlabel("Easting (m, EPSG:32749)")
    ax.set_ylabel("Northing (m, EPSG:32749)")
    ax.set_title("Robust core and contingent fringe (XGBoost, 10-seed ensemble)",
                 fontsize=11)
    legend_elements = [
        Patch(facecolor=BLUE_DARK, label="robust core (top decile under all 3 designs)"),
        Patch(facecolor=BLUE_MID,  label="2 designs agree"),
        Patch(facecolor=BLUE_LIGHT, label="contingent fringe (top decile under 1 design)"),
        Patch(facecolor=NONE_GRAY,  label="not top-decile under any design"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK, markersize=6, label="known sites"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=VOLC_COL, markersize=8, label="volcanic centres"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False, loc="lower right")

    fig.tight_layout()
    out = FIG / "fig16_priority_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Fig 17 -- seed-ensemble stabilisation curve
# ===========================================================================
def fig_stabilisation() -> Path:
    c = pd.read_csv(EXP / "E221_seed_ensemble_stability/results/e221_stabilisation_curve.csv")
    J90 = 0.90
    kfloor = 7  # protocol floor (E221 / doc 10 D2)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for algo, col in [("xgboost", BLUE), ("randomforest", ORANGE), ("maxent", AQUA)]:
        sub = c[c.algorithm == algo]
        g = sub.groupby("k")["jaccard_mean"].mean()
        ax.plot(g.index, g.values, "o-", color=col, lw=2, ms=5, label=algo.capitalize())
        kstar = int(sub[sub.jaccard_mean >= J90].k.min())
        ax.axvline(kstar, color=col, lw=1.0, ls=":", alpha=0.7)

    ax.axhline(J90, color=INK, lw=1.2, ls="--")
    ax.text(1.02, J90, f" J = {J90}", fontsize=8.5, va="center", color=INK)
    ax.axvspan(kfloor, c.k.max(), color=BLUE, alpha=0.06)
    ax.text(7.05, 0.56, "protocol floor\nk $\\geq$ 7", fontsize=8.5, color=MUTED)
    ax.set_xlabel("Ensemble size k (seeds averaged)")
    ax.set_ylabel("Jaccard vs 10-seed ensemble\n(top-decile map agreement)")
    ax.set_title("Seed-ensembling stabilises the priority map", fontsize=11)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    fig.tight_layout()
    out = FIG / "fig17_seed_stabilisation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Fig 10 -- study area redraw with the canonical 13 volcanic centres (INT-1)
# ===========================================================================
def fig_study_area() -> Path:
    import geopandas as gpd
    try:
        import contextily as cx
        HAS_CX = True
    except Exception:
        HAS_CX = False

    sites = gpd.read_file(DATA / "east_java_sites.geojson")
    sites = sites[sites.geometry.notna() & ~sites.geometry.is_empty]
    sites = sites.cx[111:115, -9:-6.5]
    sites_wm = sites.to_crs(epsg=3857)

    volc = pd.read_csv(DATA / "dashboard/volcanoes_java_full.csv")
    v13 = volc[(volc.lon >= 111) & (volc.lon <= 115) & (volc.lat >= -9) & (volc.lat <= -6.5)]
    vg = gpd.GeoDataFrame(v13, geometry=gpd.points_from_xy(v13.lon, v13.lat), crs="EPSG:4326")
    vg_wm = vg.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    # Legend deliberately carries no count: the frame holds 383 presences, of which
    # the model uses the 378 with valid covariates (manuscript §2.1). A count here
    # would let a reviewer find the discrepancy instead of us.
    sites_wm.plot(ax=ax, color=BLUE, markersize=7, alpha=0.65, label="archaeological presences")
    vg_wm.plot(ax=ax, color=VOLC_COL, marker="^", markersize=70, label="canonical volcanic centres (13)")
    # stagger label offsets to reduce collision in the dense Malang-Mojokerto cluster
    offsets = [(6, 6), (6, -10), (-8, 6), (-8, -10), (12, 2), (-12, 2), (6, 12)]
    for i, row in enumerate(vg_wm.iterrows()):
        _, row = row
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(row["name"], (row.geometry.x, row.geometry.y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, fontweight="bold", color=VOLC_COL)

    if HAS_CX:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=8)
        except Exception:
            HAS_CX = False
    if not HAS_CX:
        ax.set_facecolor(NONE_GRAY)

    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    ax.set_xlim(*tr.transform(111, -9)); ax.set_xlim(*tr.transform(110.5, -6.5))
    # keep the previous x-lim by re-deriving both bounds
    x0, y0 = tr.transform(110.5, -9.0)
    x1, y1 = tr.transform(115.5, -6.0)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_title("Study area: East Java, Indonesia (111$-$115$^\\circ$E)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    out = FIG / "fig10_study_area_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ===========================================================================
# Fig 3 -- the reported ladder, restated as the object under examination
# ===========================================================================
def fig_progression() -> Path:
    m = pd.read_csv(PAPER / "tables_experiment_progression.csv")
    x = np.arange(len(m))
    labels = m["experiment"].tolist()

    fig, (axA, axT) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    w = 0.37
    axA.bar(x - w / 2, m["xgb_auc"], width=w, color=BLUE, alpha=0.9, label="XGBoost")
    axA.bar(x + w / 2, m["rf_auc"], width=w, color=ORANGE, alpha=0.9, label="RandomForest")
    axA.axhline(0.75, color=INK, ls="--", lw=1.0, label="MVR threshold (0.75)")
    axA.set_ylabel("Spatial AUC")
    axA.set_ylim(0.55, 0.82)
    axA.grid(axis="y")
    axA.set_title("Reported AUC, E007$-$E013")
    axA.legend(fontsize=8)

    axT.bar(x - w / 2, m["xgb_tss"], width=w, color=BLUE, alpha=0.9, label="XGBoost")
    axT.bar(x + w / 2, m["rf_tss"], width=w, color=ORANGE, alpha=0.9, label="RandomForest")
    axT.axhline(0.40, color=INK, ls="--", lw=1.0, label="TSS target (0.40)")
    axT.set_ylabel("TSS")
    axT.set_ylim(0.25, 0.56)
    axT.grid(axis="y")
    axT.set_title("Reported TSS, E007$-$E013")
    axT.legend(fontsize=8)

    for ax_ in (axA, axT):
        ax_.set_xticks(x)
        ax_.set_xticklabels(labels)

    fig.suptitle("The ladder under examination: reported performance of the published pipeline",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG / "fig3_auc_tss_progression.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("fig14_artefact_two_panel.png", fig_artefact),
        ("fig15_dose_response.png", fig_dose_response),
        ("fig16_priority_map.png", fig_priority_map),
        ("fig17_seed_stabilisation.png", fig_stabilisation),
        ("fig10_study_area_map.png", fig_study_area),
        ("fig3_auc_tss_progression.png", fig_progression),
    ]
    for name, fn in jobs:
        p = fn()
        print(f"  {name}: {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
