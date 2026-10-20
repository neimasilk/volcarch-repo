"""E235 — Elevation or volcanic setting? Band-by-band control for P2 (JCAA round 2, Reviewer 2).

Pre-registered in DESIGN.md (frozen with commit 3ac41cc, pushed 2026-10-20 before this script existed).
Reads the stored E221 ensemble maps; nothing is retrained.

    python experiments/E235_elevation_vs_volcanic_control/01_elevation_vs_volcanic.py
"""
import importlib.util
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")   # Windows cp1252 console
warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
EXP = HERE.parent
RESULTS_DIR = HERE / "results"

spec = importlib.util.spec_from_file_location("e217base", EXP / "E217_maxent_benchmark" / "01_maxent_benchmark.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
spec219 = importlib.util.spec_from_file_location("e219main", EXP / "E219_map_divergence" / "01_map_divergence.py")
e219 = importlib.util.module_from_spec(spec219)
spec219.loader.exec_module(e219)

# ---- fixed by DESIGN.md §3–§5 ---------------------------------------------------------------
BAND_M = 200.0
MIN_AREA_KM2 = 25.0
MIN_BLOCKS = 2
DEFINITIONS = {"primary_20_40": (20.0, 40.0), "sens_10_30": (10.0, 30.0), "sens_15_30": (15.0, 30.0)}
PRIMARY = "primary_20_40"
ALGOS = ["xgboost", "randomforest", "maxent"]
PRIMARY_ALGO = "xgboost"
DESIGNS = ["random", "tgb", "hybrid"]
DELTA = 0.05            # suitability units (M2, M3b)
DELTA_PP = 5.0          # percentage points (M3a)
SENS_DELTAS = [0.03, 0.10]
N_BOOT = 1000
SEED = 235
CELL_KM2 = (base.DECIMATE * 30.663793749005784 / 1000.0) ** 2   # same constant as E219

OUTCOMES = ["suit_random", "suit_tgb", "suit_hybrid", "fringe", "absdiff_hybrid_random",
            "aspect_cos", "aspect_sin", "river_dist_km"]


def verdict(est, lo, hi, d):
    """DESIGN §5: SPECIFIC / NOT SPECIFIC / INCONCLUSIVE."""
    if (lo > 0 or hi < 0) and abs(est) >= d:
        return "SPECIFIC"
    if lo > -d and hi < d:
        return "NOT SPECIFIC"
    return "INCONCLUSIVE"


def rcs_basis(x, n_knots=5):
    """Restricted (natural) cubic spline basis, Harrell knots at 5/27.5/50/72.5/95 percentiles."""
    k = np.quantile(x, [0.05, 0.275, 0.5, 0.725, 0.95][:n_knots])
    cols = [x]
    norm = (k[-1] - k[0]) ** 2
    pos3 = lambda u: np.clip(u, 0, None) ** 3
    for j in range(n_knots - 2):
        cols.append((pos3(x - k[j]) - pos3(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                     + pos3(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / norm)
    return np.column_stack(cols)


def partial_r2(y, X_red, x_add):
    ones = np.ones((len(y), 1))
    Xr = np.hstack([ones, X_red])
    Xf = np.hstack([Xr, x_add.reshape(-1, 1)])
    sse = lambda X: float(np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2))
    s_r, s_f = sse(Xr), sse(Xf)
    return (s_r - s_f) / s_r if s_r > 0 else float("nan")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    frame = base.build_frame()
    maps = {a: np.load(EXP / "E221_seed_ensemble_stability" / "results" / f"e221_priority_sets_{a}.npz")
            for a in ALGOS}
    for a, m in maps.items():
        # The join must be one-to-one over all cells (DESIGN §3); the E221 maps were saved from this same frame.
        assert len(m["x"]) == len(frame) == 588_535, (a, len(m["x"]), len(frame))
        assert np.allclose(m["x"], frame["x"].to_numpy()) and np.allclose(m["y"], frame["y"].to_numpy()), a
    print(f"Frame: {len(frame):,} cells; join to E221 maps exact for {ALGOS}")

    canon = pd.read_csv(e219.CANON_VOLCANOES)          # all 30 centres count for distance (DESIGN §3)
    d_km = e219.volcano_distance_km(frame["x"].to_numpy(), frame["y"].to_numpy(),
                                    list(zip(canon["lat"], canon["lon"])))
    elev = frame["elevation"].to_numpy(dtype=np.float64)
    band = np.floor(elev / BAND_M).astype(int)
    blocks = base.assign_blocks(frame["x"].to_numpy(), frame["y"].to_numpy())
    slope, tri, twi = (frame[c].to_numpy(dtype=np.float64) for c in ("slope", "tri", "twi"))
    asp = np.deg2rad(frame["aspect"].to_numpy(dtype=np.float64))   # aspect is circular: compare cos/sin
    river_km = frame["river_dist"].to_numpy(dtype=np.float64) / 1000.0
    print(f"Elevation {elev.min():.0f}–{elev.max():.0f} m; {len(canon)} centres; "
          f"{len(np.unique(blocks))} spatial blocks")

    band_rows, outcome = [], {"design_commit": "3ac41cc", "definitions": {}}
    for dname, (v_km, n_km) in DEFINITIONS.items():
        V, N = d_km <= v_km, d_km >= n_km
        usable = []
        for b in range(band.max() + 1):
            inb = band == b
            aV, aN = (inb & V).sum() * CELL_KM2, (inb & N).sum() * CELL_KM2
            bV, bN = len(np.unique(blocks[inb & V])), len(np.unique(blocks[inb & N]))
            ok = aV >= MIN_AREA_KM2 and aN >= MIN_AREA_KM2 and bV >= MIN_BLOCKS and bN >= MIN_BLOCKS
            if ok:
                usable.append(b)
            band_rows.append({"definition": dname, "band_lo_m": b * BAND_M, "band_hi_m": (b + 1) * BAND_M,
                              "area_km2_V": aV, "area_km2_N": aN, "area_km2_all": inb.sum() * CELL_KM2,
                              "blocks_V": bV, "blocks_N": bN, "usable": ok})
        e_star = (max(usable) + 1) * BAND_M if usable else float("nan")
        above = elev >= e_star if usable else np.zeros_like(V)
        m1 = {"usable_bands_lo_m": [b * BAND_M for b in usable], "E_star_m": e_star,
              "land_above_E_star_km2": float(above.sum() * CELL_KM2),
              "share_above_E_star_V": float(V[above].mean()) if above.any() else float("nan"),
              "share_above_E_star_N": float(N[above].mean()) if above.any() else float("nan")}
        print(f"\n[{dname}] V ≤ {v_km:.0f} km, N ≥ {n_km:.0f} km — usable bands (lower edges): "
              f"{m1['usable_bands_lo_m']}; E* = {e_star:.0f} m; land above E*: {m1['land_above_E_star_km2']:,.0f} km² "
              f"(V {m1['share_above_E_star_V']:.1%}, N {m1['share_above_E_star_N']:.1%})")
        outcome["definitions"][dname] = {"M1": m1, "algorithms": {}}

        # Within-band CEM on slope × TRI × TWI terciles (terciles over the band's V ∪ N cells).
        stratum = np.full(len(frame), -1, dtype=np.int64)
        for b in usable:
            m = (band == b) & (V | N)
            key = np.zeros(m.sum(), dtype=np.int64)
            for c in (slope, tri, twi):
                key = key * 3 + np.digitize(c[m], np.quantile(c[m], [1 / 3, 2 / 3]))
            stratum[m] = b * 27 + key
        strata = np.unique(stratum[stratum >= 0])
        s_col = {s: i for i, s in enumerate(strata)}
        s_band = np.array([s // 27 for s in strata])

        for algo in ALGOS:
            mp = maps[algo]
            ys = {"suit_random": mp["suit_random"].astype(np.float64),
                  "suit_tgb": mp["suit_tgb"].astype(np.float64),
                  "suit_hybrid": mp["suit_hybrid"].astype(np.float64),
                  "fringe": (mp["n_designs_top10"] == 1).astype(np.float64) * 100.0,     # percent
                  "absdiff_hybrid_random": np.abs(mp["suit_hybrid"].astype(np.float64)
                                                  - mp["suit_random"].astype(np.float64)),
                  "aspect_cos": np.cos(asp), "aspect_sin": np.sin(asp), "river_dist_km": river_km}

            # Per arm: block × stratum matrices of counts and outcome sums (DESIGN §4 bootstrap).
            arm = {}
            for name, A in (("V", V), ("N", N)):
                idx = np.where((stratum >= 0) & A)[0]
                ub, binv = np.unique(blocks[idx], return_inverse=True)
                cols = np.array([s_col[s] for s in stratum[idx]])
                C = np.zeros((len(ub), len(strata)))
                np.add.at(C, (binv, cols), 1.0)
                S = {}
                for k in OUTCOMES:
                    S[k] = np.zeros((len(ub), len(strata)))
                    np.add.at(S[k], (binv, cols), ys[k][idx])
                arm[name] = (C, S, len(ub))

            def stats(mV, mN):
                cV, cN = mV @ arm["V"][0], mN @ arm["N"][0]
                w = np.minimum(cV, cN)
                res = {}
                for k in OUTCOMES:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        diff = (mV @ arm["V"][1][k]) / cV - (mN @ arm["N"][1][k]) / cN
                    diff = np.where(w > 0, diff, 0.0)
                    per_band = {}
                    for b in usable:
                        sb = s_band == b
                        per_band[b] = float((w[sb] * diff[sb]).sum() / w[sb].sum()) if w[sb].sum() > 0 else np.nan
                    res[k] = (float((w * diff).sum() / w.sum()), per_band)
                return res, w

            ones_V, ones_N = np.ones(arm["V"][2]), np.ones(arm["N"][2])
            point, w0 = stats(ones_V, ones_N)
            rng = np.random.default_rng(SEED)
            boot = {k: [] for k in OUTCOMES}
            boot_band = {k: {b: [] for b in usable} for k in OUTCOMES}
            for _ in range(N_BOOT):
                mV = np.bincount(rng.integers(0, arm["V"][2], arm["V"][2]), minlength=arm["V"][2]).astype(float)
                mN = np.bincount(rng.integers(0, arm["N"][2], arm["N"][2]), minlength=arm["N"][2]).astype(float)
                r, _ = stats(mV, mN)
                for k in OUTCOMES:
                    boot[k].append(r[k][0])
                    for b in usable:
                        boot_band[k][b].append(r[k][1][b])

            res_algo = {"matched_weight_cells": float(w0.sum()), "measures": {}}
            for k in OUTCOMES:
                lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
                est = point[k][0]
                d = DELTA_PP if k == "fringe" else DELTA
                entry = {"estimate": est, "ci95": [float(lo), float(hi)]}
                if k.startswith("suit_") or k in ("fringe", "absdiff_hybrid_random"):
                    entry["verdict"] = verdict(est, lo, hi, d)
                    entry["verdict_sensitivity"] = {
                        str(sd if k != "fringe" else sd * 100): verdict(est, lo, hi, sd if k != "fringe" else sd * 100)
                        for sd in SENS_DELTAS}
                res_algo["measures"][k] = entry
                for b in usable:
                    blo, bhi = np.nanpercentile(boot_band[k][b], [2.5, 97.5])
                    band_rows.append({"definition": dname, "algorithm": algo, "measure": k,
                                      "band_lo_m": b * BAND_M, "band_hi_m": (b + 1) * BAND_M,
                                      "delta_V_minus_N": point[k][1][b], "ci_lo": blo, "ci_hi": bhi,
                                      "matched_weight_cells": float(w0[s_band == b].sum())})

            # M2 summary: the verdict shared by ≥ 2 of 3 designs, else "mixed" (DESIGN §5).
            v_designs = [res_algo["measures"][f"suit_{d}"]["verdict"] for d in DESIGNS]
            top = max(set(v_designs), key=v_designs.count)
            res_algo["M2_summary"] = top if v_designs.count(top) >= 2 else "mixed"
            if len(usable) < 2:
                res_algo["M2_summary"] = "NOT JUDGED (< 2 usable bands)"

            # M4 (descriptive): partial R² of the V/N indicator after a spline in elevation.
            sel = np.isin(band, usable) & (V | N)
            Xr = rcs_basis(elev[sel])
            res_algo["M4_partial_r2_setting_after_elevation"] = {
                k: partial_r2(ys[k][sel], Xr, V[sel].astype(float))
                for k in ("suit_random", "suit_tgb", "suit_hybrid", "absdiff_hybrid_random")}
            res_algo["M4_r2_elevation_only"] = {}
            for k in ("suit_random", "suit_tgb", "suit_hybrid", "absdiff_hybrid_random"):
                y = ys[k][sel]
                X = np.hstack([np.ones((sel.sum(), 1)), Xr])
                resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
                res_algo["M4_r2_elevation_only"][k] = float(1 - resid.var() / y.var())
            outcome["definitions"][dname]["algorithms"][algo] = res_algo

            print(f"  {algo:13s} matched cells {w0.sum():>9,.0f} | M2 " + " · ".join(
                f"{d} {res_algo['measures'][f'suit_{d}']['estimate']:+.3f} "
                f"[{res_algo['measures'][f'suit_{d}']['ci95'][0]:+.3f}, {res_algo['measures'][f'suit_{d}']['ci95'][1]:+.3f}] "
                f"{res_algo['measures'][f'suit_{d}']['verdict']}" for d in DESIGNS)
                + f" → {res_algo['M2_summary']}")
            f, a_ = res_algo["measures"]["fringe"], res_algo["measures"]["absdiff_hybrid_random"]
            print(f"  {'':13s} M3a fringe {f['estimate']:+.2f} pp [{f['ci95'][0]:+.2f}, {f['ci95'][1]:+.2f}] {f['verdict']}"
                  f" | M3b |h−r| {a_['estimate']:+.4f} [{a_['ci95'][0]:+.4f}, {a_['ci95'][1]:+.4f}] {a_['verdict']}")

    pd.DataFrame(band_rows).to_csv(RESULTS_DIR / "e235_bands.csv", index=False)
    with open(RESULTS_DIR / "e235_outcome.json", "w", encoding="utf-8") as fh:
        json.dump(outcome, fh, indent=1, ensure_ascii=False, default=float)
    make_figure(frame, d_km, maps[PRIMARY_ALGO], outcome)
    print(f"\nWrote {RESULTS_DIR / 'e235_bands.csv'}, e235_outcome.json, e235_response_by_band.png")


def make_figure(frame, d_km, mp, outcome):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v_km, n_km = DEFINITIONS[PRIMARY]
    V, N = d_km <= v_km, d_km >= n_km
    elev = frame["elevation"].to_numpy()
    band = np.floor(elev / BAND_M).astype(int)
    usable = [int(b / BAND_M) for b in outcome["definitions"][PRIMARY]["M1"]["usable_bands_lo_m"]]
    bands = np.arange(band.max() + 1)
    mid = (bands + 0.5) * BAND_M / 1000.0
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True, constrained_layout=True)
    for b in usable:
        for a in ax:
            a.axvspan(b * BAND_M / 1000, (b + 1) * BAND_M / 1000, color="0.92", zorder=0)
    aV = np.array([(band[V] == b).sum() for b in bands]) * CELL_KM2
    aN = np.array([(band[N] == b).sum() for b in bands]) * CELL_KM2
    w = BAND_M / 1000 * 0.4
    # Empty bands get no bar (a floor value on a log axis would read as land that is not there).
    ax[0].bar((mid - w / 2)[aV > 0], aV[aV > 0], w, color="#b2182b", label=f"volcanic setting (≤ {v_km:.0f} km)")
    ax[0].bar((mid + w / 2)[aN > 0], aN[aN > 0], w, color="#2166ac", label=f"non-volcanic (≥ {n_km:.0f} km)")
    ax[0].axhline(MIN_AREA_KM2, color="k", lw=0.8, ls=":")
    ax[0].text(3.75, MIN_AREA_KM2 * 1.15, "25 km² (usable-band minimum)", ha="right", fontsize=7)
    ax[0].set_yscale("log"); ax[0].set_ylabel("land area (km²)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("A  Land in each 200 m elevation band (shaded: both settings present)", loc="left", fontsize=9)
    # Means are drawn only where the arm has the pre-registered minimum area in that band.
    enough = {"V": aV >= MIN_AREA_KM2, "N": aN >= MIN_AREA_KM2}
    styles = {"random": "-", "tgb": "--", "hybrid": ":"}
    for dsg, ls in styles.items():
        s = mp[f"suit_{dsg}"]
        for A, key, col in ((V, "V", "#b2182b"), (N, "N", "#2166ac")):
            mvals = [s[A & (band == b)].mean() if enough[key][b] else np.nan for b in bands]
            ax[1].plot(mid, mvals, ls=ls, color=col, lw=1.4, label=f"{dsg}" if col == "#b2182b" else None)
    ax[1].set_ylabel("mean predicted suitability")
    ax[1].legend(title="design (red = volcanic, blue = non-volcanic)", frameon=False, fontsize=8, title_fontsize=8)
    ax[1].set_title("B  Model response by band (XGBoost, 10-seed ensemble; unmatched band means)", loc="left", fontsize=9)
    fr = (mp["n_designs_top10"] == 1)
    for A, key, col in ((V, "V", "#b2182b"), (N, "N", "#2166ac")):
        vals = [fr[A & (band == b)].mean() * 100 if enough[key][b] else np.nan for b in bands]
        ax[2].plot(mid, vals, color=col, lw=1.4, marker="o", ms=3)
    ax[2].set_ylabel("cells in contingent fringe (%)")
    ax[2].set_xlabel("elevation (km)")
    ax[2].set_title("C  Disagreement between background designs by band", loc="left", fontsize=9)
    fig.savefig(RESULTS_DIR / "e235_response_by_band.png", dpi=200)


if __name__ == "__main__":
    main()
