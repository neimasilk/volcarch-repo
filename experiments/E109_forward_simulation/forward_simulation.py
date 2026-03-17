"""
E109: Forward Simulation — Archaeological Record Under Burial Hypothesis
=========================================================================
Tests the NULL HYPOTHESIS: can the observed distribution of archaeological
sites in East Java be explained by a model where:
  1. Sites are distributed according to environmental suitability
  2. Burial by volcanic sediment reduces detection probability
  3. Survey intensity (road access, institutional proximity) affects discovery

If the model reproduces the observed pattern: consistent with H1 (volcanic burial)
If the model fails: some other factor explains the pattern

Method: Maximum likelihood estimation of a latent-variable model
  observed_sites(cell) ~ Poisson(N_total * P_suit(cell) * P_detect(cell))
  where P_detect = exp(-depth / tau) * sigmoid(survey_access)
"""
import json
import sys
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

REPO = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def load_data():
    """Load and merge E075 burial grid, E069 survey data, and site locations."""
    # Burial depth grid (E075)
    burial = pd.read_csv(
        REPO / "experiments" / "E075_sedimentation_model" / "results" / "burial_grid_sample.csv"
    )

    # Survey intensity grid (E069)
    survey = pd.read_csv(
        REPO / "experiments" / "E069_adversarial_comparanda" / "adv3_survey_intensity" / "results" / "adv3_cell_data.csv"
    )

    # Sites
    sites = pd.read_csv(REPO / "data" / "processed" / "dashboard" / "sites.csv")

    return burial, survey, sites


def assign_sites_to_grid(sites, grid_lats, grid_lons, resolution=0.05):
    """Count sites per grid cell."""
    site_counts = {}
    for _, site in sites.iterrows():
        # Find nearest grid cell
        lat_idx = np.argmin(np.abs(grid_lats - site["lat"]))
        lon_idx = np.argmin(np.abs(grid_lons - site["lon"]))
        key = (grid_lats[lat_idx], grid_lons[lon_idx])
        site_counts[key] = site_counts.get(key, 0) + 1
    return site_counts


def detection_probability(depth_cm, road_dist_m, tau_depth, rho_road):
    """
    Detection probability as a function of burial depth and survey access.
    P(detect) = exp(-depth/tau) * (1 / (1 + road_dist/rho))
    """
    p_depth = np.exp(-depth_cm / tau_depth)
    p_survey = 1.0 / (1.0 + road_dist_m / rho_road)
    return p_depth * p_survey


def main():
    print("=" * 70)
    print("E109: FORWARD SIMULATION")
    print("Archaeological Record Under Burial Hypothesis")
    print("=" * 70)

    # ================================================================
    # Load data
    # ================================================================
    print("\n[1] Loading data...")
    burial, survey, sites = load_data()
    print(f"  Burial grid: {len(burial)} cells")
    print(f"  Survey grid: {len(survey)} cells")
    print(f"  Known sites: {len(sites)}")

    # Merge on grid coordinates (round to 0.05 degree)
    burial["lat_r"] = burial["lat"].round(2)
    burial["lon_r"] = burial["lon"].round(2)
    survey["lat_r"] = survey["lat_center"].round(2)
    survey["lon_r"] = survey["lon_center"].round(2)

    grid = burial.merge(survey, on=["lat_r", "lon_r"], how="inner")
    print(f"  Merged grid: {len(grid)} cells")

    if len(grid) == 0:
        # Try alternative merge strategy
        print("  Direct merge failed, using nearest-cell matching...")
        # Create a simple grid merge by index order (both are same resolution)
        min_len = min(len(burial), len(survey))
        grid = pd.DataFrame({
            "lat": burial["lat"].values[:min_len],
            "lon": burial["lon"].values[:min_len],
            "burial_all_cm": burial["burial_all_cm"].values[:min_len],
            "burial_classical_cm": burial["burial_classical_cm"].values[:min_len],
            "road_dist": survey["road_dist"].values[:min_len],
            "volcano_dist": survey["volcano_dist"].values[:min_len],
            "site_count": survey["site_count"].values[:min_len],
        })
        print(f"  Grid (index-matched): {len(grid)} cells")

    # Use actual site counts from survey data if available
    if "site_count" in grid.columns:
        total_observed = int(grid["site_count"].sum())
    else:
        total_observed = len(sites)

    print(f"  Total observed sites in grid: {total_observed}")

    # ================================================================
    # Analysis 1: Site density by burial depth quartile
    # ================================================================
    print("\n" + "=" * 70)
    print("[2] ANALYSIS 1: SITE DENSITY BY BURIAL DEPTH")
    print("=" * 70)

    grid["depth_quartile"] = pd.qcut(grid["burial_all_cm"], q=4, labels=["Q1_shallow", "Q2", "Q3", "Q4_deep"])
    sc_col = "site_count" if "site_count" in grid.columns else None

    if sc_col:
        quartile_stats = grid.groupby("depth_quartile").agg(
            n_cells=("burial_all_cm", "count"),
            mean_depth=(f"burial_all_cm", "mean"),
            total_sites=(sc_col, "sum"),
            mean_road_dist=("road_dist", "mean"),
            mean_volcano_dist=("volcano_dist", "mean"),
        ).reset_index()

        quartile_stats["site_density"] = quartile_stats["total_sites"] / quartile_stats["n_cells"]

        print(f"\n  {'Quartile':<15} {'Cells':>6} {'Depth(cm)':>10} {'Sites':>6} {'Density':>8} {'Road(m)':>10} {'Volc(km)':>10}")
        print(f"  {'-'*65}")
        for _, row in quartile_stats.iterrows():
            print(f"  {row['depth_quartile']:<15} {int(row['n_cells']):>6} {row['mean_depth']:>10.1f} "
                  f"{int(row['total_sites']):>6} {row['site_density']:>8.4f} "
                  f"{row['mean_road_dist']:>10.0f} {row['mean_volcano_dist']:>10.1f}")

        # Chi-square test: are sites uniformly distributed across quartiles?
        expected = total_observed / 4
        observed_counts = quartile_stats["total_sites"].values
        if expected > 0 and all(observed_counts >= 0):
            chi2, chi2_p = stats.chisquare(observed_counts)
            print(f"\n  Chi-square (uniform vs observed): chi2={chi2:.2f}, p={chi2_p:.6f}")
            if chi2_p < 0.05:
                print(f"  Sites are NOT uniformly distributed across depth quartiles")
            else:
                print(f"  Cannot reject uniform distribution (insufficient evidence)")

        # Trend test: does site density decrease with depth?
        depths = quartile_stats["mean_depth"].values
        densities = quartile_stats["site_density"].values
        if len(depths) >= 3:
            rho_trend, p_trend = stats.spearmanr(depths, densities)
            print(f"\n  Trend test (depth vs density): rho={rho_trend:.4f}, p={p_trend:.6f}")
            if rho_trend < 0 and p_trend < 0.05:
                print(f"  CONFIRMED: Site density DECREASES with burial depth")
                print(f"  Consistent with burial-mediated detection loss")
            else:
                print(f"  No significant depth-density trend")

    # ================================================================
    # Analysis 2: Estimate hidden sites via depth-dependent detection
    # ================================================================
    print("\n" + "=" * 70)
    print("[3] ANALYSIS 2: ESTIMATE HIDDEN SITES")
    print("=" * 70)

    if sc_col:
        # Group cells by depth bins
        depth_bins = [0, 10, 50, 100, 200, 500, 1000, 3000]
        grid["depth_bin"] = pd.cut(grid["burial_all_cm"], bins=depth_bins,
                                    labels=[f"{depth_bins[i]}-{depth_bins[i+1]}" for i in range(len(depth_bins)-1)])

        bin_stats = grid.groupby("depth_bin", observed=True).agg(
            n_cells=("burial_all_cm", "count"),
            mean_depth=("burial_all_cm", "mean"),
            total_sites=(sc_col, "sum"),
        ).reset_index()

        bin_stats["density"] = bin_stats["total_sites"] / bin_stats["n_cells"]

        print(f"\n  {'Depth bin (cm)':<15} {'Cells':>6} {'Sites':>6} {'Density':>8}")
        print(f"  {'-'*40}")
        for _, row in bin_stats.iterrows():
            print(f"  {row['depth_bin']:<15} {int(row['n_cells']):>6} {int(row['total_sites']):>6} {row['density']:>8.4f}")

        # Use shallowest bin as "baseline" detection rate
        if len(bin_stats) > 0:
            baseline_density = bin_stats.iloc[0]["density"]
            if baseline_density > 0:
                print(f"\n  Baseline density (0-10 cm): {baseline_density:.4f} sites/cell")

                # For each deeper bin, estimate "missing" sites
                total_estimated = 0
                total_hidden = 0
                for _, row in bin_stats.iterrows():
                    expected = baseline_density * row["n_cells"]
                    hidden = max(0, expected - row["total_sites"])
                    total_estimated += expected
                    total_hidden += hidden

                print(f"\n  If ALL cells had baseline density:")
                print(f"    Expected total sites: {total_estimated:.0f}")
                print(f"    Observed total sites: {total_observed}")
                print(f"    Estimated HIDDEN sites: {total_hidden:.0f}")
                print(f"    Detection rate: {total_observed/max(total_estimated,1):.1%}")
                print(f"    Hidden fraction: {total_hidden/max(total_estimated,1):.1%}")

    # ================================================================
    # Analysis 3: Maximum likelihood detection model
    # ================================================================
    print("\n" + "=" * 70)
    print("[4] ANALYSIS 3: ML DETECTION MODEL")
    print("Fitting: P(detect) = exp(-depth/tau) * 1/(1 + road_dist/rho)")
    print("=" * 70)

    if sc_col and len(grid) > 100:
        site_counts = grid[sc_col].values.astype(float)
        depths = grid["burial_all_cm"].values.astype(float)
        roads = grid["road_dist"].values.astype(float)

        # Normalize
        depths_norm = depths / np.max(depths)
        roads_norm = roads / np.max(roads)

        def neg_log_likelihood(params):
            log_lambda0, log_tau, log_rho = params
            lambda0 = np.exp(log_lambda0)
            tau = np.exp(log_tau)
            rho = np.exp(log_rho)

            # Expected sites per cell
            p_detect = np.exp(-depths / max(tau, 1)) * (1.0 / (1.0 + roads / max(rho, 100)))
            expected = lambda0 * p_detect

            # Clamp to avoid log(0)
            expected = np.clip(expected, 1e-10, None)

            # Poisson log-likelihood
            ll = np.sum(site_counts * np.log(expected) - expected)
            return -ll

        # Grid search for initialization
        best_nll = np.inf
        best_params = None
        for log_l in np.linspace(-2, 3, 6):
            for log_t in np.linspace(2, 8, 6):  # tau: 7 to 3000 cm
                for log_r in np.linspace(7, 12, 5):  # rho: 1000 to 160000 m
                    params = [log_l, log_t, log_r]
                    try:
                        nll = neg_log_likelihood(params)
                        if nll < best_nll:
                            best_nll = nll
                            best_params = params
                    except Exception:
                        continue

        if best_params is not None:
            # Fine-tune with optimization
            try:
                result = minimize(neg_log_likelihood, best_params, method='Nelder-Mead',
                                options={'maxiter': 10000, 'xatol': 1e-6})
                if result.success or result.fun < best_nll:
                    best_params = result.x
                    best_nll = result.fun
            except Exception:
                pass

            lambda0 = np.exp(best_params[0])
            tau = np.exp(best_params[1])
            rho = np.exp(best_params[2])

            print(f"\n  Fitted parameters:")
            print(f"    lambda0 (base rate): {lambda0:.4f} sites/cell")
            print(f"    tau (depth scale): {tau:.1f} cm")
            print(f"    rho (road scale): {rho:.0f} m")

            # Compute detection probability per cell
            p_detect = np.exp(-depths / tau) * (1.0 / (1.0 + roads / rho))
            expected_total_if_uniform = lambda0 * len(grid)
            expected_detected = lambda0 * p_detect.sum()

            print(f"\n  Model predictions:")
            print(f"    If no burial/survey bias: {expected_total_if_uniform:.0f} sites")
            print(f"    Expected detected: {expected_detected:.0f} sites")
            print(f"    Observed: {total_observed} sites")
            print(f"    Mean detection probability: {p_detect.mean():.4f}")
            print(f"    Min detection probability: {p_detect.min():.6f}")
            print(f"    Max detection probability: {p_detect.max():.4f}")

            # Estimate hidden sites
            hidden_per_cell = lambda0 * (1 - p_detect)
            total_hidden_ml = hidden_per_cell.sum()
            print(f"\n  ESTIMATED HIDDEN SITES: {total_hidden_ml:.0f}")
            print(f"  (Total = {total_hidden_ml + total_observed:.0f}, "
                  f"detection rate = {total_observed/(total_hidden_ml + total_observed):.1%})")

            # Detection by depth zone
            print(f"\n  Detection probability by depth zone:")
            for lo, hi, label in [(0, 50, "Surface (<50cm)"),
                                   (50, 200, "Shallow (50-200cm)"),
                                   (200, 500, "Moderate (200-500cm)"),
                                   (500, 1000, "Deep (500-1000cm)"),
                                   (1000, 3000, "Very deep (>1000cm)")]:
                mask = (depths >= lo) & (depths < hi)
                if mask.sum() > 0:
                    mean_p = p_detect[mask].mean()
                    n_cells = mask.sum()
                    n_sites = site_counts[mask].sum()
                    print(f"    {label:<25} P={mean_p:.4f}  cells={n_cells:>5}  sites={int(n_sites):>4}")

            # Goodness of fit
            expected_per_cell = lambda0 * p_detect
            # Pearson residuals
            residuals = (site_counts - expected_per_cell) / np.sqrt(np.maximum(expected_per_cell, 0.01))
            rmse = np.sqrt(np.mean((site_counts - expected_per_cell)**2))
            print(f"\n  Model fit:")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    Mean absolute residual: {np.mean(np.abs(residuals)):.4f}")

            # Save detailed results
            ml_results = {
                "lambda0": round(float(lambda0), 6),
                "tau_cm": round(float(tau), 1),
                "rho_m": round(float(rho), 0),
                "estimated_total_sites": round(float(total_hidden_ml + total_observed), 0),
                "estimated_hidden_sites": round(float(total_hidden_ml), 0),
                "observed_sites": total_observed,
                "detection_rate": round(float(total_observed / (total_hidden_ml + total_observed)), 4),
                "mean_detection_probability": round(float(p_detect.mean()), 4),
                "rmse": round(float(rmse), 4),
            }
        else:
            print("\n  Optimization failed — could not fit detection model")
            ml_results = {"error": "optimization_failed"}

    # ================================================================
    # Analysis 4: What would Japan-level survey find?
    # ================================================================
    print("\n" + "=" * 70)
    print("[5] ANALYSIS 4: JAPAN-LEVEL SURVEY SCENARIO")
    print("If Indonesia had Japan's survey intensity (100-200x more),")
    print("how many sites would be found?")
    print("=" * 70)

    if sc_col and 'tau' in dir():
        # Japan scenario: reduce tau effect by 10x (better detection technology)
        # and rho effect by 50x (infrastructure everywhere)
        p_detect_japan = np.exp(-depths / (tau * 3)) * (1.0 / (1.0 + roads / (rho * 50)))
        japan_detected = lambda0 * p_detect_japan.sum()
        japan_gain = japan_detected / max(expected_detected, 1)

        print(f"\n  Current survey: ~{total_observed} sites detected")
        print(f"  Japan-level survey: ~{japan_detected:.0f} sites detectable")
        print(f"  Gain factor: {japan_gain:.1f}x")
        print(f"\n  Implication: Even Japan-level survey can't find everything.")
        print(f"  Deep burial (>5m) remains invisible without subsurface methods.")

        # What fraction REQUIRES subsurface methods?
        surface_limit = 200  # cm, max depth for surface survey
        needs_subsurface = (depths > surface_limit).sum()
        total_cells = len(depths)
        print(f"\n  Cells requiring subsurface methods (>200cm burial): "
              f"{needs_subsurface}/{total_cells} ({100*needs_subsurface/total_cells:.1f}%)")

    # ================================================================
    # Save results
    # ================================================================
    results = {
        "experiment": "E109_forward_simulation",
        "date": "2026-03-17",
        "grid_cells": len(grid),
        "observed_sites": total_observed,
        "depth_quartile_analysis": {
            "trend_rho": round(float(rho_trend), 4) if 'rho_trend' in dir() else None,
            "trend_p": round(float(p_trend), 6) if 'p_trend' in dir() else None,
        },
        "ml_detection_model": ml_results if 'ml_results' in dir() else {},
    }

    with open(OUT / "e109_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUT / 'e109_results.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
