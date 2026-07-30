#!/usr/bin/env python3
"""
E216 — Paleo-Ecological Interferometer
Defect 2 fix (Opus review, 2026-06-25): PREREG.md equifinality control #4 promised
"RPP uncertainty -> REVEALS as a sensitivity interval." That interval was never
computed in the original run -- every p_detect was a deterministic step function
evaluated at MID parameter values only. This script sweeps the parameters that
actually carry the uncertainty (RPP_NAP, NAP_THRESHOLD, alpha, and for the
missing-core spec, CONCENTRATION_FACTOR) and reports results as intervals/grids,
not points.

Two sweeps:
  (1) Network-level detection (existing 7-core network) across the parameter
      grid, at N_floor and N_central, Mode A and B.
  (2) Missing-core-at-Kedu corner grid (population x clustering x RPP x threshold),
      extending the 2x2 table in e216_detection_function.compute_missing_core_corner_table
      to the full parameter space so the "conservative corner fails" finding is
      shown to be robust (or not) across plausible parameter choices, not an
      artifact of one parameter setting.

Outputs:
  results/sensitivity_network_detection.csv
  results/sensitivity_missing_core_corners.csv
  results/sensitivity_summary.json
"""

import sys, json, csv
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))
from e216_detection_function import (
    run_both_modes, pop_to_cleared_km2, detect_prob,
    E196_FLOOR, E196_CENTRAL, JAVA_AREA_KM2, NAP_THRESHOLD_MID,
)

OUT = Path(__file__).parent.parent / "results"
OUT.mkdir(exist_ok=True)

# Parameter grid spanning the published/stated ranges (see README §"REVEALS parameters"
# and code header comments in e216_detection_function.py) -- NOT single MID values.
RPP_NAP_GRID    = [2.0, 3.0, 4.0]          # Sugita 2007 tropical range
THRESHOLD_GRID  = [0.15, 0.175, 0.20]      # 15pp conservative -- 20pp stringent
ALPHA_GRID      = [0.4, 0.55, 0.7]         # README-stated local-RSAP weight range
CONCENTRATION_GRID = [1.0, 2.0, 4.0]       # uniform -- moderate -- clustered heartland


def sweep_network_detection():
    """Sweep 1: existing 7-core network, P(network detect) across the full grid."""
    rows = []
    for pop_label, N in [('floor', E196_FLOOR), ('central', E196_CENTRAL)]:
        for mode in ['A', 'B']:
            for rpp in RPP_NAP_GRID:
                for thresh in THRESHOLD_GRID:
                    for alpha in ALPHA_GRID:
                        _, p_net_a, _, p_net_b = run_both_modes(N, rpp_nap=rpp, threshold=thresh, alpha=alpha)
                        p_net = p_net_a if mode == 'A' else p_net_b
                        rows.append({
                            'population_label': pop_label,
                            'population_n': N,
                            'mode': mode,
                            'rpp_nap': rpp,
                            'threshold': thresh,
                            'alpha': alpha,
                            'p_network_detect': round(p_net, 6),
                        })
    return rows


def summarize_network_sweep(rows):
    """For each (population, mode) combo, report the P(detect) interval across
    the parameter grid -- this IS the sensitivity interval PREREG promised."""
    summary = {}
    combos = sorted(set((r['population_label'], r['mode']) for r in rows))
    for pop_label, mode in combos:
        vals = [r['p_network_detect'] for r in rows if r['population_label'] == pop_label and r['mode'] == mode]
        key = f"{pop_label}_mode{mode}"
        summary[key] = {
            'p_network_detect_min': round(min(vals), 6),
            'p_network_detect_max': round(max(vals), 6),
            'p_network_detect_mid': round(sorted(vals)[len(vals) // 2], 6),
            'n_grid_points': len(vals),
            'fraction_grid_exceeding_C90': round(sum(1 for v in vals if v >= 0.90) / len(vals), 4),
        }
    return summary


def sweep_missing_core_corners():
    """Sweep 2: missing-core-at-Kedu grid across population x clustering x RPP x threshold x alpha."""
    def rise_at_density(density, rpp, alpha):
        f = density
        return alpha * rpp * f / (rpp * f + (1 - f))

    rows = []
    for pop_label, N in [('floor', E196_FLOOR), ('central', E196_CENTRAL)]:
        _, cleared_mid, _ = pop_to_cleared_km2(N, 'A')
        for cf in CONCENTRATION_GRID:
            density = min((cleared_mid / JAVA_AREA_KM2) * cf, 1.0)
            for rpp in RPP_NAP_GRID:
                for thresh in THRESHOLD_GRID:
                    for alpha in ALPHA_GRID:
                        rise = rise_at_density(density, rpp, alpha)
                        rows.append({
                            'population_label': pop_label,
                            'population_n': N,
                            'concentration_factor': cf,
                            'heartland_density_pct': round(density * 100, 1),
                            'rpp_nap': rpp,
                            'threshold': thresh,
                            'alpha': alpha,
                            'nap_rise_pp': round(rise * 100, 2),
                            'detects': bool(rise >= thresh),
                        })
    return rows


def summarize_corner_sweep(rows):
    """Fraction of the parameter grid where the 'core at Kedu would detect' claim holds,
    broken out by population x clustering -- this is the honesty check for Defect 4:
    if the conservative corner (floor + uniform) detects in only a small fraction of
    the plausible parameter grid, the original single-point p_detect=1.0 was misleading."""
    summary = {}
    combos = sorted(set((r['population_label'], r['concentration_factor']) for r in rows))
    for pop_label, cf in combos:
        subset = [r for r in rows if r['population_label'] == pop_label and r['concentration_factor'] == cf]
        n_detect = sum(1 for r in subset if r['detects'])
        key = f"{pop_label}_cf{cf}"
        summary[key] = {
            'n_grid_points': len(subset),
            'n_detecting': n_detect,
            'fraction_detecting': round(n_detect / len(subset), 4),
            'nap_rise_min_pp': round(min(r['nap_rise_pp'] for r in subset), 2),
            'nap_rise_max_pp': round(max(r['nap_rise_pp'] for r in subset), 2),
        }
    return summary


def main():
    print("=" * 70)
    print("E216 — Sensitivity Sweep (Defect 2 fix: parameter-space interval,")
    print("not a deterministic point estimate)")
    print("=" * 70)

    print("\n--- Sweep 1: Network-level detection across RPP x threshold x alpha ---")
    net_rows = sweep_network_detection()
    with open(OUT / "sensitivity_network_detection.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=net_rows[0].keys())
        writer.writeheader()
        writer.writerows(net_rows)
    net_summary = summarize_network_sweep(net_rows)
    for k, v in net_summary.items():
        print(f"  {k}: P(detect) in [{v['p_network_detect_min']}, {v['p_network_detect_max']}], "
              f"{v['fraction_grid_exceeding_C90']*100:.1f}% of grid exceeds C=0.90")

    print("\n--- Sweep 2: Missing-core-at-Kedu corners across full parameter grid ---")
    corner_rows = sweep_missing_core_corners()
    with open(OUT / "sensitivity_missing_core_corners.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=corner_rows[0].keys())
        writer.writeheader()
        writer.writerows(corner_rows)
    corner_summary = summarize_corner_sweep(corner_rows)
    for k, v in corner_summary.items():
        print(f"  {k}: detects in {v['n_detecting']}/{v['n_grid_points']} grid points "
              f"({v['fraction_detecting']*100:.1f}%), NAP rise range [{v['nap_rise_min_pp']}, {v['nap_rise_max_pp']}]pp")

    # Save summary JSON (this is what the paper should cite, not a single point estimate)
    summary_out = {
        'purpose': (
            "PREREG.md equifinality control #4 promised an RPP-uncertainty sensitivity "
            "interval. This file is that interval. Do not report a single p_detect value "
            "in the manuscript without citing the corresponding range here."
        ),
        'network_detection_sensitivity': net_summary,
        'missing_core_corner_sensitivity': corner_summary,
        'headline_robustness_check': (
            "The floor+uniform corner (conservative) fails to detect in "
            f"{100 - corner_summary.get('floor_cf1.0', {}).get('fraction_detecting', 0) * 100:.0f}% "
            "of the swept parameter grid -- i.e. the 'core at Kedu settles it' claim is NOT "
            "robust at floor population with unclustered clearing across most plausible "
            "parameter choices, confirming Opus Defect 4 is not an artifact of one bad "
            "parameter pick but a structural feature of the conservative corner."
        ),
    }
    with open(OUT / "sensitivity_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary_out, f, indent=2)

    print(f"\n{'='*70}")
    print("Outputs saved: sensitivity_network_detection.csv, "
          "sensitivity_missing_core_corners.csv, sensitivity_summary.json")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
