"""
E118: Information Gain from Volcanic Context

The strongest counter-argument to VOLCARCH (Counter 4 in pre-mortem):
"Survey deficit has 40× leverage; volcanic burial only 1.7×.
Indonesia just needs more archaeology, period. Volcanism is a distraction."

This experiment quantifies WHY volcanic context matters EVEN THOUGH its
absolute leverage is smaller. The answer: SPATIAL PREDICTABILITY.

Survey deficit is uniform — knowing that Indonesia under-surveys tells you
NOTHING about WHERE to dig. Volcanic burial is geographically structured —
it tells you EXACTLY where to dig and how deep.

We formalize this as an information theory problem:
  - H(finds | no context): entropy of finding sites with no geological knowledge
  - H(finds | volcanic context): entropy of finding sites WITH volcanic knowledge
  - Information gain: H_no - H_with

If volcanic context reduces uncertainty about where to find sites,
VOLCARCH has practical value regardless of absolute leverage.
"""

import json
import sys
import io
from pathlib import Path
import numpy as np
from scipy.stats import entropy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

np.random.seed(2026)


def main():
    print("=" * 70)
    print("E118: INFORMATION GAIN FROM VOLCANIC CONTEXT")
    print("Why spatial predictability matters more than absolute leverage")
    print("=" * 70)

    # ================================================================
    # SETUP: Define the search space
    # ================================================================
    print("\n[1] SEARCH SPACE DEFINITION")
    print("=" * 70)

    # East Java habitable area ≈ 20,000 km²
    # Discretize into 1 km² cells
    n_cells = 20000
    # From E108 moderate estimate: ~3,000 pre-400 CE settlements in E. Java
    n_sites = 3000

    # Without volcanic context: sites could be ANYWHERE
    # P(site at random cell) = 3000/20000 = 0.15
    p_random = n_sites / n_cells

    print(f"\n  Search area: {n_cells:,} km² (East Java)")
    print(f"  Expected sites: {n_sites:,} (E108 moderate)")
    print(f"  P(site at random cell): {p_random:.3f}")

    # ================================================================
    # ANALYSIS 1: Uniform vs. Volcanic-Informed Search
    # ================================================================
    print("\n" + "=" * 70)
    print("[2] UNIFORM vs. VOLCANIC-INFORMED SEARCH")
    print("=" * 70)

    # With volcanic context (from E080 suitability model):
    # High-suitability zone (top 20%): ~4,000 km², contains ~70% of predicted sites
    # Medium-suitability zone (40%): ~8,000 km², contains ~25% of predicted sites
    # Low-suitability zone (40%): ~8,000 km², contains ~5% of predicted sites

    zones = {
        "high": {"area_km2": 4000, "site_fraction": 0.70, "label": "High suitability (volcanic sweet spot)"},
        "medium": {"area_km2": 8000, "site_fraction": 0.25, "label": "Medium suitability"},
        "low": {"area_km2": 8000, "site_fraction": 0.05, "label": "Low suitability"},
    }

    print(f"\n  {'Zone':<40} {'Area (km²)':<12} {'Sites':<8} {'P(site)':<10} {'Density'}")
    print("  " + "-" * 85)

    for name, z in zones.items():
        sites_in_zone = int(n_sites * z["site_fraction"])
        p_site = sites_in_zone / z["area_km2"]
        density_ratio = p_site / p_random
        z["sites"] = sites_in_zone
        z["p_site"] = p_site
        z["density_ratio"] = density_ratio
        print(f"  {z['label']:<40} {z['area_km2']:<12,} {sites_in_zone:<8} {p_site:<10.4f} {density_ratio:.2f}×")

    # ================================================================
    # ANALYSIS 2: Information Gain (Shannon Entropy)
    # ================================================================
    print("\n" + "=" * 70)
    print("[3] INFORMATION GAIN (SHANNON ENTROPY)")
    print("=" * 70)

    # H(find | no context) = -p*log2(p) - (1-p)*log2(1-p)
    h_random = entropy([p_random, 1 - p_random], base=2)

    # H(find | volcanic context) = weighted sum of zone entropies
    total_area = sum(z["area_km2"] for z in zones.values())
    h_volcanic = 0
    for name, z in zones.items():
        weight = z["area_km2"] / total_area
        p = z["p_site"]
        h_zone = entropy([p, 1 - p], base=2)
        h_volcanic += weight * h_zone
        print(f"  Zone {name}: H = {h_zone:.4f} bits (weight = {weight:.2f})")

    info_gain = h_random - h_volcanic
    info_gain_pct = (info_gain / h_random) * 100

    print(f"\n  H(find | no context): {h_random:.4f} bits")
    print(f"  H(find | volcanic context): {h_volcanic:.4f} bits")
    print(f"  Information gain: {info_gain:.4f} bits ({info_gain_pct:.1f}% reduction)")

    # ================================================================
    # ANALYSIS 3: Search Efficiency
    # ================================================================
    print("\n" + "=" * 70)
    print("[4] SEARCH EFFICIENCY — Practical Impact")
    print("=" * 70)

    # If you survey K cells, how many sites do you expect to find?
    # Strategy A: Random survey (uniform across all cells)
    # Strategy B: Volcanic-informed (prioritize high-suitability zone)
    # Strategy C: Perfect oracle (knows exactly where sites are)

    budgets = [20, 50, 100, 200, 500, 1000]

    print(f"\n  {'Budget (cells)':<16} {'Random':<12} {'Volcanic-informed':<20} {'Speedup':<10} {'Oracle'}")
    print("  " + "-" * 70)

    speedups = []
    for k in budgets:
        # Strategy A: Random
        expected_random = k * p_random

        # Strategy B: Survey high-suitability first, then medium, then low
        expected_volcanic = 0
        remaining = k
        for name in ["high", "medium", "low"]:
            z = zones[name]
            cells_in_zone = z["area_km2"]
            survey_in_zone = min(remaining, cells_in_zone)
            expected_volcanic += survey_in_zone * z["p_site"]
            remaining -= survey_in_zone
            if remaining <= 0:
                break

        # Strategy C: Oracle
        expected_oracle = min(k, n_sites)

        speedup = expected_volcanic / expected_random if expected_random > 0 else float("inf")
        speedups.append(speedup)
        print(f"  {k:<16} {expected_random:<12.1f} {expected_volcanic:<20.1f} {speedup:<10.2f}× {expected_oracle}")

    mean_speedup = np.mean(speedups)

    # ================================================================
    # ANALYSIS 4: Cost-Effectiveness of Targeting
    # ================================================================
    print("\n" + "=" * 70)
    print("[5] COST-EFFECTIVENESS OF VOLCANIC TARGETING")
    print("=" * 70)

    # GPR survey cost: $3,500 per cell (1 km²)
    cost_per_cell = 3500

    # To find the FIRST site:
    # Random: expected 1/p_random = 6.7 cells → $23,333
    # Volcanic: expected 1/p_high = 1/0.525 = 1.9 cells → $6,667
    p_high = zones["high"]["p_site"]

    cells_random = 1 / p_random
    cells_volcanic = 1 / p_high
    cost_random = cells_random * cost_per_cell
    cost_volcanic = cells_volcanic * cost_per_cell
    cost_savings = cost_random - cost_volcanic
    cost_ratio = cost_random / cost_volcanic

    print(f"\n  Cost per GPR survey: ${cost_per_cell:,}")
    print(f"\n  To find FIRST archaeological anomaly:")
    print(f"    Random search: {cells_random:.1f} surveys × ${cost_per_cell:,} = ${cost_random:,.0f}")
    print(f"    Volcanic-targeted: {cells_volcanic:.1f} surveys × ${cost_per_cell:,} = ${cost_volcanic:,.0f}")
    print(f"    Savings: ${cost_savings:,.0f} ({cost_ratio:.1f}× more efficient)")

    # For statistical significance (≥5 finds):
    target_finds = 5
    cells_random_5 = target_finds / p_random
    cells_volcanic_5 = target_finds / p_high
    cost_random_5 = cells_random_5 * cost_per_cell
    cost_volcanic_5 = cells_volcanic_5 * cost_per_cell

    print(f"\n  To find 5 anomalies (statistical significance):")
    print(f"    Random: {cells_random_5:.0f} surveys = ${cost_random_5:,.0f}")
    print(f"    Volcanic-targeted: {cells_volcanic_5:.0f} surveys = ${cost_volcanic_5:,.0f}")
    print(f"    Savings: ${cost_random_5 - cost_volcanic_5:,.0f}")

    # ================================================================
    # ANALYSIS 5: Depth Prediction Advantage
    # ================================================================
    print("\n" + "=" * 70)
    print("[6] DEPTH PREDICTION — The Unique VOLCARCH Advantage")
    print("=" * 70)

    # Without VOLCARCH: dig randomly, unclear how deep to go
    # With VOLCARCH: predict depth from distance + sedimentation rate

    # From E075 burial model (Pyle exponential):
    # Depth is a function of distance from volcano + eruption frequency
    # Correlation r=0.951 between predicted and observed depths

    distances_km = [5, 10, 15, 20, 30, 50]
    # Approximate predicted depths from E075 model
    predicted_depths = {
        5: 8.0,   # Very close to volcano
        10: 5.0,  # Typical candi zone
        15: 3.5,  # Outer candi zone
        20: 2.5,  # Court zone edge
        30: 1.5,  # Peripheral
        50: 0.5,  # Marginal volcanic influence
    }

    print(f"\n  {'Distance (km)':<16} {'Predicted depth':<18} {'Method needed':<25} {'Cost impact'}")
    print("  " + "-" * 75)

    for d, depth in predicted_depths.items():
        if depth < 1.0:
            method = "Surface survey"
            cost = "Low ($500)"
        elif depth < 2.0:
            method = "Shallow excavation"
            cost = "Medium ($2,000)"
        elif depth < 5.0:
            method = "GPR"
            cost = "Medium ($3,500)"
        else:
            method = "Deep coring/GPR"
            cost = "High ($5,000+)"
        print(f"  {d:<16} {depth:<18.1f}m {method:<25} {cost}")

    print(f"""
  KEY POINT: Without volcanic context, you don't know HOW DEEP to look.
  A shallow excavation at 5km from Kelud would find nothing (sites at 8m).
  VOLCARCH tells you not just WHERE but HOW DEEP — and this determines
  which survey method to use and therefore the total cost.
    """)

    # ================================================================
    # SYNTHESIS
    # ================================================================
    print("=" * 70)
    print("[7] SYNTHESIS — Why Volcanic Context Matters")
    print("=" * 70)

    print(f"""
  The Counter-4 critique is partially correct:
  - Survey deficit (40× leverage) IS the bigger absolute factor
  - Indonesia needs more archaeology regardless of volcanism

  BUT the critique misses VOLCARCH's actual contribution:

  1. SPATIAL TARGETING: {mean_speedup:.1f}× search efficiency improvement
     → Volcanic context reduces search space by concentrating effort
     → Information gain: {info_gain:.4f} bits ({info_gain_pct:.1f}% entropy reduction)

  2. DEPTH PREDICTION: Tells you HOW DEEP to look (r=0.951 accuracy)
     → Without VOLCARCH: dig randomly at random depth
     → With VOLCARCH: predict depth from distance, choose optimal method
     → Cost savings: ${cost_savings:,.0f} per first-find

  3. METHOD SELECTION: Determines which survey method is needed
     → <1m: surface survey ($500)
     → 1-5m: GPR ($3,500)
     → >5m: deep coring ($5,000+)
     → Wrong method = 100% wasted effort

  4. FALSIFIABILITY: VOLCARCH predicts specific things that can be tested
     → "More archaeology" is not a prediction — it's a plea
     → VOLCARCH predicts [0,6] anomalies at 20 specific locations
     → This is SCIENCE, not advocacy

  BOTTOM LINE:
  Survey deficit is the bigger PROBLEM.
  Volcanic context is the better SOLUTION.
  You can't target "do more archaeology." You CAN target "dig here, this deep."
    """)

    # Save results
    results = {
        "experiment": "E118_information_gain",
        "date": "2026-03-20",
        "purpose": "Quantify practical value of volcanic context for fieldwork targeting",
        "info_gain_bits": round(info_gain, 4),
        "entropy_reduction_pct": round(info_gain_pct, 1),
        "search_speedup_mean": round(mean_speedup, 2),
        "cost_first_find_random": round(cost_random),
        "cost_first_find_volcanic": round(cost_volcanic),
        "cost_ratio": round(cost_ratio, 1),
        "depth_prediction_r": 0.951,
        "zone_densities": {name: round(z["density_ratio"], 2) for name, z in zones.items()},
        "verdict": (
            f"Volcanic context provides {info_gain_pct:.1f}% entropy reduction in site search, "
            f"{mean_speedup:.1f}× mean search efficiency improvement, and "
            f"${cost_savings:,.0f} savings per first-find. Survey deficit is the bigger "
            f"problem; volcanic context is the better solution. VOLCARCH's value is "
            f"TARGETING (where + how deep), not explanation (why sites are missing)."
        ),
    }

    with open(OUT / "e118_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {OUT / 'e118_results.json'}")


if __name__ == "__main__":
    main()
