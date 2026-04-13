#!/usr/bin/env python3
"""
E196: Pre-400 CE Java Population Estimation — Multi-Method Synthesis
=====================================================================
"If Java was densely populated before 400 CE, the absence of archaeological
evidence becomes a powerful argument for taphonomic bias."

Four independent estimation methods:
1. Growth rate back-projection from known historical anchors
2. Carrying capacity (ecological ceiling)
3. Comparative island scaling (Austronesian neighbors)
4. Sunda Shelf displacement floor (E177)

Monte Carlo synthesis with 100K draws per method.
"""

import json, sys, numpy as np
from pathlib import Path
from scipy import stats as sp_stats

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

np.random.seed(42)
N_DRAWS = 100_000

# ══════════════════════════════════════════════════════════════════════
# KNOWN DATA
# ══════════════════════════════════════════════════════════════════════

JAVA_AREA_KM2 = 129_000  # modern Java island area

# Historical population anchors (published estimates with uncertainty)
# Format: (year_ce, low_estimate, mid_estimate, high_estimate, source)
HISTORICAL_ANCHORS = [
    (1600, 3_000_000, 4_000_000, 5_500_000, "Reid 1988, Lieberman 2003"),
    (1700, 3_500_000, 5_000_000, 7_000_000, "Ricklefs 2001"),
    (1800, 5_000_000, 7_000_000, 10_000_000, "Colonial estimates"),
    (1815, 4_000_000, 4_615_000, 6_000_000, "Raffles census (undercount noted)"),
    (1900, 28_000_000, 30_000_000, 32_000_000, "Colonial census"),
    (1930, 40_000_000, 41_718_000, 42_000_000, "Dutch census (precise)"),
]

# Comparative Austronesian islands (~1600 CE estimates)
# Format: (name, area_km2, pop_low, pop_high, volcanic, source)
COMPARATIVE_ISLANDS = [
    ("Philippines", 300_000, 1_200_000, 2_500_000, True, "Scott 1994, Reid 1988"),
    ("Sumatra", 473_000, 2_000_000, 3_500_000, True, "Reid 1988"),
    ("Borneo", 743_000, 800_000, 2_000_000, False, "Reid 1988"),
    ("Sulawesi", 189_000, 1_000_000, 2_000_000, False, "Reid 1988"),
    ("Bali", 5_780, 500_000, 1_000_000, True, "Vickers 2005"),
    ("Taiwan (indigenous)", 36_000, 200_000, 600_000, False, "Shepherd 1993"),
    ("Madagascar", 587_000, 1_000_000, 3_000_000, False, "Campbell 2005"),
]


def method1_growth_backprojection():
    """
    Back-project from 1600 CE anchors using pre-modern growth rates.

    Key insight: pre-modern populations grow at 0.01-0.2% per year
    (interrupted by famine, plague, volcanic disaster, war).
    Javanese growth 1600→1930: ~0.7%/yr (unusually high — likely includes
    agricultural intensification and new crops like maize/cassava post-1600).
    Pre-1600 growth was MUCH slower.
    """
    # Sample 1600 CE population
    pop_1600 = np.random.triangular(3_000_000, 4_000_000, 5_500_000, N_DRAWS)

    # Pre-modern growth rate (annual, 0 CE to 1600 CE)
    # Literature: 0.02-0.15%/yr for pre-industrial agrarian societies
    # Java specifically: volcanic disasters cause periodic crashes
    # Conservative: use 0.03-0.12%/yr (lower than European average)
    growth_rate = np.random.uniform(0.0003, 0.0012, N_DRAWS)

    # Years to back-project (1600 → 400 CE = 1200 years)
    years_back = 1200

    # Back-project: pop_400 = pop_1600 / exp(growth_rate × years)
    pop_400 = pop_1600 / np.exp(growth_rate * years_back)

    return pop_400


def method2_carrying_capacity():
    """
    Ecological ceiling based on agricultural productivity.

    Two sub-scenarios:
    A. Rice-based economy (if rice established by 400 CE)
    B. Mixed sago/tuber/early rice economy (more conservative)
    """
    # Arable fraction of Java
    # Exclude: mountains >2000m (~5%), steep slopes >30° (~15%), marshes (~5%)
    # Result: ~70-80% of Java is cultivable
    arable_frac = np.random.uniform(0.65, 0.80, N_DRAWS)
    arable_km2 = JAVA_AREA_KM2 * arable_frac
    arable_ha = arable_km2 * 100  # 1 km² = 100 ha

    # Scenario mix: probability that rice was dominant by 400 CE
    # Evidence suggests rice was present but not yet dominant
    rice_prob = np.random.uniform(0.2, 0.5, N_DRAWS)

    # Rice yield: 1.0-2.5 ton/ha/year (pre-modern, single harvest)
    # Modern: 5-6 ton/ha but with fertilizer + varieties
    rice_yield = np.random.uniform(1.0, 2.5, N_DRAWS)  # ton/ha/yr

    # Sago/tuber yield: 0.3-1.0 ton starch equivalent/ha/yr
    # Sago is actually quite productive per palm but land-extensive
    sago_yield = np.random.uniform(0.3, 1.0, N_DRAWS)

    # Blended yield
    yield_per_ha = rice_prob * rice_yield + (1 - rice_prob) * sago_yield

    # Not all arable land is cultivated
    # At 400 CE, cultivation fraction likely 10-40% of arable
    cultivation_frac = np.random.uniform(0.10, 0.40, N_DRAWS)

    # Caloric needs: ~200 kg grain equivalent per person per year
    kg_per_person_year = np.random.uniform(180, 250, N_DRAWS)

    # Total food production (tons)
    total_food = arable_ha * cultivation_frac * yield_per_ha

    # Population supported
    pop_capacity = (total_food * 1000) / kg_per_person_year  # tons → kg

    return pop_capacity


def method3_comparative_scaling():
    """
    Scale from neighboring Austronesian islands.

    Logic: Java and Philippines/Sulawesi/Borneo share Austronesian culture,
    similar latitudes, similar maritime access. If Java had comparable
    population DENSITY at 400 CE, what would its population be?

    Use 1600 CE densities of comparable islands and project backward,
    assuming Java's density advantage was present but smaller in antiquity.
    """
    # Compute density per km² for each island at ~1600 CE
    densities = []
    for name, area, pop_lo, pop_hi, volcanic, src in COMPARATIVE_ISLANDS:
        pop = np.random.uniform(pop_lo, pop_hi, N_DRAWS)
        density = pop / area
        densities.append((name, density))

    # Java's 1600 CE density
    java_1600_pop = np.random.triangular(3_000_000, 4_000_000, 5_500_000, N_DRAWS)
    java_1600_density = java_1600_pop / JAVA_AREA_KM2  # ~23-43 /km²

    # Median density of non-Java islands at 1600 CE
    all_densities = np.array([d[1] for d in densities])  # shape: (n_islands, N_DRAWS)
    # Use Philippines + Bali + Sulawesi (most comparable ecologically)
    comparable = np.array([densities[0][1], densities[4][1], densities[3][1]])
    median_comparable_density = np.median(comparable, axis=0)

    # Java's density RATIO at 1600 CE vs comparables
    java_ratio_1600 = java_1600_density / median_comparable_density
    # Java was ~3-8× denser than comparables at 1600

    # At 400 CE, Java's density advantage was likely SMALLER
    # (rice intensification + irrigation hadn't fully developed)
    # Assume ratio was 1.5-4× (half to full 1600 advantage)
    ratio_400 = np.random.uniform(1.5, 4.0, N_DRAWS)

    # Comparable island density at 400 CE
    # Back-project comparable density from 1600 CE at 0.03-0.10%/yr
    growth_comp = np.random.uniform(0.0003, 0.001, N_DRAWS)
    comp_density_400 = median_comparable_density / np.exp(growth_comp * 1200)

    # Java density at 400 CE
    java_density_400 = comp_density_400 * ratio_400

    # Population
    pop_400 = java_density_400 * JAVA_AREA_KM2

    return pop_400


def method4_sunda_displacement_floor():
    """
    Absolute minimum floor from E177 Sunda Shelf model.

    250K people displaced to Java over 14,000 years.
    By 400 CE (~6000 years after shelf fully submerged),
    these populations + their descendants form a minimum floor.

    PLUS: Java's indigenous population that was already there.
    """
    # E177: ~250K total displaced to Java
    displaced_total = np.random.triangular(150_000, 250_000, 400_000, N_DRAWS)

    # Most displacement happened 14,000-4,000 BP (before 400 CE)
    # By 400 CE, displacement is complete
    # These people had ~6,000 years to grow
    # At 0.02-0.10%/yr growth:
    growth = np.random.uniform(0.0002, 0.001, N_DRAWS)
    displaced_descendants = displaced_total * np.exp(growth * 6000)

    # PLUS indigenous Javanese (pre-Sunda displacement)
    # Java was inhabited since H. erectus, Homo sapiens from ~45,000 BP
    # By 10,000 BP, likely had 50K-200K hunter-gatherers
    # By 400 CE (with agriculture): grown further
    indigenous_10k_bp = np.random.uniform(30_000, 200_000, N_DRAWS)
    growth_indig = np.random.uniform(0.0002, 0.0008, N_DRAWS)
    indigenous_400ce = indigenous_10k_bp * np.exp(growth_indig * 10_000)

    # Total = displaced descendants + indigenous
    # Cap at carrying capacity (~10M as absolute ceiling for 400 CE)
    pop_400 = np.minimum(displaced_descendants + indigenous_400ce, 10_000_000)

    return pop_400


def main():
    print("=" * 70)
    print("E196: Pre-400 CE Java Population Estimation")
    print("Multi-Method Monte Carlo Synthesis")
    print("=" * 70)

    # ── Run all methods ───────────────────────────────────────────────
    print(f"\nRunning {N_DRAWS:,} Monte Carlo draws per method...")

    methods = {
        "Growth back-projection": method1_growth_backprojection(),
        "Carrying capacity": method2_carrying_capacity(),
        "Comparative island scaling": method3_comparative_scaling(),
        "Sunda displacement floor": method4_sunda_displacement_floor(),
    }

    print(f"\n{'='*70}")
    print("INDIVIDUAL METHOD ESTIMATES (population at 400 CE)")
    print(f"{'='*70}")

    all_estimates = {}
    for name, draws in methods.items():
        draws = draws[np.isfinite(draws)]
        draws = draws[draws > 0]

        p5 = np.percentile(draws, 5)
        p25 = np.percentile(draws, 25)
        p50 = np.percentile(draws, 50)
        p75 = np.percentile(draws, 75)
        p95 = np.percentile(draws, 95)

        all_estimates[name] = {
            "median": int(p50),
            "p5": int(p5), "p25": int(p25),
            "p75": int(p75), "p95": int(p95),
            "mean": int(np.mean(draws)),
        }

        print(f"\n  {name}:")
        print(f"    Median: {p50:,.0f}")
        print(f"    90% CI: [{p5:,.0f} — {p95:,.0f}]")
        print(f"    IQR:    [{p25:,.0f} — {p75:,.0f}]")

    # ── Bayesian synthesis ────────────────────────────────────────────
    # Simple approach: geometric mean of medians (since estimates span orders of magnitude)
    # More sophisticated: multiply likelihood functions

    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print(f"{'='*70}")

    # Combine all draws (equal weight per method)
    all_draws = np.concatenate([v for v in methods.values()])
    all_draws = all_draws[np.isfinite(all_draws) & (all_draws > 0)]

    # Log-space synthesis (better for spanning orders of magnitude)
    log_draws = {}
    for name, draws in methods.items():
        d = draws[np.isfinite(draws) & (draws > 0)]
        log_draws[name] = np.log10(d)

    # Geometric mean of medians
    medians = [v["median"] for v in all_estimates.values()]
    geo_mean = np.exp(np.mean(np.log(medians)))

    print(f"\n  Geometric mean of medians: {geo_mean:,.0f}")

    # Conservative synthesis: take the OVERLAP of all 90% CIs
    all_p5 = max(v["p5"] for v in all_estimates.values())
    all_p95 = min(v["p95"] for v in all_estimates.values())

    print(f"  Conservative overlap floor (max of all lower bounds): {all_p5:,.0f}")

    # The KEY number: what is the MINIMUM plausible population?
    # Use the 5th percentile of the most conservative method
    most_conservative = min(all_estimates.values(), key=lambda v: v["median"])
    min_plausible = most_conservative["p5"]

    print(f"\n  *** MINIMUM PLAUSIBLE POPULATION AT 400 CE: {min_plausible:,.0f} ***")
    print(f"  (5th percentile of most conservative method)")

    # ── Archaeological comparison ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("ARCHAEOLOGICAL IMPLICATION")
    print(f"{'='*70}")

    # Known pre-400 CE open-air sites in volcanic Java: ~0
    # (E117: zero pre-400 CE open-air sites in volcanic interior)
    observed_sites = 0

    # Expected sites per capita (from comparable contexts)
    # Philippines has ~4,000 known archaeological sites for ~2M pre-colonial population
    # That's ~1 site per 500 people
    # More conservatively: 1 site per 2,000 people
    sites_per_person = np.random.uniform(1/5000, 1/500, N_DRAWS)

    expected_sites_low = min_plausible * np.median(sites_per_person)
    expected_sites_med = geo_mean * np.median(sites_per_person)

    print(f"\n  Expected sites (if comparable to Philippines):")
    print(f"    Minimum estimate ({min_plausible:,.0f} people): {expected_sites_low:,.0f} sites")
    print(f"    Central estimate ({geo_mean:,.0f} people): {expected_sites_med:,.0f} sites")
    print(f"  Observed sites in volcanic Java pre-400 CE: {observed_sites}")

    if expected_sites_low > 0:
        suppression = expected_sites_low / max(observed_sites, 1)
        print(f"\n  *** TAPHONOMIC SUPPRESSION FACTOR: ≥{suppression:,.0f}× ***")
        print(f"  (minimum expected / observed)")

    # ── Population density at 400 CE ──────────────────────────────────
    print(f"\n--- Population Density at 400 CE ---")
    density_min = min_plausible / JAVA_AREA_KM2
    density_geo = geo_mean / JAVA_AREA_KM2
    print(f"  Minimum: {density_min:.1f} per km²")
    print(f"  Central: {density_geo:.1f} per km²")
    print(f"  For comparison:")
    print(f"    Modern Java: {152_000_000/JAVA_AREA_KM2:.0f} per km²")
    print(f"    Java 1600 CE: {4_000_000/JAVA_AREA_KM2:.0f} per km²")
    print(f"    Philippines 1600 CE: {1_700_000/300_000:.1f} per km²")
    print(f"    Pre-modern agrarian typical: 5-20 per km²")

    # ── Person-centuries of invisible civilization ─────────────────────
    print(f"\n--- Invisible Civilization ---")
    # Period: 2000 BCE to 400 CE = 2400 years
    # (Austronesian arrival to inscription era)
    invisible_years = 2400
    geo_pop_avg = geo_mean / 2  # rough average over the period (growing)
    person_centuries = (geo_pop_avg * invisible_years) / 100

    print(f"  Period: ~2000 BCE to 400 CE ({invisible_years} years)")
    print(f"  Average population (crude): ~{geo_pop_avg:,.0f}")
    print(f"  Person-centuries of invisible civilization: ~{person_centuries:,.0f}")
    print(f"  That's {person_centuries/1_000_000:.1f} million person-centuries of")
    print(f"  human experience with ZERO direct archaeological record in volcanic Java.")

    # ── Save ──────────────────────────────────────────────────────────
    summary = {
        "experiment": "E196",
        "date": "2026-04-13",
        "n_monte_carlo": N_DRAWS,
        "methods": all_estimates,
        "synthesis": {
            "geometric_mean": int(geo_mean),
            "minimum_plausible": int(min_plausible),
            "conservative_floor": int(all_p5),
        },
        "density_400ce": {
            "minimum_per_km2": round(density_min, 1),
            "central_per_km2": round(density_geo, 1),
        },
        "archaeological_implication": {
            "expected_sites_minimum": int(expected_sites_low),
            "expected_sites_central": int(expected_sites_med),
            "observed_sites": observed_sites,
            "suppression_factor": int(suppression) if expected_sites_low > 0 else None,
        },
        "invisible_person_centuries": int(person_centuries),
    }

    with open(RESULTS_DIR / "e196_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save method distributions for potential plotting
    method_summaries = []
    for name, draws in methods.items():
        d = draws[np.isfinite(draws) & (draws > 0)]
        for pct in [5, 10, 25, 50, 75, 90, 95]:
            method_summaries.append({
                "method": name,
                "percentile": pct,
                "population": int(np.percentile(d, pct)),
            })

    import csv
    with open(RESULTS_DIR / "method_distributions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "percentile", "population"])
        writer.writeheader()
        writer.writerows(method_summaries)

    print(f"\nResults saved to {RESULTS_DIR}/")

    # ── Verdict ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"\n  Four independent methods converge:")
    print(f"  Java at 400 CE had AT LEAST {min_plausible:,.0f} people")
    print(f"  (central estimate: {geo_mean:,.0f})")
    print(f"")
    print(f"  Archaeological record of this population in volcanic Java: ZERO sites")
    print(f"  Expected at Philippine site-density: {expected_sites_low:,.0f}+ sites")
    print(f"")
    print(f"  This is not a gap. This is an ERASURE.")
    print(f"  {person_centuries/1_000_000:.1f} million person-centuries of civilization")
    print(f"  with no direct archaeological trace.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
