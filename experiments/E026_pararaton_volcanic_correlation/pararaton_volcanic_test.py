"""
E026: Pararaton Volcanic Correlation Test

Tests whether Kelud eruptions temporally precede Majapahit political crises
at a rate higher than chance. If yes, volcanic stress is a plausible trigger
for the political instability that ended Majapahit.

Three analyses:
1. Eruption-crisis temporal proximity (permutation test)
2. Eruption rate comparison: peak vs decline periods
3. Pararaton geological events vs GVP cross-reference

Sources for Pararaton events:
- Brandes 1896 (original transliteration)
- Poerbatjaraka 1940 (Indonesian translation)
- Pigeaud 1960/1967 (English summary in Literature of Java)
- Noorduyn & Verstappen 1972 (geological interpretation)
- Satyana 2015 (mud volcano hypothesis)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import csv
import json
import numpy as np
from pathlib import Path

np.random.seed(42)

# ============================================================
# DATA: KELUD ERUPTIONS FROM GVP (already in repo)
# ============================================================

def load_kelud_eruptions(csv_path):
    """Load Kelud eruptions from GVP eruption_history.csv."""
    eruptions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['volcano'] == 'Kelud':
                year = int(row['year'])
                vei = float(row['vei']) if row['vei'] else None
                eruptions.append({'year': year, 'vei': vei})
    return eruptions

# ============================================================
# DATA: PARARATON EVENTS (compiled from published sources)
# ============================================================

# Political/military crises from Pararaton + epigraphic cross-references
# Sources: Pigeaud 1960, Noorduyn 1978, Ricklefs 2001
PARARATON_CRISES = [
    # (year, event, type, source)
    (1292, "Jayakatwang usurps Singasari, Kertanagara killed", "succession", "Pararaton; Nagarakretagama"),
    (1293, "Mongol invasion, Raden Wijaya founds Majapahit", "war", "Pararaton; Yuan Shi"),
    (1309, "Nambi rebellion suppressed", "rebellion", "Pararaton"),
    (1316, "Ra Kuti rebellion, Jayanagara nearly overthrown", "rebellion", "Pararaton"),
    (1328, "Jayanagara assassinated by Ra Tanca (physician)", "succession", "Pararaton"),
    (1334, "Sadeng and Keta rebellions (coincides with banyu pindah)", "rebellion", "Pararaton"),
    (1357, "Gajah Mada dies", "succession", "Pararaton; Nagarakretagama"),
    (1389, "Hayam Wuruk dies — peak of Majapahit ends", "succession", "Pararaton; Nagarakretagama"),
    (1401, "Paregreg Civil War begins (Wikramawardhana vs Wirabhumi)", "war", "Pararaton"),
    (1405, "Wirabhumi defeated and killed, Paregreg ends", "war", "Pararaton"),
    (1426, "Great famine recorded", "famine", "Pararaton"),
    (1447, "Suhita dies, succession instability", "succession", "Pararaton"),
    (1451, "Kertawijaya dies after brief reign", "succession", "Pararaton"),
    (1453, "Rajasawardhana dies, Bhre Wengker period", "succession", "Pararaton"),
    (1456, "Bhre Wengker (Singhawikramawardhana) takes power", "succession", "Pararaton"),
    (1466, "Bhre Pandan Salas (Suraprabhawa) becomes king", "succession", "Pararaton"),
    (1468, "Bhre Kertabhumi splits the kingdom — dual kingship", "civil_conflict", "Pararaton"),
    (1478, "Sirna ilang krtaning bhumi — political collapse", "collapse", "Pararaton"),
]

# Geological events recorded in Pararaton
PARARATON_GEOLOGICAL = [
    (1334, "banyu pindah", "flooding/river course change", "Pararaton"),
    (1374, "pagunung anyar", "new mountain / mud volcano emergence", "Pararaton"),
    (1481, "guntur pawatugunung", "volcanic eruption — last entry", "Pararaton"),
]

# Period definitions
MAJAPAHIT_START = 1293
MAJAPAHIT_END = 1527  # Demak conquest
PEAK_END = 1375  # End of expansion / Hayam Wuruk era
DECLINE_START = 1376  # Kelud cluster begins


# ============================================================
# ANALYSIS 1: ERUPTION-CRISIS TEMPORAL PROXIMITY
# ============================================================

def compute_proximity(crisis_years, eruption_years):
    """For each crisis, find years since the most recent preceding eruption.
    Returns list of proximities (NaN if no preceding eruption)."""
    proximities = []
    for cy in crisis_years:
        preceding = [ey for ey in eruption_years if ey <= cy]
        if preceding:
            proximities.append(cy - max(preceding))
        else:
            proximities.append(np.nan)
    return proximities


def proximity_permutation_test(crisis_years, eruption_years, n_perm=10000):
    """Permutation test: randomize crisis years within Majapahit period,
    compare mean proximity to observed."""
    observed = compute_proximity(crisis_years, eruption_years)
    observed_clean = [x for x in observed if not np.isnan(x)]
    if not observed_clean:
        return None, None, None
    observed_mean = np.mean(observed_clean)

    # Only use crises AFTER first eruption in period
    first_eruption = min(eruption_years)
    valid_crises = [y for y in crisis_years if y >= first_eruption]
    n_crises = len(valid_crises)

    count_leq = 0
    null_means = []
    for _ in range(n_perm):
        # Random crisis years within the Majapahit period (after first eruption)
        rand_years = sorted(np.random.randint(first_eruption, MAJAPAHIT_END + 1, size=n_crises))
        rand_prox = compute_proximity(list(rand_years), eruption_years)
        rand_clean = [x for x in rand_prox if not np.isnan(x)]
        if rand_clean:
            rand_mean = np.mean(rand_clean)
            null_means.append(rand_mean)
            if rand_mean <= observed_mean:
                count_leq += 1

    p_value = count_leq / n_perm
    return observed_mean, p_value, null_means


# ============================================================
# ANALYSIS 2: ERUPTION RATE COMPARISON
# ============================================================

def eruption_rate_test(eruption_years, peak_end=PEAK_END):
    """Compare eruption frequency: peak period vs decline period."""
    peak_eruptions = [y for y in eruption_years if MAJAPAHIT_START <= y <= peak_end]
    decline_eruptions = [y for y in eruption_years if peak_end < y <= MAJAPAHIT_END]

    peak_duration = peak_end - MAJAPAHIT_START + 1
    decline_duration = MAJAPAHIT_END - peak_end

    peak_rate = len(peak_eruptions) / peak_duration * 100  # per century
    decline_rate = len(decline_eruptions) / decline_duration * 100

    rate_ratio = decline_rate / peak_rate if peak_rate > 0 else float('inf')

    return {
        "peak_eruptions": len(peak_eruptions),
        "peak_years": peak_eruptions,
        "peak_duration_yr": peak_duration,
        "peak_rate_per_century": round(peak_rate, 1),
        "decline_eruptions": len(decline_eruptions),
        "decline_years": decline_eruptions,
        "decline_duration_yr": decline_duration,
        "decline_rate_per_century": round(decline_rate, 1),
        "rate_ratio": round(rate_ratio, 2),
    }


# ============================================================
# ANALYSIS 3: PARARATON GEOLOGICAL EVENTS vs GVP
# ============================================================

def cross_reference_geological(pararaton_geo, kelud_years, tolerance=5):
    """Check if Pararaton geological events match GVP eruption records."""
    results = []
    for year, name, description, source in pararaton_geo:
        matches = [ey for ey in kelud_years if abs(ey - year) <= tolerance]
        results.append({
            "year": year,
            "name": name,
            "description": description,
            "gvp_match": matches if matches else None,
            "match_within_tolerance": len(matches) > 0,
        })
    return results


# ============================================================
# ANALYSIS 4: POST-ERUPTION CRISIS WINDOW
# ============================================================

def post_eruption_window_test(crisis_years, eruption_years, window=15, n_perm=10000):
    """What fraction of crises occur within N years after an eruption?
    Compare to null distribution."""
    def fraction_in_window(cy_list, ey_list, w):
        count = 0
        for cy in cy_list:
            for ey in ey_list:
                if 0 < (cy - ey) <= w:
                    count += 1
                    break
        return count / len(cy_list) if cy_list else 0

    observed_frac = fraction_in_window(crisis_years, eruption_years, window)

    count_geq = 0
    for _ in range(n_perm):
        rand_years = sorted(np.random.randint(MAJAPAHIT_START, MAJAPAHIT_END + 1,
                                               size=len(crisis_years)))
        rand_frac = fraction_in_window(list(rand_years), eruption_years, window)
        if rand_frac >= observed_frac:
            count_geq += 1

    p_value = count_geq / n_perm
    return observed_frac, p_value


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("E026: PARARATON VOLCANIC CORRELATION TEST")
    print("Does Kelud volcanic activity precede Majapahit political crises?")
    print("=" * 70)

    # Load data
    csv_path = Path(__file__).parent.parent.parent / "data" / "processed" / "eruption_history.csv"
    all_kelud = load_kelud_eruptions(csv_path)

    # Filter to Majapahit period (1200-1530 CE)
    kelud_majapahit = [e for e in all_kelud if 1200 <= e['year'] <= 1530]
    kelud_years = [e['year'] for e in kelud_majapahit]

    crisis_years = [e[0] for e in PARARATON_CRISES]
    crisis_types = {e[0]: e[2] for e in PARARATON_CRISES}

    print(f"\n--- DATA SUMMARY ---")
    print(f"  Kelud eruptions (1200-1530): {len(kelud_years)}")
    print(f"    Years: {kelud_years}")
    print(f"    VEI:   {[e['vei'] for e in kelud_majapahit]}")
    print(f"  Pararaton political crises: {len(crisis_years)}")
    for y, desc, typ, src in PARARATON_CRISES:
        print(f"    {y}: [{typ}] {desc}")
    print(f"  Pararaton geological events: {len(PARARATON_GEOLOGICAL)}")
    for y, name, desc, src in PARARATON_GEOLOGICAL:
        print(f"    {y}: {name} — {desc}")

    results = {}

    # === ANALYSIS 1: PROXIMITY TEST ===
    print("\n" + "=" * 70)
    print("ANALYSIS 1: ERUPTION-CRISIS TEMPORAL PROXIMITY")
    print("For each crisis, how many years since the last Kelud eruption?")
    print("=" * 70)

    proximities = compute_proximity(crisis_years, kelud_years)
    print(f"\n  Observed proximities (years since last eruption):")
    for i, (y, desc, typ, _) in enumerate(PARARATON_CRISES):
        prox = proximities[i]
        prox_str = f"{prox:.0f}" if not np.isnan(prox) else "N/A"
        print(f"    {y} [{typ:>15s}]: {prox_str:>4s} years  | {desc}")

    observed_mean, p_value, null_means = proximity_permutation_test(
        crisis_years, kelud_years, n_perm=10000
    )

    print(f"\n  Observed mean proximity: {observed_mean:.1f} years")
    print(f"  Null distribution mean:  {np.mean(null_means):.1f} years")
    print(f"  Null distribution 5th percentile: {np.percentile(null_means, 5):.1f} years")
    print(f"  p-value (one-sided, crises follow eruptions more closely than chance): {p_value:.4f}")

    if p_value < 0.05:
        print(f"  >> SIGNIFICANT at alpha=0.05 — crises cluster after eruptions")
    elif p_value < 0.10:
        print(f"  >> MARGINAL (p < 0.10) — suggestive but not conclusive")
    else:
        print(f"  >> NOT SIGNIFICANT — no evidence crises follow eruptions")

    results["proximity_test"] = {
        "observed_mean_proximity_years": round(observed_mean, 1),
        "null_mean": round(np.mean(null_means), 1),
        "p_value": round(p_value, 4),
        "n_permutations": 10000,
    }

    # === ANALYSIS 1b: POST-ERUPTION WINDOW ===
    print("\n--- Analysis 1b: Post-eruption crisis window ---")
    for window in [5, 10, 15, 20]:
        frac, p = post_eruption_window_test(crisis_years, kelud_years, window=window)
        print(f"  Window={window:>2d}yr: {frac*100:.0f}% of crises within window (p={p:.4f})")
        results[f"window_{window}yr"] = {"fraction": round(frac, 3), "p_value": round(p, 4)}

    # === ANALYSIS 2: ERUPTION RATE COMPARISON ===
    print("\n" + "=" * 70)
    print("ANALYSIS 2: ERUPTION RATE — PEAK vs DECLINE")
    print(f"Peak period: {MAJAPAHIT_START}-{PEAK_END} (expansion, strong kings)")
    print(f"Decline period: {DECLINE_START}-{MAJAPAHIT_END} (post-Hayam Wuruk)")
    print("=" * 70)

    rate_results = eruption_rate_test(kelud_years)
    print(f"\n  Peak ({MAJAPAHIT_START}-{PEAK_END}):")
    print(f"    Eruptions: {rate_results['peak_eruptions']} in {rate_results['peak_duration_yr']} years")
    print(f"    Years: {rate_results['peak_years']}")
    print(f"    Rate: {rate_results['peak_rate_per_century']} per century")
    print(f"\n  Decline ({DECLINE_START}-{MAJAPAHIT_END}):")
    print(f"    Eruptions: {rate_results['decline_eruptions']} in {rate_results['decline_duration_yr']} years")
    print(f"    Years: {rate_results['decline_years']}")
    print(f"    Rate: {rate_results['decline_rate_per_century']} per century")
    print(f"\n  Rate ratio (decline/peak): {rate_results['rate_ratio']}x")

    if rate_results['rate_ratio'] >= 2.0:
        print(f"  >> MEETS THRESHOLD (ratio >= 2.0) — eruptions concentrated in decline")
    else:
        print(f"  >> BELOW THRESHOLD (ratio < 2.0)")

    results["rate_comparison"] = rate_results

    # === ANALYSIS 3: GEOLOGICAL CROSS-REFERENCE ===
    print("\n" + "=" * 70)
    print("ANALYSIS 3: PARARATON GEOLOGICAL EVENTS vs GVP RECORDS")
    print(f"Tolerance: ±5 years")
    print("=" * 70)

    geo_results = cross_reference_geological(PARARATON_GEOLOGICAL, kelud_years)
    n_matched = sum(1 for r in geo_results if r['match_within_tolerance'])
    for r in geo_results:
        match_str = f"GVP match: {r['gvp_match']}" if r['gvp_match'] else "No GVP match"
        print(f"\n  {r['year']} — {r['name']} ({r['description']})")
        print(f"    {match_str}")

    print(f"\n  Matched: {n_matched}/{len(geo_results)} Pararaton geological events have GVP counterpart")
    results["geological_crossref"] = {
        "matched": n_matched,
        "total": len(geo_results),
        "details": geo_results,
    }

    # === ANALYSIS 4: CRISIS TYPE BREAKDOWN ===
    print("\n" + "=" * 70)
    print("ANALYSIS 4: CRISIS TIMING BY TYPE")
    print("Are wars/famines (high-impact) more associated with eruptions than successions?")
    print("=" * 70)

    high_impact = [(y, t) for y, t in crisis_types.items()
                   if t in ('war', 'famine', 'collapse', 'civil_conflict')]
    low_impact = [(y, t) for y, t in crisis_types.items()
                  if t in ('succession', 'rebellion')]

    high_years = [y for y, t in high_impact]
    low_years = [y for y, t in low_impact]

    high_prox = [x for x in compute_proximity(high_years, kelud_years) if not np.isnan(x)]
    low_prox = [x for x in compute_proximity(low_years, kelud_years) if not np.isnan(x)]

    if high_prox and low_prox:
        print(f"\n  High-impact crises (war/famine/collapse): n={len(high_prox)}")
        print(f"    Mean proximity to eruption: {np.mean(high_prox):.1f} years")
        print(f"    Crises: {high_years}")
        print(f"\n  Low-impact crises (succession/rebellion): n={len(low_prox)}")
        print(f"    Mean proximity to eruption: {np.mean(low_prox):.1f} years")
        print(f"    Crises: {low_years}")

        diff = np.mean(low_prox) - np.mean(high_prox)
        print(f"\n  Difference: high-impact crises are {diff:.1f} years CLOSER to eruptions on average")
        results["crisis_type_breakdown"] = {
            "high_impact_mean_proximity": round(np.mean(high_prox), 1),
            "low_impact_mean_proximity": round(np.mean(low_prox), 1),
            "difference_years": round(diff, 1),
        }

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY — GO/NO-GO ASSESSMENT")
    print("=" * 70)

    criteria = []

    # Criterion 1: proximity p < 0.05
    c1 = p_value < 0.05
    criteria.append(c1)
    print(f"\n  1. Eruption-crisis proximity p < 0.05:  {'GO' if c1 else 'NO-GO'} (p={p_value:.4f})")

    # Criterion 2: rate ratio >= 2.0
    c2 = rate_results['rate_ratio'] >= 2.0
    criteria.append(c2)
    print(f"  2. Eruption rate ratio >= 2.0:          {'GO' if c2 else 'NO-GO'} (ratio={rate_results['rate_ratio']})")

    # Criterion 3: >= 2/3 Pararaton geo events match GVP
    c3 = n_matched >= 2
    criteria.append(c3)
    print(f"  3. Pararaton-GVP match >= 2/3:          {'GO' if c3 else 'NO-GO'} ({n_matched}/3)")

    n_go = sum(criteria)
    overall = "GO" if n_go >= 2 else "NO-GO"
    print(f"\n  OVERALL: {overall} ({n_go}/3 criteria met)")

    if overall == "GO":
        print("\n  >> Proceed to P14 paper draft.")
        print("  >> Kelud volcanic stress is a plausible trigger for Majapahit decline.")
    else:
        print("\n  >> Insufficient evidence for volcanic trigger hypothesis.")
        print("  >> Document as informative negative; consider alternative framings.")

    results["summary"] = {
        "criterion_1_proximity": {"met": c1, "p_value": round(p_value, 4)},
        "criterion_2_rate_ratio": {"met": c2, "ratio": rate_results['rate_ratio']},
        "criterion_3_geo_crossref": {"met": c3, "matched": n_matched, "total": 3},
        "overall": overall,
        "criteria_met": n_go,
    }

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_dir / "pararaton_correlation_results.json", "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n  Results saved to {output_dir / 'pararaton_correlation_results.json'}")


if __name__ == "__main__":
    main()
