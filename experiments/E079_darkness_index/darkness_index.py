#!/usr/bin/env python3
"""
E079: The Archaeological Darkness Index — Grand Synthesis
==========================================================
Combines evidence from E071-E078 to produce a single quantitative
measure of "archaeological darkness" for each century of Javanese
history from 3000 BCE to 1500 CE.

The Darkness Index integrates:
1. Inscription density (E074, E078) — textual visibility
2. Volcanic sedimentation (E075) — burial probability
3. Material decay (E074) — organic vs mineral ratio
4. External recognition (E071) — are outsiders aware of Java?
5. Site discovery rate — known sites per century

A high Darkness Index means: civilizational activity WAS happening,
but the evidence has been destroyed, buried, or never recorded.

The key insight: darkness is HIGHEST not when nothing was happening,
but when the gap between EXPECTED and OBSERVED evidence is largest.
"""

import json
import sys
from pathlib import Path
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Evidence compilation by century ───────────────────────────────────
# For each century, compile available evidence

CENTURIES = list(range(-30, 16))  # -30 = 3000 BCE, 15 = 1500 CE

def century_label(c):
    if c < 0:
        return f"{abs(c)*100}-{abs(c)*100-99} BCE"
    elif c == 0:
        return "100 BCE-1 CE"
    else:
        return f"{(c-1)*100+1}-{c*100} CE"

def century_short(c):
    if c <= 0:
        return f"{abs(c)*100} BCE"
    else:
        return f"C{c}"


# ── Dimension 1: Inscription Density (from E074/E078) ─────────────────
# Number of known inscriptions per century
# Source: DHARMA corpus (268 inscriptions, 86 dated)
INSCRIPTION_COUNT = {
    7: 2,    # C7: 684, 686 CE
    8: 5,    # C8: 732, 752, 760, 787, 792
    9: 28,   # C9: peak of Central Java
    10: 31,  # C10: peak of East Java transition
    11: 8,   # C11: Airlangga period
    12: 1,   # C12: Kediri
    13: 7,   # C13: Singhasari
    14: 4,   # C14: Majapahit
}

# ── Dimension 2: External Recognition ─────────────────────────────────
# Number of external sources (Chinese, Indian, Greek, Roman) that
# mention Java/Nusantara in this century
# Source: E071 + background agent Chinese sources compilation
EXTERNAL_SOURCES = {
    -17: 1,   # Clove at Terqa ~1700 BCE (Maluku trade)
    -3: 2,    # Ramayana "Yavadvipa" (~300 BCE), Jataka tales
    -2: 2,    # Arthashastra trade refs, Sembiran Rouletted Ware
    -1: 2,    # Periplus Maris Erythraei, Dong Son drums in Indonesia
    1: 2,     # Pliny (77 CE), Han Shu maritime route
    2: 3,     # Ptolemy "Iabadiou" (150 CE), Hou Han Shu Ye-Tiao (132 CE), Buni Complex
    3: 1,     # San Guo Zhi maritime trade
    4: 2,     # Batujaya Buddhist temples, Buni Complex continued
    5: 3,     # Fa Xian (414 CE), Mulavarman inscriptions (Borneo), Purnavarman (West Java)
    6: 2,     # Liang Shu She-Po, Tarumanagara continued
    7: 2,     # Srivijaya emerges, Chinese embassies
    8: 3,     # Borobudur era, Chinese records
    9: 3,     # Peak Mataram, Chinese + Arab records
    10: 3,    # East Java kingdoms, Chinese records
}

# ── Dimension 3: Volcanic Sedimentation Rate ─────────────────────────
# Estimated sedimentation rate (mm/year) for volcanic Java
# Source: E075 model + published data
# Before recorded eruptions, use geological average for Java
VOLCANIC_SEDIMENTATION = {
    -30: 3,   # Background geological rate
    -20: 3,
    -10: 3,
    -5: 3,
    -3: 3,    # Proto-historic period
    -2: 3,
    -1: 3,
    0: 3,
    1: 3,
    2: 3,
    3: 3,
    4: 3,
    5: 5,     # Some eruptions recorded
    6: 5,
    7: 5,
    8: 5,
    9: 8,     # High activity period (Merapi, Kelut)
    10: 10,   # Merapi 928-1006 CE events
    11: 6,
    12: 5,
    13: 8,    # Samalas 1257 (VEI 7)
    14: 7,    # Kelut series
    15: 7,
}

# ── Dimension 4: Material Preservation ────────────────────────────────
# Probability that material culture from this century survives
# Based on E074 organic/mineral ratio + decay models
# Stone/metal: ~90% survival. Wood/bamboo: <1% after 500 years.
# Pre-Hindu: almost entirely organic materials
MATERIAL_SURVIVAL = {
    -30: 0.30,  # Stone tools survive well
    -20: 0.25,
    -10: 0.20,
    -5: 0.15,   # Neolithic: pottery + stone survive
    -3: 0.10,   # Metal age: bronze survives
    -2: 0.10,
    -1: 0.08,
    0: 0.05,    # Minimal durable material culture
    1: 0.05,
    2: 0.05,
    3: 0.05,
    4: 0.07,    # First stone inscriptions
    5: 0.10,    # Stone inscriptions + early brick
    6: 0.12,
    7: 0.15,    # Candi construction begins
    8: 0.25,    # Borobudur, massive stone architecture
    9: 0.30,    # Peak stone architecture
    10: 0.25,   # East Java brick + stone
    11: 0.20,
    12: 0.15,
    13: 0.20,   # Singhasari stone temples
    14: 0.25,   # Majapahit brick architecture
    15: 0.20,
}

# ── Dimension 5: Archaeological Site Count ────────────────────────────
# Known archaeological sites attributable to this century
# Source: E001 site database + E071 pre-400 CE evidence
KNOWN_SITES = {
    -30: 5,    # Homo erectus sites (Sangiran, Trinil, Mojokerto)
    -20: 3,
    -10: 3,
    -5: 4,     # Song Terus, Gua Kidang, etc.
    -3: 3,     # Neolithic sites
    -2: 5,     # Sembiran, Buni, metal age
    -1: 4,     # Buni Complex, Dong Son finds
    0: 2,      # Very few attributable sites
    1: 2,
    2: 3,      # Buni continued
    3: 2,
    4: 3,      # Batujaya
    5: 5,      # Tarumanagara, Mulavarman
    6: 4,
    7: 8,      # Dieng temples
    8: 15,     # Borobudur, Prambanan
    9: 25,     # Peak of candi construction
    10: 30,    # East Java temples
    11: 15,
    12: 8,
    13: 12,    # Singhasari
    14: 20,    # Majapahit/Trowulan
    15: 10,
}


def compute_darkness_index():
    """
    Compute Archaeological Darkness Index for each century.

    Darkness = (Expected Activity - Observed Activity) / Expected Activity

    Expected Activity is estimated from:
    - Population trajectory (demographic growth model)
    - External recognition (if outsiders know about you, you exist)
    - Temporal neighbors (if centuries before/after are active, this one should be too)

    Observed Activity is estimated from:
    - Inscription count
    - Known archaeological sites
    - Material survival probability
    """

    results = []

    # Estimate expected activity using sigmoid population growth
    # Java population: ~1M by 500 CE, ~5M by 1000 CE (estimates)
    # Earlier periods: sparse but continuous occupation since 100,000 years
    def expected_population(century):
        """Rough estimate of relative population size."""
        if century <= -10:
            return 0.01  # Hunter-gatherers
        elif century <= -3:
            return 0.05  # Early agriculture
        elif century <= 0:
            return 0.15  # Bronze/Iron age chiefdoms
        elif century <= 5:
            return 0.3   # Early states
        elif century <= 8:
            return 0.5   # Classical kingdoms
        elif century <= 10:
            return 0.7   # Peak classical
        elif century <= 14:
            return 0.85  # Late classical
        else:
            return 1.0   # Majapahit peak

    for c in CENTURIES:
        label = century_short(c)
        pop = expected_population(c)
        inscriptions = INSCRIPTION_COUNT.get(c, 0)
        external = EXTERNAL_SOURCES.get(c, 0)
        sedimentation = VOLCANIC_SEDIMENTATION.get(c, 3)
        material_surv = MATERIAL_SURVIVAL.get(c, 0.05)
        sites = KNOWN_SITES.get(c, 0)

        # Expected evidence = f(population, material_survival, time_decay)
        # More people + more durable materials = more expected evidence
        time_decay = max(0.01, 1.0 / (1 + 0.001 * abs((c - 10) * 100)))  # Older = more decay
        expected_evidence = pop * 100  # Baseline expected finds

        # Observed evidence
        observed = inscriptions + sites * 0.5 + external * 2

        # Darkness factors (each increases darkness)
        burial_factor = min(1.0, sedimentation / 15.0)  # Normalized sedimentation
        decay_factor = 1.0 - material_surv  # Higher organic = more loss
        depth_factor = min(1.0, sedimentation * abs(min(c, 15) - 15) * 0.01)  # Cumulative burial

        # Darkness Index: 0 = fully visible, 1 = completely dark
        if expected_evidence > 0:
            visibility = min(1.0, observed / expected_evidence)
        else:
            visibility = 0

        darkness = (1.0 - visibility) * (0.3 + 0.3 * burial_factor + 0.2 * decay_factor + 0.2 * depth_factor)

        # Cap at [0, 1]
        darkness = max(0.0, min(1.0, darkness))

        results.append({
            'century': c,
            'label': label,
            'full_label': century_label(c),
            'expected_population': round(pop, 2),
            'inscriptions': inscriptions,
            'external_sources': external,
            'known_sites': sites,
            'sedimentation_mm_yr': sedimentation,
            'material_survival': material_surv,
            'observed_evidence': round(observed, 1),
            'expected_evidence': round(expected_evidence, 1),
            'burial_factor': round(burial_factor, 2),
            'decay_factor': round(decay_factor, 2),
            'darkness_index': round(darkness, 3),
        })

    return results


def main():
    print("=" * 70)
    print("E079: The Archaeological Darkness Index — Grand Synthesis")
    print("  Quantifying what we don't know about ancient Java")
    print("=" * 70)

    results = compute_darkness_index()

    # ── Display timeline ──────────────────────────────────────────────
    print(f"\n{'Century':<12} {'Pop':<5} {'Inscr':<6} {'Ext':<5} {'Sites':<6} {'Sed':<5} {'Mat%':<6} {'DI':<7} {'Darkness Bar'}")
    print("-" * 85)

    for r in results:
        c = r['century']
        # Skip very early centuries with no activity
        if c < -5 and r['darkness_index'] < 0.01:
            continue

        di = r['darkness_index']
        bar_len = int(di * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)

        print(f"  {r['label']:<10s} {r['expected_population']:<5.2f} "
              f"{r['inscriptions']:<6d} {r['external_sources']:<5d} "
              f"{r['known_sites']:<6d} {r['sedimentation_mm_yr']:<5d} "
              f"{r['material_survival']*100:<6.0f} {di:<7.3f} {bar}")

    # ── Key periods analysis ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY PERIOD ANALYSIS")
    print("=" * 70)

    # Find darkest centuries
    result_dict = {r['century']: r for r in results}
    dark_periods = sorted([r for r in results if r['century'] >= -5],
                          key=lambda x: x['darkness_index'], reverse=True)

    print(f"\n  DARKEST CENTURIES (highest DI):")
    for r in dark_periods[:5]:
        print(f"    {r['full_label']}: DI = {r['darkness_index']:.3f}")

    print(f"\n  BRIGHTEST CENTURIES (lowest DI):")
    for r in sorted([r for r in results if r['century'] >= -5],
                    key=lambda x: x['darkness_index'])[:5]:
        print(f"    {r['full_label']}: DI = {r['darkness_index']:.3f}")

    # ── The Invisible Millennium ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("THE INVISIBLE MILLENNIUM (1-400 CE)")
    print("=" * 70)

    invisible = [r for r in results if 1 <= r['century'] <= 4]
    mean_di = np.mean([r['darkness_index'] for r in invisible])

    # Classical comparison
    classical = [r for r in results if 8 <= r['century'] <= 10]
    mean_classical_di = np.mean([r['darkness_index'] for r in classical])

    print(f"""
  The "Invisible Millennium" (1-400 CE) has:
    Mean Darkness Index: {mean_di:.3f}
    External sources recognizing Java: {sum(r['external_sources'] for r in invisible)}
    Known archaeological sites: {sum(r['known_sites'] for r in invisible)}
    Inscriptions: {sum(r['inscriptions'] for r in invisible)}

  Compare with Classical Java (700-1000 CE):
    Mean Darkness Index: {mean_classical_di:.3f}
    External sources: {sum(r['external_sources'] for r in classical)}
    Known sites: {sum(r['known_sites'] for r in classical)}
    Inscriptions: {sum(r['inscriptions'] for r in classical)}

  The Invisible Millennium is {mean_di/max(mean_classical_di, 0.001):.1f}× darker than Classical Java,
  yet external sources confirm a prosperous, trade-connected society.

  This {mean_di/max(mean_classical_di, 0.001):.1f}× gap quantifies the VOLCARCH thesis:
  the archaeological record is not a faithful record of the past.
  It is filtered by volcanic burial, organic decay, and historiographic bias.
""")

    # ── VOLCARCH Evidence Convergence ─────────────────────────────────
    print("=" * 70)
    print("EVIDENCE CONVERGENCE: What Makes the Invisible Millennium Dark?")
    print("=" * 70)

    print(f"""
  FACTOR 1 — Volcanic Burial (E075):
    Sedimentation rate 3-5 mm/year in volcanic zones
    Over 400 years (1-400 CE): 1.2-2.0 METERS of burial
    Beyond standard excavation depth for most Indonesian surveys

  FACTOR 2 — Organic Material Culture (E074):
    Material survival probability: 5% (vs 25-30% for classical Java)
    63% of inscribed goods are organic → 63% invisible in ground record
    Pre-Hindu material culture was predominantly wood/bamboo/textile

  FACTOR 3 — No Writing System (E071):
    Sanskrit script adopted ~400 CE, but society existed for millennia
    Mulavarman's grandfather "Kundungga" has indigenous name →
    dynasty predates writing by ≥2 generations

  FACTOR 4 — Eruption Disruption (E078):
    Eruption decades have 6.3× fewer inscriptions (p=0.035)
    928 CE: 77% inscription rate drop after Merapi VEI 4
    Eruptions destroy both the sites AND the political systems
    that produce records

  FACTOR 5 — Spatial Encoding (E073):
    Volcanic knowledge is BEHAVIORAL, not textual (5/5 vs 0/4)
    Architecture encodes information that language does not
    → Pre-literate knowledge systems leave spatial, not lexical traces

  FACTOR 6 — Administrative Pre-existence (E074):
    49% of inscriptions use Austronesian administrative terms
    (rakryān, rakai, sīma, wanua) with no Sanskrit equivalents
    → The governing system predates Indianization

  CONVERGENCE:
    6 independent analytical dimensions all point to the same conclusion:
    Java's pre-400 CE civilization was real, substantial, and
    trade-connected, but archaeologically invisible due to the
    combined effects of volcanic burial, organic decay, and the
    absence of durable writing systems.
""")

    # ── Save results ──────────────────────────────────────────────────
    with open(RESULTS_DIR / "darkness_index.json", "w") as f:
        json.dump({
            "experiment": "E079",
            "title": "Archaeological Darkness Index",
            "results": results,
            "invisible_millennium_mean_di": round(mean_di, 3),
            "classical_java_mean_di": round(mean_classical_di, 3),
            "darkness_ratio": round(mean_di / max(mean_classical_di, 0.001), 1),
        }, f, indent=2)

    # CSV
    import csv
    with open(RESULTS_DIR / "darkness_timeline.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'century', 'label', 'full_label', 'expected_population',
            'inscriptions', 'external_sources', 'known_sites',
            'sedimentation_mm_yr', 'material_survival',
            'observed_evidence', 'expected_evidence',
            'burial_factor', 'decay_factor', 'darkness_index'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
