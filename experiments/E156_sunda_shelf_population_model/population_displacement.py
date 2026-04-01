"""
E156: Sunda Shelf Population Displacement → Java Volcanic Interior
===================================================================
L2 (Coastal Submersion) has only 2/155 experiments. This experiment
models WHERE displaced Sunda Shelf populations would have gone as
sea levels rose, and whether their destination intersects with
Java's volcanic burial zones.

The hypothesis: populations displaced from the drowning Sunda Shelf
entered Java through river mouths and migrated inland — INTO volcanic
zones where their settlements were subsequently buried.

This is the L1×L2 intersection that VOLCARCH has never tested.
"""

import numpy as np
import json
from pathlib import Path

# ============================================================
# TIMELINE: Post-LGM sea level rise
# ============================================================
# From Lambeck et al. 2014, Hanebuth et al. 2000, Voris 2000
# Sea level at LGM (~21,000 BP) was ~120m below present
# Rapid rise during Meltwater Pulse 1A (14,600-14,300 BP): ~20m in 300 years

sea_level_timeline = [
    # (years_BP, sea_level_m, notes)
    (21000, -120, "Last Glacial Maximum"),
    (19000, -115, "Early deglaciation"),
    (17000, -100, "Progressive warming"),
    (16000, -90, "Approaching MWP-1A"),
    (14600, -80, "Start of Meltwater Pulse 1A"),
    (14300, -60, "End of MWP-1A (20m in 300 years)"),
    (13000, -55, "Bølling-Allerød interstadial"),
    (12000, -50, "Younger Dryas onset"),
    (11600, -50, "Younger Dryas (stable)"),
    (10000, -40, "Early Holocene"),
    (8000, -15, "Mid-Holocene transgression"),
    (7000, -5, "Near-modern levels"),
    (6000, 0, "Holocene optimum (slight highstand +2m)"),
    (4000, 0, "Modern levels reached"),
]

print("=" * 70)
print("E156: SUNDA SHELF POPULATION DISPLACEMENT MODEL")
print("=" * 70)

# ============================================================
# SUNDA SHELF AREA VS SEA LEVEL
# ============================================================
# From E052: total exposed area at LGM = 2,089,415 km²
# Shelf bathymetry is relatively flat (E052: 81.5% flat + near river)
# Area exposed is roughly proportional to depth (simplified)

# Modern Sunda Shelf (0 to -120m) — approximate by contour
# Using simplified area-depth relationship from Voris 2000
area_by_depth = {
    -120: 2089415,  # km² exposed at LGM (from E052)
    -100: 1800000,
    -80: 1400000,
    -60: 1000000,
    -50: 800000,
    -40: 500000,
    -15: 100000,
    -5: 20000,
    0: 0,  # fully submerged
}

print(f"\n--- SUNDA SHELF AREA BY SEA LEVEL ---")
print(f"{'Sea Level (m)':<15} {'Exposed Area (km²)':<20} {'% of LGM':<10}")
for depth, area in sorted(area_by_depth.items()):
    pct = area / 2089415 * 100
    print(f"{depth:>10}m    {area:>15,}    {pct:>7.1f}%")

# ============================================================
# POPULATION MODEL
# ============================================================
# Population density estimates for hunter-gatherer/early Neolithic Sunda
# Conservative: 0.1 people/km² (hunter-gatherer, arid savanna)
# Moderate: 0.3 people/km² (hunter-gatherer, tropical forest edge)
# High: 1.0 people/km² (early Neolithic, riverine)

densities = {
    "hunter_gatherer_low": 0.1,
    "hunter_gatherer_moderate": 0.3,
    "early_neolithic": 1.0,
}

print(f"\n--- POPULATION ON SUNDA SHELF BY ERA ---")
print(f"{'Era (BP)':<12} {'Sea Level':<10} {'Area (km²)':<15} {'Pop (low)':<12} {'Pop (mod)':<12} {'Pop (high)':<12}")
print(f"{'-'*75}")

displacement_data = []

for year, sl, notes in sea_level_timeline:
    # Interpolate area from depth
    depths = sorted(area_by_depth.keys())
    area = 0
    for i in range(len(depths) - 1):
        if depths[i] <= sl <= depths[i + 1]:
            # Linear interpolation
            frac = (sl - depths[i]) / (depths[i + 1] - depths[i])
            area = area_by_depth[depths[i]] + frac * (area_by_depth[depths[i + 1]] - area_by_depth[depths[i]])
            break
    if sl <= depths[0]:
        area = area_by_depth[depths[0]]
    elif sl >= depths[-1]:
        area = area_by_depth[depths[-1]]

    pop_low = area * densities["hunter_gatherer_low"]
    pop_mod = area * densities["hunter_gatherer_moderate"]
    pop_high = area * densities["early_neolithic"]

    print(f"{year:>8} BP  {sl:>6}m  {area:>12,.0f}  {pop_low:>10,.0f}  {pop_mod:>10,.0f}  {pop_high:>10,.0f}")

    displacement_data.append({
        "year_bp": year,
        "sea_level_m": sl,
        "area_km2": area,
        "pop_low": pop_low,
        "pop_mod": pop_mod,
        "pop_high": pop_high,
        "notes": notes,
    })

# ============================================================
# DISPLACEMENT RATE CALCULATION
# ============================================================
print(f"\n{'='*70}")
print(f"DISPLACEMENT RATES (people forced to move per century)")
print(f"{'='*70}")

print(f"\n{'Period':<35} {'Area Lost (km²)':<18} {'Pop Displaced (mod)':<20} {'Rate (/century)':<15}")
print(f"{'-'*90}")

for i in range(1, len(displacement_data)):
    prev = displacement_data[i - 1]
    curr = displacement_data[i]
    area_lost = prev["area_km2"] - curr["area_km2"]
    pop_displaced = prev["pop_mod"] - curr["pop_mod"]
    duration_centuries = (prev["year_bp"] - curr["year_bp"]) / 100
    rate = pop_displaced / duration_centuries if duration_centuries > 0 else 0

    period = f"{prev['year_bp']:,}-{curr['year_bp']:,} BP"
    if area_lost > 0:
        print(f"{period:<35} {area_lost:>14,.0f}  {pop_displaced:>16,.0f}  {rate:>12,.0f}")

# ============================================================
# WHERE DO DISPLACED PEOPLE GO?
# ============================================================
print(f"\n{'='*70}")
print(f"DESTINATION ANALYSIS: Where did displaced Sunda populations go?")
print(f"{'='*70}")

# Java's north coast has multiple large river systems that were
# connected to Sunda Shelf paleo-drainages (E052, E148)
# Major entry points into Java from the Sunda Shelf:

entry_points = [
    {
        "name": "Solo River System (Bengawan Solo)",
        "modern_mouth": "Surabaya/Gresik",
        "paleo_connection": "Connected to N. Sunda River via Java Sea",
        "inland_reach_km": 540,
        "volcanic_zones_reached": ["Kelud (km 280)", "Arjuno-Welirang (km 200)", "Lawu (km 350)"],
        "known_archaeology": "Sangiran (Homo erectus), Trinil, Ngandong — ALL along Solo River",
        "volcarch_zone": "Primary Zone B/C (E080 targets cluster here)",
    },
    {
        "name": "Brantas River System",
        "modern_mouth": "Surabaya (delta)",
        "paleo_connection": "Connected to Solo system at LGM (shared delta)",
        "inland_reach_km": 320,
        "volcanic_zones_reached": ["Kelud (km 60)", "Arjuno-Welirang (km 80)", "Semeru (km 100)"],
        "known_archaeology": "Trowulan (Majapahit), Singosari — deep volcanic burial documented",
        "volcarch_zone": "Core Zone B/C — Dwarapala Singosari calibration point",
    },
    {
        "name": "Progo/Opak River System (Central Java)",
        "modern_mouth": "South coast Yogyakarta",
        "paleo_connection": "Short rivers, but south coast exposed additional shelf",
        "inland_reach_km": 80,
        "volcanic_zones_reached": ["Merapi (km 30)", "Sundoro (km 80)"],
        "known_archaeology": "Sambisari (5m buried), Kedulan (7m buried), Liangan (9m buried)",
        "volcarch_zone": "Merapi system — highest burial rates (4.8 mm/yr mean)",
    },
    {
        "name": "Citarum River System (West Java)",
        "modern_mouth": "North coast Karawang",
        "paleo_connection": "Connected to N. Sunda River system directly",
        "inland_reach_km": 250,
        "volcanic_zones_reached": ["Tangkuban Parahu (km 100)", "Papandayan (km 180)"],
        "known_archaeology": "Batujaya (pre-Hindu, NON-volcanic coast) — West Java smoking gun",
        "volcarch_zone": "Control zone — pre-Hindu sites EXIST in non-volcanic coast",
    },
]

for ep in entry_points:
    print(f"\n  {ep['name']}")
    print(f"    Modern mouth: {ep['modern_mouth']}")
    print(f"    Paleo connection: {ep['paleo_connection']}")
    print(f"    Inland reach: {ep['inland_reach_km']} km")
    print(f"    Volcanic zones reached: {', '.join(ep['volcanic_zones_reached'])}")
    print(f"    Known archaeology: {ep['known_archaeology']}")
    print(f"    VOLCARCH zone: {ep['volcarch_zone']}")

# ============================================================
# THE L1×L2 INTERSECTION MODEL
# ============================================================
print(f"\n{'='*70}")
print(f"THE L1×L2 INTERSECTION: Population pushed FROM shelf INTO volcanic zones")
print(f"{'='*70}")

# Model: as sea level rises, populations move inland along river systems.
# The further inland they go, the closer they get to volcanic zones.
# Their settlements are then buried by volcanic deposition.

# Java's geography: volcanoes run along the spine (south-center).
# Rivers flow NORTH to Java Sea (Solo, Brantas) or SOUTH to Indian Ocean.
# Sunda Shelf is to the NORTH.
# Therefore: displaced populations from the north coast move SOUTH along rivers
# → toward the volcanic spine.

# Key question: What fraction of Java's habitable area is in volcanic zones?
# From E075: 32.3% of cells have >1m burial depth
# From E013: model predicts high suitability in areas WITH volcanic burial

java_area_km2 = 129000
volcanic_zone_fraction = 0.323  # from E075
volcanic_zone_area = java_area_km2 * volcanic_zone_fraction

# Fraction of river corridors that pass through volcanic zones
# Solo River: ~50% of its length is in volcanic proximity
# Brantas: ~70% (originates near Kelud)
# Progo/Opak: ~90% (short, near Merapi)
# Citarum: ~40% (longer, passes Tangkuban Parahu zone)
river_volcanic_fraction = 0.60  # weighted average

# Population that ends up in volcanic zones
# Assumption: populations follow rivers → proportion in volcanic zones
# proportional to river corridor overlap with volcanic zones

total_displaced_mod = sum(
    displacement_data[i - 1]["pop_mod"] - displacement_data[i]["pop_mod"]
    for i in range(1, len(displacement_data))
    if displacement_data[i - 1]["pop_mod"] - displacement_data[i]["pop_mod"] > 0
)

# Some go to Java, some to Sumatra, Borneo, etc.
# Java receives ~20-30% (proportional to coastline facing Sunda Shelf)
java_fraction = 0.25
pop_to_java = total_displaced_mod * java_fraction
pop_to_volcanic = pop_to_java * river_volcanic_fraction

print(f"\n  Total displaced from Sunda (moderate density): {total_displaced_mod:,.0f}")
print(f"  Estimated fraction reaching Java: {java_fraction*100:.0f}%")
print(f"  Population reaching Java: {pop_to_java:,.0f}")
print(f"  Of which, entering volcanic zones: {pop_to_volcanic:,.0f} ({river_volcanic_fraction*100:.0f}%)")
print(f"  Cumulative over 15,000 years of sea-level rise")

# Burial model
burial_rate_mm_yr = 4.4  # from L1 calibration
years_since_arrival = 10000  # average arrival ~10,000 BP
burial_depth_m = burial_rate_mm_yr * years_since_arrival / 1000

print(f"\n  Mean burial rate: {burial_rate_mm_yr} mm/yr")
print(f"  Time since average arrival: {years_since_arrival:,} years")
print(f"  Estimated burial depth: {burial_depth_m:.1f} m")
print(f"  Detection by surface survey: IMPOSSIBLE (surface reaches ~1900 CE only, E117)")

# ============================================================
# SETTLEMENT DENSITY IN VOLCANIC ZONES
# ============================================================
print(f"\n{'='*70}")
print(f"SETTLEMENT DENSITY PREDICTION")
print(f"{'='*70}")

# If pop_to_volcanic people lived in volcanic zones at various times
# and their settlements accumulated over millennia:

# Conservative: 1 settlement per 100 people
# Moderate: 1 settlement per 50 people
# Dense: 1 settlement per 20 people

settlement_estimates = {
    "conservative (1/100)": pop_to_volcanic / 100,
    "moderate (1/50)": pop_to_volcanic / 50,
    "dense (1/20)": pop_to_volcanic / 20,
}

for label, count in settlement_estimates.items():
    density = count / volcanic_zone_area
    print(f"  {label}: {count:,.0f} settlements ({density:.2f}/km²)")

# Compare to E109's estimate
e109_hidden = 824  # from E109 forward simulation
print(f"\n  E109 forward simulation estimate: {e109_hidden} hidden sites")
print(f"  This model (moderate): {settlement_estimates['moderate (1/50)']:,.0f} settlements")
print(f"  Ratio: {settlement_estimates['moderate (1/50)'] / e109_hidden:.1f}×")

# ============================================================
# THE DOUBLE ERASURE
# ============================================================
print(f"\n{'='*70}")
print(f"THE DOUBLE ERASURE: L1 × L2 compound taphonomy")
print(f"{'='*70}")

print("""
The Sunda Shelf drowning created a DOUBLE erasure:

1. COASTAL SITES: Settlements on the shelf itself are now
   underwater at -10 to -120m depth. These are L2 losses.
   Estimated: 2.09M km² × 0.3 people/km² = ~630,000 people's
   settlements are under the Java Sea.

2. DISPLACED INLAND SITES: People who fled rising seas
   moved UP river systems INTO volcanic zones, where their
   settlements were buried by volcanic deposition. These are
   L1 losses triggered by L2.

The standard narrative says: "People arrived in Java ~400 CE with Indian culture."
This model says: "People have been in Java for 40,000+ years. The Sunda Shelf
drowning pushed additional populations into volcanic zones 20,000-6,000 BP.
Their coastal ancestors are underwater. Their inland descendants are buried
under 20-40 meters of volcanic sediment."

Neither the underwater sites (L2) nor the buried sites (L1) are visible
to conventional archaeological survey. The ONLY visible sites are in the
narrow non-volcanic, non-submerged zone — which is exactly where the
known pre-Hindu sites are (Buni Complex, Batujaya = north coast,
non-volcanic West Java).

This is not a coincidence. It's a PREDICTION of the L1×L2 model.
""")

# ============================================================
# TESTABLE PREDICTIONS
# ============================================================
print(f"{'='*70}")
print(f"TESTABLE PREDICTIONS FROM L1×L2 MODEL")
print(f"{'='*70}")

predictions = [
    "1. River mouth sites (Solo, Brantas deltas) should show continuous occupation from Pleistocene through Holocene — if they exist above current sea level.",
    "2. Inland sites along Solo River (Sangiran corridor) should have pre-Neolithic through Neolithic cultural layers at 5-20m depth.",
    "3. Batujaya/Buni (non-volcanic coast) should have the LONGEST continuous sequence of any Java site — they escaped both L1 and L2.",
    "4. Sonar/sub-bottom profiling of the Java Sea floor near Solo River paleo-mouth should reveal submerged settlement features at -40 to -80m depth.",
    "5. Phytolith analysis of deep volcanic cores near Brantas River should show agricultural signatures (rice, millet) at pre-400 CE depths.",
]

for p in predictions:
    print(f"  {p}")

# Save results
output_path = Path("D:/documents/volcarch-repo/experiments/E156_sunda_shelf_population_model/results")
output_path.mkdir(exist_ok=True)

results = {
    "timeline": displacement_data,
    "entry_points": [ep["name"] for ep in entry_points],
    "total_displaced_moderate": total_displaced_mod,
    "pop_to_java": pop_to_java,
    "pop_to_volcanic_zones": pop_to_volcanic,
    "estimated_burial_depth_m": burial_depth_m,
    "settlement_estimates": {k: v for k, v in settlement_estimates.items()},
    "double_erasure": "L1 (volcanic burial) × L2 (coastal submersion) compound taphonomy",
}

with open(output_path / "displacement_model.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {output_path / 'displacement_model.json'}")
print(f"\nDONE.")
