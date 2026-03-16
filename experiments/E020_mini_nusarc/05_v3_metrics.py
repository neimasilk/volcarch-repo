"""
E020: Re-run Metrics 1 & 2 on Mini-NusaRC v3 (80 sites)
Tests H-TOM predictions with expanded dataset.
"""
import csv
import sys
import io
from collections import Counter
from pathlib import Path

# Fix Windows cp1252 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA = Path("experiments/E020_mini_nusarc/data/mini_nusarc_v3.csv")

# Volcanic vs non-volcanic classification
VOLCANIC_REGIONS = {"Java", "Sulawesi", "Nusa_Tenggara", "Sumatra", "Philippines", "Maluku"}
NON_VOLCANIC_REGIONS = {"Kalimantan", "Madagascar"}

def load():
    with open(DATA, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def metric1_cave_ratio(sites):
    """Metric 1: Cave/open-air ratio in volcanic vs non-volcanic regions."""
    print("=" * 60)
    print("METRIC 1: Cave/Open-Air Ratio")
    print("=" * 60)

    volcanic = [s for s in sites if s['region'] in VOLCANIC_REGIONS]
    non_volcanic = [s for s in sites if s['region'] in NON_VOLCANIC_REGIONS]

    def cave_stats(group, label):
        cave = sum(1 for s in group if s['site_type'] in ('cave', 'rockshelter'))
        open_air = sum(1 for s in group if s['site_type'] in ('open_air', 'river_terrace'))
        total = cave + open_air
        ratio = cave / total if total > 0 else 0
        print(f"  {label}: {cave} cave/shelter, {open_air} open-air, ratio = {ratio:.3f} ({cave}/{total})")
        return cave, open_air

    print("\nAll sites:")
    v_cave, v_open = cave_stats(volcanic, "Volcanic regions")
    nv_cave, nv_open = cave_stats(non_volcanic, "Non-volcanic regions")

    # Fisher's exact test
    try:
        from scipy.stats import fisher_exact
        table = [[v_cave, v_open], [nv_cave, nv_open]]
        odds, p = fisher_exact(table)
        print(f"\n  Fisher's exact: odds ratio = {odds:.3f}, p = {p:.4f}")
        if p < 0.05:
            print("  → SIGNIFICANT: volcanic regions differ in cave ratio")
        else:
            print("  → Not significant at α=0.05")
    except ImportError:
        print("  (scipy not available for Fisher's test)")

    # Deep-time only (>10,000 BP)
    print("\nDeep-time sites only (>10,000 BP):")
    deep_vol = [s for s in volcanic if int(s.get('date_bp', 0)) > 10000]
    deep_nv = [s for s in non_volcanic if int(s.get('date_bp', 0)) > 10000]
    dv_cave, dv_open = cave_stats(deep_vol, "Volcanic >10ka")
    dnv_cave, dnv_open = cave_stats(deep_nv, "Non-volcanic >10ka")

    try:
        from scipy.stats import fisher_exact
        table2 = [[dv_cave, dv_open], [dnv_cave, dnv_open]]
        odds2, p2 = fisher_exact(table2)
        print(f"\n  Fisher's exact (>10ka): odds ratio = {odds2:.3f}, p = {p2:.4f}")
    except ImportError:
        pass

    # Per region breakdown
    print("\nPer-region cave ratios:")
    for region in sorted(set(s['region'] for s in sites)):
        rs = [s for s in sites if s['region'] == region]
        cave = sum(1 for s in rs if s['site_type'] in ('cave', 'rockshelter'))
        total = len(rs)
        ratio = cave / total if total > 0 else 0
        volc = "VOLCANIC" if region in VOLCANIC_REGIONS else "control"
        print(f"  {region:20s}: {cave}/{total} = {ratio:.2f}  [{volc}]")


def metric2_density_dropoff(sites):
    """Metric 2: Site density per time bin — volcanic vs non-volcanic."""
    print("\n" + "=" * 60)
    print("METRIC 2: Site Density per Time Bin")
    print("=" * 60)

    bins = [(0, 5000), (5000, 10000), (10000, 50000), (50000, 200000), (200000, 2000000)]
    bin_labels = ["0-5ka", "5-10ka", "10-50ka", "50-200ka", "200ka+"]

    volcanic = [s for s in sites if s['region'] in VOLCANIC_REGIONS]
    non_volcanic = [s for s in sites if s['region'] in NON_VOLCANIC_REGIONS]

    print(f"\n{'Bin':>10s} | {'Volcanic':>10s} | {'Non-volcanic':>14s} | {'V ratio':>8s} | {'NV ratio':>8s}")
    print("-" * 60)

    v_total = len(volcanic)
    nv_total = len(non_volcanic)

    for (lo, hi), label in zip(bins, bin_labels):
        v_count = sum(1 for s in volcanic if lo <= int(s.get('date_bp', 0)) < hi)
        nv_count = sum(1 for s in non_volcanic if lo <= int(s.get('date_bp', 0)) < hi)
        v_ratio = v_count / v_total if v_total else 0
        nv_ratio = nv_count / nv_total if nv_total else 0
        print(f"{label:>10s} | {v_count:>10d} | {nv_count:>14d} | {v_ratio:>8.2f} | {nv_ratio:>8.2f}")

    # H-TOM prediction: volcanic regions should have steeper dropoff
    # (fewer sites in older bins relative to total)
    print("\nH-TOM prediction: volcanic regions should have proportionally")
    print("fewer sites in the deep-time bins (>10ka) due to burial/destruction.")


def metric3_java_distance_time(sites):
    """Bonus: For Java sites, check if older sites are exclusively far from volcanoes."""
    print("\n" + "=" * 60)
    print("BONUS: Java Site Age vs Site Type")
    print("=" * 60)

    java = [s for s in sites if s['region'] == 'Java']
    java_sorted = sorted(java, key=lambda s: -int(s.get('date_bp', 0)))

    print(f"\n{'Site':30s} | {'Age BP':>12s} | {'Type':15s} | {'Period':20s}")
    print("-" * 85)
    for s in java_sorted:
        age = int(s.get('date_bp', 0))
        print(f"{s['site_name'][:30]:30s} | {age:>12,d} | {s['site_type']:15s} | {s['cultural_period']:20s}")

    # Count by period
    cave_by_period = Counter()
    open_by_period = Counter()
    for s in java:
        period = s['cultural_period']
        if s['site_type'] in ('cave', 'rockshelter'):
            cave_by_period[period] += 1
        else:
            open_by_period[period] += 1

    print("\nJava site type by period:")
    all_periods = sorted(set(list(cave_by_period.keys()) + list(open_by_period.keys())))
    for p in all_periods:
        c = cave_by_period.get(p, 0)
        o = open_by_period.get(p, 0)
        total = c + o
        cave_pct = c / total * 100 if total else 0
        print(f"  {p:20s}: {c} cave, {o} open ({cave_pct:.0f}% cave)")


def main():
    sites = load()
    print(f"Mini-NusaRC v3: {len(sites)} sites loaded\n")
    metric1_cave_ratio(sites)
    metric2_density_dropoff(sites)
    metric3_java_distance_time(sites)

if __name__ == "__main__":
    main()
