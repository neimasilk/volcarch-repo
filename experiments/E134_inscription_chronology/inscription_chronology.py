"""
E134: Inscription Chronology Gap Analysis
Distribution of 166 dated DHARMA inscriptions by century.
Identify "dark centuries" and test correlation with volcanic eruption frequency.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
REPO = Path(__file__).parent.parent.parent

# === LOAD DATA ===

df = pd.read_csv(REPO / "experiments/E030_prasasti_temporal_nlp/results/dated_inscriptions.csv")
df = df[df["year_ce"].notna()].copy()
df["century"] = df["year_ce"].apply(lambda y: int(y // 100) + 1 if y > 0 else int(y // 100))
print(f"Dated inscriptions: {len(df)}")

# === CENTURY DISTRIBUTION ===

print(f"\n{'=' * 70}")
print("INSCRIPTION COUNT BY CENTURY")
print("=" * 70)

century_counts = df["century"].value_counts().sort_index()
all_centuries = range(min(century_counts.index), max(century_counts.index) + 1)

print(f"\n  {'Century':>8} {'Count':>6} {'Bar'}")
print(f"  {'-'*8} {'-'*6} {'-'*40}")
for c in all_centuries:
    count = century_counts.get(c, 0)
    bar = "#" * count
    label = f"C{c}" if c > 0 else f"C{c}"
    highlight = " <<<" if count == 0 else " ***" if count >= 20 else ""
    print(f"  {label:>8} {count:>6} {bar}{highlight}")

# === DARK CENTURIES ===

print(f"\n{'=' * 70}")
print("DARK CENTURIES (zero or very few inscriptions)")
print("=" * 70)

for c in all_centuries:
    count = century_counts.get(c, 0)
    if count <= 2:
        approx_year = f"{(c-1)*100}-{c*100} CE" if c > 0 else f"{c*100}-{(c+1)*100} BCE"
        print(f"  C{c} ({approx_year}): {count} inscriptions")

# === PRE-INDIC RATIO BY CENTURY ===

print(f"\n{'=' * 70}")
print("PRE-INDIC VOCABULARY RATIO BY CENTURY")
print("=" * 70)

century_preindic = df.groupby("century")["pre_indic_ratio"].mean()
print(f"\n  {'Century':>8} {'N':>4} {'Pre-Indic Ratio':>15} {'Trend'}")
print(f"  {'-'*8} {'-'*4} {'-'*15} {'-'*20}")
for c in sorted(century_preindic.index):
    n = century_counts.get(c, 0)
    ratio = century_preindic[c]
    trend = "LOW (Indic dominant)" if ratio < 0.15 else "RISING" if ratio > 0.25 else "MODERATE"
    print(f"  C{c:>7} {n:>4} {ratio:>14.3f} {trend}")

# Test temporal trend
dated_cents = [(c, century_preindic[c]) for c in sorted(century_preindic.index) if century_counts.get(c, 0) >= 3]
if len(dated_cents) >= 5:
    x = [d[0] for d in dated_cents]
    y = [d[1] for d in dated_cents]
    rho, p = stats.spearmanr(x, y)
    print(f"\n  Spearman trend (pre-Indic ratio vs century): rho={rho:.3f}, p={p:.4f}")
    if rho > 0 and p < 0.05:
        print(f"  RISING TREND: Pre-Indic vocabulary INCREASES over time")
    elif rho < 0 and p < 0.05:
        print(f"  FALLING TREND: Pre-Indic vocabulary DECREASES over time")
    else:
        print(f"  No significant trend")

# === WORD COUNT TREND ===

print(f"\n{'=' * 70}")
print("INSCRIPTION LENGTH BY CENTURY")
print("=" * 70)

century_wordcount = df.groupby("century")["word_count"].agg(["mean", "median", "count"])
for c in sorted(century_wordcount.index):
    row = century_wordcount.loc[c]
    if row["count"] >= 3:
        print(f"  C{c:>3}: mean={row['mean']:.0f} words, median={row['median']:.0f}, n={row['count']:.0f}")

# === ERUPTION CORRELATION ===

print(f"\n{'=' * 70}")
print("INSCRIPTION COUNT vs VOLCANIC ERUPTIONS")
print("=" * 70)

# Major eruptions per century (from GVP data, Java + nearby)
eruptions_per_century = {
    7: 2,   # early historical
    8: 3,   # Merapi active
    9: 5,   # Merapi VEI 4-5
    10: 4,  # Merapi active
    11: 3,  # Merapi
    12: 2,  # reduced activity
    13: 3,  # Samalas 1257 (VEI 7)
    14: 4,  # Kelud active
    15: 2,  # post-Majapahit
}

common_centuries = [c for c in eruptions_per_century if c in century_counts.index]
if len(common_centuries) >= 5:
    x_erupt = [eruptions_per_century[c] for c in common_centuries]
    y_insc = [century_counts[c] for c in common_centuries]

    rho_e, p_e = stats.spearmanr(x_erupt, y_insc)
    print(f"\n  Spearman(eruptions, inscriptions): rho={rho_e:.3f}, p={p_e:.4f}")
    if rho_e < 0 and p_e < 0.1:
        print(f"  NEGATIVE correlation: More eruptions = fewer inscriptions (suggestive)")
    elif rho_e > 0:
        print(f"  POSITIVE correlation: More eruptions = more inscriptions (?)")
    else:
        print(f"  No significant correlation")

    print(f"\n  {'Century':>8} {'Eruptions':>10} {'Inscriptions':>13}")
    print(f"  {'-'*8} {'-'*10} {'-'*13}")
    for c in common_centuries:
        print(f"  C{c:>7} {eruptions_per_century[c]:>10} {century_counts[c]:>13}")

# === HYANG ANALYSIS ===

print(f"\n{'=' * 70}")
print("HYANG (pre-Indic divine concept) FREQUENCY BY CENTURY")
print("=" * 70)

century_hyang = df.groupby("century")["has_hyang"].mean()
for c in sorted(century_hyang.index):
    n = century_counts.get(c, 0)
    if n >= 3:
        pct = century_hyang[c] * 100
        bar = "#" * int(pct)
        print(f"  C{c:>3} (n={n:>3}): {pct:>5.1f}% {bar}")

# === SAVE ===

summary = {
    "experiment": "E134_inscription_chronology",
    "total_dated": len(df),
    "century_distribution": {f"C{c}": int(century_counts.get(c, 0)) for c in all_centuries},
    "dark_centuries": [f"C{c}" for c in all_centuries if century_counts.get(c, 0) <= 2],
    "peak_century": f"C{century_counts.idxmax()} ({century_counts.max()} inscriptions)",
    "pre_indic_trend": f"rho={rho:.3f}, p={p:.4f}" if len(dated_cents) >= 5 else "insufficient data",
}

with open(RESULTS_DIR / "inscription_chronology.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved to {RESULTS_DIR}/")
