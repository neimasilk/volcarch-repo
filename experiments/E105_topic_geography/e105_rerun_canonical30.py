"""
E105 CANONICAL-30 RE-RUN: Topic x Zone contingency (929 CE discontinuity)
=========================================================================
WS-E / SIG G1 re-derivation (2026-08-13, P11->SPAFA; G9 finding #3).

E105's zone table (57% court pre-929 / 91% Sanskrit-dominant / 53% periphery
post-929 / 89% mixed-indigenous) was computed 2026-03-17, before the
volcanoes.csv defect was found — its zone classification depended on the
incomplete volcano inventory. This re-runs the same classification with
canonical 30-volcano distances (E082 canonical-30 CSV).

Classification rules (identical to E105 README):
- Topic by pre-Indic ratio: Sanskrit-dominant (<0.05), Mixed (0.05-0.20),
  Indigenous-rich (>0.20).
- Zones by distance to nearest canonical volcano:
  Volcano <15 km, Court 15-30 km, Periphery >30 km.
- Era split at 929 CE.
"""
import json
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
E062 = REPO / "experiments" / "E062_temporal_synthesis" / "results" / "joined_dated_inscriptions.csv"
E082 = REPO / "experiments" / "E082_inscription_georeferencing" / "results" / "canonical30" / "geocoded_inscriptions_canonical30.csv"
OUT = Path(__file__).parent / "results" / "e105_results_canonical30.json"

# Load E062: per-inscription pre-Indic ratio
ratios = {}
with open(E062, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        fn = row["filename"].strip()
        try:
            year = float(row["year_ce"]) if row["year_ce"] not in ("", "nan") else None
        except ValueError:
            year = None
        try:
            ratio = float(row["pre_indic_ratio"])
        except ValueError:
            ratio = None
        ratios[fn] = {"year_ce": year, "pre_indic_ratio": ratio}

# Load E082 canonical-30: per-inscription canonical volcano distance
dists = {}
with open(E082, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        fn = row["filename"].strip()
        try:
            d = float(row["volcano_dist_km_c30"])
        except (ValueError, TypeError):
            d = None
        dists[fn] = d

# Join
def topic_of(r):
    if r is None:
        return None
    if r < 0.05:
        return "Sanskrit-dominant"
    if r <= 0.20:
        return "Mixed"
    return "Indigenous-rich"

def zone_of(d):
    if d is None:
        return None
    if d < 15:
        return "Volcano"
    if d <= 30:
        return "Court"
    return "Periphery"

rows = []
for fn, e62 in ratios.items():
    d = dists.get(fn)
    z = zone_of(d)
    t = topic_of(e62["pre_indic_ratio"])
    y = e62["year_ce"]
    if z is None or t is None or y is None:
        continue
    rows.append({"filename": fn, "year_ce": y, "topic": t, "zone": z})

n = len(rows)
print(f"Joined dated+geocoded+ratio inscriptions: {n}")

pre = [r for r in rows if r["year_ce"] < 929]
post = [r for r in rows if r["year_ce"] >= 929]
print(f"Pre-929: {len(pre)}, Post-929: {len(post)}")


def zone_table(subset):
    t = {"Volcano": {}, "Court": {}, "Periphery": {}}
    for r in subset:
        t[r["zone"]][r["topic"]] = t[r["zone"]].get(r["topic"], 0) + 1
    return t


pre_tab = zone_table(pre)
post_tab = zone_table(post)

pre_court_n = sum(pre_tab["Court"].values())
pre_court_sans = pre_tab["Court"].get("Sanskrit-dominant", 0)
post_peri_n = sum(post_tab["Periphery"].values())
post_peri_mixind = post_tab["Periphery"].get("Mixed", 0) + post_tab["Periphery"].get("Indigenous-rich", 0)

results = {
    "experiment": "E105 topic x zone contingency — CANONICAL-30 RE-RUN",
    "date": "2026-08-13",
    "n_joined": n,
    "n_pre_929": len(pre),
    "n_post_929": len(post),
    "zone_table_pre_929": pre_tab,
    "zone_table_post_929": post_tab,
    "headline": {
        "pre_929_court_pct": round(100 * pre_court_n / len(pre), 1) if pre else None,
        "pre_929_court_sanskrit_pct": round(100 * pre_court_sans / pre_court_n, 1) if pre_court_n else None,
        "post_929_periphery_pct": round(100 * post_peri_n / len(post), 1) if post else None,
        "post_929_periphery_mixedindigenous_pct": round(100 * post_peri_mixind / post_peri_n, 1) if post_peri_n else None,
    },
    "baseline_20260317": {
        "pre_court": 57, "pre_court_sanskrit": 91, "post_periphery": 53, "post_periphery_mixind": 89,
    },
}
print(json.dumps(results["zone_table_pre_929"], indent=1))
print(json.dumps(results["zone_table_post_929"], indent=1))
print("HEADLINE:", json.dumps(results["headline"], indent=1))

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {OUT}")
