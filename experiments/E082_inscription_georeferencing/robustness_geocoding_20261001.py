"""Independent check (2026-10-01) of the verifier's claim that the candi-vs-inscription contrast (P17 core,
P11 pillar 2) is driven by geocoding artefacts in E082's canonical inscription file."""
import csv
import math
import statistics
from collections import Counter

from scipy.stats import mannwhitneyu

R = "D:/documents/volcarch-repo"
volc = [(float(v["lat"]), float(v["lon"])) for v in
        csv.DictReader(open(f"{R}/data/processed/dashboard/volcanoes_java_full.csv", encoding="utf-8"))]


def hav(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(x))


def dmin(lat, lon):
    return min(hav(lat, lon, a, b) for a, b in volc)


ins = list(csv.DictReader(open(
    f"{R}/experiments/E082_inscription_georeferencing/results/canonical30/geocoded_inscriptions_canonical30.csv",
    encoding="utf-8")))
print("inscription rows:", len(ins))
print("geocode_method:", Counter(r["geocode_method"] for r in ins).most_common())
print("confidence:", Counter(r["confidence"] for r in ins).most_common())
relief = [r for r in ins if "relief" in r["title"].lower() or "borobudur" in r["title"].lower()]
print("titles mentioning relief/Borobudur:", len(relief), "| distinct coords among them:",
      len({(r["lat"], r["lon"]) for r in relief}))
coords = Counter((r["lat"], r["lon"]) for r in ins)
top5 = coords.most_common(5)
print("distinct coordinates:", len(coords), "| share on top-5 points:",
      round(100 * sum(c for _, c in top5) / len(ins), 1), "%")
for (la, lo), c in top5:
    ex = next(r for r in ins if (r["lat"], r["lon"]) == (la, lo))
    print(f"   {c:3d} rows at ({la},{lo})  method={ex['geocode_method']} note={ex['geocode_note'][:60]!r}")

candi = [(float(r["lat"]), float(r["lon"])) for r in csv.DictReader(
    open(f"{R}/experiments/E031_candi_orientation/results/candi_volcano_pairs.csv", encoding="utf-8"))]
candi_u = sorted(set((round(a, 6), round(b, 6)) for a, b in candi))


def test(label, inscr, cand):
    di = [dmin(float(r["lat"]), float(r["lon"])) for r in inscr]
    dc = [dmin(a, b) for a, b in cand]
    u, p = mannwhitneyu(dc, di, alternative="two-sided")
    print(f"{label:<62} n_ins={len(di):<4} n_candi={len(dc):<4} med candi {statistics.median(dc):5.1f} | "
          f"med ins {statistics.median(di):5.1f} | gap {statistics.median(di) - statistics.median(dc):5.1f} km | p={p:.2g}")


test("as published (175 inscriptions, 142 candi rows)", ins, candi)
no_relief = [r for r in ins if r not in relief]
test("minus relief/Borobudur captions", no_relief, candi)
# 'precise' = geocode methods that are not region/province placeholders, as recorded in the file itself
placeholder_methods = {m for m, _ in Counter(r["geocode_method"] for r in ins).items()
                       if any(k in m.lower() for k in ("region", "province", "kingdom", "default", "centroid"))}
print("methods treated as placeholders:", placeholder_methods)
precise = [r for r in no_relief if r["geocode_method"] not in placeholder_methods]
test("minus captions and placeholder-geocoded rows", precise, candi)
test("minus captions and placeholders, de-duplicated candi", precise, candi_u)
ej = [r for r in precise if float(r["lon"]) >= 111.0]
test("East Java only (lon >= 111) precise inscriptions, de-dup candi", ej, candi_u)


# --- placeholder-aware filter (added 2026-10-01): geocode_method marks everything 'known_location',
# so region-level placeholders are identified from geocode_note instead.
REGION = {"East Java", "Mataram Central Java", "Central Java", "West Java", "Java", "Bali", "Sumatra"}
precise2 = [r for r in no_relief if r["geocode_note"].strip() not in REGION]
print()
print("region placeholders dropped:", len(no_relief) - len(precise2), "| precise findspots kept:", len(precise2))
test("precise findspots (no captions, no region placeholders), 142 candi", precise2, candi)
test("precise findspots, de-duplicated candi (103)", precise2, candi_u)
test("precise findspots, East Java only (lon>=111), de-dup candi", [r for r in precise2 if float(r["lon"]) >= 111.0], candi_u)
