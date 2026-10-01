"""E153 robustness check (2026-10-01, re-entry audit, ledger C029).

The 'non-temple archaeological sites' in E153 come from data/processed/east_java_sites.geojson
(OSM/Wikipedia, 662 of 666 with period 'unknown'). Named modern features (Tugu, Monumen, Patung,
museums, heroes' cemeteries, etc.) can sit in that set. This script re-computes E153 Test 1 (distance
from each non-temple site to the nearest of the 142 E031 candi, vs a random-point Monte Carlo null in the
same bounding box) after removing them, so P11's 'candi-settlement proximity' support can be judged.
It reproduces E153's own classification rule first, so the baseline can be compared with its README.
"""
import csv
import json
import math
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
random.seed(153)

MODERN = re.compile(
    r"\b(tugu|monumen|monument|patung|museum|makam pahlawan|taman makam|memorial|stasiun|gedung|"
    r"jembatan|alun|lapangan|kantor|sekolah|masjid|gereja|klenteng|vihara|benteng|pabrik|"
    r"gelanggang|stadion|taman)\b", re.I)


def hav(a, b, c, d):
    r = 6371.0
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = p2 - p1, math.radians(d - b)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(x))


candi = []
with open(REPO / "experiments/E031_candi_orientation/results/candi_volcano_pairs.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lat = float(row.get("candi_lat") or row.get("lat"))
        lon = float(row.get("candi_lon") or row.get("lon"))
        candi.append((lat, lon))

feats = json.load(open(REPO / "data/processed/east_java_sites.geojson", encoding="utf-8"))["features"]
sites = []
for ft in feats:
    g = ft.get("geometry") or {}
    if not g.get("coordinates"):
        continue
    p = ft["properties"]
    sites.append({"name": str(p.get("name", "")), "type": p.get("type", "unknown"),
                  "lat": g["coordinates"][1], "lon": g["coordinates"][0]})

# E153's own rule
non_temple = [s for s in sites if not (any(k in s["name"].lower() for k in ("candi", "temple", "pura"))
                                       or s["type"] in {"monument", "kuil"})]
bbox = (min(s["lat"] for s in sites), max(s["lat"] for s in sites),
        min(s["lon"] for s in sites), max(s["lon"] for s in sites))


def nearest(lat, lon):
    return min(hav(lat, lon, c[0], c[1]) for c in candi)


def describe(label, subset, n_sim=2000):
    d = sorted(nearest(s["lat"], s["lon"]) for s in subset)
    n = len(d)
    mean = sum(d) / n
    med = d[n // 2]
    w10 = sum(1 for x in d if x < 10) / n
    sims = []
    for _ in range(n_sim):
        tot = 0.0
        for _ in range(n):
            tot += nearest(random.uniform(bbox[0], bbox[1]), random.uniform(bbox[2], bbox[3]))
        sims.append(tot / n)
    p = (1 + sum(1 for m in sims if m <= mean)) / (n_sim + 1)
    null_mean = sum(sims) / n_sim
    print(f"{label:<46} n={n:<4} mean={mean:6.2f} km  median={med:5.2f}  <10km={100*w10:5.1f}%  "
          f"null mean={null_mean:6.2f}  MC p={p:.4f}")
    return mean, med, w10, p


print(f"candi: {len(candi)}  sites: {len(sites)}  non-temple (E153 rule): {len(non_temple)}")
modern = [s for s in non_temple if MODERN.search(s["name"])]
print(f"non-temple sites with modern/non-ancient names: {len(modern)} e.g. {[s['name'] for s in modern[:10]]}")
describe("E153 rule (baseline, should match README)", non_temple)
clean = [s for s in non_temple if not MODERN.search(s["name"])]
describe("minus modern/non-ancient names", clean)
strict = [s for s in clean if s["type"] in {"archaeological_site", "ruins", "situs_arkeologi"}]
describe("strict: typed archaeological_site/ruins only", strict)
