"""
E070: Cross-reference colonial archaeological register with modern site databases.

Matching strategy:
1. Name matching (fuzzy) — Levenshtein-based on site_name and modern_name vs modern DB names
2. Coordinate proximity (<10 km) — haversine distance between coordinate pairs

Modern databases used:
- data/processed/east_java_sites_wiki.csv  (391 sites, Wikidata+Wikipedia)
- data/processed/dashboard/sites.csv       (380 sites, OSM+Wiki merged for E001)

Output: experiments/E070_colonial_literature_mining/results/colonial_vs_modern_comparison.csv
"""

import sys
import os
import csv
import math
import re
from pathlib import Path
from collections import defaultdict

# UTF-8 stdout for Windows
sys.stdout.reconfigure(encoding='utf-8')

REPO = Path(__file__).parent.parent.parent

# ── Load colonial register ──────────────────────────────────────────────

def load_colonial():
    path = REPO / "experiments" / "E070_colonial_literature_mining" / "results" / "colonial_site_register_v1.0.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Parse lat/lon safely
            lat = parse_float(r.get("lat", ""))
            lon = parse_float(r.get("lon", ""))
            rows.append({
                "colonial_site_name": r.get("site_name", "").strip(),
                "colonial_modern_name": r.get("modern_name", "").strip(),
                "source": r.get("source", "").strip(),
                "year_report": r.get("year_report", "").strip(),
                "province": r.get("province", "").strip(),
                "regency": r.get("regency", "").strip(),
                "lat": lat,
                "lon": lon,
                "burial_depth_m": r.get("burial_depth_m", "").strip(),
                "condition": r.get("condition", "").strip(),
                "context": r.get("context", "").strip(),
                "volcanic_system": r.get("volcanic_system", "").strip(),
            })
    return rows


def parse_float(s):
    if not s:
        return None
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


# ── Load modern databases ───────────────────────────────────────────────

def load_wiki_sites():
    path = REPO / "data" / "processed" / "east_java_sites_wiki.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r.get("name", "").strip()
            lat = parse_float(r.get("lat", ""))
            lon = parse_float(r.get("lon", ""))
            if name:
                rows.append({"name": name, "lat": lat, "lon": lon, "db": "wiki"})
    return rows


def load_dashboard_sites():
    path = REPO / "data" / "processed" / "dashboard" / "sites.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r.get("name", "").strip()
            lat = parse_float(r.get("lat", ""))
            lon = parse_float(r.get("lon", ""))
            if name:
                rows.append({"name": name, "lat": lat, "lon": lon, "db": "dashboard"})
    return rows


# ── Merge modern databases (deduplicate by name) ───────────────────────

def merge_modern(wiki, dashboard):
    """Merge both modern DBs, dedup by normalized name."""
    seen = {}
    merged = []
    for s in wiki + dashboard:
        key = normalize(s["name"])
        if key not in seen:
            seen[key] = s
            merged.append(s)
        else:
            # If existing has no coords but new one does, replace
            if seen[key]["lat"] is None and s["lat"] is not None:
                seen[key] = s
    return merged


# ── Matching utilities ──────────────────────────────────────────────────

def normalize(name):
    """Normalize site name for comparison."""
    if not name:
        return ""
    s = name.lower().strip()
    # Common spelling variants
    s = s.replace("tj", "c").replace("dj", "j").replace("oe", "u")
    # Remove prefixes
    for prefix in ["candi ", "situs ", "prasasti ", "arca ", "pura ", "cagar budaya "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Remove non-alphanumeric
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def levenshtein(s1, s2):
    """Simple Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(s2)]


def name_similarity(n1, n2):
    """Normalized similarity 0-1 based on Levenshtein."""
    a = normalize(n1)
    b = normalize(n2)
    if not a or not b:
        return 0.0
    dist = levenshtein(a, b)
    maxlen = max(len(a), len(b))
    return 1.0 - dist / maxlen


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return float('inf')
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Main cross-reference ───────────────────────────────────────────────

def find_best_match(colonial_row, modern_sites, name_threshold=0.65, dist_threshold_km=10.0):
    """
    Find best match for a colonial site in modern database.
    Returns (match_type, match_name, match_db, dist_km, name_sim, details).
    """
    c_name = colonial_row["colonial_site_name"]
    c_modern = colonial_row["colonial_modern_name"]
    c_lat = colonial_row["lat"]
    c_lon = colonial_row["lon"]

    best_name_match = None
    best_name_sim = 0.0
    best_coord_match = None
    best_coord_dist = float('inf')

    for ms in modern_sites:
        m_name = ms["name"]

        # Name matching: try both colonial name and colonial-assigned modern name
        sim1 = name_similarity(c_name, m_name)
        sim2 = name_similarity(c_modern, m_name)
        sim = max(sim1, sim2)

        if sim > best_name_sim:
            best_name_sim = sim
            best_name_match = ms

        # Coordinate proximity
        dist = haversine_km(c_lat, c_lon, ms["lat"], ms["lon"])
        if dist < best_coord_dist:
            best_coord_dist = dist
            best_coord_match = ms

    # Determine match
    name_hit = best_name_sim >= name_threshold
    coord_hit = best_coord_dist <= dist_threshold_km

    if name_hit and coord_hit:
        # Both name AND coordinate match — check if same site
        # Use the name match as primary
        dist_to_name_match = haversine_km(c_lat, c_lon, best_name_match["lat"], best_name_match["lon"])
        return {
            "match_type": "BOTH",
            "matched_modern_name": best_name_match["name"],
            "matched_db": best_name_match["db"],
            "distance_km": round(dist_to_name_match, 2),
            "name_similarity": round(best_name_sim, 3),
        }
    elif name_hit:
        dist_to_name_match = haversine_km(c_lat, c_lon, best_name_match["lat"], best_name_match["lon"])
        return {
            "match_type": "NAME_ONLY",
            "matched_modern_name": best_name_match["name"],
            "matched_db": best_name_match["db"],
            "distance_km": round(dist_to_name_match, 2) if dist_to_name_match != float('inf') else "",
            "name_similarity": round(best_name_sim, 3),
        }
    elif coord_hit:
        sim_to_coord = max(
            name_similarity(c_name, best_coord_match["name"]),
            name_similarity(c_modern, best_coord_match["name"])
        )
        return {
            "match_type": "COORD_ONLY",
            "matched_modern_name": best_coord_match["name"],
            "matched_db": best_coord_match["db"],
            "distance_km": round(best_coord_dist, 2),
            "name_similarity": round(sim_to_coord, 3),
        }
    else:
        # No match — "lost site"
        nearest_name = best_name_match["name"] if best_name_match else ""
        nearest_coord = best_coord_match["name"] if best_coord_match else ""
        return {
            "match_type": "NO_MATCH",
            "matched_modern_name": "",
            "matched_db": "",
            "distance_km": round(best_coord_dist, 2) if best_coord_dist != float('inf') else "",
            "name_similarity": round(best_name_sim, 3),
            "nearest_by_name": nearest_name,
            "nearest_by_coord": nearest_coord,
            "nearest_coord_dist_km": round(best_coord_dist, 2) if best_coord_dist != float('inf') else "",
        }


def main():
    print("=" * 70)
    print("E070: COLONIAL vs MODERN SITE CROSS-REFERENCE")
    print("=" * 70)

    # Load data
    colonial = load_colonial()
    print(f"\nColonial register: {len(colonial)} entries")

    wiki = load_wiki_sites()
    dashboard = load_dashboard_sites()
    print(f"Modern wiki DB: {len(wiki)} sites")
    print(f"Modern dashboard DB: {len(dashboard)} sites")

    modern = merge_modern(wiki, dashboard)
    print(f"Modern merged (deduplicated): {len(modern)} sites")

    # Cross-reference each colonial site
    results = []
    for c in colonial:
        match = find_best_match(c, modern)
        row = {
            "colonial_site_name": c["colonial_site_name"],
            "colonial_modern_name": c["colonial_modern_name"],
            "source": c["source"],
            "year_report": c["year_report"],
            "province": c["province"],
            "regency": c["regency"],
            "colonial_lat": c["lat"] if c["lat"] is not None else "",
            "colonial_lon": c["lon"] if c["lon"] is not None else "",
            "burial_depth_m": c["burial_depth_m"],
            "condition": c["condition"],
            "context": c["context"],
            "volcanic_system": c["volcanic_system"],
            "match_type": match["match_type"],
            "matched_modern_name": match.get("matched_modern_name", ""),
            "matched_db": match.get("matched_db", ""),
            "distance_km": match.get("distance_km", ""),
            "name_similarity": match.get("name_similarity", ""),
            "nearest_by_name": match.get("nearest_by_name", ""),
            "nearest_by_coord": match.get("nearest_by_coord", ""),
            "nearest_coord_dist_km": match.get("nearest_coord_dist_km", ""),
        }
        results.append(row)

    # ── Write output ────────────────────────────────────────────────────
    outpath = REPO / "experiments" / "E070_colonial_literature_mining" / "results" / "colonial_vs_modern_comparison.csv"
    fieldnames = [
        "colonial_site_name", "colonial_modern_name", "source", "year_report",
        "province", "regency", "colonial_lat", "colonial_lon",
        "burial_depth_m", "condition", "context", "volcanic_system",
        "match_type", "matched_modern_name", "matched_db",
        "distance_km", "name_similarity",
        "nearest_by_name", "nearest_by_coord", "nearest_coord_dist_km",
    ]
    with open(outpath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nResults written to: {outpath}")

    # ── Summary statistics ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    match_types = defaultdict(list)
    for r in results:
        match_types[r["match_type"]].append(r)

    total = len(results)
    matched = len(match_types["BOTH"]) + len(match_types["NAME_ONLY"]) + len(match_types["COORD_ONLY"])
    lost = len(match_types["NO_MATCH"])

    print(f"\nTotal colonial entries:  {total}")
    print(f"  MATCHED (both):       {len(match_types['BOTH'])}  (name + coordinate match)")
    print(f"  MATCHED (name only):  {len(match_types['NAME_ONLY'])}  (name match, coords differ/missing)")
    print(f"  MATCHED (coord only): {len(match_types['COORD_ONLY'])}  (nearby coords, different name)")
    print(f"  ─────────────────────")
    print(f"  TOTAL MATCHED:        {matched}")
    print(f"  NO MATCH ('lost'):    {lost}")
    print(f"  Match rate:           {matched/total*100:.1f}%")

    # Detail: matched sites
    print("\n── MATCHED SITES ──")
    for mtype in ["BOTH", "NAME_ONLY", "COORD_ONLY"]:
        if match_types[mtype]:
            print(f"\n  [{mtype}]")
            for r in match_types[mtype]:
                dist_str = f"{r['distance_km']} km" if r['distance_km'] != "" else "no coords"
                print(f"    {r['colonial_site_name']:45s} → {r['matched_modern_name']:35s} "
                      f"(sim={r['name_similarity']}, dist={dist_str})")

    # Detail: lost sites
    print("\n── LOST SITES (colonial entries NOT in modern databases) ──")
    for r in match_types["NO_MATCH"]:
        depth_str = f"depth={r['burial_depth_m']}m" if r["burial_depth_m"] else "no depth"
        print(f"  {r['colonial_site_name']:45s} [{r['source']} {r['year_report']}] "
              f"({r['condition']}, {depth_str})")

    # Volcanic context analysis
    print("\n── VOLCANIC CONTEXT ANALYSIS ──")
    volcanic_lost = [r for r in match_types["NO_MATCH"] if r["context"] == "volcanic"]
    volcanic_matched = [r for r in results if r["match_type"] != "NO_MATCH" and r["context"] == "volcanic"]
    print(f"  Volcanic-context entries matched:     {len(volcanic_matched)}")
    print(f"  Volcanic-context entries LOST:         {len(volcanic_lost)}")
    if volcanic_lost:
        print(f"  Lost volcanic sites:")
        for r in volcanic_lost:
            print(f"    - {r['colonial_site_name']} ({r['volcanic_system']}, depth={r['burial_depth_m']}m)")

    # Sites with burial depth that are lost
    print("\n── LOST SITES WITH BURIAL DEPTH DATA ──")
    depth_lost = [r for r in match_types["NO_MATCH"] if r["burial_depth_m"]]
    for r in depth_lost:
        print(f"  {r['colonial_site_name']:45s} depth={r['burial_depth_m']}m  [{r['condition']}]")

    # Now check modern sites NOT in colonial register
    print("\n── MODERN SITES AS COMPARISON ──")
    matched_modern_names = set()
    for r in results:
        if r["match_type"] != "NO_MATCH":
            matched_modern_names.add(normalize(r["matched_modern_name"]))

    # Count modern "candi" sites not matched
    modern_candi = [m for m in modern if "candi" in m["name"].lower()]
    modern_candi_not_colonial = [m for m in modern_candi
                                  if normalize(m["name"]) not in matched_modern_names]
    print(f"  Modern DB candi sites: {len(modern_candi)}")
    print(f"  Modern candis NOT in colonial register: {len(modern_candi_not_colonial)}")
    print(f"  (These are post-colonial discoveries or sites not covered by OV reports)")

    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print("=" * 70)
    print(f"""
Of {total} colonial-era site entries:
- {matched} ({matched/total*100:.1f}%) can be identified in modern databases
- {lost} ({lost/total*100:.1f}%) are 'lost' — not found in modern registries

'Lost' sites include:
1. Generic/unnamed entries (excavation reports without location)
2. Sites destroyed by volcanic activity or looting
3. Sites still buried and not yet rediscovered
4. Sites with imprecise locations that prevent matching

This {lost/total*100:.0f}% loss rate between colonial records and modern databases
is itself evidence of taphonomic processes — both natural (volcanic burial)
and cultural (looting, development) — erasing archaeological heritage.
""")


if __name__ == "__main__":
    main()
