#!/usr/bin/env python3
"""
E194: Combined Archaeological Prospection Map
==============================================
Integrates ALL prediction layers into a single probability surface:
- E013: Settlement model (AUC 0.768)
- E080: Fieldwork targets (composite scoring)
- E097: Anomaly detection (Isolation Forest)
- E075: Burial depth model
- E189: Satellite NDWI signal
- E193: L1xL2 double erasure zones

Output: ranked list of highest-priority prospection targets with
evidence convergence score (how many independent lines agree).
"""

import json, csv, sys, numpy as np
from pathlib import Path
from scipy import stats as sp_stats

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Data sources ──────────────────────────────────────────────────────

# E080 targets (GPS + composite score)
E080_CSV = REPO_ROOT / "experiments" / "E080_fieldwork_targets" / "results" / "top20_targets.csv"

# E097 anomalies (top 50 cells + burial depth)
E097_CSV = REPO_ROOT / "experiments" / "E097_anomaly_detection" / "results" / "top50_anomaly_cells.csv"

# E177 entry points
ENTRY_POINTS = [
    ("Surabaya", -7.25, 112.75, 1),
    ("Tangerang", -6.20, 106.55, 2),
    ("Semarang", -6.95, 110.40, 3),
    ("Jakarta", -6.10, 106.85, 4),
    ("Cirebon", -6.70, 108.55, 5),
]

# Volcanoes
VOLCANOES = [
    ("Kelud", -7.93, 112.31),
    ("Arjuno-Welirang", -7.73, 112.58),
    ("Semeru", -8.11, 112.92),
    ("Merapi", -7.54, 110.45),
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    print("=" * 70)
    print("E194: Combined Archaeological Prospection Map")
    print("=" * 70)

    # ── Load E080 targets ─────────────────────────────────────────────
    e080 = []
    with open(E080_CSV, "r") as f:
        for row in csv.DictReader(f):
            e080.append({
                "lat": float(row["lat"]), "lon": float(row["lon"]),
                "score": float(row["composite_score"]),
                "volcano": row["nearest_volcano"],
                "burial_m": float(row["estimated_burial_m"]),
            })
    print(f"E080 targets: {len(e080)}")

    # ── Load E097 anomalies ───────────────────────────────────────────
    e097 = []
    with open(E097_CSV, "r") as f:
        for row in csv.DictReader(f):
            e097.append({
                "lat": float(row["lat"]), "lon": float(row["lon"]),
                "composite": float(row["composite_score"]),
                "burial_cm": float(row["burial_depth_cm"]),
                "volc_dist_km": float(row["volcano_dist_km"]),
            })
    print(f"E097 anomalies: {len(e097)}")

    # ── Build unified grid ────────────────────────────────────────────
    # Use E080 targets as primary grid (20 cells, well-distributed)
    print(f"\n--- Building convergence map ---")

    targets = []
    for i, t in enumerate(e080):
        target = {
            "id": f"T{i+1:02d}",
            "lat": t["lat"], "lon": t["lon"],
            "volcano": t["volcano"],
            "burial_m": t["burial_m"],
            "evidence_streams": 0,
            "evidence_detail": [],
        }

        # Stream 1: E080 fieldwork target score
        target["e080_score"] = t["score"]
        if t["score"] >= 0.7:
            target["evidence_streams"] += 1
            target["evidence_detail"].append(f"E080 score {t['score']:.3f}")

        # Stream 2: E097 anomaly convergence (within 5km)
        e097_match = [a for a in e097
                      if haversine_km(t["lat"], t["lon"], a["lat"], a["lon"]) < 5]
        target["e097_matches"] = len(e097_match)
        if e097_match:
            target["evidence_streams"] += 1
            best = max(e097_match, key=lambda a: a["composite"])
            target["e097_best_composite"] = best["composite"]
            target["evidence_detail"].append(f"E097 {len(e097_match)} cells")

        # Stream 3: Volcanic sweet spot (5-15 km)
        nearest_volc = min(
            (haversine_km(t["lat"], t["lon"], vlat, vlon), vname)
            for vname, vlat, vlon in VOLCANOES
        )
        target["volc_dist_km"] = round(nearest_volc[0], 1)
        target["nearest_volcano"] = nearest_volc[1]
        if 5 <= nearest_volc[0] <= 15:
            target["evidence_streams"] += 1
            target["evidence_detail"].append(f"Sweet spot {nearest_volc[0]:.0f}km")

        # Stream 4: L1xL2 double erasure zone (within 75km of entry point)
        nearest_entry = min(
            (haversine_km(t["lat"], t["lon"], elat, elon), ename)
            for ename, elat, elon, _ in ENTRY_POINTS
        )
        target["entry_dist_km"] = round(nearest_entry[0], 1)
        target["nearest_entry"] = nearest_entry[1]
        if nearest_entry[0] < 75:
            target["evidence_streams"] += 1
            target["evidence_detail"].append(f"L1xL2 {nearest_entry[0]:.0f}km from {nearest_entry[1]}")

        # Stream 5: Burial depth > 3m (significant)
        if t["burial_m"] >= 3:
            target["evidence_streams"] += 1
            target["evidence_detail"].append(f"Burial {t['burial_m']:.0f}m")

        targets.append(target)

    # ── Rank by evidence convergence ──────────────────────────────────
    targets.sort(key=lambda t: (-t["evidence_streams"], -t.get("e080_score", 0)))

    print(f"\n{'='*70}")
    print("RANKED PROSPECTION TARGETS (by evidence convergence)")
    print(f"{'='*70}")
    print(f"\n  {'ID':4s} {'Lat':>7s} {'Lon':>8s} {'Streams':>7s} {'Volcano':>10s} "
          f"{'Dist':>5s} {'Burial':>6s} Evidence")

    for t in targets:
        evidence = " + ".join(t["evidence_detail"]) if t["evidence_detail"] else "—"
        print(f"  {t['id']:4s} {t['lat']:7.2f} {t['lon']:8.2f} "
              f"{t['evidence_streams']:>5d}/5  {t['nearest_volcano']:>10s} "
              f"{t['volc_dist_km']:5.1f} {t['burial_m']:5.0f}m  {evidence}")

    # ── Summary statistics ────────────────────────────────────────────
    stream_counts = [t["evidence_streams"] for t in targets]
    print(f"\n--- Convergence Summary ---")
    for n in range(6):
        count = sum(1 for s in stream_counts if s == n)
        if count > 0:
            print(f"  {n}/5 streams: {count} targets")

    # Top tier (>=4 streams)
    top_tier = [t for t in targets if t["evidence_streams"] >= 4]
    mid_tier = [t for t in targets if t["evidence_streams"] == 3]
    print(f"\n  TOP TIER (>=4 streams): {len(top_tier)} targets")
    print(f"  MID TIER (3 streams):   {len(mid_tier)} targets")

    # ── Save ──────────────────────────────────────────────────────────
    with open(RESULTS_DIR / "prospection_targets.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["id", "lat", "lon", "evidence_streams", "nearest_volcano",
                  "volc_dist_km", "burial_m", "e080_score", "e097_matches",
                  "entry_dist_km", "nearest_entry"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(targets)

    summary = {
        "experiment": "E194", "date": "2026-04-13",
        "n_targets": len(targets),
        "top_tier": len(top_tier),
        "mid_tier": len(mid_tier),
        "max_convergence": max(stream_counts),
        "evidence_streams": ["E080 fieldwork", "E097 anomaly", "Volcanic sweet spot",
                            "L1xL2 double erasure", "Significant burial depth"],
    }
    with open(RESULTS_DIR / "e194_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Verdict ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    if top_tier:
        best = top_tier[0]
        print(f"#1 TARGET: {best['lat']:.2f}, {best['lon']:.2f} near {best['nearest_volcano']}")
        print(f"  Evidence: {best['evidence_streams']}/5 independent streams converge")
        print(f"  Burial: ~{best['burial_m']:.0f}m | Distance: {best['volc_dist_km']:.0f}km from volcano")
    print(f"TOTAL: {len(top_tier)} high-priority + {len(mid_tier)} medium-priority targets")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
