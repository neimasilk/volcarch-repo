#!/usr/bin/env python3
"""
E209 Step 01b: Add stratified random negatives to training set.

Problem: Step 01 produced only 5 hard negatives vs 115 positives — severe class
imbalance that renders K-fold CV AUC meaningless (see step 03 results: 0.479 ± 0.283).

Fix: Generate N random-sample locations in Java that are
  - ≥ 5 km from any known positive site
  - Within the Java geographic bounds
  - Stratified across terrain types (lowland / slope / upland)
    (terrain stratum assigned after Step 02 download fetches DEM)

Appends to data/training_sites.csv as label='random_negative', class=-1.

Usage:
  python 01b_add_random_negatives.py [--n 200]
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

E209_DIR = Path(__file__).resolve().parents[1]
SITES_CSV = E209_DIR / "data" / "training_sites.csv"

# Java bounds
LAT_MIN, LAT_MAX = -8.9, -6.0
LON_MIN, LON_MAX = 105.0, 115.0

# Exclusion threshold from any positive site (in degrees; ~5 km at the equator)
EXCLUSION_DEG = 0.045


def load_sites(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def near_any(lat: float, lon: float, points: list[tuple[float, float]],
             threshold_deg: float = EXCLUSION_DEG) -> bool:
    for la, lo in points:
        if abs(lat - la) < threshold_deg and abs(lon - lo) < threshold_deg:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200,
                        help="Number of random negatives to generate.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    existing = load_sites(SITES_CSV)
    already_random = [s for s in existing if s["label"] == "random_negative"]
    print(f"Already in file: {len(existing)} sites (inc. {len(already_random)} random negatives)")

    if already_random:
        print(f"  Random negatives already exist; appending more")

    # All positive points (class > 0) to exclude
    positive_points = []
    for s in existing:
        try:
            cls = int(s["class"] or "0")
            if cls > 0:
                positive_points.append((float(s["lat"]), float(s["lon"])))
        except (ValueError, KeyError):
            continue

    print(f"Excluding ≥5km from {len(positive_points)} positive sites")

    # Generate
    generated = []
    tries = 0
    max_tries = args.n * 50
    next_id = len(already_random) + 1
    while len(generated) < args.n and tries < max_tries:
        tries += 1
        lat = random.uniform(LAT_MIN, LAT_MAX)
        lon = random.uniform(LON_MIN, LON_MAX)
        # Also bias toward Java main island (exclude very far-east Bali bounds)
        # Rough Java bbox: 105.0..114.5, -8.8..-5.9
        if lon > 114.5:
            continue
        # Exclude Madura strait if lat between -7.0 and -6.8 and lon > 112.5
        if -7.0 < lat < -6.8 and lon > 112.5:
            continue
        if near_any(lat, lon, positive_points):
            continue
        # Also exclude from previously generated randoms to avoid clustering
        if near_any(lat, lon, [(g["lat"], g["lon"]) for g in generated],
                    threshold_deg=0.02):  # ~2km spacing
            continue
        generated.append({
            "site_id": f"RN{next_id:04d}",
            "name": f"RandomNeg_{next_id:04d}",
            "lat": lat,
            "lon": lon,
            "label": "random_negative",
            "class": -1,
            "category": "random_control",
            "source": "auto_generated",
            "notes": f"Random negative ≥{EXCLUSION_DEG*111:.1f}km from any positive site",
        })
        next_id += 1

    print(f"Generated {len(generated)} random negatives (after {tries} tries)")
    if len(generated) < args.n:
        print(f"  WARN: requested {args.n}, only generated {len(generated)}")

    # Append to CSV
    fieldnames = ["site_id", "name", "lat", "lon", "label", "class",
                  "category", "source", "notes"]
    with open(SITES_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for s in generated:
            row = {k: s.get(k, "") for k in fieldnames}
            w.writerow(row)

    print(f"Appended to: {SITES_CSV}")
    print()
    print("Next: run 02_extract_s2_features.py to extract features for new sites")
    print(f"      (it will skip already-processed sites via the checkpoint)")


if __name__ == "__main__":
    main()
