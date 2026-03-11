"""
E023: Cross-reference Pulotu culture locations with ISRIC SoilGrids soil pH data.
Tests prediction: volcanic/acidic soil areas → more elaborate mortuary timing.

Run: python experiments/E023_ritual_screening/isric_soil_crossref.py
"""
import csv
import io
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PULOTU_DIR = Path(__file__).parent / "data" / "pulotu" / "cldf"
RESULTS_DIR = Path(__file__).parent / "results"
MORTUARY_CSV = RESULTS_DIR / "pulotu_mortuary_comparison.csv"
OUT_CSV = RESULTS_DIR / "pulotu_soil_ph_crossref.csv"

ISRIC_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"


def query_isric_ph(lat, lon, retries=2):
    """Query ISRIC SoilGrids for soil pH at a location."""
    params = (
        f"?lon={lon}&lat={lat}"
        f"&property=phh2o&depth=0-5cm&value=mean"
    )
    url = ISRIC_URL + params

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # Navigate JSON response to extract pH value
            props = data.get("properties", {})
            layers = props.get("layers", [])
            for layer in layers:
                if layer.get("name") == "phh2o":
                    depths = layer.get("depths", [])
                    for depth in depths:
                        if "0-5" in depth.get("label", ""):
                            values = depth.get("values", {})
                            mean_val = values.get("mean")
                            if mean_val is not None:
                                # SoilGrids returns pH × 10
                                return mean_val / 10.0
            return None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # No data for this location (ocean/island)
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"  HTTP error {e.code} for ({lat}, {lon})")
                return None
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"  Error for ({lat}, {lon}): {e}")
                return None


def main():
    # Load cultures with coordinates
    cultures = {}
    with open(PULOTU_DIR / "cultures.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cultures[row["Name"]] = {
                "id": row["ID"],
                "lat": float(row["Latitude"]) if row["Latitude"] else None,
                "lon": float(row["Longitude"]) if row["Longitude"] else None,
            }

    # Load full mortuary package cultures
    full_package_names = set()
    with open(MORTUARY_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full_package_names.add(row["culture"])

    print("=" * 70)
    print("PULOTU × ISRIC SOIL pH CROSS-REFERENCE")
    print(f"Total cultures: {len(cultures)}")
    print(f"Full mortuary package: {len(full_package_names)}")
    print("=" * 70)

    # Query ISRIC for all cultures with valid coordinates
    results = []
    valid_count = 0
    for name, info in sorted(cultures.items()):
        lat, lon = info["lat"], info["lon"]
        if lat is None or lon is None:
            print(f"  SKIP {name}: no coordinates")
            continue

        valid_count += 1
        has_package = name in full_package_names
        print(f"  [{valid_count}] {name:30s} ({lat:>7.1f}, {lon:>7.1f}) package={has_package}", end="")

        ph = query_isric_ph(lat, lon)
        if ph is not None:
            print(f"  pH={ph:.1f}")
        else:
            print(f"  pH=N/A")

        results.append({
            "culture": name,
            "culture_id": info["id"],
            "latitude": lat,
            "longitude": lon,
            "soil_ph": ph if ph is not None else "",
            "has_full_mortuary_package": "yes" if has_package else "no",
        })

        # Rate limit: 1 request per second
        time.sleep(1.0)

    # Save results
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {OUT_CSV}")

    # Compute statistics
    pkg_ph = [float(r["soil_ph"]) for r in results
              if r["soil_ph"] and r["has_full_mortuary_package"] == "yes"]
    non_pkg_ph = [float(r["soil_ph"]) for r in results
                  if r["soil_ph"] and r["has_full_mortuary_package"] == "no"]
    all_ph = [float(r["soil_ph"]) for r in results if r["soil_ph"]]
    no_data = sum(1 for r in results if not r["soil_ph"])

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Total queried: {len(results)}")
    print(f"With pH data: {len(all_ph)}")
    print(f"No data (ocean/island): {no_data}")

    if pkg_ph:
        print(f"\nFull mortuary package cultures (n={len(pkg_ph)}):")
        print(f"  Mean pH: {sum(pkg_ph)/len(pkg_ph):.2f}")
        print(f"  Min pH:  {min(pkg_ph):.1f}")
        print(f"  Max pH:  {max(pkg_ph):.1f}")

    if non_pkg_ph:
        print(f"\nNon-package cultures (n={len(non_pkg_ph)}):")
        print(f"  Mean pH: {sum(non_pkg_ph)/len(non_pkg_ph):.2f}")
        print(f"  Min pH:  {min(non_pkg_ph):.1f}")
        print(f"  Max pH:  {max(non_pkg_ph):.1f}")

    if pkg_ph and non_pkg_ph:
        diff = sum(non_pkg_ph)/len(non_pkg_ph) - sum(pkg_ph)/len(pkg_ph)
        print(f"\n  pH difference (non-package - package): {diff:+.2f}")
        if diff > 0:
            print("  → Full-package cultures have MORE ACIDIC soil on average")
        else:
            print("  → Full-package cultures have LESS ACIDIC soil on average")

    # Identify volcanic-zone cultures (pH < 5.5)
    acidic = [(r["culture"], float(r["soil_ph"]))
              for r in results if r["soil_ph"] and float(r["soil_ph"]) < 5.5]
    if acidic:
        print(f"\nCultures in acidic soil (pH < 5.5): {len(acidic)}")
        for name, ph in sorted(acidic, key=lambda x: x[1]):
            pkg = "***" if any(r["culture"] == name and r["has_full_mortuary_package"] == "yes"
                              for r in results) else "   "
            print(f"  {pkg} {name:30s} pH={ph:.1f}")
        print("  *** = has full mortuary package")

    print("\nDone.")


if __name__ == "__main__":
    main()
