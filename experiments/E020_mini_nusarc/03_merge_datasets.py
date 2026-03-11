"""
E020: Merge mini_nusarc_v1.csv with nusarc_v0.1.csv (agent-harvested).

Strategy:
- mini_nusarc_v1 = primary (1 row per unique site, with site_id)
- nusarc_v0.1 = secondary (multiple dates per site, more detail)
- Identify new unique sites in v0.1 not in v1
- For each new site, take oldest date as representative
- Assign new NUSARC-00XX IDs
- Output: mini_nusarc_v2.csv

Run from repo root:
    python experiments/E020_mini_nusarc/03_merge_datasets.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Cultural period mapping based on date_bp
def classify_period(date_bp):
    if date_bp >= 300000:
        return "Lower_Paleolithic"
    elif date_bp >= 40000:
        return "Upper_Paleolithic"
    elif date_bp >= 10000:
        return "Mesolithic"
    elif date_bp >= 3000:
        return "Neolithic"
    elif date_bp >= 500:
        return "Metal_Age"
    else:
        return "Historical"

# Species normalization
SPECIES_MAP = {
    "H. sapiens": "Homo_sapiens",
    "H. sapiens (Deep Skull)": "Homo_sapiens",
    "H. sapiens (Wajak 1 & 2)": "Homo_sapiens",
    "H. sapiens (not H. floresiensis)": "Homo_sapiens",
    "H. floresiensis": "Homo_floresiensis",
    "H. luzonensis": "Homo_luzonensis",
    "H. erectus": "Homo_erectus",
    "H. erectus?": "Homo_erectus",
    "unknown hominin": "unknown",
    "fauna only (no human remains at depth)": "fauna_only",
    "Stegodon florensis": "fauna_only",
    "Stegodon florensis; Varanus komodoensis": "fauna_only",
}

# Country mapping by region
REGION_COUNTRY = {
    "Java": "ID",
    "Sumatra": "ID",
    "Sulawesi": "ID",
    "Kalimantan": "MY",  # Most Kalimantan sites are Sarawak/Sabah
    "Nusa_Tenggara": "ID",  # Default; Timor-Leste sites will be overridden
    "Philippines": "PH",
    "Maluku": "ID",
    "Madagascar": "MG",
}


def normalize_site_name(name):
    """Normalize site name for matching."""
    return (name.strip()
            .replace("(Great Cave)", "")
            .replace("(Lobang Hangus)", "")
            .replace("Laili", "Laili Cave")  # Match v1 naming
            .strip())


def main():
    print("=" * 60)
    print("E020: Merge mini_nusarc_v1 + nusarc_v0.1")
    print("=" * 60)

    # Load v1 (primary)
    v1 = pd.read_csv(DATA_DIR / "mini_nusarc_v1.csv")
    print(f"\n  v1: {len(v1)} sites")

    # Load v0.1 (secondary)
    v01 = pd.read_csv(RAW_DIR / "nusarc_v0.1.csv")
    print(f"  v0.1: {len(v01)} records")

    # Normalize v1 site names for matching
    v1_names = set()
    for name in v1["site_name"]:
        v1_names.add(name.strip().lower())
        # Also add without parenthetical
        base = name.split("(")[0].strip().lower()
        v1_names.add(base)

    # Find unique sites in v0.1 not in v1
    # Group v0.1 by site_name, take oldest date per site
    v01_grouped = v01.groupby("site_name").agg({
        "lat": "first",
        "lon": "first",
        "coord_precision": "first",
        "region": "first",
        "date_bp": "max",  # oldest = highest BP
        "date_type": "first",
        "date_error": "first",
        "site_type": "first",
        "context_detail": "first",
        "species": "first",
        "source_citation": "first",
        "confidence": "first",
        "notes": "first",
    }).reset_index()

    print(f"  v0.1 unique sites: {len(v01_grouped)}")

    new_sites = []
    for _, row in v01_grouped.iterrows():
        name_lower = row["site_name"].strip().lower()
        base_lower = name_lower.split("(")[0].strip()

        # Check if already in v1
        if name_lower in v1_names or base_lower in v1_names:
            continue

        # Also check partial match for tricky names
        matched = False
        for v1_name in v1_names:
            if name_lower in v1_name or v1_name in name_lower:
                matched = True
                break
        if matched:
            continue

        new_sites.append(row)

    print(f"  New sites from v0.1: {len(new_sites)}")
    for s in new_sites:
        print(f"    - {s['site_name']} ({s['region']}, {s['date_bp']} BP, {s['site_type']})")

    # Assign new NUSARC IDs starting from max existing
    max_id = v1["site_id"].str.extract(r"(\d+)")[0].astype(int).max()
    next_id = max_id + 1

    # Build new rows in v1 schema
    new_rows = []
    for s in new_sites:
        species = SPECIES_MAP.get(s["species"], "unknown")
        region = s["region"]
        country = REGION_COUNTRY.get(region, "ID")

        # Special case: Timor-Leste sites
        if "Timor" in str(s.get("context_detail", "")):
            country = "TL"

        # Special case: Kalimantan East (Indonesia) vs Sarawak (Malaysia)
        if region == "Kalimantan" and s["lat"] > 0 and s["lon"] > 115 and s["lon"] < 118:
            country = "ID"  # East Kalimantan

        date_bp = int(s["date_bp"])
        period = classify_period(date_bp)

        new_rows.append({
            "site_id": f"NUSARC-{next_id:04d}",
            "site_name": s["site_name"],
            "lat": s["lat"],
            "lon": s["lon"],
            "coord_precision": s["coord_precision"],
            "region": region,
            "country": country,
            "date_bp": date_bp,
            "date_type": s["date_type"],
            "date_error": s["date_error"] if pd.notna(s["date_error"]) else "",
            "site_type": s["site_type"],
            "context_detail": s["context_detail"],
            "cultural_period": period,
            "species": species,
            "source_citation": s["source_citation"],
            "confidence": s["confidence"],
            "notes": s["notes"],
        })
        next_id += 1

    # Combine
    new_df = pd.DataFrame(new_rows)
    v2 = pd.concat([v1, new_df], ignore_index=True)

    # Save
    v2.to_csv(DATA_DIR / "mini_nusarc_v2.csv", index=False)
    print(f"\n  OUTPUT: mini_nusarc_v2.csv — {len(v2)} sites")

    # Summary by region
    print(f"\n  By region:")
    for region in sorted(v2["region"].unique()):
        n = len(v2[v2["region"] == region])
        print(f"    {region:20s}: {n} sites")

    # Summary by site_type
    print(f"\n  By site type:")
    for st in sorted(v2["site_type"].unique()):
        n = len(v2[v2["site_type"] == st])
        print(f"    {st:20s}: {n}")

    # Check minimum viable dataset targets
    print(f"\n  Minimum Viable Dataset check:")
    targets = {
        "Java": 8, "Sumatra": 5, "Sulawesi": 8, "Kalimantan": 5,
        "Nusa_Tenggara": 4, "Philippines": 4, "Maluku": 3, "Madagascar": 3,
    }
    all_met = True
    for region, minimum in targets.items():
        n = len(v2[v2["region"] == region])
        status = "OK" if n >= minimum else "NEED MORE"
        if n < minimum:
            all_met = False
        print(f"    {region:20s}: {n}/{minimum} {'OK' if n >= minimum else f'NEED {minimum - n} MORE'}")

    if all_met:
        print(f"\n  ALL MINIMUM TARGETS MET!")
    else:
        print(f"\n  Some regions still below minimum.")

    print(f"\nMerge COMPLETE.")


if __name__ == "__main__":
    main()
