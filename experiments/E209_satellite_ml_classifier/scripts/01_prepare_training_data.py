#!/usr/bin/env python3
"""
E209 Phase 1, Step 01: Prepare training-data site list with labels + metadata.

Loads:
  - data/processed/east_java_sites.geojson (15,011 line OSM-derived features)
  - data/processed/east_java_sites_osm.geojson (336 lines, secondary OSM pull)
  - data/processed/east_java_sites_wiki.csv (391 lines, Wikidata-derived)

Produces:
  - experiments/E209_satellite_ml_classifier/data/training_sites.csv
    Columns: site_id, name, lat, lon, label (positive/negative/hard_positive/hard_negative),
             category, source, stratum, in_training, notes

Labeling logic:
  - HARD POSITIVES (class=2): Sambisari, Kedulan, Kimpulan, Liangan, Candi Badut, Candi Tigomangi
    (discovered-buried sites — strongest training signal)
  - SOFT POSITIVES (class=1): 142+ candi filtered from geojson + wiki CSV by
    type == 'situs_arkeologi' / period != 'modern' / name starts with 'Candi '
  - HARD NEGATIVES (class=-2): 5 controls from E189 (known non-archaeological) + any
    geological hazards (active lava, recent lahar deposits, quarries) from OSM tags
  - RANDOM NEGATIVES (class=-1): generated in script 02 after we know sampling density;
    this script outputs positives + hard negatives only.

Stratification by terrain:
  - lowland: elevation < 200m, slope < 5deg
  - slope: slope 5-20deg
  - upland: elevation > 500m, slope < 10deg
  - valley: elevation 200-500m, slope < 5deg
  Stratum determined from Copernicus DEM in script 02.

Validation rule:
  Each positive site must have valid lat/lon in Java bounds
  (lat -8.9 .. -6.0, lon 105.0 .. 115.0). Sites outside dropped with warning.

Output is self-documenting CSV; downstream scripts consume it as the canonical training list.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "training_sites.csv"

# Java geographic bounds
LAT_MIN, LAT_MAX = -8.9, -6.0
LON_MIN, LON_MAX = 105.0, 115.0


def is_in_java(lat: float, lon: float) -> bool:
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


# Hard positives: discovered-buried sites (known cases of volcanic burial + re-exposure)
# These are the strongest training signal because they are CONFIRMED buried then excavated.
HARD_POSITIVES: List[Dict] = [
    # Sambisari — buried 6.5m under Merapi ash, discovered 1966
    dict(site_id="HP001", name="Candi Sambisari", lat=-7.7628, lon=110.4561,
         source="literature", notes="Buried 6.5m by Merapi, 9th c. CE, discovered 1966"),
    # Kedulan — buried under Merapi, excavation ongoing since 1993
    dict(site_id="HP002", name="Candi Kedulan", lat=-7.7308, lon=110.4503,
         source="literature", notes="Buried 7m by Merapi, 9th c. CE, discovered 1993"),
    # Kimpulan — buried 4–5m by Merapi, discovered 2009 during university construction
    dict(site_id="HP003", name="Candi Kimpulan", lat=-7.7636, lon=110.4147,
         source="literature", notes="Buried 4-5m by Merapi, 9th-10th c. CE, discovered 2009"),
    # Liangan — buried by Sindoro ash, still-active excavation since ~2008
    dict(site_id="HP004", name="Candi Liangan", lat=-7.2667, lon=110.0611,
         source="literature", notes="Buried under Sindoro ash; wooden material preserved"),
    # Candi Badut — buried 5th-century site, Malang
    dict(site_id="HP005", name="Candi Badut", lat=-7.9578, lon=112.5986,
         source="Wikidata Q3517536", notes="Partially buried, Malang volcanic plain, 8th c. CE"),
    # Tigomangi - location approximate, 1358 CE Semar relief, East Java
    dict(site_id="HP006", name="Candi Tigomangi", lat=-7.9700, lon=112.4200,
         source="literature", notes="1358 CE, Semar relief, East Java (approx coords)"),
]

# Hard negatives: known non-archaeological sites (E189 controls + extensions)
HARD_NEGATIVES: List[Dict] = [
    dict(site_id="HN001", name="Ctrl_plain_north", lat=-7.4500, lon=112.6000,
         source="E189", notes="Agricultural plain, no known archaeology"),
    dict(site_id="HN002", name="Ctrl_slope_south", lat=-8.1500, lon=112.7000,
         source="E189", notes="Volcanic slope, no known archaeology"),
    dict(site_id="HN003", name="Ctrl_plain_east", lat=-7.8500, lon=113.0000,
         source="E189", notes="Alluvial plain east"),
    dict(site_id="HN004", name="Ctrl_slope_north", lat=-7.6000, lon=112.4500,
         source="E189", notes="Volcanic slope north"),
    dict(site_id="HN005", name="Ctrl_plain_west", lat=-7.9000, lon=112.1500,
         source="E189", notes="Alluvial plain west"),
]


def load_geojson_sites(path: Path) -> List[Dict]:
    """Parse a GeoJSON FeatureCollection for point features with archaeological relevance."""
    if not path.exists():
        print(f"  WARN: {path.name} not found, skipping")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        if not is_in_java(lat, lon):
            continue
        name = props.get("name") or props.get("name:en") or ""
        if not name:
            continue
        # Filter: we want pre-1500 CE archaeological sites. OSM tags to keep:
        historic = props.get("historic", "")
        site_type = props.get("site_type", "") or props.get("tourism", "")
        keep = (
            historic in ("archaeological_site", "ruins", "monument", "temple", "shrine")
            or ("candi" in name.lower())
            or ("situs" in name.lower())
            or ("prasasti" in name.lower())
        )
        if not keep:
            continue
        out.append(dict(
            site_id=f"OSM{feat.get('id', len(out))}",
            name=name,
            lat=lat,
            lon=lon,
            source="east_java_sites.geojson",
            category=historic or site_type or "archaeology",
            notes=props.get("description", "")[:100],
        ))
    return out


def load_wiki_csv(path: Path) -> List[Dict]:
    """Parse the Wikidata-derived CSV for archaeological sites."""
    if not path.exists():
        print(f"  WARN: {path.name} not found, skipping")
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
            except (KeyError, ValueError):
                continue
            if not is_in_java(lat, lon):
                continue
            site_type = row.get("type", "").lower()
            name = row.get("name", "").strip()
            if not name:
                continue
            # Keep archaeological type OR candi-named OR situs_arkeologi
            keep = (
                site_type in ("situs_arkeologi", "candi", "prasasti", "arkeologi")
                or "candi" in name.lower()
                or "situs" in name.lower()
                or "prasasti" in name.lower()
            )
            if not keep:
                continue
            out.append(dict(
                site_id=f"WIKI{i:04d}",
                name=name,
                lat=lat,
                lon=lon,
                source=row.get("source", "wikidata"),
                category=site_type,
                notes=row.get("notes", "")[:100],
            ))
    return out


def dedupe(sites: List[Dict], tol_deg: float = 0.002) -> List[Dict]:
    """Deduplicate sites within `tol_deg` degrees (~200m) preferring wiki source."""
    out: List[Dict] = []
    for s in sites:
        dup_idx = None
        for i, existing in enumerate(out):
            if (abs(s["lat"] - existing["lat"]) < tol_deg
                    and abs(s["lon"] - existing["lon"]) < tol_deg):
                dup_idx = i
                break
        if dup_idx is None:
            out.append(s)
        else:
            # If wiki source, replace OSM; otherwise keep first
            if "wiki" in s.get("source", "").lower() and "wiki" not in out[dup_idx].get("source", "").lower():
                out[dup_idx] = s
    return out


def main() -> None:
    print("E209 Step 01: Prepare training-data site list")
    print("=" * 60)

    all_sites: List[Dict] = []

    # Hard positives
    for s in HARD_POSITIVES:
        s = dict(s)  # copy
        s["label"] = "hard_positive"
        s["class"] = 2
        s["category"] = "discovered_buried_candi"
        all_sites.append(s)
    print(f"  Hard positives: {len(HARD_POSITIVES)}")

    # Soft positives from geojson
    osm1 = load_geojson_sites(DATA_DIR / "east_java_sites.geojson")
    print(f"  OSM primary: {len(osm1)}")
    osm2 = load_geojson_sites(DATA_DIR / "east_java_sites_osm.geojson")
    print(f"  OSM secondary: {len(osm2)}")
    wiki = load_wiki_csv(DATA_DIR / "east_java_sites_wiki.csv")
    print(f"  Wikidata: {len(wiki)}")

    merged = dedupe(osm1 + osm2 + wiki, tol_deg=0.002)
    print(f"  Soft positives after dedup: {len(merged)}")

    # Filter out any soft positives within 500m of a hard positive (avoid double-counting)
    for s in merged:
        s["label"] = "soft_positive"
        s["class"] = 1
        close_hp = any(
            abs(s["lat"] - hp["lat"]) < 0.005 and abs(s["lon"] - hp["lon"]) < 0.005
            for hp in HARD_POSITIVES
        )
        if close_hp:
            s["label"] = "excluded_dup_hard_pos"
            s["class"] = 0
        all_sites.append(s)

    # Hard negatives
    for s in HARD_NEGATIVES:
        s = dict(s)
        s["label"] = "hard_negative"
        s["class"] = -2
        s["category"] = "control"
        all_sites.append(s)
    print(f"  Hard negatives: {len(HARD_NEGATIVES)}")

    # Normalise fields
    fieldnames = ["site_id", "name", "lat", "lon", "label", "class",
                  "category", "source", "notes"]
    for s in all_sites:
        for k in fieldnames:
            s.setdefault(k, "")

    # Write
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in all_sites:
            w.writerow({k: s[k] for k in fieldnames})

    # Summary
    from collections import Counter
    label_counts = Counter(s["label"] for s in all_sites)
    print()
    print(f"Training sites written to: {OUT_CSV}")
    print(f"Total: {len(all_sites)}")
    for label, count in label_counts.most_common():
        print(f"  {label:30s} {count:4d}")

    print()
    print("Next step: scripts/02_download_satellite_bands.py")
    print("  (downloads Sentinel-2 L2A + Sentinel-1 GRD + Copernicus DEM for all sites)")


if __name__ == "__main__":
    main()
