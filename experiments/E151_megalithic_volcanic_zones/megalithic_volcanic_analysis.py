#!/usr/bin/env python3
"""
E151: Megalithic Distribution vs Volcanic Zones

Question:
  Do visible megalithic monuments in volcanic zones refute VOLCARCH?

Answer tested here:
  No. They refine it. Stone monuments survive in volcanic landscapes, while
  organic/domestic settlement signatures remain overwhelmingly absent.

Method:
  - Curated 4-case volcanic megalith dataset requested in WORKSTATE:
      Gunung Padang, Cipari, Bondowoso cluster, Pasemah highlands
  - Compute distance to nearest active volcano
  - Code whether STONE monument survives and whether ORGANIC settlement is
    archaeologically visible
  - Cross-reference project-wide asymmetry metrics from E117 / E129 / E140
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


BASE = Path("experiments/E151_megalithic_volcanic_zones")
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# East Java volcano coordinates come from E016. West Java / Dempo are stable
# reference coordinates added here because E151 explicitly needs those cases.
VOLCANOES = {
    "Gede-Pangrango": {"lat": -6.770, "lon": 106.980, "source": "stable reference"},
    "Ciremai": {"lat": -6.892, "lon": 108.408, "source": "stable reference"},
    "Raung": {"lat": -8.125, "lon": 114.042, "source": "E016"},
    "Ijen": {"lat": -8.058, "lon": 114.242, "source": "E016"},
    "Dempo": {"lat": -4.030, "lon": 103.130, "source": "stable reference"},
}


CASES = [
    {
        "site": "Gunung Padang",
        "lat": -6.99,
        "lon": 107.06,
        "coordinate_note": "exact site coordinate from E071",
        "volcano_candidates": ["Gede-Pangrango"],
        "material": "stone terraces / basalt columns",
        "stone_monument_survives": True,
        "organic_settlement_visible": False,
        "evidence": (
            "Massive punden berundak survives in volcanic West Java, but no paired "
            "lowland organic settlement horizon is visible in the record."
        ),
        "reference": "E071; Blind Spot BS-1",
    },
    {
        "site": "Cipari / Kuningan megalithic",
        "lat": -6.95,
        "lon": 108.48,
        "coordinate_note": "exact site coordinate from E071",
        "volcano_candidates": ["Ciremai"],
        "material": "menhirs / stone terraces / sarcophagi",
        "stone_monument_survives": True,
        "organic_settlement_visible": False,
        "evidence": (
            "Stone mortuary architecture survives within the Ciremai volcanic zone; "
            "the domestic/organic settlement package does not."
        ),
        "reference": "E071; Blind Spot BS-1",
    },
    {
        "site": "Bondowoso megalithic cluster",
        "lat": -7.914444581620128,
        "lon": 113.82185071188832,
        "coordinate_note": "kabupaten centroid proxy from E056 (cluster-level, not single monument)",
        "volcano_candidates": ["Raung", "Ijen"],
        "material": "dolmens / sarcophagi / stone graves",
        "stone_monument_survives": True,
        "organic_settlement_visible": False,
        "evidence": (
            "Megalithic Bondowoso survives as stone features in East Java's volcanic belt. "
            "Nearby Garahan grave shows ~18 cm ash from Raung directly above a megalithic burial."
        ),
        "reference": "Blind Spot BS-1; JOURNAL 2026-03-21 volcanic evidence log",
    },
    {
        "site": "Pasemah highlands",
        "lat": -3.79673,
        "lon": 103.137,
        "coordinate_note": "Besemah highland proxy from ABVD/Pulotu language coordinate",
        "volcano_candidates": ["Dempo"],
        "material": "stone statues / megalithic sculpture",
        "stone_monument_survives": True,
        "organic_settlement_visible": False,
        "evidence": (
            "OV 1922 places Pasemah antiquities at the foot of Dempo; carved stone statues remain visible, "
            "but domestic settlement archaeology is not comparably preserved."
        ),
        "reference": "OV 1922; Pulotu Besemah note; Blind Spot BS-1",
    },
]


def nearest_volcano(case: dict) -> tuple[str, float]:
    best_name = None
    best_dist = None
    for volcano in case["volcano_candidates"]:
        entry = VOLCANOES[volcano]
        dist = haversine_km(case["lat"], case["lon"], entry["lat"], entry["lon"])
        if best_dist is None or dist < best_dist:
            best_name = volcano
            best_dist = dist
    return best_name, best_dist or 0.0


def main() -> None:
    print("=" * 72)
    print("E151: Megalithic Distribution vs Volcanic Zones")
    print("=" * 72)

    e117 = json.load(
        open("experiments/E117_archaeological_onset/results/e117_results.json", "r", encoding="utf-8")
    )
    e129 = json.load(
        open("experiments/E129_survey_asymmetry/results/survey_asymmetry.json", "r", encoding="utf-8")
    )
    e140 = json.load(
        open("experiments/E140_material_culture_index/results/material_culture.json", "r", encoding="utf-8")
    )

    rows = []
    for case in CASES:
        volcano, dist_km = nearest_volcano(case)
        rows.append(
            {
                **case,
                "nearest_volcano": volcano,
                "distance_km": round(dist_km, 2),
                "within_35km": dist_km <= 35.0,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "case_studies.csv", index=False)

    n_cases = len(df)
    within_35 = int(df["within_35km"].sum())
    stone_survival = int(df["stone_monument_survives"].sum())
    organic_survival = int(df["organic_settlement_visible"].sum())
    mean_distance = float(df["distance_km"].mean())

    status = "SUCCESS" if within_35 == n_cases and stone_survival == n_cases and organic_survival == 0 else "MIXED"

    garahan_excerpt = (
        "OV 1921: grave between Garahan and Mrawan was covered by ~20 cm earth, ~18 cm ash/sand, "
        "then humus; the ash layer was attributed to nearby Raung."
    )
    pasemah_excerpt = (
        "OV 1922: Pasemah antiquities were described as lying at the foot of volcano Dempo in South Sumatra."
    )

    results = {
        "experiment": "E151_megalithic_volcanic_zones",
        "title": "Megalithic Distribution vs Volcanic Zones",
        "date": "2026-03-30",
        "status": status,
        "sample": {
            "n_cases": n_cases,
            "sites": df["site"].tolist(),
            "mean_distance_km": mean_distance,
            "max_distance_km": float(df["distance_km"].max()),
            "within_35km_count": within_35,
        },
        "core_asymmetry": {
            "stone_monuments_survive": f"{stone_survival}/{n_cases}",
            "organic_settlement_visible": f"{organic_survival}/{n_cases}",
            "interpretation": (
                "All four megalithic case studies remain visible as stone monuments in volcanic landscapes; "
                "none yields a comparably visible organic/domestic settlement package."
            ),
        },
        "cross_experiment_context": {
            "E117": {
                "pre_400ce_predicted_depth_m": e117["pre_400ce_predicted_depth_m"],
                "verdict": e117["verdict"],
            },
            "E129": {
                "temple_monument_pct": e129["temple_monument_pct"],
                "settlement_pct": e129["settlement_pct"],
                "conclusion": e129["conclusion"],
            },
            "E140": {
                "organic_pct": e140["organic_pct"],
                "key_finding": e140["key_finding"],
            },
        },
        "taphonomic_support": {
            "Garahan_ash_layer": garahan_excerpt,
            "Pasemah_Dempo_context": pasemah_excerpt,
        },
        "verdict": (
            "Megaliths do not refute VOLCARCH. They specify the exception class: stone, monumental, mortuary, "
            "often upland contexts survive. The missing record is lowland, domestic, organic settlement archaeology."
        ),
        "implications": [
            "The claim should be refined from 'pre-Hindu evidence is absent' to 'organic settlement-scale evidence is absent.'",
            "Visible megaliths are exactly what a volcanic taphonomic model predicts should survive best: stone and monumental features.",
            "Bondowoso/Garahan shows that even megalithic contexts can sit beneath ash horizons while the stone marker survives.",
            "Pasemah generalizes the same logic beyond Java: volcanic highlands preserve sculpture better than domestic life.",
        ],
        "limitations": [
            "This is a curated case-study test, not a complete Indonesian megalith database.",
            "Bondowoso and Pasemah use cluster/highland proxy coordinates because exact monument coordinates are not yet cataloged in repo.",
            "Organic absence is archaeological visibility, not proof of true absence.",
        ],
    }

    with open(OUT_DIR / "e151_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"\nCases analyzed: {n_cases}")
    print(f"Mean distance to nearest volcano: {mean_distance:.2f} km")
    print(f"Within 35 km of active volcano: {within_35}/{n_cases}")
    print(f"Stone monuments visible: {stone_survival}/{n_cases}")
    print(f"Organic settlement visible: {organic_survival}/{n_cases}")
    print(f"E129 context: monuments={e129['temple_monument_pct']:.1f}% vs settlements={e129['settlement_pct']:.2f}%")
    print(f"E140 context: organic material mentions={e140['organic_pct']:.1f}%")
    print(f"\nE151 COMPLETE - Status: {status}")


if __name__ == "__main__":
    main()
