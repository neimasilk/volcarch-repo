"""
E119: The VOLCARCH Synthesis Figure

A single figure that tells the entire story:
- X-axis: time (centuries CE/BCE)
- Y-axis: predicted burial depth (meters)
- Overlay: detection horizons for different survey methods
- Data points: all known Java archaeological sites by type and age
- Shaded regions: "invisible zone" (beyond all current methods)

This is the visual Michelson-Morley: it shows WHY we can't see pre-400 CE
open-air sites, and WHAT would be needed to find them.

Output: ASCII summary table (no matplotlib on mudik laptop) + JSON data
for later rendering with matplotlib post-mudik.
"""

import json
import csv
import sys
import io
from pathlib import Path
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("E119: THE VOLCARCH SYNTHESIS FIGURE")
    print("One figure that tells the entire story")
    print("=" * 70)

    # Sedimentation rate
    rate_mm_yr = 4.0  # mm/yr best estimate from E083

    # Time axis: from 4000 BCE to 2000 CE
    centuries = list(range(-4000, 2100, 100))

    # For each century, calculate predicted burial depth
    depth_data = []
    for century_ce in centuries:
        age_yr = 2026 - century_ce
        depth_m = age_yr * rate_mm_yr / 1000
        depth_data.append({
            "century_ce": century_ce,
            "age_yr": age_yr,
            "depth_m": round(depth_m, 2),
        })

    # Detection horizons (horizontal lines)
    detection_methods = {
        "Surface survey": {"depth_m": 0.5, "color": "green", "label": "Surface survey (0.5m)"},
        "Shallow excavation": {"depth_m": 1.0, "color": "yellow", "label": "Shallow excavation (1m)"},
        "Standard excavation": {"depth_m": 2.0, "color": "orange", "label": "Standard excavation (2m)"},
        "GPR": {"depth_m": 5.0, "color": "red", "label": "GPR (5m effective in volcanic ash)"},
        "Deep coring": {"depth_m": 10.0, "color": "darkred", "label": "Deep coring (10m)"},
    }

    # Known Java sites with their ages and types
    # From E071 + mini-NusaRC
    known_sites = [
        # Caves (immune to volcanic burial — plotted at depth 0)
        {"name": "Song Terus", "century_ce": -98050, "type": "cave", "depth": 0, "note": "100ka"},
        {"name": "Gua Braholo", "century_ce": -28050, "type": "cave", "depth": 0, "note": "30ka"},
        {"name": "Wajak", "century_ce": -26050, "type": "cave", "depth": 0, "note": "28ka"},
        {"name": "Song Keplek", "century_ce": -7050, "type": "cave", "depth": 0, "note": "9ka"},
        # River terraces (exposed by erosion — plotted at depth 0)
        {"name": "Sangiran", "century_ce": -1598050, "type": "river_terrace", "depth": 0, "note": "1.6Ma"},
        {"name": "Trinil", "century_ce": -498050, "type": "river_terrace", "depth": 0, "note": "500ka"},
        # Coastal (outside volcanic zone)
        {"name": "Buni Complex", "century_ce": -100, "type": "coastal", "depth": 0, "note": "200 BCE"},
        {"name": "Batujaya", "century_ce": 400, "type": "coastal", "depth": 0, "note": "4th c CE"},
        # Open-air non-volcanic
        {"name": "Kendeng Lembu", "century_ce": -2000, "type": "open_air_nonvolcanic", "depth": 0, "note": "karst"},
        # Hindu-Buddhist temples (found because monumental stone)
        {"name": "Dieng", "century_ce": 750, "type": "temple_found", "depth": 2.5, "note": "buried 2.5m"},
        {"name": "Borobudur", "century_ce": 800, "type": "temple_found", "depth": 0, "note": "exposed"},
        {"name": "Prambanan", "century_ce": 850, "type": "temple_found", "depth": 5.5, "note": "partially buried"},
        {"name": "Sambisari", "century_ce": 850, "type": "temple_found", "depth": 6.5, "note": "fully buried"},
        {"name": "Kedulan", "century_ce": 900, "type": "temple_found", "depth": 6.0, "note": "fully buried"},
        {"name": "Singosari Dwarapala", "century_ce": 1268, "type": "temple_found", "depth": 1.85, "note": "half buried"},
        {"name": "Trowulan/Majapahit", "century_ce": 1350, "type": "temple_found", "depth": 4.3, "note": "buried"},
        # Pre-400 CE volcanic interior: NOTHING
    ]

    # Print the synthesis table
    print(f"\n  THE BURIAL DEPTH TIMELINE (at 4.0 mm/yr)")
    print(f"\n  {'Century':<12} {'Predicted Depth':<18} {'Detectable By':<30} {'Known Sites'}")
    print("  " + "-" * 90)

    key_centuries = [-4000, -3000, -2000, -1000, -500, 0, 400, 700, 900, 1000, 1200, 1400, 1600, 1800, 2000]

    for c in key_centuries:
        age = 2026 - c
        depth = age * rate_mm_yr / 1000

        # Which methods can detect at this depth?
        methods = []
        for name, m in detection_methods.items():
            if depth <= m["depth_m"]:
                methods.append(name)
        method_str = methods[0] if methods else "NONE (beyond all methods)"

        # Which sites are from this century?
        sites_here = [s for s in known_sites if abs(s["century_ce"] - c) < 150 and s["century_ce"] > -10000]
        site_str = ", ".join(f"{s['name']}" for s in sites_here) if sites_here else "—"

        ce_label = f"{c} CE" if c >= 0 else f"{abs(c)} BCE"
        print(f"  {ce_label:<12} {depth:<18.1f}m {method_str:<30} {site_str}")

    # Print the key insight
    print(f"""
  KEY INSIGHT:

  The diagonal line (burial depth increasing with age) intersects each
  detection method at a specific century:

    Surface survey (0.5m):     only reaches ~1900 CE
    Shallow excavation (1m):   only reaches ~1776 CE
    Standard excavation (2m):  only reaches ~1526 CE
    GPR (5m):                  only reaches ~776 CE
    Deep coring (10m):         only reaches ~474 BCE

  Everything below the diagonal's intersection with the method being
  used is IN THE INVISIBLE ZONE.

  OBSERVATION: All pre-400 CE sites in Java are either:
    - In caves (immune to burial)
    - On river terraces (exposed by erosion)
    - On the coast (outside volcanic zone)
    - In non-volcanic terrain (Kendeng Lembu karst)

  ZERO open-air sites in volcanic interior Java pre-400 CE.

  This is either taphonomic burial OR genuine absence.
  The figure shows WHY we can't tell which — and WHAT would resolve it:
  GPR or deep coring to 6.5m+ at E080's top 20 target locations.
    """)

    # Save figure data for later matplotlib rendering
    figure_data = {
        "experiment": "E119_synthesis_figure",
        "date": "2026-03-20",
        "sedimentation_rate_mm_yr": rate_mm_yr,
        "burial_depth_timeline": depth_data,
        "detection_methods": detection_methods,
        "known_sites": [s for s in known_sites if s["century_ce"] > -10000],
        "key_intersections": {
            name: {
                "depth_m": m["depth_m"],
                "oldest_detectable_ce": int(2026 - m["depth_m"] * 1000 / rate_mm_yr),
            }
            for name, m in detection_methods.items()
        },
        "figure_description": (
            "X-axis: time (centuries CE/BCE). "
            "Y-axis: predicted burial depth at 4mm/yr (meters, increasing downward). "
            "Diagonal line: burial depth as function of age. "
            "Horizontal lines: detection limits for each survey method. "
            "Symbols: known Java sites (circles=caves at depth 0, triangles=temples at observed depth, "
            "squares=coastal sites at depth 0). "
            "Shaded region below diagonal: INVISIBLE ZONE. "
            "The figure shows why pre-400 CE open-air sites cannot be found by any current method."
        ),
        "rendering_instructions": (
            "Use matplotlib with inverted Y-axis (depth increases downward). "
            "Diagonal from top-left to bottom-right = burial depth vs time. "
            "Horizontal dashed lines = detection limits. "
            "Color zones: green (detectable), yellow (marginal), red (invisible). "
            "Mark known sites with different symbols per type. "
            "Highlight the 'invisible zone' with gray shading."
        ),
    }

    with open(OUT / "e119_figure_data.json", "w", encoding="utf-8") as f:
        json.dump(figure_data, f, indent=2, ensure_ascii=False)
    print(f"  Figure data saved: {OUT / 'e119_figure_data.json'}")
    print(f"  Render with matplotlib post-mudik (no GPU needed, just matplotlib)")


if __name__ == "__main__":
    main()
