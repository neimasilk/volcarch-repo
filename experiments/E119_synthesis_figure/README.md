# E119: The VOLCARCH Synthesis Figure

**Status:** SUCCESS (data ready, rendering post-mudik)
**Date:** 2026-03-20
**Papers:** P1, All
**Depends on:** E083 (burial depths), E110 (cascade), E117 (detection horizon), E071 (pre-400 CE sites)

---

## Purpose

Create a single figure that tells the entire VOLCARCH story: burial depth vs. time, overlaid with detection horizons for different survey methods and the locations of all known Java archaeological sites by type.

## The Figure Concept

- **X-axis:** Time (centuries CE/BCE)
- **Y-axis:** Predicted burial depth at 4 mm/yr sedimentation (meters, increasing downward)
- **Diagonal line:** Burial depth = age × sedimentation rate
- **Horizontal lines:** Detection limits (surface 0.5m, excavation 1-2m, GPR 5m, coring 10m)
- **Symbols:** Known sites by type (caves at depth=0, temples at observed depth, coastal at depth=0)
- **Shaded region:** "Invisible zone" — below the diagonal, beyond current methods

## Key Insight

The diagonal line intersects each detection method at a specific century:

| Method | Depth Limit | Oldest Detectable |
|--------|-------------|-------------------|
| Surface survey | 0.5m | ~1900 CE |
| Standard excavation | 2.0m | ~1526 CE |
| GPR | 5.0m | ~776 CE |
| Deep coring | 10.0m | ~474 BCE |

Pre-400 CE open-air sites at 6.5m+ depth are beyond GPR and require deep coring. All known pre-400 CE Java sites are in taphonomically protected contexts (caves, river terraces, coastal, non-volcanic terrain). Zero in volcanic interior.

## Status

Data generated as JSON for post-mudik matplotlib rendering. ASCII table version demonstrates the concept. No GPU needed — just matplotlib.

## Files

- `synthesis_figure.py` — Data generation script
- `results/e119_figure_data.json` — All data needed to render the figure
