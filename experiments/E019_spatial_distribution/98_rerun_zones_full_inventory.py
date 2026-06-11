"""
E019 RE-ANALYSIS (post-hoc, integrity audit): re-run the Zone A-vs-B-vs-C
distance test using a COMPLETE Java volcano inventory instead of the 7-volcano
list that produced the published Cohen's d = 1.005.

Triggered by Antiquity AQY-2026-0104 Reviewer 2: the original volcanoes.csv
omits Lawu, Wilis, Kawi-Butak, Penanggungan, Argopuro, Baluran, and every
Central Java volcano (Merapi/Merbabu/Lawu/Muria) -- yet the grid extends west
to lon ~110.9 (Central Java). Western grid cells were therefore measured to
far-eastern volcanoes. This script measures whether the headline result survives.

Run: python experiments/E019_spatial_distribution/98_rerun_zones_full_inventory.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

REPO = Path(__file__).parent.parent.parent
DASH = REPO / "data" / "processed" / "dashboard"

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def nearest(lat, lon, vdf):
    d = np.full(len(lat), np.inf)
    for _, v in vdf.iterrows():
        d = np.minimum(d, haversine_km(lat, lon, v["lat"], v["lon"]))
    return d

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2)/(nx+ny-2))
    return (np.mean(x) - np.mean(y)) / sp if sp else 0.0

grid = pd.read_csv(DASH / "grid_predictions.csv")
v_old = pd.read_csv(DASH / "volcanoes.csv")          # 7 volcanoes (published)
v_new = pd.read_csv(DASH / "volcanoes_java_full.csv") # 30 volcanoes (complete)

print(f"Grid cells: {len(grid):,} | lon range {grid.lon.min():.2f}-{grid.lon.max():.2f} "
      f"| lat range {grid.lat.min():.2f}-{grid.lat.max():.2f}")
print(f"Old inventory: {len(v_old)} volcanoes | New inventory: {len(v_new)} volcanoes\n")

for label, vdf in [("OLD (7 volcanoes, published)", v_old), ("NEW (30 volcanoes, full)", v_new)]:
    grid["d"] = nearest(grid.lat.values, grid.lon.values, vdf)
    a = grid.loc[grid.zone == "A", "d"].values
    b = grid.loc[grid.zone == "B", "d"].values
    c = grid.loc[grid.zone == "C", "d"].values
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    d = cohens_d(a, b)
    print(f"=== {label} ===")
    print(f"  Zone A: median={np.median(a):5.1f} km  mean={a.mean():5.1f}  (n={len(a):,})")
    print(f"  Zone B: median={np.median(b):5.1f} km  mean={b.mean():5.1f}  (n={len(b):,})")
    print(f"  Zone C: median={np.median(c):5.1f} km  mean={c.mean():5.1f}  (n={len(c):,})")
    direction = "B CLOSER than A (supports H-TOM)" if np.median(b) < np.median(a) else "B FARTHER than A (COUNTER)"
    print(f"  A vs B: p={p:.2e}  Cohen's d={d:.3f}  -> {direction}\n")
