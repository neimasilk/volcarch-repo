"""
E004 RE-RUN (integrity remediation, 2026-06-08): P1 §spatial distance-band site
density vs volcanic proximity, recomputed with the CANONICAL 30-volcano inventory
instead of the hardcoded 7 (which omits Lawu, Wilis, all Central Java).

Uses bounding-box study area (no OSM dependency) for reproducibility. Reports the
Spearman(density, distance) for 7-volcano vs 30-volcano so we know if P1's
descriptive spatial claim survives.

Run: python experiments/E004_density_analysis/02_rerun_full_inventory.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from scipy.stats import spearmanr

REPO = Path(__file__).parent.parent.parent
SITES = REPO / "data" / "processed" / "east_java_sites.geojson"
VFULL = REPO / "data" / "processed" / "dashboard" / "volcanoes_java_full.csv"

BIN_EDGES = [0, 25, 50, 75, 100, 150, 200, 10000]
BIN_LABELS = ["0-25","25-50","50-75","75-100","100-150","150-200","200+"]
MIDPTS = [12.5,37.5,62.5,87.5,125,175,250]

V7 = pd.DataFrame([  # the legacy hardcoded set
    ("Kelud",-7.93,112.308),("Semeru",-8.108,112.922),("Arjuno-Welirang",-7.729,112.575),
    ("Bromo",-7.942,112.95),("Lamongan",-7.977,113.343),("Raung",-8.125,114.042),("Ijen",-8.058,114.242),
], columns=["name","lat","lon"])
V30 = pd.read_csv(VFULL)

sites = gpd.read_file(SITES)
sites = sites[sites.geometry.notna() & ~sites.geometry.is_empty].to_crs("EPSG:4326")
b = (-9.0,111.0,-6.5,115.0)
sites = sites[(sites.geometry.y>=b[0])&(sites.geometry.x>=b[1])&(sites.geometry.y<=b[2])&(sites.geometry.x<=b[3])]
sites_p = sites.to_crs("EPSG:32749")
study = box(4.9e5,9.06e6,8.8e5,9.26e6)  # East Java bbox, UTM49S
print(f"Sites in East Java bbox: {len(sites)}")

def analyze(vdf, tag):
    vgdf = gpd.GeoDataFrame(vdf, geometry=gpd.points_from_xy(vdf.lon, vdf.lat), crs="EPSG:4326").to_crs("EPSG:32749")
    # min distance per site
    d = sites_p.geometry.apply(lambda g: vgdf.geometry.distance(g).min()/1000)
    dist_bin = pd.cut(d, bins=BIN_EDGES, labels=BIN_LABELS, right=False)
    counts = dist_bin.value_counts().reindex(BIN_LABELS, fill_value=0)
    # area per bin (buffers around volcano union, clipped to study bbox)
    vu = unary_union(vgdf.geometry.values)
    edges_m = [e*1000 for e in BIN_EDGES]
    areas=[]
    for i in range(len(BIN_LABELS)):
        outer = vu.buffer(edges_m[i+1]); inner = vu.buffer(edges_m[i]) if edges_m[i]>0 else None
        ring = outer if inner is None else outer.difference(inner)
        areas.append(ring.intersection(study).area/1e6)
    dens = np.where(np.array(areas)>0, counts.values/np.array(areas)*1000, 0)
    mask = np.array(areas)>0
    rho,p = spearmanr(np.array(MIDPTS)[mask], dens[mask])
    print(f"\n=== {tag} ({len(vdf)} volcanoes) ===")
    df = pd.DataFrame({"band":BIN_LABELS,"sites":counts.values,"area_km2":np.round(areas).astype(int),"dens/1000km2":np.round(dens,3)})
    print(df.to_string(index=False))
    print(f"Spearman(density, distance) rho={rho:.3f}, p={p:.4f}  -> "
          f"{'positive: more sites FARTHER (supports H1)' if rho>0 and p<0.05 else 'NOT significant' if p>=0.05 else 'negative'}")
    return rho,p

r7 = analyze(V7, "OLD 7-volcano (P1 as-published)")
r30 = analyze(V30, "NEW 30-volcano (canonical)")
print(f"\nSUMMARY: rho 7-volc={r7[0]:.3f} (p={r7[1]:.3f})  vs  30-volc={r30[0]:.3f} (p={r30[1]:.3f})")
