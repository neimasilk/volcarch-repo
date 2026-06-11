"""
E104 CLEAN REBUILD (2026-06-08) — integrity remediation after P7/Antiquity audit.

The original E104 (a) saved candi:0 in its result JSON (headline not reproducible),
(b) compared candi distances (9-volcano list) vs inscription distances (15-volcano
list) -- apples to oranges, (c) did not consistently restrict inscriptions to Java.

This rebuild: BOTH groups, ONE canonical 30-volcano inventory, explicit Java filter.
Reports whether the candi-vs-inscription "spatial segregation" (P17's core claim)
survives. Also a region-matched version (lon>=111, where candi exist) for fairness.

Run: python experiments/E104_court_zone_hypothesis/rebuild_clean_full_inventory.py
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
        d = np.minimum(d, haversine_km(np.asarray(lat), np.asarray(lon), v["lat"], v["lon"]))
    return d

# --- Load ---
candi = pd.read_csv(REPO / "experiments/E031_candi_orientation/results/candi_volcano_pairs.csv")[["name","lat","lon"]].copy()
ins = pd.read_csv(REPO / "experiments/E082_inscription_georeferencing/results/geocoded_inscriptions.csv")
volc = pd.read_csv(DASH / "volcanoes_java_full.csv")

# --- Java filter (exclude Sumatra lat>-5.8, Bali/Lombok lon>114.8, and far outliers) ---
def java_only(df):
    return df[(df.lat.between(-8.9, -5.8)) & (df.lon.between(105.0, 114.8))].copy()

candi_j = java_only(candi)
ins_j = java_only(ins)
print(f"Candi: {len(candi)} -> {len(candi_j)} Java | Inscriptions: {len(ins)} -> {len(ins_j)} Java")
print(f"Inventory: {len(volc)} volcanoes\n")

candi_j["d"] = nearest(candi_j.lat.values, candi_j.lon.values, volc)
ins_j["d"]   = nearest(ins_j.lat.values,   ins_j.lon.values,   volc)

def report(c, i, tag):
    u, p = mannwhitneyu(c, i, alternative="two-sided")
    print(f"=== {tag} ===")
    print(f"  Candi:        median={np.median(c):5.1f} km  mean={c.mean():5.1f}  (n={len(c)})")
    print(f"  Inscriptions: median={np.median(i):5.1f} km  mean={i.mean():5.1f}  (n={len(i)})")
    surv = "SURVIVES (candi closer, p<0.05)" if (np.median(c) < np.median(i) and p < 0.05) else \
           "DOES NOT survive" if p >= 0.05 else "REVERSED (candi farther!)"
    print(f"  Mann-Whitney p={p:.2e}  ->  segregation {surv}")
    # distance bands
    bands = [(0,10),(10,20),(20,30),(30,40),(40,60),(60,100)]
    print(f"  {'band':<10}{'candi%':>8}{'inscr%':>8}")
    for lo,hi in bands:
        cp = 100*((c>=lo)&(c<hi)).sum()/len(c)
        ip = 100*((i>=lo)&(i<hi)).sum()/len(i)
        print(f"  {f'{lo}-{hi}km':<10}{cp:7.1f}%{ip:7.1f}%")
    print()

# Full Java
report(candi_j["d"].values, ins_j["d"].values, "FULL JAVA (canonical 30-volcano inventory)")

# Region-matched: both lon>=111 (where candi actually exist) -- fair within-region test
cm = candi_j[candi_j.lon >= 111.0]["d"].values
im = ins_j[ins_j.lon >= 111.0]["d"].values
report(cm, im, "REGION-MATCHED (lon>=111, fair comparison)")

print("Original P17/E104 claim: candi median 14.6 km vs inscriptions 27.6 km (p<1e-6)")
