"""
E019 VERIFICATION (post-hoc): Re-test the deep-time site distance claim against
the Antiquity (AQY-2026-0104) Reviewer 2 critique.

Reviewer 2 asserted the four deep-time sites are NEAR volcanoes (30-70 km), not
"90-170 km from the nearest volcanic centre" as the P7 submission claimed.

The original analysis (01_site_volcano_distance.py) used volcanoes.csv, which
contains ONLY 7 eastern-East-Java volcanoes (lon 112.3-114.2). Lawu and Wilis --
the volcanoes actually nearest these western sites -- and all Central Java
volcanoes (Sangiran is in Central Java) were absent.

This script recomputes nearest-volcano distance for the 4 sites using (a) the
original 7-volcano list and (b) a fuller Holocene/active Java inventory, to
measure the size of the artifact. Honest self-audit per CLAUDE.md integrity rule.

Run: python experiments/E019_spatial_distribution/99_verify_reviewer_distance_critique.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# The four deep-time sites (from E019 data/deep_time_sites.csv)
sites = [
    ("Song Terus", -8.017, 110.917, "cave (Gunung Sewu karst, East Java)"),
    ("Trinil",     -7.374, 111.358, "Solo R. terrace, Ngawi (East Java)"),
    ("Sangiran",   -7.450, 110.850, "dome eroded by Cemoro R. (CENTRAL Java)"),
    ("Wajak",      -8.033, 111.500, "cave, Tulungagung (East Java)"),
]

# ORIGINAL 7-volcano list actually used by E019 (data/processed/dashboard/volcanoes.csv)
volc_original = [
    ("Kelud", -7.93, 112.308), ("Semeru", -8.108, 112.922),
    ("Arjuno-Welirang", -7.729, 112.575), ("Bromo", -7.942, 112.95),
    ("Lamongan", -7.977, 113.343), ("Raung", -8.125, 114.042),
    ("Ijen", -8.058, 114.242),
]

# Fuller inventory of Holocene / Pleistocene-active Java volcanoes (Smithsonian GVP
# coordinates). The crucial omissions vs the original list are Lawu and Wilis.
volc_full = volc_original + [
    ("Lawu", -7.625, 111.192),        # nearest to Trinil / Song Terus
    ("Wilis", -7.808, 111.758),       # large volcano near Trinil / Wajak
    ("Kawi-Butak", -7.92, 112.45),
    ("Penanggungan", -7.62, 112.63),
    ("Argopuro", -7.97, 113.57),
    # Central Java (Sangiran sits among these):
    ("Merapi", -7.541, 110.446),
    ("Merbabu", -7.45, 110.43),
    ("Telomoyo", -7.37, 110.40),
    ("Ungaran", -7.18, 110.33),
    ("Muria", -6.62, 110.88),
    ("Sumbing", -7.384, 110.07),
    ("Sundoro", -7.30, 109.992),
    ("Dieng", -7.20, 109.92),
]

def nearest(site, vlist):
    best_d, best_n = np.inf, None
    for n, vlat, vlon in vlist:
        d = haversine_km(site[1], site[2], vlat, vlon)
        if d < best_d:
            best_d, best_n = d, n
    return best_d, best_n

print("="*78)
print("DEEP-TIME SITE -> NEAREST VOLCANO: original 7-volcano list vs full inventory")
print("="*78)
print(f"{'Site':<12}{'7-volcano list':<28}{'Full inventory':<28}{'factor'}")
print("-"*78)
rows = []
for s in sites:
    d_old, n_old = nearest(s, volc_original)
    d_new, n_new = nearest(s, volc_full)
    factor = d_old / d_new
    print(f"{s[0]:<12}{f'{d_old:5.0f} km ({n_old})':<28}{f'{d_new:5.0f} km ({n_new})':<28}{factor:4.1f}x")
    rows.append(dict(site=s[0], context=s[3], dist_7volc_km=round(d_old,1),
                     nearest_7volc=n_old, dist_full_km=round(d_new,1),
                     nearest_full=n_new, inflation_factor=round(factor,1)))

print("-"*78)
print(f"P7 submission claimed: 'All four ... 90-170 km from nearest volcanic centre'")
print(f"Reviewer 2 claimed:    Sangiran ~50-60 km; Song Terus ~60-70 km; Trinil ~30-40 km")
print("="*78)

out = Path(__file__).parent / "results" / "verify_distance_critique.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Saved: {out}")
