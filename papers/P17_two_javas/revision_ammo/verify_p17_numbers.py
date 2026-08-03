#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WS-E integrity sweep, paper P17 ("Two Javas", ArchCalc #365, under review).

Every headline number in P17 is a function of "distance to the nearest volcano", and the
submitted manuscript computes that against a hand-picked list of **10** volcanoes
(draft_v0.3_archcalc.tex l.169). The canonical inventory
`data/processed/dashboard/volcanoes_java_full.csv` has **30**. That is the same class of
defect that sank P7 at Antiquity, so every number is re-derived here on the canonical
inventory and reported next to the published one.

E104's clean rebuild (2026-06-08) already re-derived the two medians. It did NOT re-derive
the zone distribution, the Fisher odds ratio, or the Mann-Whitney U, and the distribution
block in `e104_court_zone.json` still carries `candi: 0` from the original non-reproducible
run. This script closes that gap.

Run from the repo root:  python papers/P17_two_javas/revision_ammo/verify_p17_numbers.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

REPO = Path(__file__).resolve().parents[3]
DASH = REPO / "data" / "processed" / "dashboard"

# The 10 centres named in the submitted manuscript (l.110 and l.169).
# "Sindoro" is the Indonesian spelling; the canonical file uses the Smithsonian GVP
# form "Sundoro". Same volcano - the alias is needed or the match silently drops it.
PUBLISHED_10 = ["Merapi", "Kelud", "Arjuno", "Semeru", "Bromo",
                "Penanggungan", "Lawu", "Merbabu", "Sundoro", "Sumbing"]
ALIASES = {"Sundoro": "Sindoro"}

# Published headline values, read out of draft_v0.3_archcalc.tex.
PUBLISHED = {
    "candi_median": 14.6, "ins_median": 27.6, "gap": 13.0,
    "candi_peak_share": 42.3, "ins_peak_share": 39.2,
    "mw_U": 8081.0, "fisher_or": 1.86, "fisher_p": 0.012,
    "n_candi": 142, "n_ins": 176,
}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def nearest(lat, lon, vdf):
    d = np.full(len(lat), np.inf)
    for _, v in vdf.iterrows():
        d = np.minimum(d, haversine_km(np.asarray(lat), np.asarray(lon), v["lat"], v["lon"]))
    return d


def java_only(df):
    """E104's rebuild filter: Java proper, excluding Sumatra, Bali and outliers."""
    return df[df.lat.between(-8.9, -5.8) & df.lon.between(105.0, 114.8)].copy()


def zone_table(d, edges=((0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 1e9))):
    return {f"{a}-{b if b < 1e8 else 'inf'}km": int(((d >= a) & (d < b)).sum()) for a, b in edges}


def main() -> None:
    candi = pd.read_csv(REPO / "experiments/E031_candi_orientation/results/"
                               "candi_volcano_pairs.csv")[["name", "lat", "lon"]]
    ins = pd.read_csv(REPO / "experiments/E082_inscription_georeferencing/results/"
                             "geocoded_inscriptions.csv")
    volc_all = pd.read_csv(DASH / "volcanoes_java_full.csv")

    # the manuscript's 10, matched by prefix so "Arjuno-Welirang" resolves to "Arjuno"
    mask = volc_all["name"].apply(
        lambda n: any(n.lower().startswith(p.lower()) or p.lower() in n.lower()
                      for p in PUBLISHED_10))
    volc_10 = volc_all[mask]

    c, i = java_only(candi), java_only(ins)
    print("=" * 78)
    print("WS-E / P17 - headline numbers on the published vs the canonical inventory")
    print("=" * 78)
    print(f"  candi (Java): {len(c)}   inscriptions (Java): {len(i)}")
    print(f"  manuscript's 10 centres matched in the canonical file: {len(volc_10)} "
          f"-> {sorted(volc_10.name.tolist())}")
    missing = [p for p in PUBLISHED_10
               if not any(p.lower() in n.lower() for n in volc_10.name)]
    if missing:
        print(f"  !! named in the manuscript but absent from the canonical file: {missing}")
    print(f"  canonical inventory: {len(volc_all)} centres\n")

    rows = []
    for tag, vdf in (("published_10", volc_10), ("canonical_30", volc_all)):
        dc, di = nearest(c.lat.values, c.lon.values, vdf), nearest(i.lat.values, i.lon.values, vdf)
        u, p = mannwhitneyu(dc, di, alternative="two-sided")
        zc, zi = zone_table(dc), zone_table(di)
        # volcano zone (0-20 km) vs court zone (20-40 km), as in the manuscript
        cv, cc = int(((dc >= 0) & (dc < 20)).sum()), int(((dc >= 20) & (dc < 40)).sum())
        iv, ic = int(((di >= 0) & (di < 20)).sum()), int(((di >= 20) & (di < 40)).sum())
        # fisher_exact([[candi_volcano, candi_court], [ins_volcano, ins_court]]) returns
        # odds(court:volcano | inscriptions) / odds(court:volcano | candi) -- i.e. exactly
        # the manuscript's "inscriptions are N times more court-concentrated than candi".
        orat, fp = fisher_exact([[cv, cc], [iv, ic]])
        rows.append(dict(
            inventory=tag, n_volcanoes=len(vdf),
            candi_median=float(np.median(dc)), ins_median=float(np.median(di)),
            gap=float(np.median(di) - np.median(dc)),
            mw_U=float(u), mw_p=float(p),
            candi_peak_zone=max(zc, key=zc.get), candi_peak_share=100 * max(zc.values()) / len(dc),
            ins_peak_zone=max(zi, key=zi.get), ins_peak_share=100 * max(zi.values()) / len(di),
            fisher_or=float(orat), fisher_p=float(fp),
            zones_candi=zc, zones_ins=zi,
            candi_volc=cv, candi_court=cc, ins_volc=iv, ins_court=ic))

    df = pd.DataFrame(rows)
    out_dir = Path(__file__).parent
    df.drop(columns=["zones_candi", "zones_ins"]).to_csv(
        out_dir / "p17_inventory_comparison.csv", index=False)

    for r in rows:
        print(f"--- {r['inventory']} ({r['n_volcanoes']} centres)")
        print(f"    candi median        {r['candi_median']:6.1f} km   "
              f"(published {PUBLISHED['candi_median']})")
        print(f"    inscription median  {r['ins_median']:6.1f} km   "
              f"(published {PUBLISHED['ins_median']})")
        print(f"    median gap          {r['gap']:6.1f} km   (published {PUBLISHED['gap']})")
        print(f"    Mann-Whitney        U={r['mw_U']:.0f}  p={r['mw_p']:.3e}   "
              f"(published U={PUBLISHED['mw_U']:.0f}, p<1e-6)")
        print(f"    candi peak zone     {r['candi_peak_zone']} at {r['candi_peak_share']:.1f}%  "
              f"(published 0-10km at {PUBLISHED['candi_peak_share']}%)")
        print(f"    inscr. peak zone    {r['ins_peak_zone']} at {r['ins_peak_share']:.1f}%  "
              f"(published 20-30km at {PUBLISHED['ins_peak_share']}%)")
        print(f"    Fisher court conc.  OR={r['fisher_or']:.2f}  p={r['fisher_p']:.4f}   "
              f"(published OR={PUBLISHED['fisher_or']}, p={PUBLISHED['fisher_p']})")
        print(f"    zones candi         {r['zones_candi']}")
        print(f"    zones inscriptions  {r['zones_ins']}\n")

    a, b = rows[0], rows[1]
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    surv = (b["candi_median"] < b["ins_median"]) and b["mw_p"] < 0.05
    print(f"  segregation on the canonical inventory : "
          f"{'SURVIVES' if surv else 'DOES NOT SURVIVE'}")
    print(f"  median gap {a['gap']:.1f} km -> {b['gap']:.1f} km "
          f"({b['gap'] - a['gap']:+.1f} km when the omitted centres are restored)")
    print(f"  court concentration OR {a['fisher_or']:.2f} (p={a['fisher_p']:.4f}) -> "
          f"{b['fisher_or']:.2f} (p={b['fisher_p']:.4f})")
    print(f"\n  -> {out_dir / 'p17_inventory_comparison.csv'}")


if __name__ == "__main__":
    main()
