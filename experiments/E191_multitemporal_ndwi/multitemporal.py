#!/usr/bin/env python3
"""
E191: Multi-temporal NDWI — Dry vs Wet Season at Candi Sites
==============================================================
E189: NDWI p=0.032 in dry season. Does wet season amplify the signal?
Buried structures impede infiltration → larger NDWI anomaly when water table high.
"""

import json, csv, sys, numpy as np, requests
from pathlib import Path
from scipy import stats as sp_stats
from scipy.ndimage import zoom

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

KNOWN_CANDI = [
    ("Candi Jawi", -7.7619, 112.7281),
    ("Candi Tikus", -7.5589, 112.3756),
    ("Candi Brahu", -7.5481, 112.3856),
    ("Candi Bajang Ratu", -7.5561, 112.3978),
    ("Candi Kidal", -8.0156, 112.5928),
    ("Candi Singosari", -7.8894, 112.6592),
    ("Candi Sumberawan", -7.8403, 112.5456),
    ("Candi Songgoriti", -7.8714, 112.4894),
    ("Candi Penataran", -7.9250, 112.2069),
    ("Candi Sawentar", -7.9500, 112.1806),
    ("Candi Gambar Wetan", -7.9769, 112.2597),
    ("Candi Tegowangi", -7.7733, 112.1194),
    ("Candi Surawana", -7.7756, 112.1494),
    ("Situs Trowulan", -7.5500, 112.3800),
    ("Candi Wringin Lawang", -7.5397, 112.3908),
]

CONTROLS = [
    ("Ctrl_volc_kelud", -7.95, 112.45),
    ("Ctrl_volc_arjuno", -7.80, 112.55),
    ("Ctrl_plain_north", -7.45, 112.40),
    ("Ctrl_plain_east", -7.60, 112.80),
    ("Ctrl_forest_south", -8.05, 113.00),
]


def search_s2(bbox, date_range, max_cloud=20, max_items=20):
    body = {
        "collections": ["sentinel-2-l2a"], "bbox": bbox,
        "datetime": date_range, "limit": max_items,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]
    }
    try:
        resp = requests.post(f"{STAC_URL}/search", json=body, timeout=30)
        return resp.json().get("features", []) if resp.status_code == 200 else []
    except Exception:
        return []


def get_ndwi_at_site(scenes, lat, lon, token):
    """Extract NDWI center-ring diff and local variance at a site."""
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
    cx, cy = transformer.transform(lon, lat)
    buffer_m = 500

    for scene in scenes:
        try:
            # Read B03 (green) and B08 (NIR)
            for band in ["B03", "B08"]:
                href = scene["assets"].get(band, {}).get("href")
                if not href:
                    break
            else:
                b03_href = scene["assets"]["B03"]["href"]
                b08_href = scene["assets"]["B08"]["href"]

                sign = lambda h: f"{h}&{token}" if "?" in h else f"{h}?{token}"

                with rasterio.open(sign(b03_href)) as src:
                    if not (src.bounds.left <= cx <= src.bounds.right and
                            src.bounds.bottom <= cy <= src.bounds.top):
                        continue
                    w = from_bounds(cx - buffer_m, cy - buffer_m,
                                    cx + buffer_m, cy + buffer_m, src.transform)
                    b03 = src.read(1, window=w).astype(float)
                    if np.count_nonzero(b03) / max(b03.size, 1) < 0.5:
                        continue

                with rasterio.open(sign(b08_href)) as src:
                    w = from_bounds(cx - buffer_m, cy - buffer_m,
                                    cx + buffer_m, cy + buffer_m, src.transform)
                    b08 = src.read(1, window=w).astype(float)
                    if np.count_nonzero(b08) / max(b08.size, 1) < 0.5:
                        continue

                # Resize b03 if needed
                if b03.shape != b08.shape:
                    b03 = zoom(b03, (b08.shape[0]/b03.shape[0], b08.shape[1]/b03.shape[1]), order=0)

                # NDWI
                denom = b03 + b08
                ndwi = np.where(denom > 0, (b03 - b08) / denom, np.nan)

                h, w_px = ndwi.shape
                if h < 6 or w_px < 6:
                    continue

                # Center vs ring
                ch, cw = h // 5, w_px // 5
                center = ndwi[2*ch:3*ch, 2*cw:3*cw]
                ring = np.concatenate([ndwi[:ch, :].flatten(), ndwi[-ch:, :].flatten(),
                                       ndwi[ch:-ch, :cw].flatten(), ndwi[ch:-ch, -cw:].flatten()])

                ndwi_diff = float(np.nanmean(center) - np.nanmean(ring))
                ndwi_mean = float(np.nanmean(ndwi))

                # Local variance
                lvars = []
                for i in range(1, h - 1):
                    for j in range(1, w_px - 1):
                        v = np.nanvar(ndwi[i-1:i+2, j-1:j+2])
                        if np.isfinite(v):
                            lvars.append(v)

                return {
                    "ndwi_mean": round(ndwi_mean, 5),
                    "ndwi_diff": round(ndwi_diff, 5),
                    "ndwi_lvar": round(float(np.mean(lvars)), 8) if lvars else None,
                    "scene": scene["properties"].get("datetime", "")[:10],
                }
        except Exception:
            continue
    return None


def main():
    print("=" * 70)
    print("E191: Multi-temporal NDWI — Dry vs Wet Season")
    print("=" * 70)

    all_sites = [(n, la, lo, "candi") for n, la, lo in KNOWN_CANDI] + \
                [(n, la, lo, "control") for n, la, lo in CONTROLS]

    print(f"Sites: {len(KNOWN_CANDI)} candi + {len(CONTROLS)} control")

    # Bounding box
    all_lats = [s[1] for s in all_sites]
    all_lons = [s[2] for s in all_sites]
    bbox = [min(all_lons) - 0.1, min(all_lats) - 0.1,
            max(all_lons) + 0.1, max(all_lats) + 0.1]

    # Search both seasons
    seasons = {
        "dry": "2024-07-01/2024-09-30",
        "wet": "2023-12-01/2024-02-29",  # wet season: Dec-Feb
    }

    season_scenes = {}
    for season, date_range in seasons.items():
        print(f"\nSearching {season} season ({date_range})...")
        # Wet season has more clouds — allow higher threshold
        max_cloud = 20 if season == "dry" else 40
        found = search_s2(bbox, date_range, max_cloud=max_cloud, max_items=20)
        season_scenes[season] = found
        print(f"  Found: {len(found)} scenes")
        for i, s in enumerate(found[:3]):
            p = s["properties"]
            print(f"  [{i}] {p['datetime'][:10]} cloud={p['eo:cloud_cover']:.1f}%")

    token = requests.get(SIGN_URL, timeout=10).json().get("token", "")
    print(f"\nToken: {len(token)} chars")

    # Extract NDWI per site per season
    print(f"\nExtracting NDWI for both seasons...")
    results = []

    for name, lat, lon, cat in all_sites:
        dry_data = get_ndwi_at_site(season_scenes["dry"], lat, lon, token)
        wet_data = get_ndwi_at_site(season_scenes["wet"], lat, lon, token)

        if dry_data and wet_data:
            delta_diff = wet_data["ndwi_diff"] - dry_data["ndwi_diff"]
            delta_mean = wet_data["ndwi_mean"] - dry_data["ndwi_mean"]
            delta_lvar = None
            if dry_data["ndwi_lvar"] is not None and wet_data["ndwi_lvar"] is not None:
                delta_lvar = wet_data["ndwi_lvar"] - dry_data["ndwi_lvar"]

            r = {
                "site": name, "cat": cat, "lat": lat, "lon": lon,
                "dry_ndwi_diff": dry_data["ndwi_diff"],
                "wet_ndwi_diff": wet_data["ndwi_diff"],
                "delta_diff": round(delta_diff, 5),
                "dry_ndwi_mean": dry_data["ndwi_mean"],
                "wet_ndwi_mean": wet_data["ndwi_mean"],
                "delta_mean": round(delta_mean, 5),
                "dry_lvar": dry_data["ndwi_lvar"],
                "wet_lvar": wet_data["ndwi_lvar"],
                "delta_lvar": round(delta_lvar, 8) if delta_lvar is not None else None,
                "dry_scene": dry_data["scene"],
                "wet_scene": wet_data["scene"],
            }
            results.append(r)
            print(f"  [{cat:7s}] {name:25s} dry={dry_data['ndwi_diff']:+.5f} "
                  f"wet={wet_data['ndwi_diff']:+.5f} delta={delta_diff:+.5f}")
        elif dry_data:
            print(f"  [{cat:7s}] {name:25s} dry only")
        elif wet_data:
            print(f"  [{cat:7s}] {name:25s} wet only")
        else:
            print(f"  [{cat:7s}] {name:25s} SKIP")

    # Analysis
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(results)}/{len(all_sites)} sites with both seasons")
    print(f"{'='*70}")

    candi_r = [r for r in results if r["cat"] == "candi"]
    ctrl_r = [r for r in results if r["cat"] == "control"]

    for label, group in [("CANDI", candi_r), ("CONTROL", ctrl_r)]:
        if not group:
            continue
        print(f"\n  {label} (n={len(group)}):")
        for k in ["dry_ndwi_diff", "wet_ndwi_diff", "delta_diff", "delta_mean"]:
            vals = [r[k] for r in group if r[k] is not None]
            if vals:
                print(f"    {k:18s}: mean={np.mean(vals):+.5f} std={np.std(vals):.5f}")
        dlvar = [r["delta_lvar"] for r in group if r["delta_lvar"] is not None]
        if dlvar:
            print(f"    {'delta_lvar':18s}: mean={np.mean(dlvar):+.8f}")

    # Tests
    print(f"\n--- Statistical Tests ---")
    tests_out = {}

    # Test 1: Is wet-season NDWI diff stronger at candi than dry?
    # (paired test within candi)
    c_dry = [r["dry_ndwi_diff"] for r in candi_r]
    c_wet = [r["wet_ndwi_diff"] for r in candi_r]
    if len(c_dry) >= 3:
        t_stat, p_paired = sp_stats.wilcoxon([abs(w) for w in c_wet],
                                              [abs(d) for d in c_dry],
                                              alternative="greater")
        sig = "***" if p_paired < 0.05 else "   "
        print(f"  {sig} Candi wet>dry |NDWI diff|: T={t_stat:.1f}, p={p_paired:.5f}")
        tests_out["candi_wet_vs_dry"] = {"T": float(t_stat), "p": float(p_paired), "sig": p_paired < 0.05}

    # Test 2: Is delta_diff (seasonal change) different candi vs control?
    c_delta = [r["delta_diff"] for r in candi_r]
    t_delta = [r["delta_diff"] for r in ctrl_r]
    if len(c_delta) >= 3 and len(t_delta) >= 3:
        u, p = sp_stats.mannwhitneyu(c_delta, t_delta, alternative="two-sided")
        sig = "***" if p < 0.05 else "   "
        print(f"  {sig} Delta_diff candi vs ctrl: U={u:.1f}, p={p:.5f}")
        tests_out["delta_diff_candi_ctrl"] = {"U": float(u), "p": float(p), "sig": p < 0.05}

    # Test 3: Wet-season NDWI diff candi vs control
    t_wet = [abs(r["wet_ndwi_diff"]) for r in ctrl_r]
    c_wet_abs = [abs(r["wet_ndwi_diff"]) for r in candi_r]
    if len(c_wet_abs) >= 3 and len(t_wet) >= 3:
        u, p = sp_stats.mannwhitneyu(c_wet_abs, t_wet, alternative="greater")
        sig = "***" if p < 0.05 else "   "
        print(f"  {sig} Wet |NDWI diff| candi>ctrl: U={u:.1f}, p={p:.5f}")
        tests_out["wet_ndwi_candi_ctrl"] = {"U": float(u), "p": float(p), "sig": p < 0.05}

    # Test 4: Delta local variance
    c_dlvar = [r["delta_lvar"] for r in candi_r if r["delta_lvar"] is not None]
    t_dlvar = [r["delta_lvar"] for r in ctrl_r if r["delta_lvar"] is not None]
    if len(c_dlvar) >= 3 and len(t_dlvar) >= 3:
        u, p = sp_stats.mannwhitneyu(c_dlvar, t_dlvar, alternative="two-sided")
        sig = "***" if p < 0.05 else "   "
        print(f"  {sig} Delta lvar candi vs ctrl: U={u:.1f}, p={p:.5f}")
        tests_out["delta_lvar"] = {"U": float(u), "p": float(p), "sig": p < 0.05}

    # Comparison with E189
    print(f"\n--- Comparison with E189 ---")
    print(f"  E189 dry NDWI: p=0.032 (significant)")
    if "wet_ndwi_candi_ctrl" in tests_out:
        wp = tests_out["wet_ndwi_candi_ctrl"]["p"]
        if wp < 0.032:
            print(f"  E191 wet NDWI: p={wp:.5f} (STRONGER than dry!)")
        else:
            print(f"  E191 wet NDWI: p={wp:.5f} (weaker than dry)")

    # Save
    with open(RESULTS_DIR / "multitemporal_profiles.csv", "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    n_sig = sum(1 for t in tests_out.values() if t.get("sig"))
    summary = {
        "experiment": "E191", "date": "2026-04-13",
        "n_candi": len(candi_r), "n_control": len(ctrl_r),
        "tests": tests_out, "n_significant": n_sig,
    }
    with open(RESULTS_DIR / "e191_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"VERDICT: {n_sig}/{len(tests_out)} tests significant")
    if any(t.get("sig") for t in tests_out.values()):
        print("Multi-temporal analysis provides additional evidence.")
    else:
        print("Seasonal contrast does not amplify the signal.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
