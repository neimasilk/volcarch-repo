#!/usr/bin/env python3
"""
E190: SAR Feasibility — Can Sentinel-1 Radar See Buried Candi?
================================================================
E189 showed NDWI (moisture) is the strongest optical signal at candi (p=0.032).
SAR directly measures soil moisture and penetrates vegetation.
This experiment tests whether SAR provides a stronger archaeological signal.
"""

import json, csv, sys, numpy as np, requests
from pathlib import Path
from scipy import stats as sp_stats

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-grd"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Same sites as E189 core
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


def search_sar(bbox, date_range="2024-07-01/2024-09-30", max_items=30):
    """Search for Sentinel-1 GRD scenes."""
    body = {
        "collections": ["sentinel-1-grd"],
        "bbox": bbox,
        "datetime": date_range,
        "limit": max_items,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}]
    }
    try:
        resp = requests.post(f"{STAC_URL}/search", json=body, timeout=30)
        if resp.status_code != 200:
            print(f"  STAC error: {resp.status_code}")
            return []
        return resp.json().get("features", [])
    except Exception as e:
        print(f"  Network error: {e}")
        return []


def get_token():
    """Get SAS token for Sentinel-1 data access."""
    try:
        resp = requests.get(SIGN_URL, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("token", "")
    except Exception:
        pass
    return ""


def read_sar_band(scene, band_name, target_lon, target_lat, buffer_m, token):
    """
    Read SAR band (VV or VH) around a point.
    Sentinel-1 GRD on Planetary Computer uses GCPs, not affine transforms.
    We find the nearest GCP to locate the pixel, then read a window.
    """
    import rasterio

    href = scene["assets"].get(band_name, {}).get("href")
    if not href:
        return None
    signed = f"{href}&{token}" if "?" in href else f"{href}?{token}"

    try:
        with rasterio.open(signed) as src:
            gcps, gcp_crs = src.gcps
            if not gcps:
                return None

            # Build a simple inverse mapping from lon/lat to pixel col/row
            # using nearest GCPs and linear interpolation
            gcp_lons = np.array([g.x for g in gcps])
            gcp_lats = np.array([g.y for g in gcps])
            gcp_cols = np.array([g.col for g in gcps])
            gcp_rows = np.array([g.row for g in gcps])

            # Check if target is within GCP footprint
            if (target_lon < gcp_lons.min() - 0.1 or target_lon > gcp_lons.max() + 0.1 or
                target_lat < gcp_lats.min() - 0.1 or target_lat > gcp_lats.max() + 0.1):
                return None

            # Use inverse distance weighting from nearest GCPs to estimate col/row
            dists = np.sqrt((gcp_lons - target_lon)**2 + (gcp_lats - target_lat)**2)
            nearest_idx = np.argsort(dists)[:6]  # 6 nearest GCPs

            weights = 1.0 / (dists[nearest_idx] + 1e-10)
            weights /= weights.sum()

            est_col = int(np.round(np.sum(weights * gcp_cols[nearest_idx])))
            est_row = int(np.round(np.sum(weights * gcp_rows[nearest_idx])))

            # Buffer in pixels (~10m per pixel for GRD IW)
            pix_buffer = max(int(buffer_m / 10), 25)

            row_start = max(0, est_row - pix_buffer)
            row_end = min(src.height, est_row + pix_buffer)
            col_start = max(0, est_col - pix_buffer)
            col_end = min(src.width, est_col + pix_buffer)

            if row_end - row_start < 10 or col_end - col_start < 10:
                return None

            window = rasterio.windows.Window(
                col_start, row_start,
                col_end - col_start, row_end - row_start
            )
            data = src.read(1, window=window).astype(float)

            # Check nodata
            valid_frac = np.count_nonzero(data) / max(data.size, 1)
            if valid_frac < 0.5:
                return None

            return data
    except Exception as e:
        return None


def analyze_sar(vv, vh, site_name):
    """Analyze SAR backscatter at a site."""
    result = {"site": site_name}

    for band_name, arr in [("VV", vv), ("VH", vh)]:
        if arr is None or arr.size == 0:
            continue

        h, w = arr.shape
        if h < 6 or w < 6:
            continue

        # Convert to dB (SAR standard)
        arr_db = np.where(arr > 0, 10 * np.log10(arr), np.nan)

        result[f"{band_name}_mean_db"] = round(float(np.nanmean(arr_db)), 3)
        result[f"{band_name}_std_db"] = round(float(np.nanstd(arr_db)), 3)

        # Center vs ring
        ch, cw = h // 5, w // 5
        center = arr_db[2*ch:3*ch, 2*cw:3*cw]
        ring = np.concatenate([arr_db[:ch, :].flatten(), arr_db[-ch:, :].flatten(),
                               arr_db[ch:-ch, :cw].flatten(), arr_db[ch:-ch, -cw:].flatten()])

        c_mean = float(np.nanmean(center))
        r_mean = float(np.nanmean(ring))
        result[f"{band_name}_center_db"] = round(c_mean, 3)
        result[f"{band_name}_ring_db"] = round(r_mean, 3)
        result[f"{band_name}_diff_db"] = round(c_mean - r_mean, 4)

        # Local variance (3x3)
        lvars = []
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                patch = arr_db[i-1:i+2, j-1:j+2]
                v = np.nanvar(patch)
                if np.isfinite(v):
                    lvars.append(v)
        if lvars:
            result[f"{band_name}_lvar_db"] = round(float(np.mean(lvars)), 6)

        result[f"{band_name}_pixels"] = int(arr.size)

    # Cross-polarization ratio (VH/VV) — sensitive to surface roughness/structure
    if vv is not None and vh is not None and vv.shape == vh.shape:
        ratio = np.where(vv > 0, vh / vv, np.nan)
        result["xpol_ratio_mean"] = round(float(np.nanmean(ratio)), 5)
        result["xpol_ratio_std"] = round(float(np.nanstd(ratio)), 5)

        # Center vs ring for ratio
        h, w = ratio.shape
        if h >= 6 and w >= 6:
            ch, cw = h // 5, w // 5
            rc = ratio[2*ch:3*ch, 2*cw:3*cw]
            rr = np.concatenate([ratio[:ch, :].flatten(), ratio[-ch:, :].flatten(),
                                 ratio[ch:-ch, :cw].flatten(), ratio[ch:-ch, -cw:].flatten()])
            result["xpol_center"] = round(float(np.nanmean(rc)), 5)
            result["xpol_ring"] = round(float(np.nanmean(rr)), 5)
            result["xpol_diff"] = round(float(np.nanmean(rc) - np.nanmean(rr)), 5)

    return result


def main():
    print("=" * 70)
    print("E190: SAR Feasibility — Can Sentinel-1 See Buried Candi?")
    print("=" * 70)

    all_sites = [(n, la, lo, "candi") for n, la, lo in KNOWN_CANDI] + \
                [(n, la, lo, "control") for n, la, lo in CONTROLS]

    print(f"Sites: {len(KNOWN_CANDI)} candi + {len(CONTROLS)} control = {len(all_sites)}")

    # Search SAR scenes
    all_lats = [s[1] for s in all_sites]
    all_lons = [s[2] for s in all_sites]
    bbox = [min(all_lons) - 0.1, min(all_lats) - 0.1,
            max(all_lons) + 0.1, max(all_lats) + 0.1]

    print(f"\nSearching Sentinel-1 SAR scenes...")
    scenes = search_sar(bbox, "2024-07-01/2024-09-30", max_items=30)
    if not scenes:
        scenes = search_sar(bbox, "2023-07-01/2023-09-30", max_items=30)
    if not scenes:
        print("ERROR: No SAR scenes found.")
        return

    print(f"Found: {len(scenes)} scenes")
    for i, s in enumerate(scenes[:5]):
        p = s["properties"]
        print(f"  [{i}] {p.get('datetime', '?')[:10]} | Mode: {p.get('sar:instrument_mode', '?')} | "
              f"Pol: {p.get('sar:polarizations', '?')}")

    token = get_token()
    print(f"Token: {len(token)} chars")

    # Process sites
    print(f"\nExtracting SAR backscatter at {len(all_sites)} sites...")
    results = []
    for name, lat, lon, cat in all_sites:
        success = False
        for scene in scenes:
            vv = read_sar_band(scene, "vv", lon, lat, 500, token)
            if vv is None:
                continue
            vh = read_sar_band(scene, "vh", lon, lat, 500, token)
            if vh is None:
                continue

            analysis = analyze_sar(vv, vh, name)
            analysis["lat"] = lat
            analysis["lon"] = lon
            analysis["cat"] = cat
            analysis["scene"] = scene["properties"].get("datetime", "")[:10]
            results.append(analysis)

            vv_diff = analysis.get("VV_diff_db", "N/A")
            xpol = analysis.get("xpol_diff", "N/A")
            lvar = analysis.get("VV_lvar_db", "N/A")
            print(f"  [{cat:7s}] {name:25s} VV_diff={vv_diff:+.3f}dB xpol_diff={xpol:+.5f} "
                  f"lvar={lvar:.4f}" if isinstance(vv_diff, float) else
                  f"  [{cat:7s}] {name:25s} extracted")
            success = True
            break
        if not success:
            print(f"  [{cat:7s}] {name:25s} SKIP (no data)")

    # Results
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(results)}/{len(all_sites)} sites analyzed")
    print(f"{'='*70}")

    candi_r = [r for r in results if r["cat"] == "candi"]
    ctrl_r = [r for r in results if r["cat"] == "control"]

    for label, group in [("CANDI", candi_r), ("CONTROL", ctrl_r)]:
        if not group:
            continue
        print(f"\n  {label} (n={len(group)}):")
        for metric in ["VV_diff_db", "VH_diff_db", "xpol_diff", "VV_lvar_db", "VH_lvar_db"]:
            vals = [r[metric] for r in group if metric in r and r[metric] is not None]
            if vals:
                print(f"    {metric:15s}: mean={np.mean(vals):+.5f} std={np.std(vals):.5f}")

    # Statistical tests
    print(f"\n--- Statistical Tests ---")
    tests_out = {}
    for metric_name, get_val in [
        ("VV_diff", lambda r: abs(r.get("VV_diff_db", 0))),
        ("VH_diff", lambda r: abs(r.get("VH_diff_db", 0))),
        ("xpol_diff", lambda r: abs(r.get("xpol_diff", 0))),
        ("VV_lvar", lambda r: r.get("VV_lvar_db")),
        ("VH_lvar", lambda r: r.get("VH_lvar_db")),
    ]:
        c_vals = [get_val(r) for r in candi_r if get_val(r) is not None]
        t_vals = [get_val(r) for r in ctrl_r if get_val(r) is not None]
        if len(c_vals) >= 3 and len(t_vals) >= 3:
            u, p = sp_stats.mannwhitneyu(c_vals, t_vals, alternative="greater")
            sig = "***" if p < 0.05 else "   "
            print(f"  {sig} {metric_name:12s}: U={u:5.1f}, p={p:.5f}, "
                  f"candi={np.mean(c_vals):.5f}, ctrl={np.mean(t_vals):.5f}")
            tests_out[metric_name] = {"U": float(u), "p": float(p), "sig": p < 0.05}

    # Compare with E189 optical
    print(f"\n--- Comparison with E189 (Optical) ---")
    print(f"  E189 best: NDWI p=0.032 (significant)")
    sar_best_p = min([t["p"] for t in tests_out.values()]) if tests_out else 1.0
    print(f"  E190 best: p={sar_best_p:.5f} {'(BEATS E189!)' if sar_best_p < 0.032 else '(weaker than E189)' if sar_best_p > 0.05 else '(comparable to E189)'}")

    # Effect sizes
    for metric in ["VV_diff", "VV_lvar", "xpol_diff"]:
        c = [abs(r.get(f"{metric.replace('_diff','_diff_db').replace('_lvar','_lvar_db')}", 0))
             for r in candi_r]
        t = [abs(r.get(f"{metric.replace('_diff','_diff_db').replace('_lvar','_lvar_db')}", 0))
             for r in ctrl_r]
        if c and t:
            pooled = np.sqrt((np.var(c) + np.var(t)) / 2)
            if pooled > 0:
                d = (np.mean(c) - np.mean(t)) / pooled
                print(f"  Cohen d ({metric}): {d:.3f}")

    # Save
    with open(RESULTS_DIR / "sar_profiles.csv", "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    n_sig = sum(1 for t in tests_out.values() if t.get("sig"))
    verdict = "SIGNAL" if n_sig >= 2 else "WEAK_SIGNAL" if n_sig >= 1 else "NO_SIGNAL"

    summary = {
        "experiment": "E190", "date": "2026-04-13",
        "n_candi": len(candi_r), "n_control": len(ctrl_r),
        "tests": tests_out, "verdict": verdict,
        "comparison_e189": {"e189_best_p": 0.032, "e190_best_p": sar_best_p},
    }
    with open(RESULTS_DIR / "e190_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict} ({n_sig}/{len(tests_out)} significant)")
    if sar_best_p < 0.032:
        print("SAR BEATS optical! Proceed to Phase C (ML with SAR features).")
    elif n_sig > 0:
        print("SAR shows signal. Combined optical+SAR fusion is the path forward.")
    else:
        print("SAR does not improve on optical. Try multi-temporal or L-band SAR (ALOS PALSAR).")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
