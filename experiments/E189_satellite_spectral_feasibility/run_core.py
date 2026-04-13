#!/usr/bin/env python3
"""
E189 Core Run: Candi vs Control spectral comparison.
Focused version — 15 candi + 5 control = 20 sites.
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

all_sites = [(n, la, lo, "candi") for n, la, lo in KNOWN_CANDI] + \
            [(n, la, lo, "control") for n, la, lo in CONTROLS]

print(f"Sites: {len(KNOWN_CANDI)} candi + {len(CONTROLS)} control = {len(all_sites)}")

# Search STAC
all_lats = [s[1] for s in all_sites]
all_lons = [s[2] for s in all_sites]
bbox = [min(all_lons)-0.1, min(all_lats)-0.1, max(all_lons)+0.1, max(all_lats)+0.1]

print(f"Searching STAC...")
scenes = []
for dr in ["2024-07-01/2024-09-30", "2023-07-01/2023-09-30"]:
    body = {
        "collections": ["sentinel-2-l2a"], "bbox": bbox,
        "datetime": dr, "limit": 20,
        "query": {"eo:cloud_cover": {"lt": 15}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]
    }
    resp = requests.post(f"{STAC_URL}/search", json=body, timeout=30)
    found = resp.json().get("features", [])
    scenes.extend(found)
    print(f"  {dr}: {len(found)} scenes")
    if len(scenes) >= 15:
        break

scenes.sort(key=lambda s: s["properties"].get("eo:cloud_cover", 100))
print(f"Total: {len(scenes)} scenes")
for i, s in enumerate(scenes[:5]):
    p = s["properties"]
    print(f"  [{i}] {p['datetime'][:10]} cloud={p['eo:cloud_cover']:.1f}% tile={p.get('s2:mgrs_tile','?')}")

# Token
token = requests.get(SIGN_URL, timeout=10).json().get("token", "")
print(f"Token: {len(token)} chars")

import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)


def read_band(scene, band_name, cx, cy, buffer_m=500):
    href = scene["assets"].get(band_name, {}).get("href")
    if not href:
        return None
    signed = f"{href}&{token}" if "?" in href else f"{href}?{token}"
    with rasterio.open(signed) as src:
        if not (src.bounds.left <= cx <= src.bounds.right and
                src.bounds.bottom <= cy <= src.bounds.top):
            return None
        w = from_bounds(cx - buffer_m, cy - buffer_m,
                        cx + buffer_m, cy + buffer_m, src.transform)
        data = src.read(1, window=w).astype(float)
        # Sentinel-2 uses 0 as nodata at tile edges — reject mostly-empty windows
        valid_frac = np.count_nonzero(data) / max(data.size, 1)
        if valid_frac < 0.5:
            return None
        return data


def resize_match(src, tgt):
    if src.shape == tgt.shape:
        return src
    return zoom(src, (tgt.shape[0]/src.shape[0], tgt.shape[1]/src.shape[1]), order=0)


# Process
print(f"\nExtracting spectral indices...")
results = []
for name, lat, lon, cat in all_sites:
    cx, cy = transformer.transform(lon, lat)
    success = False
    for scene in scenes:
        try:
            b04 = read_band(scene, "B04", cx, cy)
            if b04 is None:
                continue
            b08 = read_band(scene, "B08", cx, cy)
            b03 = read_band(scene, "B03", cx, cy)
            if b08 is None or b03 is None:
                continue

            # NDVI
            d = b08 + b04
            ndvi = np.where(d > 0, (b08 - b04) / d, np.nan)
            # NDWI
            b03r = resize_match(b03, b08)
            d2 = b03r + b08
            ndwi = np.where(d2 > 0, (b03r - b08) / d2, np.nan)
            # MSAVI
            term = (2 * b08 + 1)**2 - 8 * (b08 - b04)
            term = np.maximum(term, 0)
            msavi = (2 * b08 + 1 - np.sqrt(term)) / 2

            h, w = ndvi.shape
            if h < 6 or w < 6:
                continue

            # Center vs ring
            ch, cw = h // 5, w // 5
            nc = ndvi[2*ch:3*ch, 2*cw:3*cw]
            nr = np.concatenate([ndvi[:ch, :].flatten(), ndvi[-ch:, :].flatten(),
                                 ndvi[ch:-ch, :cw].flatten(), ndvi[ch:-ch, -cw:].flatten()])
            wc = ndwi[2*ch:3*ch, 2*cw:3*cw]
            wr = np.concatenate([ndwi[:ch, :].flatten(), ndwi[-ch:, :].flatten(),
                                 ndwi[ch:-ch, :cw].flatten(), ndwi[ch:-ch, -cw:].flatten()])
            mc = msavi[2*ch:3*ch, 2*cw:3*cw]
            mr = np.concatenate([msavi[:ch, :].flatten(), msavi[-ch:, :].flatten(),
                                 msavi[ch:-ch, :cw].flatten(), msavi[ch:-ch, -cw:].flatten()])

            # Local variance (3x3)
            lvars_ndvi = []
            lvars_ndwi = []
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    v = np.nanvar(ndvi[i-1:i+2, j-1:j+2])
                    if np.isfinite(v):
                        lvars_ndvi.append(v)
                    v2 = np.nanvar(ndwi[i-1:i+2, j-1:j+2])
                    if np.isfinite(v2):
                        lvars_ndwi.append(v2)

            r = {
                "site": name, "cat": cat, "lat": lat, "lon": lon,
                "ndvi_mean": round(float(np.nanmean(ndvi)), 5),
                "ndvi_std": round(float(np.nanstd(ndvi)), 5),
                "ndvi_center": round(float(np.nanmean(nc)), 5),
                "ndvi_ring": round(float(np.nanmean(nr)), 5),
                "ndvi_diff": round(float(np.nanmean(nc) - np.nanmean(nr)), 5),
                "ndvi_lvar": round(float(np.mean(lvars_ndvi)), 8) if lvars_ndvi else None,
                "ndwi_mean": round(float(np.nanmean(ndwi)), 5),
                "ndwi_diff": round(float(np.nanmean(wc) - np.nanmean(wr)), 5),
                "ndwi_lvar": round(float(np.mean(lvars_ndwi)), 8) if lvars_ndwi else None,
                "msavi_mean": round(float(np.nanmean(msavi)), 5),
                "msavi_diff": round(float(np.nanmean(mc) - np.nanmean(mr)), 5),
                "pixels": ndvi.size,
                "scene": scene["properties"]["datetime"][:10],
            }
            results.append(r)
            print(f"  [{cat:7s}] {name:25s} NDVI_diff={r['ndvi_diff']:+.5f} "
                  f"lvar={r['ndvi_lvar']:.6f}" if r['ndvi_lvar'] else
                  f"  [{cat:7s}] {name:25s} NDVI_diff={r['ndvi_diff']:+.5f} lvar=N/A")
            success = True
            break
        except Exception as e:
            continue
    if not success:
        print(f"  [{cat:7s}] {name:25s} SKIP")

# ── Results ────────────────��──────────────────────────────────────────
print(f"\n{'='*70}")
print(f"RESULTS: {len(results)}/{len(all_sites)} sites analyzed")
print(f"{'='*70}")

candi_r = [r for r in results if r["cat"] == "candi"]
ctrl_r = [r for r in results if r["cat"] == "control"]

for label, group in [("CANDI", candi_r), ("CONTROL", ctrl_r)]:
    if not group:
        continue
    diffs = [r["ndvi_diff"] for r in group]
    lvars = [r["ndvi_lvar"] for r in group if r["ndvi_lvar"] is not None]
    ndwi_d = [r["ndwi_diff"] for r in group]
    msavi_d = [r["msavi_diff"] for r in group]
    print(f"\n  {label} (n={len(group)}):")
    print(f"    NDVI  diff: mean={np.mean(diffs):+.5f} std={np.std(diffs):.5f}")
    print(f"    NDWI  diff: mean={np.mean(ndwi_d):+.5f} std={np.std(ndwi_d):.5f}")
    print(f"    MSAVI diff: mean={np.mean(msavi_d):+.5f} std={np.std(msavi_d):.5f}")
    if lvars:
        print(f"    NDVI local_var: mean={np.mean(lvars):.8f}")

# Statistical tests
print(f"\n--- Statistical Tests ---")
tests_out = {}

for idx_name, get_val in [
    ("NDVI_diff", lambda r: abs(r["ndvi_diff"])),
    ("NDWI_diff", lambda r: abs(r["ndwi_diff"])),
    ("MSAVI_diff", lambda r: abs(r["msavi_diff"])),
    ("NDVI_lvar", lambda r: r["ndvi_lvar"]),
    ("NDWI_lvar", lambda r: r["ndwi_lvar"]),
]:
    c_vals = [get_val(r) for r in candi_r if get_val(r) is not None]
    t_vals = [get_val(r) for r in ctrl_r if get_val(r) is not None]
    if len(c_vals) >= 3 and len(t_vals) >= 3:
        u, p = sp_stats.mannwhitneyu(c_vals, t_vals, alternative="greater")
        sig = "***" if p < 0.05 else "   "
        print(f"  {sig} {idx_name:12s}: U={u:5.1f}, p={p:.5f}, "
              f"candi_mean={np.mean(c_vals):.6f}, ctrl_mean={np.mean(t_vals):.6f}")
        tests_out[idx_name] = {"U": float(u), "p": float(p), "sig": p < 0.05}

# Effect size
c_d = [abs(r["ndvi_diff"]) for r in candi_r]
t_d = [abs(r["ndvi_diff"]) for r in ctrl_r]
if c_d and t_d:
    pooled = np.sqrt((np.var(c_d) + np.var(t_d)) / 2)
    if pooled > 0:
        cohen_d = (np.mean(c_d) - np.mean(t_d)) / pooled
        print(f"\n  Cohen's d (NDVI |diff|): {cohen_d:.3f}")

# ── Per-site table ────────────────────────────────────────────────────
print(f"\n--- Per-site NDVI Anomaly ---")
print(f"  {'Site':25s} {'Cat':7s} {'NDVI_diff':>10s} {'NDWI_diff':>10s} {'Lvar':>12s}")
for r in sorted(results, key=lambda x: abs(x["ndvi_diff"]), reverse=True):
    lv = f"{r['ndvi_lvar']:.6f}" if r['ndvi_lvar'] else "N/A"
    print(f"  {r['site']:25s} {r['cat']:7s} {r['ndvi_diff']:+10.5f} "
          f"{r['ndwi_diff']:+10.5f} {lv:>12s}")

# ── Save ────────────────��──────────────────────────────��──────────────
with open(RESULTS_DIR / "spectral_profiles.csv", "w", newline="", encoding="utf-8") as f:
    if results:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

n_sig = sum(1 for t in tests_out.values() if t.get("sig"))
if n_sig == 0:
    verdict = "NO_SIGNAL"
elif n_sig <= len(tests_out) // 3:
    verdict = "WEAK_SIGNAL"
else:
    verdict = "SIGNAL_DETECTED"

summary = {
    "experiment": "E189",
    "title": "Satellite Spectral Feasibility",
    "date": "2026-04-13",
    "n_candi": len(candi_r),
    "n_control": len(ctrl_r),
    "tests": tests_out,
    "verdict": verdict,
    "n_significant": n_sig,
    "n_tests": len(tests_out),
}
with open(RESULTS_DIR / "e189_results.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"VERDICT: {verdict} ({n_sig}/{len(tests_out)} tests significant)")
if verdict == "NO_SIGNAL":
    print("Andosol may be too homogeneous for passive optical. Proceed to SAR (Phase B).")
elif verdict == "WEAK_SIGNAL":
    print("Some signal detected. Multi-temporal or SAR analysis recommended.")
else:
    print("Spectral anomalies at candi differ from controls. Proceed to ML (Phase C).")
print(f"{'='*70}")
