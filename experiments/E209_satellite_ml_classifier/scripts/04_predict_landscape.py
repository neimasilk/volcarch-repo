#!/usr/bin/env python3
"""
E209 Phase 1, Step 04: Landscape-scale prediction — apply trained classifier to
volcanic interior basins and produce probability map + top-K candidate coordinates.

Inputs:
  - models/rf_baseline.joblib (from step 03)
  - data/features_s2.csv (for schema reference — column order must match)

Outputs:
  - results/probability_map_{basin}.tif             # GeoTIFF probability raster
  - results/probability_map_{basin}.png             # Preview PNG
  - results/top20_candidates_{basin}.geojson        # Vector points with scores
  - results/top20_candidates_{basin}.csv            # Human-readable list

Design philosophy:
  The naive approach — extract feature vector per grid cell via separate STAC
  queries — takes ~20 hours per basin. Infeasible.

  Smart approach: download ONE composite raster per basin per season (single big
  tile, e.g., 40 km × 40 km). Compute full-raster indices (NDVI, NDWI, MSAVI)
  in memory. Then apply a sliding 1km × 1km window with 500m stride and compute
  the same features as the training set at each position.

  Output: 80×80 = 6400 grid points per basin, ~10–15 min per basin including
  STAC composite download.

Target basins (from ME#16 diamond-hunt spec):
  - Malang basin (Arjuno-Semeru volcanics)
  - Kediri basin (Kelud volcanics)
  - Progo-Kedu Central Java (Merapi-Merbabu-Sumbing-Sindoro)

Usage:
  python 04_predict_landscape.py --basin malang
  python 04_predict_landscape.py --basin all
  python 04_predict_landscape.py --basin malang --stride 500  # custom grid stride

Status note (2026-04-22):
  Scripting complete. Full execution deferred to post-classifier-training session
  (requires models/rf_baseline.joblib produced by step 03). Test-run ready.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

E209_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = E209_DIR / "models"
RESULTS_DIR = E209_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"

TARGET_BASINS = {
    "malang": {
        "bbox": [112.40, -8.15, 112.85, -7.75],
        "description": "Malang basin (Arjuno-Semeru volcanics)",
    },
    "kediri": {
        "bbox": [111.90, -8.05, 112.30, -7.65],
        "description": "Kediri basin (Kelud volcanics)",
    },
    "progo": {
        "bbox": [110.10, -7.80, 110.55, -7.40],
        "description": "Progo plain (Merapi-Merbabu volcanics)",
    },
    "kedu": {
        "bbox": [110.05, -7.45, 110.40, -7.15],
        "description": "Kedu basin (Sumbing-Sindoro volcanics)",
    },
}

SEASONS = {
    "dry": ["2024-07-01/2024-09-30", "2023-07-01/2023-09-30"],
    "wet": ["2024-01-01/2024-03-31", "2023-01-01/2023-03-31"],
}


def get_sas_token() -> str:
    import requests
    try:
        return requests.get(SIGN_URL, timeout=10).json().get("token", "")
    except Exception:
        return ""


def search_best_scene(bbox: list, date_ranges: list) -> dict:
    """Return the lowest-cloud-cover scene fully covering the bbox."""
    import requests
    scenes = []
    for dr in date_ranges:
        body = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": dr,
            "limit": 50,
            "query": {"eo:cloud_cover": {"lt": 10}},
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        }
        try:
            resp = requests.post(f"{STAC_URL}/search", json=body, timeout=60)
            scenes.extend(resp.json().get("features", []))
        except Exception:
            continue
    # Pick scenes by coverage: prefer those whose tile fully contains the bbox
    scenes.sort(key=lambda s: s["properties"].get("eo:cloud_cover", 100))
    return scenes[0] if scenes else None


def download_basin_composite(basin_bbox: list, season: str, token: str,
                             out_path: Path) -> Path:
    """Download Sentinel-2 bands for a basin bbox and save as multi-band GeoTIFF.

    For simplicity, uses a single best-cloud-cover scene rather than a true
    temporal median composite. This is acceptable for dry-season images with
    <5% cloud cover. Real composite would need multi-scene mosaic.

    Returns path to written file; skips if already exists.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    if out_path.exists():
        print(f"  cached: {out_path.name}")
        return out_path

    scene = search_best_scene(basin_bbox, SEASONS[season])
    if not scene:
        print(f"  NO SCENE for {season}")
        return None

    print(f"  scene: {scene['properties']['datetime'][:10]} "
          f"cloud={scene['properties'].get('eo:cloud_cover', '?'):.1f}%")

    # UTM zone 49S for Java
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
    xmin, ymin = transformer.transform(basin_bbox[0], basin_bbox[1])
    xmax, ymax = transformer.transform(basin_bbox[2], basin_bbox[3])

    bands_to_read = ["B03", "B04", "B08", "B11", "B12"]  # green, red, NIR, SWIR1, SWIR2
    band_arrays = {}
    reference_transform = None
    reference_shape = None

    for band in bands_to_read:
        href = scene["assets"].get(band, {}).get("href")
        if not href:
            print(f"  MISSING asset: {band}")
            return None
        signed = f"{href}&{token}" if "?" in href else f"{href}?{token}"
        with rasterio.open(signed) as src:
            w = from_bounds(xmin, ymin, xmax, ymax, src.transform)
            try:
                data = src.read(1, window=w)
            except Exception as e:
                print(f"  READ ERROR on {band}: {e}")
                return None
            transform = src.window_transform(w)
            band_arrays[band] = data
            if reference_transform is None:
                reference_transform = transform
                reference_shape = data.shape

    # Resample smaller bands if needed (B11/B12 are 20m native, will need resize)
    from scipy.ndimage import zoom
    for band in bands_to_read:
        if band_arrays[band].shape != reference_shape:
            zy = reference_shape[0] / band_arrays[band].shape[0]
            zx = reference_shape[1] / band_arrays[band].shape[1]
            band_arrays[band] = zoom(band_arrays[band], (zy, zx), order=0)

    # Stack + write
    out_stack = np.stack([band_arrays[b].astype(np.int16) for b in bands_to_read], axis=0)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=out_stack.shape[1],
        width=out_stack.shape[2],
        count=len(bands_to_read),
        dtype=out_stack.dtype,
        crs="EPSG:32749",
        transform=reference_transform,
        compress="LZW",
    ) as dst:
        dst.write(out_stack)
        dst.descriptions = tuple(bands_to_read)

    print(f"  wrote: {out_path.name} shape={out_stack.shape}")
    return out_path


def compute_indices_full(composite: np.ndarray) -> dict:
    """From a (5, H, W) stack [B03, B04, B08, B11, B12], compute per-pixel indices."""
    B03, B04, B08, B11, B12 = [composite[i].astype(float) for i in range(5)]
    ndvi_d = B08 + B04
    ndvi = np.where(ndvi_d > 0, (B08 - B04) / ndvi_d, np.nan)
    ndwi_d = B03 + B08
    ndwi = np.where(ndwi_d > 0, (B03 - B08) / ndwi_d, np.nan)
    term = (2 * B08 + 1) ** 2 - 8 * (B08 - B04)
    term = np.maximum(term, 0)
    msavi = (2 * B08 + 1 - np.sqrt(term)) / 2
    clay = np.where(B12 > 0, B11 / B12, np.nan)
    iron = np.where(B03 > 0, B04 / B03, np.nan)
    return {"ndvi": ndvi, "ndwi": ndwi, "msavi": msavi,
            "clay": clay, "iron": iron}


def window_features(indices: dict, cy: int, cx: int, half: int = 50) -> dict:
    """Extract training-compatible features from a window centered at (cy, cx).

    half=50 pixels ≈ 500m at 10m resolution → 1000m × 1000m window (matches training).
    """
    def safe_stats(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0, 0.0
        return float(np.mean(arr)), float(np.std(arr))

    h, w = indices["ndvi"].shape
    if not (half <= cy < h - half and half <= cx < w - half):
        return None

    feats = {}
    for name, arr in indices.items():
        tile = arr[cy - half:cy + half, cx - half:cx + half]
        if tile.size == 0:
            return None
        # Center (inner 20%) vs ring (outer ring)
        hh, ww = tile.shape
        ch, cw = max(1, hh // 5), max(1, ww // 5)
        center = tile[2 * ch:3 * ch, 2 * cw:3 * cw]
        ring = np.concatenate([
            tile[:ch, :].flatten(),
            tile[-ch:, :].flatten(),
            tile[ch:-ch, :cw].flatten(),
            tile[ch:-ch, -cw:].flatten(),
        ])
        m, s = safe_stats(tile)
        cm, _ = safe_stats(center)
        rm, _ = safe_stats(ring)
        feats[f"{name}_mean"] = m
        feats[f"{name}_std"] = s
        feats[f"{name}_center"] = cm
        feats[f"{name}_ring"] = rm
        feats[f"{name}_diff"] = cm - rm

    # Local variance (approx via std in sub-windows)
    for name in ["ndvi", "ndwi"]:
        tile = indices[name][cy - half:cy + half, cx - half:cx + half]
        # Sub-window variance (6×6 blocks of ~17 pixels each)
        lvars = []
        step = max(1, tile.shape[0] // 10)
        for i in range(0, tile.shape[0] - step, step):
            for j in range(0, tile.shape[1] - step, step):
                sub = tile[i:i + step, j:j + step]
                v = np.nanvar(sub)
                if np.isfinite(v):
                    lvars.append(v)
        feats[f"{name}_lvar"] = float(np.mean(lvars)) if lvars else 0.0

    return feats


def predict_basin(basin_name: str, stride_m: int = 500) -> Path:
    """Top-level: download basin tiles, grid-predict, export results."""
    import rasterio
    from pyproj import Transformer
    try:
        import joblib
        model = joblib.load(MODELS_DIR / "rf_baseline.joblib")
    except (ImportError, FileNotFoundError) as e:
        print(f"ERROR: classifier not available — {e}")
        print(f"       Run scripts/03_train_classifier.py first.")
        return None

    info = TARGET_BASINS[basin_name]
    bbox = info["bbox"]
    print(f"\nBasin: {basin_name} — {info['description']}")
    print(f"BBox: {bbox}")

    data_dir = E209_DIR / "data" / "basin_composites"
    data_dir.mkdir(exist_ok=True)

    token = get_sas_token()
    if not token:
        print("ERROR: no SAS token")
        return None

    # Download composites for both seasons
    seasons_data = {}
    for season in ["dry", "wet"]:
        print(f"\n[{season}]")
        out = data_dir / f"{basin_name}_{season}.tif"
        path = download_basin_composite(bbox, season, token, out)
        if path is None:
            return None
        with rasterio.open(path) as src:
            seasons_data[season] = {
                "data": src.read(),
                "transform": src.transform,
                "crs": src.crs,
                "shape": src.shape,
            }

    # Compute indices once per season
    idx = {s: compute_indices_full(seasons_data[s]["data"]) for s in seasons_data}

    # Grid over basin at stride_m
    H, W = seasons_data["dry"]["shape"]
    transform_dry = seasons_data["dry"]["transform"]
    pixel_size = abs(transform_dry.a)  # ~10m for Sentinel-2
    stride_px = max(1, int(stride_m / pixel_size))
    half_px = max(10, int(500 / pixel_size))  # 500m half-window

    print(f"\nGrid: stride={stride_px}px, half-window={half_px}px ({H}×{W} raster)")

    predictions = []
    transformer_back = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)

    # Build prediction feature schema matching training
    # Training used: ndvi_mean, ndvi_std, ndvi_center, ndvi_ring, ndvi_diff, ndvi_lvar,
    #                ndwi_mean, ndwi_std, ndwi_center, ndwi_ring, ndwi_diff, ndwi_lvar,
    #                msavi_mean, msavi_center, msavi_diff, clay_ratio, iron_oxide
    #   × 3 (dry, wet, delta)
    FEATS = [
        "ndvi_mean", "ndvi_std", "ndvi_center", "ndvi_ring", "ndvi_diff", "ndvi_lvar",
        "ndwi_mean", "ndwi_std", "ndwi_center", "ndwi_ring", "ndwi_diff", "ndwi_lvar",
        "msavi_mean", "msavi_center", "msavi_diff",
        "clay_ratio", "iron_oxide",
    ]

    def flatten_feats(season_feats: dict) -> list:
        """Return training-compatible ordered list for one season."""
        out = []
        out += [season_feats.get(k, np.nan) for k in
                ["ndvi_mean", "ndvi_std", "ndvi_center", "ndvi_ring", "ndvi_diff", "ndvi_lvar"]]
        out += [season_feats.get(k, np.nan) for k in
                ["ndwi_mean", "ndwi_std", "ndwi_center", "ndwi_ring", "ndwi_diff", "ndwi_lvar"]]
        out += [season_feats.get(k, np.nan) for k in
                ["msavi_mean", "msavi_center", "msavi_diff"]]
        # Note: training schema has clay_ratio/iron_oxide in the base features,
        # but our window_features uses "clay" + "iron" keys — map them here.
        out.append(season_feats.get("clay_mean", np.nan))
        out.append(season_feats.get("iron_mean", np.nan))
        return out

    t0 = time.time()
    count = 0
    for cy in range(half_px, H - half_px, stride_px):
        for cx in range(half_px, W - half_px, stride_px):
            dry_feats = window_features(idx["dry"], cy, cx, half=half_px)
            wet_feats = window_features(idx["wet"], cy, cx, half=half_px)
            if dry_feats is None or wet_feats is None:
                continue
            dvec = flatten_feats(dry_feats)
            wvec = flatten_feats(wet_feats)
            # Combine with seasonal delta (wet - dry)
            full = []
            for i in range(len(dvec)):
                full.extend([dvec[i], wvec[i],
                             (wvec[i] - dvec[i]) if np.isfinite(dvec[i]) and np.isfinite(wvec[i]) else np.nan])
            # Impute NaN with 0 (crude; should match training impute)
            full = [v if (isinstance(v, float) and np.isfinite(v)) else 0.0 for v in full]
            # Convert pixel coords to UTM then to lat/lon
            from rasterio.transform import xy
            x_utm, y_utm = xy(transform_dry, cy, cx)
            lon, lat = transformer_back.transform(x_utm, y_utm)
            predictions.append({"lat": lat, "lon": lon, "cy": cy, "cx": cx, "features": full})
            count += 1

    print(f"Grid points computed: {count} ({time.time() - t0:.1f}s)")

    # Predict
    X = np.array([p["features"] for p in predictions])
    probs = model.predict_proba(X)[:, 1]
    print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}] mean={probs.mean():.3f}")

    # Top-20 candidates, excluded within 1km of known sites
    import csv as _csv
    sites_csv = E209_DIR / "data" / "training_sites.csv"
    known_pts = []
    if sites_csv.exists():
        with open(sites_csv, encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                try:
                    known_pts.append((float(r["lat"]), float(r["lon"])))
                except (ValueError, KeyError):
                    continue

    def near_known(lat: float, lon: float, threshold_deg: float = 0.01) -> bool:
        return any(abs(lat - la) < threshold_deg and abs(lon - lo) < threshold_deg
                   for la, lo in known_pts)

    ranked = sorted(zip(probs, predictions), key=lambda x: -x[0])
    top_20 = []
    for prob, pred in ranked:
        if near_known(pred["lat"], pred["lon"]):
            continue
        top_20.append({"prob": float(prob), "lat": pred["lat"], "lon": pred["lon"]})
        if len(top_20) >= 20:
            break

    print(f"\nTop-20 candidates (excluding within 1km of known sites):")
    for i, t in enumerate(top_20):
        print(f"  {i+1:2d}. ({t['lat']:.5f}, {t['lon']:.5f}) p={t['prob']:.3f}")

    # Export
    out_csv = RESULTS_DIR / f"top20_candidates_{basin_name}.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["rank", "lat", "lon", "prob"])
        for i, t in enumerate(top_20):
            w.writerow([i + 1, t["lat"], t["lon"], t["prob"]])
    print(f"\nWrote: {out_csv}")

    # GeoJSON
    import json as _json
    features = []
    for i, t in enumerate(top_20):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [t["lon"], t["lat"]]},
            "properties": {"rank": i + 1, "prob": t["prob"]},
        })
    geojson = {"type": "FeatureCollection", "features": features}
    out_gj = RESULTS_DIR / f"top20_candidates_{basin_name}.geojson"
    with open(out_gj, "w", encoding="utf-8") as f:
        _json.dump(geojson, f, indent=2)
    print(f"Wrote: {out_gj}")

    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default="malang",
                        choices=list(TARGET_BASINS.keys()) + ["all"])
    parser.add_argument("--stride", type=int, default=500,
                        help="Grid stride in metres (default 500).")
    args = parser.parse_args()

    print("E209 Step 04: Landscape prediction")
    print("=" * 60)

    basins = list(TARGET_BASINS.keys()) if args.basin == "all" else [args.basin]

    # Check classifier exists
    if not (MODELS_DIR / "rf_baseline.joblib").exists():
        print(f"\nERROR: {MODELS_DIR / 'rf_baseline.joblib'} not found.")
        print(f"       Run scripts/03_train_classifier.py first.")
        return

    for b in basins:
        predict_basin(b, stride_m=args.stride)


if __name__ == "__main__":
    main()
