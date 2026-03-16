#!/usr/bin/env python3
"""
E097: Anomaly Detection on Settlement Model Feature Stack
==========================================================
Uses Isolation Forest to identify grid cells that are environmentally
SUITABLE for settlement (based on E013 feature distributions) but have
NO known archaeological sites. These are candidate "hidden site" zones.

Logic:
1. Extract environmental features at known site locations (E013 pipeline)
2. Train Isolation Forest on site feature distributions
3. Score ALL grid cells: low anomaly = "looks like a site"
4. Combine with E075 burial depth: high burial + low anomaly = buried site
5. Cross-reference top anomalies with E080 fieldwork targets

Key question: Do statistically site-like environments with high burial
depth converge with independently-derived fieldwork targets?

Run from repo root:
    py experiments/E097_anomaly_detection/anomaly_detection.py
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import rowcol
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import cdist
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import folium
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install geopandas rasterio scikit-learn scipy matplotlib folium")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
DEM_DIR = REPO_ROOT / "data" / "processed" / "dem"
SITES_PATH = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
BURIAL_CSV = REPO_ROOT / "experiments" / "E075_sedimentation_model" / "results" / "burial_grid_sample.csv"
TARGETS_CSV = REPO_ROOT / "experiments" / "E080_fieldwork_targets" / "results" / "top20_targets.csv"
RESULTS_DIR = Path(__file__).parent / "results"

FEAT_COLS = ["elevation", "slope", "twi", "tri", "aspect", "river_dist"]

VOLCANOES = {
    "Kelud": (-7.9300, 112.3080),
    "Semeru": (-8.1080, 112.9220),
    "Arjuno-Welirang": (-7.7290, 112.5750),
    "Bromo": (-7.9420, 112.9500),
    "Lamongan": (-7.9770, 113.3430),
    "Raung": (-8.1250, 114.0420),
    "Ijen": (-8.0580, 114.2420),
}

GRID_STEP = 10  # sample every 10th pixel from rasters
RANDOM_SEED = 42


# ── Raster utilities (from E013) ──────────────────────────────────────

def load_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, src.transform, src.crs, src.bounds


def extract_at_points(points_xy: np.ndarray, raster_arr: np.ndarray, transform) -> np.ndarray:
    rows, cols = rowcol(transform, points_xy[:, 0], points_xy[:, 1])
    rows = np.array(rows)
    cols = np.array(cols)
    h, w = raster_arr.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    values = np.full(len(points_xy), np.nan, dtype=np.float32)
    values[valid] = raster_arr[rows[valid], cols[valid]]
    return values


def min_volcano_distance_km(lats, lons):
    """Minimum distance to any volcano in km (haversine)."""
    import math
    min_dists = np.full(len(lats), np.inf)
    for _, (vlat, vlon) in VOLCANOES.items():
        dlat = np.radians(lats - vlat)
        dlon = np.radians(lons - vlon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lats)) * np.cos(np.radians(vlat)) * np.sin(dlon/2)**2
        dists = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        min_dists = np.minimum(min_dists, dists)
    return min_dists


# ── Main ──────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("E097: Anomaly Detection on Settlement Model Feature Stack")
    print("=" * 70)

    # ── 1. Load rasters ───────────────────────────────────────────────
    raster_files = {
        "elevation": DEM_DIR / "jatim_dem.tif",
        "slope": DEM_DIR / "jatim_slope.tif",
        "twi": DEM_DIR / "jatim_twi.tif",
        "tri": DEM_DIR / "jatim_tri.tif",
        "aspect": DEM_DIR / "jatim_aspect.tif",
        "river_dist": DEM_DIR / "jatim_river_dist.tif",
    }

    print("\nLoading rasters...")
    rasters = {}
    for name, path in raster_files.items():
        if not path.exists():
            print(f"  ERROR: Missing raster {path}")
            sys.exit(1)
        arr, transform, crs, bounds = load_raster(path)
        rasters[name] = (arr, transform, crs, bounds)
        print(f"  {name}: {arr.shape}, range [{np.nanmin(arr):.1f}, {np.nanmax(arr):.1f}]")

    ref_arr, ref_transform, ref_crs, ref_bounds = list(rasters.values())[0]

    # ── 2. Load sites & extract features ──────────────────────────────
    print("\nLoading archaeological sites...")
    gdf = gpd.read_file(SITES_PATH)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs("EPSG:4326")
    jatim = (-9.0, 111.0, -6.5, 115.0)
    gdf = gdf[(gdf.geometry.y >= jatim[0]) & (gdf.geometry.x >= jatim[1]) &
              (gdf.geometry.y <= jatim[2]) & (gdf.geometry.x <= jatim[3])]
    print(f"  Sites in East Java: {len(gdf)}")

    sites_proj = gdf.to_crs("EPSG:32749")
    site_xy = np.column_stack([sites_proj.geometry.x, sites_proj.geometry.y])

    site_feats = {}
    for name, (arr, transform, *_) in rasters.items():
        site_feats[name] = extract_at_points(site_xy, arr, transform)
    site_df = pd.DataFrame(site_feats)
    site_df["x_utm"] = site_xy[:, 0]
    site_df["y_utm"] = site_xy[:, 1]
    site_df["lat"] = gdf.geometry.y.values
    site_df["lon"] = gdf.geometry.x.values
    site_df = site_df.dropna(subset=FEAT_COLS)
    site_df = site_df[site_df["elevation"] > 0].reset_index(drop=True)
    print(f"  Sites with valid features: {len(site_df)}")

    # ── 3. Build feature grid ─────────────────────────────────────────
    print("\nBuilding feature grid (every 10th pixel)...")
    h, w = ref_arr.shape
    rows_idx = np.arange(0, h, GRID_STEP)
    cols_idx = np.arange(0, w, GRID_STEP)
    rr, cc = np.meshgrid(rows_idx, cols_idx, indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()
    xs, ys = rasterio.transform.xy(ref_transform, rr, cc)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    grid_data = {"x_utm": xs, "y_utm": ys}
    for name, (arr, transform, *_) in rasters.items():
        grid_data[name] = arr[rr, cc]

    grid_df = pd.DataFrame(grid_data)
    mask = grid_df[FEAT_COLS].notna().all(axis=1) & (grid_df["elevation"] > 0)
    grid_df = grid_df[mask].reset_index(drop=True)

    # Convert to WGS84 for lat/lon
    import pyproj
    transformer = pyproj.Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(grid_df["x_utm"].values, grid_df["y_utm"].values)
    grid_df["lat"] = lats
    grid_df["lon"] = lons

    print(f"  Grid cells: {len(grid_df)}")

    # ── 4. Train Isolation Forest ─────────────────────────────────────
    print("\nTraining Isolation Forest on site feature distributions...")
    scaler = StandardScaler()
    site_X = scaler.fit_transform(site_df[FEAT_COLS].values)
    grid_X = scaler.transform(grid_df[FEAT_COLS].values)

    iso_forest = IsolationForest(
        n_estimators=500,
        max_samples='auto',
        contamination=0.1,  # expect ~10% of sites are "unusual"
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso_forest.fit(site_X)

    # Score grid: decision_function returns anomaly scores
    # More negative = more anomalous relative to site distribution
    # More positive = more "normal" (site-like)
    grid_scores = iso_forest.decision_function(grid_X)
    grid_df["anomaly_score_raw"] = grid_scores

    # Invert: we want HIGH score = site-like
    # decision_function: positive = inlier (site-like), negative = outlier
    # So "site_likeness" = raw score (higher is more site-like)
    grid_df["site_likeness"] = grid_scores

    # Also score sites themselves for validation
    site_scores = iso_forest.decision_function(site_X)
    site_df["site_likeness"] = site_scores
    print(f"  Site likeness stats: mean={site_scores.mean():.3f}, "
          f"std={site_scores.std():.3f}, min={site_scores.min():.3f}")
    print(f"  Grid likeness stats: mean={grid_scores.mean():.3f}, "
          f"std={grid_scores.std():.3f}")

    # ── 5. Load E075 burial depth data ────────────────────────────────
    print("\nLoading E075 burial depth predictions...")
    burial_df = pd.read_csv(BURIAL_CSV)
    print(f"  Burial grid: {len(burial_df)} cells")
    print(f"  Burial range: {burial_df['burial_all_cm'].min():.1f} - "
          f"{burial_df['burial_all_cm'].max():.1f} cm")

    # Match burial depth to our grid cells via nearest neighbor (lat/lon)
    from scipy.spatial import cKDTree
    burial_coords = burial_df[["lat", "lon"]].values
    grid_coords = grid_df[["lat", "lon"]].values
    tree = cKDTree(burial_coords)
    dists, indices = tree.query(grid_coords, k=1)
    grid_df["burial_depth_cm"] = burial_df.iloc[indices]["burial_all_cm"].values
    grid_df["burial_match_dist_deg"] = dists

    # Filter: only keep matches within ~0.1 degree (~11 km)
    good_match = grid_df["burial_match_dist_deg"] < 0.1
    print(f"  Grid cells with burial match: {good_match.sum()}/{len(grid_df)}")

    # ── 6. Composite score: site-likeness x burial depth ──────────────
    print("\nComputing composite anomaly scores...")

    # Normalize both to [0, 1]
    sl_min, sl_max = grid_df["site_likeness"].min(), grid_df["site_likeness"].max()
    grid_df["site_likeness_norm"] = (grid_df["site_likeness"] - sl_min) / (sl_max - sl_min)

    bd_max = grid_df["burial_depth_cm"].max()
    grid_df["burial_norm"] = grid_df["burial_depth_cm"] / bd_max

    # Composite: site-like AND deeply buried = highest priority
    # Only consider cells that are site-like (positive decision function)
    grid_df["composite_score"] = grid_df["site_likeness_norm"] * grid_df["burial_norm"]

    # Filter to site-like cells only (inliers)
    sitelike = grid_df[grid_df["site_likeness"] > 0].copy()
    print(f"  Site-like grid cells (inliers): {len(sitelike)}/{len(grid_df)} "
          f"({100*len(sitelike)/len(grid_df):.1f}%)")

    # Volcano distance for context
    grid_df["volcano_dist_km"] = min_volcano_distance_km(
        grid_df["lat"].values, grid_df["lon"].values
    )

    # ── 7. Top 50 anomalous cells (site-like + buried) ────────────────
    print("\nRanking top 50 candidate buried-site cells...")
    top50 = (grid_df[grid_df["site_likeness"] > 0]
             .nlargest(50, "composite_score")
             .copy()
             .reset_index(drop=True))

    print(f"\n  Top 10 candidate cells:")
    print(f"  {'Rank':<6} {'Lat':<8} {'Lon':<8} {'Burial(cm)':<12} {'SiteLike':<10} "
          f"{'Composite':<12} {'VolcDist(km)'}")
    print(f"  {'-'*70}")
    for i, row in top50.head(10).iterrows():
        print(f"  {i+1:<6} {row['lat']:<8.3f} {row['lon']:<8.3f} "
              f"{row['burial_depth_cm']:<12.0f} {row['site_likeness']:<10.3f} "
              f"{row['composite_score']:<12.4f} {row['volcano_dist_km']:.1f}")

    # Save top 50
    top50_save = top50[["lat", "lon", "elevation", "slope", "twi", "tri",
                         "aspect", "river_dist", "site_likeness",
                         "burial_depth_cm", "composite_score", "volcano_dist_km"]].copy()
    top50_save.to_csv(RESULTS_DIR / "top50_anomaly_cells.csv", index=False)
    print(f"\n  Saved: {RESULTS_DIR / 'top50_anomaly_cells.csv'}")

    # ── 8. Cross-reference with E080 fieldwork targets ────────────────
    print("\nCross-referencing with E080 top 20 fieldwork targets...")
    targets_df = pd.read_csv(TARGETS_CSV)
    print(f"  E080 targets loaded: {len(targets_df)}")

    # Match: anomaly cell within ~5 km of a fieldwork target
    MATCH_THRESHOLD_KM = 5.0
    target_coords = targets_df[["lat", "lon"]].values
    top50_coords = top50[["lat", "lon"]].values

    matches = []
    matched_targets = set()
    for i, (alat, alon) in enumerate(top50_coords):
        for j, (tlat, tlon) in enumerate(target_coords):
            # Haversine
            dlat = np.radians(alat - tlat)
            dlon = np.radians(alon - tlon)
            a = (np.sin(dlat/2)**2 +
                 np.cos(np.radians(alat)) * np.cos(np.radians(tlat)) *
                 np.sin(dlon/2)**2)
            dist_km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
            if dist_km <= MATCH_THRESHOLD_KM:
                matches.append({
                    "anomaly_rank": i + 1,
                    "anomaly_lat": float(alat),
                    "anomaly_lon": float(alon),
                    "target_rank": j + 1,
                    "target_lat": float(tlat),
                    "target_lon": float(tlon),
                    "distance_km": float(dist_km),
                    "anomaly_composite": float(top50.iloc[i]["composite_score"]),
                    "anomaly_burial_cm": float(top50.iloc[i]["burial_depth_cm"]),
                    "target_composite": float(targets_df.iloc[j]["composite_score"]),
                    "target_nearest_volcano": targets_df.iloc[j]["nearest_volcano"],
                })
                matched_targets.add(j)

    n_matched_targets = len(matched_targets)
    overlap_pct = 100 * n_matched_targets / len(targets_df)
    print(f"\n  Matches found (within {MATCH_THRESHOLD_KM} km):")
    print(f"    Matched anomaly cells: {len(matches)}")
    print(f"    Matched E080 targets: {n_matched_targets}/{len(targets_df)} ({overlap_pct:.0f}%)")
    print(f"    Expectation: >30% overlap = strong independent validation")

    if overlap_pct >= 30:
        overlap_verdict = f"STRONG CONVERGENCE ({overlap_pct:.0f}% >= 30%)"
    elif overlap_pct >= 15:
        overlap_verdict = f"MODERATE CONVERGENCE ({overlap_pct:.0f}%)"
    else:
        overlap_verdict = f"WEAK CONVERGENCE ({overlap_pct:.0f}% < 15%)"
    print(f"    Verdict: {overlap_verdict}")

    if matches:
        print(f"\n    Top matches:")
        for m in sorted(matches, key=lambda x: x["distance_km"])[:10]:
            print(f"      Anomaly #{m['anomaly_rank']} ({m['anomaly_lat']:.2f}, {m['anomaly_lon']:.2f}) "
                  f"↔ Target #{m['target_rank']} ({m['target_lat']:.2f}, {m['target_lon']:.2f}): "
                  f"{m['distance_km']:.1f} km — near {m['target_nearest_volcano']}")

    # Save overlap analysis
    overlap_results = {
        "match_threshold_km": MATCH_THRESHOLD_KM,
        "n_anomaly_cells": 50,
        "n_fieldwork_targets": len(targets_df),
        "n_matches": len(matches),
        "n_matched_targets": n_matched_targets,
        "overlap_pct": float(overlap_pct),
        "verdict": overlap_verdict,
        "matches": matches,
    }
    with open(RESULTS_DIR / "overlap_analysis.json", "w", encoding="utf-8") as f:
        json.dump(overlap_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {RESULTS_DIR / 'overlap_analysis.json'}")

    # ── 9. Feature importance via permutation ─────────────────────────
    print("\nComputing feature importance (permutation-based)...")
    baseline_scores = iso_forest.decision_function(grid_X)
    baseline_mean = baseline_scores.mean()
    importances = {}
    rng = np.random.default_rng(RANDOM_SEED)
    for i, feat in enumerate(FEAT_COLS):
        grid_X_perm = grid_X.copy()
        grid_X_perm[:, i] = rng.permutation(grid_X_perm[:, i])
        perm_scores = iso_forest.decision_function(grid_X_perm)
        importances[feat] = float(abs(baseline_mean - perm_scores.mean()))

    # Normalize
    total_imp = sum(importances.values())
    if total_imp > 0:
        importances = {k: v / total_imp for k, v in importances.items()}
    print("  Feature importance (permutation):")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {feat:<15s}: {imp:.3f} {bar}")

    # ── 10. Spatial statistics ────────────────────────────────────────
    print("\nSpatial statistics of site-like cells...")
    sitelike_pct_near_volc = float(
        (grid_df[(grid_df["site_likeness"] > 0) &
                 (grid_df["volcano_dist_km"] <= 50)].shape[0]) /
        max(len(sitelike), 1) * 100
    )
    buried_sitelike = grid_df[
        (grid_df["site_likeness"] > 0) & (grid_df["burial_depth_cm"] > 100)
    ]
    print(f"  Site-like cells within 50 km of volcano: {sitelike_pct_near_volc:.1f}%")
    print(f"  Site-like cells with >1m burial: {len(buried_sitelike)} "
          f"({100*len(buried_sitelike)/max(len(sitelike),1):.1f}%)")

    # ── 11. Visualizations ────────────────────────────────────────────
    print("\nGenerating visualizations...")

    # 11a. Scatter: site_likeness vs burial_depth
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("E097: Anomaly Detection — Site-Likeness vs Burial Depth", fontsize=12)

    ax = axes[0]
    sc = ax.scatter(grid_df["burial_depth_cm"], grid_df["site_likeness"],
                    c=grid_df["volcano_dist_km"], cmap="RdYlBu", alpha=0.3, s=5)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7, label="Inlier/outlier boundary")
    ax.set_xlabel("Burial depth (cm, from E075)")
    ax.set_ylabel("Site-likeness (Isolation Forest)")
    ax.set_title("All grid cells")
    ax.legend(fontsize=7)
    plt.colorbar(sc, ax=ax, label="Volcano dist (km)")

    # 11b. Top 50 on burial vs site-likeness
    ax = axes[1]
    ax.scatter(top50["burial_depth_cm"], top50["site_likeness"],
               c="red", s=30, alpha=0.8, label="Top 50")
    ax.scatter(site_df.get("burial_depth_cm", [0]*len(site_df)) if "burial_depth_cm" in site_df else [0]*len(site_df),
               site_df["site_likeness"],
               c="blue", s=15, alpha=0.5, marker="^", label="Known sites")
    ax.set_xlabel("Burial depth (cm)")
    ax.set_ylabel("Site-likeness")
    ax.set_title("Top 50 candidates vs known sites")
    ax.legend(fontsize=7)

    # 11c. Feature importance
    ax = axes[2]
    feats_sorted = sorted(importances.items(), key=lambda x: x[1])
    ax.barh([k for k, _ in feats_sorted], [v for _, v in feats_sorted],
            color="#E53935", alpha=0.85)
    ax.set_xlabel("Importance (permutation)")
    ax.set_title("Feature importance")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "anomaly_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'anomaly_analysis.png'}")

    # 11d. Folium map
    print("  Generating anomaly map...")
    center_lat = float(grid_df["lat"].mean())
    center_lon = float(grid_df["lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9,
                   tiles="CartoDB positron")

    # Plot site-like cells colored by composite score
    colormap = plt.cm.YlOrRd
    high_composite = grid_df[grid_df["composite_score"] > grid_df["composite_score"].quantile(0.9)]
    for _, row in high_composite.iterrows():
        score_norm = min(row["composite_score"] / max(grid_df["composite_score"].max(), 1e-6), 1.0)
        color = mcolors.to_hex(colormap(score_norm))
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3, color=color, fill=True, fill_color=color,
            fill_opacity=0.5, weight=0,
            tooltip=(f"Composite: {row['composite_score']:.3f} | "
                     f"Burial: {row['burial_depth_cm']:.0f}cm | "
                     f"SiteLike: {row['site_likeness']:.3f}"),
        ).add_to(m)

    # Plot top 50 in red
    for i, row in top50.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6, color="red", fill=True, fill_color="red",
            fill_opacity=0.9, weight=1,
            tooltip=(f"TOP {i+1} | Composite: {row['composite_score']:.3f} | "
                     f"Burial: {row['burial_depth_cm']:.0f}cm"),
        ).add_to(m)

    # Plot E080 targets in green
    for j, row in targets_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5, color="green", fill=True, fill_color="green",
            fill_opacity=0.8, weight=1,
            tooltip=(f"E080 Target #{j+1} | Score: {row['composite_score']:.3f} | "
                     f"Near: {row['nearest_volcano']}"),
        ).add_to(m)

    # Plot known sites in blue
    sites_wgs = gdf.to_crs("EPSG:4326")
    for _, row in sites_wgs.iterrows():
        if row.geometry is None:
            continue
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3, color="blue", fill=True, fill_color="blue",
            fill_opacity=0.7, weight=0,
            tooltip=f"Known site: {row.get('name', '?')}",
        ).add_to(m)

    # Volcanoes
    for vname, (vlat, vlon) in VOLCANOES.items():
        folium.Marker(
            location=[vlat, vlon],
            icon=folium.Icon(color="darkred", icon="fire", prefix="fa"),
            tooltip=f"Volcano: {vname}",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:50px; left:50px; background:white;
         padding:10px; border:2px solid grey; z-index:9999; font-size:12px;">
    <b>E097 Anomaly Map</b><br>
    <span style="color:red;">●</span> Top 50 anomaly cells<br>
    <span style="color:green;">●</span> E080 fieldwork targets<br>
    <span style="color:blue;">●</span> Known archaeological sites<br>
    <span style="color:darkred;">▲</span> Volcanoes<br>
    <span style="color:orange;">●</span> High composite score cells
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(str(RESULTS_DIR / "anomaly_map.html"))
    print(f"  Saved: {RESULTS_DIR / 'anomaly_map.html'}")

    # ── 12. Save full results ─────────────────────────────────────────
    results = {
        "experiment": "E097",
        "title": "Anomaly Detection on Settlement Model Feature Stack",
        "method": "Isolation Forest (n_estimators=500, contamination=0.1)",
        "features": FEAT_COLS,
        "n_sites": len(site_df),
        "n_grid_cells": len(grid_df),
        "n_sitelike_cells": len(sitelike),
        "pct_sitelike": float(100 * len(sitelike) / len(grid_df)),
        "site_likeness_stats": {
            "sites_mean": float(site_scores.mean()),
            "sites_std": float(site_scores.std()),
            "grid_mean": float(grid_scores.mean()),
            "grid_std": float(grid_scores.std()),
        },
        "feature_importance": importances,
        "spatial_stats": {
            "sitelike_within_50km_volcano_pct": sitelike_pct_near_volc,
            "sitelike_with_1m_burial": len(buried_sitelike),
        },
        "overlap_analysis": overlap_results,
        "top10_cells": [
            {
                "rank": i + 1,
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "burial_depth_cm": float(row["burial_depth_cm"]),
                "site_likeness": float(row["site_likeness"]),
                "composite_score": float(row["composite_score"]),
                "volcano_dist_km": float(row["volcano_dist_km"]),
            }
            for i, (_, row) in enumerate(top50.head(10).iterrows())
        ],
    }

    with open(RESULTS_DIR / "e097_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {RESULTS_DIR / 'e097_results.json'}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("E097 SUMMARY")
    print("=" * 70)
    print(f"  Isolation Forest trained on {len(site_df)} known sites")
    print(f"  {len(grid_df)} grid cells scored")
    print(f"  {len(sitelike)} cells ({100*len(sitelike)/len(grid_df):.1f}%) are site-like")
    print(f"  {len(buried_sitelike)} site-like cells have >1m burial depth")
    print(f"  Top feature: {max(importances, key=importances.get)} "
          f"({importances[max(importances, key=importances.get)]:.3f})")
    print(f"\n  OVERLAP WITH E080 FIELDWORK TARGETS:")
    print(f"    {n_matched_targets}/{len(targets_df)} targets matched ({overlap_pct:.0f}%)")
    print(f"    {overlap_verdict}")
    print(f"\n  INTERPRETATION:")
    if overlap_pct >= 30:
        print(f"    The anomaly detection (purely environmental) converges with")
        print(f"    independently-derived fieldwork targets (E080). This provides")
        print(f"    INDEPENDENT VALIDATION that the Zone B/C cells identified by")
        print(f"    the settlement model are genuinely site-like environments")
        print(f"    buried under volcanic deposits.")
    else:
        print(f"    The overlap is lower than expected. This may indicate that")
        print(f"    the E080 targets are driven by different factors (volcano")
        print(f"    proximity, candi proximity) than pure environmental suitability.")
        print(f"    The approaches are COMPLEMENTARY rather than redundant.")
    print("=" * 70)


if __name__ == "__main__":
    main()
