"""
E213: Aggradation-Exposure Geomorphic Asymmetry (the corrected basis for P7).

After the Antiquity rejection killed the "distance from volcano" variable, this
tests the CORRECT taphonomic variable: deep archaeology is visible only where
erosion/karst EXPOSES it. The novel, falsifiable claim:

  Settlement suitability favors LOW-RELIEF plains; archaeological VISIBILITY
  favors RELIEF (denudation hills / karst) or river incision. The two
  anti-correlate -> the land people would settle is exactly the land where the
  deep record is buried. Surface absence on the plains is therefore uninformative
  (a detection-horizon statement, NOT a claim that buried sites exist).

Data: Copernicus 30m DEM derivatives for East Java (UTM 49S / EPSG:32749).
Run: python experiments/E213_aggradation_exposure_asymmetry/analyze.py
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from pathlib import Path
from scipy.stats import spearmanr

REPO = Path(__file__).parent.parent.parent
DEM = REPO / "data" / "processed" / "dem"
OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)

FLAT_DEG = 2.0  # slope below this = low-relief plain (net deposition / burial)

def sample_raster(path, lons, lats):
    """Sample a UTM raster at WGS84 lon/lat points; return array (nodata->nan)."""
    with rasterio.open(path) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, list(lons), list(lats))
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype="float64")
        nod = src.nodata
    if nod is not None:
        vals[vals == nod] = np.nan
    vals[vals < -1e30] = np.nan
    return vals

# ---------- 1. The four known deep-time sites: terrain signature ----------
sites = pd.read_csv(REPO / "experiments/E019_spatial_distribution/data/deep_time_sites.csv")
geomorph_ctx = {  # established geomorphic context from the literature
    "Song Terus": "karst cave (Gunung Sewu) - EXPOSURE",
    "Trinil":     "Solo River terrace (incision) - EXPOSURE",
    "Sangiran":   "eroding dome / Cemoro R. - EXPOSURE (edge of DEM)",
    "Wajak":      "karst cave (S. Mountains) - EXPOSURE",
}
for col, path in [("dem","jatim_dem.tif"),("slope","jatim_slope.tif"),
                  ("tri","jatim_tri.tif"),("river_dist_m","jatim_river_dist.tif")]:
    sites[col] = sample_raster(DEM/path, sites.lon.values, sites.lat.values)
sites["geomorph_context"] = sites.name.map(geomorph_ctx)
print("="*72)
print("KNOWN DEEP-TIME SITES — terrain signature")
print("="*72)
print(sites[["name","elev_m" if False else "dem","slope","tri","river_dist_m","geomorph_context"]]
      .rename(columns={"dem":"elev_m"}).to_string(index=False))

# ---------- 2. Visibility deficit over the settlement-suitability grid ----------
grid = pd.read_csv(REPO / "data/processed/dashboard/grid_predictions.csv")
# subsample for speed (every 3rd cell ~ 21k points; plenty for stats)
g = grid.iloc[::3].copy()
g["slope"] = sample_raster(DEM/"jatim_slope.tif", g.lon.values, g.lat.values)
g["river_dist_m"] = sample_raster(DEM/"jatim_river_dist.tif", g.lon.values, g.lat.values)
g = g.dropna(subset=["slope","suitability"])
print(f"\nGrid cells sampled (in DEM coverage): {len(g):,}")

# suitability vs slope: do suitable cells cluster on flat (buryable) terrain?
rho, p = spearmanr(g.suitability, g.slope)
print(f"\nSpearman(suitability, slope) = {rho:.3f} (p={p:.1e})  "
      f"-> {'suitable land is FLAT (buryable)' if rho<0 else 'suitable land is sloped'}")

hi = g[g.suitability >= 0.5]                      # where the model says people settle
flat_hi = (hi.slope < FLAT_DEG).mean() * 100      # buried-unless-incised fraction
print(f"\nHigh-suitability cells (suit>=0.5): {len(hi):,}")
print(f"  % that are LOW-RELIEF plains (slope<{FLAT_DEG} deg) = {flat_hi:.1f}%   <-- buried-record fraction")
# narrow 'visible via incision' sliver (upper bound; floodplain-conflated, flagged)
incised = hi[(hi.slope < FLAT_DEG) & (hi.river_dist_m < 1000)]
print(f"  of those flat cells, within 1 km of a river (terrace OR floodplain, UPPER bound visible): "
      f"{len(incised)/max(len(hi[hi.slope<FLAT_DEG]),1)*100:.1f}%")

# the asymmetry, stated as two conditional means
print(f"\n  mean slope | high suitability (>=0.5): {hi.slope.mean():.2f} deg")
print(f"  mean slope | low  suitability (<0.5):  {g[g.suitability<0.5].slope.mean():.2f} deg")

sites.to_csv(OUT/"deep_time_terrain_signature.csv", index=False)
summary = {
    "spearman_suit_slope": round(float(rho),3),
    "pct_high_suit_lowrelief_buried": round(float(flat_hi),1),
    "mean_slope_high_suit": round(float(hi.slope.mean()),2),
    "mean_slope_low_suit": round(float(g[g.suitability<0.5].slope.mean()),2),
    "n_grid_sampled": int(len(g)),
}
import json
(OUT/"e213_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSaved: {OUT/'deep_time_terrain_signature.csv'} + e213_summary.json")
