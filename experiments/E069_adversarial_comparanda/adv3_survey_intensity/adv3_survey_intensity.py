"""
ADV-3: Survey Intensity Sufficiency Test
=========================================
ADVERSARIAL EXPERIMENT — Can survey intensity alone explain site distribution?

H0: Site distribution is FULLY explained by survey intensity proxies.
    No residual volcanic signal exists.

Method:
    1. Grid East Java into ~10km cells
    2. Count archaeological sites per cell
    3. Compute survey intensity proxies per cell:
       - Mean road distance (from E013 raster)
       - Min distance to BPCB offices
       - Min distance to archaeology departments
    4. Compute min volcanic proximity per cell
    5. Nested Poisson regression:
       - Model 1 (survey only): site_count ~ road_dist + bpcb_dist + uni_dist
       - Model 2 (survey + volcanic): site_count ~ road_dist + bpcb_dist + uni_dist + volcano_dist
    6. Likelihood ratio test

Interpretation:
    - If Model 2 significantly better (p < 0.05): volcanic adds explanatory power → VOLCARCH SUPPORTED
    - If Model 1 sufficient, no improvement: volcanic is redundant → VOLCARCH WEAKENED

Data:
    - Sites: data/processed/east_java_sites.geojson (E001)
    - Road distance: data/processed/dem/jatim_road_dist_expanded.tif (E013)
    - Volcano coordinates: hardcoded from E013
    - BPCB / University coordinates: defined below
"""
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

# Try imports
try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import rowcol
    from sklearn.linear_model import PoissonRegressor
    from scipy import stats
    from scipy.special import gammaln
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install geopandas rasterio scikit-learn scipy")
    raise

# === PATHS ===
BASE_DIR = Path(".")
SITE_FILE = BASE_DIR / "data" / "processed" / "east_java_sites.geojson"
ROAD_RASTER = BASE_DIR / "data" / "processed" / "dem" / "jatim_road_dist_expanded.tif"
RESULTS_DIR = BASE_DIR / "experiments" / "E069_adversarial_comparanda" / "adv3_survey_intensity" / "results"

# === REFERENCE COORDINATES (WGS84) ===
# East Java volcanoes (from E013)
VOLCANOES = {
    'Kelud': (-7.93, 112.31),
    'Semeru': (-8.108, 112.922),
    'Arjuno-Welirang': (-7.732, 112.577),
    'Bromo': (-7.942, 112.950),
    'Lamongan': (-7.979, 113.342),
    'Raung': (-8.125, 114.042),
    'Ijen': (-8.058, 114.242),
}

# BPCB offices (heritage conservation offices)
BPCB_OFFICES = {
    'BPCB Jawa Timur (Trowulan)': (-7.5639, 112.3788),
    'BPCB DIY (Yogyakarta)': (-7.7956, 110.3695),
    'BPCB Jawa Tengah (Prambanan)': (-7.7520, 110.4910),
}

# Archaeology departments at universities
UNIVERSITIES = {
    'UGM (Yogyakarta)': (-7.7713, 110.3778),
    'Universitas Indonesia (Depok)': (-6.3650, 106.8300),
    'Universitas Brawijaya (Malang)': (-7.9633, 112.6150),
    'Universitas Airlangga (Surabaya)': (-7.2700, 112.7500),
    'Universitas Udayana (Bali)': (-8.7950, 115.1740),
}

# === GRID PARAMETERS ===
# East Java bounding box
LON_MIN, LON_MAX = 110.8, 114.6
LAT_MIN, LAT_MAX = -8.8, -7.0
CELL_SIZE_DEG = 0.1  # ~11km at equator, ~10.5km at -8 lat


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def min_distance_to_set(lat, lon, locations_dict):
    """Min haversine distance from point to nearest location in dict."""
    dists = [haversine_km(lat, lon, loc[0], loc[1]) for loc in locations_dict.values()]
    return min(dists) if dists else np.nan


def build_grid():
    """Build regular grid over East Java."""
    lons = np.arange(LON_MIN, LON_MAX, CELL_SIZE_DEG)
    lats = np.arange(LAT_MIN, LAT_MAX, CELL_SIZE_DEG)
    cells = []
    for lon in lons:
        for lat in lats:
            cells.append({
                'lon_center': lon + CELL_SIZE_DEG / 2,
                'lat_center': lat + CELL_SIZE_DEG / 2,
                'lon_min': lon,
                'lon_max': lon + CELL_SIZE_DEG,
                'lat_min': lat,
                'lat_max': lat + CELL_SIZE_DEG,
            })
    return cells


def count_sites_per_cell(cells, sites_lon, sites_lat):
    """Count archaeological sites in each grid cell."""
    counts = np.zeros(len(cells), dtype=int)
    for i, cell in enumerate(cells):
        mask = ((sites_lon >= cell['lon_min']) & (sites_lon < cell['lon_max']) &
                (sites_lat >= cell['lat_min']) & (sites_lat < cell['lat_max']))
        counts[i] = mask.sum()
    return counts


def wgs84_to_utm49s(lon, lat):
    """Convert WGS84 lon/lat to UTM Zone 49S (EPSG:32749) using formula."""
    import math
    # UTM Zone 49S: central meridian = 111E
    lon0 = 111.0
    k0 = 0.9996
    a = 6378137.0  # WGS84 semi-major axis
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    e_prime2 = e2 / (1 - e2)

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon0_rad = math.radians(lon0)

    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    T = math.tan(lat_rad) ** 2
    C = e_prime2 * math.cos(lat_rad) ** 2
    A = math.cos(lat_rad) * (lon_rad - lon0_rad)

    M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat_rad
             - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * math.sin(2*lat_rad)
             + (15*e2**2/256 + 45*e2**3/1024) * math.sin(4*lat_rad)
             - (35*e2**3/3072) * math.sin(6*lat_rad))

    easting = k0 * N * (A + (1-T+C)*A**3/6 + (5-18*T+T**2+72*C-58*e_prime2)*A**5/120) + 500000
    northing = k0 * (M + N * math.tan(lat_rad) * (A**2/2 + (5-T+9*C+4*C**2)*A**4/24
                + (61-58*T+T**2+600*C-330*e_prime2)*A**6/720))
    if lat < 0:
        northing += 10000000  # Southern hemisphere

    return easting, northing


def sample_road_distance(cells, raster_path):
    """Sample road distance per cell from raster (UTM Zone 49S)."""
    road_dists = np.full(len(cells), np.nan)

    if not raster_path.exists():
        print(f"WARNING: Road distance raster not found: {raster_path}")
        print("  Using distance-to-Surabaya as proxy for accessibility")
        for i, cell in enumerate(cells):
            road_dists[i] = haversine_km(cell['lat_center'], cell['lon_center'],
                                          -7.2575, 112.7521)  # Surabaya
        return road_dists

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        nodata = src.nodata

        for i, cell in enumerate(cells):
            # Convert WGS84 to UTM Zone 49S
            try:
                utm_x, utm_y = wgs84_to_utm49s(cell['lon_center'], cell['lat_center'])
                row, col = rowcol(src.transform, utm_x, utm_y)
                if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                    val = data[row, col]
                    if nodata is not None and val == nodata:
                        road_dists[i] = np.nan
                    else:
                        road_dists[i] = float(val)
                else:
                    road_dists[i] = np.nan
            except Exception:
                road_dists[i] = np.nan

    return road_dists


def main():
    print("=" * 70)
    print("ADV-3: Survey Intensity Sufficiency Test")
    print("ADVERSARIAL EXPERIMENT — Falsification test for VOLCARCH")
    print("=" * 70)

    # --- Load site data ---
    print("\n1. Loading archaeological site data...")
    if not SITE_FILE.exists():
        print(f"ERROR: Site file not found: {SITE_FILE}")
        return

    gdf = gpd.read_file(SITE_FILE)
    sites_lon = gdf.geometry.x.values
    sites_lat = gdf.geometry.y.values
    n_sites = len(gdf)
    print(f"   Loaded {n_sites} sites from {SITE_FILE.name}")

    # --- Build grid ---
    print("\n2. Building grid...")
    cells = build_grid()
    print(f"   {len(cells)} cells ({CELL_SIZE_DEG}deg ~ {CELL_SIZE_DEG * 111:.1f}km)")

    # --- Count sites per cell ---
    print("\n3. Counting sites per cell...")
    site_counts = count_sites_per_cell(cells, sites_lon, sites_lat)
    n_occupied = (site_counts > 0).sum()
    print(f"   {n_occupied}/{len(cells)} cells have >=1 site")
    print(f"   Max sites in one cell: {site_counts.max()}")
    print(f"   Total sites counted: {site_counts.sum()} (of {n_sites} loaded)")

    # --- Compute survey proxies ---
    print("\n4. Computing survey intensity proxies...")

    # 4a. Road distance
    print("   4a. Road distance (from raster)...")
    road_dist = sample_road_distance(cells, ROAD_RASTER)
    n_valid_road = np.isfinite(road_dist).sum()
    print(f"       {n_valid_road}/{len(cells)} cells with valid road distance")

    # 4b. BPCB distance
    print("   4b. Distance to nearest BPCB office...")
    bpcb_dist = np.array([min_distance_to_set(c['lat_center'], c['lon_center'], BPCB_OFFICES) for c in cells])

    # 4c. University distance
    print("   4c. Distance to nearest archaeology department...")
    uni_dist = np.array([min_distance_to_set(c['lat_center'], c['lon_center'], UNIVERSITIES) for c in cells])

    # --- Compute volcanic proximity ---
    print("\n5. Computing volcanic proximity...")
    volcano_dist = np.array([min_distance_to_set(c['lat_center'], c['lon_center'], VOLCANOES) for c in cells])

    # --- Filter to valid cells (has road distance data = on land) ---
    valid = np.isfinite(road_dist) & np.isfinite(bpcb_dist) & np.isfinite(uni_dist) & np.isfinite(volcano_dist)
    print(f"\n6. Valid cells for regression: {valid.sum()}/{len(cells)}")

    y = site_counts[valid]
    X_survey = np.column_stack([
        road_dist[valid],
        bpcb_dist[valid],
        uni_dist[valid],
    ])
    X_full = np.column_stack([
        road_dist[valid],
        bpcb_dist[valid],
        uni_dist[valid],
        volcano_dist[valid],
    ])

    # Standardize predictors for numerical stability
    X_survey_z = (X_survey - X_survey.mean(axis=0)) / (X_survey.std(axis=0) + 1e-10)
    X_full_z = (X_full - X_full.mean(axis=0)) / (X_full.std(axis=0) + 1e-10)

    survey_cols = ['road_dist', 'bpcb_dist', 'uni_dist']
    full_cols = survey_cols + ['volcano_dist']

    def poisson_loglik(y_true, mu):
        """Poisson log-likelihood."""
        mu = np.maximum(mu, 1e-10)
        return np.sum(y_true * np.log(mu) - mu - gammaln(y_true + 1))

    # --- Fit Poisson regression ---
    print("\n7. Fitting Poisson regression models...")

    # Null model (intercept only)
    null_mu = np.full_like(y, y.mean(), dtype=float)
    null_ll = poisson_loglik(y, null_mu)
    print(f"   Null model LL: {null_ll:.2f} (mean={y.mean():.3f})")

    # Model 1: Survey only
    m1 = PoissonRegressor(alpha=0, max_iter=1000, fit_intercept=True)
    m1.fit(X_survey_z, y)
    mu1 = m1.predict(X_survey_z)
    ll1 = poisson_loglik(y, mu1)
    k1 = len(survey_cols) + 1  # coefficients + intercept
    aic1 = 2 * k1 - 2 * ll1
    pr2_1 = 1 - ll1 / null_ll

    print(f"\n   MODEL 1 (Survey Only): {', '.join(survey_cols)}")
    print(f"   Log-likelihood: {ll1:.2f}")
    print(f"   AIC: {aic1:.2f}")
    print(f"   Pseudo-R2 (McFadden): {pr2_1:.4f}")
    print(f"   Intercept: {m1.intercept_:.4f}")
    print("   Coefficients:")
    for name, coef in zip(survey_cols, m1.coef_):
        print(f"     {name:15s}: {coef:+.4f}")

    # Model 2: Survey + Volcanic
    m2 = PoissonRegressor(alpha=0, max_iter=1000, fit_intercept=True)
    m2.fit(X_full_z, y)
    mu2 = m2.predict(X_full_z)
    ll2 = poisson_loglik(y, mu2)
    k2 = len(full_cols) + 1
    aic2 = 2 * k2 - 2 * ll2
    pr2_2 = 1 - ll2 / null_ll

    print(f"\n   MODEL 2 (Survey + Volcanic): {', '.join(full_cols)}")
    print(f"   Log-likelihood: {ll2:.2f}")
    print(f"   AIC: {aic2:.2f}")
    print(f"   Pseudo-R2 (McFadden): {pr2_2:.4f}")
    print(f"   Intercept: {m2.intercept_:.4f}")
    print("   Coefficients:")
    for name, coef in zip(full_cols, m2.coef_):
        print(f"     {name:15s}: {coef:+.4f}")

    # --- Likelihood Ratio Test ---
    print("\n8. LIKELIHOOD RATIO TEST (Model 1 vs Model 2)...")
    lr_stat = 2 * (ll2 - ll1)
    df_diff = k2 - k1  # Should be 1
    lr_pvalue = stats.chi2.sf(lr_stat, df_diff) if lr_stat > 0 else 1.0

    print(f"   LR statistic: {lr_stat:.4f}")
    print(f"   Degrees of freedom: {df_diff}")
    print(f"   p-value: {lr_pvalue:.6f}")

    # --- Check overdispersion ---
    print("\n9. Checking for overdispersion...")
    dispersion = y.var() / max(y.mean(), 0.001)
    print(f"   Mean site count: {y.mean():.3f}")
    print(f"   Variance: {y.var():.3f}")
    print(f"   Dispersion ratio: {dispersion:.2f} (>1 = overdispersed)")

    if dispersion > 1.5:
        print("   Overdispersion detected — Poisson p-values may be anti-conservative.")
        print("   Applying quasi-Poisson correction (Pearson scale)...")
        # Pearson chi2 dispersion estimate for model 2
        pearson_chi2 = np.sum((y - mu2)**2 / np.maximum(mu2, 1e-10))
        phi_hat = pearson_chi2 / (len(y) - k2)
        print(f"   Estimated dispersion (phi): {phi_hat:.2f}")
        # Adjusted LR test
        adj_lr = lr_stat / phi_hat
        adj_p = stats.chi2.sf(adj_lr, df_diff) if adj_lr > 0 else 1.0
        print(f"   Adjusted LR stat: {adj_lr:.4f}, adjusted p: {adj_p:.6f}")
    else:
        print("   No severe overdispersion — Poisson adequate")
        phi_hat = 1.0
        adj_lr, adj_p = lr_stat, lr_pvalue

    # --- Interpretation ---
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Direction of volcanic coefficient
    volc_coef = m2.coef_[-1]  # Last coefficient = volcano_dist
    # Use the quasi-adjusted p-value if overdispersed
    final_p = adj_p if dispersion > 1.5 else lr_pvalue

    if final_p < 0.05:
        if volc_coef < 0:
            verdict = "VOLCARCH SUPPORTED"
            explanation = (
                f"Adding volcanic proximity SIGNIFICANTLY improves the model (p={final_p:.4f}).\n"
                f"Volcanic coefficient is NEGATIVE (beta={volc_coef:.4f}), meaning:\n"
                f"  Fewer sites closer to volcanoes, EVEN AFTER controlling for survey intensity.\n"
                f"  This is consistent with volcanic burial suppressing site discovery."
            )
        else:
            verdict = "VOLCARCH COMPLICATED"
            explanation = (
                f"Adding volcanic proximity significantly improves the model (p={final_p:.4f}),\n"
                f"but coefficient is POSITIVE (beta={volc_coef:.4f}), meaning:\n"
                f"  MORE sites near volcanoes after survey control.\n"
                f"  Volcanic fertile soil may ATTRACT settlement (confound)."
            )
    else:
        verdict = "VOLCARCH WEAKENED"
        explanation = (
            f"Adding volcanic proximity does NOT significantly improve the model (p={final_p:.4f}).\n"
            f"Survey intensity alone may explain site distribution.\n"
            f"However: this does NOT disprove volcanic burial —\n"
            f"  it means the signal is not detectable at this spatial resolution\n"
            f"  with current data ({n_sites} sites, {valid.sum()} grid cells)."
        )

    print(f"\nVERDICT: {verdict}")
    print(f"\n{explanation}")

    # --- Effect sizes ---
    print(f"\n--- Effect sizes ---")
    delta_r2 = pr2_2 - pr2_1
    print(f"Survey-only pseudo-R2: {pr2_1:.4f}")
    print(f"Full model pseudo-R2:  {pr2_2:.4f}")
    print(f"Delta pseudo-R2:       {delta_r2:.4f}")
    print(f"AIC improvement:       {aic1 - aic2:.2f} (positive = Model 2 better)")

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'experiment': 'ADV-3 Survey Intensity Sufficiency',
        'n_sites': int(n_sites),
        'n_cells': int(len(cells)),
        'n_valid_cells': int(valid.sum()),
        'n_occupied_cells': int(n_occupied),
        'cell_size_deg': CELL_SIZE_DEG,
        'model1': {
            'name': 'Survey Only',
            'predictors': survey_cols,
            'log_likelihood': float(ll1),
            'aic': float(aic1),
            'pseudo_r2': float(pr2_1),
            'intercept': float(m1.intercept_),
            'coefficients': {name: float(c) for name, c in zip(survey_cols, m1.coef_)},
        },
        'model2': {
            'name': 'Survey + Volcanic',
            'predictors': full_cols,
            'log_likelihood': float(ll2),
            'aic': float(aic2),
            'pseudo_r2': float(pr2_2),
            'intercept': float(m2.intercept_),
            'coefficients': {name: float(c) for name, c in zip(full_cols, m2.coef_)},
        },
        'likelihood_ratio_test': {
            'lr_statistic': float(lr_stat),
            'df': int(df_diff),
            'p_value': float(lr_pvalue),
        },
        'dispersion_ratio': float(dispersion),
        'dispersion_phi': float(phi_hat),
        'adjusted_lr_p': float(adj_p) if dispersion > 1.5 else float(lr_pvalue),
        'delta_pseudo_r2': float(delta_r2),
        'aic_improvement': float(aic1 - aic2),
        'verdict': verdict,
    }

    results_file = RESULTS_DIR / "adv3_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Descriptive stats CSV
    desc_file = RESULTS_DIR / "adv3_cell_data.csv"
    import csv
    with open(desc_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lon_center', 'lat_center', 'site_count', 'road_dist', 'bpcb_dist', 'uni_dist', 'volcano_dist'])
        for i in range(len(cells)):
            if valid[i]:
                writer.writerow([
                    f"{cells[i]['lon_center']:.3f}",
                    f"{cells[i]['lat_center']:.3f}",
                    site_counts[i],
                    f"{road_dist[i]:.1f}",
                    f"{bpcb_dist[i]:.1f}",
                    f"{uni_dist[i]:.1f}",
                    f"{volcano_dist[i]:.1f}",
                ])
    print(f"Cell data saved to {desc_file}")


if __name__ == "__main__":
    main()
