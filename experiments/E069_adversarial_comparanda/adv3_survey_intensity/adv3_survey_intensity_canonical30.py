"""
ADV-3 CANONICAL-30 RE-RUN: Survey Intensity Sufficiency Test
=============================================================
WS-E / SIG G1 re-derivation (2026-08-13, P11→SPAFA condition 1).

Identical to adv3_survey_intensity.py EXCEPT the volcano source:
the hardcoded 7 eastern-East-Java volcanoes (the volcanoes.csv defect
class) are replaced by the canonical 30-volcano Java inventory
`data/processed/dashboard/volcanoes_java_full.csv`.

Everything else is untouched: same sites, same grid, same survey
proxies, same Poisson models, same quasi-Poisson LR test.

Baseline (7 volcanoes, 2026-03-13): beta=-0.477 (standardized),
adjusted p=0.00154.
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
    raise

# === PATHS (anchored to repo root via this file) ===
REPO = Path(__file__).resolve().parents[3]
SITE_FILE = REPO / "data" / "processed" / "east_java_sites.geojson"
ROAD_RASTER = REPO / "data" / "processed" / "dem" / "jatim_road_dist_expanded.tif"
VOLCANO_CSV = REPO / "data" / "processed" / "dashboard" / "volcanoes_java_full.csv"
RESULTS_DIR = REPO / "experiments" / "E069_adversarial_comparanda" / "adv3_survey_intensity" / "results" / "canonical30"

# === CANONICAL VOLCANO INVENTORY (30, Java-wide) ===
VOLCANOES = {}
with open(VOLCANO_CSV, encoding="utf-8-sig") as f:
    next(f)  # header
    for line in f:
        name, lat, lon = line.strip().split(",")
        VOLCANOES[name] = (float(lat), float(lon))
assert len(VOLCANOES) == 30, f"Expected 30 volcanoes, got {len(VOLCANOES)}"
print(f"Canonical inventory loaded: {len(VOLCANOES)} volcanoes from {VOLCANO_CSV.name}")

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

# === GRID PARAMETERS (identical to original) ===
LON_MIN, LON_MAX = 110.8, 114.6
LAT_MIN, LAT_MAX = -8.8, -7.0
CELL_SIZE_DEG = 0.1


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def min_distance_to_set(lat, lon, locations_dict):
    dists = [haversine_km(lat, lon, loc[0], loc[1]) for loc in locations_dict.values()]
    return min(dists) if dists else np.nan


def build_grid():
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
    counts = np.zeros(len(cells), dtype=int)
    for i, cell in enumerate(cells):
        mask = ((sites_lon >= cell['lon_min']) & (sites_lon < cell['lon_max']) &
                (sites_lat >= cell['lat_min']) & (sites_lat < cell['lat_max']))
        counts[i] = mask.sum()
    return counts


def wgs84_to_utm49s(lon, lat):
    import math
    lon0 = 111.0
    k0 = 0.9996
    a = 6378137.0
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
        northing += 10000000
    return easting, northing


def sample_road_distance(cells, raster_path):
    road_dists = np.full(len(cells), np.nan)
    if not raster_path.exists():
        print(f"WARNING: Road distance raster not found: {raster_path}")
        print("  Using distance-to-Surabaya as proxy for accessibility")
        for i, cell in enumerate(cells):
            road_dists[i] = haversine_km(cell['lat_center'], cell['lon_center'],
                                          -7.2575, 112.7521)
        return road_dists
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        nodata = src.nodata
        for i, cell in enumerate(cells):
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


def poisson_loglik(y_true, mu):
    mu = np.maximum(mu, 1e-10)
    return np.sum(y_true * np.log(mu) - mu - gammaln(y_true + 1))


def main():
    print("=" * 70)
    print("ADV-3 CANONICAL-30 RE-RUN: Survey Intensity Sufficiency Test")
    print("=" * 70)

    print("\n1. Loading archaeological site data...")
    if not SITE_FILE.exists():
        print(f"ERROR: Site file not found: {SITE_FILE}")
        return
    gdf = gpd.read_file(SITE_FILE)
    sites_lon = gdf.geometry.x.values
    sites_lat = gdf.geometry.y.values
    n_sites = len(gdf)
    print(f"   Loaded {n_sites} sites from {SITE_FILE.name}")

    print("\n2. Building grid...")
    cells = build_grid()
    print(f"   {len(cells)} cells")

    print("\n3. Counting sites per cell...")
    site_counts = count_sites_per_cell(cells, sites_lon, sites_lat)
    n_occupied = (site_counts > 0).sum()
    print(f"   {n_occupied}/{len(cells)} cells have >=1 site; max {site_counts.max()}; total {site_counts.sum()}")

    print("\n4. Survey proxies...")
    road_dist = sample_road_distance(cells, ROAD_RASTER)
    n_valid_road = np.isfinite(road_dist).sum()
    print(f"   Road distance: {n_valid_road}/{len(cells)} valid")
    bpcb_dist = np.array([min_distance_to_set(c['lat_center'], c['lon_center'], BPCB_OFFICES) for c in cells])
    uni_dist = np.array([min_distance_to_set(c['lat_center'], c['lon_center'], UNIVERSITIES) for c in cells])

    print("\n5. Volcanic proximity (CANONICAL 30)...")
    volcano_dist = np.array([min_distance_to_set(c['lat_center'], c['lon_center'], VOLCANOES) for c in cells])

    valid = np.isfinite(road_dist) & np.isfinite(bpcb_dist) & np.isfinite(uni_dist) & np.isfinite(volcano_dist)
    print(f"6. Valid cells: {valid.sum()}/{len(cells)}")

    y = site_counts[valid]
    X_survey = np.column_stack([road_dist[valid], bpcb_dist[valid], uni_dist[valid]])
    X_full = np.column_stack([road_dist[valid], bpcb_dist[valid], uni_dist[valid], volcano_dist[valid]])
    X_survey_z = (X_survey - X_survey.mean(axis=0)) / (X_survey.std(axis=0) + 1e-10)
    X_full_z = (X_full - X_full.mean(axis=0)) / (X_full.std(axis=0) + 1e-10)
    survey_cols = ['road_dist', 'bpcb_dist', 'uni_dist']
    full_cols = survey_cols + ['volcano_dist']

    print("\n7. Fitting Poisson regression...")
    null_mu = np.full_like(y, y.mean(), dtype=float)
    null_ll = poisson_loglik(y, null_mu)
    print(f"   Null LL: {null_ll:.2f}")

    m1 = PoissonRegressor(alpha=0, max_iter=1000, fit_intercept=True)
    m1.fit(X_survey_z, y)
    mu1 = m1.predict(X_survey_z)
    ll1 = poisson_loglik(y, mu1)
    k1 = len(survey_cols) + 1
    aic1 = 2 * k1 - 2 * ll1
    pr2_1 = 1 - ll1 / null_ll
    print(f"   MODEL 1 (survey only): LL={ll1:.2f} AIC={aic1:.2f} pseudoR2={pr2_1:.4f}")
    for name, coef in zip(survey_cols, m1.coef_):
        print(f"     {name:15s}: {coef:+.4f}")

    m2 = PoissonRegressor(alpha=0, max_iter=1000, fit_intercept=True)
    m2.fit(X_full_z, y)
    mu2 = m2.predict(X_full_z)
    ll2 = poisson_loglik(y, mu2)
    k2 = len(full_cols) + 1
    aic2 = 2 * k2 - 2 * ll2
    pr2_2 = 1 - ll2 / null_ll
    print(f"   MODEL 2 (survey + volcanic): LL={ll2:.2f} AIC={aic2:.2f} pseudoR2={pr2_2:.4f}")
    for name, coef in zip(full_cols, m2.coef_):
        print(f"     {name:15s}: {coef:+.4f}")

    print("\n8. LIKELIHOOD RATIO TEST...")
    lr_stat = 2 * (ll2 - ll1)
    df_diff = k2 - k1
    lr_pvalue = stats.chi2.sf(lr_stat, df_diff) if lr_stat > 0 else 1.0
    print(f"   LR stat: {lr_stat:.4f}, df={df_diff}, p={lr_pvalue:.6f}")

    print("\n9. Overdispersion check...")
    dispersion = y.var() / max(y.mean(), 0.001)
    print(f"   Dispersion ratio: {dispersion:.2f}")
    if dispersion > 1.5:
        pearson_chi2 = np.sum((y - mu2)**2 / np.maximum(mu2, 1e-10))
        phi_hat = pearson_chi2 / (len(y) - k2)
        adj_lr = lr_stat / phi_hat
        adj_p = stats.chi2.sf(adj_lr, df_diff) if adj_lr > 0 else 1.0
        print(f"   Quasi-Poisson: phi={phi_hat:.2f}, adjusted LR={adj_lr:.4f}, adjusted p={adj_p:.6f}")
    else:
        print("   No severe overdispersion")
        phi_hat = 1.0
        adj_lr, adj_p = lr_stat, lr_pvalue

    volc_coef = m2.coef_[-1]
    final_p = adj_p if dispersion > 1.5 else lr_pvalue

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if final_p < 0.05:
        if volc_coef < 0:
            verdict = "VOLCARCH SUPPORTED"
        else:
            verdict = "VOLCARCH COMPLICATED"
    else:
        verdict = "VOLCARCH WEAKENED"
    print(f"VERDICT: {verdict}  (volc beta={volc_coef:.4f}, final p={final_p:.6f})")

    delta_r2 = pr2_2 - pr2_1
    print(f"Delta pseudo-R2: {delta_r2:.4f}; AIC improvement: {aic1 - aic2:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'experiment': 'ADV-3 Survey Intensity Sufficiency — CANONICAL-30 RE-RUN',
        'date': '2026-08-13',
        'volcano_source': str(VOLCANO_CSV),
        'n_volcanoes': len(VOLCANOES),
        'n_sites': int(n_sites),
        'n_cells': int(len(cells)),
        'n_valid_cells': int(valid.sum()),
        'n_occupied_cells': int(n_occupied),
        'model1': {
            'log_likelihood': float(ll1), 'aic': float(aic1), 'pseudo_r2': float(pr2_1),
            'coefficients': {name: float(c) for name, c in zip(survey_cols, m1.coef_)},
        },
        'model2': {
            'log_likelihood': float(ll2), 'aic': float(aic2), 'pseudo_r2': float(pr2_2),
            'coefficients': {name: float(c) for name, c in zip(full_cols, m2.coef_)},
        },
        'likelihood_ratio_test': {
            'lr_statistic': float(lr_stat), 'df': int(df_diff), 'p_value': float(lr_pvalue),
        },
        'dispersion_ratio': float(dispersion),
        'dispersion_phi': float(phi_hat),
        'adjusted_lr_p': float(adj_p) if dispersion > 1.5 else float(lr_pvalue),
        'delta_pseudo_r2': float(delta_r2),
        'aic_improvement': float(aic1 - aic2),
        'verdict': verdict,
        'baseline_7volcano': {
            'beta': -0.47659563995396126, 'adjusted_p': 0.0015436974286970916,
        },
    }
    with open(RESULTS_DIR / "adv3_canonical30_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_DIR / 'adv3_canonical30_results.json'}")


if __name__ == "__main__":
    main()
