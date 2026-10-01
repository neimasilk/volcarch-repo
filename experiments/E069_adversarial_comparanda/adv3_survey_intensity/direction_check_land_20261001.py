"""E069 land-only robustness check (2026-10-01, re-entry audit, ledger C023/C035).

225 of the 703 'valid' cells are mostly sea, so the original fit mixes land and sea. This refits the
survey-control model on cells with >=10% land, with distance to the canonical 30 volcanoes and log(land
fraction) as an offset, to confirm that the corrected sign (MORE recorded sites near volcanoes) is not a
sea-cell artefact.
"""
import csv
import math

import numpy as np
import rasterio
import statsmodels.api as sm
from rasterio.warp import transform as wtransform

R = "D:/documents/volcarch-repo"
rows = list(csv.DictReader(open(f"{R}/experiments/E069_adversarial_comparanda/adv3_survey_intensity/results/adv3_cell_data.csv", encoding="utf-8")))
volc = [(float(v["lat"]), float(v["lon"])) for v in csv.DictReader(open(f"{R}/data/processed/dashboard/volcanoes_java_full.csv", encoding="utf-8"))]
dem = rasterio.open(f"{R}/data/processed/dem/jatim_dem.tif")
band = dem.read(1)


def hav(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(x))


def land_frac(lat, lon, n=5):
    """Share of an n x n lattice inside the 0.1-degree cell that falls on land (DEM > 0)."""
    lats = [lat - 0.05 + 0.1 * (i + 0.5) / n for i in range(n)]
    lons = [lon - 0.05 + 0.1 * (j + 0.5) / n for j in range(n)]
    pts = [(lo, la) for la in lats for lo in lons]
    xs, ys = wtransform("EPSG:4326", dem.crs, [p[0] for p in pts], [p[1] for p in pts])
    land = 0
    for x, y in zip(xs, ys):
        r, c = dem.index(x, y)
        if 0 <= r < band.shape[0] and 0 <= c < band.shape[1] and band[r, c] > 0:
            land += 1
    return land / len(pts)


data = []
for r in rows:
    try:
        vals = [float(r[k]) for k in ("site_count", "road_dist", "bpcb_dist", "uni_dist")]
    except ValueError:
        continue
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        continue
    la, lo = float(r["lat_center"]), float(r["lon_center"])
    data.append(vals + [min(hav(la, lo, a, b) for a, b in volc), land_frac(la, lo)])
D = np.array(data)
print("cells:", len(D), "| cells with <10% land:", int((D[:, 5] < 0.1).sum()))
L = D[D[:, 5] >= 0.1]
X = sm.add_constant((L[:, 1:5] - L[:, 1:5].mean(0)) / L[:, 1:5].std(0))
off = np.log(L[:, 5])
nb = sm.GLM(L[:, 0], X, family=sm.families.NegativeBinomial(alpha=1.0), offset=off).fit()
po = sm.GLM(L[:, 0], X, family=sm.families.Poisson(), offset=off).fit(scale="X2")
print(f"land cells: {len(L)} | NB2(alpha=1) beta(volcano distance, canonical 30) = {nb.params[4]:+.3f}, p = {nb.pvalues[4]:.2g}")
print(f"land cells: {len(L)} | quasi-Poisson beta(volcano distance, canonical 30) = {po.params[4]:+.3f}, p = {po.pvalues[4]:.2g}")
print("Negative = more recorded sites closer to volcanoes (the corrected reading).")
