"""Independent direction check of E069 ADV-3 (2026-10-01).

Question: does the negative coefficient on volcano_dist mean FEWER sites near volcanoes
(the reading in the README, the verdict logic and P11 v0.7) or MORE sites near volcanoes?
Uses the per-cell table saved by the original run, recomputes distance to the canonical
30-volcano inventory, refits the same Poisson model, and reports partial effects in
counts, so the sign cannot be misread.
"""
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

REPO = "D:/documents/volcarch-repo"
cells = pd.read_csv(f"{REPO}/experiments/E069_adversarial_comparanda/adv3_survey_intensity/results/adv3_cell_data.csv")
volc = pd.read_csv(f"{REPO}/data/processed/dashboard/volcanoes_java_full.csv")
print("volcano csv columns:", list(volc.columns)[:8], "| n =", len(volc))

latc = next(c for c in volc.columns if c.lower() in ("lat", "latitude"))
lonc = next(c for c in volc.columns if c.lower() in ("lon", "lng", "longitude"))


def hav(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


cells["volc30_km"] = [
    min(hav(la, lo, v_la, v_lo) for v_la, v_lo in zip(volc[latc], volc[lonc]))
    for la, lo in zip(cells["lat_center"], cells["lon_center"])
]
d = cells.replace([np.inf, -np.inf], np.nan).dropna(subset=["road_dist", "bpcb_dist", "uni_dist", "volc30_km"])
print("valid cells:", len(d), "| sites in grid:", int(d.site_count.sum()))

print("\nRAW: sites per cell by distance band to nearest of the 30 volcanoes")
bands = pd.cut(d.volc30_km, [0, 10, 25, 50, 100, 1000])
print(d.groupby(bands, observed=True).site_count.agg(["count", "sum", "mean"]).round(3))

for label, vcol in [("7-volcano (as saved)", "volcano_dist"), ("canonical-30 (recomputed)", "volc30_km")]:
    X = d[["road_dist", "bpcb_dist", "uni_dist", vcol]].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-10
    Xz = (X - mu) / sd
    m = PoissonRegressor(alpha=0, max_iter=2000).fit(Xz, d.site_count)
    beta = m.coef_[-1]
    # partial effect: expected sites per cell at the 10th vs 90th percentile of volcano distance,
    # other predictors held at their means (z = 0)
    lo_km, hi_km = np.percentile(d[vcol], [10, 90])
    z_lo, z_hi = (lo_km - mu[-1]) / sd[-1], (hi_km - mu[-1]) / sd[-1]
    base = m.intercept_
    near = math.exp(base + beta * z_lo)
    far = math.exp(base + beta * z_hi)
    print(f"\n{label}: beta(volcano distance, z-scored) = {beta:+.3f}")
    print(f"  expected sites/cell NEAR (p10 = {lo_km:.0f} km) = {near:.3f}  vs  FAR (p90 = {hi_km:.0f} km) = {far:.3f}")
    print("  => " + ("MORE sites near volcanoes (no burial deficit at this scale)" if near > far
                     else "FEWER sites near volcanoes (deficit)"))
