"""Post hoc check (2026-10-22, ledger C080): how far from the sites are the "hard negatives"?

The hybrid design draws "hard negatives" from the band 2 <= zdist <= 5, where zdist is the Euclidean distance from the
mean site environment in standard-deviation units across the six features. The manuscript and E013's README call
these points "environmentally dissimilar from presences". With six features, though, a typical point sits near
sqrt(6) ~ 2.4 from the mean, so the band may hold many of the sites themselves. This script measures that, with
E217's own loaders so the definition is exactly the benchmark's.

Output: results/e217_zdist_sites_check.txt
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e217", HERE / "01_maxent_benchmark.py")
m = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]  # the benchmark module reads no arguments at import, but keep argv clean
spec.loader.exec_module(m)

frame = m.build_frame()
sites = m.load_sites()
xy = np.column_stack([sites.geometry.x, sites.geometry.y])
pres = pd.DataFrame({c: m.sample_at_points(m.DEM_DIR / m.RASTER_FILES[c], xy) for c in m.FULL_COLS})
pres = pres.dropna(subset=m.FULL_COLS)
pres = pres[pres["elevation"] > 0].reset_index(drop=True)

# same standardisation as 01_maxent_benchmark.py (presence mean and SD)
mu = pres[m.FULL_COLS].mean().to_numpy(float)
sd = pres[m.FULL_COLS].std().replace(0, 1.0).to_numpy(float)
z_sites = np.sqrt((((pres[m.FULL_COLS].to_numpy(float) - mu) / sd) ** 2).sum(axis=1))
z_frame = np.sqrt((((frame[m.FULL_COLS].to_numpy(float) - mu) / sd) ** 2).sum(axis=1))

lines = [f"features: {m.FULL_COLS} | presences: {len(pres)} | frame cells (complete covariates): {len(frame)}"]
for name, z in (("sites", z_sites), ("frame (uniform)", z_frame)):
    q = np.percentile(z, [25, 50, 75])
    lines.append(f"{name}: median {q[1]:.2f} (IQR {q[0]:.2f}-{q[2]:.2f}); share z>=2 {np.mean(z >= 2):.3f}; "
                 f"share 2<=z<=5 {np.mean((z >= 2) & (z <= 5)):.3f}; share z>5 {np.mean(z > 5):.3f}")
raw = pd.read_csv(HERE / "results" / "e217_raw_results.csv")
share = raw.groupby("background")["frac_zdist_ge2"].mean()
lines.append("realised share z>=2 per background design (e217_raw_results.csv, mean of 30 runs): "
             + ", ".join(f"{k} {v:.3f}" for k, v in share.items()))
text = "\n".join(lines)
print(text)
(HERE / "results" / "e217_zdist_sites_check.txt").write_text(text + "\n", encoding="utf-8")
