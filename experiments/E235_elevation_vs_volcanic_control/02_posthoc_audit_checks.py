# E235 POST HOC checks (NOT pre-registered): the orchestrator's re-run of the 2026-10-20 audit's main claims. See README §Audit.
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[2]; EXP = R / "experiments"
sp = importlib.util.spec_from_file_location("b", EXP / "E217_maxent_benchmark/01_maxent_benchmark.py"); b = importlib.util.module_from_spec(sp); sp.loader.exec_module(b)
f = b.build_frame()
import pyproj
from scipy.spatial import cKDTree
tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
vol = pd.read_csv(R / "data/processed/dashboard/volcanoes_java_full.csv")
def dist(v):
    vx, vy = tr.transform(v["lon"].to_numpy(), v["lat"].to_numpy())
    return cKDTree(np.column_stack([vx, vy])).query(f[["x", "y"]].to_numpy())[0] / 1000
d30 = dist(vol)
v13 = vol[(vol.lon >= 111) & (vol.lon <= 115) & (vol.lat >= -9) & (vol.lat <= -6.5)]
d13 = dist(v13)
m = np.load(EXP / "E221_seed_ensemble_stability/results/e221_priority_sets_xgboost.npz")
elev = f.elevation.to_numpy(); band = (elev // 200).astype(int)
blocks = b.assign_blocks(f.x.to_numpy(), f.y.to_numpy())
cell = (10 * 30.663793749005784 / 1000) ** 2

def run(d, vk, nk, cols=("slope", "tri", "twi"), nb=3, outcome="suit_random", boot=0):
    V, N = d <= vk, d >= nk
    usable = [bb for bb in range(band.max() + 1)
              if ((band == bb) & V).sum() * cell >= 25 and ((band == bb) & N).sum() * cell >= 25
              and len(np.unique(blocks[(band == bb) & V])) >= 2 and len(np.unique(blocks[(band == bb) & N])) >= 2]
    st = np.full(len(f), -1)
    for bb in usable:
        sel = (band == bb) & (V | N); key = np.zeros(sel.sum(), int)
        for c in cols:
            v = f[c].to_numpy()[sel]; key = key * nb + np.digitize(v, np.quantile(v, np.linspace(0, 1, nb + 1)[1:-1]))
        st[sel] = bb * 10000 + key
    y = (m["n_designs_top10"] == 1) * 100.0 if outcome == "fringe" else m[outcome].astype(float)
    def est(wv=None, wn=None):
        g = pd.DataFrame({"st": st, "y": y, "V": V, "N": N, "blk": blocks})
        g = g[g.st >= 0]
        g["w"] = 1.0
        if wv is not None:
            g.loc[g.V, "w"] = g.loc[g.V, "blk"].map(wv).fillna(0)
            g.loc[g.N, "w"] = g.loc[g.N, "blk"].map(wn).fillna(0)
        gv = g[g.V].assign(wy=lambda q: q.w * q.y).groupby("st")[["w", "wy"]].sum()
        gn = g[g.N].assign(wy=lambda q: q.w * q.y).groupby("st")[["w", "wy"]].sum()
        j = gv.join(gn, lsuffix="_v", rsuffix="_n", how="inner")
        j = j[(j.w_v > 0) & (j.w_n > 0)]
        w = np.minimum(j.w_v, j.w_n)
        return float((w * (j.wy_v / j.w_v - j.wy_n / j.w_n)).sum() / w.sum())
    e = est()
    if not boot:
        return e, usable
    rng = np.random.default_rng(235); bv = np.unique(blocks[V & (st >= 0)]); bn = np.unique(blocks[N & (st >= 0)])
    reps = []
    for _ in range(boot):
        wv = pd.Series(np.bincount(rng.integers(0, len(bv), len(bv)), minlength=len(bv)), index=bv)
        wn = pd.Series(np.bincount(rng.integers(0, len(bn), len(bn)), minlength=len(bn)), index=bn)
        reps.append(est(wv, wn))
    return e, np.percentile(reps, [2.5, 97.5]), usable

print("primary 20/40 random (should be +0.0491):", round(run(d30, 20, 40)[0], 4))
print("tighter V only, 10/40 random:", round(run(d30, 10, 40)[0], 4), "| 15/40:", round(run(d30, 15, 40)[0], 4))
for o in ("suit_random", "suit_tgb", "suit_hybrid"):
    e, ci, _ = run(d30, 20, 50, outcome=o, boot=200); print(f"20/50 {o}: {e:+.4f} CI {ci.round(3)}")
for o in ("suit_random", "suit_tgb", "suit_hybrid"):
    e, ci, _ = run(d13, 20, 40, outcome=o, boot=200); print(f"13-centre 20/40 {o}: {e:+.4f} CI {ci.round(3)}")
print("+river_dist matched, random:", round(run(d30, 20, 40, cols=("slope", "tri", "twi", "river_dist"))[0], 4))
print("+river_dist, 5 bins, random:", round(run(d30, 20, 40, cols=("slope", "tri", "twi", "river_dist"), nb=5)[0], 4))
V, N = d30 <= 20, d30 >= 40
lo = elev < 1000
fr = (m["n_designs_top10"] == 1)
print(f"fringe rate <1000 m: V {fr[V & lo].mean()*100:.2f} %  N {fr[N & lo].mean()*100:.2f} %")
ad = np.abs(m["suit_hybrid"].astype(float) - m["suit_random"].astype(float))
print(f"mean |hybrid-random| <1000 m: V {ad[V & lo].mean():.4f}  N {ad[N & lo].mean():.4f}")
# Where does N land above 400 m sit? Rough regions by longitude (UTM -> lon).
inv = pyproj.Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)
lon, lat = inv.transform(f.x.to_numpy(), f.y.to_numpy())
sel = N & (elev >= 400) & (elev < 1000)
print("N cells 400-1000 m:", sel.sum(), "| south of 7.9°S:", round((lat[sel] < -7.9).mean() * 100, 1), "% | west of 112.2°E:", round((lon[sel] < 112.2).mean() * 100, 1), "%")
print("frame lon range:", round(lon.min(), 2), round(lon.max(), 2), "| cells east of 114.43°E (Bali side):", int((lon > 114.43).sum()))
