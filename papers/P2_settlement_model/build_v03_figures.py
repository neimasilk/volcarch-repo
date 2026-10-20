"""Round-2 revision figures for P2 (JCAA #280): Figure 1 (study area) and Figure 5 (priority map).

Reviewer requests (REVISION_JCAA_R2_20261019.md): R1-7/R2-2 — Figure 1 must be a real map and show topography;
R1-8 — Figure 5's legend must not sit on the map. The v0.2 figures stay as submitted (build_v02_figures.py).

    python papers/P2_settlement_model/build_v03_figures.py

Outputs (figures/): fig01_study_area_v03.png, fig05_priority_map_v03.png
Data: data/processed/dem/jatim_dem.tif (the model's own DEM), data/processed/east_java_sites.geojson,
data/processed/dashboard/volcanoes_java_full.csv, E221 priority sets, data/raw/naturalearth/ne_50m_land.zip (inset only).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import matplotlib.patheffects as pe                  # noqa: E402
from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap   # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import Patch, Rectangle      # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v02", HERE / "build_v02_figures.py")
v02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v02)                         # colours + rcParams of the submitted figure set

REPO, EXP, DATA, FIG = v02.REPO, v02.EXP, v02.DATA, v02.FIG
UTM = "EPSG:32749"
ZONE_V_KM, ZONE_N_KM = 20.0, 40.0                    # E235 primary definition (DESIGN §3)
SEA = "#dde9f2"
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

# Hand-placed label offsets (points) so the dense central cluster does not collide.
LABEL_OFFSET = {
    "Lawu": (7, 5), "Wilis": (7, -9), "Kelud": (-34, -11), "Kawi-Butak": (2, -14),
    "Arjuno-Welirang": (-86, -2), "Penanggungan": (7, 5), "Bromo-Tengger": (-44, 12), "Semeru": (6, -11),
    "Lamongan": (-22, 10), "Iyang-Argapura": (5, -12), "Raung": (-30, -12), "Ijen": (6, -10), "Baluran": (6, 5),
}


def read_dem(factor: int):
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(DATA / "dem" / "jatim_dem.tif") as src:
        h, w = src.height // factor, src.width // factor
        arr = src.read(1, out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
        nod = src.nodata
        b = src.bounds
    if nod is not None:
        arr[arr == nod] = np.nan
    return arr, (b.left, b.right, b.bottom, b.top)


def centres_utm():
    import pyproj
    v = pd.read_csv(DATA / "dashboard" / "volcanoes_java_full.csv")
    tr = pyproj.Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    v["x"], v["y"] = tr.transform(v["lon"].to_numpy(), v["lat"].to_numpy())
    inb = (v.lon >= 111) & (v.lon <= 115) & (v.lat >= -9) & (v.lat <= -6.5)
    return v, v[inb]


def sites_utm(extent):
    import geopandas as gpd
    s = gpd.read_file(DATA / "east_java_sites.geojson")
    s = s[s.geometry.notna() & ~s.geometry.is_empty]
    s = s.cx[111:115, -9:-6.5].to_crs(UTM)       # same selection as the submitted Figure 1
    return s.geometry.x.to_numpy(), s.geometry.y.to_numpy()


def graticule(ax, extent, lons=np.arange(111.0, 115.01, 0.5), lats=np.arange(-9.0, -6.49, 0.5)):
    """Draw meridians/parallels projected into UTM and label them where they cross the frame."""
    import pyproj
    tr = pyproj.Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    left, right, bottom, top = extent
    xt, xl, yt, yl = [], [], [], []
    for lon in lons:
        la = np.linspace(-9.5, -6.0, 200)
        x, y = tr.transform(np.full_like(la, lon), la)
        ax.plot(x, y, color="0.35", lw=0.4, alpha=0.5, zorder=3)
        xb = np.interp(bottom, y, x)
        if left < xb < right:
            xt.append(xb); xl.append(f"{lon:g}°E")
    for lat in lats:
        lo = np.linspace(110.5, 115.5, 200)
        x, y = tr.transform(lo, np.full_like(lo, lat))
        ax.plot(x, y, color="0.35", lw=0.4, alpha=0.5, zorder=3)
        yb = np.interp(left, x, y)
        if bottom < yb < top:
            yt.append(yb); yl.append(f"{abs(lat):g}°S")
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.set_xlim(left, right); ax.set_ylim(bottom, top)
    ax.tick_params(length=3, labelsize=8)


def scale_bar(ax, extent, km=50):
    left, right, bottom, top = extent
    x0, y0 = left + 0.03 * (right - left), bottom + 0.06 * (top - bottom)
    ax.add_patch(Rectangle((x0 - 4000, y0 - 7000), km * 1000 + 8000, 19000, fc="white", ec="none", alpha=0.85, zorder=5))
    ax.plot([x0, x0 + km * 1000], [y0, y0], color="k", lw=2.5, solid_capstyle="butt", zorder=6)
    for xx in (x0, x0 + km * 500, x0 + km * 1000):
        ax.plot([xx, xx], [y0, y0 + 2500], color="k", lw=1, zorder=6)
    ax.text(x0 + km * 500, y0 + 4000, f"{km} km", ha="center", va="bottom", fontsize=8, zorder=6)
    ax.annotate("N", xy=(x0 + km * 1000 + 22000, y0 + 16000), xytext=(x0 + km * 1000 + 22000, y0 - 1000),
                ha="center", va="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2), zorder=6)


def coastline(ax, dem, extent, **kw):
    land = np.where(np.isfinite(dem) & (dem > 0), 1.0, 0.0)
    left, right, bottom, top = extent
    xs = np.linspace(left, right, dem.shape[1]); ys = np.linspace(top, bottom, dem.shape[0])
    ax.contour(xs, ys, land, levels=[0.5], colors=kw.get("color", "0.25"), linewidths=kw.get("lw", 0.6), zorder=4)
    return land


def fig_study_area() -> Path:
    from scipy.spatial import cKDTree
    dem, extent = read_dem(factor=4)
    left, right, bottom, top = extent
    land = np.isfinite(dem) & (dem > 0)
    z = np.where(land, dem, 0.0)

    hyps = LinearSegmentedColormap.from_list("hyps", [
        (0.00, "#5d8c4f"), (0.04, "#8fb36a"), (0.12, "#d8d08f"), (0.27, "#c79f6a"),
        (0.50, "#9a7353"), (0.75, "#8c8078"), (1.00, "#f4f4f2")])
    vmax = 3700.0
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(z, cmap=hyps, vmin=0, vmax=vmax, blend_mode="soft", vert_exag=0.08, dx=122.7, dy=122.7)
    rgb[~land, :3] = matplotlib.colors.to_rgb(SEA)

    fig = plt.figure(figsize=(9.6, 4.95))
    ax = fig.add_axes([0.06, 0.09, 0.70, 0.88])
    ax.imshow(rgb, extent=extent, origin="upper", zorder=1, interpolation="bilinear")
    coastline(ax, dem, extent)

    # E235 zones: distance to all 30 canonical centres, contoured on a coarse grid.
    vall, v13 = centres_utm()
    gx = np.linspace(left, right, 400); gy = np.linspace(bottom, top, 260)
    GX, GY = np.meshgrid(gx, gy)
    dist = cKDTree(np.column_stack([vall.x, vall.y])).query(np.column_stack([GX.ravel(), GY.ravel()]))[0] / 1000
    dist = dist.reshape(GX.shape)
    ax.contour(GX, GY, dist, levels=[ZONE_V_KM], colors="#8b1a1a", linewidths=1.0, zorder=4)
    ax.contour(GX, GY, dist, levels=[ZONE_N_KM], colors="#1f3f7a", linewidths=1.0, linestyles="--", zorder=4)

    sx, sy = sites_utm(extent)
    ax.scatter(sx, sy, s=9, c="black", edgecolors="white", linewidths=0.4, zorder=6)
    ax.scatter(v13.x, v13.y, s=58, marker="^", c=v02.VOLC_COL, edgecolors="black", linewidths=0.6, zorder=7)
    for _, r in v13.iterrows():
        dx, dy = LABEL_OFFSET.get(r["name"], (6, 6))
        ax.annotate(r["name"], (r.x, r.y), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    fontweight="bold", color="#5a1010", path_effects=HALO, zorder=8)
    graticule(ax, extent)
    scale_bar(ax, extent)
    ax.set_aspect("equal")

    # Right column: locator inset, legend, elevation scale.
    import geopandas as gpd
    ins = fig.add_axes([0.785, 0.66, 0.20, 0.30])
    ne = gpd.read_file("zip://" + str(REPO / "data" / "raw" / "naturalearth" / "ne_50m_land.zip"))
    ne.cx[94:142, -12:8].plot(ax=ins, color="#c9c6bb", edgecolor="#8a877d", linewidth=0.3)
    import pyproj
    inv = pyproj.Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    lo0, la0 = inv.transform(left, bottom); lo1, la1 = inv.transform(right, top)
    ins.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fc="none", ec=v02.VOLC_COL, lw=1.4))
    ins.set_xlim(94, 142); ins.set_ylim(-12, 8); ins.set_aspect("equal")
    ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor(SEA)
    ins.text(96, 5.5, "Indonesia", fontsize=7, color="0.25")
    ins.text(lo1 + 0.8, la0 - 2.6, "study area", fontsize=7, color=v02.VOLC_COL)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="black", markeredgecolor="white", markersize=5,
               label="archaeological presences"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=v02.VOLC_COL, markeredgecolor="black",
               markersize=7, label="volcanic centres (13 in bounds)"),
        Line2D([0], [0], color="#8b1a1a", lw=1.0, label=f"{ZONE_V_KM:.0f} km from a volcanic centre"),
        Line2D([0], [0], color="#1f3f7a", lw=1.0, ls="--", label=f"{ZONE_N_KM:.0f} km from a volcanic centre"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.775, 0.63), frameon=False, fontsize=7.5)
    cax = fig.add_axes([0.80, 0.11, 0.018, 0.28])
    sm = plt.cm.ScalarMappable(cmap=hyps, norm=matplotlib.colors.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("elevation (m)", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.text(0.06, 0.012, "Elevation: Copernicus GLO-30 DEM, as used by the model. Inset: Natural Earth.", fontsize=6.5, color="0.35")

    out = FIG / "fig01_study_area_v03.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig_priority_map() -> Path:
    d = np.load(EXP / "E221_seed_ensemble_stability/results/e221_priority_sets_xgboost.npz")
    x, y, nd = d["x"], d["y"], d["n_designs_top10"]
    # The E221 cells sit on the decimated DEM lattice: rebuild the raster instead of scattering points.
    dx = np.median(np.diff(np.unique(x))); dy = np.median(np.diff(np.unique(y)))
    col = np.rint((x - x.min()) / dx).astype(int); row = np.rint((y.max() - y) / dy).astype(int)
    grid = np.full((row.max() + 1, col.max() + 1), np.nan)
    grid[row, col] = nd
    gext = (x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2)

    dem, extent = read_dem(factor=4)
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.19)
    ax.set_facecolor(SEA)
    land = np.isfinite(dem) & (dem > 0)
    ax.imshow(np.where(land, 1.0, np.nan), extent=extent, origin="upper", cmap=ListedColormap(["#f3f2ee"]), zorder=1)
    cmap = ListedColormap([v02.NONE_GRAY, v02.BLUE_LIGHT, v02.BLUE_MID, v02.BLUE_DARK])
    ax.imshow(np.ma.masked_invalid(grid), extent=gext, origin="upper", cmap=cmap, vmin=-0.5, vmax=3.5,
              interpolation="nearest", zorder=2)
    coastline(ax, dem, extent)
    sx, sy = sites_utm(extent)
    ax.scatter(sx, sy, s=7, c=v02.INK, alpha=0.6, linewidths=0, zorder=5)
    _, v13 = centres_utm()
    ax.scatter(v13.x, v13.y, s=50, marker="^", c=v02.VOLC_COL, edgecolors="black", linewidths=0.6, zorder=6)
    graticule(ax, extent)
    scale_bar(ax, extent)
    ax.set_aspect("equal")
    handles = [
        Patch(facecolor=v02.BLUE_DARK, label="robust core: top decile under all 3 designs"),
        Patch(facecolor=v02.BLUE_MID, label="top decile under 2 designs"),
        Patch(facecolor=v02.BLUE_LIGHT, label="contingent fringe: top decile under 1 design"),
        Patch(facecolor=v02.NONE_GRAY, label="not top decile under any design"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=v02.INK, markersize=5, label="known sites"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=v02.VOLC_COL, markeredgecolor="black",
               markersize=7, label="volcanic centres"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.53, 0.0))
    out = FIG / "fig05_priority_map_v03.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for f in (fig_study_area, fig_priority_map):
        p = f()
        print(f"  {p.name}: {p.stat().st_size / 1024:.0f} KB")
