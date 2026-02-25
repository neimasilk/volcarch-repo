"""
E004 / Step 1: Site density vs volcanic proximity — First test of H1.

Requires:
  - data/processed/east_java_sites.geojson  (from E001)
  - Internet (for East Java province boundary from OSM, unless cached)

Produces:
  - results/density_by_distance.csv         — site count & density per distance band
  - results/correlation_stats.txt           — Spearman test results
  - results/map_sites_by_distance.html      — interactive Folium map
  - results/density_chart.png               — bar chart: density vs distance band

Run from repo root:
    python experiments/E004_density_analysis/01_analyze_density.py
"""

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# Check imports
try:
    import geopandas as gpd
    from shapely.geometry import Point, shape
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import folium
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent.parent
SITES_PATH = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
RESULTS_DIR = Path(__file__).parent / "results"

# Reference volcano coordinates (WGS84 lat/lon)
# Sources: GVP Smithsonian, Wikipedia
VOLCANOES = {
    "Kelud":           {"lat": -7.9300, "lon": 112.3080, "gvp_id": "263280"},
    "Semeru":          {"lat": -8.1080, "lon": 112.9220, "gvp_id": "263300"},
    "Arjuno-Welirang": {"lat": -7.7290, "lon": 112.5750, "gvp_id": "263260"},
    "Bromo":           {"lat": -7.9420, "lon": 112.9500, "gvp_id": "263310"},
    "Lamongan":        {"lat": -7.9770, "lon": 113.3430, "gvp_id": "263350"},
    "Raung":           {"lat": -8.1250, "lon": 114.0420, "gvp_id": "263340"},
    "Ijen":            {"lat": -8.0580, "lon": 114.2420, "gvp_id": "263350"},
}

# Distance bins (km from nearest volcano)
# BIN_EDGES defines the boundaries; BIN_LABELS has exactly len(BIN_EDGES)-1 entries
BIN_EDGES = [0, 25, 50, 75, 100, 150, 200, 10000]   # last bin = 200+ km (open ended)
BIN_LABELS = ["0-25", "25-50", "50-75", "75-100", "100-150", "150-200", "200+"]

# Colour palette for map (distance bands)
BIN_COLORS = ["#d73027", "#fc8d59", "#fee090", "#e0f3f8", "#91bfdb", "#4575b4", "#313695"]


def load_sites() -> gpd.GeoDataFrame:
    """Load the site dataset from E001 output."""
    if not SITES_PATH.exists():
        print(f"ERROR: Sites file not found: {SITES_PATH}")
        print("Run E001 first: python experiments/E001_site_density_vs_volcanic_proximity/01_collect_sites.py")
        sys.exit(1)

    gdf = gpd.read_file(SITES_PATH)
    original_len = len(gdf)

    # Drop sites with no geometry
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf.to_crs("EPSG:4326")

    # Filter to East Java bounding box (rough sanity check)
    jatim_bounds = (-9.0, 111.0, -6.5, 115.0)  # south, west, north, east
    gdf = gdf[
        (gdf.geometry.y >= jatim_bounds[0]) &
        (gdf.geometry.x >= jatim_bounds[1]) &
        (gdf.geometry.y <= jatim_bounds[2]) &
        (gdf.geometry.x <= jatim_bounds[3])
    ]

    print(f"Sites loaded: {original_len} total, {len(gdf)} in East Java bounds")
    return gdf


def build_volcano_gdf() -> gpd.GeoDataFrame:
    """Build GeoDataFrame for reference volcanoes."""
    rows = []
    for name, info in VOLCANOES.items():
        rows.append({
            "name": name,
            "lat": info["lat"],
            "lon": info["lon"],
            "gvp_id": info["gvp_id"],
            "geometry": Point(info["lon"], info["lat"]),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def compute_min_volcano_distance(sites: gpd.GeoDataFrame,
                                  volcs: gpd.GeoDataFrame) -> pd.Series:
    """
    For each site, compute distance (km) to nearest volcano.
    Uses projected CRS (UTM 49S) for accurate distance in meters.
    """
    sites_proj = sites.to_crs("EPSG:32749")
    volcs_proj = volcs.to_crs("EPSG:32749")

    min_dists = []
    nearest_names = []

    for _, site_row in sites_proj.iterrows():
        dists = volcs_proj.geometry.distance(site_row.geometry) / 1000  # km
        min_idx = dists.idxmin()
        min_dists.append(dists[min_idx])
        nearest_names.append(volcs_proj.loc[min_idx, "name"])

    return pd.Series(min_dists, index=sites.index), pd.Series(nearest_names, index=sites.index)


def get_jatim_area_polygon() -> gpd.GeoDataFrame | None:
    """
    Try to fetch East Java province polygon from Overpass API.
    Returns GeoDataFrame or None if unavailable.
    Used to compute accurate area per distance ring.
    """
    import requests
    query = """
[out:json][timeout:60];
relation["name"="Jawa Timur"]["boundary"="administrative"]["admin_level"="4"];
out geom;
"""
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=90,
            headers={"User-Agent": "VOLCARCH-research/0.1"},
        )
        resp.raise_for_status()
        data = resp.json()
        elements = data.get("elements", [])
        if not elements:
            return None

        # Build a rough polygon from the relation geometry
        from shapely.ops import unary_union, polygonize
        from shapely.geometry import MultiLineString

        lines = []
        for el in elements:
            if el.get("type") == "relation":
                for member in el.get("members", []):
                    if member.get("type") == "way" and "geometry" in member:
                        coords = [(p["lon"], p["lat"]) for p in member["geometry"]]
                        if len(coords) >= 2:
                            from shapely.geometry import LineString
                            lines.append(LineString(coords))

        if lines:
            polys = list(polygonize(unary_union(lines)))
            if polys:
                merged = unary_union(polys)
                gdf = gpd.GeoDataFrame({"geometry": [merged]}, crs="EPSG:4326")
                print(f"  East Java polygon fetched: {merged.geom_type}")
                return gdf

    except Exception as e:
        print(f"  Could not fetch Jawa Timur polygon: {e}")
    return None


def compute_area_per_bin(volcs: gpd.GeoDataFrame,
                          jatim_poly: gpd.GeoDataFrame | None) -> pd.Series:
    """
    Compute land area (km²) in each distance bin from any volcano.
    If jatim_poly is None, use a bounding box approximation.
    """
    from shapely.ops import unary_union

    volcs_proj = volcs.to_crs("EPSG:32749")

    if jatim_poly is not None:
        study_area = jatim_poly.to_crs("EPSG:32749").unary_union
    else:
        # Fallback: use Jawa Timur approximate bounding box (~47,799 km²)
        from shapely.geometry import box
        # UTM 49S approx bounding box for Jawa Timur
        study_area = box(4.9e5, 9.06e6, 8.8e5, 9.26e6)
        print("  WARNING: Using bounding box approximation for study area.")
        print("  Area calculations will be rough. For accuracy, fix OSM polygon fetch.")

    total_study_area_km2 = study_area.area / 1e6
    print(f"  Study area: {total_study_area_km2:.0f} km²")

    # Build union of all volcano buffer rings
    volcano_union = unary_union(volcs_proj.geometry.values)

    areas = []
    # BIN_EDGES has one more entry than BIN_LABELS (e.g., 8 edges, 7 bins).
    # The last edge (10000 km) acts as "open infinity" for the 200+ bin.
    bin_edges_m = [b * 1000 for b in BIN_EDGES]  # convert km to meters

    for i in range(len(BIN_LABELS)):
        lower_m = bin_edges_m[i]
        upper_m = bin_edges_m[i + 1]
        outer = volcano_union.buffer(upper_m)
        inner = volcano_union.buffer(lower_m) if lower_m > 0 else None

        ring = outer if inner is None else outer.difference(inner)
        ring_clipped = ring.intersection(study_area)
        area_km2 = ring_clipped.area / 1e6
        areas.append(area_km2)

    return pd.Series(areas, index=BIN_LABELS)


def assign_distance_bins(min_dist_km: pd.Series) -> pd.Series:
    """Assign each site to a distance bin."""
    bins = BIN_EDGES
    labels = BIN_LABELS
    return pd.cut(min_dist_km, bins=bins, labels=labels, right=False)


def run_correlation_test(sites: gpd.GeoDataFrame,
                          density_df: pd.DataFrame) -> dict:
    """
    Spearman correlation between site density and distance-from-volcano midpoint.
    Tests H1: more sites farther from volcanoes (positive rho expected).
    """
    # Use midpoint of each distance bin
    bin_midpoints = [12.5, 37.5, 62.5, 87.5, 125.0, 175.0, 250.0]
    density = density_df["density_per_1000km2"].values
    dist = np.array(bin_midpoints[:len(density)])

    # Only include bins with non-zero area
    mask = density_df["area_km2"].values > 0
    density_valid = density[mask]
    dist_valid = dist[mask]

    rho, pval = spearmanr(dist_valid, density_valid)

    result = {
        "spearman_rho": rho,
        "p_value": pval,
        "n_bins": int(mask.sum()),
        "total_sites": int(len(sites)),
        "h1_supported": rho > 0 and pval < 0.05,
        "interpretation": (
            f"rho = {rho:.3f}, p = {pval:.4f}. "
            + ("H1 SUPPORTED: More sites farther from volcanoes (p < 0.05)."
               if rho > 0 and pval < 0.05
               else "H1 NOT SUPPORTED at p < 0.05. "
                    + ("Positive trend but not significant." if rho > 0 else "No positive trend."))
        ),
    }
    return result


def plot_density_chart(density_df: pd.DataFrame, stats: dict, output_path: Path) -> None:
    """Bar chart of site density by distance band."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(density_df))
    bars = ax.bar(x, density_df["density_per_1000km2"], color=BIN_COLORS, edgecolor="white", linewidth=0.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{lab} km" for lab in density_df["bin"]], rotation=30, ha="right")
    ax.set_ylabel("Known archaeological sites per 1,000 km²")
    ax.set_xlabel("Distance from nearest active volcano")
    ax.set_title(
        "Archaeological Site Density vs Volcanic Proximity — East Java\n"
        f"Spearman rho = {stats['spearman_rho']:.3f}, p = {stats['p_value']:.4f}  |  "
        f"n = {stats['total_sites']} sites"
    )

    # Add count labels on bars
    for bar, (_, row) in zip(bars, density_df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"n={int(row['site_count'])}",
            ha="center", va="bottom", fontsize=8
        )

    ax.set_ylim(0, density_df["density_per_1000km2"].max() * 1.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {output_path}")


def build_folium_map(sites: gpd.GeoDataFrame,
                      volcs: gpd.GeoDataFrame,
                      output_path: Path) -> None:
    """Interactive Folium map: sites colored by distance band, volcanoes marked."""
    m = folium.Map(location=[-7.8, 112.7], zoom_start=8, tiles="CartoDB positron")

    # Add volcano markers
    for _, row in volcs.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            tooltip=f"🌋 {row['name']}",
            icon=folium.Icon(color="red", icon="fire", prefix="fa"),
        ).add_to(m)

    # Add distance ring circles (visual only)
    for dist_km, color in zip([25, 50, 100], ["#d73027", "#fc8d59", "#fee090"]):
        for _, vrow in volcs.iterrows():
            folium.Circle(
                location=[vrow["lat"], vrow["lon"]],
                radius=dist_km * 1000,
                color=color,
                fill=False,
                weight=1,
                opacity=0.4,
                tooltip=f"{dist_km} km from {vrow['name']}",
            ).add_to(m)

    # Add sites colored by distance bin
    bin_labels_list = BIN_LABELS
    for _, row in sites.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        dist_bin = row.get("distance_bin", "unknown")
        try:
            color_idx = bin_labels_list.index(str(dist_bin))
            color = BIN_COLORS[color_idx]
        except (ValueError, IndexError):
            color = "gray"

        name = row.get("name") or "Unnamed site"
        site_type = row.get("type") or "unknown"
        period = row.get("period") or "unknown"
        dist_km = row.get("min_dist_km")
        dist_str = f"{dist_km:.1f} km" if dist_km is not None else "?"

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=f"{name} | {site_type} | {period} | {dist_str} from volcano",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px; border:2px solid #ccc; font-size:12px;">
    <b>Distance from nearest volcano</b><br>
    """ + "".join(
        f'<span style="background:{c};display:inline-block;width:12px;height:12px;margin-right:4px;"></span>{l} km<br>'
        for l, c in zip(BIN_LABELS, BIN_COLORS)
    ) + "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"  Map saved: {output_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("E004: Site Density vs Volcanic Proximity")
    print("=" * 60)

    # 1. Load data
    print("\nLoading data...")
    sites = load_sites()
    volcs = build_volcano_gdf()

    # 2. Compute distances
    print("\nComputing distances to volcanoes...")
    min_dist, nearest_volc = compute_min_volcano_distance(sites, volcs)
    sites = sites.copy()
    sites["min_dist_km"] = min_dist.values
    sites["nearest_volcano"] = nearest_volc.values
    sites["distance_bin"] = assign_distance_bins(sites["min_dist_km"])

    # 3. Fetch study area polygon for area calculations
    print("\nFetching East Java boundary for area calculations...")
    jatim_poly = get_jatim_area_polygon()

    # 4. Compute area per bin
    print("\nComputing area per distance bin...")
    area_per_bin = compute_area_per_bin(volcs, jatim_poly)

    # 5. Count sites per bin
    site_counts = sites.groupby("distance_bin", observed=True).size()

    density_df = pd.DataFrame({
        "bin": BIN_LABELS,
        "site_count": [site_counts.get(b, 0) for b in BIN_LABELS],
        "area_km2": area_per_bin.values,
    })
    # Sites per 1000 km²
    density_df["density_per_1000km2"] = np.where(
        density_df["area_km2"] > 0,
        density_df["site_count"] / density_df["area_km2"] * 1000,
        0,
    )

    print("\nDensity by distance band:")
    print(density_df.to_string(index=False))
    density_df.to_csv(RESULTS_DIR / "density_by_distance.csv", index=False)

    # 6. Correlation test
    print("\nRunning Spearman correlation test...")
    stats = run_correlation_test(sites, density_df)

    print(f"\n  {stats['interpretation']}")
    print(f"  H1 {'SUPPORTED' if stats['h1_supported'] else 'NOT SUPPORTED'} at p < 0.05")

    stats_text = textwrap.dedent(f"""
        E004 — Correlation Test Results
        ================================
        Date: 2026-02-23
        Total sites analyzed: {stats['total_sites']}
        Distance bins with data: {stats['n_bins']}

        Spearman rho (density vs distance from volcano): {stats['spearman_rho']:.4f}
        p-value: {stats['p_value']:.6f}

        Interpretation:
        {stats['interpretation']}

        MVR: rho > 0.5 and p < 0.05
        MVR {'MET' if stats['h1_supported'] and stats['spearman_rho'] > 0.5 else 'NOT MET'}

        H1 (Taphonomic Bias): {'SUPPORTED' if stats['h1_supported'] else 'NOT SUPPORTED'}

        Notes:
        - This is a distance-based proxy for volcanic deposition.
          A stronger test would use actual tephra thickness estimates (E002 data).
        - OSM site data is known to be incomplete. Undercounting is expected.
        - Sites without coordinates are excluded ({len(sites)} geocoded out of full dataset).
    """).strip()

    print("\n" + stats_text)
    (RESULTS_DIR / "correlation_stats.txt").write_text(stats_text, encoding="utf-8")

    # 7. Chart
    print("\nGenerating charts...")
    plot_density_chart(density_df, stats, RESULTS_DIR / "density_chart.png")

    # 8. Map
    print("Generating interactive map...")
    build_folium_map(sites, volcs, RESULTS_DIR / "map_sites_by_distance.html")

    print(f"\nAll results saved to {RESULTS_DIR}")
    print("\nNext steps:")
    if not stats['h1_supported']:
        print("  - H1 not supported. Check: is OSM data too sparse? Consider Wikipedia supplement.")
        print("  - Consider E005: control for terrain suitability before concluding bias absence.")
    else:
        print("  - H1 supported! Draft Paper 1 with these results as core quantitative finding.")
        print("  - Run with Wikipedia supplement data for stronger site count.")


if __name__ == "__main__":
    main()
