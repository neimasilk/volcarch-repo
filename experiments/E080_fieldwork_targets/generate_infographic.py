#!/usr/bin/env python3
"""
generate_infographic.py — VOLCARCH Fieldwork Target Convergence Map
Overlays E080 fieldwork targets, E097 anomaly detections, known sites, and volcanoes.
Highlights overlap zones where E097 anomalies fall within 5 km of E080 targets.

Output: experiments/E080_fieldwork_targets/results/fieldwork_infographic.html
"""

import math
import json
from pathlib import Path

import pandas as pd
import geopandas as gpd
import folium
from folium import IFrame

# ── paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent
E080_CSV = REPO / "experiments" / "E080_fieldwork_targets" / "results" / "top20_targets.csv"
E097_CSV = REPO / "experiments" / "E097_anomaly_detection" / "results" / "top50_anomaly_cells.csv"
SITES_GEOJSON = REPO / "data" / "processed" / "east_java_sites.geojson"
OUT_HTML = REPO / "experiments" / "E080_fieldwork_targets" / "results" / "fieldwork_infographic.html"

# ── volcanoes ────────────────────────────────────────────────────────────────
VOLCANOES = {
    "Kelud":            (-7.9300, 112.3080),
    "Semeru":           (-8.1080, 112.9220),
    "Arjuno-Welirang":  (-7.7290, 112.5750),
    "Bromo":            (-7.9420, 112.9500),
    "Lamongan":         (-7.9770, 113.3430),
    "Raung":            (-8.1250, 114.0420),
    "Ijen":             (-8.0580, 114.2420),
}

# ── helper: haversine distance (km) ─────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_map():
    # ── load data ────────────────────────────────────────────────────────────
    e080 = pd.read_csv(E080_CSV)
    e097 = pd.read_csv(E097_CSV)
    sites = gpd.read_file(SITES_GEOJSON)

    # ── base map ─────────────────────────────────────────────────────────────
    m = folium.Map(
        location=[-7.85, 112.75],
        zoom_start=9,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # ── title bar ────────────────────────────────────────────────────────────
    title_html = """
    <div style="
        position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
        z-index: 9999; background: rgba(255,255,255,0.92);
        border: 2px solid #333; border-radius: 8px;
        padding: 8px 24px; font-family: 'Segoe UI', Arial, sans-serif;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25); pointer-events: none;">
        <b style="font-size:14px; color:#222;">
            VOLCARCH Fieldwork Target Convergence Map: E080 Targets &times; E097 Anomaly Detection
        </b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # ── feature groups (for layer control) ───────────────────────────────────
    fg_sites   = folium.FeatureGroup(name="Known archaeological sites (blue)", show=True)
    fg_e080    = folium.FeatureGroup(name="E080 fieldwork targets (green)", show=True)
    fg_e097    = folium.FeatureGroup(name="E097 anomaly detections (red)", show=True)
    fg_volc    = folium.FeatureGroup(name="Volcanoes", show=True)
    fg_overlap = folium.FeatureGroup(name="Overlap zones (5 km)", show=True)

    # ── 3. known archaeological sites — small blue dots ──────────────────────
    for _, row in sites.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        lat, lon = geom.y, geom.x
        name = row.get("name", "Unknown site")
        stype = row.get("type", "")
        period = row.get("period", "")
        popup_text = (
            f"<b>{name}</b><br>"
            f"Type: {stype}<br>"
            f"Period: {period}<br>"
            f"Lat: {lat:.4f}, Lon: {lon:.4f}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#2166ac",
            fill=True,
            fill_color="#4393c3",
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup(popup_text, max_width=250),
        ).add_to(fg_sites)

    # ── 1. E080 fieldwork targets — large green markers ──────────────────────
    for i, row in e080.iterrows():
        popup_html = (
            f"<div style='font-family:Arial;font-size:12px;'>"
            f"<b style='color:#1a7a1a;'>E080 Target #{i+1}</b><br>"
            f"<hr style='margin:4px 0'>"
            f"Composite: <b>{row['composite_score']:.3f}</b><br>"
            f"Volc: {row['volc_score']:.2f} | Candi: {row['candi_score']:.2f}<br>"
            f"Gap: {row['gap_score']:.2f} | Terrain: {row['terrain_score']:.2f}<br>"
            f"<hr style='margin:4px 0'>"
            f"Nearest volcano: <b>{row['nearest_volcano']}</b> ({row['dist_volcano_km']:.1f} km)<br>"
            f"Nearest candi: <b>{row['nearest_candi']}</b> ({row['dist_candi_km']:.1f} km)<br>"
            f"Est. burial: <b>{row['estimated_burial_m']:.1f} m</b><br>"
            f"<hr style='margin:4px 0'>"
            f"Lat: {row['lat']:.4f}, Lon: {row['lon']:.4f}"
            f"</div>"
        )
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=320),
            icon=folium.Icon(color="green", icon="bullseye", prefix="fa"),
        ).add_to(fg_e080)

    # ── 2. E097 anomaly detections — red circle markers ──────────────────────
    for i, row in e097.iterrows():
        popup_html = (
            f"<div style='font-family:Arial;font-size:12px;'>"
            f"<b style='color:#c62828;'>E097 Anomaly #{i+1}</b><br>"
            f"<hr style='margin:4px 0'>"
            f"Composite: <b>{row['composite_score']:.4f}</b><br>"
            f"Elevation: {row['elevation']:.1f} m | Slope: {row['slope']:.2f}&deg;<br>"
            f"TWI: {row['twi']:.2f} | TRI: {row['tri']:.2f}<br>"
            f"Site likeness: {row['site_likeness']:.4f}<br>"
            f"Burial depth: <b>{row['burial_depth_cm']:.1f} cm</b><br>"
            f"Volcano dist: {row['volcano_dist_km']:.2f} km<br>"
            f"<hr style='margin:4px 0'>"
            f"Lat: {row['lat']:.5f}, Lon: {row['lon']:.5f}"
            f"</div>"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7,
            color="#c62828",
            fill=True,
            fill_color="#ef5350",
            fill_opacity=0.8,
            weight=2,
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(fg_e097)

    # ── 4. volcanoes — triangle icon via DivIcon ─────────────────────────────
    for vname, (vlat, vlon) in VOLCANOES.items():
        # Custom triangle icon using HTML/CSS
        icon_html = (
            '<div style="font-size:22px; color:#ff6f00; text-shadow: 0 0 3px #000;">'
            '&#9650;</div>'  # filled triangle (black outline via text-shadow)
        )
        folium.Marker(
            location=[vlat, vlon],
            popup=folium.Popup(
                f"<b style='color:#d84315;'>&#x1f30b; {vname}</b><br>"
                f"Lat: {vlat:.4f}, Lon: {vlon:.4f}",
                max_width=220,
            ),
            icon=folium.DivIcon(
                html=icon_html,
                icon_size=(28, 28),
                icon_anchor=(14, 24),
            ),
        ).add_to(fg_volc)

        # Subtle label
        folium.Marker(
            location=[vlat - 0.025, vlon],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:10px; font-weight:bold; color:#bf360c;'
                    f' text-shadow:1px 1px 2px #fff; white-space:nowrap;'
                    f' text-align:center; transform:translateX(-50%);">'
                    f'{vname}</div>'
                ),
                icon_size=(100, 20),
                icon_anchor=(50, 10),
            ),
        ).add_to(fg_volc)

    # ── 5. overlap zones — one circle per E080 target that has E097 neighbours ─
    OVERLAP_RADIUS_KM = 5.0
    total_pairs = 0
    circles_drawn = 0
    for _, t in e080.iterrows():
        nearby_count = 0
        min_dist = float("inf")
        for _, a in e097.iterrows():
            d = haversine_km(t["lat"], t["lon"], a["lat"], a["lon"])
            if d <= OVERLAP_RADIUS_KM:
                nearby_count += 1
                min_dist = min(min_dist, d)
        if nearby_count > 0:
            total_pairs += nearby_count
            circles_drawn += 1
            folium.Circle(
                location=[t["lat"], t["lon"]],
                radius=OVERLAP_RADIUS_KM * 1000,  # metres
                color="#ff9800",
                fill=True,
                fill_color="#ffcc80",
                fill_opacity=0.18,
                weight=2,
                dash_array="6 4",
                popup=folium.Popup(
                    f"<b>Convergence zone</b><br>"
                    f"E080 target &harr; <b>{nearby_count}</b> E097 anomalies<br>"
                    f"Nearest anomaly: {min_dist:.2f} km<br>"
                    f"Centre: {t['lat']:.4f}, {t['lon']:.4f}",
                    max_width=260,
                ),
            ).add_to(fg_overlap)

    print(f"  E080-E097 convergence pairs: {total_pairs}")
    print(f"  Overlap circles drawn: {circles_drawn}")

    # ── add feature groups ───────────────────────────────────────────────────
    fg_overlap.add_to(m)   # bottom layer
    fg_sites.add_to(m)
    fg_e097.add_to(m)
    fg_e080.add_to(m)
    fg_volc.add_to(m)

    # ── layer control ────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)

    # ── legend ───────────────────────────────────────────────────────────────
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 20px; z-index: 9999;
        background: rgba(255,255,255,0.94); border: 2px solid #555;
        border-radius: 8px; padding: 12px 16px;
        font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3); line-height: 1.8;
        max-width: 260px;">
        <b style="font-size:13px;">Legend</b><br>
        <span style="color:#1a7a1a; font-size:16px;">&#x25CF;</span>
            E080 Fieldwork Targets (20)<br>
        <span style="color:#c62828; font-size:16px;">&#x25CF;</span>
            E097 Anomaly Detections (50)<br>
        <span style="color:#4393c3; font-size:12px;">&#x25CF;</span>
            Known Archaeological Sites<br>
        <span style="color:#ff6f00; font-size:16px;">&#9650;</span>
            Volcanoes<br>
        <span style="display:inline-block; width:16px; height:16px;
               background:rgba(255,204,128,0.4); border:2px dashed #ff9800;
               border-radius:50%; vertical-align:middle;"></span>
            Convergence Zone (5 km)<br>
        <hr style="margin:4px 0;">
        <span style="font-size:10px; color:#666;">
            VOLCARCH Project &mdash; E080 &times; E097
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── save ─────────────────────────────────────────────────────────────────
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"\nMap saved to: {OUT_HTML}")
    print(f"  E080 targets plotted: {len(e080)}")
    print(f"  E097 anomalies plotted: {len(e097)}")
    print(f"  Known sites plotted: {len(sites)}")
    print(f"  Volcanoes plotted: {len(VOLCANOES)}")


if __name__ == "__main__":
    build_map()
