"""
VOLCARCH Interactive Prediction Map
====================================
Combines archaeological sites, volcanic systems, candi locations,
burial depth estimates, and fieldwork targets into a shareable web map.

From Dissemination Roadmap item 2C.
"""

import folium
from folium import plugins
import pandas as pd
import numpy as np
import json
from pathlib import Path

print("Building VOLCARCH interactive prediction map...")

# ============================================================
# 1. Load data
# ============================================================

# Archaeological sites
try:
    import geopandas as gpd
    sites_gdf = gpd.read_file("D:/documents/volcarch-repo/data/processed/east_java_sites.geojson")
    print(f"  Sites: {len(sites_gdf)} loaded")
except Exception as e:
    print(f"  Sites: error loading GeoJSON ({e}), trying CSV fallback")
    sites_gdf = None

# Candi locations
candi_df = pd.read_csv("D:/documents/volcarch-repo/experiments/E031_candi_orientation/results/candi_volcano_pairs.csv")
print(f"  Candi: {len(candi_df)} loaded")

# Inscriptions
insc_df = pd.read_csv("D:/documents/volcarch-repo/experiments/E082_inscription_georeferencing/results/geocoded_inscriptions.csv")
print(f"  Inscriptions: {len(insc_df)} loaded")

# Volcanoes (major East Java)
volcanoes = [
    {"name": "Kelud", "lat": -7.93, "lon": 112.31, "eruptions": 37},
    {"name": "Semeru", "lat": -8.108, "lon": 112.922, "eruptions": 63},
    {"name": "Arjuno-Welirang", "lat": -7.732, "lon": 112.578, "eruptions": 12},
    {"name": "Bromo/Tengger", "lat": -7.942, "lon": 112.95, "eruptions": 50},
    {"name": "Merapi", "lat": -7.54, "lon": 110.446, "eruptions": 68},
    {"name": "Sundoro", "lat": -7.30, "lon": 109.992, "eruptions": 6},
    {"name": "Lawu", "lat": -7.625, "lon": 111.192, "eruptions": 3},
]

# Fieldwork targets (from E080)
targets = [
    {"name": "Target 1: Kelud W flank", "lat": -7.95, "lon": 112.18, "priority": "HIGH", "method": "GPR+ERT"},
    {"name": "Target 2: Kelud SW", "lat": -8.02, "lon": 112.22, "priority": "HIGH", "method": "GPR"},
    {"name": "Target 3: Arjuno N", "lat": -7.68, "lon": 112.55, "priority": "HIGH", "method": "GPR+ERT"},
    {"name": "Target 4: Singosari area", "lat": -7.89, "lon": 112.67, "priority": "HIGH", "method": "GPR"},
    {"name": "Target 5: Penanggungan W", "lat": -7.62, "lon": 112.55, "priority": "MEDIUM", "method": "GPR"},
    {"name": "Target 6: Lawu W", "lat": -7.64, "lon": 111.05, "priority": "MEDIUM", "method": "Borehole"},
    {"name": "Target 7: Merapi S (Sambisari)", "lat": -7.75, "lon": 110.44, "priority": "HIGH", "method": "ERT"},
    {"name": "Target 8: Sundoro E (Liangan)", "lat": -7.33, "lon": 110.08, "priority": "HIGH", "method": "Phytolith"},
]

# Calibration sites (known buried temples)
calibration = [
    {"name": "Dwarapala Singosari", "lat": -7.889, "lon": 112.639, "depth_m": 1.85, "rate": 3.5, "year_built": 1268},
    {"name": "Candi Sambisari", "lat": -7.752, "lon": 110.491, "depth_m": 5.75, "rate": 5.1, "year_built": 835},
    {"name": "Candi Kedulan", "lat": -7.697, "lon": 110.467, "depth_m": 6.50, "rate": 5.8, "year_built": 869},
    {"name": "Candi Kimpulan", "lat": -7.68, "lon": 110.42, "depth_m": 3.85, "rate": 3.5, "year_built": 900},
    {"name": "Candi Liangan", "lat": -7.318, "lon": 110.000, "depth_m": 7.00, "rate": None, "year_built": 800},
]

# ============================================================
# 2. Create map
# ============================================================

# Center on Java
m = folium.Map(
    location=[-7.6, 111.5],
    zoom_start=8,
    tiles='CartoDB positron',
    control_scale=True,
)

# Add satellite basemap option
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Satellite',
).add_to(m)

folium.TileLayer(
    tiles='OpenStreetMap',
    name='OpenStreetMap',
).add_to(m)

# ============================================================
# 3. Volcano layer
# ============================================================

volcano_group = folium.FeatureGroup(name="Volcanoes (active)", show=True)
for v in volcanoes:
    folium.CircleMarker(
        location=[v['lat'], v['lon']],
        radius=10,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.8,
        popup=folium.Popup(
            f"<b>{v['name']}</b><br>"
            f"Eruptions: {v['eruptions']}<br>"
            f"Lat: {v['lat']:.3f}, Lon: {v['lon']:.3f}",
            max_width=200
        ),
        tooltip=v['name'],
    ).add_to(volcano_group)

    # Add 15km radius (Zone A boundary)
    folium.Circle(
        location=[v['lat'], v['lon']],
        radius=15000,  # 15 km in meters
        color='red',
        fill=False,
        weight=1,
        opacity=0.4,
        dash_array='5 5',
        tooltip=f"{v['name']} Zone A (15 km)",
    ).add_to(volcano_group)

volcano_group.add_to(m)

# ============================================================
# 4. Candi layer
# ============================================================

candi_group = folium.FeatureGroup(name="Candi (142 temples)", show=True)
for _, row in candi_df.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=4,
        color='orange',
        fill=True,
        fill_color='orange',
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>{row['name']}</b><br>"
            f"Nearest volcano: {row['nearest_volcano']}<br>"
            f"Distance: {row['distance_km']:.1f} km<br>"
            f"Zone: {row['zone']}",
            max_width=200
        ),
        tooltip=row['name'],
    ).add_to(candi_group)

candi_group.add_to(m)

# ============================================================
# 5. Inscription layer
# ============================================================

insc_group = folium.FeatureGroup(name="Inscriptions (182 geocoded)", show=False)
for _, row in insc_df.iterrows():
    if pd.notna(row['lat']) and pd.notna(row['lon']):
        color = 'blue' if row.get('century', 10) <= 9 else 'purple'
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>{row.get('title', row['filename'])}</b><br>"
                f"Date: {row.get('date_ce', 'unknown')} CE<br>"
                f"Language: {row.get('lang', 'unknown')}<br>"
                f"Volcano dist: {row.get('volcano_dist_km', 0):.1f} km",
                max_width=250
            ),
        ).add_to(insc_group)

insc_group.add_to(m)

# ============================================================
# 6. Calibration sites (buried temples)
# ============================================================

calib_group = folium.FeatureGroup(name="Calibration: Buried Temples", show=True)
for c in calibration:
    rate_str = f"{c['rate']:.1f} mm/yr" if c['rate'] else "Catastrophic burial"
    folium.Marker(
        location=[c['lat'], c['lon']],
        icon=folium.Icon(color='darkred', icon='arrow-down', prefix='fa'),
        popup=folium.Popup(
            f"<b>{c['name']}</b><br>"
            f"Built: ~{c['year_built']} CE<br>"
            f"Burial depth: {c['depth_m']:.2f} m<br>"
            f"Sedimentation rate: {rate_str}",
            max_width=200
        ),
        tooltip=f"{c['name']} ({c['depth_m']}m buried)",
    ).add_to(calib_group)

calib_group.add_to(m)

# ============================================================
# 7. Fieldwork targets
# ============================================================

target_group = folium.FeatureGroup(name="Fieldwork Targets (E080)", show=True)
for t in targets:
    color = 'green' if t['priority'] == 'HIGH' else 'lightgreen'
    folium.Marker(
        location=[t['lat'], t['lon']],
        icon=folium.Icon(color=color, icon='bullseye', prefix='fa'),
        popup=folium.Popup(
            f"<b>{t['name']}</b><br>"
            f"Priority: {t['priority']}<br>"
            f"Method: {t['method']}",
            max_width=200
        ),
        tooltip=t['name'],
    ).add_to(target_group)

target_group.add_to(m)

# ============================================================
# 8. Archaeological sites (if available)
# ============================================================

if sites_gdf is not None:
    sites_group = folium.FeatureGroup(name="Archaeological Sites (666)", show=False)
    for _, row in sites_gdf.iterrows():
        try:
            lat = row.geometry.y
            lon = row.geometry.x
            name = row.get('name', row.get('nama', 'Unknown'))
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='gray',
                fill=True,
                fill_color='gray',
                fill_opacity=0.5,
                tooltip=str(name),
            ).add_to(sites_group)
        except:
            pass
    sites_group.add_to(m)

# ============================================================
# 9. Info box
# ============================================================

info_html = """
<div style="position:fixed; top:10px; right:10px; z-index:1000;
     background:white; padding:15px; border-radius:8px;
     border:2px solid #333; max-width:300px; font-family:Arial,sans-serif;">
<h3 style="margin:0 0 8px 0; color:#333;">VOLCARCH Prediction Map</h3>
<p style="font-size:12px; margin:0 0 5px 0; color:#555;">
<b>158 experiments</b> | 5 papers under review<br>
Burial rate: 2.4-6.2 mm/yr (4 calibration sites)<br>
3,220x demographic gap | 0.058% cascade visibility<br>
Pre-400 CE sites at 6.5m+ depth = undetectable
</p>
<p style="font-size:11px; margin:5px 0 0 0; color:#888;">
<b>Legend:</b><br>
<span style="color:red;">&#9679;</span> Active volcanoes (+ 15km Zone A)<br>
<span style="color:orange;">&#9679;</span> Candi (142 temples)<br>
<span style="color:blue;">&#9679;</span> Inscriptions (pre-C10) <span style="color:purple;">&#9679;</span> (post-C10)<br>
<span style="color:darkred;">&#x25BC;</span> Buried temple calibration sites<br>
<span style="color:green;">&#x25C9;</span> Fieldwork targets (E080)<br>
</p>
<p style="font-size:10px; margin:5px 0 0 0; color:#aaa;">
Contact: amien@ubhinus.ac.id | VOLCARCH 2026
</p>
</div>
"""
m.get_root().html.add_child(folium.Element(info_html))

# ============================================================
# 10. Layer control + fullscreen
# ============================================================

folium.LayerControl(collapsed=False).add_to(m)
plugins.Fullscreen().add_to(m)
plugins.MiniMap(toggle_display=True).add_to(m)

# ============================================================
# 11. Save
# ============================================================

output_path = Path("D:/documents/volcarch-repo/maps/volcarch_prediction_map.html")
m.save(str(output_path))
print(f"\nMap saved to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.0f} KB")
print(f"\nOpen in browser to view: file:///{output_path}")
print("\nDONE.")
