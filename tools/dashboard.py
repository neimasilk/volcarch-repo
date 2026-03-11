"""
VOLCARCH Interactive Dashboard.

Streamlit app for exploring settlement suitability model results
from Paper 2 (Volcanic Taphonomic Bias in East Java).

Requires pre-computed data from precompute_dashboard_data.py.

Run:
    streamlit run tools/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json

st.set_page_config(
    page_title="VOLCARCH: Settlement Suitability Explorer",
    layout="wide",
)

REPO_ROOT = Path(__file__).parent.parent
DASHBOARD_DIR = REPO_ROOT / "data" / "processed" / "dashboard"


# ── Data loading (cached) ───────────────────────────────────────────────

@st.cache_data
def load_grid():
    return pd.read_csv(DASHBOARD_DIR / "grid_predictions.csv")


@st.cache_data
def load_sites():
    return pd.read_csv(DASHBOARD_DIR / "sites.csv")


@st.cache_data
def load_volcanoes():
    return pd.read_csv(DASHBOARD_DIR / "volcanoes.csv")


@st.cache_data
def load_metadata():
    return json.loads((DASHBOARD_DIR / "metadata.json").read_text())


# ── Tab renderers ────────────────────────────────────────────────────────

def render_map_tab(grid, sites, volcanoes, layer, show_sites, show_volcanoes):
    try:
        import folium
        from folium.plugins import HeatMap
        from streamlit_folium import st_folium
    except ImportError:
        st.error(
            "Missing dependencies for interactive map.\n\n"
            "Run: `pip install folium streamlit-folium`"
        )
        return

    center_lat = float(grid["lat"].mean())
    center_lon = float(grid["lon"].mean())
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="CartoDB positron",
    )

    # Main layer based on sidebar selection
    if "Suitability" in layer:
        heat_data = grid[["lat", "lon", "suitability"]].values.tolist()
        HeatMap(
            heat_data, min_opacity=0.3, radius=8, blur=10, max_zoom=13,
            gradient={
                0.0: "#3288bd", 0.25: "#66c2a5",
                0.5: "#fee08b", 0.75: "#f46d43", 1.0: "#d53e4f",
            },
        ).add_to(m)

    elif "Zones" in layer:
        zone_colors = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}
        for zone, color in zone_colors.items():
            zdata = grid[grid["zone"] == zone]
            if len(zdata) == 0:
                continue
            # Subsample large zones for map performance
            if len(zdata) > 5000:
                zdata = zdata.iloc[::3]
            fg = folium.FeatureGroup(name=f"Zone {zone}")
            for lat, lon, suit, burial in zip(
                zdata["lat"].values, zdata["lon"].values,
                zdata["suitability"].values, zdata["burial_depth_cm"].values,
            ):
                folium.CircleMarker(
                    [float(lat), float(lon)], radius=3,
                    color=color, fill=True, fill_color=color,
                    fill_opacity=0.5, weight=0,
                    tooltip=(
                        f"Zone {zone} | "
                        f"Suitability: {suit:.2f} | "
                        f"Burial: {burial:.0f} cm"
                    ),
                ).add_to(fg)
            fg.add_to(m)

    elif "Burial" in layer:
        max_b = float(np.clip(grid["burial_depth_cm"].quantile(0.99), 100, 500))
        vals = grid[["lat", "lon", "burial_depth_cm"]].values.copy()
        vals[:, 2] = np.clip(vals[:, 2], 0, max_b)
        HeatMap(
            vals.tolist(), min_opacity=0.2, radius=8, blur=10,
            gradient={
                0.0: "#ffffb2", 0.25: "#fecc5c",
                0.5: "#fd8d3c", 0.75: "#f03b20", 1.0: "#bd0026",
            },
        ).add_to(m)

    # Sites overlay
    if show_sites:
        fg_sites = folium.FeatureGroup(name="Sites")
        for _, row in sites.iterrows():
            name = row.get("name", "")
            if pd.isna(name) or name == "":
                name = "Unknown"
            popup_html = (
                f"<b>{name}</b><br>"
                f"Suitability: {row['suitability']:.2f}<br>"
                f"Burial: {row['burial_depth_cm']:.0f} cm<br>"
                f"Zone: {row['zone']}"
            )
            folium.CircleMarker(
                [float(row["lat"]), float(row["lon"])],
                radius=5, color="blue", fill=True,
                fill_color="blue", fill_opacity=0.8, weight=1,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=str(name),
            ).add_to(fg_sites)
        fg_sites.add_to(m)

    # Volcanoes overlay
    if show_volcanoes:
        fg_volc = folium.FeatureGroup(name="Volcanoes")
        for _, row in volcanoes.iterrows():
            folium.Marker(
                [float(row["lat"]), float(row["lon"])],
                icon=folium.Icon(color="red", icon="info-sign"),
                tooltip=row["name"],
            ).add_to(fg_volc)
        fg_volc.add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, height=600)


def render_shap_tab(meta):
    import matplotlib.pyplot as plt

    st.subheader("SHAP Feature Importance / Pentingnya Fitur SHAP")
    st.markdown(
        "TreeSHAP provides instance-level feature attribution, showing how each "
        "feature contributes to the model's prediction for every sample."
    )

    col1, col2 = st.columns(2)

    with col1:
        beeswarm_path = DASHBOARD_DIR / "shap_beeswarm.png"
        if beeswarm_path.exists():
            st.image(str(beeswarm_path), caption="SHAP Beeswarm Plot")
        else:
            st.info("Beeswarm plot not available. Run precompute with shap installed.")

    with col2:
        bar_path = DASHBOARD_DIR / "shap_bar.png"
        if bar_path.exists():
            st.image(str(bar_path), caption="Mean |SHAP| per Feature")
        else:
            # Fallback: bar chart from gain-based feature importances
            fi = meta.get("feature_importances", {})
            if fi:
                sorted_fi = sorted(fi.items(), key=lambda x: x[1])
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(
                    [FEAT_LABELS.get(k, k) for k, _ in sorted_fi],
                    [v for _, v in sorted_fi],
                    color="#43A047",
                )
                ax.set_xlabel("Importance (gain)")
                ax.set_title("XGBoost Feature Importance (Gain-based)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # SHAP summary table
    shap_csv = DASHBOARD_DIR / "shap_summary.csv"
    if shap_csv.exists():
        st.subheader("SHAP Summary Table / Tabel Ringkasan SHAP")
        shap_df = pd.read_csv(shap_csv)
        st.dataframe(shap_df, use_container_width=True, hide_index=True)
    else:
        # Fallback: show gain-based importances
        fi = meta.get("feature_importances", {})
        if fi:
            st.subheader("Feature Importances (Gain) / Pentingnya Fitur")
            fi_df = pd.DataFrame([
                {"Feature": FEAT_LABELS.get(k, k), "Importance": v}
                for k, v in sorted(fi.items(), key=lambda x: -x[1])
            ])
            st.dataframe(fi_df, use_container_width=True, hide_index=True)


def render_zone_tab(grid, meta):
    st.subheader("Zone Classification / Klasifikasi Zona")

    st.markdown("""
| Zone | Description / Deskripsi | Survey Implication / Implikasi Survei |
|------|------------------------|--------------------------------------|
| **A** | High suitability + shallow burial (<100 cm) | Known sites expected / Situs yang sudah diketahui |
| **B** | High suitability + moderate burial (100-300 cm) | GPR survey targets / Target survei GPR |
| **C** | High suitability + deep burial (>300 cm) | Present but hard to reach / Ada tapi sulit dijangkau |
| **E** | Low suitability | Few/no sites expected / Sedikit/tidak ada situs |
""")

    # Zone statistics table
    st.subheader("Zone Statistics / Statistik Zona")
    zone_stats_path = DASHBOARD_DIR / "zone_statistics.csv"
    if zone_stats_path.exists():
        zone_stats = pd.read_csv(zone_stats_path, index_col=0)
        st.dataframe(zone_stats, use_container_width=True)
    else:
        zone_counts = meta.get("zone_counts", {})
        total = max(sum(zone_counts.values()), 1)
        stats_rows = []
        for z in ["A", "B", "C", "E"]:
            n = zone_counts.get(z, 0)
            zdata = grid[grid["zone"] == z]
            stats_rows.append({
                "Zone": z,
                "Count": n,
                "Percentage": f"{100 * n / total:.1f}%",
                "Mean Suitability": f"{zdata['suitability'].mean():.2f}" if len(zdata) > 0 else "-",
                "Mean Burial (cm)": f"{zdata['burial_depth_cm'].mean():.0f}" if len(zdata) > 0 else "-",
            })
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # Dwarapala validation
    st.subheader("Dwarapala Validation / Validasi Dwarapala")
    dw_pred = meta.get("dwarapala_predicted_cm", 0)
    dw_actual = meta.get("dwarapala_actual_cm", 185)
    dw_raw = meta.get("dwarapala_raw_cm", 0)
    lf = meta.get("loss_factor", 1.0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted / Prediksi", f"{dw_pred:.0f} cm")
    col2.metric("Actual / Aktual", f"{dw_actual} cm")
    col3.metric("Loss Factor", f"{lf:.3f}")

    st.markdown(f"""
**Dwarapala Statue** (built 1268 CE, discovered at 185 cm depth)

- Raw Pyle model prediction: **{dw_raw:.0f} cm**
- Calibrated prediction: **{dw_pred:.0f} cm** (loss factor = {lf:.3f})
- Actual observed depth: **{dw_actual} cm**
- Interpretation: Only **{lf * 100:.1f}%** of deposited tephra is retained in situ.
  The remainder is lost to erosion, compaction, lahar reworking, and fluvial transport.
""")


def render_validation_tab(meta):
    import matplotlib.pyplot as plt

    st.subheader("Model Validation / Validasi Model")

    # AUC Progression
    st.subheader("AUC Progression / Progres AUC (E007 - E013)")
    auc_prog = meta.get("auc_progression", {})
    if auc_prog:
        experiments = list(auc_prog.keys())
        aucs = list(auc_prog.values())
        colors = ["#b0b0b0"] * (len(aucs) - 1) + ["#2ecc71"]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(experiments, aucs, color=colors, edgecolor="#333", linewidth=0.5)
        ax.axhline(0.75, color="green", linestyle="--", alpha=0.7, label="MVR threshold (0.75)")
        ax.axhline(0.65, color="orange", linestyle="--", alpha=0.7, label="Kill signal (0.65)")
        ax.set_ylabel("Spatial CV AUC")
        ax.set_title("AUC Progression Across Experiments")
        ax.set_ylim(0.55, 0.85)
        ax.legend(fontsize=8, loc="upper left")
        for i, v in enumerate(aucs):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tautology + Temporal side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tautology Test / Uji Tautologi")
        rho = meta.get("tautology_rho", 0)
        verdict = meta.get("tautology_verdict", "N/A")
        st.metric("Spearman rho(suitability, volcano_dist)", f"{rho:.3f}")
        if "FREE" in verdict:
            st.success(f"Verdict: {verdict}")
        elif "MILD" in verdict:
            st.warning(f"Verdict: {verdict}")
        else:
            st.error(f"Verdict: {verdict}")
        st.markdown("""
**Interpretation:** A negative rho means high-suitability cells are NOT
systematically closer to volcanoes. The model predicts settlement based on
terrain features, not volcano proximity.
""")

    with col2:
        st.subheader("Temporal Validation (E014) / Validasi Temporal")
        tv = meta.get("temporal_validation", {})
        tv_auc = tv.get("xgb_auc", 0)
        tv_verdict = tv.get("verdict", "N/A")
        tv_spatial = tv.get("spatial_cv_xgb_auc", 0)

        st.metric("Temporal AUC (XGBoost)", f"{tv_auc:.3f}")
        st.metric("Spatial CV AUC (baseline)", f"{tv_spatial:.3f}")
        delta = tv_auc - tv_spatial
        st.metric("Temporal vs Spatial", f"{delta:+.3f}")
        if tv_verdict == "PASS":
            st.success("Verdict: PASS - Model is tautology-resistant")
        else:
            st.warning(f"Verdict: {tv_verdict}")
        st.markdown(f"""
**Test:** Train on easy-access sites, test on hard-access sites.
Temporal AUC > 0.65 means the model predicts genuinely undiscovered sites.
Split method: `{tv.get("split_method", "unknown")}`.
""")

    # Model summary table
    st.subheader("Model Summary / Ringkasan Model")
    summary_data = {
        "Parameter": [
            "Algorithm", "Features", "Pseudo-absence ratio",
            "Background sampling", "Hard-negative fraction",
            "Spatial CV AUC", "TSS", "Grid points", "Grid spacing",
        ],
        "Value": [
            "XGBoost (n_est=300, depth=4, lr=0.05)",
            "elevation, slope, TWI, TRI, aspect, river_dist",
            "5:1",
            "TGB (road-distance weighted)",
            "30% (z-dist >= 2.0)",
            f"{meta.get('xgb_auc', 0):.3f} +/- {meta.get('xgb_auc_std', 0):.3f}",
            f"{meta.get('xgb_tss', 0):.3f} +/- {meta.get('xgb_tss_std', 0):.3f}",
            f"{meta.get('n_grid_points', 0):,}",
            "~900m (30-pixel step)",
        ],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


# ── Feature label mapping (for SHAP fallback) ───────────────────────────

FEAT_LABELS = {
    "elevation": "Elevation (m)",
    "slope": "Slope (degrees)",
    "twi": "TWI",
    "tri": "TRI",
    "aspect": "Aspect (degrees)",
    "river_dist": "River distance (m)",
}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    # Check data exists
    if not DASHBOARD_DIR.exists() or not (DASHBOARD_DIR / "metadata.json").exists():
        st.error(
            "Dashboard data not found.\n\n"
            "Run the precompute script first:\n"
            "```\npython tools/precompute_dashboard_data.py\n```"
        )
        return

    grid = load_grid()
    sites = load_sites()
    volcanoes = load_volcanoes()
    meta = load_metadata()

    # Title
    st.title("VOLCARCH: Settlement Suitability Explorer")
    st.caption(
        "Volcanic Taphonomic Bias in East Java "
        "/ Bias Tafonomik Vulkanik di Jawa Timur"
    )

    # Sidebar
    with st.sidebar:
        st.header("Map Controls / Kontrol Peta")
        map_layer = st.radio(
            "Map Layer / Lapisan Peta",
            [
                "Suitability / Kesesuaian",
                "Zones / Zona",
                "Burial Depth / Kedalaman",
            ],
        )
        st.divider()
        show_sites = st.checkbox("Show Sites / Tampilkan Situs", True)
        show_volcanoes = st.checkbox("Show Volcanoes / Tampilkan Gunung Api", True)

        st.divider()
        st.subheader("Model Statistics")
        c1, c2 = st.columns(2)
        c1.metric("AUC", f"{meta.get('xgb_auc', 0):.3f}")
        c2.metric("TSS", f"{meta.get('xgb_tss', 0):.3f}")
        st.caption(f"Tautology: {meta.get('tautology_verdict', 'N/A')}")
        st.caption(f"Sites: {meta.get('n_sites', 0)}")

        st.divider()
        st.subheader("About / Tentang")
        st.markdown(
            "**Paper 2:** Settlement Suitability Model\n\n"
            "**Model:** XGBoost (E013 best config)\n\n"
            f"**Grid:** {meta.get('n_grid_points', 0):,} points (~900m)\n\n"
            "**Burial model:** Pyle (1989) + Dwarapala calibration"
        )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Peta Interaktif / Interactive Map",
        "Analisis SHAP / SHAP Analysis",
        "Klasifikasi Zona / Zone Classification",
        "Validasi Model / Model Validation",
    ])

    with tab1:
        render_map_tab(grid, sites, volcanoes, map_layer, show_sites, show_volcanoes)

    with tab2:
        render_shap_tab(meta)

    with tab3:
        render_zone_tab(grid, meta)

    with tab4:
        render_validation_tab(meta)


if __name__ == "__main__":
    main()
