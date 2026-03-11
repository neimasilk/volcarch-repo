# Dashboard Handoff Document

## Status: WORKING — All 4 Tabs Verified (2026-03-03)

Dashboard fully functional. Root cause of browser error was `streamlit 1.30.0` lacking `on_change` callback support for custom components (required by `streamlit-folium 0.26.2`). Fixed by upgrading to `streamlit 1.54.0` and changing HeatMap gradient keys from strings to floats.

All 4 tabs verified via Playwright headless browser testing. Screenshots in `tools/screenshots/`.

---

## File Inventory

### Application Files

| File | Lines | Purpose |
|------|-------|---------|
| `tools/dashboard.py` | ~454 | Streamlit app — 4 tabs + sidebar, bilingual (ID/EN) |
| `tools/precompute_dashboard_data.py` | ~608 | Retrains E013 model, generates grid predictions, SHAP, zones |
| `tools/test_dashboard.py` | ~166 | Playwright headless test — launches server, screenshots 4 tabs |

### Pre-computed Data (`data/processed/dashboard/`)

| File | Size | Content |
|------|------|---------|
| `grid_predictions.csv` | 4.6 MB | 65,432 rows: lon, lat, suitability, burial_depth_cm, zone |
| `model_xgb.json` | 548 KB | Serialized XGBoost model |
| `shap_values.npz` | 174 KB | SHAP matrix + feature names + base values |
| `shap_beeswarm.png` | 184 KB | SHAP beeswarm plot |
| `shap_bar.png` | 71 KB | Mean |SHAP| bar chart |
| `shap_summary.csv` | 355 B | Feature-level SHAP statistics |
| `sites.csv` | 27 KB | 378 sites with name, lat, lon, suitability, burial_depth_cm, zone |
| `volcanoes.csv` | 179 B | 7 volcanoes with name, lat, lon |
| `zone_statistics.csv` | 163 B | Zone-level aggregated statistics |
| `metadata.json` | 1 KB | Model stats, AUC progression, validation results |

### Screenshots (`tools/screenshots/`)

| File | Shows |
|------|-------|
| `tab1_peta_interaktif.png` | Folium heatmap + site/volcano markers |
| `tab2_shap.png` | Beeswarm + bar chart + summary table |
| `tab3_zona.png` | Zone legend, statistics table, Dwarapala validation |
| `tab4_validasi.png` | AUC progression chart, tautology test, temporal validation |

---

## Dashboard Architecture

### Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  SIDEBAR                  │  MAIN CONTENT                        │
│                           │                                      │
│  Map Controls:            │  Title: VOLCARCH Settlement Explorer │
│  ○ Suitability            │  Caption: bilingual                  │
│  ○ Zones                  │                                      │
│  ○ Burial Depth           │  ┌─Tab 1─┬─Tab 2─┬─Tab 3─┬─Tab 4─┐ │
│                           │  │ Peta   │ SHAP  │ Zona  │ Valid │ │
│  ☑ Show Sites             │  │        │       │       │       │ │
│  ☑ Show Volcanoes         │  │ Folium │ PNG   │ Table │ Chart │ │
│                           │  │ Map    │ imgs  │ Stats │ Metr. │ │
│  Model Statistics:        │  │ 600px  │ + tbl │ Dwar. │ Tauto │ │
│  AUC: 0.768  TSS: 0.507  │  └────────┴───────┴───────┴───────┘ │
│  Tautology: FREE          │                                      │
│  Sites: 378               │                                      │
│                           │                                      │
│  About / Tentang          │                                      │
└──────────────────────────────────────────────────────────────────┘
```

### Tab Details

**Tab 1 — Peta Interaktif:** Folium map with 3 switchable layers (sidebar radio):
- **Suitability HeatMap:** 65k points, gradient blue→red
- **Zone Overlay:** CircleMarkers for zones A (green), B (orange), C (red). Zone E skipped. Subsampled >5k points by 3x.
- **Burial Depth HeatMap:** 65k points, gradient yellow→red, clipped at P99
- Optional overlays: 378 site markers (blue circles, popup with name/suitability/burial/zone), 7 volcano markers (red icons)

**Tab 2 — Analisis SHAP:** Two pre-rendered PNGs side-by-side (beeswarm + bar chart) + summary CSV table. Fallback to gain-based importance if SHAP PNGs missing.

**Tab 3 — Klasifikasi Zona:** Zone legend table (A/B/C/E with descriptions), zone statistics from CSV, Dwarapala validation metrics (predicted vs actual depth, loss factor).

**Tab 4 — Validasi Model:** Matplotlib AUC progression bar chart (E007–E013, green highlight on E013), tautology test + temporal validation side-by-side with st.metric + st.success/warning, model summary table.

### Data Flow

```
Raw rasters (EPSG:32749)     E013 config (seeds, hyperparams)
         │                              │
         ▼                              ▼
  precompute_dashboard_data.py ──────────────────►  data/processed/dashboard/
    - Load 6 rasters + road_dist                       - grid_predictions.csv
    - Load sites GeoJSON                               - model_xgb.json
    - Build TGB pseudo-absence pool                    - shap_*.{npz,png,csv}
    - Train XGBoost (300 trees, depth 4)               - sites.csv, volcanoes.csv
    - Grid predictions at ~900m spacing                - zone_statistics.csv
    - Pyle burial depth + Dwarapala calibration        - metadata.json
    - TreeSHAP + plots                                         │
    - Zone assignment (A/B/C/E)                                ▼
                                                      dashboard.py (Streamlit)
                                                        - Reads CSV/JSON/PNG only
                                                        - No model training
                                                        - ~5MB total data
```

### Key Constants (from E013/E014/E016)

| Metric | Value | Source |
|--------|-------|--------|
| AUC (Spatial CV) | 0.768 ± 0.069 | E013 |
| TSS | 0.507 ± 0.167 | E013 |
| Temporal AUC | 0.755 | E014 |
| Tautology rho | -0.229 | E013 |
| Tautology verdict | TAUTOLOGY-FREE | E013 |
| Loss factor | 0.284 (28.4% retention) | E016/Dwarapala |
| Zone A | 15,217 cells (23.2%) | E016 |
| Zone B (GPR targets) | 1,093 cells (1.7%) | E016 |
| Zone C | 48 cells (0.1%) | E016 |
| Zone E | 49,074 cells (75.0%) | E016 |

---

## What Was Verified

- [x] Precompute script runs successfully (~1 min), all 10 output files generated
- [x] Zone counts match E016: A=15,217 / B=1,093 / C=48 / E=49,074
- [x] Dwarapala calibrates to exactly 185 cm (loss factor 0.284)
- [x] SHAP ranking matches E015: Elevation > TRI > River dist > Slope > Aspect > TWI
- [x] Both SHAP PNGs visually verified (correct beeswarm and bar chart)
- [x] All Python imports succeed, no missing dependencies
- [x] Streamlit server starts cleanly
- [x] **Tab 1:** Folium heatmap renders, site/volcano markers visible with popups
- [x] **Tab 2:** Beeswarm + bar chart PNGs display, SHAP summary table loads
- [x] **Tab 3:** Zone legend, statistics table (correct counts), Dwarapala metrics
- [x] **Tab 4:** AUC progression chart, tautology/temporal validation panels
- [x] No Streamlit exceptions, no JS console errors (only benign iframe warnings)

---

## Bug Fix History

### 2026-03-03: MarshallComponentException (FIXED)

**Symptom:** `MarshallComponentException: Object of type function is not JSON serializable` on every page load.

**Root cause:** `streamlit-folium 0.26.2` passes `on_change=_on_change` (a Python callback) to `_component_func()`. In `streamlit 1.30.0`, `CustomComponent.__call__` did not have `on_change` as a named parameter, so the function ended up in `**kwargs` → `json_args` → `json.dumps()` → failure.

**Fix:**
1. `pip install --upgrade streamlit` → 1.30.0 → 1.54.0 (adds `on_change` support)
2. HeatMap gradient keys: `"0.0"` → `0.0` (preventive, for Folium compatibility)

**Debugging note:** A stale Streamlit process on port 8502 (running old 1.30.0) caused confusion during diagnosis. Always kill leftover servers before testing: `netstat -ano | grep :8502` then `taskkill /F /PID <pid>`.

---

## Dependencies

```
# Core (pinned versions that are known to work together):
streamlit==1.54.0
streamlit-folium==0.26.2
folium==0.20.0

# ML/Data:
xgboost
geopandas
rasterio
shap
matplotlib
numpy
pandas

# Testing:
playwright==1.57.0   # with chromium browser binaries installed
```

**Critical:** `streamlit >= 1.32.0` required for `on_change` callback support in custom components. Do NOT downgrade below 1.32.0 without also downgrading `streamlit-folium`.

---

## Commands

```bash
# Launch dashboard:
python -m streamlit run tools/dashboard.py

# Launch on specific port (headless, no browser auto-open):
python -m streamlit run tools/dashboard.py --server.port 8501 --server.headless true

# Re-generate pre-computed data (only if model/data changes):
python tools/precompute_dashboard_data.py

# Run Playwright verification test:
python tools/test_dashboard.py

# Kill stale Streamlit processes (Windows):
netstat -ano | grep :8502
taskkill /F /PID <pid>
```

**Note:** Use `python -m streamlit run` (NOT `streamlit run`) — the latter triggers a CLI module error on this machine.

---

## Possible Future Improvements

These are NOT required — the dashboard is functional as-is. Listed for reference if enhancement is desired.

1. **Zone layer performance:** Currently renders ~6k CircleMarkers via Python loop. Could switch to GeoJSON overlay or vector tiles for smoother interaction.
2. **Layer switching without full re-render:** Map rebuilds on every sidebar interaction. Could use `st.session_state` to cache the Folium map object per layer.
3. **Export functionality:** Add download buttons for grid_predictions.csv, zone map PNG, or site list filtered by zone.
4. **Mobile responsiveness:** Current layout is optimized for desktop (1400px+ width).
5. **Additional maps:** Burial depth contours, volcano eruption radius rings, site density kernel.

---

## Prompt for New Session

See next section.
