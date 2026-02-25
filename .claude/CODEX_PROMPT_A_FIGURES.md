# CODEX PROMPT A: Generate Missing Critical Figures for Paper 2

**Context:** VOLCARCH project, Paper 2 (settlement suitability model for East Java). The LaTeX manuscript at `papers/P2_settlement_model/submission_remote_sensing_v0.2.tex` is missing three critical figures that any Remote Sensing reviewer will expect. Generate them as PNG files.

**Rules:**
- Do NOT push to GitHub.
- Do NOT modify any existing experiment results or data.
- Output figures to `papers/P2_settlement_model/figures/`.
- Use Python with matplotlib, geopandas, rasterio. If cartopy is not installed, use a simpler approach with geopandas + contextily or plain matplotlib.
- All figures must be publication-quality: 300 DPI, clear labels, readable at A4 page width.

---

## Figure A: Study Area Map (`fig10_study_area_map.png`)

**What it shows:** Map of East Java province with:
1. Province boundary outline (can be derived from Natural Earth data or a simple bounding box: 111-115°E, 9-6.5°S)
2. Seven volcano locations as red triangles with labels:
   - Kelud (-7.930, 112.308)
   - Semeru (-8.108, 112.922)
   - Arjuno-Welirang (-7.729, 112.575)
   - Bromo (-7.942, 112.950)
   - Lamongan (-7.977, 113.343)
   - Raung (-8.125, 114.042)
   - Ijen (-8.058, 114.242)
3. Archaeological site locations as small blue dots — load from `data/processed/east_java_sites.geojson`
4. Inset showing Java island within Indonesia (optional, skip if complex)
5. Scale bar and north arrow
6. Coordinate grid (lat/lon ticks)

**Styling:** White background, grayscale terrain if DEM is available (`data/processed/dem/jatim_dem.tif`), otherwise plain. Title: "Study area: East Java Province, Indonesia"

---

## Figure B: Suitability Probability Map (`fig11_suitability_map_static.png`)

**What it shows:** The E013 model output as a static heatmap.

**How to generate:**
1. Run the E013 model to generate grid predictions, OR if you want to avoid re-running the full model, reconstruct from the existing experiment data. The simplest approach:
   - Load rasters from `data/processed/dem/` (jatim_dem.tif, jatim_slope.tif, jatim_twi.tif, jatim_tri.tif, jatim_aspect.tif, jatim_river_dist.tif)
   - Load sites from `data/processed/east_java_sites.geojson`
   - Load road distance from `data/processed/dem/jatim_road_dist_expanded.tif`
   - Replicate E013 best config: region_blend=0.00, hard_frac_target=0.30, seed=375 (cfg_seed = 42 + 3*111 = 375), base TGB params decay=12000, max_road=20000, min_prob=0.03
   - Train XGBoost with same params as in `experiments/E013_settlement_model_v7/01_settlement_model_v7.py`
   - Predict on grid (step=10 pixels) and plot as filled contour/imshow
2. Use YlOrRd colormap, add colorbar labeled "Settlement Suitability Probability"
3. Overlay volcano triangles (red) and known sites (small white/black dots)
4. Title: "E013 Settlement suitability probability map (XGBoost, AUC = 0.751)"
   - NOTE: use seed-averaged AUC 0.751, NOT single-seed 0.768

**Important:** This figure is THE most important output of the paper. Take care with it.

---

## Figure C: Feature Importance (`fig12_feature_importance.png`)

**What it shows:** Horizontal bar chart of XGBoost feature importances from E013.

**Data (from E013 results):**
- elevation: 0.215
- TRI: 0.185
- TWI: 0.166
- river_dist: 0.160
- slope: 0.155
- aspect: 0.118

**Styling:**
- Horizontal bars, sorted descending (elevation at top)
- Use readable labels: "Elevation", "Terrain Ruggedness (TRI)", "Topographic Wetness (TWI)", "River Distance", "Slope", "Aspect"
- Color: single color (e.g., steelblue) or gradient
- Title: "XGBoost feature importance (E013 best configuration)"
- X-axis: "Relative Importance"

---

## Deliverables

After generating, list the files created and their sizes. Update `papers/P2_settlement_model/submission_checklist.md` to mark these figures as generated.

Script should be saved as `papers/P2_settlement_model/build_submission_figures.py` for reproducibility.
