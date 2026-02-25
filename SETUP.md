# VOLCARCH — Setup Guide

## Requirements

- Python 3.10 or newer
- ~2 GB free disk space (DEM files)
- Internet connection (for data scraping)

---

## 1. Install Python

Download from: https://www.python.org/downloads/
Choose **Python 3.11** (recommended). Check "Add Python to PATH" during install.

Verify:
```
python --version
```

---

## 2. Create a virtual environment

From the repo root:
```
python -m venv .venv
```

Activate it:
```
# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate
```

---

## 3. Install dependencies

```
pip install -r requirements.txt
```

This installs: geopandas, rasterio, scikit-learn, xgboost, folium, requests, beautifulsoup4, scipy, matplotlib, tqdm, and others.

> **Note for Windows:** geopandas on Windows sometimes needs binary wheels.
> If `pip install geopandas` fails, try:
> ```
> pip install pipwin
> pipwin install gdal
> pipwin install fiona
> pipwin install geopandas
> ```
> Or install via conda: `conda install -c conda-forge geopandas rasterio`

---

## 4. Verify GPU (optional, for future ML work)

```python
import torch
print(torch.cuda.is_available())   # should print True on RTX 4080
print(torch.cuda.get_device_name(0))
```

GPU is not required for Sprint 0 data collection tasks.

---

## 5. Run experiments in order

### Step 1 — Collect archaeological sites (E001)

```
python experiments/E001_site_density_vs_volcanic_proximity/01_collect_sites.py
```

This scrapes OpenStreetMap for East Java archaeological sites and writes:
`data/processed/east_java_sites.geojson`

To also include Wikipedia data, run the supplement scraper first:
```
python tools/scrape_wikipedia_sites.py
```
Then re-run `01_collect_sites.py`.

### Step 2 — Compile eruption history (E002)

```
python experiments/E002_eruption_history/01_compile_eruptions.py
```

This will attempt to auto-download from GVP Smithsonian. If GVP returns HTML
instead of CSV, it falls back to a manually-compiled seed dataset.

**For full GVP data (recommended):**
1. Go to: https://volcano.si.edu/database/search_eruption_excel.cfm
2. Search each volcano by number: Kelud (263280), Semeru (263300), Arjuno-Welirang (263260), Bromo (263310)
3. Export as Excel → save to `data/raw/gvp/gvp_<id>.xlsx`
4. Re-run the script — it will auto-detect the xlsx files

### Step 3 — Download DEM (E003)

```
python experiments/E003_dem_acquisition/01_download_dem.py
```

Downloads SRTM 30m DEM from OpenTopography (free, no auth needed) and computes
slope, aspect, TWI, TRI layers for Malang Raya.

Output: `data/processed/dem/malang_*.tif`

### Step 4 — Density analysis (E004) — Paper 1 core

```
python experiments/E004_density_analysis/01_analyze_density.py
```

**Requires E001 output.** Runs the first statistical test of H1.

Output:
- `experiments/E004_density_analysis/results/density_chart.png`
- `experiments/E004_density_analysis/results/map_sites_by_distance.html`
- `experiments/E004_density_analysis/results/correlation_stats.txt`

---

## 6. Data provenance

See `data/sources.md` for full data source documentation.

All data in `data/raw/` must not be modified. Process into `data/processed/`.

---

## 7. Repo navigation

See `CLAUDE.md` for full repo structure and rules.

Quick orientation:
- `docs/L1_CONSTITUTION.md` — why this project exists
- `docs/L2_STRATEGY.md` — current phase and active papers
- `docs/L3_EXECUTION.md` — what to work on now
- `docs/JOURNAL.md` — research log (append-only)
