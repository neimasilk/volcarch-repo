# Data Availability Statement — Paper 2

**Paper:** Tautology-Free Settlement Suitability Modeling in East Java Under Survey and Taphonomic Bias  
**Journal:** Remote Sensing (MDPI)  
**Date:** 2026-02-26  

---

## 1. ARCHAEOLOGICAL SITE DATA

| Dataset | Description | Source | Access | License |
|---------|-------------|--------|--------|---------|
| `east_java_sites.geojson` | 666 archaeological sites (candi, monuments, ruins) in East Java | Compiled from OpenStreetMap (OSM), Wikidata, Wikipedia Indonesia, and BPCB publications | `data/processed/east_java_sites.geojson` in this repository | ODbL (OSM data), CC BY-SA (Wikidata) |

**Attribution:**
- OpenStreetMap contributors (via Overpass API)
- Wikidata (SPARQL endpoint)
- Wikipedia Indonesia ("Daftar candi di Indonesia")
- Balai Pelestarian Cagar Budaya (BPCB) Jawa Timur publications

**Geocoding:**
- 297 sites: OSM centroid (`accuracy_level: osm_centroid`)
- 94 sites: Nominatim geocoding (`accuracy_level: nominatim`)
- 16 sites: Wikidata P625 (`accuracy_level: wikidata_p625`)
- 259 sites: No coordinates (excluded from modeling)

---

## 2. ENVIRONMENTAL COVARIATES

### 2.1 Digital Elevation Model (DEM)

| Dataset | Description | Source | Resolution | License |
|---------|-------------|--------|------------|---------|
| Copernicus GLO-30 | Global DEM | Copernicus Programme (ESA) | 30 m | Free and open (Copernicus Terms) |

**Access:** https://dataspace.copernicus.eu/  
**Download method:** AWS Open Data Registry  
**Citation:** European Space Agency (ESA), Copernicus Digital Elevation Model (DEM)

**Derived layers:**
- `jatim_dem.tif` — Elevation (m)
- `jatim_slope.tif` — Slope (degrees)
- `jatim_aspect.tif` — Aspect (degrees)
- `jatim_twi.tif` — Topographic Wetness Index (unitless)
- `jatim_tri.tif` — Terrain Ruggedness Index (unitless)
- `jatim_river_dist.tif` — Distance to nearest river (m)
- `jatim_road_dist_expanded.tif` — Distance to nearest road (m)

### 2.2 River Network

| Dataset | Description | Source | Access | License |
|---------|-------------|--------|--------|---------|
| OpenStreetMap waterways | Rivers, streams, canals | OpenStreetMap | Full Java extract | ODbL |

**Download method:** Overpass API query  
**Query:** `way["waterway"~"river|stream|canal"] in "Jawa Timur, Indonesia"`

### 2.3 Road Network

| Dataset | Description | Source | Access | License |
|---------|-------------|--------|--------|---------|
| OpenStreetMap roads | Major and minor roads | OpenStreetMap | Full Java extract | ODbL |

**Road classes used:**
- `motorway`, `trunk`, `primary`, `secondary`, `tertiary` (E011-E012)
- Plus `unclassified`, `residential`, `service` (E012-E013)

---

## 3. CODE AND SOFTWARE

### 3.1 Repository

**GitHub:** https://github.com/neimasilk/volcarch-repo  
**Commit (Paper 2 submission):** `453f36b`  
**License:** MIT (code), CC BY 4.0 (papers/documents)

### 3.2 Key Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| `01_settlement_model_v7.py` | E013 hybrid model implementation | `experiments/E013_settlement_model_v7/` |
| `01_temporal_split_test.py` | E014 temporal validation | `experiments/E014_temporal_validation/` |
| `robustness_checks.py` | Seed stability analysis | `papers/P2_settlement_model/` |
| `block_size_sensitivity.py` | CV block size testing | `papers/P2_settlement_model/` |
| `build_figures.py` | Figure generation | `papers/P2_settlement_model/` |

### 3.3 Software Versions

| Package | Version |
|---------|---------|
| Python | 3.11+ |
| geopandas | 0.14.0 |
| rasterio | 1.3.8 |
| xgboost | 2.0.0 |
| scikit-learn | 1.3.0 |
| numpy | 1.24.3 |
| pandas | 2.0.3 |
| matplotlib | 3.7.2 |
| folium | 0.14.0 |

**Full dependency lock:** `papers/P2_settlement_model/requirements_submission_lock.txt`

---

## 4. REPRODUCIBILITY

### 4.1 Reproducing E013 (Main Model)

```bash
# 1. Clone repository
git clone https://github.com/neimasilk/volcarch-repo.git
cd volcarch-repo

# 2. Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run E013 (requires E012 road proxy first)
python experiments/E012_settlement_model_v6/00_prepare_road_proxy_expanded.py
python experiments/E013_settlement_model_v7/01_settlement_model_v7.py

# 4. Results in: experiments/E013_settlement_model_v7/results/
```

### 4.2 Reproducing Temporal Validation (E014)

```bash
python experiments/E014_temporal_validation/01_temporal_split_test.py
```

### 4.3 Expected Runtime

- E013 (single run): ~5 minutes (CPU only)
- E013 (full sweep): ~30 minutes
- Robustness (20 seeds): ~45 minutes
- Block sensitivity: ~60 minutes
- Temporal validation: ~5 minutes

---

## 5. DATA ACCESSIBILITY SUMMARY

| Category | Status | Location |
|----------|--------|----------|
| Site coordinates | ✅ Available | This repository |
| DEM data | ✅ Available | Copernicus (public) |
| OSM extracts | ✅ Available | OSM (ODbL) |
| Source code | ✅ Available | GitHub (MIT) |
| Trained models | ✅ Available | `models/` directory |
| Suitability maps | ✅ Available | `maps/` directory |

---

## 6. LIMITATIONS AND NOTES

1. **Site data quality:** 41% of sites lack precise coordinates (275/666). Model uses only geocoded sites (n=378).

2. **Temporal data:** Discovery year available for only 28 sites. Temporal validation (E014) uses accessibility as proxy.

3. **Road network:** OSM completeness varies by region. Malang area has good coverage; remote areas may be underrepresented.

4. **DEM accuracy:** Copernicus GLO-30 vertical accuracy ~4m (1σ) in flat terrain, ~10m in mountainous terrain.

5. **Archaeological bias:** Known sites are predominantly stone monuments (candi). Wooden/settlement sites are underrepresented due to preservation bias (see Paper 1).

---

## 7. CONTACT

For data access questions:
- **Author:** Mukhlis Amien (amien@ubhinus.ac.id)
- **Institution:** Lab Data Sains, Universitas Bhinneka Nusantara
- **Repository:** https://github.com/neimasilk/volcarch-repo

---

*Generated: 2026-02-26*  
*Version: Paper 2 Submission*
