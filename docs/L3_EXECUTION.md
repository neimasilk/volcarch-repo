# L3: EXECUTION (Active Tasks)

**Status:** ACTIVE — Updates per week or per experiment.
**Sprint:** Sprint 0 — Project Setup & Initial Data Collection
**Last updated:** 2026-02-23

---

## Current Focus

**Immediate goal:** Collect and geocode archaeological site data for East Java to enable the first statistical test of H1 (taphonomic bias).

## Active Tasks

### TASK-001: Set up repo and environment
**Status:** IN PROGRESS
**Description:** Initialize Python environment, install core dependencies, verify GPU access.
**Acceptance criteria:**
- [ ] Python 3.10+ with geopandas, rasterio, scikit-learn, folium installed
- [ ] GPU accessible (verify with torch.cuda.is_available())
- [ ] Repo structure matches CLAUDE.md specification
- [ ] All L1–L3 docs in place

### TASK-002: Scrape and geocode East Java archaeological sites
**Status:** NOT STARTED
**Description:** Build a comprehensive dataset of known archaeological sites in East Java with coordinates.
**Sources to scrape/compile:**
- Wikipedia list of candi in East Java
- BPCB Jawa Timur website (kebudayaan.kemdikbud.go.id)
- OpenStreetMap (tag: historic=archaeological_site in East Java)
- Published archaeological survey papers
- Google Scholar search for site location data
**Output:** `data/processed/east_java_sites.geojson` with fields: name, type (candi/arca/prasasti/other), period, lat, lon, source, discovery_year, notes
**Experiment:** → E001

### TASK-003: Compile volcanic eruption history
**Status:** NOT STARTED
**Description:** Build structured dataset of eruptions affecting Malang basin.
**Volcanoes:** Kelud, Semeru, Arjuno, Welirang, Bromo
**Sources:** Global Volcanism Program (volcano.si.edu), published papers
**Output:** `data/processed/eruption_history.csv` with fields: volcano, year, VEI, ashfall_malang_cm (if documented), source
**Experiment:** → E002

### TASK-004: Download and process DEM
**Status:** NOT STARTED
**Description:** Acquire DEM for Malang Raya study area. SRTM 30m as starting point (easier access), DEMNAS 8.5m if available.
**Output:** `data/raw/dem_malang.tif` and derived layers (slope, aspect, TWI, TRI)
**Experiment:** → E003

## Upcoming Tasks (Backlog)

- TASK-005: Statistical test — site density vs distance from active volcanoes (Paper 1 core analysis)
- TASK-006: Generate isopach-style volcanic influence map for Malang basin
- TASK-007: Literature review — global precedents for volcanic taphonomic bias
- TASK-008: Draft Paper 1 outline and figures

## Blocked / Waiting

*Nothing currently blocked.*

## Recently Completed

*No completed tasks yet — project just initialized.*

---

## Experiment Queue

| ID | Name | Status | Paper | Notes |
|----|------|--------|-------|-------|
| E001 | Archaeological site geocoding | QUEUED | P1 | First priority |
| E002 | Eruption history compilation | QUEUED | P1, P3 | Can parallel with E001 |
| E003 | DEM acquisition and processing | QUEUED | P2 | Can parallel with E001 |
| E004 | Site density vs volcanic proximity | BLOCKED (needs E001, E002) | P1 | The first real test of H1 |

---

*Update this document whenever tasks change status. Keep it honest — if something is stuck, say so.*
