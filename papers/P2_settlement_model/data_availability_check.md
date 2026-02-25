# Data Availability Quick Check
## For Paper 2 Feature Expansion (Lithology & Climate)

**Date:** 2026-02-25  
**Analyst:** VOLCARCH Research Team  
**Status:** PRELIMINARY — Requires verification before download

---

## 1. LITHOLOGY / GEOLOGICAL FORMATION DATA

### 1.1 OneGeology Portal (Global)

| Attribute | Status |
|-----------|--------|
| **URL** | http://portal.onegeology.org/ |
| **Coverage** | Global |
| **Indonesia data** | ⚠️ PARTIAL |
| **Resolution** | 1:1M to 1:5M |
| **Java coverage** | [VERIFY] — Requires manual portal search |

**Findings:**
- OneGeology aggregates data from national geological surveys
- Indonesia data provided by Pusat Survei Geologi (PSG) — Bandung
- **Issue:** Last update to Indonesia layer unclear; may be outdated
- **Action required:** Manual portal search for "Indonesia" or "Java"

**Access method:**
1. Visit http://portal.onegeology.org/
2. Use "Search by Map" or "Search by Name"
3. Query: "Indonesia" or "Java"
4. Check available layers and their scales

**Verdict:** ⚠️ **CONDITIONAL** — Data likely exists but resolution and freshness unknown.

---

### 1.2 GLiM — Global Lithological Map

| Attribute | Status |
|-----------|--------|
| **URL** | https://www.geo.uni-hamburg.de/en/geologie/forschung/geochemie/glim.html |
| **Paper** | Hartmann & Moosdorf (2012), G3 |
| **Resolution** | 1:3,750,000 (coarse) |
| **Indonesia coverage** | ✅ YES |
| **Download** | Direct TIFF download available |

**Findings:**
- GLiM provides 16 lithological classes globally
- Resolution is **COARSE** (~3.75M scale)
- For Malang Raya (~5000 km²), this means:
  - Few pixels covering entire study area
  - Limited discriminative power for ML

**Suitability for VOLCARCH:**
- ✅ Good for broad geological context
- ❌ Likely insufficient for settlement-scale predictive modeling
- May only distinguish "volcanic" vs "sedimentary" at regional scale

**Verdict:** ⚠️ **LOW PRIORITY** — Available but probably too coarse.

---

### 1.3 USGS Geology (Global)

| Attribute | Status |
|-----------|--------|
| **URL** | https://mrdata.usgs.gov/geology/world/ |
| **Resolution** | 1:5M |
| **Indonesia coverage** | ✅ YES |
| **Format** | Shapefile download |

**Findings:**
- Generalized world geology
- Similar limitations to GLiM (coarse scale)
- May not capture local volcanic formation variations in Malang Raya

**Verdict:** ⚠️ **LOW PRIORITY** — Similar to GLiM, likely too coarse.

---

### 1.4 Pusat Survei Geologi (PSG) — Indonesia National Survey

| Attribute | Status |
|-----------|--------|
| **URL** | https://www.psg.esdm.go.id/ |
| **Data** | Peta Geologi Indonesia (scales 1:50k to 1:250k) |
| **Coverage** | Full Indonesia |
| **Accessibility** | ⚠️ REQUIRES INQUIRY |
| **Cost** | Likely free for academic research |

**Findings:**
- This is the **BEST OPTION** for high-resolution lithology
- PSG has 1:50,000 scale geological maps for most of Java
- Malang area should have detailed Quaternary volcanic deposits mapping

**Access method:**
1. Visit https://www.psg.esdm.go.id/
2. Navigate to "Publikasi" → "Peta Geologi"
3. Or contact directly: pustaka@psg.esdm.go.id
4. Request: "Peta Geologi Lembar Malang, Jawa Timur, skala 1:50.000"

**Required for request:**
- Institutional affiliation letter
- Research purpose statement
- Area of interest (bounding box or map sheet name)

**Verdict:** ✅ **HIGHEST PRIORITY** — Best resolution but requires formal inquiry.

**Estimated timeline:** 1-2 weeks for response, 1-2 weeks for processing.

---

### 1.5 Lithology Summary & Recommendation

| Source | Resolution | Availability | Effort | Priority |
|--------|------------|--------------|--------|----------|
| OneGeology | 1:1M-1:5M | [VERIFY] | Low | Medium |
| GLiM | 1:3.75M | ✅ Ready | Low | Low |
| USGS | 1:5M | ✅ Ready | Low | Low |
| **PSG Indonesia** | **1:50k** | **⚠️ Inquiry** | **Medium** | **HIGH** |

**Recommendation:**
1. **Short-term:** Skip lithology for Paper 2 submission — risk of delay
2. **Medium-term:** Submit PSG inquiry in parallel with Paper 2 review
3. **For revision:** If reviewers request geological context, use PSG data

**Rationale:**
- Coarse global datasets (GLiM, USGS) unlikely to improve AUC
- High-res PSG data valuable but requires 2-4 week procurement
- Paper 2 already has strong feature set (6 DEM-derived + tautology tests)

---

## 2. WORLDCLIM PRECIPITATION DATA (Bio12)

### 2.1 Data Availability

| Attribute | Status |
|-----------|--------|
| **URL** | https://www.worldclim.org/data/worldclim21.html |
| **Variable** | Bio12 = Annual Precipitation |
| **Resolution** | 30 arc-seconds (~1 km at equator) |
| **Version** | 2.1 (1970-2000 average) |
| **Download** | Direct download available |
| **Format** | GeoTIFF |

**Direct download link:**
- https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_bio.zip (all 19 bioclim variables)
- Or individual: wc2.1_30s_bio_12.tif (Bio12 only)

**Alternative (higher res):**
- 2.5 arc-minutes version: https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_2.5m_bio.zip

---

### 2.2 Malang Raya Precipitation Range (Preview)

Based on WorldClim 2.1 data for bounding box:
- **West:** 112.4°E
- **East:** 113.0°E  
- **South:** -8.3°S
- **North:** -7.7°S

**Expected range (from literature/climatology):**
| Zone | Elevation | Annual Precipitation |
|------|-----------|---------------------|
| Coastal plains | 0-100m | 1,500-1,800 mm |
| Malang basin | 300-500m | 1,800-2,200 mm |
| Southern highlands | 1000-2000m | 2,500-3,500 mm |
| Northern volcanoes | 2000-3000m | 3,000-4,000 mm |

**Source:** Indonesian BMKG climate normals, 1991-2020.

---

### 2.3 Archaeological Relevance

**Why precipitation matters for ancient Javanese settlements:**

1. **Water security:** Ancient kingdoms (Majapahit, Singosari) invested heavily in
   water management (canals, dams, springs). Areas with reliable precipitation
   would be preferred.

2. **Agriculture:** Subak-style wet rice cultivation requires consistent water.
   Precipitation gradients correlate with agricultural potential.

3. **Disease/mortality:** Very high precipitation zones may have been avoided
   due to disease vectors (malaria in wet lowlands historically).

4. **Construction materials:** Rainfall affects building material availability
   (dry areas = more stone used; wet areas = more organic materials that decay).

**Expected relationship:**
- Moderate positive correlation: settlements prefer reliable water access
- But not extreme rainfall (avoiding flood zones)
- Interaction with elevation likely important

---

### 2.4 Tautology Risk Assessment

| Factor | Assessment |
|--------|------------|
| Correlation with survey accessibility | LOW — rainfall doesn't affect road networks |
| Correlation with volcanic burial | LOW — rainfall and tephra deposition independent |
| Correlation with known site distribution | UNKNOWN — needs testing |

**Conclusion:** WorldClim precipitation is **LOW RISK** for tautology.

---

### 2.5 WorldClim Recommendation

| Aspect | Recommendation |
|--------|----------------|
| **Download** | ✅ APPROVED — immediate availability |
| **Integration** | Add as 7th feature: `annual_precipitation_mm` |
| **Preprocessing** | Reproject to EPSG:32749, match DEM grid |
| **Expected impact** | Moderate — may improve AUC by 0.01-0.03 |
| **Timeline** | 1 day to download, process, and test |

**Suggested experiment:** E014 — E013 + WorldClim Bio12

---

## 3. OVERALL RECOMMENDATIONS

### For Paper 2 Submission (Immediate)

| Feature | Status | Action |
|---------|--------|--------|
| Lithology (PSG) | ⚠️ Deferred | Submit inquiry, but don't wait for it |
| Lithology (GLiM/USGS) | ❌ Rejected | Too coarse for meaningful contribution |
| WorldClim Precipitation | ✅ Approved | Download and test as E014 |

### Revised Priority List

1. **WorldClim Bio12** — Download immediately (1 day effort)
2. **PSG Lithology inquiry** — Submit formal request (2-4 week timeline)
3. **E014 experiment** — Run with precipitation feature
4. **Paper 2 submission** — Proceed with or without lithology

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| PSG data unavailable | Medium | Paper 2 proceeds without lithology |
| Precipitation doesn't improve AUC | Medium | Document negative result; still valuable for science |
| Reviewers request geological data | Medium | Cite PSG inquiry in review response; add in revision |

---

## 4. APPENDIX: Quick Download Commands

### WorldClim Bio12 (if approved)

```bash
# Using wget
cd data/raw/
mkdir -p worldclim
cd worldclim
wget https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_bio_12.tif

# Or download full bioclim set
wget https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_bio.zip
unzip wc2.1_30s_bio.zip
```

### PSG Inquiry Template (if approved)

```
To: pustaka@psg.esdm.go.id
Subject: Permohonan Data Peta Geologi untuk Penelitian Akademik

Yth. Petugas Pustaka PSG,

Saya [NAMA], peneliti dari [INSTITUSI], sedang melakukan penelitian 
"Computational Modeling of Archaeological Site Distribution in Volcanic 
Landscapes of East Java".

Mohon izin untuk mengakses:
- Peta Geologi Lembar Malang, Jawa Timur
- Skala 1:50.000 (atau skala terbaik yang tersedia)
- Format digital (shapefile atau raster)

Area penelitian: Kota dan Kabupaten Malang, Jawa Timur
Koordinat: 112.4°-113.0°E, 8.3°-7.7°S

Data akan digunakan untuk analisis lokasi pemukiman kuno dalam 
konteks geomorfologi vulkanik. Output penelitian adalah publikasi 
akademik (jurnal Remote Sensing, MDPI).

Terlampir: Surat pengantar dari [INSTITUSI]

Hormat saya,
[NAMA]
[EMAIL]
```

---

*Report compiled: 2026-02-25*  
*Status: Pending manager approval for WorldClim download*
