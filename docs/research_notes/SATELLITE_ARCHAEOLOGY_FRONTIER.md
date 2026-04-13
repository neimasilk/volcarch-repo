# Satellite Archaeology Frontier — VOLCARCH Phase 2 Computational

**Date:** 2026-04-09
**Status:** PROPOSED — next computational frontier after DHARMA mining closure
**Decision:** Pak Amien approved direction 2026-04-09

---

## Kenapa Ini Frontier Berikutnya

DHARMA mining officially CLOSED (37 eksperimen, 268 prasasti, exhausted). VOLCARCH butuh DATA BARU. Satellite imagery adalah sumber data gratis, proven, dan belum pernah diterapkan di konteks vulkanik tropis Jawa.

### Precedent (sudah proven di tempat lain)

| Project | Location | Method | Result |
|---------|----------|--------|--------|
| Parcak (2016) | Egypt | Sentinel/WorldView + ML | 17 pyramids, 3,100 settlements detected |
| Evans (2013) | Angkor, Cambodia | LiDAR | Medieval city 10x larger than thought |
| de Souza (2024) | Amazon | LiDAR + SAR | Thousands of pre-Columbian cities under canopy |
| Tapete & Cigna (2019) | Global | Sentinel-1 SAR | Subsurface detection in arid contexts |
| Chase (2012) | Belize | LiDAR | Maya cities invisible from ground |

**NONE of these** has been applied to volcanic tropical Java. This is a genuine gap.

---

## Apa yang Tersedia (Gratis)

### Data Satelit
| Satellite | Type | Resolution | Access | Relevance |
|-----------|------|------------|--------|-----------|
| **Sentinel-2** | Multispectral (13 bands) | 10m | Free (Copernicus) | Crop marks, vegetation anomalies |
| **Sentinel-1** | SAR (C-band radar) | 5-20m | Free (Copernicus) | Penetrates vegetation, detects subsurface moisture |
| **Landsat 8/9** | Multispectral (11 bands) | 30m | Free (USGS) | Long time series (1984-now) |
| **ALOS PALSAR** | SAR (L-band radar) | 12.5m | Free (JAXA) | Deeper penetration than C-band |
| **GLO-30 DEM** | Elevation | 30m | Free (Copernicus) | Already used (E003) |
| **SRTM** | Elevation | 30m | Free (NASA) | Microtopography analysis |

### Compute
- **RTX 4080** (16 GB VRAM) — sufficient for U-Net, ResNet, or Vision Transformer training
- **Python ecosystem**: rasterio, GDAL, PyTorch, segmentation_models_pytorch
- **Google Earth Engine** (free) — cloud processing for large-area analysis

---

## Strategi Bertahap

### Phase A: Feasibility Study — **COMPLETED (E189)**
1. ~~Download Sentinel-2 tiles untuk 3 target zones dari E080/E097~~
2. ~~Compute vegetation indices (NDVI, NDWI, MSAVI) + band ratios~~
3. ~~Compare known candi locations with spectral signatures~~
4. ~~Question: apakah buried structures menghasilkan anomali spektral di andosol?~~
**Result: WEAK BUT REAL SIGNAL.** NDWI (water index) p=0.032, NDVI p=0.095, all 5 metrics favor candi (sign test p=0.031). Buried stone alters soil moisture → detectable at 10m. But insufficient for standalone prospection. **→ Proceed to Phase B (SAR).**

### Phase A.2: SAR Feasibility — **COMPLETED (E190, RULED OUT)**
- Sentinel-1 GRD C-band (VV/VH) tested at 15 candi + 5 controls
- **C-band SAR RULED OUT:** reflects off canopy, not ground (Cohen's d = -0.92 wrong direction)
- L-band SAR (ALOS PALSAR, 24 cm) could penetrate deeper — untested

### Phase A.3: Multi-temporal — **COMPLETED (E191)**
- Dry vs wet season NDWI comparison at all 20 sites
- **New metric: delta local variance (p=0.066)** — candi lvar increases wet season, controls decrease
- Dry-season optical NDWI remains best single metric (p=0.032)

### Phase B: Training Data (~1 minggu)
1. Compile GLOBAL training dataset dari published satellite archaeology:
   - Egyptian buried structures (Parcak dataset jika available)
   - Angkor crop marks
   - Amazon geometric earthworks
2. Augment dengan VOLCARCH data: 142 candi locations as POSITIVE samples
3. Generate NEGATIVE samples dari area tanpa situs

### Phase C: Model Training (~1-2 minggu)
1. Transfer learning: pretrained ResNet/EfficientNet → fine-tune on archaeology
2. U-Net semantic segmentation: pixel-level prediction of "archaeological probability"
3. Train on global data → apply to Java tiles
4. RTX 4080: batch training overnight, inference on full East Java ~2-4 jam

### Phase D: Validation & Integration (~1 minggu)
1. Compare model predictions vs E080/E097 fieldwork targets
2. Overlap analysis: do spectral anomalies coincide with settlement model (E013)?
3. Generate "dig here" probability map overlay
4. Register predictions for Zenodo (falsifiable, GPS-precise)

---

## Apa yang Bisa Dideteksi

### Kemungkinan deteksi di andosol vulkanik:
| Feature | Mechanism | Confidence |
|---------|-----------|:---:|
| **Crop marks** | Differential vegetation growth over buried walls/floors | MEDIUM — works in temperate, unknown in tropical andosol |
| **Soil moisture anomalies** | Buried structures alter drainage → SAR signature | MEDIUM-HIGH — SAR penetrates vegetation |
| **Microtopography** | Subtle elevation changes from buried structures | LOW-MEDIUM — 30m DEM too coarse, LiDAR would be better |
| **Thermal anomalies** | Buried stone retains heat differently | LOW — Landsat thermal band too coarse (100m) |
| **Vegetation species changes** | Different plants grow over buried structures | HIGH — proven in Amazon and Mediterranean |

### Paling menjanjikan: SAR (Sentinel-1)
SAR menembus vegetasi dan sensitif terhadap kelembaban tanah. Struktur terkubur mengubah drainase → anomali kelembaban → detectable oleh SAR. Ini bekerja di SEMUA musim, siang-malam.

---

## Hubungan dengan Eksperimen yang Ada

| Existing | Connection |
|----------|------------|
| E013 (settlement model, AUC 0.768) | Overlay: model prediksi + satelit anomali |
| E080 (fieldwork targets, 20 GPS) | Validasi: apakah target tumpang tindih? |
| E097 (anomaly detection, 65% overlap) | Triple convergence: model + anomaly + satellite |
| E166 (burial depth map) | Filter: hanya area dengan kedalaman <3m (SAR max depth) |
| E076 v2 (NDVI script, belum jalan) | Starting point: script sudah ada |

---

## Risiko & Mitigasi

| Risk | Mitigation |
|------|------------|
| Andosol terlalu homogen → no spectral contrast | Multi-temporal analysis (dry vs wet season) |
| Vegetasi tropis terlalu tebal → no ground signal | SAR penetrates vegetation; use L-band PALSAR |
| Training data dari konteks berbeda → poor transfer | Fine-tune with Javanese candi positives |
| False positives dari geologi (lava flows, faults) | Cross-reference with geological maps |
| Resolution terlalu kasar untuk individual sites | Use as AREA-level screening, not site-level detection |

---

## Ekspektasi Realistis

**Best case:** Menemukan cluster anomali spektral di Zone B/C yang tumpang tindih dengan settlement model predictions. Ini = "computational GPR" yang mengarahkan fieldwork ke lokasi paling menjanjikan. Paper-worthy (Remote Sensing of Environment, Q1).

**Worst case:** Andosol vulkanik tropis terlalu homogen untuk deteksi satelit. Ini sendiri = informative negative yang belum pernah dipublikasikan. Still paper-worthy sebagai "limits of satellite archaeology in volcanic tropical contexts."

**Either outcome is a contribution.**

---

## Ide Paper: P23 "Seeing Through Volcanic Soil"

**Title:** "Seeing Through Volcanic Soil: Satellite Remote Sensing for Archaeological Prospection in Tropical Andosol Landscapes"

**Target:** Remote Sensing of Environment (Q1, IF 13.5) atau Journal of Archaeological Science (Q1)

**Hook:** "Satellite archaeology has found cities under Amazon canopy and pyramids under Egyptian sand. Can it find settlements under Javanese volcanic deposits?"

---

*Frontier ini tidak menggantikan fieldwork — tapi bisa mengarahkan fieldwork ke tempat yang PALING mungkin menghasilkan temuan, menghemat ratusan ribu dolar biaya survei.*
