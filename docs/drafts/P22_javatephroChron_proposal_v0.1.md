# VOLCARCH Working Proposal — Paper 19 (Draft v0.1)

# JavaTephroChron: A Multi-Eruption Stratigraphic Framework for Archaeological Site Prediction and Time-Resolved Sedimentation History in Volcanic Java

**Status:** Idea capture — NOT FOR CIRCULATION
**Tanggal:** Maret 2026
**Author:** Mukhlis Amien, Universitas Bhinneka Nusantara / STIKI Malang
**Compute resources tersedia:** 4× Intel Core i9, 4× RTX 4080, 128GB RAM total
**Depends on:** Paper 17 (TobaSim-Nusantara) — FALL3D infrastructure
**Target journal (kandidat):**
- *Quaternary Science Reviews* (Q1, Elsevier)
- *Journal of Archaeological Science* (Q1)
- *Geoarchaeology* (Wiley)
- *Quaternary International*

---

## 1. Latar Belakang dan Kegelisahan

### 1.1 Masalah dengan Satu Angka

VOLCARCH Paper 1 menetapkan laju sedimentation vulkanik di dataran Jawa sebesar 3.5–6.2 mm/tahun berdasarkan lima titik kalibrasi independen. Ini adalah kontribusi yang signifikan — pertama kalinya ada quantified estimate yang robust untuk proses ini.

Namun satu angka rata-rata menyembunyikan sesuatu yang penting: **laju sedimentation tidak konstan sepanjang waktu**. Jawa mengalami puluhan erupsi besar dalam 2.000 tahun terakhir saja. Setiap erupsi besar mendeposit material secara tiba-tiba dan masif — diikuti periode yang relatif tenang. Laju "rata-rata" adalah campuran dari periode deposition cepat (segera setelah erupsi) dan periode deposition lambat (antara erupsi).

Untuk arkeologi, perbedaan ini sangat penting:
- Artefak yang terkubur segera setelah erupsi besar bisa berada di kedalaman yang **jauh lebih dalam** dari yang diprediksi oleh laju rata-rata
- Artefak yang terkubur di periode tenang bisa berada di kedalaman yang **jauh lebih dangkal**

Model satu angka tidak bisa membedakan ini. Yang dibutuhkan adalah **time-resolved sedimentation history** — sebuah clock yang mencatat tidak hanya berapa dalam, tapi *kapan* setiap lapisan terbentuk.

### 1.2 Solusi: Multi-Eruption Stratigraphic Clock

Alam sudah menyediakan solusinya. Setiap erupsi vulkanik besar meninggalkan **tephra layer** — lapisan abu vulkanik dengan komposisi geokimia yang unik seperti fingerprint. Jika kita tahu:
1. Tanggal setiap erupsi besar (dari ice cores, radiocarbon, Ar-Ar dating, dan catatan historis)
2. Distribusi dan ketebalan ash deposit setiap erupsi (dari measurements yang sudah ada)
3. Komposisi geokimia setiap erupsi (dari published geochemical databases)

Maka kita bisa membangun sebuah **stratigraphic clock** — model 3D dari tanah Jawa di mana setiap layer bisa diberi label waktu yang presisi.

Bayangkan tanah Jawa sebagai buku dengan ratusan halaman bertanggal:

```
Permukaan (2026)
━━━━━━━━━━━━━━━━━ ← Halaman 1:  Krakatau 1883     (143 tahun lalu)
━━━━━━━━━━━━━━━━━ ← Halaman 2:  Tambora 1815      (211 tahun lalu)
━━━━━━━━━━━━━━━━━ ← Halaman 3:  Samalas 1257      (769 tahun lalu)
━━━━━━━━━━━━━━━━━ ← Halaman 4:  Merapi ~1006 CE   (~1020 tahun lalu)
━━━━━━━━━━━━━━━━━ ← Halaman 5:  Kelud ~1000 CE    (~1026 tahun lalu)
━━━━━━━━━━━━━━━━━ ← Halaman 6:  Merapi ~930 CE    (~1096 tahun lalu)
         ...      ← Halaman N:  [multiple events]
━━━━━━━━━━━━━━━━━ ← TARGET:     Pre-400 CE zone
         ...
━━━━━━━━━━━━━━━━━ ← Deep:       Toba 74.000 tahun lalu
━━━━━━━━━━━━━━━━━ ← Deeper:     Middle Toba 501.000 tahun lalu
━━━━━━━━━━━━━━━━━ ← Deepest:    Old Toba 788.000 tahun lalu
```

Setiap "halaman" ini adalah **isochronal horizon** — permukaan yang terbentuk pada waktu yang sama di seluruh area yang terdampak. Dengan membangun model ini secara komputasional, kita bisa prediksi: *di koordinat mana pun di Jawa, pada kedalaman berapa kita akan menemukan lapisan dari setiap periode?*

### 1.3 Implikasi Langsung untuk VOLCARCH

Ini bukan hanya akademik. Untuk tujuan VOLCARCH — menemukan situs arkeologi pre-400 CE yang terkubur — model ini memberikan sesuatu yang sangat konkret:

> *"Di Sangiran (koordinat 7.45°S, 110.83°E), situs arkeologi dari abad ke-3 CE kemungkinan berada antara kedalaman X cm (di bawah layer Samalas 1257) dan Y cm (di atas layer dari erupsi unknown ~300 CE). Untuk menemukan situs dari 200 BCE, gali hingga kedalaman Z cm."*

Ini mengubah "gali lebih dalam" menjadi **targeted excavation dengan depth prediction**.

---

## 2. Hipotesis dan Research Questions

### 2.1 Central Hypothesis

**H1 (Stratigraphic Clock Hypothesis):**
Kombinasi dari (a) published tephra deposit measurements, (b) FALL3D computational dispersal simulations, dan (c) published eruption chronologies dapat menghasilkan sebuah **3D time-resolved sedimentation model** untuk Jawa yang memiliki temporal resolution lebih tinggi dari model laju rata-rata yang ada, dan yang menghasilkan testable predictions tentang kedalaman artefak arkeologis dari periode tertentu.

**H2 (Depth Prediction Hypothesis):**
Predicted depths dari model ini akan berkorelasi secara signifikan dengan observed depths dari independent datasets — terutama dari colonial records (VOLCARCH Paper 18) dan dari published excavation reports.

**H3 (Visibility Window Hypothesis):**
Antara setiap dua eruption layers yang berdekatan, ada "visibility window" — zona kedalaman di mana artefak dari periode itu tersimpan. Ketebalan dan integritas visibility window ini bervariasi secara spasial (lebih tipis di dekat gunung api, lebih tebal di daerah yang terlindungi) dan bisa diprediksi secara komputasional.

**H4 (Sedimentation Rate Variability Hypothesis):**
Laju sedimentation di dataran vulkanik Jawa tidak konstan — ia berfluktuasi secara episodik mengikuti siklus erupsi. Model time-resolved akan menunjukkan variance yang signifikan di sekitar nilai rata-rata Paper 1 (3.5–6.2 mm/tahun), dengan episode post-erupsi yang jauh lebih tinggi dan periode inter-erupsi yang jauh lebih rendah.

---

## 3. Data Inventory

### 3.1 Eruption Chronology Database

Target: semua eruptions signifikan (VEI ≥ 4) dari gunung api Jawa, Sumatra, dan pulau-pulau sekitarnya yang memiliki published dates.

**Tier 1 — Presisi sangat tinggi (tanggal exact atau ±decade):**

| Erupsi | Tanggal | VEI | Dating Method | Coverage Jawa |
|---|---|---|---|---|
| Krakatau | 27 Agustus 1883 | 6 | Historical | Seluruh Jawa |
| Tambora | 10 April 1815 | 7 | Historical | Seluruh Jawa |
| Samalas/Rinjani | 1257 CE | 7 | Ice core bipolar | Seluruh Jawa |
| Kelud | 1586 CE | 5 | Historical | Jawa Timur-Tengah |
| Merapi | 1006 CE | 4-5 | Historical+radiocarbon | Jawa Tengah |

**Tier 2 — Presisi menengah (±50–100 tahun):**

| Erupsi | Tanggal | VEI | Dating Method |
|---|---|---|---|
| Merapi Holocene VEI 4+ | Multiple (400–1900 CE) | 4-5 | Radiocarbon |
| Kelud Holocene | Multiple | 4-5 | Radiocarbon |
| Lawu | ~1885 CE (last) | 3-4 | Historical |
| Merbabu | ~1797 CE (last) | 3-4 | Historical |

**Tier 3 — Presisi rendah (±1.000 tahun) tapi secara stratigrafis penting:**

| Erupsi | Tanggal | VEI | Dating Method |
|---|---|---|---|
| Toba YTT | 73.880 ± 320 ka | 8 | Ar-Ar |
| Toba MTT | 501 ± 0.7 ka | 8 | Ar-Ar |
| Toba OTT | 788–792 ka | 8 | Ar-Ar |
| Ranau Caldera | ~33 ka | 7 | Radiocarbon |
| Maninjau Caldera | ~52 ka | 7 | Radiocarbon |

**Sumber data:**
- Global Volcanism Program (GVP): volcano.si.edu — comprehensive eruption database
- PVMBG: Database letusan gunung api Indonesia
- Published literature: Newhall et al. 2000 (Merapi), Gertisser & Keller 2003 (Merapi), Lavigne et al. 2013 (Samalas)

### 3.2 Tephra Thickness Measurements

Untuk setiap eruption di atas, existing deposit thickness measurements tersedia dari:
- **PVMBG hazard assessment reports** — ground measurements di berbagai jarak
- **Published volcanological papers** — isopach maps (contour maps of ash thickness)
- **GVP eruption records** — deposit descriptions
- **Colonial records** (dari Paper 18 pipeline) — additional data points

Target: minimum 10 thickness measurements per eruption untuk inversion modeling.

### 3.3 Geochemical Fingerprint Database

Untuk setiap eruption layer, geochemical fingerprint tersedia dari:
- Published glass shard analysis (SiO₂, TiO₂, FeO, MgO, CaO, Na₂O, K₂O)
- Ini yang memungkinkan identification layer di soil core tanpa mengetahui konteks stratigrafis

**Key fingerprints yang sudah published:**
- Toba YTT: distinctive high-K rhyolite, FeO/MgO = 2.1–2.6 (biotite), SiO₂ ~76%
- Samalas 1257: trachyandesite, distinctive trace element signature
- Tambora 1815: trachyandesite dengan elevated Ba/La ratio
- Krakatau 1883: basaltic andesite, distinctive Sr isotope ratio

---

## 4. Metodologi

### 4.1 Overview Pipeline

```
INPUT: Eruption chronology + thickness measurements
       + geochemical fingerprints
       
       ↓
       
STEP 1: Single-eruption dispersal modeling
        (FALL3D per eruption, reuse infrastructure dari Paper 17)
        
       ↓
       
STEP 2: Isopach construction per eruption
        (predicted thickness map untuk setiap erupsi)
        
       ↓
       
STEP 3: Layer stacking — build 3D model
        (cumulative depth dari setiap eruption layer
         di setiap grid cell 0.1° × 0.1°)
        
       ↓
       
STEP 4: Inter-layer sediment modeling
        (antara eruption layers ada non-volcanic sedimentation)
        
       ↓
       
STEP 5: Depth prediction per coordinate per period
        ("Di koordinat X, pre-400 CE zone ada di kedalaman Y±Z cm")
        
       ↓
       
STEP 6: Visibility window mapping
        (map of where different archaeological periods
         are accessible at what depths)
        
       ↓
       
STEP 7: Validation
        (cross-check dengan colonial records database,
         published excavation reports, known site depths)
         
       ↓
       
OUTPUT: JavaTephroChron 3D model + Archaeological Target Map
```

### 4.2 FALL3D Infrastructure (dari Paper 17)

Paper 17 sudah membangun seluruh FALL3D pipeline untuk simulasi Toba. Paper 19 *reuses* infrastructure yang sama untuk setiap eruption dalam database.

Perbedaan parameter per eruption:
```python
eruption_params = {
    'Krakatau_1883': {
        'source_location': (-6.10, 105.42),
        'erupted_mass': 2.1e13,  # kg
        'column_height': 40,     # km
        'duration': 18,          # hours
        'vei': 6,
        'grain_size': 'medium_fine'
    },
    'Tambora_1815': {
        'source_location': (-8.25, 117.98),
        'erupted_mass': 1.4e14,  # kg
        'column_height': 43,     # km
        'duration': 36,          # hours
        'vei': 7,
        'grain_size': 'fine'
    },
    'Samalas_1257': {
        'source_location': (-8.42, 116.46),
        'erupted_mass': 2.0e14,  # kg — larger than Tambora
        'column_height': 43,     # km
        'duration': 48,          # hours
        'vei': 7,
        'grain_size': 'fine'
    },
    # ... semua eruptions dari database
}
```

**Compute requirement:**
- 1 simulation per eruption
- ~20–50 major eruptions dalam full database
- Dengan RTX 4080: ~2–4 jam per simulation = **2–8 hari total compute time**
- Jauh lebih ringan dari Toba ensemble (100+ simulations)

### 4.3 Layer Stacking Algorithm

```python
import numpy as np
import xarray as xr

def build_stratigraphic_model(eruption_results, inter_eruption_model):
    """
    Build 3D stratigraphic model dari stack eruption layers.
    
    Parameters:
    - eruption_results: dict {eruption_name: thickness_grid}
    - inter_eruption_model: sedimentation rate untuk non-volcanic periods
    
    Returns:
    - stratigraphy: xarray Dataset dengan depth per eruption layer
                   per grid cell
    """
    # Sort eruptions chronologically
    eruptions_sorted = sorted(
        eruption_results.items(),
        key=lambda x: x[1]['date_ka'],
        reverse=True  # oldest first
    )
    
    # Initialize cumulative depth grid
    lat_grid = np.arange(-10, 8, 0.1)
    lon_grid = np.arange(100, 125, 0.1)
    cumulative_depth = np.zeros((len(lat_grid), len(lon_grid)))
    
    # Stack layers
    layer_depths = {}
    for i, (eruption_name, data) in enumerate(eruptions_sorted):
        
        # Add inter-eruption sedimentation
        if i > 0:
            prev_date = eruptions_sorted[i-1][1]['date_ka']
            curr_date = data['date_ka']
            years_between = (prev_date - curr_date) * 1000
            inter_sediment = inter_eruption_model * years_between
            cumulative_depth += inter_sediment
        
        # Record depth of this eruption layer
        layer_depths[eruption_name] = cumulative_depth.copy()
        
        # Add tephra from this eruption
        cumulative_depth += data['thickness_grid']
    
    return layer_depths

# Output: untuk setiap koordinat, depth dari setiap layer
# e.g., layer_depths['Samalas_1257'][lat_idx][lon_idx] = 
#        depth in cm dari permukaan ke layer Samalas 1257
```

### 4.4 Inter-eruption Sedimentation Model

Antara eruption layers, ada non-volcanic sedimentation (alluvial, aeolian, soil formation). Ini dimodelkan sebagai:

```
r_inter(x, y) = f(distance_to_volcano, rainfall, slope, vegetation)
```

**Data untuk kalibrasi:**
- Paper 1 values: 3.5–6.2 mm/yr total rate
- Total rate = volcanic episodes + inter-volcanic background
- Background rate estimated dari paleosol thickness antara tephra layers dalam published sections

**Literature estimate:** Background inter-volcanic rate di dataran Jawa ~0.1–0.5 mm/yr (Newhall et al. 2000; Gertisser & Keller 2003)

### 4.5 Archaeological Visibility Windows

```python
def compute_visibility_windows(layer_depths, period_boundaries):
    """
    Compute archaeological visibility windows per coordinate.
    
    period_boundaries: {
        'pre_400_CE': (-400, 400),   # BCE/CE years
        'early_hindu': (400, 700),
        'mataram': (700, 1100),
        'majapahit': (1293, 1527),
        'post_majapahit': (1527, 1830)
    }
    """
    windows = {}
    
    for period_name, (start_year, end_year) in period_boundaries.items():
        # Find eruption layers bracketing this period
        layer_above = find_nearest_eruption_above(start_year, layer_depths)
        layer_below = find_nearest_eruption_below(end_year, layer_depths)
        
        # Compute depth range
        depth_top = layer_depths[layer_above]
        depth_bottom = layer_depths[layer_below]
        
        windows[period_name] = {
            'depth_min': depth_top,     # cm dari permukaan
            'depth_max': depth_bottom,  # cm dari permukaan
            'thickness': depth_bottom - depth_top,  # ketebalan window
            'bounded_by': (layer_above, layer_below)
        }
    
    return windows
```

**Output visual:**

```
VISIBILITY WINDOWS DI SANGIRAN (prediksi):

Pre-Hindu (pre-400 CE):
  Depth: 280–320 cm
  Bounded: [Toba-era + Holocene] to [unknown ~400 CE eruption]
  Window thickness: ~40 cm
  
Early Hindu (400–700 CE):  
  Depth: 200–280 cm
  Bounded: [~400 CE eruption] to [~700 CE eruption]
  Window thickness: ~80 cm

Mataram (700–1100 CE):
  Depth: 100–200 cm
  Bounded: [~700 CE] to [Merapi ~1006 CE]
  Window thickness: ~100 cm

Majapahit (1293–1527 CE):
  Depth: 40–100 cm
  Bounded: [Samalas 1257] to [~1500 CE eruption]
  Window thickness: ~60 cm

Colonial-era surface finds (1600–1900 CE):
  Depth: 0–40 cm
  Bounded: [~1500 CE] to [Krakatau 1883]
  Window thickness: ~40 cm
```

### 4.6 Validation Strategy

**Validation Dataset 1 — Colonial Records (Paper 18):**
Colonial records database mengandung reported depths untuk accidental finds. Untuk finds yang bisa di-date (dari description), predicted depth dari model harus konsisten dengan observed depth.

**Validation Dataset 2 — Published Excavation Reports:**
Excavation reports dari Candi Sambisari, Candi Kedulan, Candi Liyangan, Liangan settlement — semua sudah memiliki published depths. Model harus reproduce these depths.

**Validation Dataset 3 — Tephra Layer Observations:**
Beberapa published stratigraphic sections di Jawa sudah memiliki tephra layers yang diidentifikasi. Model predictions harus konsisten dengan observed layer positions.

**Statistical validation:**
```python
# Pearson correlation: predicted vs observed depths
from scipy import stats

predicted = [model.depth_prediction(coord, period) 
             for coord, period in validation_set]
observed = [v['reported_depth'] for v in validation_set]

r, p_value = stats.pearsonr(predicted, observed)
rmse = np.sqrt(np.mean((np.array(predicted) - np.array(observed))**2))
```

---

## 5. Primary Output: JavaTephroChron Database

### 5.1 Format

**Raster format:**
- Resolution: 0.1° × 0.1° (~11 km)
- Layers: depth prediction untuk setiap eruption dalam database
- Format: NetCDF4 (standard geoscience format)

**Untuk setiap grid cell:**
```
depth_to_Krakatau_1883: float (cm)
depth_to_Tambora_1815:  float (cm)
depth_to_Samalas_1257:  float (cm)
depth_to_Merapi_1006:   float (cm)
...
depth_to_Toba_74ka:     float (cm)

visibility_window_pre400CE_top:    float (cm)
visibility_window_pre400CE_bottom: float (cm)
visibility_window_pre400CE_thick:  float (cm)
...
```

**Query interface:**
```python
from javatephroChron import StratigraphicClock

clock = StratigraphicClock('java_tephroChron_v1.nc')

# Query specific location
sangiran = clock.query(lat=-7.45, lon=110.83)
print(sangiran.depth_to('Samalas_1257'))
# Output: 42.3 ± 8.1 cm

print(sangiran.visibility_window('pre_400_CE'))
# Output: {'top': 285, 'bottom': 340, 'thickness': 55}
# = Artefak pre-400 CE di Sangiran ada di kedalaman 285-340 cm

# Generate excavation target map
clock.plot_visibility_window(
    period='pre_400_CE',
    overlay='VOLCARCH_ZoneBC',
    output='excavation_targets.png'
)
```

### 5.2 Archaeological Target Map

Output utama untuk komunitas arkeologi: **peta target ekskavasi** yang menunjukkan:
- Lokasi dengan Zona B/C overlap (high suitability + no surface sites)
- Predicted depth untuk setiap periode arkeologis di lokasi tersebut
- Confidence interval (uncertainty dari model)

Contoh output untuk satu titik:
```
EXCAVATION TARGET REPORT — SITE: Jombang, East Java
Coordinates: 7.55°S, 112.23°E
VOLCARCH Zone: B (high suitability, no surface sites)

Recommended excavation depths:
  Colonial (1600–1880 CE):    0–15 cm
  Majapahit (1293–1527 CE):   15–55 cm  ← [Layer: Samalas 1257 at ~15 cm]
  Singhasari (1222–1292 CE):  55–90 cm
  Mataram (700–1100 CE):      90–180 cm ← [Layer: Merapi ~1006 CE at ~90 cm]
  Early Hindu (400–700 CE):   180–260 cm
  PRE-HINDU TARGET (pre-400 CE): 260–340 cm ← PRIMARY VOLCARCH TARGET
  
NOTE: Toba layer predicted at ~18 meters depth
NOTE: Uncertainty ±20% on all depth estimates
```

---

## 6. Samalas 1257 — The Key Anchor

Satu eruption yang layak mendapat perhatian khusus adalah **Samalas 1257** dari Gunung Rinjani, Lombok.

**Mengapa Samalas sangat penting untuk JavaTephroChron:**

1. **Tanggal sangat presisi:** Diidentifikasi secara independen di ice cores Greenland (GISP2, GRIP) DAN Antartika (Law Dome, Vostok) — memberikan date yang presisi ke dalam *musim* letusan (musim panas 1257 CE). Ini adalah tanggal yang tidak bisa diperdebatkan.

2. **Skala sangat besar:** VEI 7, lebih besar dari Tambora 1815 berdasarkan sulfate deposition. Sehingga layer-nya harusnya terdeteksi hampir di seluruh Jawa.

3. **Historically significant:** Letusan ini kemungkinan besar menyebabkan "Medieval Global Cooling" (~1–3°C drop). Di Jawa, dampaknya terhadap pertanian dan populasi kemungkinan sangat signifikan — ini adalah event yang mungkin terekam dalam sejarah lokal juga.

4. **Geokimia distinctive:** Trachyandesite dengan trace element signature yang unik — bisa dibedakan dari Tambora dan Krakatau.

5. **769 tahun yang lalu:** Cukup dalam untuk membentuk lapisan yang bisa diexcavate, tapi cukup muda untuk masih terdefinisi dengan baik.

**Prediksi:** Layer Samalas 1257 di dataran Jawa Tengah kemungkinan berada di kedalaman 10–50 cm dari permukaan (bervariasi tergantung lokasi). Ini adalah **layer yang paling feasible untuk diverifikasi dengan soil core sederhana** — tidak perlu bor dalam, cukup auger biasa.

**Proposed verification experiment:**
10 soil cores di berbagai lokasi di Jawa Tengah, masing-masing 100 cm dalam. Kirim sampel tiap 5 cm untuk geochemical analysis. Cari Samalas signature (trachyandesite + distinctive trace elements). Ini adalah pilot fieldwork yang affordable dan hasilnya langsung valuable.

---

## 7. Koneksi ke VOLCARCH Series

```
Paper 1  → Average sedimentation rate (3.5–6.2 mm/yr)
            Paper 19 REFINES ini menjadi time-resolved model

Paper 2  → Zona B/C spatial predictions
            Paper 19 ADDS depth predictions untuk setiap zone

Paper 17 → FALL3D infrastructure untuk Toba
            Paper 19 REUSES infrastructure untuk semua eruptions

Paper 18 → Colonial records depth database
            Paper 19 USES sebagai independent VALIDATION

Paper 19 → JavaTephroChron 3D model ← PROPOSAL INI
            Paper 20+ → Fieldwork validation
```

Paper 19 adalah **central synthesis paper** yang menghubungkan semua komponen VOLCARCH:
- Geological data (eruption records)
- Computational modeling (FALL3D)
- Spatial modeling (GIS)
- Historical records (colonial data)
- Archaeological predictions (visibility windows)

---

## 8. Novelty dan Kontribusi

1. **Pertama:** Multi-eruption stratigraphic 3D model untuk Jawa dengan resolusi temporal dan spasial yang belum pernah ada sebelumnya
2. **Pertama:** Archaeological visibility window mapping berbasis computational volcanology untuk Indonesia
3. **Pertama:** Integration tephrochronology dengan computational archaeology untuk site prediction di volcanic Java
4. **Methodological contribution:** JavaTephroChron sebagai reusable framework yang bisa diaplikasikan untuk volcanic regions di mana saja di dunia
5. **Open science:** Database dipublikasikan sebagai open access via Zenodo — memungkinkan arkeolog Indonesia menggunakannya tanpa perlu membangun model sendiri
6. **Practical output:** Excavation target map yang directly actionable untuk survey lapangan

---

## 9. Compute Resource Assessment

**Per eruption simulation (reuse FALL3D dari Paper 17):**
- Runtime: ~1–2 jam per simulation (resolusi 0.1°)
- 50 eruptions × 2 jam = 100 jam sequential
- 4× RTX 4080 parallel: ~25 jam wall time

**Total compute untuk Paper 19:**
- FALL3D simulations: ~25 jam
- Stacking + analysis: ~5 jam (CPU-intensive, tidak butuh GPU)
- **Total: ~1–2 hari compute time**

**Storage:**
- Raw simulation output: ~200 GB
- Final database (compressed): ~10 GB
- External HDD yang ada sudah cukup

---

## 10. Timeline

| Phase | Duration | Dependencies |
|---|---|---|
| Eruption database compilation | 2 minggu | Literature review |
| FALL3D batch simulations | 1 minggu (compute) | Paper 17 infrastructure |
| Stacking algorithm development | 2 minggu | Python/NumPy |
| Visibility window computation | 1 minggu | Above |
| Validation terhadap Paper 18 data | 2 minggu | Paper 18 in progress |
| Archaeological target map generation | 1 minggu | GIS/matplotlib |
| Writing | 4–6 minggu | All above |
| **Total** | **3–4 bulan** | After Paper 17 setup |

---

## 11. Immediate Action Items

**Sekarang (tanpa compute):**
- [ ] Download GVP database untuk Indonesia: volcano.si.edu/search_eruption_results.cfm
- [ ] Compile tabel eruptions VEI≥4 dari Java + Sumatra dengan tanggal yang sudah published
- [ ] Cari isopach maps untuk Tambora 1815 dan Krakatau 1883 di literature
- [ ] Download Lavigne et al. 2013 (Samalas paper) — full methods

**Setelah Paper 17 FALL3D setup selesai:**
- [ ] Run batch simulations untuk Tambora, Krakatau, Samalas sebagai Tier 1 priority
- [ ] Validate terhadap known deposit measurements
- [ ] Stack tiga layers pertama sebagai proof of concept

**Pilot fieldwork proposal (bisa diajukan ke BALARJATIM):**
- [ ] 10 soil cores @ 100 cm depth di Jawa Tengah
- [ ] Target: detect Samalas 1257 layer
- [ ] Budget estimate: Rp 10–30 juta (sangat affordable)
- [ ] Ini adalah **minimal viable fieldwork** untuk validate model

---

## 12. Closing Note

Ada sesuatu yang puitis dalam paper ini.

VOLCARCH bermula dari pengamatan sederhana: Dwarapala Singosari yang setengah terkubur. Satu patung, satu pengamatan, satu kegelisahan. Dari sana berkembang menjadi 90 eksperimen, 19 paper proposals, dan sekarang sebuah model yang mencoba memetakan setiap layer tanah Jawa dari sekarang hingga jutaan tahun ke belakang.

JavaTephroChron adalah, dalam satu makna, **peta waktu** — bukan peta ruang. Ia memberitahu kamu bukan di mana situs arkeologis berada secara horizontal, tapi **seberapa dalam kamu harus menggali, dan pada kedalaman berapa kamu akan menemukan setiap era peradaban Jawa yang telah hilang.**

Alat ini tidak akan menemukan peradaban itu sendiri. Tapi ia adalah senter yang lebih terang dan lebih fokus — yang akan memandu siapapun yang akhirnya mengangkat sekop dan mulai menggali.

> *"We cannot recover what was buried. But we can compute where it is buried — and to what depth. The rest is fieldwork."*

---

## Referensi Kunci

**Volcanic eruption data:**
- Global Volcanism Program (GVP). Smithsonian Institution. volcano.si.edu
- Newhall, C.G. et al. 2000. *Journal of Volcanology and Geothermal Research* 100: 271–338. [Merapi]
- Gertisser, R. & Keller, J. 2003. *Bulletin of Volcanology* 65: 228–249. [Merapi Holocene]
- Lavigne, F. et al. 2013. "Source of the great A.D. 1257 mystery eruption unveiled." *PNAS* 110: 16742–16747. [Samalas]
- Self, S. & Rampino, M.R. 1981. *Nature* 294: 699–704. [Krakatau 1883]
- Stothers, R.B. 1984. *Science* 224: 1191–1198. [Tambora 1815]

**Tephrochronology methods:**
- Pyle, D.M. 2015. "Sizes of volcanic eruptions." In *The Encyclopedia of Volcanoes*. Academic Press.
- Costa, A. et al. 2014. *Frontiers in Earth Science* 2:16. [FALL3D + Toba]
- Lowe, D.J. 2011. "Tephrochronology and its application." *Quaternary Geochronology* 6: 107–153.

**Indonesian tephrostratigraphy:**
- Fontijn, K. et al. 2025. *Bulletin of Volcanology* [Sumatra tephrochronology]
- Chesner, C.A. & Rose, W.I. 1991. *Bulletin of Volcanology* 53: 343–356. [Toba]

**VOLCARCH Series:**
- Amien, M. & Gunawan, G.F. Papers 1–18 (this series)

---

*Working Proposal v0.1 — Maret 2026*
*"We cannot recover what was buried. But we can compute where it is buried — and to what depth. The rest is fieldwork."*
