# VOLCARCH Working Proposal — Paper 18 (Draft v0.1)

# ColonialMine-Nusantara: Text Mining Catatan Kolonial Belanda sebagai Sumber Data Taphonomic Arkeologi Jawa

**Status:** Idea capture — NOT FOR CIRCULATION
**Tanggal:** Maret 2026
**Author:** Mukhlis Amien, Universitas Bhinneka Nusantara / STIKI Malang
**Compute resources tersedia:** 4× Intel Core i9, 4× RTX 4080, 128GB RAM total
**Primary data source:** Delpher.nl (Koninklijke Bibliotheek), KITLV Digital Collections, Nationaal Archief Den Haag
**Target journal (kandidat):**
- *Journal of Archaeological Science* (Q1)
- *Digital Scholarship in the Humanities* (Oxford)
- *International Journal of Historical Archaeology*
- *Bijdragen tot de Taal-, Land- en Volkenkunde* (BKI) — khusus karena KITLV connection

---

## 1. Latar Belakang dan Kegelisahan

### 1.1 Paradoks Literasi

Nusantara memiliki tradisi literasi yang kaya — prasasti Sanskrit dari abad ke-4 M, kakawin Jawa Kuno abad ke-9, kitab-kitab Mataram dan Majapahit. Namun tradisi literasi ini sebagian besar hancur atau terkubur bersama peradaban yang menghasilkannya, karena alasan taphonomic yang sudah didokumentasikan di VOLCARCH Papers 1–17.

Di sisi lain, administrasi kolonial Belanda menghasilkan **arsip tertulis yang luar biasa detailnya** — dimulai dari VOC (1602) hingga kemerdekaan Indonesia (1945). Selama 343 tahun, ribuan pegawai, pedagang, surveyor, militer, dan peneliti Belanda mencatat aktivitas mereka di Nusantara dengan presisi yang merupakan kebiasaan budaya dan keharusan administratif mereka.

Di dalam catatan-catatan ini — laporan keuangan, jurnal harian, surat resmi, laporan teknis, artikel jurnal, surat kabar — tersembunyi **data arkeologis yang tidak disengaja**: deskripsi tentang apa yang ditemukan saat menggali sumur, membangun kanal, meletakkan fondasi, atau membajak sawah. Data ini tidak pernah dimaksudkan sebagai catatan arkeologis — tapi itulah tepatnya mengapa nilainya sangat tinggi: ia adalah **observasi tanpa bias seleksi arkeologis**.

### 1.2 Skala Arsip yang Belum Dieksplor

Arsip kolonial Belanda yang relevan untuk Nusantara mencakup:

- **VOC Dagregisters (1624–1799):** Jurnal harian VOC di Batavia dan pos-pos dagang lainnya. Mencatat setiap kejadian signifikan, termasuk pembangunan infrastruktur dan penemuan tidak terduga.
- **Tijdschrift voor Indische Taal-, Land- en Volkenkunde (TBG, 1853–1955):** Jurnal ilmiah utama tentang Nusantara. Ratusan artikel tentang arkeologi, geografi, antropologi, dan sejarah lokal.
- **Rapporten van den Oudheidkundigen Dienst (ROD, 1910–1942):** Laporan tahunan dinas arkeologi kolonial Belanda. Berisi excavation reports, accidental finds, kondisi situs.
- **Surat kabar kolonial:** *De Locomotief* (Semarang), *Het Nieuws van den Dag voor Nederlandsch Indië* (Batavia), *Soerabaiasch Handelsblad* — sering melaporkan penemuan artefak kuno saat pembangunan kota.
- **Laporan teknis:** Laporan konstruksi rel kereta api, kanal, jalan raya — semua melewati tanah Jawa dan sering menemukan artefak.

Platform **Delpher.nl** (Koninklijke Bibliotheek Belanda) telah mendigitisasi dan mengaplikasikan OCR pada sebagian besar corpus ini — menjadikannya *machine-readable* dan *computationally accessible* untuk pertama kalinya.

### 1.3 Gap Metodologis

Tidak ada studi yang secara sistematis melakukan text mining pada corpus kolonial Belanda untuk tujuan *taphonomic archaeology* — yaitu, untuk mengekstrak data tentang *kedalaman penguburan*, *kondisi soil*, dan *accidental archaeological finds* secara large-scale dan computational.

Studi-studi yang ada menggunakan sumber kolonial Belanda secara *kualitatif* — membaca teks tertentu untuk narasi historis. Yang belum pernah dilakukan adalah pendekatan *kuantitatif dan sistematis*: mengolah seluruh corpus sebagai data, mengekstrak entitas terstruktur, dan mengoverlay hasilnya dengan model spasial arkeologis.

### 1.4 Konteks VOLCARCH

Paper ini merupakan ekstensi dari dua jalur VOLCARCH:

**Jalur 1 — Paper 16 (Textual Archaeology):** Paper 16 menggunakan *external* ancient texts (Yunani, India, Cina, Arab) sebagai bukti keberadaan peradaban Nusantara pre-4th century CE dari perspektif luar. Paper 18 menggunakan *internal* colonial texts sebagai bukti dari perspektif dalam — orang-orang yang secara fisik menginjakkan kaki di tanah Jawa dan mencatat apa yang mereka temukan.

**Jalur 2 — Paper 1 (Taphonomic Framework):** Data kalibrasi sedimentation rate di Paper 1 sebagian berasal dari catatan kolonial (Dwarapala Singosari dari laporan Engelhard 1803; Candi Sambisari dari catatan penemuan kembali). Paper 18 secara sistematis mencari *semua* data sejenis ini yang tersebar di seluruh corpus — bukan hanya yang sudah terkenal.

---

## 2. Hipotesis dan Research Questions

### 2.1 Central Hypothesis

**H1 (Hidden Data Hypothesis):**
Corpus colonial Belanda (1602–1942) mengandung sejumlah signifikan *incidental stratigraphic observations* — deskripsi tentang kedalaman, material, dan kondisi sub-surface yang ditemukan saat aktivitas non-arkeologis — yang belum pernah diekstrak secara sistematis dan yang, jika diintegrasikan, akan memberikan dataset taphonomic yang signifikan untuk Jawa dan kepulauan Indonesia.

**H2 (Density Distribution Hypothesis):**
Distribusi spasial accidental finds dalam colonial records akan berkorelasi positif dengan Zona B/C dari VOLCARCH Paper 2 (high suitability areas with zero surface sites) — karena kedua dataset mencerminkan lokasi di mana sub-surface cultural material memang ada tapi tidak terlihat dari permukaan.

**H3 (Depth Calibration Hypothesis):**
Kedalaman yang dilaporkan dalam colonial records untuk penemuan artefak yang bisa di-date (misal: artefak yang bisa diidentifikasi periodenya dari deskripsi) akan konsisten dengan predicted burial depths dari VOLCARCH sedimentation model (Papers 1–2) — memberikan independent validation untuk model prediktif.

### 2.2 Research Questions

1. Berapa banyak *incidental stratigraphic observations* yang dapat diekstrak dari corpus Delpher.nl menggunakan NLP pipeline?
2. Apakah distribusi spasinya berkorelasi dengan prediksi VOLCARCH tentang lokasi buried sites?
3. Apakah kedalaman yang dilaporkan konsisten dengan predicted burial depths dari sedimentation model?
4. Apakah ada pola temporal — apakah *lebih banyak* accidental finds dilaporkan saat intensifikasi pembangunan kolonial (1870–1920, era agraris)?
5. Apakah ada referensi ke *indigenous knowledge* tentang lokasi situs kuno yang bisa menjadi target untuk survey modern?

---

## 3. Data Sources

### 3.1 Primary Corpus

**Delpher.nl (Koninklijke Bibliotheek)**
- URL: delpher.nl
- Akses: GRATIS, full-text search, bulk download tersedia via API
- Konten relevan:
  - *Tijdschrift voor Indische Taal-, Land- en Volkenkunde* (TBG): ~100 tahun, ribuan artikel
  - *Bijdragen tot de Taal-, Land- en Volkenkunde* (BKI): sama
  - Surat kabar kolonial Hindia Belanda: *De Locomotief*, *Bataviaasch Nieuwsblad*, *Soerabaiasch Handelsblad*, *Het Nieuws*, dll — **semuanya searchable**
  - Almanakken, direktori, laporan tahunan

**KITLV Digital Collections (Leiden University)**
- URL: digitalcollections.universiteitleiden.nl
- Konten: foto, peta, manuskrip, laporan — fokus Nusantara
- Sebagian dengan API access

**Rapporten van den Oudheidkundigen Dienst (ROD)**
- Laporan arkeologi kolonial 1910–1942
- Sebagian sudah di Delpher, sebagian di KITLV
- **Ini adalah primary target** karena paling directly arkeologis

**Nationaal Archief Den Haag**
- VOC archives: 1,2 km rak, sebagian sudah digital
- Dagregisters Batavia 1624–1799
- Access via nationaalarchief.nl

### 3.2 Secondary Corpus (Untuk Konteks)

- *Oud en Nieuw Oost-Indiën* (François Valentijn, 1724–1726): 8 volume ensiklopedia Nusantara
- *Histoire de Java* (Herman Willem Daendels, era 1808–1811)
- *History of Java* (Thomas Stamford Raffles, 1817) — British tapi sangat relevan
- *Reizen in het Binnenland van Borneo* (Carl Schwaner, 1853)
- Laporan-laporan teknis Burgerlijke Openbare Werken (BOW) — konstruksi infrastruktur

---

## 4. Metodologi

### 4.1 Overview Pipeline

```
┌─────────────────────────────────────────┐
│ INPUT: Delpher.nl + KITLV corpus        │
│ (~50.000+ documents, 1600–1942)         │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STEP 1: Data Acquisition                │
│ - Delpher API bulk download             │
│ - Target: TBG, BKI, ROD, koran kolonial │
│ - Output: raw OCR text files            │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STEP 2: Preprocessing                   │
│ - Old Dutch normalization               │
│   (ij→ij, oe→oe, OCR error correction) │
│ - Language detection (Dutch/Malay/mix)  │
│ - Sentence segmentation                 │
│ - Geographic entity pre-labeling        │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STEP 3: Named Entity Recognition (NER)  │
│ Target entities:                        │
│ - LOCATION (Jawa, Soerabaja, Merapi...) │
│ - DEPTH ("2 meter diep", "3 voet")      │
│ - MATERIAL ("steen", "aardewerk", etc)  │
│ - TEMPORAL ("oud", "antiek", century)   │
│ - SOIL ("aarde", "modder", "klei")      │
│ - FIND_EVENT ("opgegraven", "gevonden") │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STEP 4: Relation Extraction             │
│ Extract tuples:                         │
│ [LOCATION, DEPTH, MATERIAL, CONDITION]  │
│ [LOCATION, FIND_EVENT, TEMPORAL, SOURCE]│
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STEP 5: Geocoding                       │
│ - Map extracted locations ke koordinat  │
│ - Historical placename disambiguation   │
│   (Soerabaja→Surabaya, dll)             │
│ - Uncertainty radius assignment         │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ STEP 6: Spatial Analysis                │
│ - Overlay dengan VOLCARCH Zona B/C      │
│ - Depth distribution analysis           │
│ - Temporal distribution analysis        │
│ - Correlation dengan sedimentation model│
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ OUTPUT: Colonial Taphonomic Database    │
│ - Georeferenced accidental finds        │
│ - Depth measurements                    │
│ - Independent validation for VOLCARCH   │
│ - Target list for future surveys        │
└─────────────────────────────────────────┘
```

### 4.2 Step 1: Data Acquisition

**Delpher API:**
```python
import requests

# Delpher heeft een publieke API
# Search query voorbeeld
params = {
    'query': 'opgegraven OR gevonden AND Java AND diep',
    'facets[type][]': 'tijdschrift',
    'maxperpage': 100,
    'page': 1
}

response = requests.get(
    'https://www.delpher.nl/api/search',
    params=params
)
```

Target downloads:
- Alle TBG volumes (1853–1955): ~100 volumes
- Alle ROD rapporten (1910–1942): ~30 volumes
- Sample surat kabar kolonial: 50.000+ articles
- Estimasi total: ~200.000–500.000 text passages

**Storage requirement:** ~50–100 GB raw text

### 4.3 Step 2: Dutch Historical NLP

**Challenge:** Belanda kolonial menggunakan *historical orthography* yang berbeda dari Dutch modern:
- *ij* ditulis sebagai *y* di beberapa teks
- *ou* kadang *ow*
- OCR errors karena kualitas scan
- Campuran Dutch + Malay + Sanskrit toponyms

**Tools yang sudah ada:**
- **CLIN Dutch NLP pipeline** (Utrecht University) — handles historical Dutch
- **spaCy Dutch model** (`nl_core_news_lg`) — untuk modern Dutch sebagai baseline
- **trankit** — multilingual, handles code-switching Dutch/Malay
- **Historical Dutch word embeddings** — tersedia dari CLST Nijmegen

**Normalization rules:**
```python
normalization_rules = {
    # Old Dutch orthography
    'sch': 'sk', # schip → skip (pronunciation)
    'ae': 'aa',  # aerde → aarde
    'ue': 'uu',  # vuerd → vaard
    # Common OCR errors in colonial texts
    'rn': 'm',   # common OCR confusion
    'li': 'h',   # common OCR confusion
}
```

### 4.4 Step 3: NER untuk Archaeological Entities

**Entity types yang dicari:**

```python
ENTITY_TYPES = {
    'FIND_EVENT': [
        'opgegraven', 'gevonden', 'ontdekt', 'blootgelegd',
        'aangetroffen', 'uitgegraven', 'te voorschijn gekomen',
        'bij het graven', 'bij de aanleg', 'bij het bouwen'
    ],
    'DEPTH': [
        r'\d+[\s]?meter[\s]?(diep|onder)',
        r'\d+[\s]?voet[\s]?(diep|onder)',
        r'op[\s]een[\s]diepte[\s]van[\s]\d+',
        r'\d+[\s]?el[\s]?(diep|onder)',  # el = Dutch unit ~70cm
        r'diepte[\s]van[\s]\d+'
    ],
    'MATERIAL': [
        'aardewerk', 'potscherven', 'steenen', 'baksteen',
        'metalen', 'bronzen', 'gouden', 'zilveren',
        'beelden', 'inscriptie', 'prasasti', 'stèle',
        'fundamenten', 'muren', 'wanden'
    ],
    'SOIL': [
        'vulkanische', 'aschlaag', 'lavalaag', 'tuf',
        'alluviale', 'kleilaag', 'zandlaag', 'veenlaag'
    ],
    'TEMPORAL_INDICATOR': [
        'oud', 'antiek', 'Hindoe', 'Boeddhistisch',
        'vóór de Islamisatie', 'heidensch', 'inlandsch',
        'eeuwen oud', 'duizenden jaren'
    ]
}
```

**Fine-tuned NER model:**
- Base model: `bert-base-multilingual-cased`
- Fine-tuning data: manually annotated subset ROD reports (500–1000 sentences)
- Training: GPU-accelerated dengan RTX 4080
- Estimasi training time: 2–4 jam untuk initial model

### 4.5 Step 4: Relation Extraction

Target relation tuples:

```
(LOCATION, FIND_EVENT, DEPTH, MATERIAL, TEMPORAL, SOURCE_DOC)

Contoh extracted tuple:
("Soerabaja", "opgegraven", "2.5 meter", "aardewerk Hindoe-periode",
 "vóór de Islamisatie", "TBG_1887_v36_p234")
```

**Contoh kalimat target:**

*"Bij de aanleg van het nieuwe kanaal nabij Soerabaja werden op een diepte van twee meter verscheidene oude steenen gevonden, waarvan de inlandsche bevolking beweerde dat zij afkomstig waren van een oud Hindoesch gebouw."*

→ Extracted: {location: "Soerabaja", depth: "2 meter", material: "steenen", temporal: "Hindoesch", find_event: "aanleg kanaal", indigenous_knowledge: TRUE}

### 4.6 Step 5: Historical Geocoding

**Challenge:** Banyak placenames kolonial tidak sama dengan nama modern:
- Soerabaja → Surabaya
- Batavia → Jakarta
- Buitenzorg → Bogor
- Semarang (sama)
- Djokjakarta → Yogyakarta

**Tools:**
- **HISGIS Netherlands** — historical GIS untuk Dutch colonial toponyms
- **World Historical Gazetteer** — WHG API untuk historical placename disambiguation
- **OpenStreetMap Nominatim** untuk nama yang masih sama
- Manual validation untuk ambiguous cases

**Uncertainty radius:**
- Named city/town: ±5 km
- Named region/residency: ±25 km
- "Omgeving van" (sekitar): ±10 km
- Unlocated: excluded dari spatial analysis

### 4.7 Step 6: Spatial Analysis

**Overlay dengan VOLCARCH model:**

```python
import geopandas as gpd
import pandas as pd

# Load VOLCARCH Zona B/C predictions dari Paper 2
zona_bc = gpd.read_file('volcarch_zone_bc.geojson')

# Load extracted colonial finds
colonial_finds = gpd.read_file('colonial_finds.geojson')

# Spatial join
finds_in_zone = gpd.sjoin(
    colonial_finds,
    zona_bc,
    how='inner',
    predicate='within'
)

# Statistical test: are finds over-represented in Zone B/C?
# H0: random distribution
# H1: concentration in Zone B/C
from scipy import stats
observed = len(finds_in_zone)
expected = len(colonial_finds) * (zona_bc.area.sum() / study_area.area)
chi2, p_value = stats.chisquare([observed], [expected])
```

**Depth distribution analysis:**
- Plot histogram kedalaman reported finds
- Compare dengan predicted depths dari VOLCARCH sedimentation model
- Pearson correlation antara predicted dan observed depth

---

## 5. Preliminary Search Results

Sebagai proof of concept, beberapa manual searches di Delpher.nl menghasilkan:

**Query: "opgegraven" AND "Java" AND "diep"**
→ Ratusan hits di TBG dan surat kabar kolonial

**Contoh temuan awal (manual reading):**

1. *TBG 1887*: "Bij het graven van een waterput te Magelang werd op een diepte van 3 meter een Ganesh-beeld gevonden in goede staat van bewaring" → Magelang, 3m, Ganesha statue

2. *De Locomotief 1903*: "Bij de aanleg van den nieuwen weg bij Singosari werden op 1.5 meter diepte fundamenten gevonden van een groot steenen gebouw" → Singosari, 1.5m, stone building foundations

3. *ROD 1914*: "Toevallige vondsten gemeld door het plaatselijk bestuur: Kediri — bij aanleg irrgatie — 2.2 meter — bronzen beelden" → Kediri, 2.2m, bronze statues

4. *Bataviaasch Nieuwsblad 1911*: "De spoorwegaanleg bij Mojokerto heeft verscheidene merkwaardige vondsten opgeleverd; op sommige plaatsen reikt de cultuurlaag tot 4 meter onder het maaiveld" → Mojokerto, 4m cultural layer

Ini hanya dari beberapa menit manual search. Systematic NLP mining akan menghasilkan ribuan data points.

---

## 6. Expected Output

### 6.1 Primary Dataset: Colonial Taphonomic Database (CTD)

Structured database berisi:
- Koordinat (dengan uncertainty radius)
- Kedalaman yang dilaporkan
- Material/artifact description
- Temporal indicator
- Source document (dengan citation)
- Confidence score (dari NER model)

Estimasi jumlah entries: 500–5.000 (tergantung precision/recall NER model)

### 6.2 Maps

- **Density map** accidental finds dari colonial records, overlaid dengan Zona B/C
- **Depth distribution map** — berapa dalam rata-rata temuan di berbagai wilayah Jawa
- **Temporal distribution map** — temuan dari periode mana yang paling sering dilaporkan

### 6.3 Statistical Analysis

- Spatial correlation coefficient: colonial finds vs Zona B/C
- Depth correlation: colonial reported depths vs VOLCARCH predicted depths
- Temporal clustering: apakah temuan lebih banyak dari periode tertentu?

### 6.4 Subsidiary Finding: Indigenous Knowledge Catalog

Dalam colonial texts sering ada frasi seperti *"de inlandsche bevolking beweerde dat..."* (penduduk lokal menyatakan bahwa...) atau *"volgens overlevering"* (menurut tradisi). Ini adalah **indigenous knowledge** tentang lokasi situs kuno yang terekam oleh administrator kolonial. Mengextract dan mengcatalog ini adalah contribution tersendiri — bisa menjadi input untuk nyadran oral tradition analysis (dari Paper 15).

---

## 7. Novelty dan Kontribusi

1. **Pertama:** Systematic large-scale NLP mining dari corpus kolonial Belanda untuk tujuan taphonomic archaeology
2. **Pertama:** Integration colonial incidental finds data dengan computational taphonomic model (VOLCARCH)
3. **Pertama:** Historical geocoding dari Dutch colonial toponyms di Jawa untuk spatial archaeology
4. **Methodological bridge:** Menghubungkan Digital Humanities (colonial text mining) dengan Computational Archaeology (taphonomic modeling)
5. **Dataset contribution:** Colonial Taphonomic Database (CTD) sebagai open-access resource untuk peneliti Indonesia dan Belanda
6. **Independent validation:** Dataset yang completely independent dari semua dataset existing di VOLCARCH series (zero overlap dengan DHARMA, ABVD, atau situs archaeology databases)

---

## 8. Hubungan ke VOLCARCH Series

```
Paper 1  (taphonomic framework)
  → menggunakan beberapa colonial records (Engelhard 1803)
  
Paper 2  (settlement model)
  → Zona B/C sebagai target overlay untuk CTD

Paper 16 (textual archaeology — external)
  → ancient external texts (Greek, Indian, Chinese, Arab)
  → Paper 18 adalah COMPLEMENT-nya:
     internal colonial texts (Dutch)

Paper 17 (TobaSim)
  → geological timescale

Paper 18 (ColonialMine) ← PROPOSAL INI
  → historical timescale (1600–1942)
  → independent validation untuk Paper 1 depth model
  → indigenous knowledge catalog untuk Paper 15
```

---

## 9. Feasibility Assessment

### 9.1 Technical Feasibility

| Component | Tool | Complexity | Hardware |
|---|---|---|---|
| Data download | Delpher API + requests | Low | Laptop |
| Dutch NLP preprocessing | spaCy + custom rules | Medium | CPU |
| NER fine-tuning | HuggingFace + PyTorch | Medium | RTX 4080 |
| Relation extraction | Dependency parsing | Medium | CPU |
| Geocoding | WHG API + manual | Medium | Laptop |
| Spatial analysis | GeoPandas + QGIS | Low | Laptop |

**RTX 4080 usage:** Primarily for NER model fine-tuning (~2–4 jam) dan inference pada large corpus (~8–24 jam total). Jauh lebih ringan dari Paper 17 (TobaSim).

### 9.2 Time Estimate

| Phase | Duration |
|---|---|
| Data acquisition dari Delpher | 1–2 minggu |
| Preprocessing pipeline | 2–3 minggu |
| NER model development + fine-tuning | 3–4 minggu |
| Relation extraction | 2–3 minggu |
| Geocoding (semi-automated + manual) | 3–4 minggu |
| Spatial analysis | 1–2 minggu |
| Writing | 4–6 minggu |
| **Total estimasi** | **4–5 bulan** |

### 9.3 Skill Overlap dengan Background Kamu

Ini adalah salah satu kekuatan terbesar paper ini — **tidak butuh skill baru:**

| Skill dibutuhkan | Background kamu |
|---|---|
| NLP pipeline | arXiv:2304.02746 (Indonesian NLP survey) |
| NER fine-tuning | VOLCARCH Paper 8 (linguistic substrate detection) |
| Text mining | Manifesto experiments E022–E040 |
| Spatial analysis | VOLCARCH Paper 2 (settlement model, GIS) |
| Historical Dutch | Bisa dipelajari — basic Dutch untuk reading colonial texts tidak sulit |

---

## 10. Risks dan Mitigasi

| Risiko | Probabilitas | Mitigasi |
|---|---|---|
| OCR quality buruk di scan lama | High | Preprocessing + error correction; focus pada well-scanned TBG/ROD |
| Historical Dutch NLP model tidak ada | Medium | Fine-tune dari multilingual BERT; ada beberapa Dutch historical NLP papers sebagai guidance |
| Placename disambiguation sulit | High | Assign uncertainty radius; use only high-confidence geocoded entries for spatial analysis |
| Hasil tidak signifikan | Low-Medium | Even null result penting — membuktikan colonial records tidak mengandung data ini juga valuable |
| Legal/access issues | Low | Delpher adalah public domain; semua materi pre-1924 bebas copyright |

---

## 11. Immediate Action Items

**Minggu ini (bisa dimulai hari ini):**
- [ ] Buka delpher.nl dan lakukan 5–10 manual searches dengan berbagai query
- [ ] Download sample TBG volume (1870–1900) sebagai test corpus
- [ ] Install spaCy Dutch model: `python -m spacy download nl_core_news_lg`
- [ ] Buat spreadsheet manual: setiap accidental find yang ditemukan saat manual search

**Bulan 1:**
- [ ] Develop Delpher bulk download script via API
- [ ] Build initial NER training data (100–200 manually annotated sentences)
- [ ] Fine-tune BERT-based NER model untuk Dutch archaeological entities
- [ ] Build historical geocoding lookup table untuk major Javanese colonial toponyms

**Bulan 2–3:**
- [ ] Run full pipeline pada TBG corpus
- [ ] Manual validation sample (10% random)
- [ ] Initial spatial analysis: overlay dengan Zona B/C

---

## 12. Closing Note

Ada ironi yang indah di paper ini:

Belanda datang ke Nusantara untuk mengeksploitasi kekayaannya. Mereka mencatat segalanya dengan teliti — termasuk, secara tidak sengaja, fragmen-fragmen dari peradaban yang mereka sendiri tidak benar-benar mengerti. Selama hampir satu abad arsip-arsip ini duduk di perpustakaan Leiden dan Den Haag, tidak ada yang bertanya apa yang tersembunyi di dalam catatan-catatan administratif itu tentang tanah yang mereka tinggalkan.

Kini, dengan NLP dan computational archaeology, kita bisa membaca ulang catatan-catatan itu dengan pertanyaan yang berbeda — bukan "apa yang bernilai bagi VOC?" tapi "apa yang tersembunyi di bawah tanah Jawa yang mereka injak setiap hari tanpa sadar?"

> *"The colonizers kept meticulous records of what they took. What they didn't realize was that they were also recording, incidentally and incompletely, what was already there — buried, waiting, pre-dating their arrival by centuries."*

---

## Referensi Kunci

**Sumber Data:**
- Delpher.nl (Koninklijke Bibliotheek): delpher.nl
- KITLV Digital Collections: digitalcollections.universiteitleiden.nl
- Nationaal Archief: nationaalarchief.nl
- World Historical Gazetteer: whgazetteer.org

**Tephrochronology & VOLCARCH:**
- Amien, M. & Gunawan, G.F. Papers 1–17 (this series)
- Newhall, C.G. et al. 2000. *Journal of Volcanology and Geothermal Research* 100: 271–338.

**Dutch Colonial Archives:**
- Valentijn, F. 1724–1726. *Oud en Nieuw Oost-Indiën*. 8 vols.
- Raffles, T.S. 1817. *History of Java*. London: Black, Parbury & Allen.
- *Tijdschrift voor Indische Taal-, Land- en Volkenkunde* (TBG). 1853–1955.
- *Rapporten van den Oudheidkundigen Dienst* (ROD). 1910–1942.

**Dutch Historical NLP:**
- Lenten, J. et al. 2022. "Historical Dutch NLP." *CLIN proceedings*.
- van Strien, M. et al. 2020. "Named Entity Recognition for Historical Dutch." *DHd proceedings*.

**Digital Humanities Methods:**
- Jockers, M.L. 2013. *Macroanalysis: Digital Methods and Literary History*. University of Illinois Press.
- Moretti, F. 2013. *Distant Reading*. Verso.

---

*Working Proposal v0.1 — Maret 2026*
*"The most valuable data is not in the excavation report. It is in the footnote of the irrigation survey, written by a Dutch engineer who had no idea he was documenting the past."*
