# VOLCARCH Working Proposal — Paper 17 (Draft v0.1)

# TobaSim-Nusantara: Rekonstruksi Komputasional Dispersal Abu Toba 74.000 Tahun Lalu dan Implikasinya terhadap Kepunahan Homo soloensis di Jawa dan Survival Manusia Pembuat Seni Gua di Sulawesi

**Status:** Idea capture — NOT FOR CIRCULATION
**Tanggal:** Maret 2026
**Author:** Mukhlis Amien, Universitas Bhinneka Nusantara / STIKI Malang
**Compute resources tersedia:** 4× Intel Core i9, 4× RTX 4080, 128GB RAM total
**Target journal (kandidat):**
- *Quaternary Science Reviews* (Q1, Elsevier)
- *Journal of Volcanology and Geothermal Research*
- *Quaternary International*
- *Journal of Human Evolution* (kalau angle paleoanthropologi kuat)

---

## 1. Latar Belakang dan Kegelisahan

### 1.1 Gap yang Belum Diisi

Erupsi supervulkan Toba (~74.000 tahun lalu) adalah erupsi eksplosif terbesar dalam 2 juta tahun terakhir. Dampaknya telah dimodelkan secara global (Costa et al. 2014; Timmreck et al. 2021) dan regional untuk India dan Afrika (Petraglia et al. 2020; Lane et al. 2013). Namun ada satu blank spot yang mengejutkan: **tidak ada simulasi resolusi tinggi untuk Jawa dan Sundaland**, meskipun kepulauan Indonesia adalah zona paling langsung terdampak Toba dan mengandung dua pertanyaan paleoanthropologis yang belum terjawab.

**Pertanyaan 1 — Jawa:** Fosil H. soloensis di Ngandong, Jawa Tengah, di-date ke 117.000–108.000 tahun lalu (Rizal et al. 2019). Toba meletus ~74.000 tahun lalu. Gap ~34.000 tahun antara fosil terakhir dan erupsi Toba sering diinterpretasi sebagai bukti bahwa Toba tidak relevan untuk kepunahan H. soloensis. Namun interpretasi ini mengandung **fundamental methodological error**: tanggal fosil adalah tanggal *deposit yang ditemukan*, bukan tanggal kepunahan populasi. H. soloensis bisa saja survive di lokasi lain di Jawa hingga mendekati atau bahkan sampai saat erupsi Toba — dan bukti keberadaan mereka di lokasi-lokasi tersebut mungkin sudah terhancurkan oleh Toba itu sendiri.

**Pertanyaan 2 — Sulawesi:** Sebuah stencil tangan di gua Liang Metanduno, Pulau Muna, Sulawesi Tenggara, baru-baru ini di-date ke minimum **67.800 tahun lalu** — menjadikannya seni gua tertua yang diketahui di dunia (Oktaviana et al. 2026, *Nature*). Ini berarti manusia yang capable membuat seni simbolik sudah ada di Sulawesi hanya ~6.200 tahun setelah erupsi Toba. Pertanyaan yang muncul: **mengapa manusia di Sulawesi survive Toba sementara H. soloensis di Jawa — hanya ~1.200 km lebih jauh dari Toba — kemungkinan tidak?**

Simulasi resolusi tinggi dapat memberikan jawaban komputasional: berapa ketebalan abu Toba di Jawa versus Sulawesi? Apakah perbedaan topografi, jarak, dan arah angin menciptakan differential survival zones yang menjelaskan mengapa Sulawesi menjadi refugia sementara Jawa tidak?

### 1.2 Insight Kunci

Dua pertanyaan yang belum pernah diajukan secara formal dalam satu framework:

> *Jika H. soloensis masih hidup di Jawa pada 74.000 tahun lalu, apakah erupsi Toba cukup untuk memusnahkan mereka — sementara manusia pembuat seni gua di Sulawesi, yang hidup pada periode yang hampir sama, berhasil survive?*

Ini bukan pertanyaan retoris. Ini pertanyaan yang bisa dijawab dengan simulasi dispersal abu resolusi tinggi.

**The Sulawesi Paradox:** Sulawesi terletak di timur Jawa, artinya secara geografis lebih jauh dari Toba dalam satu arah tapi dengan barrier laut yang memisahkannya dari Sundaland. Jawa, di sisi lain, terhubung langsung ke Sumatra (dan karenanya ke Toba) via daratan Sundaland yang terbuka. Apakah konektivitas darat Jawa-Sumatra justru menjadi kelemahan — menghantarkan pyroclastic flows dan dampak proximal lebih efisien ke Jawa? Sementara Sulawesi, terpisah oleh Selat Makassar, mendapat abu yang jauh lebih tipis?

Ini adalah **Toba Differential Survival Hypothesis** yang bisa diuji secara komputasional.

### 1.3 Konteks VOLCARCH

Paper ini merupakan ekstensi alami dari VOLCARCH series yang sudah berjalan:
- **Paper 1** (taphonomic framework): mendokumentasikan volcanic burial sebagai mekanisme penghapusan bukti arkeologis di Jawa
- **Paper 2** (settlement model): mengidentifikasi Zona B/C sebagai area high suitability yang tidak berisi situs karena taphonomic loss
- **Paper 17** (proposal ini): memodelkan secara langsung *mekanisme* penghapusan itu pada skala geologis untuk H. soloensis, menggunakan erupsi Toba sebagai case study

---

## 2. Hipotesis Utama

**H1 (Toba-Extinction Hypothesis):**
Erupsi Toba 74.000 tahun lalu memberikan dampak yang cukup untuk menyebabkan kepunahan populasi H. soloensis yang tersisa di Jawa, melalui kombinasi: (a) abu tebal yang merusak ekosistem savanna yang menjadi habitat mereka, (b) volcanic winter yang berlangsung beberapa tahun, (c) UV spike dari ozone depletion yang persisten lebih dari satu tahun, dan (d) isolasi populasi akibat landscape change.

**H2 (Refugia Hypothesis — alternatif):**
Jika ada "shadow zones" di balik pegunungan Jawa yang relatif terlindungi dari dispersal abu Toba, populasi kecil H. soloensis mungkin survive Toba dalam refugia ini, sebelum akhirnya punah karena tekanan ekologis lanjutan (hutan tropis menggantikan savanna, kompetisi dengan H. sapiens yang mulai masuk).

**H3 (Pre-Toba Extinction — null hypothesis):**
H. soloensis sudah punah sebelum Toba, sesuai dengan dating fosil terakhir di 108.000 tahun lalu. Dalam skenario ini, simulasi Toba akan menunjukkan bahwa dampaknya di Jawa cukup parah untuk *menghancurkan bukti apapun* yang mungkin ada dari periode 108.000–74.000 tahun lalu — yang justru mendukung argumen taphonomic VOLCARCH secara lebih luas.

**H4 (Sulawesi Refugia Hypothesis):**
Sulawesi terpisah dari Sundaland oleh Selat Makassar (~100 km lebar pada sea level -70m) dan dari sumber Toba oleh jarak laut ~1.500 km. Model dispersal akan menunjukkan bahwa abu Toba di Sulawesi secara signifikan lebih tipis dari di Jawa — menciptakan differential survival zone yang menjelaskan mengapa manusia pembuat seni gua di Sulawesi dapat bertahan dan berkembang hanya 6.200 tahun setelah Toba, sementara bukti human occupation di Jawa dari periode yang sama nyaris tidak ada.

**H5 (Toba as Cultural Catalyst Hypothesis):**
Bukan hanya survival — Toba mungkin secara aktif *mendorong* migrasi dan konsentrasi populasi ke refugia yang tersedia. Manusia yang survive Toba di refugia tertentu mengalami population density increase yang mendorong inovasi budaya, termasuk seni simbolik. Ini bisa menjelaskan *mengapa* seni gua Sulawesi muncul begitu cepat setelah Toba — bukan karena bertepatan, tapi karena terhubung secara kausal.

**Catatan:** Kelima hipotesis ini menghasilkan prediksi yang bisa dibedakan secara komputasional dan arkeologis.

---

## 3. Metodologi

### 3.1 Rekonstruksi Paleotopografi Sundaland 74.000 tahun lalu

**Data sources:**
- GEBCO 2023 bathymetry (resolusi 15 arc-second, ~450m — open access)
- Sea level reconstruction untuk 74 ka dari ice core + coral data (Lambeck et al. 2014; Grant et al. 2012)
  - Estimasi sea level ~74 ka: -60 hingga -80 meter dari present
- SRTM 1 arc-second DEM untuk Jawa, Sumatra, Kalimantan (resolusi ~30m)

**Output yang dihasilkan:**
Peta topografi Sundaland pada 74.000 tahun lalu menunjukkan:
- Koneksi darat Jawa-Sumatra (tidak ada Selat Sunda)
- Koneksi darat Sumatra-Semenanjung Melayu
- Jarak langsung Toba ke Sangiran: ~800 km via darat
- Jarak langsung Toba ke Ngandong (lokasi H. soloensis): ~900 km via darat
- River systems yang mungkin ada (dari DEM + bathymetry)

**Tools:** Python + GDAL + GMT (Generic Mapping Tools) — semua open source, feasible di hardware yang ada.

### 3.2 Paleo-Wind Field Reconstruction

**Challenge:** Wind fields 74.000 tahun lalu tidak bisa diobservasi langsung.

**Solusi:**
1. **Metode 1 — Constraint dari deposit aktual:** Pola distribusi abu Toba yang sudah diketahui dari 57+ lokasi pengukuran di seluruh Samudra Hindia dan Asia Selatan (data dari Costa et al. 2014) memberikan constraint pada wind field yang harus konsisten dengan distribusi yang terobservasi.

2. **Metode 2 — Climate model output:** Costa et al. (2014) sudah mengidentifikasi bahwa kondisi meteorologis yang best-fit adalah kondisi autumn (September 2005 yang dirotasi 9° berlawanan arah jarum jam). Climate model output untuk kondisi glacial ~74 ka tersedia dari PMIP4 (Paleoclimate Modelling Intercomparison Project) — open access.

3. **Metode 3 — Sensitivity analysis:** Run multiple scenarios dengan wind field variants untuk menghasilkan envelope probabilistik, bukan single deterministic result.

### 3.3 Simulasi Dispersal Abu — FALL3D

**Model:** FALL3D v8.x (open source, GPU-capable via CUDA)
- Dikembangkan oleh Barcelona Supercomputing Center
- Telah digunakan untuk simulasi Toba oleh Costa et al. (2014)
- Mendukung parallel computing dan GPU acceleration

**Hardware assessment (resources yang tersedia):**
```
4× Intel Core i9    → 4× ~24 cores = ~96 threads total
4× RTX 4080         → 4× 9728 CUDA cores = ~38.912 CUDA cores total
128GB RAM           → Sangat cukup untuk domain regional
```

**Feasibility:**
- Costa et al. (2014) menjalankan 700 simulasi menggunakan supercomputer
- Dengan 4× RTX 4080 + GPU acceleration FALL3D:
  - Single simulation: estimasi 2–8 jam
  - Ensemble 100 simulasi: estimasi 200–800 jam total
  - Dengan 4 GPU parallel: ~50–200 jam wall time
  - **Feasible dalam 1–2 minggu compute time**

**Domain komputasional:**
- Fokus: 90°E–125°E, 15°S–10°N (mencakup Sundaland + Jawa + Sulawesi + sebagian India)
- Resolusi: 0.1° × 0.1° (~11 km) — jauh lebih detail dari Costa et al. (1°×1°)
- Vertical: 30 lapisan dari permukaan hingga 50 km

**Input parameters dari literature:**
- Total erupted mass: 3.800 km³ DRE (Costa et al. 2014)
- Mass eruption rate: ~10¹¹ kg/s
- Column height: ~35–52 km (stratospheric injection)
- Eruption duration: 9–14 hari
- Grain size distribution: dari Costa et al. (2014)

**Output:**
- Peta isopach (ketebalan abu) untuk seluruh domain dengan resolusi 0.1°
- Ketebalan prediksi di lokasi kunci:
  - Sangiran (Jawa Tengah): koordinat 7.45°S, 110.83°E
  - Ngandong (Jawa Tengah): koordinat 7.33°S, 111.37°E
  - Lake Toba source: 2.75°N, 98.83°E
  - **Liang Metanduno, Muna Island (seni gua 67.800 tahun):** ~5.1°S, 122.5°E
  - **Maros-Pangkep (seni gua 51.200 tahun):** ~5.0°S, 119.6°E
  - **Leang Bulu Bettue (hominin occupation record):** ~5.0°S, 119.5°E
  - Talepu (stone tools 194 ka): ~4.7°S, 119.9°E
- **Differential ash thickness map:** Jawa vs Sulawesi — quantifying the survival advantage
- **Selat Makassar sebagai barrier:** Model apakah laut 74 ka menurunkan ash loading di Sulawesi secara signifikan

### 3.4 Analisis Dampak Ekologis

**Input dari simulasi dispersal:**
- Ketebalan abu di setiap lokasi
- Durasi volcanic winter (dari climate model)
- UV spike duration (dari ozone depletion model)

**Framework analisis:**
1. **Habitat disruption:** Abu >10 cm → complete vegetation drop. Abu 1–10 cm → partial disruption. Abu <1 cm → recoverable dalam 1–3 tahun.
2. **Savanna loss:** H. soloensis adalah spesies savanna-adapted (punah saat savanna → hutan tropis). Abu tebal di savanna corridor Jawa Tengah = habitat loss langsung.
3. **Food web collapse:** Extinctions megafauna pasca-Toba terdokumentasi. Loss of primary prey species → population crash untuk predator/scavenger besar.
4. **Population viability analysis:** Dengan population size estimates dari fosil record, berapa lama isolated population bisa bertahan di refugia pasca-Toba?

### 3.5 Sulawesi Differential Survival Analysis

**Core question:** Berapa ketebalan abu Toba di Sulawesi dibandingkan Jawa — dan apakah perbedaan ini cukup untuk menjelaskan differential survival?

**Framework analisis:**

| Parameter | Jawa (Sangiran) | Sulawesi (Maros) |
|---|---|---|
| Jarak dari Toba | ~800 km (darat) | ~1.500 km (via laut) |
| Barrier | Tidak ada (Sundaland) | Selat Makassar ~100 km |
| Prediksi abu | Model output | Model output |
| Evidence human occupation ~67 ka | Nyaris tidak ada | Seni gua tertua di dunia |
| Evidence human occupation ~74 ka | Tidak ada | Archaic hominin tools |

**Analisis tambahan:**

1. **Selat Makassar sebagai aerosol barrier:** Apakah laut yang memisahkan Sulawesi dari Sundaland secara signifikan menurunkan ash loading? Wet deposition di atas laut vs dry deposition di darat — ada perbedaan mekanisme yang bisa dimodelkan.

2. **Topographic sheltering Sulawesi:** Sulawesi memiliki pegunungan tinggi (Pegunungan Latimojong, 3.478m) yang bisa menciptakan rain shadow effect untuk ash dispersal dari arah barat.

3. **Timeline artistic emergence:** Jika model menunjukkan abu <5 mm di Maros-Pangkep pada 74 ka — konsisten dengan ecosystem recovery dalam 5–10 tahun — maka kemunculan seni gua 67.800 tahun lalu (~6.200 tahun setelah Toba) menjadi **perfectly timed** dengan post-Toba ecological recovery.

4. **Migration modeling:** Jika Jawa terlalu parah terdampak Toba untuk dihuni, model bisa digunakan untuk melacak probable migration routes dari mainland Asia ke Sulawesi yang menghindari zona abu tebal — apakah ada corridor yang melalui Kalimantan timur atau Filipina yang menjelaskan Sulawesi sebagai destinasi?

### 3.6 Tephrochronological Prediction

**Output penting dari simulasi:**
Prediksi ketebalan abu Toba di Sangiran dan sekitarnya → **testable prediction** untuk fieldwork.

Jika model prediksi mengatakan abu Toba setebal X cm di koordinat Y,Z — maka soil core di koordinat tersebut harus menunjukkan lapisan geokimia yang konsisten dengan YTT (geochemical fingerprint biotite FeO/MgO: 2.1–2.6) pada kedalaman yang sesuai dengan ~74.000 tahun lalu.

Ini adalah **falsifiable prediction** yang menghubungkan simulasi komputasional dengan physical fieldwork — jembatan antara Paper 17 (komputasional) dan potensi Paper 18 (fieldwork validation).

---

## 4. Timeline dan Resource Planning

### 4.1 Setup Phase (Minggu 1–2)
- Install dan konfigurasi FALL3D v8.x dengan CUDA support
- Download dan proses GEBCO bathymetry untuk paleotopografi
- Download PMIP4 climate model output untuk ~74 ka
- Validasi setup dengan menjalankan test simulation skala kecil

### 4.2 Paleotopography Reconstruction (Minggu 3–4)
- Python scripting: combine SRTM DEM + GEBCO bathymetry
- Apply sea level correction (-70m) menggunakan GDAL
- Generate coastline Sundaland 74 ka
- Visualize hasilnya dengan GMT atau matplotlib

### 4.3 Pilot Simulations (Minggu 5–8)
- Run 20–30 pilot simulations dengan parameter variation
- Validate terhadap known deposit locations (57 measurement points dari Costa et al.)
- Identify best-fit parameter range
- Document GPU performance metrics

### 4.4 Full Ensemble (Minggu 9–16)
- Run 100+ simulations dengan full parameter ensemble
- Focus on Sundaland domain dengan resolusi 0.1°
- Extract time series di lokasi kunci
- Generate probabilistic isopach maps

### 4.5 Analysis dan Writing (Minggu 17–24)
- Ecological impact analysis
- Population viability modeling
- Tephrochronological prediction generation
- Draft paper

**Total timeline estimasi: 6 bulan** (bisa parallel dengan paper lain yang sedang review)

---

## 5. Compute Resource Assessment

### Hardware yang Tersedia
```
Node 1: Core i9 + 32GB RAM + RTX 4080
Node 2: Core i9 + 32GB RAM + RTX 4080
Node 3: Core i9 + 32GB RAM + RTX 4080
Node 4: Core i9 + 32GB RAM + RTX 4080
Total:  ~96 CPU threads + 38.912 CUDA cores + 128GB RAM
```

### FALL3D GPU Requirements
- FALL3D v8.x mendukung OpenMP (CPU parallel) dan CUDA (GPU)
- Memory requirement per simulation: ~4–8 GB GPU memory
- RTX 4080 memiliki 16GB VRAM → bisa run 2 simulations per GPU
- 4 GPU × 2 = **8 simulations simultaneously**

### Estimasi Waktu
| Scenario | Single sim | 100 sims (sequential) | 100 sims (8 parallel) |
|---|---|---|---|
| Low res (0.25°) | ~30 min | ~50 jam | ~6 jam |
| Med res (0.1°) | ~2 jam | ~200 jam | ~25 jam |
| High res (0.05°) | ~8 jam | ~800 jam | ~100 jam |

**Kesimpulan:** 4× RTX 4080 **lebih dari cukup** untuk ensemble 100 simulasi pada resolusi 0.1° dalam 1–2 hari compute time. Ini jauh lebih feasible dari yang dibutuhkan.

### Tambahan yang Mungkin Diperlukan
- Storage: ~500GB–1TB untuk output simulation files (bisa gunakan external HDD)
- Koordinasi 4 node: bisa menggunakan simple bash scripts atau MPI, tidak butuh dedicated cluster software

---

## 6. Novelty dan Kontribusi

### 6.1 Yang Baru dari Paper Ini
1. **Pertama:** Simulasi dispersal abu Toba dengan resolusi tinggi (0.1°) khusus untuk Sundaland, Jawa, dan Sulawesi
2. **Pertama:** Menggunakan paleotopografi Sundaland yang realistis (sea level -70m) bukan topografi modern
3. **Pertama:** Menghubungkan simulasi Toba secara eksplisit dengan kepunahan H. soloensis di Jawa melalui habitat disruption analysis
4. **Pertama:** Menggunakan seni gua Sulawesi (67.800 tahun — tertua di dunia, Oktaviana et al. 2026) sebagai independent survival marker untuk validasi model
5. **Pertama:** Memodelkan *differential survival* Jawa vs Sulawesi pasca-Toba secara komputasional — menjelaskan mengapa bukti human occupation muncul di Sulawesi tapi tidak di Jawa pada periode yang sama
6. **Pertama:** Menghasilkan testable tephrochronological predictions untuk Sangiran, Ngandong, dan Maros-Pangkep
7. **Extension VOLCARCH:** Memperlihatkan bahwa volcanic erasure bukan hanya terjadi pada timescale historical (3.5–6.2 mm/yr) tapi juga pada timescale catastrophic (Toba reset) — dan bahwa Sulawesi menjadi "control group" alami karena terlindungi dari erasure yang sama

### 6.2 Posisi dalam Literature
Paper ini mengisi gap antara:
- Global Toba climate modeling (Costa et al. 2014; Timmreck et al. 2021)
- H. soloensis paleoanthropology (Rizal et al. 2019)
- Sulawesi rock art archaeology (Oktaviana et al. 2026; Oktaviana et al. 2024)
- VOLCARCH taphonomic framework (Amien & Gunawan, Papers 1–16)

Tidak ada paper yang saat ini menghubungkan keempat domain ini secara komputasional.

**The key insight yang membedakan paper ini:** Seni gua Sulawesi 67.800 tahun lalu bukan hanya temuan arkeologis biasa — dalam konteks paper ini, ia berfungsi sebagai **independent proxy untuk human survival post-Toba di Sulawesi**. Kombinasi dengan absennya bukti setara di Jawa menciptakan natural experiment: dua wilayah, satu catastrophic event, dua outcomes yang berbeda. Simulasi kita memberikan mekanisme komputasional untuk menjelaskan perbedaan tersebut.

---

## 7. Risiko dan Mitigasi

| Risiko | Probabilitas | Mitigasi |
|---|---|---|
| FALL3D setup kompleks | Medium | Ada dokumentasi lengkap + community support |
| Paleo-wind field tidak representatif | Medium | Sensitivity analysis dengan multiple scenarios |
| H. soloensis sudah punah sebelum Toba | Medium | Null hypothesis tetap menghasilkan taphonomic argument yang valuable |
| Hasil tidak dramatis (abu tipis di Jawa) | Low-Medium | Tetap publishable — menunjukkan mengapa Toba bukan penyebab langsung |
| Review butuh geologist co-author | Medium | Target geolog ITB/UGM via TARGET B dissemination roadmap |

---

## 8. Co-authorship Strategy

Paper ini sangat membutuhkan co-author dari:
1. **Geologist / Volcanologist** — untuk validasi parameter FALL3D dan interpretasi hasil
   - Target: kontak di Departemen Geologi ITB atau UGM
   - Pendekatan: "Saya punya model dan compute resources, Anda punya expertise — mari kolaborasi"

2. **Paleoanthropologist** — untuk interpretasi H. soloensis extinction
   - Target: peneliti BRIN yang kerja di Sangiran atau Homo erectus Java
   - Pendekatan: via BALARJATIM connection

3. **Go Frendi Gunawan** — existing collaborator yang sudah familiar dengan VOLCARCH methodology

---

## 9. Connections ke VOLCARCH Series

```
Paper 1  (taphonomic framework)
         ↓ establishes sedimentation rates
Paper 2  (settlement model)
         ↓ identifies buried sites
Paper 17 (TobaSim-Java) ← PROPOSAL INI
         ↓ models catastrophic reset event
         ↓ generates tephrochronological predictions
Paper 18 (TBD - fieldwork validation?)
         ↓ soil core verification
```

Paper 17 adalah **missing link** antara computational modeling dan physical fieldwork dalam VOLCARCH series.

---

## 10. Quick Action Items

**Immediate (minggu ini):**
- [ ] Download FALL3D v8.x dari Barcelona Supercomputing Center (fall3d.readthedocs.io)
- [ ] Download GEBCO 2023 bathymetry untuk region 80°E–130°E, 20°S–15°N
- [ ] Test CUDA compatibility FALL3D dengan RTX 4080
- [ ] Download 5–10 deposit thickness measurements dari Costa et al. (2014) supplementary data

**Short term (bulan 1):**
- [ ] Rekonstruksi paleotopografi Sundaland 74 ka
- [ ] Run 5 pilot simulations untuk test pipeline
- [ ] Validate terhadap known deposit di Andaman Sea / Bay of Bengal

**Collaboration target:**
- [ ] Identifikasi satu geologist di ITB/UGM yang kerja di Quaternary volcanology
- [ ] Email pendek: "Saya punya model Toba + compute resources, butuh validasi geological"

---

## Referensi Kunci

- Costa, A. et al. 2014. "The magnitude and impact of the Youngest Toba Tuff super-eruption." *Frontiers in Earth Science* 2:16.
- Rizal, Y. et al. 2019. "Last appearance of Homo erectus at Ngandong, Java, 117,000–108,000 years ago." *Nature* 577: 381–385.
- Timmreck, C. et al. 2021. "Global climate disruption and regional climate shelters after the Toba supereruption." *PNAS* 118(28).
- Lane, C.S. et al. 2013. "Ash from the Toba supereruption in Lake Malawi shows no volcanic winter in East Africa at 75 ka." *PNAS* 110: 8025–8029.
- Lambeck, K. et al. 2014. "Sea level and global ice volumes from the Last Glacial Maximum to the Holocene." *PNAS* 111: 15296–15303.
- Chesner, C.A. & Rose, W.I. 1991. "Stratigraphy of the Toba Tuffs." *Bulletin of Volcanology* 53: 343–356.
- Oktaviana, A.A. et al. 2026. "Rock art from at least 67,800 years ago in Sulawesi." *Nature* 650: 652. doi:10.1038/s41586-025-09968-y
- Oktaviana, A.A. et al. 2024. "Narrative cave art in Indonesia by 51,200 years ago." *Nature* 631: 814–818.
- Burhan, B. et al. 2025. "A near-continuous archaeological record of Pleistocene human occupation at Leang Bulu Bettue, Sulawesi, Indonesia." *PLoS ONE*.
- Hakim, B. et al. 2025. "Hominins on Sulawesi during the Early Pleistocene." *Nature* 646: 378–383.
- GEBCO Compilation Group. 2023. GEBCO 2023 Grid. doi:10.5285/f98b053b-0cbc-6c23-e053-6c86abc0af7b
- PMIP4 (Paleoclimate Modelling Intercomparison Project 4): https://pmip4.lsce.ipsl.fr/

---

*Working Proposal v0.1 — Maret 2026*
*"The same volcano that may have erased H. soloensis in Java also failed to erase the artists of Sulawesi — and that difference is a question waiting to be modelled."*
