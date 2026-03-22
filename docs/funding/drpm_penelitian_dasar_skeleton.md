# Proposal Penelitian Dasar — Skeleton DRPM

**Skema:** Penelitian Dasar (Fundamental Research)
**Pendanaan:** Rp 100-500 juta (2-3 tahun)
**Sumber:** DRPM Kemendikbudristek (via BIMA)
**Deadline:** Biasanya Januari-Februari setiap tahun
**TRL:** 1-3 (dasar/fundamental)

---

## 1. Judul Penelitian

**Prediksi Lokasi Situs Arkeologi Terkubur di Lansekap Vulkanik Jawa: Pendekatan Komputasional Multi-Faktor**

*Alternatif:*
- Model Prediktif Penguburan Vulkanik Situs Arkeologi Pra-Hindu di Jawa
- Bias Tafonomi Vulkanik dalam Rekam Arkeologi Indonesia: Validasi Lapangan Model Komputasional

## 2. Identitas Pengusul

**Ketua:**
- Nama: Dr. Mukhlis Amien
- NIDN: [isi]
- Jabatan Fungsional: [isi]
- Program Studi: [isi]
- Universitas: Universitas Bhinneka Nusantara, Malang
- SINTA ID: [isi]
- Scopus ID: [isi]
- ORCID: 0000-0002-1848-167X
- H-Index: [isi]

**Anggota 1:**
- Nama: Go Frendi Gunawan, S.Kom., M.T.
- ORCID: 0000-0003-3029-9354
- Universitas: Universitas Bhinneka Nusantara
- Keahlian: Data science, machine learning

**Anggota 2 (direkrut):**
- Keahlian dibutuhkan: Arkeologi / Geologi vulkanik / Geofisika
- Kandidat institusi: Balai Arkeologi Yogyakarta, UGM, ITB

## 3. Abstrak (~250 kata)

Pulau Jawa memiliki 45 gunung api aktif di area 129.000 km². Tidak ada titik di Jawa yang berjarak lebih dari 27 km dari gunung api aktif. Sedimentasi vulkanik terus-menerus mengubur permukaan tanah dengan kecepatan rata-rata 4,4 ± 1,2 mm/tahun — divalidasi dari empat situs kalibrasi (Dwarapala Singosari, Candi Sambisari, Candi Kedulan, Candi Kimpulan). Pada kecepatan ini, situs arkeologi dari periode pra-Hindu (~400 M) kini terkubur di bawah 7+ meter overburden vulkanik, tidak terdeteksi oleh survei permukaan konvensional.

Penelitian ini menggunakan pendekatan komputasional multi-faktor untuk memprediksi lokasi situs arkeologi terkubur di Jawa. Melalui 120 eksperimen komputasional, kami telah mengembangkan: (1) model kalibrasi laju sedimentasi multi-situs, (2) model kesesuaian pemukiman berbasis machine learning (AUC = 0,768), dan (3) model cascade visibilitas arkeologis lima faktor yang memprediksi visibilitas 0,058% — cocok dengan observasi (0,031%) dalam faktor dua.

Penelitian ini bertujuan memvalidasi model prediktif tersebut melalui survei geofisika (GPR dan ERT) di 5-10 lokasi target prioritas tinggi, analisis fitolith dari sampel inti bor, dan penanggalan radiometrik. Hasil penelitian akan memberikan kontribusi metodologis untuk arkeologi lansekap vulkanik secara global, serta peta prediksi untuk pengelolaan cagar budaya di Indonesia.

**Kata Kunci:** arkeologi komputasional, bias tafonomi, sedimentasi vulkanik, Jawa, model prediktif

## 4. Pendahuluan

### 4.1 Latar Belakang

[Struktur yang direkomendasikan:]

**Paragraf 1 — Konteks global:**
Indonesia memiliki catatan arkeologis pra-Hindu yang sangat minim dibandingkan negara-negara tetangga di Asia Tenggara (Thailand: Ban Chiang 3.600 SM; Vietnam: Dong Son 1.000 SM; Filipina: Gua Tabon 50.000 tahun). Paradoks ini terutama mencolok di Jawa, pulau terpadat dan tersubur di kepulauan ini.

**Paragraf 2 — Gap spesifik:**
Estimasi konservatif berdasarkan daya dukung pertanian padi tradisional menunjukkan populasi 590.000-3.900.000 jiwa di Jawa sebelum 400 M. Dengan ukuran desa Austronesia standar (100-200 orang), ini berarti ~3.000-20.000 pemukiman. Jumlah situs pra-400 M yang ditemukan di interior vulkanik: 0-3. Gap 3.220× lipat ini tidak dapat dijelaskan oleh kepadatan populasi rendah saja.

**Paragraf 3 — Hipotesis vulkanik:**
Kami mengajukan hipotesis bahwa bias tafonomi vulkanik — proses sedimentasi terus-menerus yang mengubur bukti material — adalah penyebab utama kekosongan rekam arkeologi ini. Penemuan aksidental Liangan (2008) — pemukiman lengkap era Mataram Kuno terkubur di bawah 6-8m deposit piroklastik — memberikan validasi dramatis terhadap hipotesis ini.

**Paragraf 4 — Kebaruan pendekatan:**
Meskipun hubungan antara vulkanisme dan preservasi arkeologi telah diakui (Ceren, Pompeii), belum ada upaya sistematis untuk mengkuantifikasi bias ini pada skala lansekap dan menggunakannya untuk prediksi lokasi situs terkubur. Proyek VOLCARCH mengisi gap ini dengan pendekatan komputasional.

### 4.2 Rumusan Masalah

1. Bagaimana laju sedimentasi vulkanik mempengaruhi visibilitas arkeologis di Jawa secara kuantitatif?
2. Di mana lokasi situs arkeologi terkubur dengan probabilitas tertinggi berdasarkan model prediktif multi-faktor?
3. Apakah survei geofisika di lokasi prediksi mengkonfirmasi keberadaan lapisan budaya terkubur?

### 4.3 Tujuan Penelitian

**Tujuan Umum:**
Mengembangkan dan memvalidasi model prediktif komputasional untuk lokasi situs arkeologi terkubur di lansekap vulkanik Jawa.

**Tujuan Khusus:**
1. Mengkalibrasi model laju sedimentasi vulkanik menggunakan data multi-situs dari dua sistem vulkanik (Kelud dan Merapi)
2. Mengembangkan model kesesuaian pemukiman berbasis machine learning yang mengintegrasikan faktor topografi, hidrologi, vulkanologi, dan proximitas candi
3. Memvalidasi prediksi model melalui survei geofisika (GPR/ERT) di 5-10 lokasi prioritas
4. Menyusun peta prediksi situs terkubur untuk pengelolaan cagar budaya oleh BPCB

### 4.4 Urgensi (Keutamaan) Penelitian

- **Penyelamatan warisan budaya:** Ekspansi pembangunan (infrastruktur, perumahan, pertambangan pasir) mengancam situs terkubur yang belum teridentifikasi
- **Gap metodologis:** Indonesia belum memiliki framework survei arkeologi bawah permukaan yang sistematis (cf. Jepang: 8.300 ekskavasi penyelamatan/tahun)
- **Kontribusi global:** Metodologi applicable untuk lansekap vulkanik di seluruh dunia (Filipina, Amerika Tengah, Mediterania)

## 5. Tinjauan Pustaka

[Topik yang harus dibahas:]

### 5.1 Bias Tafonomi dalam Arkeologi
- Schiffer (1987) — formation processes
- Prinsip bahwa rekam arkeologi ≠ rekam perilaku masa lalu
- Volcanic taphonomy sebagai subset underexplored

### 5.2 Vulkanisme dan Preservasi Arkeologi
- Pompeii (79 M) — preservasi oleh tefra
- Joya de Ceren, El Salvador (Sheets, 2002) — desa Maya terkubur letusan ~600 M
- Liangan, Jawa Tengah (Riyanto, 2014) — pemukiman Mataram Kuno terkubur piroklastik
- Perbedaan kunci: kasus di atas = peristiwa katastrofik tunggal; Jawa = sedimentasi kontinu

### 5.3 Model Prediktif dalam Arkeologi
- Predictive modeling literature (Verhagen, 2007; Westcott & Brandon, 2000)
- Machine learning dalam arkeologi prediktif (recent examples)
- Gap: belum ada model yang mengintegrasikan faktor penguburan vulkanik

### 5.4 Konteks Arkeologi Jawa Pra-Hindu
- Historiografi "kekosongan" pra-400 M
- Perspektif kolonial dan bias survei
- Bukti linguistik dan demografis untuk populasi signifikan

### 5.5 Survei Geofisika untuk Arkeologi
- GPR (Ground Penetrating Radar) — prinsip dan aplikasi
- ERT (Electrical Resistivity Tomography) — keunggulan di tanah vulkanik
- Analisis fitolith sebagai proxy pemukiman

## 6. Metode Penelitian

### 6.1 Desain Penelitian
Mixed-methods: komputasional (modeling) + empiris (survei geofisika + analisis laboratorium)

### 6.2 Tahap 1: Pengembangan Model (Tahun 1, Bulan 1-6)
- Kalibrasi laju sedimentasi dari data situs terukur (n ≥ 4)
- Refinement model kesesuaian pemukiman (XGBoost/Random Forest)
- Validasi silang dengan distribusi candi (n = 142)
- Identifikasi 10 lokasi target prioritas tinggi

### 6.3 Tahap 2: Survei Geofisika (Tahun 1 Bulan 7-12, Tahun 2 Bulan 1-6)
- **GPR:** Antena 400 MHz, grid 0,5m, penetrasi 2-5m
- **ERT:** Array Wenner, spasi elektroda 1-2m, penetrasi 10-20m
- **Coring:** Inti bor 10m di anomali GPR/ERT
- Lokasi: Kelud barat (2 situs), Penanggungan barat (2 situs), Arjuno-Welirang (1 situs)
- Koordinasi dengan BPCB Jawa Timur untuk izin dan akses

### 6.4 Tahap 3: Analisis Laboratorium (Tahun 2, Bulan 7-12)
- Analisis fitolith dari sampel inti bor (identifikasi indikator pemukiman)
- Analisis stratigrafi dan sedimentologi
- Penanggalan C-14 (jika ditemukan material organik)
- Analisis geokimia tanah

### 6.5 Tahap 4: Sintesis dan Diseminasi (Tahun 3)
- Perbandingan kedalaman anomali vs prediksi model
- Penyusunan peta prediksi final
- Publikasi di jurnal internasional bereputasi
- Penyerahan data dan peta ke BPCB

### 6.6 Analisis Data
- Statistical comparison: predicted vs observed burial depths
- Spatial analysis: GIS-based overlay of model predictions and survey results
- Machine learning model evaluation: AUC, precision-recall, feature importance

## 7. Luaran Penelitian

### 7.1 Luaran Wajib

| Tahun | Luaran | Target |
|-------|--------|--------|
| 1 | Artikel jurnal internasional bereputasi | 2 artikel (Scopus Q2+) |
| 2 | Artikel jurnal internasional bereputasi | 2 artikel (Scopus Q2+) |
| 3 | Artikel jurnal internasional bereputasi | 1 artikel (Scopus Q1) |
| 1-3 | Prosiding internasional | 2 papers |
| 3 | HKI (Hak Kekayaan Intelektual) | Software model prediktif |

### 7.2 Luaran Tambahan

- Peta prediksi situs terkubur skala 1:100.000 (diserahkan ke BPCB Jawa Timur)
- Dataset terbuka (repositori GitHub + Zenodo)
- Buku referensi / monograf
- Media populer (YouTube series, artikel populer)
- Proposal kolaborasi internasional (lanjutan)

## 8. Anggaran (3 Tahun)

### Tahun 1: Rp 180.000.000

| Komponen | Rincian | Biaya (Rp) |
|----------|---------|-------------|
| Gaji & Honor | Ketua, 2 anggota, 2 mahasiswa | 54.000.000 |
| Bahan & Peralatan | Lisensi software, komputasi cloud, GPS | 30.000.000 |
| Perjalanan | Survei lapangan awal (5 lokasi) | 36.000.000 |
| Sewa Peralatan | GPR rental (2 minggu) | 40.000.000 |
| Lain-lain | Publikasi, seminar, laporan | 20.000.000 |

### Tahun 2: Rp 200.000.000

| Komponen | Rincian | Biaya (Rp) |
|----------|---------|-------------|
| Gaji & Honor | Tim peneliti + asisten lapangan | 54.000.000 |
| Bahan & Peralatan | Consumables, core sampling | 25.000.000 |
| Perjalanan | Survei geofisika intensif | 40.000.000 |
| Sewa Peralatan | GPR + ERT (3 minggu) | 50.000.000 |
| Analisis Lab | Fitolith (20 sampel), C-14 (5 sampel) | 15.000.000 |
| Lain-lain | Publikasi, seminar, laporan | 16.000.000 |

### Tahun 3: Rp 120.000.000

| Komponen | Rincian | Biaya (Rp) |
|----------|---------|-------------|
| Gaji & Honor | Tim peneliti | 40.000.000 |
| Analisis Data | Komputasi, GIS processing | 15.000.000 |
| Perjalanan | Konfirmasi lapangan + diseminasi | 25.000.000 |
| Publikasi | Open access, prosiding, buku | 20.000.000 |
| Lain-lain | HKI, laporan akhir | 20.000.000 |

**TOTAL 3 TAHUN: Rp 500.000.000**

## 9. Jadwal Penelitian

| Kegiatan | T1-S1 | T1-S2 | T2-S1 | T2-S2 | T3-S1 | T3-S2 |
|----------|-------|-------|-------|-------|-------|-------|
| Pengembangan model | ██ | | | | | |
| Identifikasi target | ██ | | | | | |
| Survei geofisika | | ██ | ██ | | | |
| Coring & sampling | | | ██ | | | |
| Analisis laboratorium | | | | ██ | | |
| Analisis data & sintesis | | | | ██ | ██ | |
| Penyusunan peta prediksi | | | | | ██ | |
| Publikasi artikel | | ██ | | ██ | | ██ |
| Seminar/konferensi | | ██ | | ██ | | ██ |
| Laporan & HKI | | ██ | | ██ | | ██ |

*T = Tahun, S = Semester*

## 10. Daftar Pustaka

[Minimal 20 referensi, mayoritas 10 tahun terakhir, mayoritas jurnal internasional]

1. Amien, M. (2026). Multi-Site Calibration of Volcanic Sedimentation Rates across Java... *E&G Quaternary Science Journal*. [P1]
2. Amien, M. & Gunawan, G.F. (2026). Settlement Suitability Model... *Journal of Computer Applications in Archaeology*. [P2]
3. [Tambahkan paper VOLCARCH lain yang telah diterima]
4. Riyanto, S. (2014). Liangan: Mozaik Peradaban Mataram Kuno di Lereng Sindoro. *Balai Arkeologi Yogyakarta*.
5. Schiffer, M.B. (1987). *Formation Processes of the Archaeological Record*. University of New Mexico Press.
6. Sheets, P.D. (2002). *Before the Volcano Erupted: The Ancient Cerén Village in Central America*. University of Texas Press.
7. Verhagen, P. (2007). *Case Studies in Archaeological Predictive Modelling*. Leiden University Press.
8. [Tambahkan 12+ referensi relevan lainnya]

---

## Catatan Penyusunan

### Dokumen Pendukung yang Diperlukan
- [ ] Surat pernyataan ketua peneliti
- [ ] CV dan track record semua anggota tim
- [ ] Surat kerjasama dengan BPCB Jawa Timur
- [ ] Surat kerjasama dengan Balai Arkeologi (jika anggota dari sana)
- [ ] Ethical clearance (jika diperlukan)
- [ ] Bukti publikasi sebelumnya (Scopus/WoS)

### Tips Pengajuan DRPM
1. **TKT (Tingkat Kesiapan Teknologi):** Proposal ini TRL 1-3, cocok untuk skema Penelitian Dasar
2. **Kebaruan (novelty):** Tekankan bahwa ini PERTAMA KALINYA model prediktif komputasional diterapkan untuk arkeologi vulkanik di Indonesia
3. **Reviewer bias:** Reviewer DRPM mungkin tidak familiar dengan computational archaeology — jelaskan dengan bahasa sederhana
4. **Roadmap ke hulu:** Tunjukkan bagaimana ini bisa berlanjut ke Penelitian Terapan (TRL 4-6) dan kemudian PPUPT
5. **Kolaborasi lintas disiplin:** DRPM menghargai tim multi-disiplin — rekrut anggota dari arkeologi/geologi
6. **Luaran realistis:** 5 artikel Scopus dalam 3 tahun ambisius tapi achievable jika 2+ sudah dalam pipeline

### Timeline Pengajuan
- **Persiapan:** Q4 2026 (setelah 2+ paper diterima untuk track record)
- **Submission:** Januari-Februari 2027
- **Review:** Maret-April 2027
- **Pengumuman:** Mei 2027
- **Dana cair:** Juli 2027

---

*Skeleton ini mengikuti format DRPM Kemendikbudristek via sistem BIMA. Sesuaikan dengan panduan terbaru saat submission.*
