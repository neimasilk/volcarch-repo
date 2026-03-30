# Peradaban yang Terkubur
## Pendekatan Sains Data untuk Misteri Arkeologi Indonesia

**Penulis:** Mukhlis Amien
**Target penerbit:** UB Press / UGM Press / Deepublish (ISBN, diakui SISTER)
**Bahasa:** Indonesia (bilingual key terms)
**Estimasi halaman:** 200-250
**Status:** OUTLINE v0.1

---

## Mengapa Buku Ini Perlu Ditulis

Indonesia memiliki sejarah hunian manusia selama lebih dari 1 juta tahun (Homo erectus di Sangiran), namun "sejarah resmi" baru dimulai dari abad ke-4 Masehi — ketika prasasti pertama muncul. Buku ini menjawab pertanyaan: ke mana 999.600 tahun sisanya?

Berdasarkan 147 eksperimen komputasional, 15 sumber kuno independen, dan analisis 268 prasasti, buku ini menunjukkan bahwa sejarah Nusantara tidak "dimulai terlambat" — sejarahnya terkubur di bawah 5-10 meter sedimen vulkanik.

**Nilai untuk dosen/mahasiswa:**
- Studi kasus nyata data science + arkeologi
- Metode: Python, GIS, NLP, ML, Monte Carlo
- Dataset terbuka (GitHub)
- Bisa jadi buku ajar "Digital Humanities" / "Computational Social Science"

---

## DAFTAR ISI

### Bagian I: Pertanyaan

**Bab 1. Patung yang Ditelan Bumi (15 hal)**
- Dwarapala Singosari: foto kolonial vs modern
- 185 cm terkubur dalam 535 tahun = 3.5 mm/tahun
- Kalau patung batu saja tenggelam, bagaimana rumah kayu?
- *Dari observasi sederhana ke pertanyaan besar*

**Bab 2. Mengapa Sejarah Indonesia "Dimulai" dari 400 M? (20 hal)**
- Kutai vs Jawa: yang tertua = yang paling terlihat
- 15 sumber kuno (Yunani, Romawi, India, Cina, Arab) sudah tahu Nusantara ribuan tahun sebelumnya
- Gap 3.220x: berapa situs yang seharusnya ada vs yang ditemukan
- *Bukan ketidakhadiran — tapi ketidaktampakan*

**Bab 3. Enam Lapisan Kegelapan (15 hal)**
- L1: Penguburan vulkanik (4.4 mm/tahun)
- L2: Penenggelaman pesisir (2 juta km2 Paparan Sunda)
- L3: Bias historiografis (perspektif kolonial)
- L4: Penimpaan kosmologis (Sanskrit menggantikan kosakata asli)
- L5: Filter genre (format prasasti menyeleksi apa yang dicatat)
- L6: Periodisitas historiografis (gelombang Indianisasi)
- *Cascade multiplikatif: 0.058% visibilitas*

### Bagian II: Bukti

**Bab 4. Kalibrasi: Berapa Cepat Tanah Naik? (20 hal)**
- 4 titik kalibrasi: Dwarapala, Sambisari, Kedulan, Kimpulan
- 51 pasangan erupsi-situs (validasi independen)
- 25 data kedalaman dari laporan kolonial Belanda (E128)
- Median 2.50m — dua dataset independen, hasil identik
- *Data science: regresi, Monte Carlo, bootstrap*

**Bab 5. Peta yang Berbicara: Analisis Spasial 666 Situs (20 hal)**
- Dataset dari OpenStreetMap + Wikidata + Wikipedia
- 73% situs adalah candi — bias survei masif
- Situs mengelompok di dekat gunung api (bukan karena preferensi, tapi karena survei)
- Model prediksi: di mana harus menggali?
- *Data science: GIS, XGBoost, anomaly detection*

**Bab 6. Apa yang Dikatakan Prasasti (20 hal)**
- 268 prasasti DHARMA: 63.4% menyebut material organik
- Paradoks C8: era paling banyak prasasti = paling sedikit konten pra-Indic
- Hyang (konsep ketuhanan pra-Hindu) justru naik dari 1.8% ke 72.7%
- Indianisasi adalah gelombang, bukan transformasi permanen
- *Data science: NLP, topic modeling, diachronic analysis*

**Bab 7. 438 Kata yang Hilang (15 hal)**
- Substrat linguistik pra-Indic terdeteksi oleh machine learning
- Kata-kata aksi (memasak, berburu, memotong) = vocabulary kehidupan sehari-hari
- Tanda fonologis: lebih banyak glottal stop, consonant cluster lebih kompleks
- Peradaban yang tidak terlihat tapi terdengar dalam bahasanya
- *Data science: ML classification, feature importance, SHAP*

**Bab 8. Dunia Sudah Tahu (15 hal)**
- Ptolemy memetakan Jawa pada 150 M
- Ramayana menyebut Yavadvipa pada 300 SM
- Fa Xian mengunjungi Jawa 414 M: Hindu-Buddha masih marginal
- Duta besar ke istana Han, 132 M
- Keramik Rouletted Ware di Buni, 200 SM — bukti fisik perdagangan
- *Bukan absensi peradaban — tapi absensi bukti lokal*

### Bagian III: Prediksi

**Bab 9. Di Mana Harus Menggali? (20 hal)**
- 20 koordinat GPS target (E080)
- Anomaly detection: 195.382 sel "mirip situs" (E097, 65% overlap)
- Strategi fieldwork 3-fase: LiDAR → GPR/ERT → coring
- Budget: $6.000 (pilot) sampai $100.000 (definitif)
- P(menemukan sesuatu dengan 20 GPR) = 93%
- *Data science: spatial modeling, cost-benefit analysis*

**Bab 10. Jawa dan Dunia: Perspektif Komparatif (15 hal)**
- Jawa vs Filipina: 4.6x lebih sedikit gunung api = sedikit lebih baik
- Jawa vs Jepang: gunung api mirip, 5.000x lebih banyak situs (rescue archaeology!)
- Jawa = satu-satunya region vulkanik dengan 1M+ tahun hunian + zero pre-400M sites
- Densitas prasasti: non-vulkanik 30x lebih tinggi dari vulkanik
- *Fenomena global, bukan anomali lokal*

### Bagian IV: Implikasi

**Bab 11. Peradaban Bambu (15 hal)**
- 60% budaya material dalam prasasti = organik (tidak survive)
- Rumah bambu, naskah lontar, pasar kain — semua hilang
- Jawa arkeologis = Jawa elite. Jawa sehari-hari = invisible.
- Inca tanpa tulisan menjalankan 12 juta orang. Nusantara pra-Hindu serupa.

**Bab 12. Apa yang Harus Dilakukan (15 hal)**
- Rescue archaeology: Indonesia butuh seperti Jepang
- Citizen science: jaringan penambang pasir
- LiDAR: preseden Amazon 2024
- Phytolith: bukti pertanian bisa bertahan 90.000 tahun di tephra
- *Dari riset ke aksi: roadmap untuk arkeologi Indonesia*

**Epilog. 400 Masehi Bukan Awal (5 hal)**
- Kembali ke Dwarapala: 200 tahun kita melihat, baru sekarang kita mendengar
- "Kerajaan tertua Indonesia bukan kerajaan yang paling tua — kerajaan yang paling terlihat"

### Lampiran

- A. Daftar 147 eksperimen VOLCARCH
- B. Tutorial Python: analisis sedimentasi (bab 4)
- C. Tutorial GIS: peta prediksi (bab 5)
- D. Tutorial NLP: analisis prasasti (bab 6)
- E. Dataset terbuka (link GitHub + Zenodo)
- F. Glosarium istilah arkeologi + data science

---

## Estimasi Penulisan

| Bagian | Halaman | Waktu |
|--------|:---:|---|
| Bagian I (Pertanyaan) | 50 | 2 bulan |
| Bagian II (Bukti) | 90 | 3 bulan |
| Bagian III (Prediksi) | 35 | 1 bulan |
| Bagian IV (Implikasi) | 30 | 1 bulan |
| Lampiran + editing | 30 | 1 bulan |
| **Total** | **~235** | **~8 bulan** |

## Nilai Tridarma

| Tridarma | Kontribusi | KUM |
|----------|-----------|-----|
| Penelitian | Rangkuman riset VOLCARCH | Sudah tercover oleh paper |
| Pengajaran | Buku ajar (ISBN) | ~20-40 angka kredit |
| Pengabdian | Tutorial Python/GIS (lampiran) | Bisa diklaim terpisah |

---

*"Buku ini bukan tentang gunung api. Buku ini tentang apa yang terkubur di bawahnya — dan bagaimana data science membantunya ditemukan."*
