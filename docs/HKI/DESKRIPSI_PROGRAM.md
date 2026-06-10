# Deskripsi Program Komputer — VOC-ArchNLP v1.0.0

---

## Identitas Karya

| Informasi | Detail |
|---|---|
| **Nama Program** | VOC-ArchNLP: Sistem Penambangan Arsip Kolonial Belanda untuk Data Arkeologi Indonesia |
| **Singkatan** | VOC-ArchNLP |
| **Versi** | 1.0.0 |
| **Tahun Pembuatan** | 2026 |
| **Pencipta** | Mukhlis Amien, S.Kom., M.Cs. |
| **ORCID** | 0000-0002-1848-167X |
| **Email** | amien@ubhinus.ac.id |
| **Afiliasi** | Universitas Bhinneka Nusantara (Ubhinus), Malang, Jawa Timur |
| **Lisensi** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Bahasa Pemrograman** | Python 3.10+ |
| **Platform** | Windows, Linux, macOS |
| **Proyek Induk** | VOLCARCH (Volcanic Taphonomic Bias in Indonesian Archaeological Records) |

---

## Deskripsi Singkat

VOC-ArchNLP adalah perangkat lunak pengolahan bahasa alami (Natural Language Processing/NLP) untuk penambangan data arkeologis dari korpus teks arsip kolonial Belanda abad ke-17 hingga ke-18. Program ini memproses transkrip digital arsip Vereenigde Oost-Indische Compagnie (VOC, 1602–1799) yang tersedia secara terbuka melalui repositori GLOBALISE Dataverse (CC0), dan mengekstraksi penyebutan situs, artefak, kedalaman penguburan, dan fitur arkeologi lainnya ke dalam basis data spasial terstruktur.

---

## Latar Belakang dan Urgensi

Rekaman arkeologi Indonesia menghadapi bias struktural yang diakibatkan oleh proses tafonomik vulkanik: situs-situs pra-Hindu, terutama di interior Jawa, tertimbun material vulkanik hingga kedalaman 3–10 meter, sehingga tidak terekam dalam survei permukaan modern. Arsip VOC, yang mencakup lebih dari 5 juta halaman catatan administratif komprehensif dari seluruh Nusantara, memuat ribuan laporan pengamatan langsung oleh pegawai VOC terhadap bangunan kuno, arca, prasasti, dan temuan bawah tanah—sebagian besar belum pernah dianalisis secara sistematis dengan metode komputasional.

VOC-ArchNLP mengisi celah ini dengan menyediakan pipeline pemrosesan teks otomatis yang dapat dijalankan pada ribuan berkas sekaligus, sehingga menghasilkan basis data penyebutan arkeologi yang komprehensif dan dapat diverifikasi secara independen.

---

## Kebaruan (Orisinalitas)

Program ini merupakan **karya asli** yang menggabungkan empat komponen inovatif:

1. **Pengunduh korpus khusus VOC** — mengakses Dataverse API GLOBALISE dengan indeks berkas ter-cache, mendukung unduhan batch ribuan berkas dengan pembatasan laju server otomatis.

2. **Prapemroses teks HTR (Handwritten Text Recognition)** — menangani artefak khusus output HTR Loghi (karakter pemecah baris `¬`, `„`), menyambung kembali kata yang terputus antarbarisian, dan menghasilkan unit paragraf siap-NLP.

3. **Normalisasi ejaan Belanda-kolonial** — memetakan ejaan pra-1947 (Soerabaja→Surabaya, tjandi→candi, M=r→Mijnheer) menggunakan tiga lapisan: kamus toponim, ekspansi singkatan, dan aturan ortografis sistematis. Tidak ada perangkat lunak yang secara spesifik menangani ejaan VOC abad ke-17.

4. **Ekstraktor penyebutan arkeologi berbasis pola** — mengklasifikasikan kalimat ke dalam enam tipe entitas (MONUMENT, GRAVE, RUIN, ARTIFACT, INSCRIPTION, DEPTH), mengekstraksi nilai kedalaman dalam satuan meter dari satuan VOC-era (voet, el, palm, duim), dan menghasilkan keluaran CSV/JSON berprovenans (nama berkas, ID kalimat, konteks sekitar).

Kombinasi keempat komponen ini **tidak ada padanannya** pada perangkat lunak yang tersedia secara publik per Januari 2026. Proyek GLOBALISE (Universiteit van Amsterdam/VU Amsterdam) menyediakan data dan infrastruktur, tetapi tidak menyediakan pipeline ekstraksi arkeologi.

---

## Fungsi Program

### Fungsi Utama

1. Mengunduh berkas transkripsi VOC dari GLOBALISE Dataverse (Arsip Internasional Sejarah Sosial, Amsterdam).
2. Membersihkan dan menyegmentasi teks HTR menjadi paragraf yang dapat diproses NLP.
3. Menormalisasi ortografi Belanda-kolonial ke Belanda modern untuk meningkatkan kompatibilitas model bahasa.
4. Mengekstraksi kalimat yang mengandung penyebutan arkeologi dengan penandaan tipe entitas.
5. Menghasilkan basis data CSV/JSON penyebutan arkeologi berprovenans untuk analisis spasial lanjutan.
6. Menyediakan antarmuka baris perintah (CLI) terpadu untuk seluruh tahapan.

### Keluaran

- `voc_archaeological_mentions.csv` — basis data tabular; setiap baris = 1 penyebutan arkeologi dengan kolom: `source_file`, `sentence_id`, `sentence_text`, `mention_types`, `keywords_found`, `depth_value_m`, `context_before`, `context_after`.
- `voc_archaeological_mentions.json` — representasi JSON untuk integrasi sistem lain.
- `pipeline_summary.json` — statistik agregat per tahapan (jumlah berkas, paragraf, kata, penyebutan per tipe).

### Pengguna Target

- Peneliti arkeologi dan sejarah Indonesia yang tidak memiliki latar belakang komputasional.
- Peneliti humanistik digital yang mempelajari arsip kolonial VOC.
- Mahasiswa dan dosen perguruan tinggi dalam mata kuliah Digital Humanities dan Text Mining.
- Pengembang sistem informasi kebudayaan dan warisan budaya.

---

## Spesifikasi Teknis

| Komponen | Keterangan |
|---|---|
| Bahasa | Python 3.10+ |
| Dependensi inti | `requests` ≥ 2.28 (pengunduhan) |
| Dependensi opsional | `spaCy` ≥ 3.5, `transformers` ≥ 4.35 (untuk Fase 2 NER) |
| Sistem operasi | Windows 10/11, Ubuntu 20.04+, macOS 12+ |
| RAM minimum | 4 GB (pipeline dasar) |
| Penyimpanan | ±10 GB untuk 6.893 berkas korpus penuh |
| Antarmuka | Baris perintah (CLI) + modul Python yang dapat diimpor |

---

## Contoh Penggunaan

```bash
# Unduh 500 berkas dari GLOBALISE Dataverse
python -m voc_archnlp download --n 500 --output data/raw/globalise_voc/

# Jalankan pipeline lengkap (4 tahap sekaligus)
python -m voc_archnlp run --raw data/raw/globalise_voc/ --output results/

# Hanya ekstraksi (jika teks sudah diproses sebelumnya)
python -m voc_archnlp extract --input data/normalized/ --output results/mentions.csv
```

---

## Hubungan dengan Penelitian Induk

Program ini adalah produk turunan dari proyek penelitian VOLCARCH yang didanai secara mandiri. Komponen-komponennya dikembangkan melalui 7 eksperimen penelitian:

| Eksperimen | Kontribusi ke VOC-ArchNLP |
|---|---|
| E091 | Validasi konsep ekstraksi OV 1925–1949 (22.162 penyebutan) |
| E141 | Ekstraksi Delpher 1854–1942 (1.768 artikel, 165 tergeokode) |
| E197 | Validasi kedalaman kolonial 1,0–4,0 m |
| E206 | Evaluasi ArcheoBERTje: gap 60% tipe entitas pada teks VOC |
| E207 | Pilot GLOBALISE (50 berkas, 6,26 juta kata) |
| E211 | Desain pipeline E211 = fondasi VOC-ArchNLP v1.0 |

---

## Pernyataan Keaslian

Program ini merupakan karya asli yang dibuat oleh Mukhlis Amien pada tahun 2026. Tidak ada bagian dari program ini yang merupakan salinan dari program komputer lain tanpa izin. Data pelatihan (teks VOC) berasal dari GLOBALISE Dataverse yang berlisensi CC0 (domain publik). Model bahasa yang dirujuk (ArcheoBERTje, XLM-R) adalah komponen terpisah yang tidak disertakan dalam paket ini dan tunduk pada lisensi masing-masing.

---

*Dokumen ini disiapkan untuk pengajuan Hak Cipta (Program Komputer) ke Direktorat Jenderal Kekayaan Intelektual (DJKI), Kementerian Hukum dan HAM Republik Indonesia.*
*Tanggal pembuatan: 23 April 2026*
