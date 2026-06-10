# Manual Pengguna — VOC-ArchNLP v1.0.0

*Sistem Penambangan Arsip Kolonial Belanda untuk Data Arkeologi Indonesia*

---

## 1. Pendahuluan

VOC-ArchNLP adalah perangkat lunak baris perintah (command-line tool) yang dirancang untuk peneliti yang ingin mengekstraksi informasi arkeologi dari teks arsip VOC (Vereenigde Oost-Indische Compagnie) yang telah didigitalkan oleh proyek GLOBALISE.

### Prasyarat

- Python 3.10 atau lebih baru
- Koneksi internet (untuk pengunduhan data)
- Minimal 4 GB RAM
- ±10 GB ruang penyimpanan untuk korpus penuh

---

## 2. Instalasi

### Opsi A: Langsung dari kode sumber (direkomendasikan untuk penelitian)

```bash
# Clone repositori VOLCARCH
git clone https://github.com/mukhlisamin/volcarch.git
cd volcarch

# Instal dependensi
pip install requests

# Verifikasi instalasi
python -m voc_archnlp --version
```

### Opsi B: Melalui pip (setelah publikasi ke PyPI)

```bash
pip install voc-archnlp
```

---

## 3. Alur Kerja

Pipeline VOC-ArchNLP terdiri dari 4 tahap berurutan:

```
[Arsip GLOBALISE] → [Prapemrosesan HTR] → [Normalisasi Ejaan] → [Ekstraksi Arkeologi]
      ↓                     ↓                      ↓                      ↓
 data/raw/           data/processed/         data/normalized/      results/mentions.csv
```

---

## 4. Perintah Dasar

### 4.1 Unduh Data (Tahap 0)

Mengunduh berkas transkripsi VOC dari GLOBALISE Dataverse:

```bash
# Unduh 500 berkas pertama
python -m voc_archnlp download --n 500 --output data/raw/globalise_voc/

# Unduh rentang nomor inventaris tertentu
python -m voc_archnlp download --range 1053-1200 --output data/raw/globalise_voc/
```

**Catatan:** Berkas yang sudah ada akan dilewati secara otomatis. Perintah ini dapat dihentikan dan dilanjutkan kapan saja.

---

### 4.2 Prapemrosesan (Tahap 1)

Membersihkan teks HTR dan menyegmentasi menjadi paragraf:

```bash
python -m voc_archnlp preprocess \
  --input data/raw/globalise_voc/ \
  --output data/processed/globalise_voc/
```

**Keluaran:** Untuk setiap berkas `1053.txt`, dihasilkan dua berkas:
- `clean_1053.txt` — baris bersih (artefak HTR dihapus)
- `paras_1053.txt` — satu paragraf per baris (siap NLP)

---

### 4.3 Normalisasi Ejaan (Tahap 2)

Mengonversi ejaan Belanda-kolonial ke Belanda modern:

```bash
python -m voc_archnlp normalize \
  --input data/processed/globalise_voc/ \
  --output data/normalized/globalise_voc/ \
  --level full
```

**Tingkat normalisasi:**
- `light` — hanya nama tempat + singkatan (aman, perubahan minimal)
- `medium` — tambah format tanggal + nama bulan
- `full` — semua aturan ejaan (direkomendasikan)

**Contoh transformasi:**
| Teks Asli | Hasil `full` |
|---|---|
| `Soerabaja` | `Surabaya` |
| `tjandi Singosari` | `candi Singhasari` |
| `den 15=e Januarij 1786` | `den 15e Januari 1786` |
| `M=r Willem Arnold Alting` | `Mijnheer Willem Arnold Alting` |

---

### 4.4 Ekstraksi Arkeologi (Tahap 3)

Mengekstraksi kalimat yang mengandung penyebutan arkeologi:

```bash
python -m voc_archnlp extract \
  --input data/normalized/globalise_voc/ \
  --output results/voc_mentions.csv \
  --json-out results/voc_mentions.json \
  --context 2
```

**Parameter:**
- `--context N` — jumlah kalimat sekitar yang disertakan sebagai konteks (default: 1)
- `--glob PATTERN` — pola berkas yang diproses (default: `paras_*.txt`)

**Tipe penyebutan yang dideteksi:**
| Tipe | Kata kunci contoh |
|---|---|
| `MONUMENT` | candi, tempel, arca, stupa, pagode, lingga, yoni |
| `GRAVE` | graf, begraven, kubur, makam, grafkuil |
| `RUIN` | ruïne, puing, vervallen, instorting, opgravingen |
| `ARTIFACT` | oudheden, penning, beeld, inscriptie, antiek |
| `INSCRIPTION` | inscriptie, prasasti, opschrift, gegraveerd |
| `DEPTH` | `N voet onder de grond`, `N el diep` |

---

### 4.5 Pipeline Lengkap (Semua Tahap Sekaligus)

```bash
python -m voc_archnlp run \
  --raw data/raw/globalise_voc/ \
  --output results/ \
  --norm-level full \
  --context 1
```

---

## 5. Memahami Keluaran

### Berkas `voc_archaeological_mentions.csv`

Setiap baris mewakili satu penyebutan arkeologi:

| Kolom | Keterangan |
|---|---|
| `source_file` | Nama berkas asal (misal: `norm_paras_1053.txt`) |
| `sentence_id` | Nomor kalimat dalam berkas (untuk reprodusibilitas) |
| `sentence_text` | Teks kalimat yang mengandung penyebutan |
| `mention_types` | Tipe entitas, dipisah `\|` (misal: `MONUMENT\|DEPTH`) |
| `keywords_found` | Kata kunci yang ditemukan (misal: `candi\|tempel`) |
| `depth_value_m` | Kedalaman dalam meter (kosong jika tidak ada pengukuran) |
| `context_before` | Kalimat sebelum (window konteks) |
| `context_after` | Kalimat sesudah (window konteks) |

### Contoh baris keluaran

```
source_file: norm_paras_1053.txt
sentence_id: 47
sentence_text: "Op 3 voet diepte werden eenige steenen gevonden welke tot een
               oud tempel schijnen te behooren nabij de dessa Trowulan."
mention_types: MONUMENT|DEPTH
keywords_found: tempel|stenen
depth_value_m: 0.914
context_before: "Den grond aldaar is sedert eeuwen door vulkaanuitworp bedekt."
context_after: "De bewoners verklaarden dit een oud heiligdom te zijn."
```

---

## 6. Reprodusibilitas dan Sitasi

### Cara mengutip program ini

```
Amien, M. (2026). VOC-ArchNLP: Dutch Colonial Archive Mining for Indonesian
Archaeological Data (Version 1.0.0) [Computer software].
Universitas Bhinneka Nusantara. Hak Cipta No. [EC-XXXXXXXX].
https://github.com/mukhlisamin/volcarch
```

### Cara mengutip data keluaran

```
Amien, M. (2026). VOC Archaeological Mentions Database v1.0 [Data set].
Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

---

## 7. Pertanyaan dan Dukungan

Hubungi: **amien@ubhinus.ac.id**
GitHub Issues: https://github.com/mukhlisamin/volcarch/issues

---

*VOC-ArchNLP v1.0.0 — Universitas Bhinneka Nusantara, 2026*
*Disiapkan untuk pengajuan Hak Cipta Program Komputer ke DJKI*
