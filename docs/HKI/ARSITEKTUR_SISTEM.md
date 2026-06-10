# Arsitektur Sistem — VOC-ArchNLP v1.0.0

---

## 1. Gambaran Umum

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VOC-ArchNLP v1.0.0                          │
│         Sistem Penambangan Arsip VOC untuk Data Arkeologi           │
├──────────┬──────────────┬───────────────┬──────────────────────────┤
│ Modul 1  │   Modul 2    │    Modul 3    │         Modul 4          │
│Downloader│ Preprocessor │  Normalizer   │        Extractor         │
│          │              │               │                          │
│GLOBALISE │HTR artifact  │Colonial Dutch │Archaeological mention    │
│Dataverse │removal +     │spelling →     │detection (6 entity       │
│API client│paragraph seg │modern Dutch   │types) + depth parsing    │
└──────────┴──────────────┴───────────────┴──────────────────────────┘
                                 ↓
                     ┌─────────────────┐
                     │   Pipeline.py   │
                     │  Orchestrator   │
                     └─────────────────┘
                                 ↓
                     ┌─────────────────┐
                     │    CLI (cli.py) │
                     │  Unified entry  │
                     └─────────────────┘
```

---

## 2. Komponen Rinci

### 2.1 Modul 1: Downloader (`download_globalise.py`)

**Fungsi:** Mengakses GLOBALISE Dataverse API dan mengunduh berkas transkripsi VOC.

**Algoritma:**
1. Memanggil Dataverse Persistent ID API (`hdl:10622/LVXSBW`) untuk mendapatkan daftar berkas.
2. Menyimpan indeks berkas ke cache JSON lokal (menghindari permintaan berulang).
3. Mengunduh berkas secara batch dengan pembatasan laju (0,5 detik antar-unduhan).
4. Mendukung resume: berkas yang sudah ada dilewati.
5. Retry otomatis dengan exponential backoff (3x).

**Input:** Parameter CLI (jumlah berkas / rentang nomor inventaris)
**Output:** Berkas `*.txt` di `data/raw/globalise_voc/`

---

### 2.2 Modul 2: Preprocessor (`preprocess_voc.py`)

**Fungsi:** Membersihkan artefak HTR dan menyegmentasi teks menjadi paragraf.

**Tahapan pemrosesan:**
1. `strip_metadata()` — menghapus komentar header GLOBALISE (`#+...`)
2. `rejoin_broken_words()` — menyambung kata terputus di akhir baris (karakter `¬` dan `„`)
3. `normalize_chars()` — normalisasi karakter Unicode khusus (ƒ→f, tanda kutip)
4. `filter_noise()` — membuang baris sangat pendek dan nomor halaman terisolasi
5. `segment_paragraphs()` — mengelompokkan baris menjadi paragraf (batas: baris kosong)

**Input:** Berkas `.txt` hasil unduhan GLOBALISE
**Output:** `clean_*.txt` (baris bersih) + `paras_*.txt` (satu paragraf per baris)

---

### 2.3 Modul 3: Normalizer (`normalize_colonial_dutch.py`)

**Fungsi:** Memetakan ejaan Belanda-kolonial (abad ke-17 hingga ke-18) ke Belanda modern.

**Tiga lapisan normalisasi:**

| Lapisan | Metode | Contoh |
|---|---|---|
| Kamus toponim | Penggantian string eksak | `Soerabaja` → `Surabaya` |
| Ekspansi singkatan | Regex | `M=r` → `Mijnheer` |
| Aturan ortografis | Regex berurutan | `tjandi` → `candi` |

**Kelas utama:** `ColonialDutchNormalizer` dengan metode `normalize(text, level)`.

---

### 2.4 Modul 4: Extractor (`extractor.py`) — **Komponen Baru**

**Fungsi:** Mengekstraksi kalimat yang mengandung penyebutan arkeologi dan menandai tipe entitas.

**Lexikon kata kunci (multi-bahasa):**
- Belanda: `oudheden, oudheidkundig, tempel, graf, ruïne, inscriptie, penning, begraven, gedelfd, ontgraven`
- Pinjaman Jawa/Melayu: `candi, prasasti, arca, yoni, lingga, stupa, kubur, makam`
- Pola kedalaman: `N voet/el/palm/duim onder de grond`

**Enam tipe entitas:**
- `MONUMENT` — bangunan keagamaan (candi, tempel, stupa, arca)
- `GRAVE` — konteks penguburan (graf, begraven, kubur)
- `RUIN` — bangunan runtuh (ruïne, puing, vervallen)
- `ARTIFACT` — benda portabel (penning, beeld, oudheden)
- `INSCRIPTION` — benda bertulisan (inscriptie, prasasti, opschrift)
- `DEPTH` — pengukuran kedalaman (dengan konversi satuan ke meter)

**Konversi satuan kedalaman (VOC → meter):**
- 1 voet = 0,3048 m (kaki Rhineland)
- 1 el = 0,6858 m (el Rhineland)
- 1 palm = 0,10 m
- 1 duim = 0,0254 m

**Algoritma ekstraksi:**
1. Memisahkan teks menjadi kalimat (regex berdasarkan tanda baca + huruf kapital)
2. Mencocokkan setiap kalimat dengan lexikon per tipe
3. Jika cocok: merekam kalimat + window konteks + nilai kedalaman (jika ada)
4. Keluaran: daftar dict dengan 8 kolom (CSV/JSON)

---

### 2.5 Pipeline Orchestrator (`pipeline.py`)

**Fungsi:** Mengintegrasikan Modul 2–4 menjadi pipeline 4-tahap satu perintah.

**Aliran data:**
```
raw/*.txt
  → [Modul 2] → processed/clean_*.txt + processed/paras_*.txt
  → [Modul 3] → normalized/norm_paras_*.txt
  → [Modul 4] → mentions/voc_archaeological_mentions.csv
                 mentions/voc_archaeological_mentions.json
                 pipeline_summary.json
```

---

### 2.6 CLI Terpadu (`cli.py`)

**Fungsi:** Antarmuka baris perintah tunggal untuk semua operasi.

**Subperintah:**
```
voc_archnlp download   -- Modul 1
voc_archnlp preprocess -- Modul 2
voc_archnlp normalize  -- Modul 3
voc_archnlp extract    -- Modul 4
voc_archnlp run        -- Pipeline (Modul 2+3+4)
```

---

## 3. Diagram Aliran Data

```
GLOBALISE Dataverse (https://datasets.iisg.amsterdam)
       │ HTTP API
       ▼
data/raw/globalise_voc/
  ├── 1053.txt   (HTR transkrip, ~12K baris/berkas)
  ├── 1054.txt
  └── ...
       │ Modul 2 (preprocess)
       ▼
data/processed/globalise_voc/
  ├── clean_1053.txt  (baris bersih)
  ├── paras_1053.txt  (paragraf, 1/baris)
  └── preprocessing_stats.json
       │ Modul 3 (normalize)
       ▼
data/normalized/globalise_voc/
  ├── norm_paras_1053.txt
  └── ...
       │ Modul 4 (extract)
       ▼
results/mentions/
  ├── voc_archaeological_mentions.csv   ← KELUARAN UTAMA
  ├── voc_archaeological_mentions.json
  └── pipeline_summary.json
```

---

## 4. Keamanan dan Privasi

- **Tidak ada data pribadi:** Semua teks yang diproses adalah arsip publik abad ke-17/18 (CC0).
- **Tidak ada koneksi keluar selain pengunduhan:** Program tidak mengirimkan data ke server eksternal.
- **Reprodusibel:** Nomor inventaris GLOBALISE memastikan jejak audit penuh untuk setiap penyebutan.

---

## 5. Keterbatasan

1. **Kualitas OCR/HTR:** Beberapa berkas GLOBALISE memiliki tingkat kesalahan HTR tinggi (terutama tangan tulisan abad ke-17); normalisasi tidak dapat memperbaiki semua kesalahan transkripsi.
2. **Ambiguitas kata kunci:** Kata seperti `tempel` dalam konteks non-arkeologi (misalnya `tempel des hemels` = kuil surga [metaforis]) dapat menghasilkan positif palsu.
3. **Cakupan gazetteer:** Normalisasi nama tempat terbatas pada ~60 entri; nama desa-desa kecil tidak selalu terpetakan.
4. **Belum termasuk NER berbasis ML:** Versi 1.0 menggunakan pencocokan kata kunci; model NER berbasis transformer (ArcheoBERTje yang telah di-fine-tune) direncanakan untuk v2.0.

---

*VOC-ArchNLP v1.0.0 — Universitas Bhinneka Nusantara, 2026*
*Disiapkan untuk pengajuan Hak Cipta Program Komputer ke DJKI*
