# HANDOFF - Mata Elang #11 Closeout (2026-03-30)

## Ringkasan

Mata Elang #11 sudah **selesai penuh**. Tiga eksperimen yang tertunda (E150-E152) sudah dibuat, dijalankan, dan didokumentasikan. Semua dokumen kanonik sekarang sinkron ke **153 eksperimen**, dan `python tools/check_doc_sync.py` sudah PASS.

---

## Yang Diselesaikan di Sesi Ini

### 1. E150 - Babad Tanah Jawi substrate NLP
- **Folder:** `experiments/E150_babad_substrate_nlp/`
- **Script:** `babad_substrate_analysis.py`
- **README:** `README.md`
- **Output utama:** `results/e150_results.json`, `classified_top_tokens.csv`, `native_content_terms.csv`, `chapter_token_summary.csv`, `domain_comparison.csv`
- **Hasil inti:**
  - 25 chapter HTML diparse
  - 25,743 token, 4,455 tipe unik
  - Top token stratum = **83.9% native/non-Sanskrit**, **6.6% Sanskrit**, **9.4% foreign**
  - Domain profile native = **GRAMMAR > OTHER > ACTION**, berbeda dari E130 yang **ACTION-heavy**
- **Makna:** blind spot DHARMA monoculture tertutup; corpus non-DHARMA juga menunjukkan backbone leksikal native yang kuat

### 2. E151 - Megalithic distribution vs volcanic zones
- **Folder:** `experiments/E151_megalithic_volcanic_zones/`
- **Script:** `megalithic_volcanic_analysis.py`
- **README:** `README.md`
- **Output utama:** `results/e151_results.json`, `case_studies.csv`
- **Hasil inti:**
  - 4 kasus sesuai WORKSTATE: Gunung Padang, Cipari, Bondowoso, Pasemah
  - Semua kasus berada dalam **35 km** dari gunung api aktif
  - Jarak rata-rata ke gunung api terdekat = **23.98 km**
  - **Stone survives 4/4**, **organic settlement visible 0/4**
- **Makna:** megalith tidak membantah VOLCARCH; ia memperjelas bahwa yang hilang adalah archaeology settlement-organik, bukan semua bukti pra-Hindu

### 3. E152 - Post-929 natural experiment
- **Folder:** `experiments/E152_post929_natural_experiment/`
- **Script:** `post929_analysis.py` (dipatch agar century parser menerima `11.0` dan `C11`)
- **README:** `README.md`
- **Output utama:** `results/e152_results.json`, `period_summary.csv`
- **Hasil inti:**
  - POST-929 inscriptions **12.71 km lebih jauh** dari gunung api (`p=0.000668`)
  - Center of gravity record bergeser **187 km ke timur**
  - Pre-Indic ratio naik dari **0.088 -> 0.231** (`p=0.000136`)
  - Mean word count naik dari **268.6 -> 648.1** (`p=0.000025`)
  - Status akhir eksperimen: **SUCCESS**
- **Makna:** blind spot mekanisme post-929 tertutup; pergeseran politik dan pergeseran taphonomik sama-sama terukur

---

## Sinkronisasi Dokumen

Dokumen berikut sudah diupdate ke state **153 eksperimen**:

- `README.md`
- `docs/EXPERIMENT_INDEX.md`
- `docs/JOURNAL.md`
- `docs/WORKSTATE.md`
- `docs/L1_CONSTITUTION.md`
- `docs/L2_STRATEGY.md`
- `docs/L3_EXECUTION.md`
- `docs/EVAL.md`
- `docs/DISSEMINATION_ROADMAP.md`
- `docs/SUSTAINABILITY_ROADMAP.md`

Tambahan penting:
- `docs/JOURNAL.md` sudah di-append entry closeout ME#11
- `docs/WORKSTATE.md` sekarang menyatakan **ME#11 fully closed**
- `docs/EXPERIMENT_INDEX.md` sudah memuat **E148-E152**

Verifikasi:
- `python tools/check_doc_sync.py` -> **SYNC OK: All docs agree on 153 experiments**

---

## Catatan Repo State

- Worktree **memang kotor** dan berisi perubahan lama + file baru yang belum di-commit. Jangan melakukan revert massal.
- `git status --short` menunjukkan file lama yang sudah dirty dari sesi sebelumnya (`.claude/settings.local.json`, `docs/drafts/manifesto.md`, `experiments/E136...`, `experiments/E137...`, dst). Perlakukan sebagai existing worktree state.
- Cek `py_compile` gagal karena `WinError 5` saat menulis `__pycache__`, tetapi **ketiga script E150-E152 berhasil dijalankan penuh**. Jadi masalahnya write-permission ke cache, bukan syntax/runtime logic.

---

## Prioritas Sesi Berikutnya

### Tidak perlu diulang
- Jangan mengulang E150-E152
- Jangan mengulang doc sync 153
- Jangan membuka kembali ME#11 kecuali ada permintaan baru

### Lanjutkan dari sini
1. **P11 target decision + submit prep**
   - Rekomendasi yang sudah tertulis di WORKSTATE tetap: **Indonesia (Cornell)**
   - Jika target disetujui, fokus sesi berikutnya: format conversion + final submission prep
2. **P17 ArchCalc prep**
   - Blocking item tetap sama: editorial rules dari ArchCalc
   - Setelah rules tersedia: adapt format + audit blind review
3. **JCAA APC / waiver check**
   - Status masih blocked, perlu cek email / waiver
4. **Repo go-public review**
   - README sudah 153, jadi pekerjaan berikutnya kalau dipilih adalah review file sensitif dan kebersihan repo

---

## File Kunci yang Perlu Dibaca Besok

Urutan baca paling efisien:

1. `docs/HANDOFF_20260330_ME11_CLOSEOUT.md` (dokumen ini)
2. `docs/WORKSTATE.md`
3. `docs/JOURNAL.md` (entry closeout paling atas)
4. `papers/P11_volcanic_informedness/`
5. `papers/P17_two_javas/`

---

## Status Akhir

- **ME#11: CLOSED**
- **Experiment count: 153**
- **Doc sync: PASS**
- **Next real decision point:** P11 target / P17 rules / JCAA waiver
