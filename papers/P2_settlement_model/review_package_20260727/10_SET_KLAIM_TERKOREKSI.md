# Dokumen 10 — SET KLAIM TERKOREKSI untuk naskah v0.2

**Tanggal:** 2026-08-03 · **Sifat:** penerapan K1–K3 (dokumen 09) + hasil SIG G1 re-derivasi buta
**Menggantikan:** dokumen `08_HANDOFF_BABAK2.md` **§3** ("Set klaim FINAL"). §3 itu **tidak final** —
tiga klaimnya tidak didukung datanya sendiri.
**Dasar angka:** `revision_ammo/SIG_G1_VERIFICATION_20260803.md` (57 pemeriksaan, dihitung ulang dari
file hasil per-run; `*_outcome.json` sengaja **tidak** dibaca) · skrip:
`revision_ammo/verify_headline_numbers.py`

> **Aturan pakai dokumen ini.** Setiap kalimat berangka di naskah v0.2, abstrak, surat balasan
> reviewer, dan email ke editor harus punya baris di §2 di bawah. Kalau sebuah angka tidak ada di
> sini, angka itu belum lolos G1 dan **tidak boleh dipakai**.

---

## 1. Enam koreksi, dan dari mana asalnya

K1–K3 datang dari review dokumen 09 (27 Jul). **K5–K7 baru hari ini** — ditemukan justru oleh proses
re-derivasi buta yang seharusnya sekadar mengonfirmasi K1–K3. Ketiganya ada di **dokumen 08 §3**,
yaitu daftar yang menyebut dirinya "set klaim FINAL".

| | Klaim yang tertulis | Yang sebenarnya terukur | Status |
|---|---|---|---|
| **K1** | "aturan seleksi naskah memilih desain terburuk 60/60" | Benar hanya kalau dial diperpanjang sampai `hard_frac=1.0`. Di grid yang naskah benar-benar pakai (`{0.0, 0.15, 0.30}`): biaya median **+0.0000** (sintetik), **+0.0044** (nyata). Yang bertahan: kriterianya **tak punya optimum interior**. | wajib |
| **K2** | "angka laporan bergerak ~10× lebih cepat daripada kebenaran" | **2.0×** sintetik (endpoint 2.01×; slope OLS per-run 2.12×). Di data nyata keduanya bergerak **berlawanan arah**. | wajib |
| **K3** | "evaluasi di background sendiri **selalu** terinflasi" | **343/360 = 95.3%**, minimum **−0.031**, median **+0.187**. | wajib |
| **K4** | null TGB dilaporkan sebagai kejutan | `road_dist` **bukan fitur model** → bias survei tak bisa direpresentasikan di ruang fitur → TGB tak punya apa pun untuk dibatalkan. Null yang **terprediksi**, bukan bug. | konstruktif |
| **K5** | "aturan naskah memilih **konfigurasi terburuk** di 100% kasus" (dok. 08 §3 klaim 2) | **SALAH.** Aturan memilih hybrid(1.0) 60/60; yang terburuk-menurut-kebenaran adalah **hybrid(0.0)** di 50/60 kasus (hybrid(0.3) 8, hybrid(0.7) 2). hybrid(1.0) **tidak pernah** jadi yang terburuk. Yang benar: pilihan itu berbiaya **+0.194 AUC_true terhadap desain terbaik**. | **baru** |
| **K6** | "AUC laporan naik **monoton** sampai ujung dial di kedua dunia" (dok. 09 §2) | Benar di sintetik (0.7367→0.7820→0.8412→0.8904). Di data nyata **ada satu penurunan** 0.0→0.1 (−0.0071), lalu naik sampai 0.8435. Rumusan yang tahan: **maksimumnya selalu di tepi grid, di mana pun grid itu dihentikan.** | **baru** |
| **K7** | "densitas situs inti robust 2–5,6× fringe" (dok. 08 §3 klaim 4) | **1.93×** (randomforest), 4.34× (xgboost), 5.62× (maxent). Batas bawahnya 1.93, bukan 2. Sebut **"1,9–5,6×"** atau tiga angkanya. | **baru** |

**K5, K6, dan K7 adalah kelas kesalahan yang persis sama dengan yang sedang kita koreksi di naskah
asli:** kuantor/superlatif yang lebih kuat daripada datanya. Bahwa ketiganya lolos sampai ke dokumen
yang berlabel "FINAL", dan baru tertangkap oleh re-derivasi buta, adalah argumen paling langsung untuk
mempertahankan G1 sebagai gerbang — dan layak disebut satu kalimat di *Response to Reviewers*.

---

## 2. Set klaim untuk v0.2 — hanya yang ada di sini yang boleh masuk naskah

Kolom **G1** = sudah diverifikasi ulang dari file per-run hari ini.

### K-A · Temuan inti: angka yang dilaporkan adalah artefak desain evaluasi

| # | Klaim (rumusan yang boleh dipakai) | Angka | Sumber | G1 |
|---|---|---|---|---|
| A1 | Model yang dinilai pada background yang ia pilih sendiri memperoleh AUC yang **secara sistematis** lebih tinggi daripada kebenarannya. | 343/360 run (95,3%); median +0.187; minimum −0.031 | `e222_runs.csv` | ✅ |
| A2 | Pada data nyata, desain hybrid memperoleh inflasi di setiap sel seed × algoritma. | 60/60 positif; rata-rata +0.037; rentang +0.005…+0.084 | `e218_stageA_raw.csv` | ✅ |
| A3 | Ditahan pada background evaluasi bersama, keunggulan hybrid **hilang**: ia hanya menang ketika dinilai di background-nya sendiri. | menang 3/3 algoritma pada eval-background hybrid; **0/3** pada uniform, tgb, dan stratified | `e218_stageA_raw.csv` | ✅ |
| A4 | Perancangan ulang background menyumbang ≈0 pada background bersama, sementara **satu** fitur hidrologi menyumbang jauh lebih besar. | redesain **−0.0142**; fitur sungai **+0.0424** (12/12 positif) | `e217b_raw_results.csv` | ✅ |
| A5 | Tangga AUC +0.092 yang terbit **ditolak**, bukan sekadar tak terbukti. | 12/12 CI sel mengecualikannya; bootstrap blok 3/3 (batas atas +0.008 / +0.025 / +0.026) | `e223a_*`, `e223b_*` | ✅ |
| A6 | Bukan artefak regularisasi MaxEnt. | β 0.5–4.0: −0.0198…−0.0217, 1/10 run positif | `e223c_beta_summary.csv` | ✅ |

**Catatan A4:** −0.0142 berlaku untuk himpunan fitur penuh (`terrain_river`), yaitu model naskah. Pada
`terrain` saja angkanya **+0.0054**. Keduanya ≈0 dibandingkan efek fitur; sebutkan himpunan fiturnya
supaya reviewer tidak menemukan selisih itu sendiri.

### K-B · Kriteria seleksi: yang bertahan setelah K1 dan K5

| # | Klaim | Angka | Sumber | G1 |
|---|---|---|---|---|
| B1 | Kriteria AUC-di-background-sendiri **tidak punya optimum interior**: maksimumnya selalu jatuh di tepi grid yang disapu, jadi kriteria itu tidak pernah menyuruh berhenti. | sintetik 0.7367→0.8904 (monoton); nyata 0.7208→0.8435, monoton dari 0.1 ke atas, satu penurunan −0.0071 di 0.0→0.1 | `e222_runs.csv`, `e218b_hardfrac_sweep.csv` | ✅ |
| B2 | **Naskah kami berhenti di hard_frac 0.30 karena gridnya berhenti di situ, bukan karena kriterianya menyuruh berhenti.** Biaya di titik operasi itu kecil. | biaya **+0.0044** (nyata), median **+0.0000** (sintetik; rata-rata +0.0012, maksimum +0.0088) | re-seleksi | ✅ |
| B3 | Kalau dial diperpanjang, biayanya besar — dan itulah bahayanya: kriteria ini tidak menyediakan pengaman apa pun. | biaya **+0.0973** (nyata), **+0.1937** median (sintetik), 100% positif | re-seleksi | ✅ |
| B4 | Di grid naskah, kriteria itu tetap **tidak** menunjuk desain terbaik-menurut-kebenaran, tetapi selisihnya praktis nol. | pilihan ≠ terbaik di 29/60; kriteria memilih random 50 / tgb 10, kebenaran memilih random 33 / tgb 27 | re-seleksi | ✅ |
| B5 | Di dial penuh, pilihannya berbiaya +0.194 **terhadap desain terbaik** — dan **bukan** desain terburuk. | terburuk = hybrid(0.0) 50/60, hybrid(0.3) 8, hybrid(0.7) 2; hybrid(1.0) 0/60 | re-seleksi | ✅ |

> ⛔ **Dilarang di naskah:** "memilih yang terburuk", "selalu", "tidak pernah", "60/60 salah",
> "~10× lebih cepat". Semua sudah diuji dan tidak didukung. (Aturan jalur 01 no. 4: tidak ada kuantor
> absolut tanpa pecahannya di sebelahnya.)

### K-C · Divergensi angka-laporan vs kebenaran

| # | Klaim | Angka | Sumber | G1 |
|---|---|---|---|---|
| C1 | Di dunia sintetik, angka laporan bergerak sekitar **dua kali** lebih cepat daripada kebenaran sepanjang dial. | laporan +0.1538 vs kebenaran +0.0764 → **2.01×**; slope OLS per-run 2.12×; median rasio per-run 2.00× | `e222_runs.csv` | ✅ |
| C2 | Di data nyata keduanya bergerak **berlawanan arah** — pernyataan yang lebih kuat dan tak butuh pembesaran. | laporan **+0.1227**, background-bersama **−0.0973** | `e218b_hardfrac_sweep.csv` | ✅ |
| C3 | Prediksi pra-registrasi P1 (inflasi naik dengan hard_frac di semua rejim) **GAGAL** dan dilaporkan gagal. | Spearman pooled **0.4395** < ambang 0.5 | `e222_runs.csv` | ✅ |

**Wajib:** sebutkan estimatornya. "2.12×" hanya benar untuk slope OLS per-run; definisi endpoint
memberi 2.01×. Tulis **"sekitar dua kali"** dan lampirkan definisinya di catatan kaki.

### K-D · Peta prioritas dan protokol

| # | Klaim | Angka | Sumber | G1 |
|---|---|---|---|---|
| D1 | Peta prioritas tidak stabil terhadap seed saja. | turnover 1−J **28,1%–47,4%** antar pasangan seed | `e221_turnover_pairs.csv` | ✅ |
| D2 | Obatnya murah dan bisa dijadikan protokol: ensemble seed. | k\* = 2–5 (J≥0.85), **4–7** (J≥0.90), 7–9 (J≥0.95) | `e223d_kstar_thresholds.csv` | ✅ |
| D3 | Produk lapangan: inti *robust* punya densitas situs lebih tinggi daripada *fringe* pada ketiga algoritma. | **1,9× / 4,3× / 5,6×** (rf / xgb / maxent) — bukan "2–5,6×" | `e221_priority_sets.csv` | ✅ |

### K-E · Null TGB — ⛔ DIAGNOSIS K4 DIUJI DAN GAGAL (E224, 2026-08-03)

| # | Klaim | Angka | Sumber | G1 |
|---|---|---|---|---|
| E1 | TGB tidak menolong di simulasi kami. **Nullnya nyata dan tetap dilaporkan.** | TGB − random pada map Jaccard: **−0.010**, 46,7% positif | `e222_runs.csv` | ✅ |
| ~~E2~~ | ~~"Alasannya struktural: `road_dist` bukan fitur, jadi TGB tak punya apa pun untuk dibatalkan"~~ | **DIUJI DI E224 DAN TIDAK DIDUKUNG.** Dengan `road_dist` dimasukkan ke fitur, TGB − random = **−0.0217** (30% positif); tanpa itu **−0.0254** (30% positif). Tidak berubah. | `e224_outcome.json` | ✅ |
| E2′ | Rumusan pengganti yang boleh dipakai: **null-nya belum terjelaskan.** Kami mengajukan satu penjelasan, mengujinya, dan penjelasan itu gagal. | lihat E224 README | pra-registrasi | ✅ |
| E2″ | Keterbatasan uji itu sendiri, wajib disebut: `road_dist` **tidak ortogonal** terhadap fitur yang sudah ada (`river_dist` **+0.49**, elevation +0.31), jadi separuh sinyalnya sudah terjangkau sebelum manipulasi. Uji yang bersih butuh permukaan bias yang ortogonal by construction — **future work**. | korelasi di `e224_outcome.json` | ✅ |
| E3 | TGB **netral, bukan merugikan**, di rejim bias regional. | World C −0.0010 (56,7% positif), World D **+0.0022 (73,3% positif)** | `e222c/d_runs.csv` | ✅ |
| E4 | Satu-satunya kondisi di mana TGB unggul adalah misspecification — kecil, tapi disebutkan (m-b). | Jaccard tgb **0.4504** vs random 0.4458 (dunia B) | `e222_runs.csv` | ✅ |

> **Pelajaran metodologisnya justru lebih berharga daripada diagnosis yang gugur:** K4 lahir sebagai
> penjelasan yang rapi dan masuk akal untuk hasil yang mengganggu, dan sudah hampir masuk naskah
> sebagai "null yang terprediksi". Yang menghentikannya adalah pra-registrasi dengan dua cabang
> keputusan tertulis di depan. Kalau E224 dijalankan tanpa itu, godaan menafsir ulang `auc_true`
> (yang memang naik dari 36,7% → 60% positif — metrik **sekunder**, jauh di bawah ambang) akan sangat
> besar. **Jangan lakukan itu.** Ambangnya dikunci pada `map_jaccard`, dan `map_jaccard` bilang tidak.

### K-F · Fork kuota yang jatuh (dilaporkan apa adanya)

| # | Klaim | Angka | Sumber | G1 |
|---|---|---|---|---|
| F1 | Desain kuota diuji **di rejim rumahnya sendiri** (bias survei regional) dan gagal. | World C: −0.2457 AUC_true, −0.4688 Jaccard, **0/30** | `e222c_runs.csv` | ✅ |
| F2 | Dan gagal lagi di rejim yang paling ramah baginya (kebenaran seimbang-regional). | World D: −0.2027 AUC_true, −0.2826 Jaccard, **0/30** | `e222d_runs.csv` | ✅ |

### K-G · INT-1 (inventaris gunung api) dan kontrol R2-F

| # | Klaim | Angka | Sumber | G1 |
|---|---|---|---|---|
| G1a | Inventaris gunung api naskah memuat **7** pusat; file kanonik memuat **13** di dalam batas 111–115°E yang naskah nyatakan sendiri. Test 1 dihitung ulang dengan yang kanonik. | ρ = **−0.281** (kanonik 13) vs −0.243 (legacy 7, re-run) | `e219_outcome.json` | ✅ |
| G1b | **Vonis Test 1 tidak berubah**: \|ρ\| tetap < 0.5, jadi tesnya tetap lolos. Angkanya berubah, kesimpulannya tidak. | ambang FAIL \|ρ\|>0.5 | naskah §T1 | ✅ |
| G1c | ⚠ **Angka terbit −0.163 tidak tereproduksi**, bahkan pada inventaris 7-gunung yang sama: re-run 5-seed memberi **−0.243**. Nilai terbit berasal dari **satu instance model**. | −0.163 (terbit, `submission_jcaa_v0.1.tex` brs. 319) vs −0.243 (re-run) | perbandingan | ✅ |
| G1d | G1c bukan kecelakaan terpisah: itu **ketidakstabilan-seed D1 yang muncul di dalam diagnostik tautologi naskah sendiri**. Kutip nilai ensemble, dan pakai ini sebagai bukti internal bahwa protokol D2 memang perlu. | lihat D1/D2 | sintesis | — |
| G2a | Kontrol R2-F (matched terrain, 90 strata): suitability dataran tinggi vulkanik sedikit lebih tinggi daripada non-vulkanik yang tercocokkan. | **0.2249 vs 0.1702** (+0.055) | `e219_terrain_matched.csv` | ✅ |
| G2b | Densitas situs jauh lebih tinggi di lengan vulkanik — **konsistensi, bukan validasi**. | 0.01377 vs **0.00048** situs/km²; lengan non-vulkanik hanya **2 situs** | `e219_terrain_matched.csv` | ✅ |

> **G1c wajib masuk *Response to Reviewers*.** Reviewer 2 secara eksplisit menyebut reproducibility
> ("It should be a way to reproduce the model in a clear way"). Menemukannya sendiri dan melaporkannya
> jauh lebih baik daripada dia yang menemukan.

---

## 3. Keterbatasan yang wajib dinyatakan (jangan dilebihkan, jangan disembunyikan)

1. **Celah kalibrasi sintetik↔nyata (dari K1).** Di titik operasi naskah, kriteria memilih **hybrid**
   pada data nyata tetapi **random** pada data sintetik. Dunia sintetik karena itu **tidak** mereproduksi
   perilaku seleksi data nyata di titik itu. Ini membatasi transfer kesimpulan E222 kembali ke kasus
   nyata dan harus tertulis di naskah, bukan di supplement.
2. **Tanda slope kebenaran kontingen terhadap rejim** — nyata: turun; sintetik: naik. Jangan klaim
   "hard negative selalu buruk". (Koreksi atas klaim kami sendiri, dokumen 09 §7.)
3. **Empat rejim sintetik** (A bias jalan, B misspecified, C bias regional, D kebenaran seimbang +
   bias regional): tak ada desain yang mengalahkan uniform pada kebenaran — dengan **n≈300–500** dan
   bentuk bias yang diuji. **Bukan** bantahan universal terhadap Phillips dkk. 2009.
4. **Lantai deteksi bootstrap ~+0.03** pada n=378: efek di bawah itu tidak bisa disingkirkan.
5. **Boyce = sanity check arah, bukan selektor** (optimumnya tak terkalibrasi ke kebenaran; median
   +0.50/+0.54, borderline). Flag `P4_boyce_tracks_truth: true` di `e222_outcome.json` **lebih longgar
   daripada narasinya** — selaraskan flag itu (m-a) supaya jejak audit tidak berkontradiksi.
6. **n=2** pada lengan non-vulkanik; densitas robust/contingent = konsistensi, **bukan** validasi.
7. **INT-1 sudah tertutup, tetapi memunculkan masalah kedua yang harus diungkap** — lihat §2 K-G.

---

## 4. Yang berubah untuk penulis naskah

- **Dokumen 08 §3 jangan dipakai lagi.** Klaim 1 ("selalu"), klaim 2 ("terburuk 100%", "~10×"), dan
  klaim 4 ("2–5,6×") sudah gugur di sini.
- **Judul.** Kandidat yang bertumpu pada "seleksi memilih yang terburuk" ikut gugur. Yang bertahan
  bertumpu pada **inkomparabilitas evaluasi** — kandidat 1 dan 3 di `JCAA_R1_RESPONSE_PLAN` §3 masih
  aman; kandidat yang menjanjikan "protokol terkoreksi" harus benar-benar memuat protokolnya (D2).
- **Surat balasan reviewer.** K1–K7 masuk ke sana secara eksplisit, dengan riwayatnya. Itu kanal yang
  memang disediakan untuk koreksi klaim, dan menaikkan kredibilitas justru karena kita yang menemukan.

---

*Dibuat 2026-08-03. Semua angka di dokumen ini keluar dari `verify_headline_numbers.py`; jalankan ulang
skrip itu sebelum submit dan pastikan mismatch tinggal K5/K6/K7 (yang memang disengaja tercatat sebagai
klaim lama yang salah).*
