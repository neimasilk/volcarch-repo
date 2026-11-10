# Sesi 2 — Results §3.4–3.6 — lembar fakta untuk PI (9 Nov 2026)

**Aturan lembar ini (C093, `AGENDA_PI_20261103.md` §5):** hanya butir dan angka, tanpa kalimat siap-salin. Bapak menulis
dengan kata-kata Bapak sendiri, dalam bahasa Indonesia, urutan bebas. Saya menerjemahkan tanpa menambah isi; setiap
perubahan di luar terjemahan saya sampaikan dulu. Kalau ada metode atau angka yang belum jelas, tanyakan di chat dan saya
jelaskan — penjelasan itu tidak untuk disalin ke naskah.

**Angka:** ditulis seperti di naskah (titik desimal). Semua dicek ulang hari ini (9 Nov) dari berkas sumber:
`revision_ammo/verify_headline_numbers.py` **60/64** (4 sisanya klaim yang memang ditarik: K5/K6/K7/G1c — jangan dipakai);
angka E224 dan pembagian E222 per dunia dihitung ulang langsung dari `e224_runs.csv` dan `e222_runs.csv` (semua cocok).

**Panjang (keputusan PI 9 Nov: plafon = panjang v0.3):** §3.4 **≤ 261** kata · §3.5 **≤ 92** · §3.6 **≤ 247** —
total ≤ 600 kata bahasa Inggris (± 680–700 kata bahasa Indonesia). Lebih pendek boleh.

**Istilah:** sama dengan sesi 1 / Tabel 1 "Key terms": *background*, *own background*, *common evaluation background*,
*hard-negative share*, *seed*, AUC, *map agreement (Jaccard)*, TGB.

**Yang tetap (tidak ditulis ulang sekarang):** Gambar 4 (kriteria seleksi; caption ditulis Bapak di sesi akhir).

**Tumpang-tindih dengan §3.2–3.3 (sudah Bapak tulis — jangan diulang):**
- §3.2 blok D: 240/240, 52/60, 51/60, 343/360 dan median selisihnya → §3.5 **tidak** mengulang hitungan ini.
- §3.3 blok D: kriteria di set bersih (0.697 → 0.866; common 0.680 / 0.713 / 0.656) → §3.4 cukup merujuk ke §3.3.
- §3.3 blok C: langkah TGB tidak ada di kedua set (−0.005 own, +0.001 common) → §3.6 cukup merujuk.

---

## §3.4 Kriteria seleksi tidak punya optimum interior ("The selection criterion has no interior optimum")

**Fungsi bagian:** menjawab "kenapa tangga itu bisa terjadi?" — aturan yang memilih desain (AUC di own background) tidak
pernah menyuruh berhenti. Plus: berapa biayanya di grid yang benar-benar dipakai E013, dan berapa kalau grid diteruskan.

**A. Bentuk kurva kriteria (Gambar 4):**

| Fakta | Angka | Sumber |
|---|---|---|
| Sintetis (E222), AUC own background di hard-negative share 0.0 / 0.3 / 0.7 / 1.0 | 0.7367 / 0.7820 / 0.8412 / 0.8904 | `experiments/E222_synthetic_ground_truth/results/e222_runs.csv` (G1 #19) |
| Grid sintetis | hanya 4 titik (0.0, 0.3, 0.7, 1.0) | idem |
| Nyata (E218b, 5 seed), 11 titik 0.0–1.0 | 0.7208 → 0.7137 → 0.7253 → … → 0.8435 | `experiments/E218_evaluation_artefact/results/e218b_hardfrac_sweep.csv` (G1 #40) |
| Satu-satunya turunan di data nyata | 0.0 → 0.1: **−0.0071**; setelah itu naik terus | idem |
| Di mana maksimumnya | di ujung rentang yang dicoba, di kedua dunia | idem |

- Isi yang perlu ada: kriteria tidak pernah "menunjuk ke dalam" → tidak pernah memberi tanda berhenti.

**B. Biaya di grid E013 (hard-negative share {0.0, 0.15, 0.30}; "biaya" = selisih AUC sebenarnya / common antara desain
terbaik di grid itu dan desain yang dipilih kriteria):**

| Fakta | Angka | Sumber |
|---|---|---|
| E013 berhenti di 0.30 karena… | grid-nya berhenti di situ, bukan karena kriteria | naskah §2.4; `e218b` |
| Biaya, data nyata | **+0.0044** | G1 #37 |
| Biaya, sintetis (60 kasus) | median **+0.0000**, rata-rata +0.0012, maksimum +0.0088 | G1 #17–18 |
| Kriteria tidak memilih desain terbaik-menurut-kebenaran | 29/60 kasus — tapi selisihnya hampir nol | G1 #18 |
| Yang dipilih kriteria di grid itu (sintetis) | acak 50, TGB 10 (bukan hybrid) | G1 #15 |

- Catatan: di 0.30 kriteria memilih hybrid di data nyata tapi acak di sintetis → "calibration gap" yang sudah ada di
  Limitations. Tidak perlu diulang di sini kecuali Bapak mau.

**C. Biaya kalau grid diteruskan sampai 1.0:**

| Fakta | Angka | Sumber |
|---|---|---|
| Biaya, data nyata | **+0.0973** | G1 #39 |
| Biaya, sintetis | median **+0.1937**, positif di **60/60** kasus | G1 #12–13 |
| Desain yang dipilih (sintetis) | hybrid(1.0) di 60/60 | G1 #11 |
| Desain yang benar-benar terburuk (sintetis) | hybrid(0.0) 50/60 · hybrid(0.3) 8/60 · hybrid(0.7) 2/60 | G1 #14 |
| hybrid(1.0) jadi yang terburuk | **0/60** — pilihan itu mahal, tapi bukan yang terburuk | G1 #14 (K5) |
| Biaya pilihan itu terhadap desain terbaik | **+0.194** AUC sebenarnya (= median +0.1937) | G1 #12 |

- Isi yang perlu ada: bahayanya struktural — kriteria tidak punya rem kalau share dinaikkan.

**D. Set bersih:** sudah di §3.3 blok D (di set itu kriteria malah mengarah ke setting dengan skor common terendah).
Cukup rujukan ke Section 3.3 kalau Bapak mau menyambungkan.

**⚑ F2 (opsional, Bapak putuskan):** E220 mengulang uji yang sama di data nyata dengan **20 seed** (bukan 5):
AUC own 0.707 → 0.842, naik di **setiap** langkah (tanpa turunan 0.0→0.1); AUC common 0.695 → 0.601
(`experiments/E220_wrong_direction_selection/results/e220_sweep.csv`, dihitung ulang hari ini). Naskah v0.3 dan Gambar 4
memakai E218b (5 seed). Saran saya: **tetap pakai E218b** (cocok dengan Gambar 4, sudah lolos G1); E220 cukup disebut
kalau Bapak ingin menyatakan bahwa polanya terulang dengan 20 seed. Menambah E220 memakan ± 25 kata dari plafon 261.

---

## §3.5 Performa yang dilaporkan dan yang sebenarnya berpisah ("Reported and true performance diverge")

**Fungsi bagian:** berapa cepat angka laporan bergerak dibanding kebenaran ketika hard-negative share dinaikkan — rasio dan
arahnya. **Bukan** hitungan run (itu sudah di §3.2).

| Fakta | Angka | Sumber |
|---|---|---|
| Sintetis, 0.0 → 1.0: perubahan AUC laporan (own) | **+0.1538** | `e222_runs.csv` (G1 #5) |
| Sintetis, 0.0 → 1.0: perubahan AUC sebenarnya | **+0.0764** | idem (G1 #6) |
| Rasio, tiga cara menghitung | endpoint **2.01×** · kemiringan OLS per run **2.12×** · median rasio per run **2.00×** | G1 #7, #10 |
| Kesimpulan yang aman | "sekitar dua kali"; sebut estimatornya kalau memakai satu angka | G1 catatan K2 |
| Data nyata, 0.0 → 1.0: AUC laporan | **+0.1227** | `e218b_hardfrac_sweep.csv` (G1 #33) |
| Data nyata, 0.0 → 1.0: AUC common background | **−0.0973** | idem (G1 #34) |
| Arah di data nyata | **berlawanan** (laporan naik, common turun) | idem |

**Prediksi yang gagal (E222, desain tertulis — bukan pra-registrasi yang ber-cap waktu, C082):**
- Prediksi P1 di `experiments/E222_synthetic_ground_truth/DESIGN.md`: inflasi (AUC own − AUC sebenarnya) naik bersama
  hard-negative share, **Spearman > 0.5 di semua run (gabungan)**.
- Hasil: Spearman gabungan **0.4395** → di bawah ambang → **gagal**, dilaporkan sebagai gagal (G1 #20).

**⚑ F3 (kecil):** v0.3 menyebut prediksi itu berlaku "in every regime"; desainnya sendiri menulis "across all runs"
(satu Spearman gabungan). Saran: tulis sesuai desain (gabungan semua run).

---

## §3.6 Hasil nol TGB, dilaporkan sebagai belum terjelaskan ("The target-group-background null, reported as unexplained")

**Fungsi bagian:** TGB diberi kondisi yang diminta teorinya (proses pengamatan bias survei yang diketahui) dan tetap tidak
membantu peta; satu penjelasan diuji (E224) dan gagal → nol dilaporkan apa adanya.

**A. E222 — dunia sintetis dengan bias survei jalan (dunia A) dan model salah spesifikasi (dunia B):**

| Fakta | Angka | Sumber |
|---|---|---|
| TGB − acak, kesepakatan peta desil teratas (Jaccard), dunia A + B | **−0.010**, positif **28/60 (46.7 %)** | `e222_runs.csv` (G1 #21–22; dihitung ulang hari ini) |
| Ambang di desain tertulis (P3) | TGB menang di ≥ 60 % kasus → **tidak tercapai** | `E222/DESIGN.md` |
| Dunia A saja (bias survei, spesifikasi benar) | −0.0247, positif 30 % (Jaccard 0.6898 vs 0.7145) | dihitung ulang hari ini; G1 #23 |
| Dunia B (salah spesifikasi) — satu-satunya tempat TGB unggul | Jaccard **0.4504 vs 0.4458**; positif 63.3 % | G1 #24; dihitung ulang hari ini |
| Dunia C (bias regional), TGB − acak, AUC sebenarnya | **−0.0010**, positif 56.7 % (17/30) | `e222c_runs.csv` (G1 #28) |
| Dunia D (kebenaran regional seimbang), idem | **+0.0022**, positif 73.3 % (22/30) | `e222d_runs.csv` (G1 #32) |

- Bacaan v0.3: di dunia bias regional TGB **netral**, bukan merugikan; menang hanya di dunia salah spesifikasi — dilaporkan
  karena benar, tapi tidak dijadikan dasar.

**B. E224 — menguji satu penjelasan (pra-registrasi: `DESIGN.md` di commit `d4f44af` sebelum hasil `c826b40`, 3 Agu):**
- Penjelasan yang diuji: `road_dist` bukan fitur model → bias survei tak terlihat di ruang fitur → TGB tak punya apa-apa
  untuk dikoreksi.
- Desain: dunia A E222 persis sama (lattice, koefisien, seed, CV), **satu** beda: `road_dist` masuk fitur atau tidak.
  10 dunia × 2 desain × 2 lengan × 3 algoritma = 120 run. Dua cabang keputusan ditulis sebelum run.
- Metrik utama (ditetapkan dulu): Jaccard peta, TGB − acak, 30 pasangan per lengan; ambang: rata-rata > 0 **dan** ≥ 60 %
  pasangan positif.

| Lengan | TGB − acak, Jaccard | Positif | TGB − acak, AUC sebenarnya | Positif |
|---|---|---|---|---|
| tanpa `road_dist` (kontrol) | **−0.0254** | 30 % | −0.0006 | 36.7 % |
| dengan `road_dist` (uji) | **−0.0217** | 30 % | −0.0000 | 60.0 % |

Sumber: `experiments/E224_road_feature_tgb/results/e224_runs.csv`, `e224_outcome.json` (dihitung ulang hari ini, cocok).

- Hasil: tidak ada beda → penjelasan **gagal**; status eksperimen FAILED (bukan REFUTED, lihat batasan di bawah).
- Metrik sekunder (AUC sebenarnya) naik dari 36.7 % ke 60 % positif — **tidak** dipakai untuk menafsir ulang, karena
  metrik utama yang ditetapkan dulu adalah Jaccard.
- Batasan uji kami sendiri: `road_dist` berkorelasi **+0.49** dengan `river_dist` dan **+0.31** dengan elevasi
  (`e224_outcome.json`, `road_dist_collinearity`) → di kondisi kontrol model sudah sebagian "melihat" sinyal bias.
- Uji yang bersih butuh permukaan upaya survei yang ortogonal terhadap terrain sejak dirancang → pekerjaan mendatang.

**C. Data nyata:** langkah TGB tidak ada di kedua set (§3.3 blok C). Cukup rujukan.

---

## Jawaban untuk reviewer di bagian ini

- **R1:** naskah berdiri sendiri — tanpa "reviewer", "revisi", "versi sebelumnya", "originally reported", "we withdraw";
  statistik dijelaskan untuk pembaca non-statistikawan; tanpa idiom ("dial", "judging at home", "home court"; "ladder"
  boleh — sudah didefinisikan Bapak di §1); "Section", bukan "§".
- **R1-3 (istilah):** TGB, Jaccard, AUC sebenarnya vs laporan — beri intuisi satu kalimat kalau belum dijelaskan di §1
  atau Tabel 1.
- **Bukan permintaan reviewer, tapi batas klaim:** hasil TGB berlaku untuk empat rezim dan ukuran sampel yang diuji;
  Limitations sudah menyatakannya (dengan rujukan Phillips et al. 2009), jadi di sini cukup tidak melampauinya.

## Jangan ditulis (sudah terbukti salah atau terlalu kuat)

- "kriteria memilih desain terburuk" (**K5** — salah: 0/60; yang benar "+0.194 terhadap desain terbaik").
- "naik monoton" untuk data nyata (**K6** — ada turunan −0.0071 di E218b). Untuk 4 titik sintetis memang naik di setiap
  titik; rumusan yang aman untuk keduanya: **maksimum selalu di ujung grid yang dicoba**.
- "± 10× lebih cepat" (**K2** — sekitar 2×).
- "selalu terinflasi" tanpa pecahan (**K3** — 343/360, dan itu sudah di §3.2).
- "nol TGB sudah diprediksi / dijelaskan oleh representabilitas ruang fitur" (**K4** — diuji di E224 dan gagal).
- "TGB tidak bekerja" secara umum (hanya untuk rezim yang diuji).
- "pra-registrasi" untuk E222 (desain tertulis, masuk git bersama hasil; C082). Untuk **E224 boleh**.
- "selalu" / "tidak pernah" tanpa pecahan di sebelahnya (aturan line 01 #4): 60/60, 0/60, 29/60, 28/60.
