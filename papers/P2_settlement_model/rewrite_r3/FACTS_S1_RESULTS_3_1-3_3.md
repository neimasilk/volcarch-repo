# Sesi 1 — Results §3.1–3.3 — lembar fakta untuk PI (3 Nov 2026)

**Aturan lembar ini (C093, `AGENDA_PI_20261103.md` §5):** hanya butir dan angka, tanpa kalimat siap-salin. Bapak menulis
dengan kata-kata Bapak sendiri, dalam bahasa Indonesia, urutan bebas. Saya menerjemahkan tanpa menambah isi; setiap
perubahan di luar terjemahan saya sampaikan dulu. Kalau ada metode atau angka yang belum jelas, tanyakan di chat dan saya
jelaskan — penjelasan itu tidak untuk disalin ke naskah.

**Angka:** ditulis seperti di naskah (titik desimal). Semua dicek ulang hari ini dari berkas sumber:
`revision_ammo/verify_k1_numbers.py` **39/39**, `revision_ammo/verify_headline_numbers.py` **60/64** (4 sisanya klaim yang
memang sudah ditarik, K5/K6/K7/G1c — jangan dipakai), plus tabel tangga dan hitungan sintetis di bawah (cek manual hari ini).

**Panjang:** naskah sekarang 11.234 kata dari batas 11.500 (termasuk referensi). Bagian ini sekarang ±1.380 kata
(§3.1 ±80, §3.2 ±710, §3.3 ±590). Lebih pendek boleh; lebih panjang tidak.

**Istilah:** pakai istilah Tabel 1 "Key terms" (tulisan Bapak sendiri, passage 5): *background*, *own background*,
*common evaluation background*, *seed*, AUC, *hard-negative share*, *robust core*, *contingent fringe*.

**Yang tetap (tidak ditulis ulang sekarang):** Tabel 4 (tangga E007–E013), Tabel 5 (model nol), Gambar 2, Gambar 3, Tabel S9.
Keterangan (caption) gambar dan tabel ditulis Bapak di sesi akhir (jadwal 22–24 Nov).

---

## §3.1 Tangga yang dilaporkan ("The reported ladder")

**Fungsi bagian:** menampilkan ulang klaim yang diuji — kenaikan AUC dalam tujuh eksperimen — sebelum diuji di §3.2.

| Fakta | Angka | Sumber |
|---|---|---|
| AUC XGBoost E007 (background acak) | 0.659 ± 0.077 | `experiments/E007_*/results/model_results.txt` |
| AUC XGBoost E013 (hybrid), run asli | 0.768 ± 0.069 | `experiments/E013_settlement_model_v7/results/model_results.txt` |
| AUC XGBoost E013, rata-rata 20 seed | 0.751 ± 0.013 | `supplement/e013_robustness_summary.txt` |
| Rentang 20 seed (XGBoost) | 0.729–0.774 | idem |
| Kenaikan tangga yang dilaporkan | +0.092 (= 0.751 − 0.659) | turunan |
| Model nol, AUC (spatial CV) | acak 0.500; heuristik sungai < 2 km 0.581; DKNS 0.646 | `supplement/null_model_comparison.txt` |
| Seed terburuk vs DKNS | +0.083 (= 0.729 − 0.646) | turunan |
| Selisih rata-rata seed vs DKNS | +0.105 (bukan +0.122 — angka lama memakai run terbaik 0.768) | `SIG_G1_VERIFICATION_20261026.md` |
| Langkah tangga (XGBoost) | E008 sungai 0.685 · E009 tanah 0.664 · E010 TGB 0.711 · E011 TGB tuned 0.725 · E012 TGB diperluas 0.730 · E013 hybrid 0.768 | `model_results.txt` tiap eksperimen (semua cocok dengan Tabel 4) |

- Model nol dinilai di grid semua sel, label = dalam 2 km dari situs; E013 dinilai di *own background*-nya → perbandingan
  di Tabel 5 mencampur dua background (koreksinya di §3.2, lihat di bawah).
- DKNS memakai lokasi situs yang sebenarnya → batas atas yang tidak adil, bukan pesaing.
- Isi yang perlu ada: tangga dibaca apa adanya = desain background makin realistis → model makin baik. Itu bacaan yang
  diuji, bukan kesimpulan.

**⚑ Untuk Bapak putuskan (F1):** naskah sekarang menaruh "TSS 0.507 ± 0.167" di samping AUC rata-rata seed 0.751. TSS 0.507
milik **run asli** (`model_results.txt`); TSS **rata-rata 20 seed** = **0.465 ± 0.021** (`e013_robustness_summary.txt`).
Saran: pasangkan yang sejenis (0.751 dengan 0.465, atau 0.768 dengan 0.507).

---

## §3.2 Tangga tidak bertahan di common evaluation background

**Fungsi bagian:** uji utama di 378 rekaman kompilasi — apakah kenaikan tangga tetap ada kalau semua desain dinilai di
background yang sama.

**A. Dekomposisi (E217b, 5 seed × 3 algoritma; `experiments/E217_maxent_benchmark/results/e217b_raw_results.csv`):**

| Fakta | Angka |
|---|---|
| Kontribusi redesign background, common background, fitur lengkap (terrain + sungai), rata-rata 3 algoritma | **−0.0142** AUC |
| Interval seed-cluster | −0.021 s.d. −0.005 |
| Sel seed × algoritma yang positif | 3 dari 15 |
| Kontribusi redesign, fitur terrain saja | +0.0054 |
| Kontribusi fitur sungai, common background | **+0.0424**, positif 12/12 pasangan |
| Redesign yang sama dinilai di own background (fitur lengkap) | rata-rata +0.0316 (MaxEnt +0.0145, XGBoost +0.0431, RF +0.0373) |

- Hasil kompilasi ini **tidak** berlaku umum: di set bersih hasilnya positif (§3.3).

**B. Desain hybrid vs acak per background evaluasi (E218, `experiments/E218_evaluation_artefact/results/e218_stageA_raw.csv`):**
- hybrid > acak: 3/3 algoritma di background hybrid; **0/3** di background uniform, TGB, dan stratified.

**C. Inflasi own background (E218, 378 rekaman):**

| Fakta | Angka |
|---|---|
| Hybrid: own − uniform, sel positif | **60/60**, rata-rata +0.037, rentang +0.005 s.d. +0.084 |
| Ukuran dibanding kenaikan tangga | orde yang sama dengan +0.092 |
| TGB: own − uniform | rata-rata **−0.0054**, 22/60 positif (TGB malah lebih tinggi di uniform) |
| TGB − acak di own background (E217 benchmark) | +0.003 |

- Jadi langkah TGB (E010–E012) **tidak terulang** di benchmark — bukan "dijelaskan oleh inflasi". Jangan sebut artefak.

**D. Dunia sintetis dengan kebenaran diketahui (E222, `experiments/E222_synthetic_ground_truth/results/e222_runs.csv`;
dicek hari ini):**

| Desain | Run dengan reported > true | Median selisih |
|---|---|---|
| hybrid | **240/240** | +0.238 |
| acak | 52/60 | +0.020 |
| TGB | 51/60 | +0.014 |
| semua (dua permukaan) | 343/360 | — |

- Pada acak dan TGB sebagian selisih datang dari beda sampel presensi teramati vs presensi sebenarnya, bukan dari
  background.

**E. Uji statistik (E223):**
- CI ekuivalensi: menolak +0.092 di **12/12** sel algoritma × background evaluasi (`experiments/E223_statistical_robustness/results/e223a_equivalence_ci.csv`).
- Bootstrap blok presensi: 30 replikasi, 1 dilewati (< 50 situs out-of-bag) → 29 dipakai; fit in-bag, nilai out-of-bag;
  batas atas CI +0.008 / +0.025 / +0.026 (tiga algoritma) → menolak +0.092 untuk ketiganya (`…/E223_statistical_robustness/results/e223b_bootstrap_summary.csv`).
- 29 replikasi = bootstrap kecil; dipakai karena margin penolakan lebar, bukan karena ujung interval presisi.
- Regularisasi MaxEnt β 0.5–4.0: −0.020 s.d. −0.022, 1/10 run positif → bukan overfit model tanpa regularisasi
  (`…/E223_statistical_robustness/results/e223c_beta_summary.csv`).

**F. Akibat untuk Tabel 5 (model nol):**
- Desain E013 di common (uniform) background: **0.706** (XGBoost, rata-rata seed; E218).
- Masih di atas DKNS 0.646, tapi margin turun dari +0.105 ke **± +0.06**; margin seed terburuk (+0.083) mendekati
  batas deteksi **+0.03** (§3.7).
- 0.751 (pipeline asli) dan 0.706 (benchmark berbasis lattice) tidak diukur dengan sampling yang sama → baca penyusutan
  sebagai orde besaran saja.
- Yang bertahan: fitur lingkungan menangkap struktur di luar interpolasi situs terdekat. Yang tidak: besarnya. Tabel 5 =
  batas atas, bukan lantai.

**Gambar 3:** kiri = AUC own vs common per desain; kanan = reported − true di 360 run sintetis per desain.

---

## §3.3 Rekaman presensi dan subset bersih

**Fungsi bagian:** menunjukkan bahwa hasil §3.2 bergantung pada data presensi — semua analisis diulang di 161 situs bersih
(E236, pra-registrasi `f9d2eb9`), skrip analisis tidak diubah selain berkas presensi.

**A. Data:**
- 378 rekaman di 335 koordinat berbeda; ±⅓ monumen/patung/tugu modern atau salah letak.
- Pembersihan (aturan di §2.1, ditulis di sesi Methods): 217 dibuang = 85 hasil geocode dari nama (Wikipedia) + 124
  monumen modern + 8 candi daerah lain/obyek wisata/institusi → **161** (155 OSM + 6 Wikidata).
  Sumber: `experiments/E236_p2_presence_sensitivity/clean_sites.py`, `README.md`; hitungan dicek 2 Nov (C092).
- Semua angka di bagian ini ada untuk kedua set di **Tabel S9**.

**B. Kontras redesign di set bersih (common background; `E236/results/A_e217b`, `A_seeds10`):**

| Fakta | Angka |
|---|---|
| 5 seed (pra-registrasi) | **+0.0507** |
| 10 seed (perluasan, dideklarasikan sebelum run tapi **setelah** hasil 5 seed) | **+0.0510** [+0.039, +0.063] |
| Sel positif | 29 dari 30 |
| Per algoritma | XGBoost +0.038 · RandomForest +0.044 · MaxEnt +0.072 |
| Dua ensemble 5 seed tanpa seed bersama | +0.0505 dan +0.0516 |
| Kontras yang sama di own background (10 seed) | +0.098 |
| Inflasi = own − common | **+0.047** (± sebesar gain-nya sendiri) |

**C. Inflasi = besaran yang tidak bergantung pada data presensi (`E236/results/B_e218a`):**
- hybrid own > uniform: bersih **54/60** sel (rata-rata +0.041) vs kompilasi **60/60** (+0.037).
- Beda 1: di set bersih hybrid > acak di **semua** background evaluasi (3/3 algoritma di uniform, TGB, hybrid,
  stratified); di kompilasi hanya di own background.
- Beda 2: langkah TGB tetap tidak ada di kedua set — TGB − acak: −0.005 (own), +0.001 (common).

**D. Kriteria seleksi (`E236/results/C_e218b`):**
- AUC own background: 0.697 (hard-negative share 0.0) → 0.866 (1.0); maksimum di ujung rentang.
- AUC common background di 0.0 / 0.3 / 1.0: 0.680 / 0.713 / 0.656.
- Artinya di set ini kriteria merekomendasikan setting dengan skor common **terendah** — bukan sekadar tidak berhenti.

**E. Peta prioritas (`E236/results/D_e221`, `F_disjoint_clean.csv`, `F_disjoint_full.csv`):**
- Rasio densitas robust core / contingent fringe: RF 2.54 · XGBoost 8.51 · MaxEnt 5.30.
- Turnover desil teratas antar pasangan seed: 41.6–54.8 %.
- Dua ensemble 5 seed disjoint: Jaccard 0.70–0.77 (bersih) vs 0.76–0.86 (kompilasi).
- Fringe hanya berisi 5 situs (XGBoost), 15 (RF), 5 (MaxEnt) → rasio = arah, bukan besaran presisi.

**F. Kontrol elevasi (`E236/results/E_e235`; detail di §3.10):**
- < 1.000 m, selisih kesesuaian dekat − jauh: bersih +0.046 (acak dan TGB), +0.060 (hybrid); kompilasi +0.049, +0.049,
  +0.060.
- Hybrid saja yang memenuhi kriteria SPECIFIC (pra-registrasi) di kedua set; ringkasan tiga desain = inconclusive; dua
  definisi sensitivitas bertukar label (inconclusive / not specific) antara kedua set.

---

## Jawaban untuk reviewer di bagian ini

- **R1:** naskah berdiri sendiri — tanpa "reviewer", "revisi", "versi sebelumnya", "originally reported", "we withdraw";
  statistik dijelaskan untuk pembaca non-statistikawan; tanpa idiom ("dial", "judging at home", "home court", "ladder"
  boleh — sudah didefinisikan Bapak di §1); "Section", bukan "§".
- **R2:** volcanic vs elevasi — di bagian ini hanya ringkasan §3.3 F; pembahasan penuh di §3.10.

## Jangan ditulis (sudah terbukti salah atau terlalu kuat)

- "selalu" / "always" tanpa pecahan di sebelahnya (aturan line 01 #4) — tulis 343/360, 60/60, 54/60.
- "aturan seleksi memilih desain terburuk" (K5 — salah), "naik monoton" (K6 — di data nyata ada satu turunan),
  "± 10× lebih cepat" (K2 — 2.01×), "densitas 2–5.6×" (K7 — 1.93×/4.34×/5.62× di kompilasi).
- "langkah TGB adalah artefak" (C084 — tidak terulang, bukan dijelaskan).
- "kenaikan tangga tidak nyata" untuk kedua set (C090/K1 — di set bersih redesign memberi +0.051).
- "pra-registrasi" untuk E217 dan E219 (tidak punya desain tertulis; C082). E224, E235, E236 boleh.

## Catatan kecil (tidak mengubah naskah)

- `supplement/null_model_comparison.txt` dan skripnya berlabel "Malang Raya", padahal batas yang dipakai seluruh Jawa Timur
  (111–115° BT, 9–6.5° LS). Label usang saja; angkanya tidak terpengaruh.
