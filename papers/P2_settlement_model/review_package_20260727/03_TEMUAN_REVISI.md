# Dokumen 3 — Temuan Revisi P2/JCAA #280 dan Letak Eksperimennya

**Tanggal:** 2026-07-27 | **Untuk:** Go Frendi Gunawan (co-author) | **Status:** menunggu review co-author
**Semua kode, data, dan hasil ada di repo — bisa dijalankan ulang. Perintahnya di §7.**

> ⚠ **DIKOREKSI SEBAGIAN — 2026-08-03.** Dokumen ini benar untuk babak 1, tetapi set klaim yang akan
> masuk naskah v0.2 **sudah bergerak dua kali** sejak ditulis: dokumen 09 (K1–K3) lalu dokumen 10
> (K5–K7 + hasil SIG G1). **Untuk angka yang boleh dikutip, pakai `10_SET_KLAIM_TERKOREKSI.md`.**
> Yang berubah dari dokumen ini: inflasi "+0.041…+0.051, 15/15" adalah rata-rata per-algoritma —
> per sel seed × algoritma rentangnya **+0.005…+0.084, 60/60 positif**; dan INT-1 ternyata memunculkan
> temuan kedua (nilai terbit ρ = −0.163 tidak tereproduksi; re-run 5-seed = −0.243).


---

## 1. Ringkasan Satu Halaman

Naskah yang kita submit ke JCAA berkata, di abstraknya:

> *"The main finding is that pseudo-absence realism, not feature count alone, is the dominant lever for
> spatial transfer under survey-biased archaeological data."*

Buktinya adalah tangga AUC dari E007 (0.659) ke E013 (0.751 seed-averaged / 0.768 run terbaik) seiring
desain background diperbaiki.

**Klaim itu tidak bertahan.** Tangga tersebut adalah **artefak evaluasi**: setiap rung dinilai terhadap
negatif yang dipilihnya sendiri. Begitu semua desain dinilai pada satu evaluation background yang dipatok
tetap, tangganya rata.

| Yang diukur | Hasil |
|---|---|
| Efek redesign background (evaluasi dicocokkan) | **−0.014 AUC** — MaxEnt 0/20 seed positif, XGBoost 4/20, RF 4/20 |
| Efek menambah satu fitur (jarak sungai) | **+0.042 AUC — positif di 60/60** perbandingan berpasangan |
| Inflasi akibat menilai tiap desain pada background sendiri | **+0.041…+0.051 AUC — 15/15 positif** |

Baris ketiga besarnya sama dengan **seluruh** kenaikan E007→E013 yang kita laporkan.

**Dan lebih dalam dari itu** (temuan yang belum ada pagi tadi): ketika knob `hard_frac` milik naskah kita
sendiri disapu dari 0.0 ke 1.0, **angka yang dilaporkan dan generalisasi sesungguhnya bergerak berlawanan
arah** — AUC-background-sendiri naik 0.721→0.844 sementara AUC-background-bersama turun 0.699→0.602
(Spearman +0.961 vs −0.708). Artinya: **prosedur tuning kita mendaki gradien yang salah.**

**Yang menggantikan klaim lama** (bukan hanya pembongkaran):
1. Desain background **mengubah sel mana yang didatangi surveyor** (tumpang tindih desil-teratas turun
   0.684 → 0.345) padahal semua metrik diskriminasi bilang model-modelnya setara.
2. Peta prioritas **tidak stabil terhadap seed acak saja** — 31–45% sel desil-teratas berganti.
3. Cacat inventaris gunung api (7 vs 13) sudah diperbaiki; **vonis tautologi naskah tetap selamat**.
4. Permintaan kontrol non-vulkanik dari Reviewer 2 sudah dijawab dengan desain matching.

**Tidak ada yang dikirim ke editor.** Menunggu persetujuan Mas Go Frendi.

---

## 2. Klaim yang Gugur — persisnya apa

| Lokasi di naskah | Kalimat |
|---|---|
| Abstract, kalimat terakhir | "pseudo-absence realism, not feature count alone, is the dominant lever for spatial transfer" |
| §1 Introduction | "under survey-biased data, better background design should produce larger transfer gains than feature accumulation alone" |
| §4.1 Discussion | "The strongest performance gains came from correcting background-label realism, not from adding more covariates." |
| §5 Conclusions | "under survey-biased archaeological data, pseudo-absence realism is the dominant determinant of spatial transfer" |

Reviewer 1 membaca klaim itu persis sebagaimana dimaksud: *"The key takeaway—that pseudo-absence realism
plays a dominant role in model transferability—is clearly stated."* Jadi ini bukan salah tafsir yang bisa
diperbaiki dengan menulis ulang kalimat.

Naskah **sudah mengutip Lobo dkk.** (sebagai `lobo2010` di §1.2) — peringatan baku bahwa AUC tidak bisa
dibandingkan lintas sampel background berbeda — lalu tidak menerapkannya pada tangganya sendiri.

---

## 3. E217 — bagaimana ini ketahuan

**Letak:** `experiments/E217_maxent_benchmark/`
**Pemicu:** permintaan Reviewer 1 untuk membandingkan dengan MaxEnt ("essential", disebut 2×).

### 3.1 Validasi pipeline dulu — ini yang membuat sisanya berdiri

Sebelum menyimpulkan apa pun, reimplementasi independen diadu dengan angka yang terbit:

| Besaran | Naskah terbit | Reimplementasi E217 |
|---|---|---|
| E013 hybrid, AUC XGBoost seed-averaged (background sendiri) | **0.751** | **0.750** |
| E007 terrain-only background acak, AUC XGBoost | 0.659 | 0.670 |
| Fraksi hard-negative terealisasi (zdist ≥ 2) di desain hybrid | **0.62** | **0.623** |

Baris ketiga yang paling penting: naskah menandai angka 0.62 itu sebagai anomali yang belum terjelaskan
(§2.4 Methods, "This pool composition effect should be considered..."). Reimplementasi independen mendarat
di angka ganjil yang sama. **Jadi apa pun yang menyusul adalah sifat desain kita, bukan akibat implementasi
yang berbeda.** Ini penting kalau ada reviewer yang mau bilang "kode Anda beda".

### 3.2 Dua run

| Skrip | Isi |
|---|---|
| `01_maxent_benchmark.py` | 3 desain background × 2 set fitur × 3 algoritma × 5 seed, tiap desain dinilai pada background-nya sendiri (= cara naskah) |
| `02_matched_evaluation.py` | Ulangi dengan **satu evaluation background bersama** + ablasi site-buffer |

Run 01 sudah menunjukkan tidak ada algoritma — MaxEnt sekalipun — yang menghasilkan tangga monoton, dan
efek background (+0.022) sudah lebih kecil dari efek satu fitur (+0.045).

Run 02 menguraikannya (5 seed, set fitur penuh):

| Komponen | MaxEnt | XGBoost | RandomForest |
|---|---|---|---|
| Ekslusi site-buffer saja | −0.006 | −0.000 | +0.006 |
| TGB di atas random ber-buffer | +0.002 | −0.002 | −0.002 |
| Hybrid di atas TGB | −0.027 | −0.005 | −0.007 |
| **TOTAL, evaluasi bersama** | **−0.032** | **−0.007** | **−0.003** |
| TOTAL, background sendiri (= yang dilaporkan naskah) | +0.015 | +0.043 | +0.037 |

**Mengapa artefak ini muncul:** background hybrid duduk lebih jauh dari situs di ruang lingkungan (fraksi
zdist ≥ 2 = 0.623 vs 0.503 pada random). Membedakan situs dari negatif yang lebih tidak mirip adalah soal
yang lebih mudah — AUC naik tanpa transfer membaik.

---

## 4. E218 — apakah refutasinya sendiri bertahan?

**Letak:** `experiments/E218_evaluation_artefact/` | **Pre-registrasi:** `DESIGN.md` (ditulis sebelum run)

Prinsipnya: refutasi tidak boleh diuji dengan standar lebih longgar daripada klaim yang digugurkannya.
Empat ancaman diuji.

### 4.1 Tes yang menentukan (Stage A) — 20 seed, 4 evaluation background

Prediksi yang di-pre-register: **kalau artefaknya nyata, hybrid menang HANYA melawan negatif mirip-hybrid.
Kalau naskah yang benar, hybrid menang di keempat-empatnya.**

Hybrid menempati peringkat teratas (dari 3 algoritma):

| Evaluation background | uniform | tgb | **hybrid** | stratified |
|---|---|---|---|---|
| AUC | 0/3 | 0/3 | **3/3** | 0/3 |
| TSS | 0/3 | 0/3 | 2/3 | 0/3 |

Berpasangan per seed (hybrid − random AUC, 20 seed):

| Evaluation background | MaxEnt | XGBoost | RandomForest |
|---|---|---|---|
| uniform | −0.033 (0/20) | −0.009 (4/20) | −0.009 (4/20) |
| tgb | −0.027 (0/20) | −0.004 (8/20) | −0.007 (6/20) |
| **hybrid** | **+0.007 (14/20)** | **+0.015 (19/20)** | **+0.010 (18/20)** |
| stratified | −0.032 (0/20) | −0.004 (7/20) | −0.006 (4/20) |

Tandanya berbalik — dan hanya berbalik — ketika background evaluasi cocok dengan desain latihnya.
XGBoost: 4/20 seed di evaluasi uniform, **19/20** di evaluasi hybrid.

### 4.2 Metrik yang kebal artefak

**Continuous Boyce index** (Hirzel dkk. 2006) — presence-only, dihitung terhadap sampel availability
uniform yang dipatok, jadi tidak bisa diinflasi oleh pilihan background latih. Hybrid − random, 20 seed:

| Algoritma | rata-rata | seed yang memihak hybrid |
|---|---|---|
| MaxEnt | +0.017 | 11/20 — setara lemparan koin |
| XGBoost | +0.041 | 13/20 — lemah |
| RandomForest | **−0.095** | **2/20 — jelas lebih buruk** |

**Rumusan yang didukung bukti: "tidak ada manfaat yang andal di bawah metrik yang jujur."** BUKAN "desain
background tidak berguna" — itu akan jadi overclaim versi kita sendiri, kesalahan yang persis sedang
kita koreksi.

### 4.3 Ancaman lain

| Ancaman | Hasil |
|---|---|
| Ukuran block (Stage B) | hybrid − random di evaluasi bersama: −0.020…+0.004 di ~40/50/60 km. Datar di semua skala. |
| Resolusi lattice (Stage D) | Di lattice ~150 m: tangga tetap ada di background sendiri (+0.047), lenyap di background bersama (−0.001) |

---

## 5. E218b + E219 — mekanismenya, dan apa yang menggantikan

### 5.1 E218b — mekanisme, percobaan kedua

**Letak:** `experiments/E218_evaluation_artefact/02_mechanism_hardfrac.py`

Percobaan **pertama** (E218 Stage C) **gagal dan instrumennya rusak** — saya laporkan karena ini bagian
dari jejak audit. Ia menyampel *pita* zdist sempit, yang menghasilkan background terkonsentrasi di cangkang
tipis: terpisah secara trivial berapa pun jaraknya (auc_own **0.98** justru di pita terdekat — kebalikan
hipotesis) dan tak berguna untuk generalisasi (auc_common 0.55). Rancangan itu mengacaukan *jarak* dengan
*konsentrasi*. Null dari instrumen rusak bukan bukti.

Redesainnya menyapu knob **`hard_frac` milik naskah kita sendiri** (0.0 → 1.0) dari pool kandidat alami:

| hard_frac | mean zdist | **AUC background sendiri** | **AUC background bersama** | inflasi |
|---|---|---|---|---|
| 0.0 | 2.10 | 0.721 | 0.699 | 0.022 |
| 0.2 | 2.26 | 0.725 | 0.695 | 0.031 |
| **0.3 ← pilihan E013** | **2.31** | **0.738** | **0.695** | **0.044** |
| 0.5 | 2.48 | 0.760 | 0.681 | 0.079 |
| 0.7 | 2.61 | 0.783 | 0.662 | 0.121 |
| 1.0 | 2.81 | **0.844** | **0.602** | 0.242 |

| Hubungan | Spearman | p |
|---|---|---|
| ketidakmiripan → inflasi | **+0.961** | 1.1e-92 |
| ketidakmiripan → AUC background sendiri | +0.886 | 2.3e-56 |
| ketidakmiripan → AUC background bersama | **−0.708** | 2.0e-26 |

**Kedua kurva berlawanan arah.** Dan ini menuding tuning kita sendiri: E013 menyapu
`hard_frac` ∈ {0.0, 0.15, 0.30} lalu memilih **0.30** — nilai maksimum yang ditawarkan. Di rentang itu AUC
laporan naik **+0.018** sementara generalisasi turun **−0.004**. Tuningnya tidak memperoleh apa pun yang
nyata; ia mengoptimalkan inflasi.

Catatan kejujuran: cabang ketiga sudah di-pre-register — kalau `auc_common` **naik** bersama `hard_frac`,
berarti hard negative memang membantu dan intuisi asli naskah sebagian direhabilitasi. Yang terjadi
kebalikannya.

### 5.2 E219 — apa yang menggantikan klaim lama

**Letak:** `experiments/E219_map_divergence/`
Skala: 378 presence, 588.535 sel frame, 5 seed × 3 desain × 3 algoritma = 45 permukaan prediksi.

**Bagian A — peta berubah walau skor tidak.** Kontrol yang membuat ini falsifiable: peta dibandingkan
antar-desain (seed sama) **dan** dalam-desain (seed beda). Kalau dua undian desain yang sama berbeda
sebanyak dua desain berbeda, itu cuma noise.

Tumpang tindih Jaccard 10% teratas ("tier prioritas survei"):

| Algoritma | dalam-desain (lantai bising) | antar-desain | lolos noise? |
|---|---|---|---|
| MaxEnt | 0.684 | **0.466** | ya |
| XGBoost | 0.549 | 0.488 | ya (tipis) |
| RandomForest | 0.690 | 0.651 | tidak |

Per pasangan, yang menggerakkan peta adalah hybrid: random↔tgb sepakat 0.55–0.73, tapi
**random↔hybrid jatuh ke 0.345** di MaxEnt.

**Temuan kedua, mungkin lebih penting: peta prioritas tidak stabil terhadap seed acak saja** — menjalankan
ulang desain yang sama dengan seed berbeda mengganti **31–45%** sel desil teratas. Ini temuan
reproduktibilitas dengan konsekuensi lapangan langsung, dan tidak ada di naskah yang disubmit.

**Bagian B — apakah ketidaksepakatan itu terorganisasi seperti yang dijanjikan koreksi bias?**
Pergeseran rank persentil relatif terhadap background random, per kuintil jarak-jalan:

| Algoritma | desain | Q1 (dekat jalan) | Q2 | Q3 | Q4 | Q5 (terpencil) |
|---|---|---|---|---|---|---|
| MaxEnt | hybrid | −0.025 | −0.024 | −0.033 | −0.018 | **+0.101** |
| XGBoost | hybrid | −0.019 | −0.018 | −0.023 | −0.015 | **+0.076** |
| RandomForest | hybrid | −0.021 | −0.017 | −0.019 | −0.009 | **+0.067** |

**Dukungan parsial, dan disebut parsial.** Arahnya benar (menggeser ke tanah sulit dijangkau), tapi efeknya
terkurung di kuintil ekstrem, korelasi keseluruhan lemah (Spearman +0.065…+0.124), dan **elevasi adalah
penjelasan tandingan yang sama kuat atau lebih** (sampai +0.305 di MaxEnt).

**Bagian C-1 — cacat inventaris gunung api (INT-1) DITUTUP.**
Kode yang disubmit meng-hardcode **7** gunung. File kanonik
(`data/processed/dashboard/volcanoes_java_full.csv`) punya **13** di dalam batas studi naskah sendiri
(111–115°E): tambahannya **Lawu, Wilis, Kawi-Butak, Penanggungan, Iyang-Argapura, Baluran**. Kawi-Butak dan
Penanggungan ada persis di konsentrasi situs Malang–Mojokerto.

| Inventaris | Spearman ρ (suitability vs jarak gunung) |
|---|---|
| 7 gunung legacy | −0.243 |
| **13 gunung kanonik** | **−0.281** |

(Naskah melaporkan −0.163 untuk set legacy; selisih dengan −0.243 kami berasal dari model dan frame yang
diimplementasi ulang secara independen — jadi ini koreksi **berarah**, bukan klaim mereproduksi angka
persis mereka.)

**Koreksinya menguatkan korelasi tapi tetap jauh di bawah ambang FAIL 0.5 — vonis GREY_ZONE Test 1
SELAMAT.** Cacatnya nyata dan wajib diungkap; ia tidak membalikkan kesimpulan tautologi. Ini kelas cacat
yang sama dengan yang menjatuhkan P7 di Antiquity.

**Bagian C-2 — permintaan Reviewer 2 (kontrol non-vulkanik) dijawab.**
Dataran tinggi = elevasi ≥ 200 m. Vulkanik = ≤20 km dari pusat kanonik (112.093 sel); non-vulkanik = ≥40 km
(44.495 sel). Coarsened exact matching pada elevation × slope × TRI × TWI (5 bin tiap variabel); 90 dari
100 strata terisi di kedua lengan.

| | dataran tinggi vulkanik | non-vulkanik |
|---|---|---|
| Suitability prediksi (tertimbang matching) | 0.2249 | 0.1702 (**+0.055**) |
| Densitas situs teramati | **0.01377/km²** (145 situs / 10.528 km²) | **0.00048/km²** (2 situs / 4.183 km²) |

Reviewer 2 khawatir model diam-diam mendeteksi efek vulkanik lewat elevasi/slope. **Ternyata tidak** —
model cuma memprediksi selisih +0.055 sementara densitas situs teramati berbeda ~29×. Model terrain tidak
menangkap konsentrasi vulkanik; ia justru sangat kurang memprediksinya.

Memotong dua arah, dan keduanya harus disebut:
- **Baik:** model bukan detektor jarak-gunung terselubung. Keberatan spesifik R2 terjawab negatif, dengan
  desain matching, bukan klaim.
- **Buruk:** apa pun yang sebenarnya menstrukturkan sebaran situs ini, kovariat terrain nyaris tidak
  melihatnya.

**Peringatan yang wajib menempel:** lengan non-vulkanik hanya berisi **2 situs**. Arah (145 vs 2) tak
ambigu; **rasionya rapuh dan tidak boleh dikutip sebagai kelipatan presisi.**

---

## 6. Risiko terbesar bagi arah revisi ini

Reviewer 1 sudah menulis: *"the empirical finding is not entirely novel and is well established in adjacent
fields such as ecological niche modeling."*

Temuan artefak telanjang **memang** dekat dengan Lobo dkk. (2008). Reviewer bisa dengan sah bilang "Anda
menemukan ulang hal yang sudah diketahui". Sampai kemarin saya tidak punya jawaban untuk itu.

**E218b adalah jawabannya.** Lobo bilang AUC tidak bisa dibandingkan lintas sampel background. Temuan kita
lebih tajam: **di ruang desain ini, mengoptimalkan metrik yang dilaporkan secara sistematis memilih model
yang lebih buruk**, dengan dose-response monoton (+0.961 / −0.708) di sepanjang parameter yang praktisi
memang putar. Itu patologi terukur, bukan pengulangan peringatan.

Risiko kedua: reframe metodologis **memperparah** keberatan R2 ("sites function mainly as spatial
observations"). E219 dirancang khusus untuk itu — konsekuensinya arkeologis (ke mana orang dikirim ke
lapangan), bukan statistik.

---

## 7. Cara memverifikasi sendiri

Semua dari root repo (`D:\documents\volcarch-repo`). Butuh `pip install elapid` (sudah terpasang di mesin
Pak Amien; versi 1.0.4).

```bash
# E217 — benchmark MaxEnt + evaluasi dicocokkan   (~10 mnt)
python experiments/E217_maxent_benchmark/01_maxent_benchmark.py
python experiments/E217_maxent_benchmark/02_matched_evaluation.py

# E218 — ketahanan artefak, 4 tahap                (~40 mnt)
python experiments/E218_evaluation_artefact/01_artefact_robustness.py

# E218b — mekanisme, sapuan hard_frac              (~20 mnt)
python experiments/E218_evaluation_artefact/02_mechanism_hardfrac.py

# E219 — divergensi peta + INT-1 + kontrol R2-F    (~25 mnt)
python experiments/E219_map_divergence/01_map_divergence.py
```

### Letak tiap angka

| Angka di dokumen ini | File |
|---|---|
| Reproduksi 0.750 vs 0.751 terbit | `experiments/E217_maxent_benchmark/results/e217b_raw_results.csv` |
| Dekomposisi kenaikan tangga | `.../results/e217b_outcome.json` |
| Matriks 4 evaluation background | `experiments/E218_evaluation_artefact/results/e218_stageA_auc_matrix.csv` |
| Boyce / TSS | `.../results/e218_stageA_{boyce,tss}_matrix.csv` |
| Sensitivitas block size | `.../results/e218_stageB_blocksize.csv` |
| Stage C yang GAGAL (jejak audit) | `.../results/e218_stageC_dissimilarity.csv` |
| Sapuan hard_frac | `.../results/e218b_hardfrac_sweep.csv`, `e218b_summary.csv`, `e218b_outcome.json` |
| Divergensi peta | `experiments/E219_map_divergence/results/e219_map_divergence.csv` |
| Ketidaksepakatan per kuintil jalan | `.../results/e219_disagreement_by_road.csv` |
| Matching vulkanik/non-vulkanik | `.../results/e219_terrain_matched.csv` |
| INT-1 ρ lama vs baru | `.../results/e219_outcome.json` |

### Dokumentasi naratif tiap eksperimen

- `experiments/E217_maxent_benchmark/README.md`
- `experiments/E218_evaluation_artefact/DESIGN.md` (pre-registrasi) + `README.md`
- `experiments/E219_map_divergence/README.md`
- Log riset: `docs/JOURNAL.md`, entri 2026-07-27 (1) sampai (5)

---

## 8. Empat pertanyaan untuk Mas Go Frendi

**Q1 — Setuju klaim inti dicabut?**
Ini pertanyaan utamanya. Kalau Mas melihat cacat di argumen saya, sekaranglah saat mengatakannya — belum
ada apa pun yang dikirim ke editor.

**Q2 — Apakah "common evaluation background" itu tolok ukur yang benar?**
Ini pilihan metodologis paling menentukan di seluruh dokumen. Alasan saya: membandingkan desain background
mengharuskan distribusi test dipatok tetap sementara distribusi latih divariasikan; kalau tidak, perbedaan
AUC mencerminkan komposisi test set, bukan mutu model. Kalau premis itu salah, seluruh refutasi runtuh.
Mas orang ML-nya — saya betul-betul ingin ini ditembak.

**Q3 — Apakah E218b cukup baru untuk melawan komentar "not entirely novel" dari R1?**
Penilaian saya di §6 mungkin terlalu optimistis. Mas lebih sering baca literatur ML; apakah hubungan
terbalik ini sudah pernah didemonstrasikan di tempat lain dengan bentuk sejelas ini?

**Q4 — Kepengarangan.**
Naskah yang akan keluar punya kesimpulan yang **berlawanan** dengan yang Mas ikut tanda tangani di
2026-03-11. Mas berhak: (a) tetap co-author di versi baru, (b) mundur, atau (c) minta perubahan sebelum
setuju. Tidak ada jawaban yang bermasalah — tapi keputusannya harus eksplisit dan tercatat.

---

## 9. Yang TIDAK terbukti — jangan ditulis seolah terbukti

1. **Bahwa peta yang berbeda itu lebih baik.** Tidak ada ground truth untuk mengadili. Naskah harus bilang
   "berbeda dan berkonsekuensi", tidak pernah "membaik".
2. **Bahwa desain background tidak berguna.** Boyce bilang "tidak ada manfaat diskriminasi yang andal";
   E219 bilang petanya berubah banyak. Keduanya klaim berbeda dan keduanya lebih lemah dari "tidak berguna".
3. **Bahwa ketiadaan situs di dataran tinggi non-vulkanik itu nyata.** n=2. Bisa jadi bias survei — dan itu
   justru pertanyaan yang paper ini ada untuk mengajukannya.
4. **Bahwa temuan ini berlaku di luar Jawa Timur atau di luar ruang desain ini.** Belum diuji.
