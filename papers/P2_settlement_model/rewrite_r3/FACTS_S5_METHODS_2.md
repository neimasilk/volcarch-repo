# Sesi 5 — Methods §2.1–2.6 — lembar fakta untuk PI (12–13 Nov 2026)

**Aturan lembar ini (C093, `AGENDA_PI_20261103.md` §5):** hanya butir dan angka, tanpa kalimat siap-salin. Bapak menulis
dengan kata-kata Bapak sendiri, dalam bahasa Indonesia. Saya menerjemahkan tanpa menambah isi; setiap perubahan di luar
terjemahan saya sampaikan dulu.

**Ini sesi yang berbeda dari sesi 2–4.** Metode (± 1.420 kata) **hampir seluruhnya tulisan Claude** (era v0.2 +
suntingan sesi 5–6). Jadi bukan "baca ulang": Bapak **menulis ulang**, dengan suara peneliti yang menjelaskan
keputusan dan prosedurnya sendiri. Yang boleh dipertahankan hanya bagian yang memang tulisan Bapak (lihat §0).

**Prioritas Bapak (11 Nov):** (1) F12 dunia sintetis A–D; (2) C094 tiga tingkat vonis Test 1; (3) pengurangan kata.
Sesi 5 = F12 + C094 + §2.1 + §2.2. Sesi 6 (14 Nov) = §2.3 + §2.4 + §2.5 + sisa §2.6.

---

## 0. Peta authorship §2.1–2.6 (yang Claude tulis = Bapak tulis ulang)

| Bagian | Baris v0.4 | Kata | Penulis teks sekarang |
|---|---|---|---|
| §2.1 Data, covariates, role separation | 184–217 | 321 | **Claude** (v0.2 + paragraf K1 sesi 6) → tulis ulang |
| §2.2 The modelling pipeline | 222–229 | 152 | **Claude** (termasuk H3 "the original pipeline set itself a target") → tulis ulang |
| §2.3 The experiment progression | 234–267 | 244 | Claude, **kecuali ¶ hard negative (l. 265–267) = passage 7 Bapak** → ¶ itu baca ulang, sisanya tulis ulang |
| §2.4 The four roles of road_dist | 272–277 | ± 120 | Claude + **passage 8B Bapak** (l. 274–276, dengan sisipan C1/C2 Claude); kalimat terakhir (l. 277) = Claude → tulis ulang + C094 |
| §2.5 The accessibility-proxy holdout (E014) | 283–284 | 58 | **Passage 8C Bapak** → baca ulang saja |
| §2.6 The refutation design E217–E224 | 289–318 | ± 300 | **Claude** (termasuk R8 "eight experiments…", H4 lattice, dan kalimat "Four synthetic worlds") → tulis ulang, **F12 masuk di sini** |

**Total plafon v0.3 (aturan PI 9 Nov):** §2.1 **321** · §2.2 **152** · §2.3 **244** · §2.4 ± 120 · §2.5 58 · §2.6 ± 300,
total Methods **1.420**. Results sudah +462 terhadap v0.3 → **penghematan di sini yang menutupnya**.

---

## 1. F12 — empat dunia sintetis A–D (prioritas 1). Tidak boleh hilang dari Metode

**Fungsi:** naskah menyebut "World C" dan "World D" di §3.11 dan "four synthetic worlds" di §2.6, tetapi dunia-dunia itu
**tidak dijelaskan di mana pun**. Pembaca (dan R1) tidak bisa menilai klaim tanpa tahu apa yang disimulasikan.

| Hal | Fakta | Sumber |
|---|---|---|
| Panggung | kisi nyata Jawa Timur, kode pipeline yang sama; intensitas diketahui; **10 dunia per tipe**; presensi teramati ± 300–500 per dunia (kekuatan data nyata) | E222 README §Method |
| **A** | intensitas **teramati penuh** (elevation, slope, river, TWI); **bias survei berbentuk jalan** (road-decay, bentuk TGB) diterapkan | idem |
| **B** | A **+ clay**, raster nyata **ditahan dari fitur** → model *misspecified* | idem |
| **C** | A + **usaha survei tidak merata antarkawasan**, bobot [1,0; 0,4; 0,15; 0,05]; share kawasan-0 pada presensi teramati **43 %** | idem |
| **D** | **kebenaran seimbang antarkawasan** (setiap kuadran intensitas sama) + usaha survei regional; share kawasan-0 **62,6 %** → rezim **paling menguntungkan** desain kuota | idem |
| Bias survei jalan | ada di **keempat** dunia — itu yang membuat TGB "model bias yang benar" di A | idem |
| Konfigurasi yang dibandingkan | random, TGB, hybrid pada `hard_frac` ∈ {0,0; 0,3; 0,7; 1,0}; 3 algoritma; hiperparameter E217; fungsi sampling E217 apa adanya | idem |
| Cara menilai "kebenaran" | `auc_true` = AUC terhadap **sampel presensi tak-bias independen** + ketersediaan seragam tetap; pemulihan peta = Jaccard desil teratas + Spearman terhadap permukaan intensitas; plus `auc_own` (angka yang dilaporkan) dan Boyce | idem |
| Jangan dikacaukan | grid hybrid **E013** (pipeline asli) memakai `hard_frac` {0,0; 0,15; 0,30}; grid **E222** memakai {0,0; 0,3; 0,7; 1,0}. Dua grid berbeda | §2.3 vs E222 |

**Yang perlu tersampaikan (± 100–120 kata):** apa yang disimulasikan dan **kenapa**; beda A/B (spesifikasi model), C
(bias regional — dunia yang memotivasi desain kuota), D (kebenaran seimbang — ujian paling lunak bagi kuota); apa yang
sama di keempatnya (bias survei bentuk jalan, intensitas diketahui); bagaimana "benar" diukur.

---

## 2. C094 — Test 1 punya tiga tingkat, bukan satu (prioritas 2)

**Fungsi:** Metode baris 274 sekarang hanya menyebut ambang gagal 0,5. Itu menyembunyikan aturan aslinya dan membuat
"the verdict does not change" di §3.9 tidak bisa diperiksa pembaca.

| Hal | Fakta | Sumber |
|---|---|---|
| Apa yang diuji Test 1 | korelasi Spearman antara **kesesuaian prediksi** dan tiga proksi: jarak ke gunung api, jarak ke jalan, jarak ke situs terdekat | `enhanced_tautology_tests.py` l. 191–196 |
| Statistik vonis | **maksimum \|ρ\| dari ketiga proksi** (bukan tiap proksi sendiri) | idem l. 213–234 |
| **PASS** | maks \|ρ\| **≤ 0,30** | l. 62 (`RHO_WARNING_THRESHOLD = 0.30`), l. 232–234 |
| **GREY_ZONE** | 0,30 < maks \|ρ\| **≤ 0,50** | l. 224–229 |
| **FAIL** | maks \|ρ\| **> 0,50** | l. 216–222 (`RHO_FAIL_THRESHOLD = 0.50`) |
| Vonis Test 1 asli | **GREY_ZONE**, karena **jarak ke jalan** \|ρ\| = **0,307** (jalan satu-satunya proksi di atas 0,30) | v0.1 l. 307; E219 README; C094 |
| Komponen gunung api | ρ = −0,243 (7 pusat, 5 seed) → −0,281 (13 pusat kanonik); keduanya **di bawah 0,30** | E219 `e219_outcome.json`; §3.9 |
| **Test 3** | memotong wilayah menurut kuartil jarak-jalan (Q1 paling mudah dijangkau … Q4 paling sulit) dan menjalankan uji-tahan 50/50 **di dalam tiap kuartil**, dengan latar dari desain yang sama → dinilai pada latar sendiri, jadi hasilnya tidak dipakai di sini | `enhanced_tautology_tests.py` l. 354–420; sesi 5 R11 |

**Yang perlu tersampaikan (± 70–90 kata):** Test 1 mengambil **maksimum** dari tiga proksi; tiga tingkat, bukan satu;
vonis asli **GREY_ZONE** karena jarak-jalan 0,307; komponen gunung api saja di bawah 0,30; Test 3 ada tetapi tidak dipakai
(alasan: dinilai pada latar sendiri). Jangan menulis "Test 1 passes" — tulis vonisnya.

---

## 3. §2.1 Data, covariates, role separation (plafon 321 kata)

| Fakta | Nilai | Sumber |
|---|---|---|
| Presensi terpakai setelah filter kovariat | **378** (111–115° BT); subset tutupan tanah E009 = 259 | E217 `e217_outcome.json` `n_presences` |
| Sumber | OpenStreetMap **280**, Wikipedia id **71**, Wikipedia en **16**, Wikidata **16** (dalam bingkai; setelah filter kovariat menjadi 279 / 85 / 14) | `east_java_sites.geojson`, dihitung 12 Nov |
| Koordinat berbeda | **335** (dari 378; 43 baris berbagi koordinat, semuanya baris hasil geocoding nama) | idem; R4 tidak membuang baris lagi setelah R1–R3 |
| Bukan situs | **124** monumen/patung/memorial modern; **13** salah tempat atau bukan situs (candi Jawa Tengah dengan koordinat Jawa Timur: Borobudur/Prambanan 111,09° BT 8,20° LS; pura Bali dekat Surabaya; satu air terjun) | C081 (audit 26 Okt), E236 DESIGN §3 |
| Aturan pembersihan (R1–R4) | R1 buang baris **yang koordinatnya hasil geocoding nama** (semua dari daftar Wikipedia) = **85**; R2 monumen modern = **124**; R3 candi daerah lain, objek wisata, institusi modern = **8**; R4 satu baris per koordinat = **0** tambahan | `exp/E236.../clean_sites.py`; saya reproduksi 12 Nov: 85 + 124 + 8 = **217**, sisa **161** (OSM 155 + Wikidata 6) |
| Catatan hitung | 85/124/8 adalah **hitungan eksklusif** (R1 didahulukan). Di luar bingkai penuh aturan yang sama mengenai 87/125/10 dari 383 baris → jangan menjumlahkan dua kali | idem |
| Tetap di 378 | tangga E007–E013 (objek yang diperiksa) tetap dilatih pada 378; kontras §3.2 dan §3.3 dilaporkan pada kedua set | Methods l. 189; K1 |

**Tabel peran kovariat (Tabel 2) — angka yang harus tetap benar:**

| Variabel | Sumber | Peran | Masuk sebagai |
|---|---|---|---|
| elevation, slope, TWI, TRI, aspect | Copernicus GLO-30 DEM 30 m, diperkecil ×10 (≈ 300 m) | kesesuaian (medan) | fitur latihan |
| river_dist | OSM/HydroSHEDS | kesesuaian (air); **ditambahkan di E008** | fitur latihan |
| clay, silt | SoilGrids topsoil | preservasi/substrat | **hanya E009**; dibuang (menurunkan AUC) |
| road_dist | OSM, kelas jalan diperluas di E012 | aksesibilitas/usaha survei | **tidak pernah fitur latihan** |
| volcano_dist | inventaris kanonik, 30 pusat (13 dalam bingkai) | tafonomi/penguburan — diagnostik | **tidak pernah fitur latihan** |

**Prinsip pemisahan peran:** hanya blok kesesuaian menjadi prediktor; aksesibilitas diungkapkan lewat **desain latar**;
preservasi dan kedekatan gunung api hanya **diagnostik pada keluaran**. Alasannya: kalau dicampur, skor rendah tidak bisa
ditafsirkan (tidak cocok? tidak disurvei? tidak terawetkan?).

---

## 4. §2.2 The modelling pipeline (plafon 152 kata)

| Fakta | Nilai | Sumber |
|---|---|---|
| Rasio presensi : latar | **1 : 5** sepanjang penelitian (378 presensi → 1.890 latar) | E217 `e217_outcome.json`; kode `n_background_target` |
| Tiga learner | **XGBoost (utama)**, RandomForest, MaxEnt | §2.2; kode |
| XGBoost | `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `scale_pos_weight` seimbang | `01_maxent_benchmark.py` l. 275–280 |
| RandomForest | `n_estimators=200`, `max_depth=6`, `class_weight="balanced"` | l. 282–284 |
| MaxEnt | implementasi `elapid` dari maxnet; fitur **linear, hinge, product**; **β = 1,5**; transformasi **cloglog**; β disapu 0,5–4,0 di §3.2 | l. 267–270 |
| Validasi | **5-fold**, blok spasial deterministik, ukuran blok dasar **0,45°** (≈ 50 km) | E217 `cv`; `BLOCK_SIZE_DEG` |
| Metrik | **AUC** dan **TSS** (TSS dipilih daripada kappa: tidak bergantung prevalensi seperti kappa) | Allouche 2006 |
| Sifat yang menjadi pokok makalah | keduanya metrik diskriminasi yang dihitung **terhadap latar apa pun yang diberikan** — itulah objek yang diperiksa | Lobo 2008; Guillera-Arroita 2015 |
| Target pipeline asli | AUC spasial **> 0,75** di bawah blok CV | H3; E013 |

---

## 5. §2.3 (sesi 6, tetapi angkanya di sini) — ringkas

Tangga E007–E013 (Tabel 4, sudah G1): 0,659 · 0,695 · 0,664 · 0,711 · 0,725 · 0,730 · 0,768 (terbaik) / 0,751
(rata-rata seed). River **+0,036** (E007→E008); tanah **−0,031** (E009). Grid hybrid E013: `region_blend` ∈
{0,0; 0,3; 0,5; 0,7} × `hard_frac` ∈ {0,0; 0,15; 0,30}. Paragraf hard negative = **passage 7 Bapak** (0,62 terealisasi;
TGB 0,485 vs acak 0,503 — C080, sudah Bapak tulis ulang 22 Okt).

---

## 6. Anggaran kata dan di mana penghematan bisa datang

| Bagian | v0.3 | Usulan |
|---|---|---|
| §2.1 | 321 | kompresi ¶ sumber+aturan bersih (± 110 → ± 75) dan ¶ "This separation is not a technicality" (± 45 → ± 25) |
| §2.2 | 152 | hampir tidak bisa dipotong (parameter); ¶ metrik bisa dipadatkan sedikit |
| §2.3 | 244 | "Two paragraphs of interpretation." (3 kata, kalimat kosong) dibuang; ¶ tangga bisa dipadatkan |
| §2.4 | ± 120 | kalimat penutup "This is also what makes the approach specifically archaeological…" (± 35) mengulang §1.3 → kandidat buang |
| §2.5 | 58 | sudah tulisan Bapak; biarkan |
| §2.6 | ± 300 | ¶ lattice (± 90 → ± 60); paragraf E217–E224 dipadatkan |

F12 (+110) dan C094 (+25 dibanding kalimat lamanya) menambah kata; penghematan harus ≥ itu supaya Metode tetap ≤ 1.420.

---

## 7. Jangan ditulis

- **"pre-registered" untuk E217/E219** (tidak ada desain tertulis) atau untuk E218/E220–E223 (desain dan hasil di-commit
  bersama). **Hanya E224 dan E235** yang sah dipra-registrasi (C082). Kalimat R8 sudah benar, pertahankan faktanya.
- **"Test 1 passes"** — vonisnya GREY_ZONE (C094).
- **"the background designs are pre-registered"** atau "the ladder was pre-registered" — tidak.
- **124** monumen modern dan **13** salah tempat dipakai seolah kategori yang sama dengan 85/8 pada aturan bersih;
  keduanya menjawab pertanyaan berbeda (komposisi record vs yang dibuang R1–R4).
- "TGB lebih memilih titik sulit" — angka C080: TGB 0,485 **lebih rendah** dari acak 0,503; 0,62 muncul dari komposisi pool.
- Angka F12 disamakan dengan grid E013 (dua grid berbeda).
- "E217–E224 memakai sampling yang sama dengan E007–E013" — tidak; lattice ≈ 300 m (H4), dan itu disebut sebagai deviasi.

## 8. Sumber yang dipakai (12 Nov)

E217 `results/e217_outcome.json`, `01_maxent_benchmark.py` (l. 128–175, 267–284) · `enhanced_tautology_tests.py` (l. 62–66,
191–234, 354–420) · `experiments/E236_p2_presence_sensitivity/clean_sites.py`, `README.md`, `DESIGN.md` ·
`experiments/E014_temporal_validation/HANDOFF_E014.md` (AUC 0,755) · `experiments/E222_synthetic_ground_truth/README.md`,
`DESIGN.md` · `data/processed/east_java_sites.geojson` (dihitung ulang 12 Nov: 383 baris dalam bingkai → 378 setelah filter
kovariat; R1 85/87, R2 124/125, R3 8/10; sisa 161 = 155 OSM + 6 Wikidata) · `PI_PROSE_LOG_R2.md` passage 7 dan 8B–8C ·
ledger C080, C081, C082, C094, K1 · `K1_WRITING_PACK.md` §Methods 2.1.
