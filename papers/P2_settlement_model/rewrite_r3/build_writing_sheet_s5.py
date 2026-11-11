# Builds the P2 author-rewrite writing sheet for session 5 (Methods 2.1-2.2, plus F12 and C094) as a .docx.
# Content rule (C093): bullets, numbers and questions only -- no prose the PI could copy.
# Session 5 is a REWRITE session: almost all of Methods is Claude's text, so the sheet asks the PI to explain the
# decisions and procedures in his own voice, not to polish the old sentences.
# Facts and sources: FACTS_S5_METHODS_2.md (all numbers re-checked 2026-11-12).
#   python papers/P2_settlement_model/rewrite_r3/build_writing_sheet_s5.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheet_lib import Sheet  # noqa: E402

OUT = Path(__file__).resolve().parent / "LEMBAR_TULIS_S5_20261112.docx"

s = Sheet()

# ================================================================ 1. COVER + TASK
s.cover("LEMBAR TULIS — SESI 5",
        "P2 / JCAA #280 — naskah pengganti, penulisan oleh penulis\nMetode §2.1–2.2 + F12 (dunia A–D) + C094 (Test 1) · 12–13 November 2026")

s.h("Sesi ini menulis, bukan memoles: ± 85 % Metode adalah teks Claude", level=1)
s.para("Gaya yang dituju: peneliti yang menjelaskan keputusan dan prosedurnya sendiri — kenapa begini, apa yang "
       "dikorbankan, apa yang tidak dilakukan. Bukan dokumentasi teknis.", italic=True)

s.task_card(
    "1", "F12 — Empat dunia sintetis A–D (prioritas Bapak #1)", "± 110 kata — satu paragraf baru di §2.6",
    [
        "Apa yang disimulasikan, dan kenapa ground truth perlu dibuat?",
        "Dunia A: apa yang teramati penuh, dan bias survei apa yang diterapkan?",
        "Dunia B: apa yang ditambahkan dan di mana spesifikasinya gagal (dan kenapa itu penting)?",
        "Dunia C: apa yang berubah — dan kenapa dunia ini yang memotivasi desain kuota?",
        "Dunia D: kenapa disebut rezim paling menguntungkan bagi kuota?",
        "Apa yang sama di keempat dunia, dan bagaimana \"benar\" diukur?",
    ],
    "halaman 2",
)
s.task_card(
    "2", "C094 — Test 1: tiga tingkat vonis (prioritas Bapak #2)", "± 70–90 kata — ganti dua kalimat di §2.4",
    [
        "Test 1 menguji apa, terhadap proksi apa saja?",
        "Vonis diambil dari statistik apa (maksimum ketiga proksi?), dan apa tiga tingkatnya?",
        "Berapa vonis aslinya, dan proksi mana yang menyebabkannya?",
        "Kenapa Test 3 ada tetapi hasilnya tidak dipakai?",
    ],
    "halaman 3",
)
s.task_card(
    "3", "§2.1  Data, covariates, and the principle of role separation", "plafon 321 kata",
    [
        "Apa record 378 itu, dari mana asalnya, dan kenapa komposisinya harus diakui terbuka?",
        "Apa aturan pembersihan (R1–R4), apa yang dibuang, dan berapa yang tersisa?",
        "Kenapa tangga tetap dilatih pada 378 sedangkan kontras dilaporkan pada dua set?",
        "Kenapa setiap variabel hanya boleh punya satu peran — dan di mana aksesibilitas, preservasi, dan gunung api masuk?",
    ],
    "halaman 4",
)
s.task_card(
    "4", "§2.2  The modelling pipeline", "plafon 152 kata",
    [
        "Desain pemodelan: rasio latar, tiga learner, siapa yang utama?",
        "MaxEnt dikonfigurasi bagaimana, dan mengapa parameternya perlu disebut?",
        "Validasi: blok spasial seperti apa?",
        "Metrik apa yang dipakai, kenapa TSS dan bukan kappa?",
        "Kenapa metrik ini justru menjadi pokok makalah?",
    ],
    "halaman 5",
)

s.table(
    [
        ["Cara pakai", ""],
        ["1", "Buka halaman detail bagian yang mau ditulis. Baca tabel angka dan butir \"yang perlu tersampaikan\"."],
        ["2", "Tulis di kotak abu-abu, bahasa Indonesia, kata-kata sendiri. Tidak perlu kalimat jadi."],
        ["3", "Kalau ada angka atau metode yang belum jelas, tanya di chat — saya jelaskan, bukan untuk disalin."],
        ["4", "Kirim balik berkasnya, atau tempel teksnya di chat. Boleh satu bagian per pesan."],
        ["5", "Saya terjemahkan, tunjukkan hasilnya, baru masukkan ke naskah. Setiap perubahan di luar terjemahan minta izin dulu."],
    ],
    widths=[1.2, 16.0],
)
s.para("")
s.para("Panjang: Metode 1.420 kata di v0.3 dan v0.4. Results sudah +462 terhadap v0.3; penghematan yang menutupnya dicari "
       "di Metode dan Diskusi. F12 menambah ± 110 kata dan C094 ± 25, jadi pemadatan lain harus menutupinya. Usulan "
       "pemadatan ada di halaman 6.", italic=True, size=9.5)
s.para("Yang bukan tulisan Bapak di Metode: §2.1, §2.2, §2.6 seluruhnya; §2.3 kecuali paragraf hard negative; §2.4 kecuali "
       "kalimat-kalimat passage 8B. Yang tulisan Bapak: paragraf hard negative (passage 7), bagian besar §2.4 (passage 8B), "
       "dan seluruh §2.5 (passage 8C) — tiga itu cukup dibaca ulang di sesi 6.", italic=True, size=9.5)

# ================================================================ 2. F12
s.page_break()
s.h("F12 — Empat dunia sintetis A–D  ·  ± 110 kata, satu paragraf baru di §2.6", level=1)
s.para("Fungsi: §3.11 menyebut World C dan World D, dan §2.6 hanya menulis \"four synthetic worlds of known intensity "
       "surface\". Tanpa paragraf ini pembaca tidak bisa menilai klaimnya — dan ini salah satu hal yang diminta R1 "
       "(naskah berdiri sendiri).", italic=True)
s.table(
    [
        ["Hal", "Fakta", "Sumber"],
        ["Panggung", "kisi nyata Jawa Timur, kode pipeline yang sama, intensitas diketahui, 10 dunia per tipe, presensi teramati ± 300–500", "E222 README"],
        ["Dunia A", "intensitas teramati penuh (elevation, slope, river, TWI) + bias survei berbentuk jalan (road-decay, bentuk TGB)", "idem"],
        ["Dunia B", "A + clay, raster nyata ditahan dari fitur → model misspecified", "idem"],
        ["Dunia C", "A + usaha survei tidak merata antarkawasan, bobot [1,0; 0,4; 0,15; 0,05]; share kawasan-0 pada presensi teramati 43 %", "idem"],
        ["Dunia D", "kebenaran seimbang antarkawasan + usaha survei regional; share kawasan-0 62,6 % → rezim paling menguntungkan kuota", "idem"],
        ["Sama di keempatnya", "bias survei jalan (itulah yang membuat TGB model bias yang benar di A)", "idem"],
        ["Konfigurasi", "random, TGB, hybrid pada hard_frac {0,0; 0,3; 0,7; 1,0}; 3 algoritma; hiperparameter dan fungsi sampling E217 apa adanya", "idem"],
        ["Mengukur \"benar\"", "auc_true = AUC terhadap sampel presensi tak-bias independen + ketersediaan seragam; pemulihan peta = Jaccard desil teratas + Spearman vs permukaan intensitas; plus auc_own dan Boyce", "idem"],
        ["Jangan dikacaukan", "grid hybrid E013 memakai hard_frac {0,0; 0,15; 0,30}; grid E222 {0,0; 0,3; 0,7; 1,0}", "§2.3 vs E222"],
    ],
    widths=[3.4, 9.6, 3.4],
)
s.para("Yang perlu tersampaikan: apa yang disimulasikan dan kenapa; apa yang berbeda di A/B/C/D; apa yang sama di "
       "keempatnya; bagaimana \"benar\" diukur.", bold=True, space_before=6)
s.writebox(lines=12, label="TULIS PARAGRAF F12 DI SINI — ± 110 kata")

# ================================================================ 3. C094
s.page_break()
s.h("C094 — Test 1 punya tiga tingkat  ·  ganti dua kalimat di §2.4 (Tempat 8B Bapak)", level=1)
s.para("Fungsi: Metode sekarang hanya menyebut ambang gagal 0,5. Itu menyembunyikan aturan aslinya, dan membuat kalimat "
       "§3.9 (\"the verdict does not change\") tidak bisa diperiksa pembaca.", italic=True)
s.table(
    [
        ["Hal", "Fakta", "Sumber"],
        ["Yang diuji Test 1", "korelasi Spearman antara kesesuaian prediksi dan tiga proksi: jarak ke gunung api, jarak ke jalan, jarak ke situs terdekat", "enhanced_tautology_tests.py l. 191–196"],
        ["Statistik vonis", "maksimum |ρ| dari ketiga proksi", "l. 213–234"],
        ["PASS", "maks |ρ| ≤ 0,30", "l. 62, 232–234"],
        ["GREY_ZONE", "0,30 < maks |ρ| ≤ 0,50", "l. 224–229"],
        ["FAIL", "maks |ρ| > 0,50", "l. 216–222"],
        ["Vonis Test 1 asli", "GREY_ZONE", "v0.1 l. 307; E219; C094"],
        ["Penyebabnya", "jarak ke jalan |ρ| = 0,307 — satu-satunya proksi di atas 0,30", "idem"],
        ["Komponen gunung api", "−0,243 (7 pusat, 5 seed) dan −0,281 (13 pusat kanonik) — keduanya di bawah 0,30", "E219; §3.9"],
        ["Test 3", "kuartil jarak-jalan (Q1 paling mudah … Q4 paling sulit), uji-tahan 50/50 di dalam tiap kuartil, latar dari desain yang sama → dinilai pada latar sendiri → tidak dipakai", "l. 354–420; sesi 5 R11"],
    ],
    widths=[3.4, 9.6, 3.4],
)
s.bullets(
    "Test 1 mengambil MAKSIMUM dari tiga proksi, bukan tiap proksi sendiri.",
    "Jangan menulis \"Test 1 passes\". Tulis vonisnya: GREY_ZONE, dan angkanya.",
    "Kalimat passage 8B Bapak yang lain (kenapa road_dist dibahas di Metode) tetap dipakai.",
)
s.writebox(lines=10, label="TULIS KALIMAT TEST 1 DI SINI — ± 70–90 kata (boleh 3–4 kalimat)")

# ================================================================ 4. Section 2.1
s.page_break()
s.h("§2.1  Data, covariates, and the principle of role separation  ·  plafon 321 kata", level=1)
s.para("Fungsi: menyatakan apa record itu, apa yang dibuang dan kenapa, lalu prinsip bahwa setiap variabel hanya punya "
       "satu peran. Prinsip ini bukan formalitas — ia yang membuat skor kesesuaian bisa ditafsirkan.", italic=True)
s.h("A. Record dan komposisinya", level=2)
s.table(
    [
        ["Fakta", "Nilai", "Sumber"],
        ["Presensi terpakai setelah filter kovariat", "378 (111–115° BT); subset tutupan tanah E009 = 259", "E217 e217_outcome.json"],
        ["Sumber (dalam bingkai)", "OpenStreetMap 280 · Wikipedia id 71 · Wikipedia en 16 · Wikidata 16", "east_java_sites.geojson, dihitung 12 Nov"],
        ["Koordinat berbeda", "335 dari 378 (43 baris berbagi koordinat; semuanya hasil geocoding nama)", "idem"],
        ["Bukan situs", "124 monumen/patung/memorial modern · 13 salah tempat atau bukan situs", "C081, E236 DESIGN §3"],
        ["Contoh yang salah tempat", "Borobudur/Prambanan di koordinat Jawa Timur (111,09° BT 8,20° LS), pura Bali dekat Surabaya, satu air terjun", "C081"],
    ],
    widths=[4.6, 9.4, 3.4],
)
s.h("B. Aturan pembersihan dan hasilnya", level=2)
s.table(
    [
        ["Aturan", "Membuang", "Jumlah"],
        ["R1", "baris yang koordinatnya hasil geocoding nama (semua dari daftar Wikipedia)", "85"],
        ["R2", "monumen/patung/memorial modern (kecuali nama yang jelas kuno: candi, prasasti, …)", "124"],
        ["R3", "candi daerah lain, objek wisata, institusi modern (mis. pura, ponpes)", "8"],
        ["R4", "satu baris per koordinat", "0 tambahan"],
        ["Sisa", "161 situs: OpenStreetMap 155 + Wikidata 6", "—"],
    ],
    widths=[2.6, 11.4, 3.4],
)
s.bullets(
    "85 + 124 + 8 = 217; 378 − 217 = 161. Hitungan eksklusif (R1 didahulukan) — jangan dijumlahkan dua kali.",
    "Tangga E007–E013 tetap dilatih pada 378 (objek yang diperiksa); kontras §3.2 dan §3.3 dilaporkan pada kedua set.",
    "Kalau Bapak ingin lebih pendek: ¶ sumber+aturan sekarang ± 110 kata; cukup menjelaskan mengapa R1 ada "
    "(geocoding nama menghasilkan titik yang bukan findspot) dan bahwa sisa 161 situs adalah set bersih K1.",
)
s.h("C. Peran variabel (Tabel 2 tetap dipakai; angkanya harus benar)", level=2)
s.table(
    [
        ["Variabel", "Sumber", "Peran", "Masuk sebagai"],
        ["elevation, slope, TWI, TRI, aspect", "Copernicus GLO-30, 30 m, diperkecil ×10 (≈ 300 m)", "kesesuaian (medan)", "fitur latihan"],
        ["river_dist", "OSM/HydroSHEDS", "kesesuaian (air); ditambahkan di E008", "fitur latihan"],
        ["clay, silt", "SoilGrids topsoil", "preservasi/substrat", "hanya E009; dibuang (menurunkan AUC)"],
        ["road_dist", "OSM, kelas diperluas di E012", "aksesibilitas/usaha survei", "tidak pernah fitur"],
        ["volcano_dist", "inventaris kanonik, 30 pusat (13 dalam bingkai)", "tafonomi/penguburan — diagnostik", "tidak pernah fitur"],
    ],
    widths=[4.2, 5.4, 4.2, 3.6],
)
s.bullets(
    "Pertanyaan yang dijawab tabel ini: kesesuaian = \"apakah orang akan menetap di sini?\"; aksesibilitas = \"apakah kita "
    "akan menemukannya?\"; preservasi = \"apakah ia masih ada untuk ditemukan?\".",
    "Kenapa tidak boleh dicampur: skor rendah jadi ambigu — tidak cocok, tidak disurvei, atau tidak terawetkan.",
)
s.writebox(lines=16, label="TULIS §2.1 DI SINI — plafon 321 kata (boleh lebih pendek)")

# ================================================================ 5. Section 2.2
s.page_break()
s.h("§2.2  The modelling pipeline  ·  plafon 152 kata", level=1)
s.para("Fungsi: menyatakan desain pemodelan secukupnya agar pembaca bisa menilai, dan menutup dengan sifat metrik yang "
       "menjadi pokok makalah.", italic=True)
s.table(
    [
        ["Fakta", "Nilai", "Sumber"],
        ["Rasio presensi : latar", "1 : 5 (378 → 1.890 latar)", "E217 e217_outcome.json"],
        ["Learner", "XGBoost (utama) · RandomForest · MaxEnt", "§2.2"],
        ["XGBoost", "300 pohon, kedalaman 4, laju 0,05, bobot kelas seimbang", "01_maxent_benchmark.py l. 275–280"],
        ["RandomForest", "200 pohon, kedalaman 6, class_weight balanced", "l. 282–284"],
        ["MaxEnt", "elapid/maxnet; fitur linear, hinge, product; β = 1,5; cloglog; β disapu 0,5–4,0 di §3.2", "l. 267–270"],
        ["Validasi", "5-fold, blok spasial deterministik, blok dasar 0,45° (≈ 50 km)", "E217 cv; BLOCK_SIZE_DEG"],
        ["Metrik", "AUC dan TSS; TSS dipilih daripada kappa (tidak bergantung prevalensi seperti kappa)", "Allouche 2006"],
        ["Sifat metrik", "keduanya dihitung terhadap latar apa pun yang diberikan — itulah yang diperiksa makalah", "Lobo 2008; Guillera-Arroita 2015"],
        ["Target pipeline asli", "AUC spasial > 0,75 di bawah blok CV", "E013"],
    ],
    widths=[3.6, 9.8, 3.0],
)
s.bullets(
    "Kalimat penutup yang paling penting: metrik diskriminasi tidak punya arti tanpa menyebut latar evaluasinya — itu "
    "yang membuat §3 masuk akal.",
    "Kalau memadatkan: ¶ metrik (AUC/TSS) bisa digabung dengan ¶ sifat metrik.",
)
s.writebox(lines=12, label="TULIS §2.2 DI SINI — plafon 152 kata")

# ================================================================ 6. Budget + next + rules
s.page_break()
s.h("Anggaran kata dan di mana penghematan bisa datang", level=1)
s.table(
    [
        ["Bagian", "v0.3", "Usulan"],
        ["§2.1", "321", "¶ sumber+aturan ± 110 → ± 75; ¶ \"This separation is not a technicality\" ± 45 → ± 25"],
        ["§2.2", "152", "¶ metrik digabung dengan ¶ sifat metrik"],
        ["§2.3", "244", "\"Two paragraphs of interpretation.\" (kalimat kosong) dibuang; ¶ tangga dipadatkan"],
        ["§2.4", "± 120", "kalimat penutup \"This is also what makes the approach specifically archaeological…\" (± 35) mengulang §1.3"],
        ["§2.5", "58", "tulisan Bapak; biarkan"],
        ["§2.6", "± 300", "¶ lattice ± 90 → ± 60; paragraf E217–E224 dipadatkan"],
        ["F12 + C094", "+135", "harus ditutup oleh penghematan di atas"],
    ],
    widths=[3.4, 2.4, 12.4],
)

s.h("Sesi 6 (14 Nov) — yang tersisa", level=1)
s.table(
    [
        ["Bagian", "Kata", "Catatan"],
        ["§2.3 The experiment progression", "244", "¶ hard negative = passage 7 Bapak (baca ulang); sisanya tulis ulang. Tabel 4 tetap"],
        ["§2.4 The four roles of road_dist", "± 120", "passage 8B Bapak + C094 (dikerjakan sesi ini)"],
        ["§2.5 The accessibility-proxy holdout (E014)", "58", "passage 8C Bapak — cukup baca ulang"],
        ["§2.6 E217–E224 paragraphs", "± 200", "tulis ulang; F12 masuk di sini. Faktanya: E217 common-background benchmark; E218 artefak terisolasi; E220 tidak ada optimum interior; E221 stabilitas seed; E222 ground truth; E223 CI + sapuan β; E224 penjelasan TGB diuji lalu gagal (null unexplained)"],
    ],
    widths=[5.6, 1.8, 10.8],
)

s.h("Aturan main sesi ini", level=1)
s.bullets(
    "**Bapak menulis dalam bahasa Indonesia**, dengan kata-kata Bapak sendiri — tanpa alat AI lain untuk menyusun atau merapikan teks (aturan editor JCAA).",
    "**Suara peneliti, bukan dokumentasi.** Yang dibaca pembaca: keputusan, alasan, dan apa yang dikorbankan.",
    "**Angka di lembar ini sudah dicek ulang** dari berkas sumbernya (12 Nov). Pakai angka seperti tertulis.",
    "**Saya menerjemahkan tanpa menambah isi.** Perubahan di luar terjemahan saya tanyakan dulu dan saya catat di PI_PROSE_LOG_R3.md.",
    "**Boleh bertanya di chat** soal metode atau statistik. Penjelasan itu untuk dipahami, bukan disalin.",
    "**Naskah berdiri sendiri:** tanpa \"reviewer\", \"revisi\", \"versi sebelumnya\". Tulis \"Section\", bukan \"§\".",
)

s.h("Jangan ditulis (sudah terbukti salah atau terlalu kuat)", level=1)
s.bullets(
    "\"pre-registered\" untuk E217/E219 (tidak ada desain tertulis) atau E218/E220–E223 (desain dan hasil di-commit bersama). Hanya E224 dan E235 yang sah (C082).",
    "\"Test 1 passes\" — vonisnya GREY_ZONE, penyebabnya jarak-jalan 0,307 (C094).",
    "\"the ladder was pre-registered\" — tidak.",
    "\"TGB memilih titik sulit\" — TGB 0,485 lebih rendah dari acak 0,503; 0,62 berasal dari komposisi pool (C080).",
    "\"E217–E224 memakai sampling yang sama dengan E007–E013\" — tidak; lattice ≈ 300 m, dan itu disebut sebagai deviasi (H4).",
    "Angka dunia sintetis disamakan dengan grid E013 — dua grid berbeda.",
    "\"selalu\" / \"tidak pernah\" tanpa pecahan di sebelahnya.",
)

s.save(OUT)
