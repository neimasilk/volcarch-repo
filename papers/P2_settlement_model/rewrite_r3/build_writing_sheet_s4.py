# Builds the P2 author-rewrite writing sheet for session 4 (Results 3.10-3.11) as a .docx.
# Content rule (C093): bullets, numbers and questions only -- no prose the PI could copy.
# The Indonesian passage quoted on the sheet is the PI's own text (R2 passage 2), shown for re-reading.
# Layout rule (from session 1): the first page states WHAT TO WRITE; the facts come after; the rules at the end.
# Facts and sources: FACTS_S4_RESULTS_3_10-3_11.md (all numbers re-checked 2026-11-11).
#   python papers/P2_settlement_model/rewrite_r3/build_writing_sheet_s4.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheet_lib import Sheet  # noqa: E402

OUT = Path(__file__).resolve().parent / "LEMBAR_TULIS_S4_20261111.docx"

s = Sheet()

# ================================================================ 1. COVER + TASK
s.cover("LEMBAR TULIS — SESI 4",
        "P2 / JCAA #280 — naskah pengganti, penulisan oleh penulis\nHasil penelitian §3.10–3.11 · 11 November 2026")

s.h("Sesi ini pendek: sebagian besar membaca ulang tulisan Bapak sendiri", level=1)

s.task_card(
    "1", "§3.10 ¶1–¶2  Volcanic setting or elevation? — BACA ULANG", "tulisan Bapak 21 Okt (325 kata)",
    [
        "Tujuh kalimat pertama (pertanyaan, metode, definisi zona, CEM, peta seed, bootstrap, label vonis): dipakai?",
        "Lima kalimat tumpang tindih di atas 1.000 m (3.435 km² / 97,5 % / 0,4 % / 15 km² / batas 800–1.200 m): dipakai?",
        "Angka di naskah sudah dibulatkan ke angka Gambar 7A (3.434,5 → 3.435; 97,48 → 97,5; 0,435 → 0,4) — setuju?",
    ],
    "halaman 2",
)
s.task_card(
    "2", "§3.10 ¶3  kalimat sensitivitas + post hoc — TULIS ULANG (kalimat Claude)", "± 80 kata (kalimat lama 79)",
    [
        "Dua definisi pra-registrasi memberi apa? (−0,002 NOT SPECIFIC; +0,016 INCONCLUSIVE)",
        "Varian post hoc bergerak dari +0,005 sampai +0,07 — apa akibatnya pada klaim?",
        "Apakah arah (+/−) boleh diklaim di bawah 1.000 m? Kenapa tidak?",
    ],
    "halaman 3",
)
s.task_card(
    "3", "§3.10 ¶3  tiga kalimat Bapak dengan sisipan Claude — BACA ULANG", "Pegunungan Selatan · fringe 1.107 m · daya uji",
    [
        "Klausa \"In the uplands (400–1.000 m),\" (F28) dan rujukan Section 3.6 (F29): dipertahankan?",
        "Kalimat daya uji: sudah menyebut objeknya (perbedaan antardesain di dekat vs jauh)?",
    ],
    "halaman 4",
)
s.task_card(
    "4", "§3.11  dua kalimat + satu keputusan", "± 70 kata (17 + 51)",
    [
        "Ganti kalimat Claude (H20): kesesuaian rendah dekat gunung api = bukti permukiman terkubur, atau bukan?",
        "Kalimat desain kuota (teks v0.2, bukan tulisan Bapak): tulis ulang dengan angka yang sama, atau pakai kalimat lama (F26)?",
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
s.para("Panjang: plafon = panjang v0.3 (keputusan Bapak 9 Nov). §3.10 sekarang 582/582 kata dan §3.11 86/86 — keduanya "
       "TEPAT di plafon. Tulisan baru harus sepanjang yang diganti; tidak ada kata tambahan. Kebutuhan ± 150 kata baru "
       "(kalimat 15 — 79; kalimat 20 — 17; kalimat 21 — 51).", italic=True, size=9.5)
s.para("Sudah Bapak tulis dan tidak perlu diulang: seluruh §3.10 ¶1–¶2 (21 Okt), kalimat Pegunungan Selatan, fringe, dan "
       "daya uji (22 Okt), kalimat pertama §3.11 (passage 8E). Angka set bersih E235 sudah ada di §3.3 — jangan diulang.",
       italic=True, size=9.5)

# ================================================================ 2. Section 3.10 p1-p2
s.page_break()
s.h("§3.10 ¶1–¶2  —  tulisan Bapak 21 Okt, mohon dibaca ulang", level=1)
s.para("Fungsi: menjawab pertanyaan R2 round 2 — apakah model menunjuk lingkungan vulkanik atau sekadar dataran tinggi. "
       "Jawabannya: di atas ± 1.000 m keduanya tidak bisa dipisahkan DI WILAYAH STUDI INI; di bawah 1.000 m tidak ada "
       "yang tegas.", italic=True)
s.para("Teks Bapak (21 Okt, bahasa Indonesia asli; di naskah sudah diterjemahkan):", bold=True, space_before=6)
s.para(
    "Analisis ini menguji apakah pola kesesuaian yang teramati mencerminkan lingkungan vulkanik atau sekadar pengaruh "
    "ketinggian. Informasi jarak ke pusat vulkanik (volcano_dist) tidak digunakan sebagai fitur pelatihan model, sehingga "
    "setiap pola yang berkaitan dengan lingkungan vulkanik dalam prediksi model harus tercermin melalui variabel medan "
    "yang digunakan. Wilayah dekat didefinisikan sebagai area berjarak ≤ 20 km dari pusat vulkanik Kuarter, sedangkan "
    "wilayah jauh berjarak ≥ 40 km; analisis mencakup 30 pusat vulkanik kanonik, termasuk yang berada di luar bingkai "
    "peta. Perbandingan dilakukan pada setiap pita ketinggian 200 m dengan pencocokan tepat bertingkat (coarsened exact "
    "matching) berdasarkan kemiringan lereng (slope), indeks kekasaran medan (TRI), dan indeks kebasahan topografi (TWI). "
    "Analisis menggunakan peta ensemble dari 10 seed yang telah tersimpan tanpa melatih ulang model. Ketidakpastian "
    "diestimasi menggunakan bootstrap blok spasial sebanyak 1.000 ulangan, dengan ambang keputusan 0,05 yang ditetapkan "
    "sebelum analisis dan desain yang dipra-registrasi di repositori publik.",
    italic=True, size=10)
s.para(
    "Di atas ketinggian 1.000 m, wilayah studi mencakup sekitar 3.434,5 km² daratan, dengan 97,48% berada dalam jarak "
    "≤ 20 km dari pusat vulkanik Kuarter dan hanya 0,435% berada pada jarak ≥ 40 km. Daratan yang jauh dari pusat "
    "vulkanik Kuarter hanya mencakup sekitar 15 km² pada ketinggian 1.000–1.200 m dan tidak ditemukan di atas 1.200 m. "
    "Kondisi ini menunjukkan bahwa, di wilayah studi ini, pengaruh ketinggian dan kedekatan dengan pusat vulkanik Kuarter "
    "tidak dapat dipisahkan secara memadai pada ketinggian di atas sekitar 1.000 m. Oleh karena itu, penelitian ini tidak "
    "mengatribusikan pola tersebut secara khusus kepada ketinggian maupun lingkungan vulkanik. Batas keterpisahan ini "
    "dapat bergeser antara 800 dan 1.200 m, bergantung pada kriteria luas minimum yang digunakan.",
    italic=True, size=10)
s.para("(Di naskah: 3.434,5 → 3.435; 97,48 → 97,5; 0,435 → 0,4 — mengikuti angka yang tercetak di Gambar 7A; "
       "disetujui 21 Okt. 14,95 km² → \"sekitar 15 km²\".)", size=9)
s.para("Tujuh kalimat terakhir ¶1 di naskah = definisi label vonis (passage 3a, 21 Okt):", bold=True, space_before=6)
s.para("Label hasil mengikuti aturan pra-registrasi: SPECIFIC jika SK 95% tidak mencakup nol dan besar selisih "
       "sekurang-kurangnya 0,05; NOT SPECIFIC jika seluruh SK 95% berada dalam rentang −0,05 hingga +0,05; dan "
       "INCONCLUSIVE untuk semua hasil yang tidak memenuhi kedua kriteria tersebut.", italic=True, size=10)
s.table(
    [
        ["Dicek ulang 11 Nov — semua cocok dengan berkas sumber", "Nilai", "Sumber"],
        ["Pusat vulkanik kanonik (17 di antaranya di luar bingkai 111–115° BT)", "30", "volcanoes_java_full.csv"],
        ["Peta yang dipakai", "10 seed × 3 desain × 3 algoritma, 588.535 sel", "E221 README"],
        ["Bootstrap blok spasial", "1.000 ulangan, seed 235", "E235 DESIGN §4"],
        ["Pra-registrasi E235", "SAH — DESIGN.md dibekukan 3ac41cc, di-push sebelum skripnya ada (F27)", "e235_outcome.json"],
        ["E* dan luas di atasnya", "1.000 m · 3.434,52 km²", "e235_outcome.json M1"],
        ["Bagian dekat / jauh", "97,479 % / 0,435 %", "idem"],
        ["Daratan jauh 1.000–1.200 m; di atas 1.200 m", "14,95 km² (4 blok); nol", "e235_bands.csv"],
        ["Ketergantungan E* pada luas minimum", "1.200 m (≤ 14,9 km²) · 1.000 m (15–185) · 800 m (≥ 190)", "AUDIT §4"],
        ["Set bersih (161 situs)", "M1 identik — TIDAK perlu diulang; sudah di §3.3", "E236 E_e235"],
    ],
    widths=[7.6, 6.2, 3.4],
)
s.writebox(lines=4, label="JAWABAN ¶1–¶2: dipakai seperti adanya / ada yang diubah (tulis ubahannya + alasannya)")

# ================================================================ 3. Section 3.10 p3 -- new writing
s.page_break()
s.h("§3.10 ¶3  —  kalimat sensitivitas + post hoc: kalimat ini Claude, Bapak tulis ulang", level=1)
s.para("Yang harus tersampaikan (urutan bebas): (a) dua definisi sensitivitas pra-registrasi memberi angka yang lebih "
       "kecil; (b) varian post hoc bergerak dari +0,005 sampai +0,07 tergantung definisi zona dan pencocokan; "
       "(c) mencocokkan jarak ke sungai juga menurunkannya menjadi +0,028; (d) karena itu tidak ada ARAH yang diklaim "
       "di bawah 1.000 m.", italic=True)
s.h("A. Angka utama ¶3 (XGBoost, 53.927 sel tercocokkan)", level=2)
s.table(
    [
        ["Desain", "Selisih dekat − jauh", "SK 95 %", "Vonis δ = 0,05"],
        ["random", "+0,0491", "+0,0237 … +0,0768", "INCONCLUSIVE"],
        ["TGB", "+0,0489", "+0,0240 … +0,0762", "INCONCLUSIVE"],
        ["hybrid", "+0,0599", "+0,0298 … +0,0914", "SPECIFIC"],
        ["ringkasan (2 dari 3 desain)", "—", "—", "INCONCLUSIVE"],
    ],
    widths=[4.6, 4.0, 5.0, 3.6],
)
s.para("Algoritma lain: RandomForest INCONCLUSIVE (utama); MaxEnt INCONCLUSIVE di ketiga definisi. Hybrid SPECIFIC "
       "HANYA pada XGBoost. Sumber: E235/results/e235_outcome.json (xgboost).", size=9)
s.h("B. Dua definisi sensitivitas pra-registrasi", level=2)
s.table(
    [
        ["Definisi", "Selisih (XGB random)", "SK 95 %", "Vonis"],
        ["V ≤ 10 / N ≥ 30 km", "−0,002", "−0,0471 … +0,0458", "NOT SPECIFIC"],
        ["V ≤ 15 / N ≥ 30 km", "+0,016", "−0,0155 … +0,0514", "INCONCLUSIVE"],
    ],
    widths=[4.6, 4.0, 5.0, 3.6],
)
s.para("⚑ F24 — penting: kedua definisi itu TIDAK hanya mengubah lebar zona. Keduanya juga memindahkan batas E* ke "
       "1.200 m dan mengubah zona jauh dari 40 ke 30 km. Pita 1.000–1.200 m itulah yang menjatuhkan angkanya ke −0,002. "
       "Jadi jangan menulis bahwa selisih \"hilang karena zona vulkanik dipersempit\".", size=9.5)
s.h("C. Varian post hoc (AUDIT 20 Okt; XGB, random / TGB / hybrid)", level=2)
s.table(
    [
        ["Varian", "random · TGB · hybrid"],
        ["hanya V diperketat (≤ 10, N ≥ 40)", "+0,029 · +0,032 · +0,035"],
        ["hanya N diperlonggar (V ≤ 20, N ≥ 30)", "+0,029 · +0,029 · +0,039"],
        ["kedua perubahan, pita 0–1.000 m saja", "+0,005 · +0,008 · +0,010"],
        ["run 10/30 (termasuk pita 1.000–1.200 m, selisih −0,18)", "−0,002"],
        ["N diperketat (≥ 50)", "+0,056 · +0,056 · +0,067 (ketiganya SPECIFIC)"],
        ["jarak ke sungai ikut dicocokkan", "+0,028"],
        ["sungai + 5 bin per variabel", "+0,010"],
    ],
    widths=[10.0, 7.2],
)
s.bullets(
    "Rentang naskah \"+0,005 sampai +0,07\": ujung bawah = kedua perubahan (+0,005), ujung atas = N ≥ 50 (+0,067). "
    "README meringkasnya \"+0,01 sampai +0,07\". Keduanya sah.",
    "Titik +0,05 duduk tepat di ambang → vonis pra-registrasi tetap INCONCLUSIVE.",
    "⚑ F25 (opsional): angka-angka ini hanya XGBoost, sedangkan teks belum menyebut algoritmanya (caption Gambar 7B "
    "sudah). Kalau mau, sebut \"(XGBoost, algoritma utama)\" ± 4 kata di kalimat yang Bapak tulis — tanpa menambah "
    "total kata (potong di tempat lain §3.10).",
)
s.writebox(lines=14, label="TULIS KALIMAT PENGGANTI DI SINI — ± 80 kata (boleh 3–4 kalimat)")

# ================================================================ 4. Section 3.10 p3 -- re-read
s.page_break()
s.h("§3.10 ¶3  —  tiga kalimat Bapak dengan sisipan Claude: baca ulang", level=1)
s.para("Tiga kalimat Bapak (22 Okt) yang di naskah berubah sedikit. Perubahannya di bawah ini; isinya tetap Bapak.",
       italic=True)
s.h("1. Pegunungan Selatan  —  Claude menambah klausa pembuka (F28)", level=2)
s.para("Teks Bapak: \"…wilayah pembanding yang jauh sebagian besar mencakup Pegunungan Selatan, yaitu busur vulkanik tua "
       "berumur Eosen–Miosen yang memiliki bentang alam karst, sehingga wilayah tersebut tidak tepat disebut sebagai "
       "wilayah nonvulkanik.\"", italic=True, size=10)
s.para("Di naskah menjadi: \"In the uplands (400–1.000 m), the far comparison land lies mostly in the Southern "
       "Mountains, an old volcanic arc of Eocene–Miocene age with karst landforms (Smyth et al. 2008), so it would not "
       "be correct to call it non-volcanic.\"", size=10)
s.bullets(
    "Alasannya: tanpa \"di pita 400–1.000 m\", kalimat itu bisa dibaca berlaku untuk semua pita, padahal klaimnya "
    "khusus upland. 95 % sel jauh di pita itu ada di selatan 7,9° LS, 87 % di barat 112,2° BT (Pacitan–Ponorogo).",
    "Sitasi Smyth, Hall & Nichols 2008 (GSA Special Paper 436) ditambahkan Claude; metadata dicek di Crossref 21 Okt.",
)
s.writebox(lines=3, label="PEGUNUNGAN SELATAN: dipakai / diubah (F28 + sitasi)")
s.h("2. Fringe MaxEnt  —  Claude mengganti rujukan Gambar (F29)", level=2)
s.para("Teks Bapak: \"…contingent fringe MaxEnt pada Gambar 5 memiliki median ketinggian 1.107 m…\"", italic=True, size=10)
s.para("Di naskah: \"…the contingent fringe of the MaxEnt maps (Section 3.6) has a median elevation of 1,107 m…\"", size=10)
s.bullets("Alasannya: Gambar 5 punya dua panel; merujuk section lebih tepat. Angka 1.107 m benar (XGBoost 491 m; "
          "E221 README).")
s.writebox(lines=3, label="FRINGE: dipakai / diubah (F29)")
s.h("3. Daya uji  —  Claude memperjelas objek kalimat", level=2)
s.para("Teks Bapak: \"Daya uji yang terbatas juga tidak memungkinkan penarikan kesimpulan yang kuat mengenai perbedaan "
       "antardesain.\"", italic=True, size=10)
s.para("Di naskah: \"…about whether the disagreement between designs differs between near and far land.\"", size=10)
s.bullets(
    "Alasannya: \"perbedaan antardesain\" bisa dibaca sebagai temuan utama naskah (perbandingan desain latar), bukan tes "
    "M3. Objek yang benar: apakah perbedaan itu beda antara wilayah dekat dan jauh.",
    "Fakta (E235 README): vonis M3 NOT SPECIFIC, tetapi tes tidak mungkin keluar SPECIFIC — fringe di bawah 1.000 m "
    "hanya 2,5 % (dekat) dan 1,4 % (jauh), sedangkan ambangnya 5 poin persen. Relatif: 1,5–1,8× lebih sering dekat "
    "gunung api, SK tak mencakup nol. Karena itu kalimat ini TIDAK boleh menjadi \"perbedaan antardesain tidak "
    "bergantung pada lingkungan vulkanik\".",
)
s.writebox(lines=4, label="DAYA UJI: dipakai / diubah — dan apakah kalimat ini sudah cukup jelas")

# ================================================================ 5. Section 3.11
s.page_break()
s.h("§3.11  —  dua kalimat + satu keputusan (± 70 kata)", level=1)
s.para("Fungsi: (1) permukaan kesesuaian tidak bisa membedakan mekanisme ketidakteramatan; (2) karena itu kesesuaian "
       "rendah dekat gunung api bukan bukti permukiman terkubur; (3) desain kuota yang diuji gagal di dua rezim — "
       "dilaporkan tanpa diperhalus.", italic=True)
s.para("Kalimat 1 adalah tulisan Bapak (passage 8E, 22 Okt): \"Permukaan kesesuaian tidak dapat membedakan mekanisme "
       "yang menyebabkan suatu situs tidak teramati.\" → dipakai.", italic=True, size=10)
s.h("A. Kalimat 2 (baris 627) — kalimat Claude (H20), Bapak tulis ulang", level=2)
s.bullets(
    "Klaim yang harus tersampaikan: kesesuaian prediksi rendah di dekat gunung api, SENDIRIAN, tidak bisa dibaca sebagai "
    "bukti permukiman terkubur.",
    "Kenapa kalimat Bapak 22 Okt tidak dipakai apa adanya: kalimat itu menambahkan alasan \"sebab perbandingan model "
    "tidak bertahan dalam evaluasi latar bersama (Section 3.2)\". Di set bersih (161 situs) redesign MENDAPAT +0,051 — "
    "jadi \"perbandingan tidak bertahan\" terlalu kuat untuk kedua set.",
    "Yang tetap berlaku: evaluasi latar sendiri menggelembungkan hasil (+0,037 / +0,041); dan §3.2 sudah menjelaskan "
    "bagian mana dari tangga yang artefak.",
    "Keputusan Bapak: alasan itu (a) tidak disebut sama sekali (paling pendek), atau (b) disebut hati-hati dengan "
    "rujukan Section 3.2.",
)
s.writebox(lines=6, label="TULIS KALIMAT PENGGANTI (baris 627) — ± 20 kata · pilihan alasan: tidak disebut / disebut hati-hati")
s.h("B. Kalimat 3 (baris 628) — desain kuota: teks v0.2, bukan tulisan Bapak (F26)", level=2)
s.table(
    [
        ["Fakta", "Angka", "Sumber"],
        ["World C (rezim asal: survei tidak merata antarkawasan) — kuota − random", "AUC_true −0,2457 · Jaccard −0,4688 · 0/30", "e222c_outcome.json"],
        ["World D (rezim paling menguntungkan: kebenaran seimbang) — kuota − random", "AUC_true −0,2027 · Jaccard −0,2826 · 0/30", "e222d_outcome.json"],
        ["TGB di dua dunia itu (pembanding, sudah di §3.6)", "C −0,0010 · D +0,0022", "idem"],
    ],
    widths=[7.8, 6.0, 3.4],
)
s.bullets(
    "Yang perlu tersampaikan: desain kuota diuji di rezim yang ia diciptakan untuknya DAN di rezim paling "
    "menguntungkannya; gagal di kedua-duanya; dilaporkan tanpa diperhalus.",
    "Angka TIDAK berubah (keputusan Bapak 22 Okt: \"hasil dan angka kuota tetap\"). Yang dipertanyakan hanya kalimatnya.",
    "Aturan editor JCAA: penulis menulis sendiri. Kalimat ini dari era v0.2 (disusun Claude).",
)
s.writebox(lines=3, label="JAWABAN F26: saya tulis ulang (lanjut di kotak bawah) / pakai kalimat lama dengan angka ini")
s.writebox(lines=6, label="TULIS KALIMAT DESAIN KUOTA DI SINI — ± 50 kata (kalau memilih menulis ulang)")

# ================================================================ 6. Decisions
s.page_break()
s.h("Keputusan Bapak — satu kali, di akhir", level=1)
s.table(
    [
        ["#", "Hal", "Saran saya"],
        ["F24", "Kalimat sensitivitas: sebut sebab penurunan ke −0,002, atau hanya hasilnya?", "Hanya hasilnya: −0,002 NOT SPECIFIC; +0,016 INCONCLUSIVE; post hoc +0,005…+0,07; tidak ada arah yang diklaim. Sebab (pita 1.000–1.200 m) jangan disebut"],
        ["F25", "Sebut \"XGBoost, algoritma utama\" di ¶3?", "Ya kalau ada tempat: ± 4 kata, tanpa menambah total"],
        ["F26", "Kalimat desain kuota: tulis ulang / pakai kalimat lama?", "Tulis ulang (aturan editor); angka seperti tabel"],
        ["F27", "Pertahankan \"pra-registrasi\" di baris 597?", "Ya — E235 sah dipra-registrasi (3ac41cc di-push sebelum skrip)"],
        ["F28", "Klausa \"In the uplands (400–1.000 m),\"?", "Pertahankan"],
        ["F29", "Rujukan Gambar 5 → Section 3.6?", "Pertahankan"],
        ["F30", "Ulangi angka set bersih di §3.10?", "Tidak — sudah di §3.3; plafon 582/582"],
    ],
    widths=[1.2, 6.6, 9.4],
)

s.h("Sisa perjalanan sampai 30 November", level=1)
s.table(
    [
        ["Bagian", "Kapan", "Status"],
        ["§3.1–3.3", "3 Nov", "selesai (sesi 1)"],
        ["§3.4–3.6", "10 Nov", "selesai (sesi 2)"],
        ["§3.7–3.9", "10 Nov", "selesai (sesi 3)"],
        ["§3.10–3.11", "11 Nov", "» sesi 4 (lembar ini)"],
        ["§2 Metode §2.1–2.6 (+ dunia A–D, C094)", "12–14 Nov", "2 sesi"],
        ["(keputusan K6)", "15 Nov", "—"],
        ["§4 Diskusi + Keterbatasan (+ F15)", "16–17 Nov", "2 sesi"],
        ["§1 Pendahuluan", "18–19 Nov", "passage 4–6 bila Bapak setuju"],
        ["§5 Kesimpulan + Abstrak", "20 Nov", "abstrak ≤ 300 kata"],
        ["Surat balasan, caption, alt text", "21–23 Nov", "menyusul"],
        ["Titik periksa", "25 Nov", "kalau belum selesai: minta waktu ke editor hari itu"],
    ],
    widths=[7.4, 3.0, 5.6],
)

s.h("Aturan main sesi ini", level=1)
s.bullets(
    "**Bapak menulis dalam bahasa Indonesia**, dengan kata-kata Bapak sendiri — tanpa alat AI lain untuk menyusun atau merapikan teks (aturan editor JCAA).",
    "**Angka di lembar ini sudah dicek ulang** dari berkas sumbernya (11 Nov). Pakai angka seperti tertulis.",
    "**Saya menerjemahkan tanpa menambah isi.** Perubahan di luar terjemahan saya tanyakan dulu dan saya catat di PI_PROSE_LOG_R3.md.",
    "**Boleh bertanya di chat** soal metode atau statistik. Penjelasan itu untuk dipahami, bukan disalin.",
    "**Jangan memakai naskah v0.3 sebagai contoh kalimat.** Yang jadi bahan: butir dan angka di lembar ini, plus tulisan Bapak sendiri.",
    "**Naskah berdiri sendiri:** tanpa \"reviewer\", \"revisi\", \"versi sebelumnya\", \"originally reported\". Tulis \"Section\", bukan \"§\".",
    "**Tidak menambah kata:** §3.10 dan §3.11 tepat di plafon.",
)

s.h("Jangan ditulis (sudah terbukti salah atau terlalu kuat)", level=1)
s.bullets(
    "\"perbedaan antardesain tidak bergantung pada lingkungan vulkanik\" — M3 tidak punya daya uji.",
    "\"selisih hilang karena zona vulkanik dipersempit\" — definisi 10/30 juga menambah pita 1.000–1.200 m (F24).",
    "\"suitability lebih tinggi dekat gunung api pada semua algoritma\" — hybrid SPECIFIC hanya XGBoost.",
    "\"dataran tinggi non-vulkanik\" — Pegunungan Selatan busur Eosen–Miosen; sebut \"jauh dari pusat Kuarter\".",
    "\"+0,05 signifikan\" tanpa \"duduk di ambang / ringkasan INCONCLUSIVE\".",
    "\"kuota gagal karena …\" — tidak ada analisis sebab; cukup hasilnya di dua rezim.",
    "\"pra-registrasi\" untuk E219/E221/E223 (C082). E235 boleh (F27).",
    "Angka mentah tanpa pembulatan naskah: 3.434,5 → 3.435; 97,48 → 97,5; 0,435 → 0,4; 14,95 → 15; 0,0491 → 0,049.",
)

s.save(OUT)
