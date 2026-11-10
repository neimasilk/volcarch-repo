# Builds the P2 author-rewrite writing sheet for session 2 (Results 3.4-3.6) as a .docx.
# Content rule (C093): bullets, numbers and questions only -- no prose the PI could copy.
# Layout rule (from session 1): the first page states WHAT TO WRITE; the facts come after; the rules at the end.
# Facts and sources: FACTS_S2_RESULTS_3_4-3_6.md (all numbers re-checked 2026-11-09).
#   python papers/P2_settlement_model/rewrite_r3/build_writing_sheet_s2.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheet_lib import Sheet  # noqa: E402

OUT = Path(__file__).resolve().parent / "LEMBAR_TULIS_S2_20261109.docx"

s = Sheet()

# ================================================================ 1. COVER + TASK
s.cover("LEMBAR TULIS — SESI 2",
        "P2 / JCAA #280 — naskah pengganti, penulisan oleh penulis\nHasil penelitian §3.4–3.6 · 9 November 2026")

s.h("Yang Bapak tulis hari ini: tiga subseksi pendek", level=1)

s.task_card(
    "1", "§3.4  The selection criterion has no interior optimum", "maks. 261 kata — dua paragraf",
    [
        "Kenapa tangga itu bisa terjadi: aturan apa yang memilih desain, dan apakah aturan itu pernah memberi tanda berhenti?",
        "Bentuk kurvanya di dunia sintetis dan di data nyata: di mana maksimumnya?",
        "Berapa biaya berhenti di 0.30 seperti E013 — dan kenapa E013 berhenti di situ?",
        "Berapa biayanya kalau share dinaikkan sampai 1.0 — dan apa yang tidak dilakukan kriteria itu?",
    ],
    "halaman 2–3",
)
s.task_card(
    "2", "§3.5  Reported and true performance diverge", "maks. 92 kata — satu paragraf pendek",
    [
        "Seberapa cepat angka laporan bergerak dibanding kebenaran di dunia sintetis (dan dihitung dengan cara apa)?",
        "Ke arah mana keduanya bergerak di data nyata?",
        "Prediksi mana yang gagal, dan bagaimana kegagalan itu dilaporkan?",
    ],
    "halaman 4",
)
s.task_card(
    "3", "§3.6  The target-group-background null, reported as unexplained", "maks. 247 kata — dua paragraf",
    [
        "Kondisi apa yang diberikan pada TGB di dunia sintetis, dan apakah TGB memperbaiki peta?",
        "Di dunia mana TGB netral, dan di mana satu-satunya ia unggul?",
        "Penjelasan apa yang diuji (E224), bagaimana diuji, dan hasilnya?",
        "Apa batasan uji itu sendiri, dan kenapa metrik sekunder tidak dipakai untuk menafsir ulang?",
    ],
    "halaman 5–6",
)

s.table(
    [
        ["Cara pakai", ""],
        ["1", "Buka halaman detail bagian yang mau ditulis. Baca tabel angka dan butir \"yang perlu tersampaikan\"."],
        ["2", "Tulis di kotak abu-abu, bahasa Indonesia, kata-kata sendiri. Tidak perlu kalimat jadi."],
        ["3", "Kalau ada angka atau metode yang belum jelas, tanya di chat — saya jelaskan, bukan untuk disalin."],
        ["4", "Kirim balik berkasnya, atau tempel teksnya di chat."],
        ["5", "Saya terjemahkan, tunjukkan hasilnya, baru masukkan ke naskah. Setiap perubahan di luar terjemahan minta izin dulu."],
    ],
    widths=[1.2, 16.0],
)
s.para("")
s.para("Panjang: plafon = panjang v0.3 (keputusan Bapak 9 Nov). Tiga subseksi ini paling banyak 600 kata bahasa Inggris, "
       "kira-kira 680–700 kata bahasa Indonesia. Lebih pendek boleh.", italic=True, size=9.5)
s.para("Sudah Bapak tulis di §3.2–3.3 dan tidak perlu diulang: hitungan run sintetis (240/240, 52/60, 51/60, 343/360), "
       "kriteria di set bersih (0.697 → 0.866), langkah TGB yang tidak ada di kedua set. Cukup rujuk Section 3.2 / 3.3. Aturan lengkap ada di halaman terakhir.",
       italic=True, size=9.5)

# ================================================================ 2. Section 3.4
s.page_break()
s.h("§3.4  The selection criterion has no interior optimum  —  maks. 261 kata", level=1)
s.para("Fungsi: menjawab \"kenapa tangga itu bisa terjadi?\" — aturan yang memilih desain (AUC di own background) tidak pernah "
       "menyuruh berhenti; berapa biayanya di grid E013, dan berapa kalau grid diteruskan.", italic=True)

s.h("A. Bentuk kurva kriteria (Gambar 4)", level=2)
s.table(
    [
        ["Hal", "Angka", "Sumber"],
        ["Sintetis (E222): AUC own di share 0.0 / 0.3 / 0.7 / 1.0", "0.7367 / 0.7820 / 0.8412 / 0.8904", "e222_runs.csv"],
        ["Grid sintetis", "hanya 4 titik", "idem"],
        ["Nyata (E218b, 5 seed): 11 titik 0.0–1.0", "0.7208 → 0.7137 → 0.7253 → … → 0.8435", "e218b_hardfrac_sweep.csv"],
        ["Satu-satunya turunan di data nyata", "0.0 → 0.1: −0.0071; setelah itu naik terus", "idem"],
        ["Letak maksimum", "di ujung rentang yang dicoba, di kedua dunia", "idem"],
    ],
    widths=[6.4, 5.8, 3.8],
)

s.h("B. Biaya di grid E013 (share 0.0 / 0.15 / 0.30)", level=2)
s.para("\"Biaya\" = AUC sebenarnya (atau common) desain terbaik di grid itu, dikurangi milik desain yang dipilih kriteria.", italic=True, size=9.5)
s.table(
    [
        ["Hal", "Angka"],
        ["Kenapa E013 berhenti di 0.30", "grid-nya berhenti di situ (§2.4), bukan karena kriteria"],
        ["Biaya, data nyata", "+0.0044"],
        ["Biaya, sintetis (60 kasus)", "median +0.0000 · rata-rata +0.0012 · maks. +0.0088"],
        ["Kriteria tidak memilih desain terbaik-menurut-kebenaran", "29/60 kasus — tapi selisihnya hampir nol"],
        ["Pilihan kriteria di grid itu (sintetis)", "acak 50, TGB 10 (bukan hybrid)"],
    ],
    widths=[8.6, 7.4],
)
s.bullets("Di 0.30 kriteria memilih hybrid di data nyata tapi acak di sintetis — itu \"calibration gap\" yang sudah ada di "
          "Limitations. Tidak perlu diulang di sini kecuali Bapak mau.")

s.h("C. Biaya kalau share diteruskan sampai 1.0", level=2)
s.table(
    [
        ["Hal", "Angka"],
        ["Biaya, data nyata", "+0.0973"],
        ["Biaya, sintetis", "median +0.1937, positif di 60/60 kasus"],
        ["Desain yang dipilih (sintetis)", "hybrid(1.0) di 60/60"],
        ["Desain yang benar-benar terburuk (sintetis)", "hybrid(0.0) 50/60 · hybrid(0.3) 8/60 · hybrid(0.7) 2/60"],
        ["hybrid(1.0) menjadi yang terburuk", "0/60 — mahal, tapi bukan yang terburuk"],
        ["Biaya pilihan itu terhadap desain terbaik", "+0.194 AUC sebenarnya"],
    ],
    widths=[8.6, 7.4],
)

s.para("Yang perlu tersampaikan:", bold=True, space_before=6)
s.bullets(
    "Kriteria tidak pernah \"menunjuk ke dalam\" rentang → tidak pernah memberi tanda berhenti.",
    "Di grid E013 biayanya kecil; bahayanya struktural — kalau share dinaikkan, kriteria tidak punya rem.",
    "Set bersih: sudah di §3.3 (di sana kriteria malah mengarah ke setting dengan skor common terendah) — cukup rujukan.",
)

s.para("⚑ Pilihan F2 (opsional):", bold=True, space_before=6)
s.bullets(
    "E220 mengulang uji yang sama di data nyata dengan 20 seed (bukan 5): AUC own 0.707 → 0.842, naik di setiap langkah "
    "(tanpa turunan 0.0→0.1); AUC common 0.695 → 0.601 (e220_sweep.csv, dihitung ulang 9 Nov).",
    "Naskah lama dan Gambar 4 memakai E218b (5 seed). Saran: tetap pakai E218b. Sebut E220 hanya kalau Bapak ingin "
    "menyatakan bahwa polanya terulang dengan 20 seed (± 25 kata dari plafon 261).",
)
s.writebox(lines=2, label="JAWABAN F2 (E218b saja / tambah E220):")
s.writebox(lines=14, label="TULIS §3.4 DI SINI — maks. 261 kata, dua paragraf")

# ================================================================ 3. Section 3.5
s.page_break()
s.h("§3.5  Reported and true performance diverge  —  maks. 92 kata", level=1)
s.para("Fungsi: seberapa cepat angka laporan bergerak dibanding kebenaran ketika share dinaikkan — rasio dan arahnya. "
       "Bukan hitungan run (itu sudah di §3.2).", italic=True)
s.table(
    [
        ["Hal", "Angka"],
        ["Sintetis, share 0.0 → 1.0: perubahan AUC laporan (own)", "+0.1538"],
        ["Sintetis, share 0.0 → 1.0: perubahan AUC sebenarnya", "+0.0764"],
        ["Rasio, tiga cara menghitung", "endpoint 2.01× · kemiringan OLS per run 2.12× · median rasio per run 2.00×"],
        ["Yang aman ditulis", "\"sekitar dua kali\"; kalau memakai satu angka, sebut cara hitungnya"],
        ["Data nyata, 0.0 → 1.0: AUC laporan", "+0.1227"],
        ["Data nyata, 0.0 → 1.0: AUC common background", "−0.0973"],
        ["Arah di data nyata", "berlawanan: laporan naik, common turun"],
    ],
    widths=[8.6, 7.4],
)
s.para("Prediksi yang gagal (E222, desain tertulis — bukan pra-registrasi ber-cap waktu):", bold=True, space_before=6)
s.bullets(
    "Prediksi P1: inflasi (AUC own − AUC sebenarnya) naik bersama share, Spearman > 0.5 untuk semua run digabung.",
    "Hasil: Spearman gabungan 0.4395 → di bawah ambang → gagal, dan dilaporkan sebagai gagal.",
)
s.para("⚑ Catatan F3:", bold=True, space_before=6)
s.bullets("Naskah lama menyebut prediksi itu berlaku \"in every regime\"; desainnya sendiri menulis \"across all runs\" "
          "(satu Spearman gabungan). Saran: tulis sesuai desain.")
s.writebox(lines=8, label="TULIS §3.5 DI SINI — maks. 92 kata, satu paragraf")

# ================================================================ 4. Section 3.6
s.page_break()
s.h("§3.6  The target-group-background null, reported as unexplained  —  maks. 247 kata", level=1)
s.para("Fungsi: TGB diberi kondisi yang diminta teorinya (proses pengamatan dengan bias survei yang diketahui) dan tetap tidak "
       "memperbaiki peta; satu penjelasan diuji (E224) dan gagal → hasil nol dilaporkan apa adanya.", italic=True)

s.h("A. E222 — dunia sintetis", level=2)
s.table(
    [
        ["Hal", "Angka"],
        ["TGB − acak, kesepakatan peta desil teratas (Jaccard), dunia A + B", "−0.010, positif 28/60 (46.7 %)"],
        ["Ambang di desain tertulis (P3)", "TGB menang di ≥ 60 % kasus → tidak tercapai"],
        ["Dunia A saja (bias survei jalan, spesifikasi benar)", "−0.0247, positif 30 %"],
        ["Dunia B (model salah spesifikasi) — satu-satunya tempat TGB unggul", "Jaccard 0.4504 vs 0.4458; positif 63.3 %"],
        ["Dunia C (bias regional): TGB − acak, AUC sebenarnya", "−0.0010, positif 17/30 (56.7 %)"],
        ["Dunia D (kebenaran regional seimbang): idem", "+0.0022, positif 22/30 (73.3 %)"],
    ],
    widths=[9.6, 6.4],
)
s.bullets("Di dunia bias regional TGB netral, bukan merugikan; ia menang hanya di dunia salah spesifikasi — dilaporkan "
          "karena benar, tapi tidak dijadikan dasar.")

s.h("B. E224 — menguji satu penjelasan (pra-registrasi: desain di-commit sebelum hasil, 3 Agu)", level=2)
s.bullets(
    "Penjelasan yang diuji: road_dist bukan fitur model → bias survei tak terlihat di ruang fitur → TGB tidak punya apa-apa untuk dikoreksi.",
    "Desain: dunia A E222 persis sama, satu beda saja — road_dist masuk fitur atau tidak. 120 run. Dua cabang keputusan ditulis sebelum run.",
    "Metrik utama (ditetapkan dulu): Jaccard peta, TGB − acak, 30 pasangan per lengan; ambang: rata-rata > 0 dan ≥ 60 % pasangan positif.",
)
s.table(
    [
        ["Lengan", "TGB − acak, Jaccard", "Positif", "TGB − acak, AUC sebenarnya", "Positif"],
        ["tanpa road_dist (kontrol)", "−0.0254", "30 %", "−0.0006", "36.7 %"],
        ["dengan road_dist (uji)", "−0.0217", "30 %", "−0.0000", "60.0 %"],
    ],
    widths=[4.4, 3.4, 1.8, 4.6, 1.8],
)
s.bullets(
    "Hasil: tidak ada beda → penjelasan gagal (status eksperimen FAILED, bukan REFUTED — lihat batasan).",
    "Metrik sekunder (AUC sebenarnya) naik dari 36.7 % ke 60 % positif — tidak dipakai untuk menafsir ulang, karena metrik utama yang ditetapkan dulu adalah Jaccard.",
    "Batasan uji kami sendiri: road_dist berkorelasi +0.49 dengan river_dist dan +0.31 dengan elevasi → di kondisi kontrol model sudah sebagian \"melihat\" sinyal bias.",
    "Uji yang bersih butuh permukaan upaya survei yang ortogonal terhadap terrain sejak dirancang → pekerjaan mendatang.",
)
s.h("C. Data nyata", level=2)
s.bullets("Langkah TGB tidak ada di kedua set (§3.3). Cukup rujukan.")
s.writebox(lines=14, label="TULIS §3.6 DI SINI — maks. 247 kata, dua paragraf")

# ================================================================ 5. End matter
s.page_break()
s.h("Peta keseluruhan: yang Bapak tulis sampai 30 November (jadwal disusul ulang 9 Nov)", level=1)
s.table(
    [
        ["Bagian", "Kata (plafon v0.3)", "Kapan", "Status"],
        ["§3.1–3.3", "1.642 (tertulis)", "3 Nov", "selesai (sesi 1)"],
        ["§3.4–3.6", "≤ 600", "9 Nov", "» sesi 2 (lembar ini)"],
        ["§3.7–3.9", "v0.3", "10 Nov", "sesi 3"],
        ["§3.10–3.11", "v0.3", "11 Nov", "sesi 4"],
        ["§2 Metode §2.1–2.6", "±1.500", "12–14 Nov", "2 sesi"],
        ["(keputusan K6)", "—", "15 Nov", "—"],
        ["§4 Diskusi + Keterbatasan", "±1.030", "16–18 Nov", "2 sesi"],
        ["§1 Pendahuluan", "±1.330", "19–20 Nov", "passage 4–6 dipakai bila Bapak setuju"],
        ["§5 Kesimpulan + Abstrak", "±620", "21 Nov", "abstrak ≤ 300 kata"],
        ["Surat balasan, caption, alt text", "—", "22–24 Nov", "menyusul"],
        ["Titik periksa", "—", "25 Nov", "kalau belum selesai: minta waktu ke editor hari itu"],
    ],
    widths=[5.4, 3.2, 2.6, 4.8],
)

s.h("Aturan main sesi ini", level=1)
s.bullets(
    "**Bapak menulis dalam bahasa Indonesia**, dengan kata-kata Bapak sendiri. Tidak perlu rapi, tidak perlu istilah Inggris yang tepat, boleh setengah butir.",
    "**Angka di lembar ini sudah dicek ulang** dari berkas sumbernya (9 Nov). Pakai angka seperti tertulis; jangan dibulatkan ulang.",
    "**Saya menerjemahkan tanpa menambah isi.** Kalau ada yang harus berubah di luar terjemahan (sitasi, istilah, urutan), saya tanyakan dulu dan catat di PI_PROSE_LOG_R3.md.",
    "**Boleh bertanya di chat** soal metode atau statistik yang belum jelas. Penjelasan itu untuk Bapak pahami, bukan untuk disalin.",
    "**Jangan membaca naskah v0.3 sebagai contoh kalimat.** Teks lama itu yang sedang diganti. Yang jadi bahan: butir-butir dan angka di lembar ini.",
    "**Naskah harus berdiri sendiri:** tanpa kata \"reviewer\", \"revisi\", \"versi sebelumnya\", \"originally reported\". Tulis \"Section\", bukan \"§\".",
)

s.h("Jangan ditulis (sudah terbukti salah atau terlalu kuat)", level=1)
s.bullets(
    "\"kriteria memilih desain terburuk\" (K5 — salah: 0/60; yang benar \"+0.194 terhadap desain terbaik\").",
    "\"naik monoton\" untuk data nyata (K6 — ada turunan −0.0071). Rumusan yang aman untuk kedua dunia: maksimum selalu di ujung grid yang dicoba.",
    "\"±10× lebih cepat\" (K2 — sekitar 2×).",
    "\"selalu terinflasi\" tanpa pecahan (K3 — 343/360, dan itu sudah di §3.2).",
    "\"nol TGB sudah diprediksi / dijelaskan\" (K4 — diuji di E224 dan gagal).",
    "\"TGB tidak bekerja\" secara umum — hanya untuk rezim yang diuji.",
    "\"pra-registrasi\" untuk E222 (desain tertulis, masuk git bersama hasilnya). Untuk E224 boleh.",
    "\"selalu\" / \"tidak pernah\" tanpa pecahan di sebelahnya — tulis 60/60, 0/60, 29/60, 28/60.",
)

s.h("Untuk pembaca — cukup tersirat, jangan disebut", level=1)
s.bullets(
    "**R1-3 (istilah):** TGB, Jaccard, AUC laporan vs AUC sebenarnya — beri intuisi satu kalimat kalau belum dijelaskan di §1 atau Tabel 1.",
    "**Batas klaim (bukan permintaan reviewer):** hasil TGB berlaku untuk empat rezim dan ukuran sampel yang diuji; Limitations sudah menyatakannya.",
)

s.save(OUT)
