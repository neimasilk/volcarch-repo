# Builds the P2 author-rewrite writing sheet for session 3 (Results 3.7-3.9) as a .docx.
# Content rule (C093): bullets, numbers and questions only -- no prose the PI could copy.
# The two passages quoted on the sheet are the PI's own Indonesian text (R2 passages 1 and 8D), shown for re-reading.
# Layout rule (from session 1): the first page states WHAT TO WRITE; the facts come after; the rules at the end.
# Facts and sources: FACTS_S3_RESULTS_3_7-3_9.md (all numbers re-checked 2026-11-10).
#   python papers/P2_settlement_model/rewrite_r3/build_writing_sheet_s3.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheet_lib import Sheet  # noqa: E402

OUT = Path(__file__).resolve().parent / "LEMBAR_TULIS_S3_20261110.docx"

s = Sheet()

# ================================================================ 1. COVER + TASK
s.cover("LEMBAR TULIS — SESI 3",
        "P2 / JCAA #280 — naskah pengganti, penulisan oleh penulis\nHasil penelitian §3.7–3.9 · 10 November 2026")

s.h("Yang Bapak tulis hari ini: empat potong pendek + dua bacaan ulang", level=1)

s.task_card(
    "1", "§3.7  Confidence intervals and a detection floor", "maks. 42 kata — dua kalimat",
    [
        "Efek sekecil apa yang tidak bisa disingkirkan oleh data 378 situs ini?",
        "Kenapa itu berarti \"tidak terdeteksi\", bukan \"tidak ada\"?",
    ],
    "halaman 2",
)
s.task_card(
    "2", "§3.8 ¶1  Priority maps change with the seed", "maks. 160 kata — satu paragraf",
    [
        "Kenapa peta (bukan angka AUC) yang penting di praktik pelestarian?",
        "Seberapa banyak sel 10 % teratas berganti hanya karena seed?",
        "Berapa seed yang cukup — dan kenapa satu cara mengukurnya menyanjung, cara lain adil?",
        "Apa protokol yang Bapak rekomendasikan?",
    ],
    "halaman 3",
)
s.task_card(
    "3", "§3.8 ¶2  robust core / contingent fringe — BACA ULANG", "tulisan Bapak 21 Okt (137 kata)",
    ["Dipakai seperti adanya, atau diubah? (plus pilihan F16)"],
    "halaman 4",
)
s.task_card(
    "4", "§3.9 ¶1  The volcano inventory", "maks. 93 kata — satu paragraf",
    [
        "Berapa gunung api yang ditanam di kode lama, dan berapa di inventaris kanonik dalam batas peta?",
        "Kenapa AUC model tidak terpengaruh, tapi Test 1 terpengaruh?",
        "Berapa ρ setelah dihitung ulang, dan apakah vonis tes berubah?",
    ],
    "halaman 5",
)
s.task_card(
    "5", "§3.9 ¶2  Korelasi yang tidak konsisten — BACA ULANG + tulis penutupnya", "3 kalimat Bapak + penutup maks. 67 kata",
    [
        "Tiga kalimat pertama: dipakai seperti adanya?",
        "Penutup: apa yang bisa dan tidak bisa disimpulkan dari selisih −0.163 → −0.243?",
    ],
    "halaman 6",
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
s.para("Panjang: plafon = panjang v0.3 (keputusan Bapak 9 Nov). Yang ditulis baru paling banyak ± 362 kata bahasa Inggris "
       "(± 410–430 kata bahasa Indonesia). Results §3.1–3.6 sudah +351 dari v0.3, jadi lebih pendek lebih baik.",
       italic=True, size=9.5)
s.para("Sudah Bapak tulis dan tidak perlu diulang: CI kesetaraan 12/12 dan batas atas bootstrap (§3.2); kesepakatan "
       "ensemble di set bersih 0.70–0.77 dan pinggiran 5/15/5 situs (§3.3). Cukup rujuk Section 3.2 / 3.3.",
       italic=True, size=9.5)

# ================================================================ 2. Section 3.7
s.page_break()
s.h("§3.7  Confidence intervals and a detection floor  —  maks. 42 kata", level=1)
s.para("Fungsi: apa yang TIDAK bisa disingkirkan data ini. §3.2 dan Limitations merujuk ke sini untuk \"+0.03\".", italic=True)
s.table(
    [
        ["Hal", "Angka", "Sumber"],
        ["Batas atas CI bootstrap presensi (29 ulangan), hybrid − random, common", "MaxEnt +0.0082 · XGB +0.0253 · RF +0.0256", "e223b_bootstrap_summary.csv"],
        ["Batas bawah CI yang sama", "−0.0589 · −0.0241 · −0.0173", "idem"],
        ["Lebar CI", "0.067 · 0.049 · 0.043", "idem"],
        ["Batas deteksi yang dinyatakan", "efek < ± +0.03 tidak bisa disingkirkan pada n = 378", "E223 README; G1 #53"],
        ["Yang tegas ditolak", "+0.092 (12/12 CI kesetaraan; 3/3 bootstrap) — sudah di §3.2", "e223a_equivalence_ci.csv"],
    ],
    widths=[6.6, 5.6, 3.8],
)
s.para("Yang perlu tersampaikan:", bold=True, space_before=6)
s.bullets(
    "+0.03 adalah batas yang dinyatakan (kekuatan uji), bukan hasil.",
    "Efek di bawah batas itu tidak terdeteksi pada ukuran sampel ini — bukan berarti tidak ada.",
    "Set bersih (161 situs): batasnya lebih besar dan tidak diukur — sudah di Limitations, di sini boleh tidak disebut.",
)
s.para("⚑ Opsi O1 (struktur):", bold=True, space_before=6)
s.bullets("§3.7 bisa juga digabung ke akhir §3.2. Saran: tetap subseksi sendiri (§3.2 dan Limitations merujuk ke "
          "Section 3.7; menggabung berarti mengubah rujukan dan nomor §3.8–3.11).")
s.writebox(lines=2, label="JAWABAN O1 (tetap sendiri / gabung):")
s.writebox(lines=6, label="TULIS §3.7 DI SINI — maks. 42 kata")

# ================================================================ 3. Section 3.8 p1
s.page_break()
s.h("§3.8 ¶1  Priority maps change with the seed  —  maks. 160 kata", level=1)
s.para("Fungsi: angka AUC bisa artefak, tapi yang dipakai di lapangan adalah peta; seberapa banyak peta berubah hanya karena "
       "seed, dan berapa banyak ensemble menolong — diukur dengan cara yang adil.", italic=True)
s.h("A. Pergantian sel 10 % teratas antar-seed (E221)", level=2)
s.table(
    [
        ["Hal", "Angka"],
        ["Pergantian (1 − Jaccard) antar pasangan seed", "28.1 % – 47.4 % (per algoritma × desain; 405 pasangan)"],
        ["Paling stabil / paling tidak stabil", "RF-TGB 28.1 % · XGBoost-hybrid 47.4 %"],
        ["Konsekuensi", "peta satu-seed bukan dasar yang baik untuk keputusan survei"],
    ],
    widths=[7.0, 9.0],
)
s.h("B. Berapa seed yang cukup — dua cara mengukur", level=2)
s.table(
    [
        ["Cara", "Angka", "Catatan"],
        ["Ensemble k seed vs ensemble 10 seed yang BERISI seed itu (E221)", "Jaccard > 0.90 setelah 4–7 seed", "menyanjung: pembanding berbagi seed"],
        ["Dua ensemble yang TIDAK berbagi seed (E236)", "1 seed: 0.55–0.70 · 5 seed: XGB 0.76, MaxEnt 0.85, RF 0.86", "ukuran yang adil"],
        ["Set bersih, 5 seed", "0.70–0.77", "sudah di §3.3"],
    ],
    widths=[6.4, 5.8, 3.8],
)
s.para("Yang perlu tersampaikan:", bold=True, space_before=6)
s.bullets(
    "Rata-rata seed jelas menolong.",
    "Protokol: minimal 7 seed sebagai lantai + laporkan kesepakatan antar ensemble yang tidak berbagi seed — bukan "
    "\"sekian seed membuat peta stabil\".",
    "Gambar 6 hanya dirujuk di paragraf ini → rujukannya harus tetap ada.",
)
s.writebox(lines=14, label="TULIS §3.8 ¶1 DI SINI — maks. 160 kata")

# ================================================================ 4. Section 3.8 p2
s.page_break()
s.h("§3.8 ¶2  robust core / contingent fringe — tulisan Bapak 21 Okt, mohon dibaca ulang", level=1)
s.para("Ensemble ini menunjukkan perbedaan yang bermanfaat antara robust core (wilayah inti yang konsisten) dan contingent "
       "fringe (wilayah pinggiran yang bergantung pada desain latar). Robust core didefinisikan sebagai sel yang masuk dalam "
       "10% peringkat teratas pada ketiga desain latar (random, TGB, dan hybrid), dengan setiap peta merupakan rata-rata dari "
       "10 seed; sedangkan contingent fringe adalah sel yang masuk dalam 10% peringkat teratas pada tepat satu desain. "
       "Kepadatan situs yang teramati pada wilayah inti lebih tinggi dibandingkan wilayah pinggiran, yaitu 1,93× untuk "
       "RandomForest (30,7 dibandingkan 15,9 situs per 1.000 km²), 4,34× untuk XGBoost (40,8 dibandingkan 9,4), dan 5,62× "
       "untuk MaxEnt (31,7 dibandingkan 5,7). Perbedaan ini menunjukkan konsistensi spasial, bukan validasi independen, "
       "karena situs yang digunakan untuk menghitung kepadatan tersebut juga digunakan untuk melatih model.",
       italic=True, size=10)
s.para("(Di naskah 5,7 → 5.65, nilai sumber; disetujui 21 Okt.)", size=9)
s.table(
    [
        ["Dicek ulang 10 Nov (e221_priority_sets.csv)", "XGBoost", "RF", "MaxEnt"],
        ["Situs di inti", "169", "141", "100"],
        ["Situs di pinggiran", "15", "15", "16"],
        ["Kepadatan inti / pinggiran (per 1.000 km²)", "40.8 / 9.4", "30.7 / 15.9", "31.7 / 5.65"],
        ["Rasio", "4.34×", "1.93×", "5.62×"],
    ],
    widths=[7.0, 3.0, 3.0, 3.0],
)
s.para("⚑ F16 (opsional):", bold=True, space_before=6)
s.bullets("Di §3.3 Bapak menulis untuk set bersih bahwa pinggiran hanya 5, 15 dan 5 situs, jadi rasionya menunjukkan arah, "
          "bukan ukuran yang tepat. Di set 378 pinggirannya juga kecil (15, 15, 16 situs). Menyebutnya di sini membuat kedua "
          "set sejajar (± 12 kata).")
s.writebox(lines=3, label="JAWABAN: ¶2 dipakai seperti adanya / diubah (tulis ubahannya) · F16 ya / tidak")

# ================================================================ 5. Section 3.9
s.page_break()
s.h("§3.9  Corrections to two volcanic diagnostics  —  ¶1 maks. 93 · penutup ¶2 maks. 67", level=1)
s.para("Fungsi: dua diagnostik pipeline asli diperbaiki — (1) daftar gunung api di kode; (2) satu korelasi yang tidak stabil. "
       "Test 1 sudah dijelaskan di Metode: korelasi kesesuaian dengan jarak ke gunung api; lulus kalau |ρ| < 0.5.", italic=True)
s.h("A. ¶1 — inventaris gunung api", level=2)
s.table(
    [
        ["Hal", "Angka"],
        ["Pusat vulkanik di kode lama", "7 (Kelud, Arjuno-Welirang, Bromo-Tengger, Semeru, Lamongan, Raung, Ijen)"],
        ["Inventaris kanonik", "30 di Jawa; 13 di dalam batas naskah 111–115° BT"],
        ["Enam tambahan", "Lawu, Wilis, Kawi-Butak, Penanggungan, Iyang-Argapura, Baluran"],
        ["Dua tambahan di konsentrasi situs Malang–Mojokerto", "Penanggungan, Kawi-Butak"],
        ["volcano_dist fitur pelatihan?", "tidak → AUC model tidak terpengaruh; Test 1 terpengaruh"],
        ["ρ Test 1, inventaris kanonik (13)", "−0.281"],
        ["ρ Test 1, inventaris lama (7), 5 seed", "−0.243"],
        ["Vonis", "|ρ| < 0.5 di keduanya → tes tetap lulus"],
    ],
    widths=[7.0, 9.0],
)
s.writebox(lines=8, label="TULIS §3.9 ¶1 DI SINI — maks. 93 kata")

s.h("B. ¶2 — tiga kalimat Bapak (22 Okt), mohon dibaca ulang", level=2)
s.para("Korelasi yang tidak konsisten antar-seed. [Pada versi sebelumnya,] korelasi Spearman sebesar −0,163 dilaporkan "
       "berdasarkan satu instans model. Namun, pengulangan analisis dengan lima seed pada inventaris tujuh gunung api yang sama "
       "menghasilkan korelasi sebesar −0,243 (E219).", italic=True, size=10)
s.para("(Di naskah \"pada versi sebelumnya\" sudah dihapus — disetujui 22 Okt.)", size=9)
s.para("Penutup yang Bapak tulis baru — faktanya:", bold=True, space_before=6)
s.bullets(
    "Pengulangan E219 tidak hanya mengganti seed: ia juga memakai sampling background lattice (bukan sampling koordinat "
    "kontinu seperti pipeline asli; dijelaskan di Metode).",
    "Jadi selisih −0.163 → −0.243 bisa dari seed atau dari cara sampling — keduanya tidak bisa dipisahkan di sini.",
    "Yang tetap aman: satu instans model bukan dasar yang baik untuk diagnostik ini.",
    "(Opsional) nilai koreksi di bagian ini memakai ensemble seed; tangga E007–E013 di Tabel 4 adalah pipeline asli, "
    "direproduksi sebagai objek yang diperiksa.",
)
s.para("⚑ F17:", bold=True, space_before=6)
s.bullets("Kalau butir opsional tentang Tabel 4 tidak dipakai, Tabel 4 hanya dirujuk dari caption Gambar 2 — dan Gambar 2 "
          "belum dirujuk di teks (sisa sesi 1). Salah satunya perlu rujukan dari teks sebelum freeze; bisa di sini atau "
          "nanti di §3.1.")
s.writebox(lines=7, label="JAWABAN 3 kalimat (dipakai / diubah) + TULIS PENUTUP ¶2 — maks. 67 kata")

# ================================================================ 6. End matter
s.page_break()
s.h("Peta keseluruhan sampai 30 November (disusun ulang 10 Nov)", level=1)
s.table(
    [
        ["Bagian", "Kapan", "Status"],
        ["§3.1–3.3", "3 Nov", "selesai (sesi 1)"],
        ["§3.4–3.6", "10 Nov", "selesai (sesi 2)"],
        ["§3.7–3.9", "10–11 Nov", "» sesi 3 (lembar ini)"],
        ["§3.10–3.11", "11–12 Nov", "sesi 4"],
        ["§2 Metode §2.1–2.6 (+ dunia A–D)", "13–14 Nov", "2 sesi"],
        ["(keputusan K6)", "15 Nov", "—"],
        ["§4 Diskusi + Keterbatasan (+ uji ortogonal)", "16–17 Nov", "2 sesi"],
        ["§1 Pendahuluan", "18–19 Nov", "passage 4–6 dipakai bila Bapak setuju"],
        ["§5 Kesimpulan + Abstrak", "20 Nov", "abstrak ≤ 300 kata"],
        ["Surat balasan, caption, alt text", "21–23 Nov", "menyusul"],
        ["Titik periksa", "25 Nov", "kalau belum selesai: minta waktu ke editor hari itu"],
    ],
    widths=[7.4, 3.0, 5.6],
)

s.h("Aturan main sesi ini", level=1)
s.bullets(
    "**Bapak menulis dalam bahasa Indonesia**, dengan kata-kata Bapak sendiri — tanpa alat AI lain untuk menyusun atau merapikan teks (aturan editor JCAA).",
    "**Angka di lembar ini sudah dicek ulang** dari berkas sumbernya (10 Nov). Pakai angka seperti tertulis.",
    "**Saya menerjemahkan tanpa menambah isi.** Perubahan di luar terjemahan saya tanyakan dulu dan saya catat di PI_PROSE_LOG_R3.md.",
    "**Boleh bertanya di chat** soal metode atau statistik. Penjelasan itu untuk dipahami, bukan disalin.",
    "**Jangan memakai naskah v0.3 sebagai contoh kalimat.** Yang jadi bahan: butir dan angka di lembar ini, plus tulisan Bapak sendiri.",
    "**Naskah berdiri sendiri:** tanpa \"reviewer\", \"revisi\", \"versi sebelumnya\", \"originally reported\". Tulis \"Section\", bukan \"§\".",
)

s.h("Jangan ditulis (sudah terbukti salah atau terlalu kuat)", level=1)
s.bullets(
    "\"kepadatan inti 2–5.6× pinggiran\" (K7) — yang benar 1.93× / 4.34× / 5.62×.",
    "\"7 seed membuat peta stabil\" (C083) — 7 seed adalah lantai, bukan jaminan.",
    "−0.163 sebagai nilai yang bisa direproduksi (G1c) — 5 seed memberi −0.243.",
    "\"pra-registrasi\" untuk E219, E221, E223 — desain tertulis, di-commit bersama hasil. Hanya E224 yang sah.",
    "\"+0.03 adalah hasil\" — itu batas kekuatan uji.",
    "\"inti tervalidasi\" — kepadatan dihitung dari situs yang juga melatih model.",
    "\"selalu\" / \"tidak pernah\" tanpa pecahan di sebelahnya.",
)

s.save(OUT)
