# Builds the P2 author-rewrite writing sheet (sesi 1, Results 3.1-3.3) as a .docx.
# Content rule (C093): bullets, numbers and questions only -- no prose the PI could copy.
# Layout rule: the first page states WHAT TO WRITE; the facts come after; the rules at the end.

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = "papers/P2_settlement_model/rewrite_r3/LEMBAR_TULIS_S1_20261103_v2.docx"

doc = Document()
sec = doc.sections[0]
sec.top_margin = Cm(1.8)
sec.bottom_margin = Cm(1.8)
sec.left_margin = Cm(2.0)
sec.right_margin = Cm(2.0)

st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)

# --- page number in the footer -------------------------------------------------
_fp = sec.footer.paragraphs[0]
_fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
_run = _fp.add_run()
_f1 = OxmlElement("w:fldChar"); _f1.set(qn("w:fldCharType"), "begin")
_it = OxmlElement("w:instrText"); _it.set(qn("xml:space"), "preserve"); _it.text = "PAGE"
_f2 = OxmlElement("w:fldChar"); _f2.set(qn("w:fldCharType"), "end")
_run._r.append(_f1); _run._r.append(_it); _run._r.append(_f2)


def shade(el, hexcolor):
    pr = el.get_or_add_tcPr() if el.tag.endswith("}tc") else el.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    pr.append(shd)


def h(text, level=1, color="1F3864"):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = RGBColor.from_string(color)
    return p


def para(text, bold=False, italic=False, size=10.5, space_before=0, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    if align:
        p.alignment = align
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    return p


def bullets(*items, style="List Bullet"):
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        items = items[0]
    for it in items:
        p = doc.add_paragraph(style=style)
        p.paragraph_format.space_after = Pt(2)
        if it.startswith("**") and "**" in it[2:]:
            lead, rest = it[2:].split("**", 1)
            p.add_run(lead).bold = True
            p.add_run(rest)
        else:
            p.add_run(it)


def table(rows, widths=None, header=True, size=9.5):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(str(cell_text))
            r.font.size = Pt(size)
            if header and i == 0:
                r.bold = True
                shade(cell._tc, "D9E2F3")
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    return t


def writebox(lines=8, label="TULIS DI SINI", fill="F7F7F7"):
    p = doc.add_paragraph()
    r = p.add_run(label)
    r.bold = True
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor.from_string("7F7F7F")
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.cell(0, 0)
    shade(cell._tc, fill)
    cell.text = ""
    for _ in range(lines):
        cp = cell.add_paragraph()
        cp.paragraph_format.space_after = Pt(2)
        cp.add_run("").font.size = Pt(10.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def task_card(no, heading, words, questions, detail):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.cell(0, 0)
    shade(cell._tc, "EEF3FA")
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(f"{no}. {heading}   ·   {words}")
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor.from_string("1F3864")
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_before = Pt(3)
    rr = p2.add_run("Tulis prosa yang menjawab pertanyaan-pertanyaan ini:")
    rr.italic = True
    rr.font.size = Pt(9.5)
    for q in questions:
        qp = cell.add_paragraph()
        qp.paragraph_format.space_after = Pt(1)
        qp.paragraph_format.left_indent = Cm(0.5)
        qp.add_run("• " + q).font.size = Pt(10)
    p3 = cell.add_paragraph()
    p3.paragraph_format.space_before = Pt(3)
    r3 = p3.add_run(f"Angka dan sumbernya: {detail}")
    r3.font.size = Pt(9)
    r3.font.color.rgb = RGBColor.from_string("7F7F7F")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ================================================================ 1. COVER + TASK
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("LEMBAR TULIS — SESI 1")
r.bold = True
r.font.size = Pt(22)
r.font.color.rgb = RGBColor.from_string("1F3864")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("P2 / JCAA #280 — naskah pengganti, penulisan oleh penulis\nHasil penelitian §3.1–3.3 · 3 November 2026")
r.font.size = Pt(11.5)
r.italic = True

para("")

h("Yang Bapak tulis hari ini: tiga bagian, satu subseksi satu paragraf-besar", level=1)

task_card(
    "1",
    "§3.1  The reported ladder",
    "±80 kata — satu paragraf pendek",
    [
        "Tangga itu apa: tujuh langkah desain, dari background acak sampai hybrid.",
        "Angka yang dilaporkan berapa, dan dibanding angka apa.",
        "Pembaca harus membacanya bagaimana (desain makin realistis → skor makin baik) — dan bahwa itulah yang diuji di §3.2, bukan kesimpulan.",
        "Kenapa perbandingan dengan model nol di Tabel 5 harus dibaca hati-hati.",
    ],
    "halaman 3",
)

task_card(
    "2",
    "§3.2  The ladder does not survive a common evaluation background",
    "±700 kata — empat sampai enam paragraf",
    [
        "Apa yang diuji: kenaikan tangga diuji ulang, semua desain dinilai di background yang sama.",
        "Hasil dekomposisi: berapa kontribusi redesign background, dan berapa yang datang dari fitur sungai.",
        "Apa yang terjadi kalau desain dinilai di own background (inflasi) — dan kenapa langkah TGB bukan artefak.",
        "Bukti dari dunia sintetis dan uji statistik: apakah +0.092 bisa ditolak.",
        "Akibatnya untuk Tabel 5: berapa marginnya sekarang, dan kenapa Tabel 5 dibaca sebagai batas atas.",
    ],
    "halaman 4–5",
)

task_card(
    "3",
    "§3.3  The presence records and a cleaned subset",
    "±600 kata — tiga sampai lima paragraf",
    [
        "Data presensinya apa, berapa, dan apa masalahnya (monumen modern, salah letak).",
        "Bagaimana dibersihkan: berapa dibuang dan atas dasar apa, sisanya berapa.",
        "Apa yang berubah kalau analisis diulang di set bersih — dan apa yang tetap sama.",
        "Apa akibatnya untuk kriteria seleksi, peta prioritas, dan kontrol elevasi.",
    ],
    "halaman 6–7",
)

para("")
t = table(
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

para("")
para("Aturan lengkap dan daftar kalimat yang tidak boleh muncul ada di halaman terakhir.", italic=True, size=9.5)

doc.add_page_break()

# ================================================================ 2. Section 3.1
h("§3.1  The reported ladder  —  ±80 kata", level=1)
para("Fungsi: menyajikan kembali klaim yang akan diuji — kenaikan AUC dalam tujuh eksperimen — sebelum diuji di §3.2.", italic=True)

para("Angka:", bold=True, space_before=4)
table(
    [
        ["Hal", "Angka", "Sumber"],
        ["AUC XGBoost, background acak (E007)", "0.659 ± 0.077", "E007/model_results.txt"],
        ["AUC XGBoost, hybrid (E013), run asli", "0.768 ± 0.069", "E013/model_results.txt"],
        ["AUC XGBoost E013, rata-rata 20 seed", "0.751 ± 0.013", "supplement/e013_robustness_summary.txt"],
        ["Rentang 20 seed (XGBoost)", "0.729 – 0.774", "idem"],
        ["Kenaikan tangga yang dilaporkan", "+0.092  (= 0.751 − 0.659)", "turunan"],
        ["Model nol: acak / sungai < 2 km / DKNS", "0.500 / 0.581 / 0.646", "supplement/null_model_comparison.txt"],
        ["Seed terburuk vs DKNS", "+0.083  (= 0.729 − 0.646)", "turunan"],
        ["Rata-rata seed vs DKNS", "+0.105", "SIG_G1_VERIFICATION_20261026.md"],
        ["Langkah tangga (XGBoost)", "E008 sungai 0.685 · E009 tanah 0.664 · E010 TGB 0.711 · E011 TGB tuned 0.725 · E012 TGB diperluas 0.730 · E013 hybrid 0.768", "model_results.txt tiap eksperimen"],
    ],
    widths=[5.4, 5.6, 5.0],
)

para("Yang perlu tersampaikan:", bold=True, space_before=6)
bullets(
    "Tangga = tujuh langkah desain; tiap langkah membuat background evaluasi makin realistis, dan AUC naik.",
    "Cara membacanya: desain background makin realistis → model makin baik. Itu bacaan yang diuji di §3.2, bukan kesimpulan.",
    "Tabel 5 membandingkan dengan model nol; tapi model nol dinilai di grid semua sel, sedangkan E013 di own background-nya — dua background berbeda.",
    "DKNS memakai lokasi situs sebenarnya → batas atas yang tidak sebanding, bukan pesaing yang adil.",
)

para("⚠ Keputusan F1 — perlu jawaban Bapak:", bold=True, space_before=6)
bullets(
    "Masalahnya: naskah sekarang menaruh AUC rata-rata 20 seed (0.751) berdampingan dengan TSS dari run asli (0.507 ± 0.167). Dua angka dari dua run berbeda.",
    "TSS rata-rata 20 seed = 0.465 ± 0.021 (e013_robustness_summary.txt).",
    "Pilihan A (disarankan): 0.751 dipasangkan dengan 0.465 — sejenis, dua-duanya rata-rata seed.",
    "Pilihan B: tetap 0.768 dengan 0.507 — dua-duanya run asli.",
    "Tulis pilihan Bapak di kotak kecil di bawah, atau jawab di chat.",
)

writebox(lines=2, label="JAWABAN F1 (A / B, atau kalimat sendiri):")
writebox(lines=10, label="TULIS §3.1 DI SINI — satu paragraf, ±80 kata")

# ================================================================ 3. Section 3.2
doc.add_page_break()
h("§3.2  The ladder does not survive a common evaluation background  —  ±700 kata", level=1)
para("Fungsi: uji utama pada 378 rekaman kompilasi — apakah kenaikan tangga tetap ada kalau semua desain dinilai pada background yang sama?", italic=True)

h("A. Dekomposisi (E217b: 5 seed × 3 algoritma)", level=2)
table(
    [
        ["Hal", "Angka"],
        ["Kontribusi redesign background, common background, fitur lengkap (rata-rata 3 algoritma)", "−0.0142 AUC"],
        ["Interval seed-cluster", "−0.021 s.d. −0.005"],
        ["Sel seed × algoritma yang positif", "3 dari 15"],
        ["Kontribusi redesign, fitur terrain saja", "+0.0054"],
        ["Kontribusi fitur sungai, common background", "+0.0424, positif 12/12 pasangan"],
        ["Redesign yang sama, dinilai di own background", "rata-rata +0.0316 (MaxEnt +0.0145, XGBoost +0.0431, RF +0.0373)"],
    ],
    widths=[10.6, 5.4],
)
bullets("Hasil kompilasi ini tidak berlaku umum: di set bersih hasilnya positif (§3.3).")

h("B. Hybrid vs acak per background evaluasi (E218)", level=2)
bullets("hybrid > acak: 3/3 algoritma di background hybrid; 0/3 di background uniform, TGB, dan stratified.")

h("C. Inflasi own background (E218, 378 rekaman)", level=2)
table(
    [
        ["Hal", "Angka"],
        ["Hybrid: own − uniform, sel positif", "60/60, rata-rata +0.037, rentang +0.005 s.d. +0.084"],
        ["Ukuran dibanding kenaikan tangga", "orde yang sama dengan +0.092"],
        ["TGB: own − uniform", "rata-rata −0.0054, 22/60 positif"],
        ["TGB − acak di own background (E217)", "+0.003"],
    ],
    widths=[10.6, 5.4],
)
bullets("Langkah TGB (E010–E012) tidak terulang di benchmark. Jangan menyebutnya artefak — ia tidak dijelaskan oleh inflasi.")

h("D. Dunia sintetis dengan kebenaran diketahui (E222)", level=2)
table(
    [
        ["Desain", "Run dengan reported > true", "Median selisih"],
        ["hybrid", "240/240", "+0.238"],
        ["acak", "52/60", "+0.020"],
        ["TGB", "51/60", "+0.014"],
        ["semua (dua permukaan)", "343/360", "—"],
    ],
    widths=[7.0, 5.0, 4.0],
)
bullets("Pada acak dan TGB, sebagian selisih datang dari beda sampel presensi teramati vs presensi sebenarnya, bukan dari background.")

h("E. Uji statistik (E223)", level=2)
bullets(
    "CI ekuivalensi menolak +0.092 di 12/12 sel algoritma × background evaluasi.",
    "Bootstrap blok presensi: 30 replikasi, 1 dilewati (< 50 situs out-of-bag) → 29 dipakai; fit in-bag, nilai out-of-bag; batas atas CI +0.008 / +0.025 / +0.026 (tiga algoritma) → menolak +0.092 untuk ketiganya.",
    "29 replikasi itu bootstrap kecil — dipakai karena margin penolakannya lebar, bukan karena ujung intervalnya presisi.",
    "Regularisasi MaxEnt β 0.5–4.0: −0.020 s.d. −0.022, 1/10 run positif → bukan overfit model tanpa regularisasi.",
)

h("F. Akibat untuk Tabel 5 (model nol)", level=2)
bullets(
    "Desain E013 di common (uniform) background: 0.706 (XGBoost, rata-rata seed; E218).",
    "Masih di atas DKNS 0.646, tapi margin turun dari +0.105 ke ± +0.06; margin seed terburuk (+0.083) mendekati batas deteksi +0.03 (§3.7).",
    "0.751 (pipeline asli) dan 0.706 (benchmark berbasis lattice) tidak diukur dengan sampling yang sama → baca penyusutan sebagai orde besaran saja.",
    "Yang bertahan: fitur lingkungan menangkap struktur di luar interpolasi situs terdekat. Yang tidak: besarnya. Tabel 5 = batas atas, bukan lantai.",
)

para("Gambar 3: kiri = AUC own vs common per desain; kanan = reported − true di 360 run sintetis per desain.", italic=True, space_before=4)

writebox(lines=30, label="TULIS §3.2 DI SINI — ±700 kata (boleh dipecah: A–C dulu, D–F kemudian)")

# ================================================================ 4. Section 3.3
doc.add_page_break()
h("§3.3  The presence records and a cleaned subset  —  ±600 kata", level=1)
para("Fungsi: menunjukkan bahwa hasil §3.2 bergantung pada data presensi; semua analisis diulang pada 161 situs bersih (E236, pra-registrasi f9d2eb9), skrip tidak diubah selain berkas presensi.", italic=True)

h("A. Data", level=2)
bullets(
    "378 rekaman di 335 koordinat berbeda; ±sepertiga monumen/patung/tugu modern atau salah letak.",
    "Pembersihan (aturan rinci di §2.1, sesi Metode): 217 dibuang = 85 hasil geocode nama (Wikipedia) + 124 monumen modern + 8 candi daerah lain/obyek wisata/institusi → 161 (155 OSM + 6 Wikidata).",
    "Semua angka bagian ini tersedia untuk kedua set di Tabel S9.",
)

h("B. Kontras redesign di set bersih (common background)", level=2)
table(
    [
        ["Hal", "Angka"],
        ["5 seed (pra-registrasi)", "+0.0507"],
        ["10 seed (perluasan; dideklarasikan sebelum run tapi setelah hasil 5 seed)", "+0.0510 [+0.039, +0.063]"],
        ["Sel positif", "29 dari 30"],
        ["Per algoritma", "XGBoost +0.038 · RandomForest +0.044 · MaxEnt +0.072"],
        ["Dua ensemble 5 seed tanpa seed bersama", "+0.0505 dan +0.0516"],
        ["Kontras yang sama di own background (10 seed)", "+0.098"],
        ["Inflasi = own − common", "+0.047 (≈ sebesar gain-nya sendiri)"],
    ],
    widths=[10.6, 5.4],
)

h("C. Inflasi sebagai besaran yang tidak bergantung pada data presensi", level=2)
bullets(
    "hybrid own > uniform: bersih 54/60 sel (rata-rata +0.041) vs kompilasi 60/60 (+0.037).",
    "Beda 1: di set bersih hybrid > acak di semua background evaluasi (3/3 algoritma di uniform, TGB, hybrid, stratified); di kompilasi hanya di own background.",
    "Beda 2: langkah TGB tetap tidak ada di kedua set — TGB − acak: −0.005 (own), +0.001 (common).",
)

h("D. Kriteria seleksi", level=2)
bullets(
    "AUC own background: 0.697 (hard-negative share 0.0) → 0.866 (1.0); maksimum di ujung rentang.",
    "AUC common background di 0.0 / 0.3 / 1.0: 0.680 / 0.713 / 0.656.",
    "Artinya di set ini kriteria merekomendasikan setting dengan skor common terendah — bukan sekadar tidak berhenti.",
)

h("E. Peta prioritas", level=2)
bullets(
    "Rasio densitas robust core / contingent fringe: RF 2.54 · XGBoost 8.51 · MaxEnt 5.30.",
    "Turnover desil teratas antar pasangan seed: 41.6–54.8%.",
    "Dua ensemble 5 seed disjoint: Jaccard 0.70–0.77 (bersih) vs 0.76–0.86 (kompilasi).",
    "Fringe hanya berisi 5 situs (XGBoost), 15 (RF), 5 (MaxEnt) → rasio = arah, bukan besaran presisi.",
)

h("F. Kontrol elevasi (ringkas; penuh di §3.10)", level=2)
bullets(
    "< 1.000 m, selisih kesesuaian dekat − jauh: bersih +0.046 (acak dan TGB), +0.060 (hybrid); kompilasi +0.049, +0.049, +0.060.",
    "Hanya hybrid yang memenuhi kriteria SPECIFIC (pra-registrasi) di kedua set; ringkasan tiga desain = inconclusive; dua definisi sensitivitas bertukar label antara kedua set.",
)

writebox(lines=28, label="TULIS §3.3 DI SINI — ±600 kata")

# ================================================================ 5. End matter
doc.add_page_break()
h("Peta keseluruhan: apa saja yang Bapak tulis sampai 30 November", level=1)
table(
    [
        ["Bagian", "Kata (target)", "Kapan", "Status"],
        ["§3 Hasil §3.1–3.11", "±3.210", "4–10 Nov (5 sesi)", "» sesi 1 = §3.1–3.3 (lembar ini)"],
        ["§2 Metode §2.1–2.6", "±1.500", "11–13 Nov (2 sesi)", "lembar menyusul"],
        ["§4 Diskusi §4.1–4.6 + Keterbatasan", "±1.030", "16–18 Nov", "lembar menyusul"],
        ["§1 Pendahuluan §1.1–1.4", "±1.330", "19–20 Nov", "passage 4–6 dipakai bila Bapak setuju"],
        ["§5 Kesimpulan + Abstrak", "±620", "21 Nov", "abstrak ≤ 300 kata"],
        ["Surat balasan reviewer", "±6 hlm", "22–24 Nov", "kerangka menyusul"],
        ["Keterangan gambar + alt text", "—", "22–24 Nov", "menyusul"],
    ],
    widths=[6.2, 2.6, 3.6, 3.6],
)
para("Sudah ada dan tetap dipakai: passage 1–8 milik Bapak sendiri (±1.300–1.800 kata di §3.8, §3.10, Keterbatasan, penelitian awal). Itu tidak perlu ditulis ulang — cukup dibaca dan disetujui.", italic=True, space_before=6)

h("Aturan main sesi ini", level=1)
bullets(
    "**Bapak menulis dalam bahasa Indonesia**, dengan kata-kata Bapak sendiri. Tidak perlu rapi, tidak perlu istilah Inggris yang tepat, boleh setengah butir.",
    "**Angka di lembar ini sudah dicek ulang** dari berkas sumbernya (3 Nov). Pakai angka seperti tertulis; jangan dibulatkan ulang.",
    "**Saya menerjemahkan tanpa menambah isi.** Kalau ada yang harus berubah di luar terjemahan (sitasi, istilah, urutan), saya tanyakan dulu dan catat di `PI_PROSE_LOG_R3.md`.",
    "**Boleh bertanya di chat** soal metode atau statistik yang belum jelas. Penjelasan itu untuk Bapak pahami, bukan untuk disalin.",
    "**Jangan membaca naskah v0.3 sebagai contoh kalimat.** Teks lama itu yang sedang diganti. Yang jadi bahan: butir-butir dan angka di lembar ini.",
    "**Naskah harus berdiri sendiri:** tanpa kata \"reviewer\", \"revisi\", \"versi sebelumnya\", \"originally reported\". Tulis \"Section\", bukan \"§\".",
)

h("Jangan ditulis (sudah terbukti salah atau terlalu kuat)", level=1)
bullets(
    "\"selalu\" / \"always\" tanpa pecahan di sebelahnya — tulis 343/360, 60/60, 54/60.",
    "\"aturan seleksi memilih desain terburuk\" (K5 — salah).",
    "\"naik monoton\" (K6 — di data nyata ada satu turunan).",
    "\"±10× lebih cepat\" (K2 — sebenarnya 2.01×).",
    "\"densitas 2–5.6×\" (K7 — 1.93× / 4.34× / 5.62× di kompilasi).",
    "\"langkah TGB adalah artefak\" (C084 — tidak terulang, bukan dijelaskan).",
    "\"kenaikan tangga tidak nyata\" untuk kedua set (C090/K1 — di set bersih redesign memberi +0.051).",
    "\"pra-registrasi\" untuk E217 dan E219 (C082). E224, E235, E236 boleh.",
)

h("Jawaban untuk reviewer — cukup tersirat, jangan disebut", level=1)
bullets(
    "**R1 (kejelasan):** naskah berdiri sendiri; statistik dijelaskan untuk pembaca non-statistikawan; tanpa idiom (\"dial\", \"judging at home\", \"home court\"); \"ladder\" boleh, sudah Bapak definisikan di §1.",
    "**R2 (vulkanik vs elevasi):** di bagian ini hanya ringkasan §3.3 F; pembahasan penuh ada di §3.10.",
)

doc.save(OUT)
print("written:", OUT)
