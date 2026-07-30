# Dokumen 5 — Review Co-Author (Go Frendi) atas Paket Revisi P2/JCAA #280

**Tanggal:** 2026-07-27 | **Dari:** Go Frendi Gunawan (disusun lewat sesi Claude Code, atas permintaan Pak Amien)
**Untuk:** Mukhlis Amien (PI) | **Status:** jawaban atas Q1–Q4 dokumen 3 §8 + dua eksperimen penguat (E220, E221)

Saya membaca keempat dokumen paket, membaca **seluruh kode** E217/E218/E218b/E219 baris per baris, dan
menurunkan ulang angka-angka kunci dari CSV/JSON mentah sebelum menjawab apa pun. Di bawah: (§1) apa yang
saya verifikasi dan cocok/tidak, (§2) serangan yang saya coba lancarkan ke refutasi kita sendiri dan di mana
sisa lubangnya, (§3–6) jawaban Q1–Q4, (§7) presisi penulisan yang wajib dijaga di v0.2, (§8) dua eksperimen
baru yang saya usulkan — dengan pre-registrasi — beserta alasannya.

---

## 1. Verifikasi independen saya

| Klaim dokumen 3 | Sumber mentah | Hasil cek saya |
|---|---|---|
| E218 Stage A: hybrid−random di eval uniform = −0.033 (0/20) MaxEnt, −0.009 (4/20) XGB, −0.009 (4/20) RF | `e218_stageA_raw.csv`, pivot berpasangan per seed | **Cocok persis** (−0.0334/−0.0085/−0.0094; 0/20, 4/20, 4/20) |
| Eval hybrid: +0.007 (14/20), +0.015 (19/20), +0.010 (18/20) | sama | **Cocok persis** |
| E218b: own 0.721→0.844, common 0.699→0.602; Spearman +0.961/+0.886/−0.708 | `e218b_summary.csv`, `e218b_outcome.json` | **Cocok persis** (11 baris grid, tiap angka sama) |
| Fitur sungai +0.042, 60/60 positif | `e217b_raw_results.csv`, 5 seed × 4 desain × 3 algo | **Cocok persis** (+0.0424, 60/60) |
| Inflasi evaluasi-sendiri +0.041…+0.051, 15/15 | `e217b_raw_results.csv` | **Cocok** (rerata +0.0458; per-algo 0.0405–0.0505; 15/15, rentang seed +0.025…+0.073) |
| E219: Jaccard MaxEnt 0.684 vs 0.466; INT-1 −0.243 vs −0.281; densitas 0.01377 vs 0.00048; suitability +0.055 | `e219_outcome.json` | **Cocok persis** |
| Pipeline tervalidasi 0.750 vs 0.751 terbit; hard fraction 0.623 vs 0.62 | `e217b_raw_results.csv`, README E217 | **Cocok** |

Tidak ada satu pun angka headline yang gagal saya turunkan ulang dari file mentah. Jejak audit (Stage C yang
gagal) juga ada dan memang menunjukkan instrumen rusak seperti dilaporkan (auc_own 0.98 di pita terdekat).

## 2. Serangan yang saya coba ke refutasi kita sendiri

Sebagai orang ML di tim ini, tugas saya menembak premis Q2 sebelum menyetujuinya. Empat serangan, dua gugur,
dua meninggalkan sisa yang harus ditutup:

1. **"Common background uniform itu arbitrer."** Gugur. E218 Stage A memakai 4 evaluation background dan
   prediksi tajamnya (menang hanya di kandang sendiri) terkonfirmasi persis: hybrid menang 3/3 algoritma
   *hanya* di eval hybrid, 0/3 di tiga lainnya. Ini bukan kontingensi pilihan background.
2. **"AUC mengutuk AUC itu sirkular."** Gugur. Boyce (presence-only, availability dipatok) ikut dilaporkan
   dan tidak menyelamatkan hybrid (RF justru −0.095, 2/20). TSS konsisten.
3. **"Evaluation background ditarik dari frame *tanpa* buffer situs (E217b/E218 memakai `frame_all`),
   sedangkan training background dari frame ber-buffer. Kontaminasi sel dekat-situs di pool evaluasi bisa
   mengaburkan perbedaan desain."** Sisa. Efeknya simetris lintas desain jadi hampir pasti tidak mengubah
   ranking, tapi kalimat "hampir pasti" tidak boleh ada di paper yang sedang menuduh orang lain kurang
   kontrol. **Ditutup di E220 Bagian 2** (eval uniform dari frame ber-buffer; prediksi: ranking tidak
   berubah).
4. **"Boyce kita sendiri punya knob (lebar jendela, jumlah jendela). Kita memakai metrik yang belum
   dirobustifikasi untuk menghukum metrik lain."** Sisa. Implementasi Boyce kita memakai width = range/10,
   101 jendela. **Ditutup di E220 Bagian 3** (sapuan 3 lebar × 3 jumlah jendela; prediksi: tanda
   hybrid−random stabil).

Selain itu dua catatan minor: (a) sign count per seed sudah kuat, tapi reviewer metodologis akan minta uji
formal — **E220 Bagian 4** menambahkan Wilcoxon signed-rank ke semua kontras headline (gratis, dari CSV yang
sudah ada); (b) mutasi `base.HYBRID_HARD_FRAC` di E218b bersih (ada `finally` restore) — tidak ada bug
state bocor.

## 3. Q1 — Setuju klaim inti dicabut?

**Setuju, tanpa syarat tambahan.** Buktinya bukan satu eksperimen gagal, melainkan tiga lapis yang saling
bebas: dekomposisi (E217b: efek desain −0.014 vs fitur +0.042), matriks evaluasi (E218: tanda berbalik hanya
di kandang sendiri, 20 seed), dan mekanisme dosis-respons (E218b: +0.961/−0.708). Ditambah validasi pipeline
0.750-vs-0.751, tidak ada ruang untuk berdalih "beda implementasi". Klaim abstrak v0.1 tidak bisa
dipertahankan dengan cara apa pun yang jujur. Mencabutnya adalah satu-satunya langkah yang benar secara
ilmiah — dan, perlu dicatat, secara strategis juga: datang ke editor dengan koreksi diri + temuan pengganti
lebih kuat daripada menunggu reviewer ketiga yang menemukannya.

## 4. Q2 — Apakah "common evaluation background" tolok ukur yang benar?

**Ya — dan saya akan memperkuat alasannya di naskah.** Dalam bahasa ML: AUC adalah
P(skor(presence) > skor(negatif)) terhadap *sampel uji tertentu*. Mengganti pool negatif berarti mengganti
estimand yang diestimasi; membandingkan AUC lintas pool bukan "perbandingan yang lemah" melainkan
perbandingan dua besaran berbeda. Mematok test set adalah kondisi *minimal* agar perbandingan desain punya
arti — ini bukan preferensi kita, ini definisi prosedur pembandingan. Lobo et al. (2008) dan
Jiménez-Valverde (2012) mengatakan hal yang sama dalam bahasa SDM; kontribusi kita adalah menunjukkan
konsekuensi seleksinya (lihat Q3).

Tiga penajaman yang saya minta masuk v0.2:

1. **Beri nama estimandnya.** "AUC terhadap availability uniform atas frame studi" — karena Stage A sendiri
   menunjukkan nilai absolut bergeser antar common background (eval hybrid ~0.70 vs eval uniform ~0.66–0.71).
   Rekomendasi: uniform-availability sebagai estimand primer (interpretable: diskriminasi terhadap lokasi
   acak; konsisten dengan availability Boyce), matriks 4 background sebagai robustness.
2. **Tutup celah buffered-eval** (serangan 3 di atas) — E220 Bagian 2.
3. **Nyatakan eksplisit bahwa prosedur ini berlaku umum**: desain apa pun (bukan hanya milik kita) harus
   dinilai pada availability uji yang dipatok. Itulah yang membuat paper ini protokol, bukan sekadar koreksi
   diri.

## 5. Q3 — Apakah E218b cukup baru melawan "not entirely novel"?

Penilaian jujur saya sebagai pembaca literatur ML/SDM: **fenomenanya tidak baru; demonstrasi terkuantifikasi
+ konsekuensi keputusannya baru.** Jangan pernah kita klaim menemukan artefaknya — R1 benar, itu sudah
dikenal. Yang bisa dipertahankan, berlapis:

1. **Lobo (2008) / Jiménez-Valverde (2012)** bilang "AUC tidak komparabel lintas background" — sebuah
   *peringatan statis*. **E218b bilang sesuatu yang dinamis dan lebih jahat**: menyapu knob yang memang
   diputar praktisi, metrik yang dilaporkan dan generalisasi **bergerak berlawanan arah secara monoton**
   (Spearman +0.886 vs −0.708). Ini Goodhart's law terukur di presence-background modelling: mengoptimalkan
   metrik laporan secara sistematis memilih model yang lebih buruk. Setahu saya bentuk sebersih ini (satu
   knob, dosis-respons penuh, dua arah berlawanan, tiga algoritma) belum pernah didemonstrasikan di
   arkeologi maupun ENM.
2. **Self-case-study**: bukan simulasi — prosedur tuning yang *terbit* (E013 memilih hard_frac=0.30,
   maksimum yang ditawarkan) terbukti mendaki gradien yang salah. Ini membuat paper jadi bukti empiris
   bahwa patologi itu *benar-benar terjadi* di praktik, bukan kemungkinan teoretis.
3. **Konsekuensi level keputusan (E219)**: literatur ENM berhenti di metrik. Kita menunjukkan peta prioritas
   berubah (Jaccard 0.684→0.345) dan tidak stabil terhadap seed saja — satuan yang dipahami manajer cagar
   budaya adalah *sel mana yang didatangi*, bukan AUC. Ini jembatan ke relevansi arkeologis yang R2 minta.
4. **Keluarga masalah yang lebih luas**: inflasi yang kita ukur sekeluarga tapi berbeda mekanisme dengan
   inflasi akibat random CV di Ploton et al. (2020, *Nature Communications*) dan literatur spatial CV
   (Roberts et al. 2017, *Ecography*); serta terkait pilihan availability di Barve et al. (2011) dan pilihan
   statistik evaluasi di Fourcade et al. (2018, *Global Ecology and Biogeography* — bahkan "lukisan" pun bisa
   diprediksi kalau statistiknya dipilih sembarang). v0.2 wajib memposisikan diri eksplisit terhadap
   nama-nama ini — sekaligus menjawab permintaan R1 untuk "examples to relate to". (Kedua sitasi 2018/2020
   sudah saya verifikasi ke Crossref hari ini.)

Tambahan yang menurut saya **paling menaikkan daya tawar novelty**: formaliskan "gradien salah" menjadi
"seleksi salah". E218b menunjukkan kurvanya; **E220 Bagian 1** menunjukkan konsekuensi proseduralnya — aturan
seleksi yang dipakai praktisi (pilih konfigurasi dengan AUC tertinggi versi laporan) memilih konfigurasi
terburuk atau hampir terburuk dalam mayoritas seed, dengan biaya generalisasi terukur dan diestimasi jujur
(cross-fitted). "Tuning on the reported metric walks backwards" adalah kalimat yang bertahan di kepala
reviewer.

Kesimpulan Q3: E218b + E220 + E219 bersama-sama cukup baru. E218b sendirian tipis. Dan framing v0.2 harus
"quantified pathology + corrected protocol + decision consequences", bukan "discovery".

## 6. Q4 — Kepengarangan

Posisi saya: **tetap co-author di versi baru**, dengan tiga syarat tercatat yang saya minta dijaga PI:

1. v0.2 menyatakan pergantian kesimpulan secara eksplisit (bukan "revisi biasa") dan email pengungkapan ke
   editor benar-benar dikirim sebelum resubmit.
2. Disiplin klaim di §7 dokumen 3 ("yang TIDAK terbukti") dipertahankan kata per kata — terutama "different
   and consequential, never improved" untuk peta, dan larangan mengutip rasio n=2 sebagai presisi.
3. Angka yang masuk naskah lulus re-derivasi buta G1 (sebagian sudah; E220/E221 menambah kandidat headline
   baru yang ikut gerbang yang sama).

Catatan prosedural: dokumen ini disusun lewat sesi Claude Code atas permintaan Pak Amien untuk memerankan
review saya. Isinya adalah analisis teknis penuh saya, tetapi **persetujuan kepengarangan yang sah tetap
harus dinyatakan Pak Amien kepada saya secara langsung** (balasan atas dokumen ini cukup) sebelum apa pun
dikirim ke editor.

## 7. Presisi penulisan yang wajib dijaga di v0.2

Temuan saya saat verifikasi — kecil, tapi persis jenis yang ditangkap reviewer bermusuhan:

- **"Turnover 31–45%"**: angka itu = 1 − Jaccard (fraksi footprint gabungan desil-teratas yang hanya muncul
  di satu run). Dengan definisi "bagian top-desil suatu run yang tergantikan antar seed" baku, nilainya
  (1−J)/(1+J) = **18–29%** (XGB 31%→19%, dst.). Keduanya benar; bedanya denominator. v0.2 harus memilih satu
  definisi, menyatakannya eksplisit, dan tidak mencampurnya. E221 Bagian C melaporkan keduanya.
- **"~29×" densitas situs**: hanya boleh muncul dengan kalimat n=2 menyertainya di kalimat yang sama.
- **"No reliable benefit"**, bukan "no benefit" — Boyce XGB +0.041 (13/20) adalah sinyal lemah yang harus
  diakui apa adanya.
- Selisih −0.243 (E219) vs −0.163 (naskah) untuk Test 1: sudah benar dilaporkan sebagai koreksi berarah,
  pertahankan rumusannya.
- Judul: dari 3 kandidat di rencana revisi §3, saya dukung **kandidat 3** ("An Evaluation Artefact in
  Presence-Background Archaeological Modelling: Evidence from East Java and a Corrected Protocol") — paling
  jujur dan menjual protokol. Hook kandidat 1 ("the evaluation background is the result") bagus sebagai
  kalimat pembuka abstrak, bukan judul.

## 8. Dua eksperimen penguat yang saya usulkan (pre-registrasi ringkas)

Keduanya menyerang dua risiko terbesar (dokumen 3 §6): novelty (R1) dan relevansi arkeologis (R2).
Pre-registrasi penuh ada di `experiments/E220_wrong_direction_selection/DESIGN.md` dan
`experiments/E221_seed_ensemble_stability/DESIGN.md`.

### E220 — "Seleksi di metrik laporan berjalan mundur" + penutup dua celah robustness
- **Bagian 1 (utama):** sapuan hard_frac 0.0–1.0, **20 seed × 3 algoritma** (perluasan E218b yang 5 seed),
  mencatat auc_own, auc_common, TSS, Boyce per konfigurasi. Lalu simulasikan aturan seleksi praktisi:
  (a) argmax auc_own — aturan "cara naskah"; (b) argmax auc_common — aturan jujur (dievaluasi cross-fitted
  antar-paruh seed agar tidak optimis); (c) argmax Boyce — uji apakah metrik presence-only ikut jujur.
  **Prediksi terdaftar:** P1 aturan (a) memilih hard_frac ≥ 0.7 di ≥60% kasus; P2 pilihan aturan (a)
  terburuk/dalam 0.01 dari terburuk di ≥50% kasus; P3 biaya cross-fitted ≥ +0.05 AUC; P4 *fork* — jika
  Boyce ikut memilih hard_frac rendah, ia tervalidasi sebagai selektor jujur; jika tidak, kita laporkan
  Boyce lebih berisik dari harapan dan rekomendasi primer = common-background AUC.
- **Bagian 2:** eval uniform dari frame **ber-buffer** (celah 3 §2). Prediksi: ranking desain tidak berubah.
- **Bagian 3:** sensitivitas Boyce terhadap lebar/jumlah jendela (celah 4 §2). Prediksi: tanda stabil.
- **Bagian 4:** Wilcoxon signed-rank untuk semua kontras headline dari CSV yang sudah ada.

### E221 — Stabilitas ensemble seed + produk peta untuk figur dan jawaban R2
E219 menemukan instabilitas seed tapi tidak menyimpan petanya. E221 mengulang produksi peta **10 seed ×
3 desain × 3 algoritma** dengan peta tersimpan, lalu:
- **Bagian A:** kurva stabilisasi — Jaccard(ensemble-k, ensemble-10) terhadap k = 1…9 per desain×algoritma.
  Output: k* minimum untuk J ≥ 0.9, dan rekomendasi protokol ("ensemble ≥ k* seed") — mengubah temuan
  instabilitas menjadi panduan yang bisa dipakai BPCB.
- **Bagian B (jawaban R2 yang sebenarnya):** partisi frame menjadi *robust priority* (top-desil di ensemble
  ketiga desain), *design-contingent* (top-desil hanya di satu desain), dan *unstable*. Karakterisasi tiap
  himpunan (jarak jalan, elevasi, jarak gunung kanonik, jarak situs, densitas situs teramati) + ekspor
  peta divergensi (0–3 desain per sel) untuk figur blok F. Ini versi jujur-era-artefak dari permintaan
  two-stage R2-C: kita tidak mengklaim di mana situs terkubur; kita menunjukkan **prioritas survei mana yang
  kebal terhadap pilihan analitis arbitrer dan mana yang artefak dari pilihan itu** — persis pertanyaan yang
  harus dijawab sebelum ada orang dikirim ke lapangan.
- **Bagian C:** turnover seed dengan kedua definisi (presisi §7).

Biaya komputasi: E220 ≈ 95 menit, E221 ≈ 50 menit — keduanya memakai ulang mesin E217–E219, data yang sama,
tanpa dependensi baru.

## 9. Penilaian peluang & rekomendasi non-teknis

- Saya **mendukung mengirim email Verhagen segera** setelah keputusan PI (dokumen 4 §3 item 5), termasuk
  minta perpanjangan ke 2026-09-30. Dengan E218b + E220 + E221, email itu bukan lagi "kami menemukan klaim
  kami artefak" melainkan "kami menemukan patologi terukur + protokol korektif + bukti konsekuensi lapangan"
  — pesan yang jauh lebih kuat, dan Verhagen adalah penilai terbaiknya.
- Risiko terbesar tetap dua: novelty kalau E220 gagal memperkuat E218b (maka framing turun ke "rigorous
  confirmation + protocol"), dan perubahan lingkup prosedural kalau editor membaca ini sebagai paper baru
  (itulah gunanya email lebih awal).
- Yang sengaja TIDAK saya kerjakan: naskah v0.2 (menunggu blok A–B sesuai dokumen 4), commit repo (perlu
  izin PI), kontak editor (milik PI).

---

*Ditandatangani secara analisis: Go Frendi (lewat Claude Code). Semua perintah reproduksi tercantum di
dokumen 3 §7; tambahan E220/E221 di direktori eksperimen masing-masing.*
