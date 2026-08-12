# Kritik Sistem / Research-Designer Review — 2026-08-11

**Mode:** ORBIT · **Peran:** system/research designer · **Prinsip:** *simple is better, fail fast, pivot
early; santai dalam waktu, serius dalam standar ilmiah.*
**Status:** dokumen kerja internal — bukan publikasi, bukan sumber bukti (F10).
**Konteks:** ledger eksposur kosong (5/5), ME#19 stop-list kini keputusan PI. Sesi ini tidak
menambahkan eksperimen baru — ia menilai mesinnya.

---

## 0. Vonis ringkas

**Proyek ini telah membangun mesin *refutasi* kelas dunia, tetapi belum memiliki mesin *penemuan*.**

214 eksperimen, SIG G1–G10, pre-registration, re-derivasi buta, budaya koreksi diri (reversal P2,
E214, E217–E224) — dan keluarannya: **0 acceptance, 7 rejections**, flagship yang kini bercerita
tentang artefaknya sendiri. Mahkota tiga bulan terakhir adalah *meta-sains tentang mengapa judul
utama salah*. Itu sains yang bernilai, tapi bukan penemuan.

Rasio status: **142/214 (66%) SUCCESS, 3 FAILED, 34 UNKNOWN.** Tidak ada program dengan 66% sukses
dan 0 acceptance yang bisa disebut "sehat" — label SUCCESS telah hanyut menjadi "berjalan dan tidak
membantah desainnya", yang persis dilaporkan mesin konfirmasi. **Perasaan "belum puas" yang PI rasakan
adalah sinyal yang benar**, bukan keinginan: portofolio belum punya satu pun temuan positif yang
divalidasi eksternal, dan satu-satunya klaim yang bisa positif (eksperimen alami Jawa Barat) **belum
punya naskah**.

Kritik di bawah menembus lapisan yang sudah diserap manifesto v5.0 / SIG. Yang sudah benar — dua
mekanisme, F-CMPLX, G1–G10, standing falsification sebagai niat — tidak diulang. Yang dibedah adalah
tempat mekanismenya *ada tapi tak ditegakkan*, dan blind spot yang *belum masuk sistem*.

---

## 1. Tujuh risiko struktural

### R1 — Inflasi label sukses (masalah data, bukan hanya masalah sikap)
- 142 SUCCESS, 3 FAILED, 34 UNKNOWN. Indeks tidak bisa membedakan eksperimen **defensif**
  (menjelaskan ketidakhadiran), **ofensif** (mencari kehadiran), dan **disconfirming** (didisain untuk
  melukai tesis). KPI kesehatan "≥1 dari ~10 didisain untuk melukai" tak terlihat di indeks sama
  sekali.
- **Fix:** perpanjang `tools/scan_experiments.py` dengan klasifikasi `DEFENSIVE / OFFENSIVE /
  DISCONFIRMING / META` per eksperimen; laporkan rasionya setiap sesi. Hentikan label SUCCESS untuk
  "berjalan sesuai desain" — SUCCESS hanya untuk "mengubah peta portofolio".

### R2 — Angka hantu di dokumen kontrak (kelas volcanoes.csv, hidup)
- `lines/01_spatial/CLAUDE.md:92` menulis **"E209 (AUC 0.844 — a Hindu-Buddhist site detector)"**.
  E209 (dibaca langsung) berstatus **PHASE 1 scaffolding**, belum pernah menjalankan classifier;
  kill-criterion-nya <0.60, hipotesisnya ≥0.75. **0.844 tak punya sumber.** Ini kelas kesalahan yang
  sama dengan `volcanoes.csv` 7-gunung yang membunuh P7 — tetapi duduk di dokumen yang *dibaca pertama
  setiap sesi focus-mode*.
- Yang sama: **prosa bilang "224 eksperimen" (manifesto §3, handoff, memory), indeks bilang 214.**
  Fakta: 224 = nomor yang dialokasikan (E001–E224), 214 = folder lokal; 10 nomor (E021, E045–47,
  E053, E072, E077, E180, E203, E212) tak pernah dibuat, dua di antaranya (E053, E203) ada di repo
  eksternal. **Angka yang dipakai sebagai bukti skala tidak bisa diverifikasi dari dokumen yang
  mengklaimnya.**
- **Fix:** bangkitkan `tools/check_doc_sync.py` sebagai **kanari jalan** — dijalankan di awal setiap
  sesi, aturan pertama: *"tidak ada angka di dokumen hidup tanpa penunjuk ke sumbernya."* Rekonsiliasi
  hitungan: tulis "214 lokal (E001–E224)".

### R3 — Uji yang paling menentukan tidak pernah dijalankan, dan itu diketahui
- **E215: belum pernah ada studi mikrobotani (phytolith/starch) pada situs prasejarah mana pun di
  Jawa.** Uji ini membedakan "masyarakat tersebar berladang/arboriculture" (muncul di starch, tidak di
  pollen) dari "kosong" — persis celah yang E214 tinggalkan. Ini **satu inti tanah bisa
  menyelesaikan** apa yang komputasi tak bisa.
- Sementara itu GPU terus menghasilkan eksperimen komputasi. **Program ini lengkap secara komputasi
  dan kosong secara empiris.** Blocker-nya satu manusia (Castillo/UCL, PVMBG core) — sejak Maret belum
  terjadi.

### R4 — Eksperimen alami punya konfound bawaan dan belum punya naskah
- "Buni/Batujaya hadir, dataran rendah Jawa Timur vulkanik absen" adalah tulang punggung tesis.
  Dua cacat:
  1. **Konfound intensitas survei.** E109/E086 menetapkan defisit survei sebagai leverage terbesar
     sinyal ketidakhadiran. Reviewer akan bilang: "Buni/Batujaya ditemukan karena dicari; Jawa Timur
     tidak." Tanpa kontrol upaya survei (kompilasi jumlah ekskavasi/survei dua kawasan), letter ini
     mati oleh kritik yang proyek *sudah tahu*.
  2. **Belum ada naskah.** Prioritas #1 manifesto §2 dan "kemenangan eksposur jangka pendek" tidak
     terdraf.
- **Fix:** draf letter dengan kontrol survei **dibangun dari baris pertama**, dan framekan sebagai
  demonstrasi *recording bias*, bukan *burial*.

### R5 — Konflasi geografis: "Nusantara" ≠ "Jawa"
- Basis bukti Jawa-sentris (semua paper spasial, candi, inventori gunung). E214: Sumatra (~7500 BP)
  dan Borneo (Niah ~6000 BP) bertani **lebih awal** dari Jawa. E201: Filipina punya 275–340 situs
  pre-400 CE vs 0 di Jawa vulkanik (dijelaskan karst — E178).
- **Puzzle "400 CE" sebenarnya adalah puzzle "Jawa + adopsi aksara", bukan puzzle Nusantara.**
  Kegelisahan PI tentang "underrepresentasi budaya Nusantara" menunjuk tepat ke sini: alat proyek
  tidak menjangkau budaya non-Jawa, non-vulkanik yang cerita pre-400 CE-nya justru lebih kuat.
- **Fix:** disagregasi ruang lingkup piagam ("Jawa" vs "Nusantara"); perbandingan Jawa Barat–Jawa
  Timur diskopkan ke Jawa, sementara kanal E214 / P19 / manik / genderang perunggu memikul ruang
  lingkup Nusantara.

### R6 — Botol manusia adalah kendala pengikat sekarang, bukan kapabilitas AI
- E211: otorisasi menunggu **110 hari**. Palynologist outreach: menunggu. Amandemen L1: menunggu.
  DJKI: menunggu. Loop AI memproduksi lebih cepat daripada kapasitas keputusan PI.
- Manifesto §2 sudah benar: "agen paling berharga = satu manusia ahli yang skeptis, bukan AI yang
  lebih kuat." **Tapi itu belum terjadi.** Arsitektur mencatat diagnosisnya dan tidak mengeksekusi
  obatnya.

### R7 — Antrian paper adalah jebakan biaya-sunah + AutoResearch zombie
- P1 (tolak 2×, butuh WS-E + SIG), P5 (tolak 1×, statistik pusat E032 adalah FDR casualty, rewrite
  tertunda ~2 bulan), P9 (HOLD), P18/P19/P20/P22 (proposal), P21 (folder kosong). Semua ini
  mengonsumsi siklus workshop yang seharusnya ke pekerjaan decisive.
- `docs/AUTORESEARCH_CONCEPT.md` (Maret) adalah zombie: Program 3 (cascade) objeknya sudah pensiun;
  Program 2 (P21) terblokir GDPR; Program 4 (P20) butuh co-author; `tools/autoresearch/results/`
  **kosong**. Arsitektur yang indah, tak pernah dikirim, kini redundan dengan manifesto v5.0 §2.

---

## 2. Arsitektur kolaborasi manusia–AI

**Pemetaan kapabilitas — yang sudah benar:**
| Kerja | Paling cocok | Status |
|---|---|---|
| Re-derivasi deterministik (G1) | Skrip, bukan model | ✅ ada (`verify_headline_numbers.py`) |
| Sintesis prosa / kerangka argumen | Opus | ✅ dideklarasikan per line |
| Kerja mekanis (scan, format) | Sonnet | ✅ |
| Review skeptis lintas-model | Model lain, *diprogram untuk menolak* | ⚠ ada prompt, tidak distruktural |
| Keputusan + domain-sanity (G2) | PI | ⛔ **botol seri** |

**Tiga cacat arsitektur:**
1. **Fungsi adversarial tidak terpisah secara struktural.** Loop yang sama yang memproduksi juga yang
   meninjau. E217–224 membuktikan self-critique bekerja — tetapi hanya ketika digerakkan oleh krisis
   (P7) atau gerbang eksplisit. Sebagai rutinitas, ia nol. Solusi: fungsi "pembunuh tesis" diberi
   budget, jadwal, dan model yang diprogram untuk menolak — bukan dibayar untuk menyetujui.
2. **Gerbang 4-AI (masterpiece) menguji prosa, bukan derivasi — dan empat LLM berbagi substrat**
   (F9). Tiga dari empat model mungkin salah pada artefak yang sama. Fix: **G1 re-derivasi WAJIB
   sebelum gerbang 4-AI**, dan setiap model diberi satu prompt eksplisit untuk menolak secara
   struktural (Validity×Centrality), bukan memberi vonis holistik.
3. **Keputusan manusia tidak di-batch dan tidak punya default.** E211 menunggu 110 hari bukan karena
   PI ragu, tapi karena tidak ada slot keputusan. Fix: **satu "decision hour" mingguan** (bisa dalam
   Mata Elang), keputusan diberi default ("otorisasi E211 kecuali PI menolak"), dan daftar keputusan
   yang menggelap dipaparkan dengan umurnya.

**Kalimat yang paling penting dalam kritik ini:** kapabilitas AI BUKAN kendala. Satu arkeobotanis
dengan satu inti tanah lebih menentukan nasib tesis daripada seribu eksperimen komputasi berikutnya.

---

## 3. Framework testing interaksi + klasifikasi kegagalan

Perpanjangan dari SIG (F1–F10, R1). Yang dibutuhkan bukan kode baru, tapi **pengujian yang berdiri
(standing), bukan reaktif**. Matriks:

| Interaksi | Jenis | Uji | Kode kegagalan yang tertangkap |
|---|---|---|---|
| AI → PI (klaim) | agent→human | **T0 re-derivasi segar berdiri** — tiap angka headline sebelum keluar direkomputasi buta dari raw (perluas dari gerbang jadi layanan) | F2, F1, F6 |
| PI → AI (instruksi) | human→agent | **T3 escape-question audit** — tiap paper/tesis harus menjawab "hasil apa yang akan meng-update kerangka MELAWAN tesis?" dengan nama eksperimen+kanal konkret; tak bisa menjawab ⇒ klaim diparkir. (Operasionalisasi F10) | R1, F8 |
| AI → AI (4-AI gate) | agent→agent | **T4 correlated-error check** — sebelum mempercayai konsensus, sebutkan substrat bersama (training/literatur/dataset); minta ≥1 re-derivasi independen, bukan kesepakatan prosa; tiap model diberi prompt menolak | F9, R1 |
| AI → data/skrip | agent→tool | **T5 robustness battery** (bootstrap/jackknife/permutasi, E121/E159 pola) + cek kanonikal G3 | F2, F7 |
| AI → portal eksternal | agent→world | **G12 post-submission re-download** (pola P2: unduh balik, bandingkan) + verifikasi landing/DOI via GET | F8-adjacent |
| Human → human (co-author) | human→human | **T6 adversarial review berwujud** — G10 memberi dokumen tertulis "saya mencoba membunuh klaim ini, ini yang bertahan", bukan tanda tangan | F1, F4 |
| PI → keputusan | human→system | **T7 aging report** — uji 30-hari F8: "ada artefak yang mencapai hakim eksternal?" + daftar keputusan dengan umurnya | F8 |

**Klasifikasi kegagalan:** pertahankan F1–F10/R1 (jangan tambah kode — F-CMPLX berlaku juga untuk
taksonomi). Yang kurang bukan kode baru, tapi **siapa menjalankan uji apa, kapan** — dan itu diselesaikan
oleh jadwal berdiri + ledger kritik (§4).

**T2 status-label audit** (dari R1) dan **T1 kanari dokumen** (dari R2) menutup dua lubang yang belum
dijangkau uji di atas: yang pertama soal indeks, yang kedua soal dokumen kontrak itu sendiri.

---

## 4. Mekanisme seleksi kritik (ledger kritik)

Rubrik Validity×Centrality sudah ada di SIG (G9/G6) sebagai prosa. **Operasionalisasikan menjadi
ledger berdiri** — karena tanpa itu dua penyakit hidup berdampingan: kritik valid yang diabaikan, dan
kritik invalid yang jadi mesin penunda (F8).

**`docs/CRITIQUE_LEDGER.md`** — setiap kritik yang masuk (dari peer review, dari review AI, dari PI,
dari kritik ini) dicatat: sumber · tanggal · klaim yang dituju · Validity (0–2) · Centrality (0–2).

Empat disposisi, **tanpa drop senyap**:
| Disposisi | Aturan |
|---|---|
| **FIX** | Validity≥1 & Centrality=2 → data baru ATAU downgrade klaim; rewording dilarang |
| **FIX-CHEAP** | Validity≥1 & Centrality=1 → diperbaiki jika ≤1 sesi; jika tidak, PARK dengan kondisi unpark |
| **PARK** | Validity≥1 & Centrality=0 → dicatat dengan pemicu unpark |
| **REJECT-with-reason** | Validity=0 → **katup anti-penunda**: kritik boleh ditolak secara sadar, penolakannya diaudit |

Dua disiplin: (1) setiap kritik mendapat baris disposisi — kritik dengan disposisi REJECT **berhenti
memblokir antrian**; (2) veto PI dicatat, bukan dibisukan. Ini menjawab langsung permintaan "menentukan
secara sadar kritik mana yang diakomodasi dan mana yang diabaikan", dan mencegah dua mode kegagalan
sekaligus.

---

## 5. Rekomendasi: TERMINATE / PIVOT / REFACTOR / BUILD / KEEP

### TERMINATE / arsipkan (obsolete, redundan, tak testable)
- `tools/autoresearch/` + `docs/AUTORESEARCH_CONCEPT.md` → arsip (tak terimplementasi, superseded).
- `papers/P21_colonialmine` (folder kosong) → ganti dengan baris I-NNN; P20/P22 tetap di drafts.
- P5 ritual-clock **dalam bentuk sekarang**: reframe jujur ≤2 sesi, atau PARK dengan unpark conditions
  (statistik pusatnya FDR casualty; rewrite terus tergelincir).
- P7 preprint correction: **tetap parkir** (jangan dibuka lagi — bukan kendala).

### PIVOT (ke arah temuan positif — ini obatnya, bukan lebih banyak eksperimen)
1. **P11 → SPAFA minggu ini** — termurah, siap. ⚠ Tuntaskan dulu sweep WS-E kanonikal untuk P11
   (defek 7-gunung masih hidup di antrian).
2. **Draf letter eksperimen alami Jawa Barat (2–3 sesi)** — dengan kontrol upaya survei dari baris
   pertama. Ini tulang punggung tesis DAN kemenangan eksposur.
3. **Mulai kanal NLP literatur abu-abu** (manifesto #2a): tambang laporan Balai Arkeologi/BPCB/PUSLIT
   untuk horizon pre-400 M yang **sudah tercatat tapi terlupakan**. Kanal positif termurah, kekuatan
   inti PI, RTX 4080 + DeepSeek. Ini tempat "sisi gelap yang dibuka AI" menjadi nyata.

### REFACTOR (mekanisme yang ada → dibuat berdiri)
- Rekonsiliasi hitungan ("214 lokal, E001–E224"); tulis ulang "224" di manifesto/handoff/memory.
- Bangkitkan `check_doc_sync.py` sebagai kanari awal-sesi (T1). Hapus "AUC 0.844" E209 dari
  `01_spatial/CLAUDE.md` atau ganti dengan penunjuk ke target yang punya sumber.
- Sahkan **G11** (gerbang klaim non-numerik) dan **G12** (re-download pasca-submission) yang sudah
  diusulkan di JOURNAL.
- Terapkan T2 (label status ofensif/defensif/disconfirming di indeks).

### BUILD (dua komponen arsitektur yang hilang)
- **`docs/CRITIQUE_LEDGER.md`** (§4) — mekanisme seleksi kritik.
- **Jadwal falsifikasi berdiri** — manifesto bilang "1/kuartal"; tanpa pemaksa, itu niat. Jadwalkan
  yang berikutnya SEKARANG dengan tanggal tetap dan model yang diprogram untuk menolak (kandidat:
  menuntaskan E216 ke submission / sintesis E215→E225 / re-derivasi nomor inti P1).

### KEEP (jangan disentuh — kolaborasi aktif)
P17 (under review, koreksi terkirim) · P2 (bersama jurnal — kejar metadata portal) · P8 (under
review) · D1/D2 (terbit) · MASTERPIECE Phase 0→1 (lambat, dipertahankan) · repo genetics (eksternal
by design) · keputusan regenerasi model dashboard (keputusan model PI, bukan cleanup).

---

## 6. Sisi gelap: arah substantif yang bisa dibuka AI

Kegelisahan PI — "mengapa peradaban Nusantara dimulai 400 M setelah adopsi aksara India?" — adalah
pertanyaan yang tepat, dan data proyek sendiri sudah memberi jawaban parsial yang **belum dirangkai
menjadi klaim**: pre-400 M Nusantara **bukan** cerita Jawa, dan tafonomi vulkanik **bukan** satu-satunya
— bahkan bukan yang utama — penghapusnya. Lima kanal yang di bawahrepresentasi:

1. **Arsip penghapusan (*erasure archive*)** — NLP atas arsip kolonial (D1, E211, GLOBALISE) + rekaman
   Indologis/historiografis untuk **mengukur tafonomi genre**: apa yang dicatat, ditekankan, dan
   dihapus, serta bagaimana "pre-400 M = belum beradab" menjadi default. Ini "sisi gelap yang dibuka
   AI" secara harfiah: mengukur keheningan arsip. AI-native, data publik, kekuatan inti PI.
2. **Bukti positif selektif-selamat (*selective survival*)** — genderang perunggu (E204: ~30 di Jawa
   Timur vulkanik), manik kaca Jatim (abad 5–8 M), Dong Son, koin: masyarakat maritim yang canggih dan
   terintegrasi-perdagangan, jejak tahan-uburnya lolos dari penguburan. Reframe I-146. **Ini bentuk
   positif dari tesis.**
3. **Uji decisive yang belum dijalankan** — phytolith/starch (E215/I-125): hipotesis
   "tersebar/berladang" membuat prediksi mikrobotani spesifik dan testable. Satu inti, satu kolaborator.
   **Aksi bernilai tertinggi di seluruh program.**
4. **Bentang terendam** — batimetri GEBCO + kurva muka laut → peta target "situs pesisir pre-400 M
   yang hilang" (bentuk ofensif L2). Data gratis, bisa dikerjakan malam ini. Mengubah lapisan defensif
   menjadi prediksi positif.
5. **Kedalaman waktu budaya hidup** — I-139 "blind spot terbesar": wayang (E205: 20–30% tanpa sumber
   India, Semar), slametan, kosmologi gunung, kosa kata ritual Tengger (I-027 READY). Tanpa fieldwork.

**Meta-point:** kegelisahan PI adalah sinyal yang benar. Program telah mengoptimalkan koherensi
internal dan rigor dengan mengorbankan satu temuan positif yang decisive. Obatnya bukan eksperimen ke-215
— obatnya adalah pivoting ofensif (satu deteksi positif), uji decisive yang belum dijalankan (satu
inti), dan kanal arsip-penghapusan yang menamai "sisi gelap" yang PI rasakan.

---

## 7. Keputusan yang saya minta dari Pak Amien (satu decision hour, ≤30 menit)

1. **Otorisasi E211** (default: YA — ia menyuplai keempat trek PhD DAN kanal arsip-penghapusan).
2. **Prioritas 2 minggu ke depan:** P11→SPAFA + draf letter Jawa Barat (YA/tidak)?
3. **Kanal NLP literatur abu-abu** dijadwalkan sesi berikutnya (YA/tidak)?
4. **Outreach arkeobotanis** (Castillo/UCL atau BRIN) — layak dicoba dengan template email (≤$0)?
5. **Sahkan G11 + G12**, aktifkan ledger kritik `docs/CRITIQUE_LEDGER.md`, jadwalkan falsifikasi
   berikutnya dengan tanggal tetap.
6. **Rekonsiliasi hitungan:** konfirmasi "214 lokal (E001–E224)" sebagai standar penulisan.
7. Amandemen L1 — **disaggregasi "Jawa" vs "Nusantara"** pada piagam (YA/tidak)?

> ⚠ **Peringatan diri (F8).** Kritik ini hanya bernilai jika 2 minggu ke depan berisi *pengiriman*
> (P11, letter Jawa Barat, email outreach). Audit berikutnya ditunda sampai salah satu tindakan itu
> berhasil atau gagal di hakim eksternal. Ini bukan undangan untuk menulis kritik berikutnya — ini
> resep untuk satu tindakan.
