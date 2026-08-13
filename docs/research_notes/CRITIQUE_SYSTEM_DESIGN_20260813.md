# Kritik Sistem / Research-Designer Review — Putaran 2 — 2026-08-13

**Mode:** ORBIT · **Peran:** system/research designer · **Prinsip:** *simple is better, fail fast, pivot
early; santai dalam waktu, serius dalam standar ilmiah.*
**Baseline:** `CRITIQUE_SYSTEM_DESIGN_20260811.md` (putaran 1) + eksekusinya (commit `66e32e6`,
2026-08-12 12:34). HEAD bersih, tidak ada commit setelah eksekusi.
**Metode putaran ini:** 3 audit subagent independen (kontrak 7 line · eksekusi C001–C009 vs disk ·
inventori portfolio/tools) + pembacaan langsung dokumen inti. Setiap klaim kritik di bawah memakai
penunjuk file:baris — aturan baru yang lahir dari kegagalan putaran 1 (lihat R9).
**Konteks:** ledger eksposur kosong; stop-list ME#19 menunggu keputusan PI (WORKSTATE §1). Kritik ini
tidak menambahkan eksperimen; ia mengaudit mesin — dan mesinnya ditemukan sedang **drift di tingkat
meta**.

---

## 0. Vonis ringkas

**Putaran 1 bekerja — lebih baik daripada kritik mana pun dalam sejarah proyek ini.** Dalam 48 jam
setelahnya: ledger eksposur kosong (P17 koreksi terkirim, D1+D2 terbit dengan DOI, P7+PhD diparkir),
G11+G12 sah, ledger kritik berdiri, P11 siap-kirim konten, skeleton Jawa Barat dengan kontrol
upaya-survei sebagai seksi kelas-satu. Itu eksekusi nyata.

**Tetapi audit putaran ini menemukan bahwa peralatan yang ditambahkan putaran 1 sudah menunjukkan
penyakit yang sama yang ia bangun untuk menyembuhkan — dalam 48 jam:**

1. **Kanari MERAH, tidak terlihat.** `tools/check_doc_sync.py` — yang oleh putaran 1 disebut "bangkitkan
   sebagai kanari awal-sesi" — **sudah ada sejak 2026-03-30** dan kini berjalan exit 1: L2/L3/EVAL/L1
   masih menulis "175 experiments", pola WORKSTATE tak lagi cocok. Repo menyimpan kanari merah selama
   ≥2 hari tanpa satu sesi pun menyadarinya. "Bangkitkan" sebenarnya "jalankan".
2. **C003 ditandai DONE padahal target yang disebut eksplisit tak tersentuh.** Ledger menulis
   "✅ DONE 2026-08-11 (214 lokal, E001–E224)" — tetapi `docs/drafts/manifesto.md` (yang *disebut nama*
   dalam perintah perbaikannya: "tulis ulang '224' di manifesto/handoff/memory") masih menulis "224
   eksperimen" tiga kali (:10, :72, :107). **Ledger — mekanisme anti drop-senyap — mencatat klaim
   eksekusi yang salah.** Aturan baru mutlak: status DONE = artefak yang disebut namanya terverifikasi,
   bukan niatnya.
3. **Perbaikan E209 salah arah (kelas WRONG-FIX).** Putaran 1 menyebut "AUC 0.844 tak punya sumber" —
   **faktanya sumbernya ada**: `experiments/E209_satellite_ml_classifier/FINDINGS_v1_20260422.md:63`
   (RF 5-fold, 0.844 ± 0.060) + `results/classifier_baseline.json` (`mean_auc: 0.84416`). E209 memang
   menjalankan classifier fase-1. Perbaikan lalu mengganti angka bersumber dengan klaim baru yang
   **bertentangan dengan folder eksperimennya sendiri** ("belum ada AUC", `lines/01_spatial/CLAUDE.md:92`).
   Yang benar bukan menghapus — angka itu ada tapi **tidak layak lapor** (random-CV, seed tunggal,
   optik saja; melanggar aturan line 1–3 proyek sendiri). Kritik pun bisa membawa angka hantu: putaran 1
   menulis "dibaca langsung" padahal FINDINGS_v1 tidak dibaca.
4. **P11 mengulangi persis pola P2 yang proyek ini sendiri dokumentasikan.** "Siap-kirim konten" sejak
   08-11 15:11; kelima syarat (E069 kanonik, G9, format SPAFA, G8, konversi+submit) **tanpa tanggal**;
   tidak ada yang bergerak dalam 2 hari. Pola "siap-X-hari-tanpa-kirim" kini menjadi mode kegagalan
   bernama yang diulang oleh antrian berikutnya.
5. **Decision hour diusulkan, tidak pernah dijadwalkan.** E211 kini **112 hari** menunggu. Tujuh
   pertanyaan PI putaran 1 belum terjawab 2 hari kemudian. Mekanisme yang usulannya sendiri memakan
   satu sesi tidak punya mekanisme untuk memaksa dirinya ada.

**Diagnosis tingkat sistem:** setiap sapuan konsistensi sejauh ini memperbaiki lapisan yang dilihatnya
dan meleset dari lapisan di sebelahnya. Sapuan 08-11 memperbaiki STATE.md + WORKSTATE; ia melewatkan
CLAUDE.md kontrak (≥15 blok basi), manifesto, EVAL.md, CANONICAL pointer P2. **Sapuan manual tidak
berskala; hanya executable berdiri yang berskala.** Putaran ini menurunkan jumlah komponen proses dan
mengganti prosa dengan skrip + pemicu, bukan menambah kerangka baru.

---

## 1. Audit eksekusi putaran 1 (C001–C009 vs disk)

| Item | Klaim ledger | Realitas disk (2026-08-13) | Vonis |
|---|---|---|---|
| C002 E209 phantom | ✅ DONE | Angka dihapus dari 01 CLAUDE, **diganti klaim yang bertentangan dengan FINDINGS_v1 E209** | **WRONG-FIX** (R9) |
| C003 rekonsiliasi 224 | ✅ DONE | WORKSTATE/memory/indeks = 214 ✓; **manifesto :10,:72,:107 masih 224; L2/L3/EVAL/L1 masih 175; kanari MERAH** | **DONE overstated** |
| G11/G12 di SIG | — | LANDED (SIG :70–85); template sign-off masih G1–G10 (kosmetik) | LANDED |
| C005 skeleton Jawa Barat | IN PROGRESS | LANDED dengan §4 kontrol survei 3-lapis (150 baris) | LANDED |
| C008a AutoResearch | OPEN | Marker 🅿 SUPERSEDED ditambahkan; file belum dipindah; `results/` kosong | PARTIAL |
| C008b P5 park/reframe | OPEN | PARKED.md tidak ada; reframe tidak dimulai; ultimatum putaran 1 kedaluwarsa | NOT LANDED |
| C001 T2 label | OPEN | `scan_experiments.py` tidak punya klasifikasi DEFENSIVE/OFFENSIVE/DISCONFIRMING | NOT LANDED |
| C007 decision hour | OPEN | Tidak ada jadwal berdiri di mana pun (hanya usulan) | NOT LANDED |
| C009 outreach arkeobotanis | OPEN (PI) | Draf Castillo 10 Jun tak terkirim; tidak ada draf baru; lead BRIN Vida (8 Jun) tak tersentuh | NOT LANDED |
| C004 E215 unpark | OPEN (PI) | Sama dengan C009 | NOT LANDED |
| C006 L1 disagregasi | OPEN (PI) | Tidak ada bahasa Jawa/Nusantara di L1 | NOT LANDED |
| P11 5 syarat SPAFA | "siap-kirim konten" | Akurat untuk konten; **5 syarat semua terbuka, tanpa tanggal** | PARTIAL (R12) |
| Jadwal falsifikasi berdiri | BUILD | Manifesto §2.3: "1/kuartal" — **tanpa tanggal berikutnya** | NOT LANDED |

**Skor: 4 LANDED, 3 PARTIAL, 5 NOT, 1 WRONG-FIX.** LANDED semuanya adalah barang yang bisa dieksekusi
Claude sendiri. NOT semuanya adalah barang yang butuh PI atau butuh jadwal. Itu bukan kebetulan —
lihat R14.

---

## 2. Risiko struktural baru (R8–R14; penomoran melanjutkan putaran 1)

### R8 — EVAL.md: zombie di daftar "binding gate"
WORKSTATE §6 menyebut `docs/EVAL.md` sebagai **binding gate**. Isinya (terakhir diperbarui
2026-03-16): "Integrated Tautology Verdict (E013+E014): **CONDITIONAL PASS**" — persis kerangka yang
ditarik di P2 v0.2; P7 masih tertulis live (Antiquity); P9 "next: submit to Cornell"; P11 v0.3→Cornell;
"175 experiments (as of 2026-04-01)". **Dokumen yang dianggap mengikat memuat klaim yang sudah ditarik.**
Ini kelas volcanoes.csv, tapi lebih buruk: ia dinobatkan sebagai gerbang. Putaran 1 melewatkannya
karena menyapu `lines/`, bukan daftar gerbangnya sendiri.
**Fix:** EVAL.md diturunkan menjadi *pointer* (setiap seksi menunjuk ke line STATE / SIG / index),
atau diarsipkan. Gerbang yang mengikat = SIG (G1–G12) + F9/F10 + kanari. Jangan ada dokumen gerbang
kedua yang memuat angka.

### R9 — Kelas angka hantu kini punya tiga sub-kelas; perbaikan yang salah bisa menciptakan hantu baru
Kasus E209 mengajarkan: "angka tanpa sumber" bukan satu kelas, tapi tiga —
(a) **hantu murni** (tak ada sumber di mana pun — kelas E209 yang *dikira* putaran 1),
(b) **bersumber tapi tidak layak lapor** (E209 sungguhan: random-CV, 1 seed, optik saja — melanggar
aturan line 1–3; pengobatannya **tandai + tunjuk ke aturan yang dilanggar**, bukan hapus — menghapus
data adalah pemalsuan jenis baru),
(c) **pointer hantu** (kontrak menunjuk ke file yang salah — `papers/P2_settlement_model/CANONICAL.md:3`
masih menamai `submission_remote_sensing_v0.3.tex` sebagai "current working manuscript" padahal yang
tersubmit adalah `submission_jcaa_v0.2`).
**Fix:** kanari diperluas untuk (b) dan (c): pernyataan kontrak tentang status eksperimen harus cocok
dengan README + `results/` eksperimen itu sendiri (uji satu baris), dan setiap pointer "current" harus
memverifikasi file targetnya ada dan termodifikasi terakhir.

### R10 — Meta-drift: peralatan audit ikut drift, dan tidak ada yang menjaga para penjaga
Kanari merah 2 hari tanpa diperhatikan; ledger mencatat DONE yang salah; taksonomi putaran 1 pagi
(T1–T6, F-EXP…F-SUBST di JOURNAL :9341) dan malam (T0–T7, di dokumen kritik) **bertabrakan: "T1"
berarti dua hal berbeda dalam dua dokumen aktif**, dan "F-" berarti dua keluarga berbeda. F-CMPLX
dilanggar oleh taksonominya sendiri — dua sistem huruf baru lahir dalam 24 jam.
**Fix:** satu namespace, tidak ada yang lain: SIG memiliki F1–F10, R1, G1–G12; ledger memiliki C-NNN;
yang lain pensiun. Kanari dipasang ke awal-sesi **di CLAUDE.md root** (bukan di prosa kritik) dengan
aturan: merah ⇒ perbaiki sebelum karya baru. Kritik harus membawa penunjuk file:baris untuk setiap
klaim tentang eksperimen (G3 untuk dokumen kritik itu sendiri).

### R11 — Infrastruktur "ide tidak pernah dibuang" hidup-mati: TRIGGER_MAP & IDEA_REGISTRY
TRIGGER_MAP: 27 pemicu, 22 "FIRED" — **semuanya bertanggal 10–17 Maret 2026**. Tidak ada yang menembak
selama 5 bulan padahal ~60 eksperimen baru (E150–E224) lahir di periode itu. Sebuah peta pemicu yang
tidak pernah menembak bukan peta — ia dekorasi yang menyatakan sistem masih hidup. IDEA_REGISTRY:
baris tanggal 2026-03-16 tapi isi sampai Juni; 8–9 dari 11 entri READY basi atau yatim (I-010/I-130
sudah SELESAI via E150/E171 tapi tak diperbarui). MASTERPIECE + P0: dorman 64 hari tanpa PARKED.md.
**Fix:** TRIGGER_MAP dapat invariant: "FIRED terakhir" wajib segar (≤2 siklus Mata Elang) — bila tidak,
map diaudit atau dipensiunkan; bukan dibiarkan. Sama untuk "Updated" pada setiap dokumen governance:
tanggal basi = temuan, bukan catatan kaki.

### R12 — Antrian paper: ultimatum kedaluwarsa tanpa konsekuensi; zombie fisik bertahan
P5: putaran 1 memberi ultimatum ("reframe jujur ≤2 sesi, atau PARK") — **keduanya tidak terjadi**;
folder tak tersentuh sejak 10 Jun. Zombie fisik: `papers/P3_burial_depth/` (kosong, 171 hari),
`papers/P21_colonialmine/` (kosong, 135 hari), `models/` (kosong sejak lahir), `deploy/volcarch-dashboard`
(gitlink repo bersarang, tersupersede oleh `tools/dashboard.py`, tak pernah deploy),
`tools/references/autoresearch_karpathy/` (repo git vendored bersarang), `tools/globalise_pipeline/`
(tersupersede E211), 8 berkas CODEX/handoff Feb di `.claude/` yang masih terlacak git. 18/26 berkas di
`docs/drafts/` yatim >90 hari tanpa kolom pemilik. **Fix:** eksekusi ultimatum — PARK atau reframe,
bukan perpanjangan senyap. Folder kosong: hapus atau ganti satu baris I-NNN di IDEA_REGISTRY.
Repositori bersarang: arsip atau hapus. Setiap kematian dicatat, tidak dibiarkan membusuk.

### R13 — Botol keputusan manusia: batch tidak terjadi, dan peta peran model kini tanpa dokumen
Decision hour: usulan, bukan mekanisme. Tujuh keputusan putaran 1 menggantung. E211 = 112 hari.
**Dan fakta baru:** model sesi kini `deepseek-v4-pro[1m]` (dipilih PI 2026-08-13), sementara kontrak
line merekomendasikan Opus/Sonnet dan gerbang 4-AI mengasumsikan empat model Claude. Peta kapabilitas
putaran 1 menyebut *peran* (sintesis / mekanis / adversarial / keputusan) tapi mengikatnya ke *merek*.
Di bawah model yang berubah-ubah, peran tidak boleh dibawa merek.
**Fix:** tulis peta peran model (sintesis ↔ adversarial ↔ mekanis ↔ verifikasi) yang netral-merek di
satu tempat; adversarial wajib model *lain* dari yang memproduksi klaim (DeepSeek tersedia murah untuk
volume; siapa yang menolak diprogram, bukan dibeli). Dan keputusan PI dibatch hari ini — §8 adalah
agendanya.

### R14 — Pivot ofensif putaran 1 belum dijalankan satu pun — padahal ledger sudah kosong
Tiga PIVOT putaran 1 (P11→SPAFA "minggu ini", letter Jawa Barat 2–3 sesi, kanal NLP abu-abu) statusnya:
P11 konten-siap-tanpa-tanggal; letter ada skeleton; kanal NLP belum terjadwal. Ledger kosong berarti
stop-list gugur — tetapi §6 WORKSTATE masih menulis "⚠ ME#19 STOP-LIST IS ACTIVE" sebagai instruksi
berdiri (kontradiksi dengan §1-nya sendiri). **Antara "ledger kosong" dan "pekerjaan ofensif dimulai"
ada celah yang diisi oleh… review putaran berikutnya.** Ini F8 dengan kostum baru: memproduksi kritik
adalah keluaran paling andal sistem saat ini. Aturan baru (masuk ledger): **audit berikutnya dipicu
oleh peristiwa kirim (P11 terkirim) ATAU interval 14 hari — bukan oleh sesi.** Kritik adalah pemicu
aksi, bukan pengganti aksi.

---

## 3. Arsitektur kolaborasi manusia–AI (putaran 2)

**Yang terbukti bekerja (pertahankan):**
- **G1 re-derivasi buta** — alat terkuat di gudang. Menangkap angka hardcode di **figur** yang nyaris
  terkirim bersama P11 (JOURNAL :9453-9456). Setiap naskah yang lewat G1 menjadi lebih jujur.
- **Pola sesi-portal** — P2, P17, D1/D2 ditutup oleh sesi yang *menjalankan portal*, bukan yang
  menyempurnakan naskah. Ini korelasi empiris, bukan pendapat.
- **Ledger + disposisi** — berfungsi; kegagalan integritasnya (C003) justru membuktikan nilainya
  karena tertangkap oleh audit.
- **Pra-registrasi** (E217–E223 sebagai model) — tidak tersentuh.

**Yang masih patah (dari putaran 1, belum diperbaiki):**
1. Fungsi adversarial tetap tanpa jadwal — program falsifikasi "1/kuartal" tanpa tanggal berikutnya.
   → **Tanggal tetap: 2026-09-15** (kandidat: E216 menuju submission / re-derivasi inti P1).
2. Gerbang 4-AI masih menguji prosa, bukan derivasi; G1-wajib-sebelum-gerbang belum menjadi aturan
   tertulis di SIG.
3. Decision batching: §8.

**Yang baru (belum ada di putaran 1):**
4. **Dokumen kritik perlu gerbangnya sendiri.** E209 membuktikan kritik bisa membawa angka hantu.
   Aturan: setiap klaim kritik tentang status/angka eksperimen wajib memakai penunjuk file:baris,
   dan perbaikan yang diusulkan harus menyebut sub-kelas (a)/(b)/(c) dari R9.
5. **Peta peran model netral-merek** (R13). Kapabilitas AI bukan kendala — tetap benar — tetapi
   pembagian peran harus bertahan saat model berganti.

---

## 4. Evaluasi framework testing (T0–T7 putaran 1 vs realitas disk)

| Uji | Dirancang sebagai | Realitas 2026-08-13 |
|---|---|---|
| T0 re-derivasi berdiri | layanan | Hanya `verify_headline_numbers.py` P2; tidak digeneralisasi |
| T1 kanari dokumen | script berdiri | `check_doc_sync.py` ada (30 Mar), **MERAH, tidak dipasang ke awal sesi** |
| T2 label status | kolom indeks | Tidak ada |
| T3 escape-question | audit berdiri | Hanya pertanyaan permanen di ledger (baik) — belum dipakai pada klaim hidup |
| T4 correlated-error | cek berdiri | Prosa |
| T5 robustness | battery | Pola ada (E121/E159), tidak terstandarisasi |
| T6 adversarial human | gerbang | Outreach belum terjadi |
| T7 aging report | laporan | WORKSTATE §1 kolom "Days waiting" = satu-satunya yang benar-benar berdiri ✓ |

**Putusan: 1 dari 8 berdiri.** Kerangka 80% prosa. Prinsip *simple is better*: **runtuhkan ke tiga
executable berdiri** — (1) kanari `check_doc_sync.py` yang diperluas + dipasang ke CLAUDE.md root,
harus hijau tiap sesi; (2) kolom aging WORKSTATE §1+§4 (sudah ada; §4 ditambah kolom umur); (3) skrip
re-derivasi per-paper (pola `verify_headline_numbers.py`). Sisanya menjadi **peristiwa bertanggal di
ledger**, bukan uji berdiri. Pensikan penomoran T ganda (R10) — taksonomi uji ikut namespace tunggal.

---

## 5. Mekanisme seleksi kritik — tiga aturan tambahan

Ledger putaran 1 benar dan dipertahankan. Tiga amandemen dari temuan putaran ini:

1. **DONE = artefak bernama terverifikasi.** Status DONE hanya sah bila artefak yang disebut dalam
   disposisi (bukan niatnya) ada di disk. C003 dikoreksi: **PARTIAL, dibuka kembali** — manifesto
   dikerjakan di sesi ini.
2. **Kritik memakai penunjuk file:baris.** Kritik tanpa penunjuk tidak boleh dieksekusi (kasus E209).
   Perbaikan menyebut sub-kelas R9 (a)/(b)/(c) dan aturan yang dilanggar.
3. **Kritik tidak boleh menjadi keluaran utama.** Audit berikutnya dipicu oleh peristiwa kirim atau
   interval 14 hari (R14). Kalimat penutup putaran 1 — "ini bukan undangan untuk menulis kritik
   berikutnya" — dilanggar 2 hari kemudian; kini ia menjadi aturan, bukan kalimat.

---

## 6. TERMINATE / PIVOT / REFACTOR / BUILD / KEEP

### TERMINATE (eksekusi ≤1 sesi, tanpa menunggu PI)
- `papers/P3_burial_depth/` + `papers/P21_colonialmine/` (folder kosong) → hapus; P21 proposal tetap
  di `docs/drafts/` dengan satu baris I-NNN.
- `deploy/volcarch-dashboard/` (gitlink superseded) + `tools/references/autoresearch_karpathy/`
  (repo vendored) → arsip/hapus; `tools/globalise_pipeline/` → arsip (tersupersede E211).
- 8 berkas CODEX/handoff Feb di `.claude/` → `git rm` (atau pindah `docs/archive/`).
- Taksonomi pagi 11 Agt (T1–T6, F-EXP…F-SUBST) → pensiun; namespace = SIG + C-NNN.
- EVAL.md sebagai gerbang → turunkan jadi pointer/arsip (R8).

### PIVOT (ofensif, dengan tanggal)
1. **P11 → SPAFA, target kirim ≤ 2026-08-20** (tanggal yang sama dengan tenggat JCAA yang dipenuhi —
   simbolik). Kelima syarat adalah kerja Claude kecuali submit: E069 kanonik (1 sesi), G9 (1 sesi),
   format SPAFA (paruh sesi), G8 :104 (paruh sesi). Submit = PI.
2. **E069 re-derive kanonik (30 puncak)** — melayani TIGA hal sekaligus: gerbang P11, lapis-1 kontrol
   letter Jawa Barat, dan re-derivasi ADV-3 segar. Kerja termurah dengan hasil ganda.
3. **E209 re-run spatial-CV + ≥7 seed (revival diamond-hunt).** $0, satu sesi komputasi RTX 4080.
   Keluaran: AUC yang layak lapor ATAU kill (<0.60). Jika selamat → kandidat P23 + daftar top-20
   target lapangan. Ini "satu deteksi positif" termurah yang tersedia saat ini.
4. **Dua email outreach mikrobotani** — Castillo (UCL) + Vida Kusmartono (BRIN; lead 8 Jun, ia sudah
   menyuarakan argumen tafonomi proyek di media). Menghidupkan C004+C009 dalam satu gerakan. $0.
5. **E226 — kanal NLP literatur abu-abu** (desain di §7), pra-registrasi DESIGN.md, jalankan mining
   pass pertama. $0–10.

### REFACTOR (Claude, sesi ini/berikutnya)
- Manifesto §2 bebas angka (0.768 → pointer; 224 → 214 lokal) — §2 "permanen" tidak boleh memuat angka
  volatil; itu arsitektur dokumennya sendiri.
- E209 di `01_spatial/CLAUDE.md:92` → kalimat jujur berpointer (sub-kelas b).
- C003 dibuka kembali di ledger; entri C010–C0xx ditambahkan (tabel §2).
- `check_doc_sync.py` diperluas (pola WORKSTATE baru; cek manifesto; cek pointer current; cek per-line
  counts) dan dipasang ke CLAUDE.md root sebagai ritual awal-sesi.
- WORKSTATE §6 stop-list box → status jujur ("discharged; keputusan PI sesi ini"); §5 D1/D2 → published.
- CANONICAL P2 :3 → `submission_jcaa_v0.2.tex`.
- Sapuan kontrak: ≥15 blok basi terdaftar di audit line (lampiran di ledger C0xx) — dikerjakan
  mekanis, satu pass, dengan kanari sebagai verifikator.

### BUILD
- **Decision hour hari ini** (§8 = agenda; jawaban PI dicatat di ledger sebagai disposisi).
- **Jadwal falsifikasi berdiri: 2026-09-15** (kandidat disepakati di agenda).
- **Peta peran model netral-merek** (satu file kecil, R13).

### KEEP (jangan disentuh)
P2/JCAA (kejar metadata portal bila editor diam) · P17/ArchCalc · P8/OL · D1/D2 (terbit) ·
MASTERPIECE fallow · repo genetics (eksternal by design) · regenerasi model dashboard (keputusan PI).

---

## 7. Sisi gelap, putaran 2 — kanal dengan biaya dan tanggal

Kegelisahan PI tidak berubah; jawaban putaran ini lebih konkret karena audit menemukan **kanal positif
termurah yang sudah setengah jadi: E209** (lihat PIVOT 3) — classifier yang men-skor situs terkubur
0.86–0.97 saat di-hold-out (Sambisari 0.965, Kedulan 0.930) *sudah ada*, hanya tidak layak lapor dalam
bentuknya sekarang. Satu sesi komputasi memisahkannya dari "temuan" atau "kill".

**Desain singkat E226 (kanal arsip-penghapusan, siap pra-registrasi):**
- *Pertanyaan:* berapa banyak situs/horizon pra-400 M yang sudah tercatat di literatur abu-abu
  arkeologi Indonesia (Berita Penelitian Arkeologi, Kalpataru, Amerta, laporan Balai/BPCB/PUSLIT) tapi
  tidak ada di database modern proyek (E001/NusaRC)?
- *Metode:* unduh korpus publik (repositori kemdikbud, archive.org, GARUDA) → OCR bila perlu →
  ekstraksi terstruktur via DeepSeek batch (situs, koordinat, periode, konteks, sumber:hal) →
  dedup vs E001 → tabel "N kandidat terlupakan" dengan koordinat.
- *Kill criteria:* <1 kandidat baru per 100 dokumen → kanal mati, publikasikan sebagai negatif
  informatif (tafonomi genre itu sendiri).
- *Biaya:* korpus $0; API ≤$10 (ekstraksi 50–100k token). *Waktu:* 2 sesi ke hasil pertama.
- *Pemilik:* line 05; reuses `voc_archnlp`; **tidak butuh otorisasi E211** (sumber publik, tanpa GDPR/SCC).

**Kanal lain (peringkat sisa):** (2) mikrobotani — kekuatan decisif tertinggi, biaya $0, butuh manusia
(email §6.4); (3) gradien epigrafi 400–700 M (Kutai muncul "tiba-tiba" di Kalimantan non-vulkanik vs
ketiadaan Jawa vulkanik — R5 dibuat empiris; data inti sudah di tangan via E082 182 prasasti) →
kandidat I-NNN baru; (4) bentang terendam — **periksa dulu asal data E052**: `generate_synthetic_bathymetry.py`
berarti sebagian batimetrinya sintetik; klaim paleoshoreline di atas data sintetik adalah racun
(presisi hantu). GEBCO asli atau tidak sama sekali; (5) GPR — jangan. Biayanya ribuan dolar per hari
survei, bukan $100; posisinya validasi pasca-deteksi (manifesto §2.2b sudah benar).

**Anggaran $100 PI:** habiskan untuk kuota DeepSeek API E226 (≤$10) + bila ada sisa, satu monograf
Balai Arkeologi berbayar yang tak ada di korpus publik. Tidak untuk GPR, tidak untuk soil core
(soil core butuh kolaborator, bukan uang $100).

---

## 8. Agenda decision hour (7 keputusan, ≤30 menit)

| # | Keputusan | Default | Catatan |
|---|---|---|---|
| D1 | P11→SPAFA: target kirim ≤2026-08-20, Claude kerjakan 5 syarat | **YA** | submit tetap PI |
| D2 | Otorisasi E211 (112 hari) | **YA** | menyuplai 4 trek PhD + kanal arsip |
| D3 | Karya baru pasca-stop-list: E209 re-run + E226 pra-registrasi | **YA** | keduanya $0–10, 2–3 sesi |
| D4 | Dua email outreach (Castillo + Vida/BRIN) — Claude draf, PI approve | **YA** | $0, menghidupkan C004+C009 |
| D5 | L1 disagregasi Jawa/Nusantara | PI | amandemen piagam — tanpa default |
| D6 | Falsifikasi berdiri bertanggal: 2026-09-15 | **YA** | kandidat disepakati di jam ini |
| D7 | Aturan audit: pemicu = kiriman ATAU 14 hari; + daftar TERMINATE §6 disahkan | **YA** | anti-F8 |

---

## 9. Tabel entri ledger baru (dikerjakan sesi ini)

| id | Kritik | V×C | Disposisi | Status |
|---|---|---|---|---|
| C010 | EVAL.md zombie binding-gate (R8) | 2×2 | FIX | OPEN — rewrite sebagai pointer |
| C011 | E209: fix putaran 1 salah kelas (R9) | 2×1 | FIX | ✅ kalimat berpointer dipasang 08-13 |
| C012 | Taksonomi T/F bertabrakan (R10) | 2×1 | FIX-CHEAP | OPEN — namespace tunggal |
| C013 | Kanari merah tak terpasang (R10) | 2×2 | FIX | IN PROGRESS — skrip diperluas + dipasang CLAUDE.md |
| C014 | P11: 5 syarat tanpa tanggal (R12) | 2×1 | FIX | OPEN — target 2026-08-20 (D1) |
| C015 | TRIGGER_MAP 5 bulan tanpa FIRED (R11) | 2×1 | FIX-CHEAP | OPEN — audit atau pensiun |
| C016 | Decision hour tak pernah ada (R13) | 2×2 | FIX | IN PROGRESS — §8 diajukan hari ini |
| C017 | E209 spatial-CV re-run (R14) | 2×1 | PARK | OPEN — unpark: D3 |
| C018 | P5 ultimatum kedaluwarsa (R12) | 2×1 | FIX-CHEAP | OPEN — PARKED.md atau reframe |
| C019 | Manifesto §2 angka volatil (R10) | 2×1 | FIX | ✅ dibersihkan 08-13 |
| C020 | C003 DONE overstated (R10) | 2×1 | FIX | ✅ C003 dikoreksi ke PARTIAL 08-13 |
| C021 | Zombie fisik + CANONICAL P2 (R12) | 2×1 | FIX-CHEAP | OPEN — daftar TERMINATE §6 (D7) |

---

*Ditulis 2026-08-13. Audit: 3 subagent independen (kontrak line · eksekusi kritik · inventori
portfolio) + pembacaan langsung. Semua temuan berpenunjuk file:baris — aturan baru yang lahir dari
R9. Putaran berikutnya dipicu oleh kiriman P11 atau 2026-08-27, mana yang lebih dulu (D7).*
