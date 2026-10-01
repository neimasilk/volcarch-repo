# CRITIQUE LEDGER — mekanisme seleksi kritik VOLCARCH

**Status:** ACTIVE (2026-08-11). Dibuat atas usulan kritik sistem/research-designer
(`docs/research_notes/CRITIQUE_SYSTEM_DESIGN_20260811.md` §4).
**Tujuan:** menentukan **secara sadar** kritik mana yang diakomodasi dan mana yang diabaikan — tanpa
drop senyap di dua arah: kritik valid yang terabaikan, dan kritik invalid yang menjadi mesin penunda
(F8). Ini adalah keputusan, bukan perdebatan.

---

## 1. Cara kerja

Setiap kritik yang masuk (dari peer review, review AI, kritik sistem, kekhawatiran PI, temuan kanari)
**dicatat**: sumber · tanggal · klaim yang dituju · **Validity (0–2)** · **Centrality (0–2)**.

Skor:
- **Validity**: 0 = salah/menyesatkan/membidik hal yang bukan klaim kami; 1 = parsial/tergantung
  konteks; 2 = benar dan langsung pada sasaran.
- **Centrality**: 0 = tidak menyentuh klaim yang dimuat; 1 = menyentuh klaim sekunder/bagian; 2 =
  menyentuh klaim inti yang dimuat.

Empat disposisi — **tanpa drop senyap**:
| Disposisi | Aturan |
|---|---|
| **FIX** | Validity≥1 & Centrality=2 → data baru ATAU downgrade klaim; rewording dilarang (banned move SIG) |
| **FIX-CHEAP** | Validity≥1 & Centrality=1 → perbaiki jika ≤1 sesi; jika tidak, PARK dengan kondisi unpark |
| **PARK** | Validity≥1 & Centrality=0 → dicatat dengan pemicu unpark + pemilik |
| **REJECT-with-reason** | Validity=0 → ditolak secara sadar, alasan dicatat; kritik BERHENTI memblokir antrian |

Dua disiplin:
1. **Kritik dengan disposisi REJECT berhenti memblokir antrian.** Inilah katup anti-F8.
2. **Veto PI** boleh menimpa disposisi apa pun, tapi veto itu **dicatat** (bukan dibisukan).

Klaim inti proyek punya escape-question permanen (uji T3): *hasil apa yang akan meng-update kerangka
MELAWAN tesis?* Jika sebuah klaim tak bisa menjawab dengan nama eksperimen + kanal data konkret, ia
**PARK** otomatis sampai bisa.

---

## 2. Entri

Format baris: `[id] tanggal — klaim — V×C → DISPOSISI — status`

### Sesi 2026-08-11 (kritik sistem — semua masuk dari `CRITIQUE_SYSTEM_DESIGN_20260811.md`)
| id | Kritik | V×C | Disposisi | Status |
|---|---|---|---|---|
| C001 | Inflasi label sukses: indeks tak memilah defensif/ofensif/disconfirming (R1) | 2×1 | FIX-CHEAP | OPEN — T2 di `scan_experiments.py` |
| C002 | Angka hantu "E209 AUC 0.844" di `01_spatial/CLAUDE.md` (R2) | 2×2 | FIX | ✅ DONE 2026-08-11 (dihapus, diganti status jujur) |
| C003 | Hitungan eksperimen "224" vs indeks 214 tak rekonsiliasi (R2) | 2×1 | FIX | 🔄 **PARTIAL** — WORKSTATE/memory/indeks ✓ 08-11; manifesto+lines/README baru dibereskan 08-13; status DONE sebelumnya overstated (C020) |
| C004 | Uji decisive mikrobotani (E215) tak pernah dijalankan (R3) | 2×2 | PARK | IN PROGRESS — PI setuju 08-13 (D4): draf email Castillo+Vida dikerjakan Claude, PI approve sebelum kirim; unpark penuh saat terkirim |
| C005 | Eksperimen alami Jawa Barat: konfound upaya-survei + tanpa naskah (R4) | 2×2 | FIX | IN PROGRESS — kontrol survei dibangun di skeleton letter (2026-08-11) |
| C006 | Konflasi "Nusantara" ≠ "Jawa" di piagam (R5) | 2×2 | PARK | OPEN — unpark: amandemen L1 (disaggregasi); pemilik: PI |
| C007 | Botol manusia: keputusan tak di-batch, E211 110 hari (R6) | 2×1 | FIX-CHEAP | OPEN — decision hour mingguan |
| C008 | AutoResearch zombie + antrian paper sunk-cost (R7) | 2×1 | FIX-CHEAP | OPEN — arsip AUTORESEARCH_CONCEPT, park P5 jika tak reframe |
| C009 | Kapabilitas AI bukan kendala; satu arkeobotanis > 1000 eksperimen (R6) | 2×2 | FIX | IN PROGRESS — PI setuju 08-13 (D4): dua email, Claude draf, PI approve |

### Sesi 2026-08-13 (kritik sistem putaran 2 — semua masuk dari `CRITIQUE_SYSTEM_DESIGN_20260813.md`)
| id | Kritik | V×C | Disposisi | Status |
|---|---|---|---|---|
| C010 | EVAL.md zombie binding-gate: klaim tautologi yang sudah ditarik masih "mengikat" (R8) | 2×2 | FIX | OPEN — rewrite sebagai pointer/arsip |
| C011 | Fix E209 putaran 1 salah kelas: angka bersumber dihapus, diganti klaim kontradiktif (R9) | 2×1 | FIX | ✅ DONE 08-13 — `01_spatial/CLAUDE.md` kini berpointer ke FINDINGS_v1 |
| C012 | Taksonomi T/F bertabrakan (pagi T1–T6 vs malam T0–T7; F- dua keluarga) (R10) | 2×1 | FIX-CHEAP | OPEN — namespace tunggal: SIG + C-NNN |
| C013 | Kanari merah tak terpasang ke awal sesi; tak bandingkan disk (R10) | 2×2 | FIX | ✅ DONE 08-13 — `check_doc_sync.py` v2 hijau, dipasang ke CLAUDE.md |
| C014 | P11: 5 syarat SPAFA tanpa tanggal — pola "siap-tanpa-kirim" berulang (R12) | 2×1 | FIX | ✅ **5 SYARAT SELESAI 2026-08-13** (E069 kanonik · G9+10 perbaikan · format terverifikasi · G8 · konversi v0.7+docx+form); **tinggal aksi PI**: review + Figure Form + submit portal ≤ 2026-08-20 |
| C015 | TRIGGER_MAP 5 bulan tanpa FIRED; IDEA_REGISTRY READY basi (R11) | 2×1 | FIX-CHEAP | OPEN — audit atau pensiun |
| C016 | Decision hour tak pernah dijadwalkan; E211 112 hari (R13) | 2×2 | FIX | ✅ **DECISION HOUR HELD 2026-08-13** — D1–D4 dijawab PI (semua YA: SPAFA ≤20 Agt · E211 · E209+E225 · outreach); D5–D7 menunggu konfirmasi teks (default YA untuk D6/D7) |
| C017 | E209 spatial-CV re-run (revival diamond-hunt, $0, 1 sesi komputasi) (R14) | 2×1 | PARK | ✅ **UNPARKED 2026-08-13** (D3 YA) — spatial-CV + ≥7 seeds; selamat → kandidat P23 + top-20 target |
| C018 | P5: ultimatum putaran 1 kedaluwarsa tanpa parkir/reframe (R12) | 2×1 | FIX-CHEAP | OPEN — PARKED.md atau reframe |
| C019 | Manifesto §2 "permanen" memuat angka volatil (AUC 0.768, "224") (R10) | 2×1 | FIX | ✅ DONE 08-13 — §2 bebas angka, §3 = 214 |
| C020 | C003 DONE overstated — ledger mencatat klaim eksekusi salah (R10) | 2×1 | FIX | ✅ DONE 08-13 — C003 dikoreksi ke PARTIAL; aturan "DONE = artefak bernama terverifikasi" |
| C021 | Zombie fisik + CANONICAL P2 menunjuk file yang salah (R12) | 2×1 | FIX-CHEAP | IN PROGRESS — CANONICAL ✅ 08-13; daftar TERMINATE §6 menunggu D7 |
| C022 | Kontrak CLAUDE/STATE 6 line basi pasca-08-11 (≥15 blok; audit line 2026-08-13) | 2×1 | FIX | ✅ DONE 08-13 — kontrak 01–07 + lines/README + CANONICAL P2 disapu; kanari hijau verifikator |

### Sesi 2026-10-01 (audit re-entry — pembaca bukti WF + cek independen)
Sumber: workflow `volcarch-objective-answer` (pembaca Sonnet per klaster, read-only) 2026-10-01.
**Diverifikasi langsung dari file oleh orkestrator pada hari yang sama:** C023 (refit + efek parsial), C024, C025, C026, C027, C029, C030.
C028 juga diverifikasi (skrip E178 dijalankan ulang, read-only). **Semua C023–C030 terverifikasi pada hari yang sama.**

| id | Kritik | V×C | Disposisi | Status |
|---|---|---|---|---|
| C023 | **E069/ADV-3 salah tanda.** β pada **jarak** ke gunung api negatif (−0.477 → −0.831 kanonik) = **lebih banyak** situs tercatat di dekat gunung api setelah kontrol survei — dicatat sejak 2026-03-13 sebagai "defisit yang selamat dari kontrol survei" (README, verdict skrip, P11 v0.7, P1 v4.0, skeleton Jawa Barat, kontrak line 02, L1, scorecard "cathedral"). Kriteria pra-registrasi ADV-3 juga **tak berarah**. Re-derivasi G1 13 Agt memeriksa angka, bukan arah. | 2×2 | FIX | 🔄 **IN PROGRESS 10-01** — README/hasil kanonik/skrip E069 dikoreksi + re-run (angka identik, verdict `SURPLUS NEAR VOLCANOES`); cek independen `direction_check_20261001.py`; **P11 v0.8 membuang klaim** (SIG re-check); P1 v4.0, skeleton, kontrak 02, catatan L1/STORY/PREMORTEM. **Sisa:** audit ulang hilir E109, E120, E136, E154, E159, E073, E085, E080, E098, E158 (+P16 parkir); **rekam publik: YA** — preprint P1 Zenodo `10.5281/zenodo.19081502` (terbit 2026-03-18, `submission_v1.0.pdf`, baris ±401–405) memuat "β = −0.477, p = 0.0015 … The volcanic site deficit is not reducible to differential survey effort" → perlu koreksi publik (Zenodo mengizinkan versi baru; keputusan PI). **Pelajaran SIG:** G1 harus memverifikasi *arah/tanda* + interpretasi, bukan hanya angka. |
| C024 | Laju penguburan interior "4 mm/thn" = kedalaman ÷ **umur monumen yang diasumsikan** (daftar E132 diketik tangan; 2 titik umurnya ditetapkan analis), dilabeli "E083 best estimate" di E117; laju bervariasi ~1 orde; kalibrasi peluruhan-jarak memakai jarak salah ketik (mis. Sambisari 24.1 km kanonik vs 5.8 km di E132). | 2×2 | FIX | ✅ **VERIFIED 10-01 (orkestrator):** E117 `onset_analysis.py` l.279–281 & 382 meng-hardcode 3.5/4.0/4.4 mm/thn "dari E083"; hasil E083 hanya memuat kedalaman + tahun letusan, tanpa keluaran laju; daftar kalibrasi E132 diketik tangan, umur OV1928 (1000 thn) dan OV1925 (1500 thn) ditetapkan analis. **OPEN-fix:** horizon deteksi E117 diturunkan ke "ilustratif, dekat-ventilasi"; butuh stratigrafi bertanggal (OSL/14C). Pemilik: line 02. |
| C025 | E128 **bukan** replikasi independen E083 (13/16 nilai kedalaman sama; "independence CONFIRMED" di-hardcode). | 2×1 | FIX-CHEAP | ✅ **VERIFIED 10-01:** 13 nilai kedalaman unik sama persis (E083 18 unik, E128 19 unik termasuk outlier 60 m); keduanya bersumber laporan OV. Catatan koreksi di README E128. **OPEN-fix:** hapus frasa "independent replication" dari kontrak/naskah. |
| C026 | Validasi E075 r=0.951 = model-vs-model (kolom "observed" adalah keluaran grid Pyle). | 2×1 | FIX-CHEAP | ✅ **VERIFIED 10-01:** E075 membaca `observed_depth_cm` dari `dashboard/sites.csv`, yang kolom `burial_depth_cm`-nya dihasilkan `tools/precompute_dashboard_data.py` (Pyle 1989 + kalibrasi Dwarapala). **Jangan pernah dikutip sebagai validasi.** |
| C027 | E195: prediksi gagal (prasasti dekat gunung api justru **lebih tua**, ρ=+0.525) dilabeli ulang "AHA" — penyelamatan interpretatif. | 2×1 | FIX-CHEAP | ✅ **VERIFIED + FIXED 10-01:** JSON `verdict: UNEXPECTED` (ρ=+0.525, MW p=0.0002) vs README "SUCCESS (AHA…)"; status README → INFO NEG + catatan koreksi. |
| C028 | E178 "karst faktor tersembunyi ke-6": keluaran skripnya sendiri tak mendukung (karst ρ=0.000; vulkanik ρ=+0.185 tanpa Jepang); tanpa file hasil. | 2×1 | FIX-CHEAP | ✅ **VERIFIED 10-01 (skrip dijalankan ulang, read-only):** karst ρ=0.000 (p=1.0), vulkanik ρ=+0.185 (p=0.69, tanpa Jepang), koefisien karst −3.217; tabelnya sendiri menempatkan Bali (karst 0.05) sebagai terpadat (0.865) sementara skrip mencetak "HIGH karst → dramatically more sites". Catatan koreksi di README E178. **OPEN-fix:** turunkan di kontrak line 02 ("the key reframe" → belum teruji). |
| C029 | Basis situs E069/E109 (geojson 666 situs OSM/Wikipedia; 662 periode "unknown"; kemungkinan monumen modern) tak pernah divalidasi; E001 tak pernah dijalankan. | 2×2 | FIX | ✅ **VERIFIED 10-01:** 666 fitur, 662 periode `unknown`, 125 tipe `monument`, 78 nama monumen modern (Tugu Pahlawan, Monumen Perjuangan Polri, Patung Karapan Sapi…); hanya 391 berkoordinat. **E069 terkontaminasi; E153/P11 inti KOKOH** (non-candi bersih: rerata 6.61 km, ketat 6.22 km, >80% <10 km, null 53.7 km — `E153/robustness_modern_features_20261001.py`). **OPEN-fix:** inventori tervalidasi bertipe-periode sebelum uji jumlah-situs berikutnya. |
| C030 | "Nol situs open-air pra-400 di interior vulkanik" bersifat definisional: E071 sendiri mencantumkan Pasir Angin, Cipari, Gunung Padang 10–24 km dari gunung api. | 2×2 | FIX | ✅ **VERIFIED 10-01:** `E071/results/pre400ce_evidence.csv` mencantumkan Pasir Angin (Bogor, 500 SM–500 M), Gunung Padang (Cianjur), Cipari (Kuningan). **OPEN-fix:** definisi kelas dipra-registrasi; berdampak pada letter Jawa Barat. |
| C031 | **P11 "lereng barat" se-Jawa = efek Penanggungan + daftar candi berduplikat.** Pada inventori kanonik: Penanggungan (69) p=1.4e-20, 62% barat; **73 candi lainnya R=0.136, p=0.26**; teks mengutip p=0.0009 dari run pra-kanonik. 142 baris candi = **103 koordinat unik** (39 duplikat); sumber OSM/Wikipedia, bukan katalog terbit. | 2×2 | FIX | ✅ VERIFIED 10-01 (orkestrator) → **P11 NO-GO** (`SIG_signoff.md` §2026-10-01). Reframe = PI. |
| C032 | **Kontras candi–prasasti (inti P17 "Two Javas", pilar 2 P11) = artefak geocoding.** 175 prasasti di 42 koordinat; 50 label relief Borobudur di satu titik + 42 titik pengganti wilayah. Lokasi temuan sebenarnya saja: gap 13.1 → 2.3 km (p=0.068); candi de-dup: 0.2 km (p=0.27); Jawa Timur: 0.2 km (p=0.85). | 2×2 | FIX | ✅ VERIFIED 10-01 (verifikator Opus + re-derivasi independen, `E082/robustness_geocoding_20261001.py`). **P17 sedang di-review ArchCalc #365 → keputusan PI: beri tahu editor / tarik.** Draf: `docs/correspondence/EMAIL_ARCHCALC_P17_INTEGRITY_NOTICE_DRAFT_20261001.md`. |
| C033 | Docx SPAFA P11 (v0.7 & v0.8) **tanpa abstrak** (EN maupun ID), padahal "SIG FINAL GO" 13 Agt. | 2×1 | FIX-CHEAP | ✅ VERIFIED 10-01. Pelajaran SIG: periksa berkas yang benar-benar diunggah, bukan hanya .tex. |
| C034 | E105 "929 M" (pilar 3 P11): 48 dari 58 catatan pra-929 "court-zone" adalah label relief Borobudur; 13 dari 15 "periphery" pasca-929 di titik pengganti "East Java". | 2×2 | FIX | REPORTED 10-01 (verifikator Opus); mekanismenya sama dengan C032 yang terverifikasi. OPEN-verify angka persisnya. |

### Referensi kritik terdahulu yang sudah diproses (untuk jejak)
- **P7/Antiquity reviewer** — inventori gunung terpotong → FIX (G1/G3), menimbulkan WS-E + kanonik 30
  gunung. ✅ tertutup 2026-06-08–08-11.
- **Reviewer R2 JCAA (P2)** — reproducibility → FIX, dimasukkan ke Response to Reviewers (G1c). ✅
- **ME#16/DeepSeek/Gemini/ChatGPT** — proxy-stack tanpa anchor fisik → mengarah pada masterpiece Phase 0
  + E214/E216. ✅ diakomodasi.

---

## 3. Cara memakai

- **Sesi awal:** cek kolom Status kolom `OPEN`; kerja yang punya pemilik non-PI dikerjakan, yang
  pemiliknya PI dirangkum ke `docs/WORKSTATE.md` §4.
- **Saat menerima kritik baru:** buat baris, skor, tetapkan disposisi. Jika REJECT — tulis alasannya
  satu baris dan **lanjut** (jangan balas dengan paragraf).
- **Saat akan memperbaiki:** baca `docs/SUBMISSION_INTEGRITY_GATE.md` (banned move: tidak ada jawaban
  rewording untuk kritik struktural).

*Ledger ini bukan tempat menampung kritik agar terlihat sibuk — ia adalah katup keputusan. Kritik yang
sudah diberi disposisi dan statusnya terkunci tidak boleh muncul lagi sebagai penghenti antrian.*
