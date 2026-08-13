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
