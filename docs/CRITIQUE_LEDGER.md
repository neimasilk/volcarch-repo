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
| C003 | Hitungan eksperimen "224" vs indeks 214 tak rekonsiliasi (R2) | 2×1 | FIX | ✅ DONE 2026-08-11 (214 lokal, E001–E224) |
| C004 | Uji decisive mikrobotani (E215) tak pernah dijalankan (R3) | 2×2 | PARK | OPEN — unpark: outreach arkeobotanis (Castillo/UCL/BRIN); pemilik: PI |
| C005 | Eksperimen alami Jawa Barat: konfound upaya-survei + tanpa naskah (R4) | 2×2 | FIX | IN PROGRESS — kontrol survei dibangun di skeleton letter (2026-08-11) |
| C006 | Konflasi "Nusantara" ≠ "Jawa" di piagam (R5) | 2×2 | PARK | OPEN — unpark: amandemen L1 (disaggregasi); pemilik: PI |
| C007 | Botol manusia: keputusan tak di-batch, E211 110 hari (R6) | 2×1 | FIX-CHEAP | OPEN — decision hour mingguan |
| C008 | AutoResearch zombie + antrian paper sunk-cost (R7) | 2×1 | FIX-CHEAP | OPEN — arsip AUTORESEARCH_CONCEPT, park P5 jika tak reframe |
| C009 | Kapabilitas AI bukan kendala; satu arkeobotanis > 1000 eksperimen (R6) | 2×2 | FIX | OPEN — outreach, pemilik: PI |

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
