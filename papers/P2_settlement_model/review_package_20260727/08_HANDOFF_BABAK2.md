# Dokumen 8 — Handoff P2/JCAA #280, Babak 2 (Review Keras + E222/E223)

**Per 2026-07-27 (malam)** | Untuk: Pak Amien, Go Frendi, dan sesi kerja berikutnya
**Melanjutkan:** dokumen 4 (handoff babak 1). **Baca bersama:** dokumen 5 (review co-author), 6 (hasil
E220/E221), 7 (review keras Q1), dan `docs/WORKSTATE.md` + `docs/JOURNAL.md` entri 2026-07-27 (6)–(9).

---

## 1. Status dalam satu tabel

| Hal | Status |
|---|---|
| Keputusan JCAA / tenggat | Revisions requested 2026-07-23; tenggat 2026-08-20 (perpanjangan ke 2026-09-30 disarankan, belum diminta) |
| Klaim inti v0.1 | GUGUR — dikonfirmasi 6 eksperimen (E217, E218, E218b, E219, E220, E222) |
| Reframe (Jalur A) | DIPERKUAT: klaim mekanisme diasah ulang oleh E222 (lihat §3) |
| Review co-author (dok. 5) | SELESAI — Q1 setuju cabut; Q2 common background benar + 3 penajaman; Q3 novelty cukup (dengan E220); Q4 tetap co-author bersyarat. **Persetujuan sah tetap harus dikonfirmasi PI ke Go Frendi manusia** |
| Review keras Q1 (dok. 7) | SELESAI — 8 kritik mayor; semua tertutup eksperimen (E222/E223) atau terkonversi jadi pengakuan terukur; putusan: conditional pass dengan syarat naskah R1–R4 |
| Eksperimen baru babak 2 | **E222** (ground-truth sintetik, 4 dunia: A/B/C/D) + **E223** (paket robustness statistik) — SELESAI, pre-registered |
| Naskah v0.2 | BELUM DITULIS — kini bisa mulai setelah blok A–B (bahan klaim sudah final, lihat §3) |
| Email Verhagen | DITAHAN (milik PI). Posisi kini: koreksi diri + patologi terukur + validasi ground-truth + protokol + bukti lapangan |
| Commit repo | BELUM — seluruh kerja 2026-07-27 (2 babak) belum di-commit (perlu izin PI) |

## 2. Yang dikerjakan babak 2, berurutan

| # | Kerja | Hasil |
|---|---|---|
| 1 | Review keras Q1 atas paket babak 1 — 8 kritik mayor dirumuskan sekuat mungkin | dokumen 7 (M1–M8) |
| 2 | **E222** — validasi ground-truth sintetik (lattice nyata, arkeologi sintetik, pipeline identik) | patologi seleksi tereplikasi terhadap kebenaran (R-own 60/60 memilih konfigurasi terburuk; biaya median +0.194 AUC_true, 100% positif); **klaim slope dikoreksi** (tanda slope kebenaran kontingen rejim); wawasan kontaminasi kuota |
| 3 | **E223** — paket robustness statistik | A: 12/12 CI menolak tangga terbit +0.092; B: bootstrap blok 30 replikasi juga menolak (batas atas ≤ +0.026); C: MaxEnt beta 0.5–4.0 tidak mengubah apa pun (−0.02 di semua); D: k* = 2–5/4–7/7–9 seed (J≥0.85/0.90/0.95) |
| 4 | **E222 World C** — kuota diuji di kandangnya (bias survei regional) | **kuota gagal 0/30** (kontaminasi false-negative) |
| 5 | **E222 World D** — kebenaran seimbang-regional + bias regional (rejim paling ramah kuota) | **kuota gagal lagi 0/30** (−0.203 AUC_true, −0.283 Jaccard). Di 4 rejim sintetik tidak ada desain yang mengalahkan uniform pada kebenaran, sementara AUC laporan selalu memilih desain paling ekstrem |
| 6 | Dokumen 7 (review keras) + README E222/E223 + dokumen ini | selesai |

## 3. Set klaim FINAL untuk v0.2 (pasca R1–R4 dokumen 7)

**Terbukti dan boleh jadi headline:**
1. Evaluasi di background sendiri selalu terinflasi secara struktural (data nyata: +0.041…+0.051, 15/15;
   sintetik: +0.012…+0.273, selalu ≥ 0). Tangga terbit **ditolak** (bukan sekadar tak terbukti): 12/12 CI
   + bootstrap menolak +0.092.
2. Dial desain menggerakkan angka laporan ~10× lebih cepat daripada kebenaran, **ke arah mana pun** —
   maka seleksi model pada angka laporan rusak secara prinsip. Aturan naskah memilih konfigurasi
   terburuk di 100% kasus (data nyata: biaya cross-fitted +0.094; ground truth sintetik: +0.194).
3. Peta prioritas berubah antar desain melampaui noise seed (kontrol split-half; hybrid yang
   menggerakkan), dan tidak stabil terhadap seed saja (1−J 28–47%) — **obatnya murah: ensemble ≥ 4–7
   seed** (k* 7–9 untuk standar publikasi).
4. Produk lapangan: inti prioritas **robust** (densitas situs 2–5,6× fringe) untuk alokasi survei;
   fringe **contingent** sebagai hipotesis.
5. R2-F terjawab (matching); INT-1 tertutup (13 gunung, vonis tautologi selamat).

**Wajib dinyatakan sebagai keterbatasan (jangan dilebihkan):**
- Tanda slope kebenaran vs hard_frac **kontingen rejim** (nyata: turun; sintetik: naik). Jangan klaim
  "hard negatives selalu buruk".
- Dalam 4 rejim sintetik (A: bias jalan; B: misspecified; C: bias regional; D: kebenaran seimbang +
  bias regional), tidak ada desain (tgb/kuota) yang mengalahkan uniform pada kebenaran — dengan batas
  daya n≈300–500 dan bentuk bias yang diuji. Bukan bantahan universal Phillips 2009.
- Boyce: sanity check arah, bukan selektor (optimumnya tak terkalibrasi kebenaran; borderline +0.50).
- Bootstrap: hanya efek ≥ ~+0.03 yang bisa disingkirkan pada n=378.
- n=2 pada lengan non-vulkanik; densitas robust/contingent = konsistensi, bukan validasi.

## 4. Keputusan yang menunggu

### Milik Pak Amien (PI)
1. **Konfirmasi kepengarangan ke Go Frendi manusia** (dokumen 5 §6 + 04 §3). Tanpa ini tidak ada yang
   boleh bergerak ke editor.
2. **Email Verhagen + minta perpanjangan ke 2026-09-30** — posisi sekarang sekuat yang bisa dicapai;
   draft lama perlu diperbarui dengan hasil E220–E222 (saya bisa memperbaruinya atas perintah).
3. **Commit repo** — 2 babak kerja belum di-commit.
4. **Judul** — dengan R1 (klaim = inkomparabilitas + seleksi), kandidat 3 masih paling pas; alternatif
   baru: *"The reported number is the artefact: pseudo-absence evaluation, ground truth, and a corrected
   protocol for archaeological predictive modelling"*.
5. **GO penulisan v0.2** — bahan klaim sudah final (§3); estimasi blok C–G dokumen 4 masih berlaku
   (3–4 minggu), itulah mengapa perpanjangan direkomendasikan.

### Milik Go Frendi (sudah terjawab di dokumen 5, menunggu konfirmasi manusia)
Q1–Q4 — lihat dokumen 5 §3–6.

## 5. Sisa pekerjaan menuju resubmit (diperbarui dari dokumen 4)

| Blok | Isi | Status |
|---|---|---|
| A | Persetujuan co-author (manusia) | menunggu PI |
| B | Email editor (diperbarui dgn E220–E222) + keputusan lingkup | menunggu PI |
| C | Naskah v0.2 — dengan set klaim §3 + syarat R1–R4 dokumen 7 | siap mulai |
| D | Tabel R2 (kovariat per-eksperimen + peran analitik) | belum |
| E | Gambar lama (Fig 1, 4, 5) | belum |
| F | Gambar baru — kini ditambah: kurva dose-response E220 (20 seed), panel seleksi R-own vs kebenaran, panel sintetik E222 (reported vs truth), peta robust/contingent E221, kurva stabilisasi k* | bahan siap (`e220_*`, `e222_*`, `e221_priority_sets_*.npz`) |
| G | Surat balasan reviewer (17 item) — kini hampir semua item punya jawaban eksperimen | belum |
| H | SIG G1 re-derivasi buta semua angka headline **termasuk E220–E223** | sebagian |
| I | Cross-model review G9 | belum |

## 6. Risiko yang tersisa (setelah babak 2)

1. **Perubahan lingkup prosedural** (revisi vs submit baru) — mitigasi: email editor lebih awal.
2. **Novelty** — turun tajam sebagai risiko setelah E220+E222: bentuk klaim kini "prosedur seleksi yang
   rusak secara prinsip + validasi ground-truth", bukan temuan artefak telanjang.
3. **Reviewer menuntut rejim sintetik tambahan** — mitigasi: tulis eksplisit 4 rejim yang diuji +
   batas dayanya; tawarkan kode sebagai supplementary (semua skrip ada).
4. **n=2** dan batas daya bootstrap — sudah dijaga di §3.
5. **Tenggat** — 20 Agustus tetap tidak realistis; perpanjangan.

## 7. Peta file (tambahan babak 2)

- `papers/P2_settlement_model/review_package_20260727/07_REVIEW_KERAS_Q1_GO_FRENDI.md` — review keras + putusan R1–R4
- `papers/P2_settlement_model/review_package_20260727/08_HANDOFF_BABAK2.md` — dokumen ini
- `experiments/E222_synthetic_ground_truth/` — DESIGN.md; 01 (dunia A/B), 02 (C), 03 (D); results/
- `experiments/E223_statistical_robustness/` — DESIGN.md; 01_robustness.py; results/
- README tiap eksperimen baru.

## 8. Yang SENGAJA tidak dikerjakan babak 2

- **Rejim sintetik tambahan** (bias nonstasioner, bias berkorelasi kovariat non-jalan, wilayah kedua
  nyata) — 4 rejim cukup untuk klaim seleksi; sisanya future work, dinyatakan.
- **Menulis v0.2** — masih menunggu blok A–B (persetujuan manusia + keputusan editor).
- **Memperbarui draft email Verhagen** — menunggu perintah PI (isinya berubah materiil setelah babak 2).
- **Commit** — perlu izin PI.
