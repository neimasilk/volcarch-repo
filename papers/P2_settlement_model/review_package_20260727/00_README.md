# Paket Review P2 / JCAA #280 — untuk Go Frendi Gunawan

**Disusun:** 2026-07-27
**Dari:** Mukhlis Amien (via Claude Code)
**Untuk:** Go Frendi Gunawan, co-author
**Sifat:** MENDESAK — klaim inti naskah yang sudah kita submit ternyata tidak bertahan. Perlu persetujuan
co-author sebelum apa pun dikirim ke editor.

---

## Kenapa paket ini ada

JCAA meminta **revisi** (bukan tolak) pada 2026-07-23 — R&R pertama proyek ini dalam 14 bulan. Tenggat
2026-08-20.

Saat mengerjakan permintaan Reviewer 1 (benchmark melawan MaxEnt), muncul hasil yang **menggugurkan temuan
utama naskah kita sendiri**. Bukan temuan reviewer — temuan kita, saat menuruti permintaan reviewer.

Sebagai co-author, Mas Go Frendi berhak tahu dan berhak tidak setuju sebelum apa pun bergerak.

## Sepuluh dokumen — tapi jangan baca urut nomor

Paket ini tumbuh dalam tiga babak pada satu hari (27 Jul) plus satu babak koreksi (3 Agt), dan
**babak yang lebih baru mengoreksi yang lebih lama.** Urutan bacanya:

| # | Dokumen | Isi | Perkiraan waktu |
|---|---|---|---|
| 1 | `01_naskah_asli_jcaa_v0.1.pdf` (+ `.tex`, `.bib`) | Naskah persis seperti yang disubmit 2026-03-11 | 30–45 mnt |
| 2 | `02_LAPORAN_REVIEWER.md` | Keputusan editor + dua laporan reviewer, verbatim | 20 mnt |
| 3 | `03_TEMUAN_REVISI.md` | **Dokumen inti babak 1.** Apa yang gugur, buktinya, letak tiap eksperimen | 45–60 mnt |
| 4 | `04_HANDOFF.md` | Status babak 1 | 15 mnt |
| 5 | `05_REVIEW_COAUTHOR_GO_FRENDI.md` | Analisis posisi co-author (**bukan** tanda tangan) | 15 mnt |
| 6 | `06_HASIL_E220_E221.md` | Hasil seleksi + stabilitas seed | 10 mnt |
| 7 | `07_REVIEW_KERAS_Q1_GO_FRENDI.md` | Review bermusuhan, 8 kritik mayor, syarat R1–R4 | 20 mnt |
| 8 | `08_HANDOFF_BABAK2.md` | Handoff babak 2. **§3 "set klaim FINAL" sudah GUGUR** — pakai dokumen 10 | 10 mnt |
| 9 | `09_REVIEW_ATAS_BABAK2.md` | Verifikasi independen. Menemukan K1–K3 | 20 mnt |
| **10** | **`10_SET_KLAIM_TERKOREKSI.md`** | ⭐ **PALING BARU (3 Agt) dan otoritatif.** Set klaim v0.2 pasca K1–K7 + hasil SIG G1 | **25 mnt** |

**Kalau waktu terbatas:** baca **dokumen 10** (set klaim yang akan masuk naskah) lalu dokumen 2
(apa yang reviewer minta). Dokumen 3 §1 kalau ingin cerita lengkap bagaimana klaim lama gugur.

⚠ **Jangan mengutip angka dari dokumen 07 atau 08 tanpa mengeceknya ke dokumen 10.** Tujuh klaim di
sana lebih kuat daripada datanya sendiri; semuanya sudah dikoreksi.

## Yang diminta dari Mas Go Frendi

Ada 4 pertanyaan spesifik di akhir dokumen 3 (§8) dan daftar keputusan di dokumen 4 (§3). Yang paling
penting: **apakah Mas setuju klaim inti memang harus dicabut**, atau melihat cacat di argumen saya.

Kritik keras dipersilakan — justru itu gunanya. Semua kode dan data ada di repo, tinggal jalankan ulang;
perintahnya tercantum di dokumen 3 §7.

## Catatan integritas

Belum ada apa pun yang dikirim ke editor. Draft email pengungkapan sudah ada tapi **DITAHAN**
(`docs/correspondence/EMAIL_VERHAGEN_JCAA_DISCLOSURE_DRAFT_20260727.md`) menunggu keputusan PI dan
persetujuan co-author.
