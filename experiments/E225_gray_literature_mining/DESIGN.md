# E225 — Gray-Literature Mining: Forgotten Pre-400 CE Records — PRE-REGISTRATION

**Pre-registered:** 2026-08-13 (keputusan PI D3). **Status:** READY TO RUN — jalankan mining pass
pertama. **Line:** 05_archival_nlp (membantu 01/02/03). **Paper target:** kanal erasure-archive
(kritik sistem 2026-08-13 §7) — "mengukur keheningan arsip"; venue TBD (zero-APC).
**Catatan penomoran:** dokumen kritik menulis "E226"; nomor berikutnya yang bebas adalah **E225**
(E001–E224 teralokasi) — referensi dikoreksi ke E225.

---

## 1. Pertanyaan (dapat difalsifikasi)

Berapa banyak situs/horizon pra-400 M yang **sudah tercatat** di literatur abu-abu arkeologi
Indonesia (laporan Balai Arkeologi/BPCB/PUSLIT, Berita Penelitian Arkeologi, Kalpataru, Amerta)
tetapi **tidak ada** di database modern proyek (E001, NusaRC/D2)?

- **H1 (kanal hidup):** ≥1 kandidat pra-400 M baru per 100 dokumen yang dipindai.
- **H0 (kanal mati):** <1 per 100 dokumen — literatur abu-abu sudah terserap penuh oleh database
  modern, atau korpusnya tidak memuat situs pra-400 M. **Kedua hasil dapat dipublikasikan** —
  negatif informatif = bukti kuantitatif tafonomi genre (apa yang dicatat dan apa yang tidak).

## 2. Metode (terdefinisi sebelum eksekusi)

1. **Korpus:** publik & gratis — repositori.kemdikbud.go.id (laporan Balai), GARUDA
   (garuda.kemdikbud.go.id), archive.org (koleksi Perpusnas/perpustakaan digital), jurnal Kalpataru
   & Amerta (PDF).
2. **Pipeline:** unduh PDF → ekstraksi teks (pypdf/OCR Tesseract bila pindaian) → potong per
   dokumen → **ekstraksi terstruktur via DeepSeek batch API** dengan skema JSON: `{source, year,
   page, site_name, place_hint, period_estimate, feature_type, evidence_summary, confidence,
   original_quote}`.
3. **Dedup vs E001/NusaRC:** kandidat dianggap BARU bila >10 km dari situs modern terdekat ATAU
   klaim periodenya tidak ada di database modern. Ambang dedup dicatat di hasil.
4. **Verifikasi:** setiap kandidat "BARU" harus punya kutipan asli + nomor halaman; kandidat
   confidence rendah ditandai, tidak dibuang.

## 3. Kriteria kill / sukses (terdefinisi sebelum eksekusi)

| Hasil | Definisi | Aksi |
|---|---|---|
| SUCCESS | ≥5 kandidat pra-400 M baru dengan koordinat/penunjuk lokasi | verifikasi 3 kandidat terhadap laporan asli → daftar lead → masukan letter Jawa Barat / paper erasure-archive |
| PARTIAL | 1–4 kandidat | laporkan jujur; kanal tetap dibuka, korpus diperluas |
| FAILED (informatif) | 0 kandidat pada 100+ dokumen | publikasikan sebagai negatif (tafonomi genre); kanal ditutup dengan alasan |

## 4. Biaya & waktu

- Korpus: $0. API DeepSeek: **≤$10** (ekstraksi batch 50–100k token; PI menyediakan key bila perlu).
- Komputasi: CPU biasa (tanpa GPU); waktu: **2 sesi** ke hasil pertama.

## 5. Aturan pelaporan

- Setiap kandidat membawa kutipan asli; tidak ada angka tanpa pointer (aturan kanari).
- Jumlah dokumen dipindai, jumlah kandidat, dan rasio dilaporkan apa adanya — termasuk korpus yang
  gagal diproses (daftar kegagalan OCR).
- Output: `results/e225_candidates.csv` + `results/e225_scan_log.csv`.

## 6. Mengapa eksperimen ini (dan bukan yang lain)

Ini "sisi gelap yang dibuka AI" secara harfiah: mengukur keheningan arsip dengan NLP pada kekuatan
inti PI. Sumber publik (tanpa GDPR/SCC, tidak seperti E211), murah, dan hasilnya dua arah dapat
diterbitkan. Pra-registrasi mengikuti model E217–E223.
