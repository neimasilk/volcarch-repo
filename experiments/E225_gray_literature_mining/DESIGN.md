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

## 7. Amandemen 2026-10-15 — ditulis SEBELUM run pertama (belum ada satu dokumen pun diunduh atau diekstraksi)

**Pemicu:** PI 2026-10-15 ("kasih pandangan, dan kemudian lakukan yg bisa dilakukan"); pemeriksaan pengendali atas desain
13 Agt menemukan tiga cacat yang membuat hasilnya condong ke "sukses":
1. **Pembanding terlalu kecil.** "BARU" = tidak ada di E001/NusaRC, padahal D2 hanya 80 situs; hampir setiap situs dari
   literatur akan "baru". Kriteria ini **dicabut** sebagai ukuran sukses.
2. **Korpus tidak dikunci.** Tanpa aturan dokumen mana yang diambil, "per 100 dokumen" bisa dipilih-pilih.
3. **Tidak ada aturan tanggal.** "Prasejarah", "paleometalik", "neolitik" tidak otomatis berarti pra-400 M.

**Cek cepat sebelum amandemen (fail fast, 2026-10-15):** tidak ditemukan register tanggal arkeologi Indonesia yang terbuka
dan lengkap — hanya daftar parsial (Bronson & Glover, *SPAFA Digest*, daftar cetak; ulasan tanggal gua/ceruk di *JIPA*;
Cochrane dkk. 2021 untuk gerabah awal; satu dataset Borneo di PANGAEA). Jurnal arkeologi BRIN terbuka (OJS, CC BY-SA /
BY-NC-SA) dan dapat didaftar lengkap lewat OAI-PMH.

**Pertanyaan (dipertajam):** berapa banyak **bukti bertanggal numerik** aktivitas manusia pra-400 M yang tercatat di jurnal
arkeologi berbahasa Indonesia, dengan kutipan dan halaman — dan (fase 2) berapa yang tidak ada di kompilasi internasional?
Keluaran berguna dua arah: register terbuka (bila ada) atau negatif informatif.

**Kerangka korpus (dibekukan saat dipanen):** semua rekaman OAI-PMH (`oai_dc`) dari lima jurnal di ejournal.brin.go.id —
*Berkala Arkeologi*, *Amerta*, *Naditira Widya*, *Kalpataru*, *Purbawidya* — dipanen 2026-10-15; disimpan sebagai
`results/frame_oai_20261015.csv` dengan SHA-256 dicatat di README. Jurnal lain (*Forum Arkeologi*, *Walennae*: OAI 404) di luar
kerangka pilot.

**Pilot (fase 1), dua lengan dideklarasikan sekarang:**
- **A — acak (untuk laju):** 150 artikel diambil acak dari kerangka (seed 20261015, `random.Random`), semua jurnal bersama.
- **B — tersasar (untuk hasil register, BUKAN untuk laju):** artikel yang judul atau abstrak OAI-nya cocok dengan regex
  pertanggalan `radiokarbon|radiocarbon|C-?14|carbon dating|pertanggalan|dating|\bBP\b|Before Present|termoluminesen|OSL|AMS`
  (tanpa membedakan huruf), dikurangi yang sudah ada di A; dibatasi 150 pertama menurut urutan kerangka bila lebih.

**Pipeline:** unduh PDF galley (gagal unduh dicatat) → teks pypdf; halaman tanpa lapisan teks dicatat (OCR tidak dijalankan di
pilot; laju dilaporkan atas dokumen yang terbaca) → regex prapenyaring tanggal numerik (angka + BP/SM/M/BC/AD/cal, atau kode
lab) → DeepSeek (model `deepseek-chat`, suhu 0) hanya pada dokumen yang lolos prapenyaring, skema JSON per tanggal:
`site, admin_location, coordinates_as_stated, method, lab_code, age_value, age_error, age_unit (BP/cal BP/BC/AD/SM/M),
calibrated_range_as_stated, material, context, page, quote`.

**Penjaga anti-halusinasi (wajib):** sebuah rekaman dihitung hanya bila `quote` (spasi dinormalkan) **ditemukan verbatim**
di teks dokumen dan angka `age_value` muncul di dalam `quote`; lainnya ditandai `unverified` dan tidak dihitung. Pengendali
memeriksa 10 rekaman acak terverifikasi terhadap PDF.

**Aturan pra-400 M (tingkat):**
- **A (radiometrik):** metode numerik (¹⁴C, AMS, TL, OSL, U-series) dengan nilai dan galat; pra-400 M bila rentang kalibrasi
  yang dinyatakan penulis berakhir sebelum 400 M, atau — bila hanya umur ¹⁴C tak terkalibrasi — `age − 2×error ≥ 1750 BP`
  (konservatif: 1750 BP ≈ cal abad ke-3–4 M). Umur di sekitar ambang dicatat "dekat batas", tidak dihitung.
- **B (tanggal dinyatakan):** tanggal numerik oleh penulis (mis. "abad ke-2 M", "500 SM") tanpa kode lab; dihitung terpisah.
- **C (periode saja):** "prasejarah", "neolitik", "paleometalik" tanpa angka — **tidak dihitung**.

**Hasil yang dilaporkan:** O1 = rekaman pra-400 M tingkat A dan B per 100 dokumen terbaca di lengan A (dengan selang Wilson
untuk proporsi dokumen yang memuat ≥1); O2 = jumlah rekaman pra-400 M terverifikasi berbeda (situs × tanggal) dari A+B;
O3 = bagian O2 yang punya koordinat atau lokasi setingkat kabupaten; O4 = rekaman ditolak penjaga.

**Aturan keputusan (menggantikan §3):**
| Hasil | Definisi | Aksi |
|---|---|---|
| LANJUT | lengan A: ≥ 1 rekaman tingkat A per 100 dokumen terbaca, **atau** A+B ≥ 20 rekaman pra-400 M terverifikasi | fase 2: seluruh kerangka + perbandingan cakupan dengan kompilasi internasional (ulasan *JIPA*, suplemen Cochrane dkk. 2021, daftar Bronson & Glover) — pra-registrasi tambahan sebelum fase 2 |
| PARSIAL | di antara | laporkan; perluasan korpus diputuskan PI |
| GAGAL (informatif) | lengan A: 0 rekaman tingkat A **dan** A+B < 5 | kanal ditutup dengan alasan; dilaporkan sebagai negatif |

**Hak dan data:** PDF dan teks disimpan lokal di `data/raw/e225_corpus/` (di-gitignore); repo hanya memuat kerangka OAI,
log, dan rekaman (fakta + kutipan pendek dengan rujukan). Biaya DeepSeek dicatat; batas tetap ≤ $10.
