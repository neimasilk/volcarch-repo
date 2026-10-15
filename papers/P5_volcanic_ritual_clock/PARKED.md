# P5 — PARKED (2026-10-15, keputusan Pak Amien)

**Status: PARKED — jangan submit, jangan rewrite, sampai syarat unpark terpenuhi.**
Keputusan: agenda orbit K1, "setuju default" (Pak Amien, 2026-10-15; `docs/research_notes/ORBIT_REVIEW_20261015.md` §5).
Menutup ledger **C018** (ultimatum "reframe ≤2 sesi atau PARK" dari kritik 11 Agt yang kedaluwarsa) dan bagian P5 dari C008.

## Mengapa diparkir

1. **Ditolak BKI di meja editor (2026-03-19):** terlalu dekat dengan topiknya; perlu terlibat dengan perdebatan teoretis dan
   menunjukkan mengapa topik ini penting secara sosial (`docs/JOURNAL.md` 2026-03-20; parafrase).
2. **Reframe humaniora ke *Asian Ethnology*** direncanakan sejak 2026-04-01 (`HUMANITIES_REFRAME_STRATEGY.md`) tetapi tidak
   pernah dimulai; rewrite tertunda sejak ±Juni.
3. **Batas kerja G13** (≤1 naskah ditulis): slot penulisan dipegang P23 sampai ia dibekukan.
4. **Pemeriksaan 2026-10-15 menemukan masalah metode yang belum pernah dicatat (ledger C072, belum diverifikasi ulang
   dengan menjalankan apa pun):** uji permutasi "1 dari 120, p = 0,008" (draf §5, `draft_v0.1.tex` 336–339) tampaknya
   hampir pasti terjadi dengan sendirinya. Rentang tahap pembusukan yang dipakai berurutan dan nyaris tidak tumpang-tindih
   (segar 1–5 hari, kembung 3–14, pembusukan lanjut 20–80, skeletonisasi 60–300, pelarutan tulang 300–2.500; draf baris 340),
   sehingga hanya penugasan berurutan yang bisa cocok penuh. Akibatnya p = 1/120 akan keluar untuk *lima angka naik mana pun*
   yang masing-masing jatuh di rentangnya; uji itu tidak menguji kebetulan. Draf juga menyebut uji ini "requires no
   researcher-defined stage boundaries" (baris 465), padahal ia memakai rentang tahap, dan menyatakan hipotesisnya
   "generated from the data" (baris 465). Uji yang bermakna adalah simulasi Monte Carlo (seberapa mungkin lima angka acak jatuh
   di rentang), dan rentang itu dipilih dari literatur sesudah kecocokannya terlihat.
5. **Catatan koreksi atas kritik 11 Agt:** kritik itu menyebut statistik pusat P5 sebagai E032 yang gagal FDR. Itu tidak
   cocok: E032 adalah Pranata Mangsa (dipakai P11, bukan draf P5; kata "Pranata" tidak muncul di `draft_v0.1.tex`).
   Statistik P5 berasal dari E025 (`experiments/E025_quantitative_validation/`).

## Yang hidup dan yang tidak

| Klaim | Status | Catatan |
|---|---|---|
| Urutan peringatan slametan 3, 7, 40, 100, 1000 hari tidak bersumber pada tradisi Hindu, Buddha, Islam, atau Kristen | belum diaudit (G1) | perlu sumber etnografis/filologis primer, bukan sekunder |
| Uji permutasi p = 0,008 | **tidak dipakai lagi sebagai bukti** (C072) | lihat butir 4 |
| Monte Carlo atas rentang forensik | lemah | rentang dari konteks non-Jawa; dipilih pasca-pengamatan |
| Kerangka H-TOM (dua skala waktu) | bergantung pada P7 yang mati | P7 ditolak *Antiquity* 2026-06-04 |
| Ide besar: pengetahuan lokal tentang kematian bertahan melalui gelombang agama | ide tetap hidup | IDEA_REGISTRY; tidak dibuang |

## Syarat UNPARK (semua)

1. **Slot G13 kosong** (P23 dibekukan atau dikirim) dan PI memilih P5 sebagai naskah berikutnya.
2. **Uji permutasi dibuang**; bila ada uji kuantitatif, rentang tahap ditetapkan *sebelum* melihat angka slametan
   (pra-registrasi E-baru), dengan pembanding tradisi lain memakai aturan yang sama.
3. **Bentuk humaniora** sesuai kritik BKI: pertanyaan yang relevan secara sosial dan keterlibatan teoretis (Geertz,
   Woodward, Beatty; kajian Austronesia perbandingan), **dengan rekan penulis ranah** (antropologi/kajian agama Jawa) —
   G10 dan aturan rekan penulis 2026-10-02.
4. **Prosa ditulis PI (G16)**; SIG penuh (G1–G16) sebelum kirim.

## Peta file (utuh, tidak ada yang dihapus)

- Naskah: `draft_v0.1.tex` (versi BKI), `draft_v0.1_bki.pdf`, `title_page.tex`; `drafts/section*.md` (draf reframe parsial)
- Rencana reframe: `HUMANITIES_REFRAME_STRATEGY.md` · amunisi revisi: `revision_ammo/` · gambar: `figures/`
- Eksperimen: `experiments/E025_quantitative_validation/` (permutasi + Monte Carlo), `experiments/E023_ritual_screening/`
- Ledger: C008, C018 (ditutup oleh berkas ini), C072 (uji permutasi)
