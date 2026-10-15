# Tinjauan orbit 2026-10-15 — posisi proyek, pendapat, langkah berikutnya, dan agenda keputusan

**Untuk:** Pak Amien (PI) · **Mode:** orbit, dari akar repo · **Disusun:** Claude (Opus 5.5), sesudah P8 R1 tersubmit.
**Pemicu:** PI, 2026-10-15: "kapan hari saya melakukan review menyeluruh, proyek volcarch ini, baca ulang dan kasih
rekomendasi next action di proyek ini, dan bagaimana pendapat kamu" → "ok silahkan" (dicatat sebagai catatan orbit +
agenda keputusan dengan default).
**Status:** catatan kerja internal — bukan sumber bukti (F10). Tidak ada eksperimen baru; tidak ada yang dieksekusi
tanpa jawaban PI di §5.

**Yang dibaca ulang:** `CRITIQUE_SYSTEM_DESIGN_20260811.md` dan `_20260813.md` (dua review menyeluruh yang PI minta),
`OBJECTIVE_ANSWER_20261001.md` (§0, §5–§7), `STRATEGI_PUBLIKASI_20261002.md`, `docs/WORKSTATE.md`, kepala `STATE.md`
ketujuh line, `papers/P23_sima_village_toponyms/README.md`, ledger kritik (C018, C024).

---

## 1. Posisi proyek sekarang

- **Pertanyaan awal sudah punya jawaban obyektif** (`OBJECTIVE_ANSWER_20261001.md` §0): yang "dimulai" ±400 M adalah
  kebiasaan menulis prasasti Sanskerta di batu, datang sebagai gelombang regional bertahap; bukan batas adanya
  masyarakat dan bukan efek gunung api. Prasasti tertua Jawa adalah temuan permukaan di lanskap vulkanik (Tuk Mas,
  Canggal); Sumatra Selatan yang non-vulkanik baru berprasasti 682. Masyarakat kompleks pra-400 jelas ada (Buni/Batujaya,
  Plawangan, Sembiran/Pacung, Gilimanuk).
- **Yang masih terbuka sempit** (§0 butir 3b): tipisnya jejak permukiman pra-400 di interior vulkanik Jawa Tengah/Timur.
  Terkubur (H1b), belum disurvei (H2), tak kasat mata (H3), atau memang jarang (H6) — belum bisa dibedakan. Bisa diuji,
  tetapi uji yang memutuskan butuh data lapangan atau mitra: penampang bertanggal (T4), mikrobotani (T5, BRIN VEGAN),
  atau daftar toponim (T7, terparkir di E226 menunggu daftar Kusen/Resiyani). Uji meja 1 Okt (§9) menunjukkan sumber
  terbuka tidak cukup: **jalur komputasi dari data terbuka untuk pertanyaan ini sudah habis** (seperti vonis 11 Agt:
  "lengkap secara komputasi, kosong secara empiris").
- **Audit Oktober menggugurkan inti P7, P11, P17 dan versi awal P2** (ledger C023–C034). Yang bertahan semuanya di
  kandang PI: **P2** (artefak evaluasi + protokol terkoreksi, JCAA putaran 2 sejak 31 Agt), **P8** (linguistik
  komputasional; diterima bersyarat 5 Okt; R1 tersubmit 15 Okt), **P23** (NLP toponim desa sīma; PLANNING, menunggu
  validasi epigraf).
- **Skor:** 0 diterima final · 1 bersyarat (P8) · 1 di review (P2) · 7 ditolak · 1 ditarik (P17). Batas G13 (≤2 di
  review) penuh oleh P2 dan P8 → tidak ada kiriman baru sampai salah satu diputus jurnal.
- **Menunggu orang luar:** Nastiti (pembaca P23 + daftar toponim; ditanya 2 Okt), VEGAN BRIN (mikrobotani; 2 Okt) —
  keduanya dijadwalkan ditindaklanjuti ±23 Okt (STATE 05, STATE 03); JCAA (P2).

## 2. Pendapat

1. **Proyek lebih sehat daripada Agustus.** Ada jawaban jujur atas pertanyaan inti; ledger eksposur kosong; gerbang
   G1–G16, ledger kritik dan kanari bekerja (15 Okt pun masih menangkap satu salah hitung lama di P8, C071). Keputusan
   positif pertama datang justru untuk naskah yang paling terbuka mengoreksi dirinya.
2. **Identitas proyek perlu disesuaikan dengan buktinya.** Nama dan piagam (L1) masih "bias tafonomi vulkanik", padahal
   bukti proyek sendiri menyatakan vulkanisme tidak diperlukan untuk menjelaskan onset ±400 M. Yang nyata dihasilkan —
   dan yang diakui jurnal — adalah **pengukuran bias dalam *catatan* secara komputasional**: apa arti "tak berkode" di
   ABVD (P8), artefak protokol evaluasi (P2), lokasi toponim sebagai leher botol (P23), laju deteksi permukiman dari
   prasasti (T7), artefak geocoding (C032). Satu tema: **tafonomi catatan**, bukan tafonomi tanah. Tema itu tepat di
   keahlian NLP PI, mendukung jalur KUM/Profesor, dan bisa dikerjakan tanpa menunggu bor tanah. Pertanyaan vulkanik
   tetap hidup sebagai satu hipotesis sempit (H1b), diparkir sampai ada mitra lapangan. **Ini keputusan PI** (L1 milik
   PI; perubahan arah = tanya dulu) — agenda K6.
3. **Risiko terbesar kini adalah suara penulis**, bukan lagi non-eksposur. P8 R1 ditulis Claude (G16 dilepas, C067);
   editor secara khusus meminta pengakuan penggunaan AI (13 Okt). Enam dari tujuh penolakan terjadi di meja editor, dua
   di antaranya soal tulisan (`STRATEGI_PUBLIKASI_20261002.md` §1). Untuk P23, G16 sebaiknya berlaku lagi — agenda K5.

## 3. Langkah berikutnya (urut waktu)

| Kapan | Aksi | Pemilik | Sumber |
|---|---|---|---|
| Minggu ini (≤30 menit) | Sesi keputusan §5 (K1–K9) | PI | catatan ini |
| Okt–Des 2026 | **P23 = satu-satunya naskah yang ditulis** (G13: 1 drafting). PI menulis prosa per bagian; Claude data, tabel, verifikasi, kritik. Tenggat realistis DHQ 2027-01-15; validasi epigraf atas sampel adalah syarat sebelum teks penuh | PI + Claude | `papers/P23_…/README.md` |
| ±23 Okt | Tindak lanjut Nastiti (validasi P23 + daftar Kusen 1990/91 / Resiyani 2010) dan VEGAN bila diam | Claude draf, PI kirim | STATE 05, STATE 03 |
| ±6 Nov | Bila tetap diam: P23 → Arlo Griffiths/DHARMA atau epigraf UGM; VEGAN → Castillo (cadangan) atau T5 diparkir | PI | idem |
| ±9 Nov | P2/JCAA: satu surel sopan menanyakan status bila belum ada kabar (10 minggu putaran 2) | Claude draf, PI kirim | WORKSTATE §5 |
| Opsional, kapan saja | **E232 = I-150**: penyaringan residu sadar-afiks (aturan gaya ModernKataKupas; Tolaki dulu) pada daftar Sulawesi — menjawab batas yang P8 nyatakan sendiri ("No form was morphologically segmented"; arah pertama di Kesimpulan) dan bisa muncul di putaran review kedua. DESIGN.md dulu; **bukan** bagian R1 | Claude, atas persetujuan PI | `docs/IDEA_REGISTRY.md` I-150 |
| Sesudah P23 dibekukan (G14) | **Naskah B**: sintesis berbahasa Indonesia "mengapa sejarah Nusantara dimulai ±400 M" (§0 jawaban obyektif + tabel T1 + pelajaran negatif), rekan penulis BRIN, jurnal nasional zero-APC | PI | `OBJECTIVE_ANSWER_20261001.md` §7.3; `docs/drafts/SINTESIS_400M_OUTLINE_DRAFT_20261001.md` |
| Saat jurnal menulis | P8 putaran 2 / P2: catat parafrase, ledger, rencana | Claude + PI | handoff 15 Okt §2 |

## 4. Yang sebaiknya berhenti

- Eksperimen komputasi baru untuk tesis penguburan vulkanik **tanpa data mitra** (T4/T5/T7 butuh data dari luar).
- Proposal P20 TobaSim, P21 ColonialMine, P22 JavaTephroChron dan AutoResearch: tetap di laci (`docs/drafts/`,
  IDEA_REGISTRY), tidak dibuka sebelum P23 dan Naskah B.
- Putaran kritik sistem berikutnya: hanya dipicu keputusan jurnal (P2/P8), kiriman baru, atau 14 hari (aturan D7).

## 5. Agenda sesi keputusan (≤30 menit; jawab "ya" / "tidak" / ubah)

| # | Keputusan | Default | Bila "ya", yang dikerjakan Claude |
|---|---|---|---|
| **K1** | **P5** (ritual clock; ditolak BKI; rewrite telat sejak ±Juni; ledger C018 OPEN) → **PARKED** | YA | `papers/P5_volcanic_ritual_clock/PARKED.md` dengan syarat unpark (pola P16); C018 ditutup |
| **K2** | **P11** (NO-GO 1 Okt) → **parkir** (pilihan c); reframe Penanggungan dicatat sebagai syarat unpark, bukan pekerjaan | YA | `papers/P11_volcanic_informedness/PARKED.md`; WORKSTATE §4 dibersihkan |
| **K3** | **P1** tetap ditahan sampai audit C024 (laju "4 mm/thn" dari umur monumen yang diasumsikan) | YA | tidak ada (sudah default sejak 1 Okt) |
| **K4** | Naskah yang sedang direvisi/di-review ulang (P8 R1) **dihitung** di slot G13 → tidak ada kiriman baru sampai P2 atau P8 diputus | YA | definisi ditulis di `docs/SUBMISSION_INTEGRITY_GATE.md` G13 |
| **K5** | **G16 berlaku lagi untuk P23** (dan naskah sesudahnya): prosa ditulis PI; Claude data, analisis, verifikasi, kritik, cek bahasa | YA | catatan di P23 README + memori |
| **K6** | **Arah program (L1):** "tafonomi catatan" (bias dalam database, teks, prasasti, diukur komputasional) sebagai identitas program; H1b (penguburan vulkanik) diparkir sampai ada mitra lapangan. Sekaligus: amandemen L1 yang menunggu sejak 11 Agt (D5: Jawa vs Nusantara) dan 1 Okt (a–e) | **tanpa default** — keputusan PI; usul: diputuskan paling lambat **2026-11-15** | draf amandemen L1 untuk dibaca PI (tidak diubah sendiri) |
| **K7** | **E232 = I-150** (penyaringan residu sadar-afiks, pilot Tolaki): tulis DESIGN.md pra-registrasi; jalan hanya sesudah PI membaca desainnya | YA (desain saja) | `experiments/E232_…/DESIGN.md` |
| **K8** | Tanggal tindak lanjut: Nastiti + VEGAN **23 Okt**; cadangan **6 Nov**; P2/JCAA **9 Nov** | YA | draf surel disiapkan sehari sebelumnya |
| **K9** | Daftar berhenti §4 disahkan | YA | WORKSTATE §6 |

**Jawaban PI (2026-10-15):** "setuju default, K6 nanti saya putuskan".

| # | Dikerjakan hari yang sama |
|---|---|
| K1 | `papers/P5_volcanic_ritual_clock/PARKED.md`; C018 ditutup; temuan baru C072 (uji permutasi P5 hampir pasti terjadi dengan sendirinya; dan klaim "E032" kritik 11 Agt keliru) |
| K2 | `papers/P11_volcanic_informedness/PARKED.md` (pilihan c; jalur unpark a/b) |
| K3 | dicatat di WORKSTATE §4 (P1 tetap ditahan sampai C024) |
| K4 | definisi di `docs/SUBMISSION_INTEGRITY_GATE.md` G13 |
| K5 | `papers/P23_sima_village_toponyms/README.md` (G16 dikonfirmasi; tidak ikut pengecualian C067) |
| K6 | **terbuka — keputusan PI paling lambat 2026-11-15** (WORKSTATE §2) |
| K7 | `experiments/E232_affix_aware_residue_screen/` (DESIGN.md + README; status PRE-REGISTERED, belum dijalankan; langkah 0 = memperoleh Mead 1998, hdl:1911/19290, akses belum dicek); I-150 → EXPERIMENT; indeks diperbarui (224) |
| K8 | tanggal di WORKSTATE §2, README P23, STATE 03/05 |
| K9 | WORKSTATE §6 |
