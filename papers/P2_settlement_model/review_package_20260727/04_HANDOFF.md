# Dokumen 4 — Handoff P2/JCAA #280

**Per 2026-07-27** | Untuk: Pak Amien, Go Frendi, dan sesi kerja berikutnya
**Baca bersama:** `docs/WORKSTATE.md` (kontrak kerja utama) dan `docs/JOURNAL.md` entri 2026-07-27 (1)–(5)

---

## 1. Status dalam satu tabel

| Hal | Status |
|---|---|
| Keputusan JCAA | **Revisions requested**, 2026-07-23, Dr Philip Verhagen |
| Tenggat | **2026-08-20** (4 minggu; editor mempersilakan minta perpanjangan lebih awal) |
| Rekomendasi reviewer | R1 *Resubmit for Review*; R2 *Resubmit Elsewhere* (editor override) |
| Klaim inti naskah | **GUGUR** — artefak evaluasi, dikonfirmasi 4 eksperimen |
| Arah yang dipilih PI | **Jalur A** — reframe ke temuan artefak (diputuskan 2026-07-27) |
| Judul | Pilihan pertama ("Tautology-Controlled…") **sudah tidak relevan** setelah reframe; 3 kandidat baru di rencana revisi §3 |
| Email ke editor | **DITAHAN** atas instruksi PI. Draft siap. |
| Persetujuan co-author | **BELUM** — paket ini yang memintanya |
| Naskah v0.2 | **BELUM DITULIS** |
| Waiver APC £593 | **BELUM DIPUTUSKAN** sejak 2026-04-06 (di-acknowledge Verhagen 2026-04-07) |
| Cross-model review G9 | **BELUM** |
| Commit repo | **BELUM** — seluruh kerja hari ini belum di-commit (perlu izin PI) |

---

## 2. Yang dikerjakan hari ini, berurutan

| # | Kerja | Hasil |
|---|---|---|
| 1 | Triase 17 item reviewer | `../revision_ammo/JCAA_R1_RESPONSE_PLAN_20260727.md` |
| 2 | Temukan INT-1 (inventaris gunung 7 vs 13) dan INT-2 (`anticipated_critiques.md` basi & menyesatkan) | keduanya ditandai; INT-2 diberi header peringatan |
| 3 | **E217** — benchmark MaxEnt (permintaan R1) | klaim inti gugur; pipeline tervalidasi (0.750 vs 0.751 terbit) |
| 4 | **E218** — uji ketahanan refutasi sendiri, 20 seed, 4 evaluation background, 3 metrik | refutasi bertahan; hipotesis mekanisme saya **gagal**, instrumennya rusak |
| 5 | **E218b** — mekanisme, redesain sapuan `hard_frac` | mekanisme terbukti: metrik laporan dan generalisasi **berlawanan arah** |
| 6 | **E219** — divergensi peta + INT-1 + kontrol R2-F | klaim pengganti ditemukan; INT-1 ditutup; R2-F dijawab |
| 7 | Paket review ini | 4 dokumen untuk Go Frendi |

Semua dicatat di `docs/JOURNAL.md` dan `docs/WORKSTATE.md`.

---

## 3. Keputusan yang menunggu

### Milik Go Frendi (co-author)
1. Setuju klaim inti dicabut? *(dokumen 3 §8 Q1)*
2. Apakah "common evaluation background" tolok ukur yang benar? *(Q2 — ini yang paling menentukan)*
3. Apakah E218b cukup baru melawan komentar "not entirely novel" R1? *(Q3)*
4. Kepengarangan di versi baru: tetap / mundur / bersyarat? *(Q4)*

### Milik Pak Amien (PI)
5. **Kapan email Verhagen dikirim?** Draft siap di
   `docs/correspondence/EMAIL_VERHAGEN_JCAA_DISCLOSURE_DRAFT_20260727.md`. Argumen untuk mengirim segera
   setelah persetujuan co-author: Verhagen sendiri peneliti utama pemodelan prediktif arkeologi, jadi dia
   sekaligus penilai terbaik apakah temuan ini berguna atau sekadar mengulang Lobo — **dan** orang yang
   harus memutuskan revisi-vs-submit-baru. Risiko menunda: datang 20 Agustus dengan paper berbeda tanpa
   pemberitahuan terbaca sebagai *bait-and-switch*.
6. **Minta perpanjangan?** Rekomendasi: minta sampai 2026-09-30 di email yang sama. Menulis ulang naskah
   dengan kesimpulan terbalik tidak realistis dalam sisa 3 minggu.
7. **Judul baru** — 3 kandidat di rencana revisi §3.
8. **Cross-model review G9** sebelum kirim apa pun? (protokol proyek; ME#17 menandai risiko echo-chamber)
9. **Commit repo** — seluruh kerja hari ini belum di-commit.
10. **Waiver APC** — naikkan satu baris saat submit revisi, terpisah dari email pengungkapan (mencampur
    pengungkapan integritas dengan permintaan biaya melemahkan keduanya).

---

## 4. Sisa pekerjaan menuju resubmit

| Blok | Isi | Status |
|---|---|---|
| A | Persetujuan co-author | menunggu |
| B | Email + keputusan lingkup editor | menunggu keputusan PI |
| C | **Naskah v0.2** — judul, abstrak, §1 pertanyaan penelitian, framing baru, literatur ENM, konteks manajemen cagar budaya Jatim, glosarium jargon | belum mulai |
| D | Tabel yang diminta R2: matriks kovariat per-eksperimen + peran analitik tiap variabel | belum |
| E | Gambar: Fig 1 gambar ulang (alur data), label meluber di Fig 1 & 4, caption Fig 5 | belum |
| F | Gambar baru untuk temuan baru: kurva dose-response `hard_frac`, matriks 4-evaluation-background, peta divergensi | belum |
| G | Surat balasan ke reviewer (per item, 17 item) | belum |
| H | SIG G1 — re-derivasi buta semua angka headline | sebagian (E217 §3.1 sudah memvalidasi pipeline) |
| I | Cross-model review G9 | belum |

**Realistis:** blok C–G butuh 3–4 minggu kerja plus review PI. Karena itu perpanjangan direkomendasikan.

---

## 5. Risiko yang harus diingat

1. **Kebaruan (paling besar).** R1 sudah bilang temuan aslinya "not entirely novel... well established in
   ecological niche modeling". Temuan artefak telanjang dekat dengan Lobo dkk. (2008). Pertahanannya =
   E218b (hubungan terbalik terkuantifikasi), bukan temuan artefaknya sendiri. Kalau pertahanan itu tidak
   meyakinkan reviewer, paper ini sulit.
2. **R2 dan reframe metodologis.** Membuat paper lebih metodologis memperparah keberatan R2 bahwa situs
   cuma jadi observasi spasial. E219 adalah jawabannya; kalau reviewer tidak menerima E219 sebagai
   "arkeologis", risikonya tinggi.
3. **Perubahan lingkup (prosedural).** Revisi dengan kesimpulan terbalik bisa dianggap paper baru. Membesar
   selama kita diam.
4. **n=2.** Angka densitas situs non-vulkanik bertumpu pada 2 situs. Jangan pernah dikutip sebagai
   kelipatan presisi.
5. **Tenggat.** 20 Agustus tidak realistis untuk hasil yang bagus.

---

## 6. Peta file

**Paket review ini:** `papers/P2_settlement_model/review_package_20260727/`
- `00_README.md`, `01_naskah_asli_jcaa_v0.1.{pdf,tex}`, `01_naskah_asli_references.bib`,
  `02_LAPORAN_REVIEWER.md`, `03_TEMUAN_REVISI.md`, `04_HANDOFF.md` (ini)

**Eksperimen:**
- `experiments/E217_maxent_benchmark/` — benchmark MaxEnt + evaluasi dicocokkan
- `experiments/E218_evaluation_artefact/` — ketahanan artefak (`DESIGN.md` = pre-registrasi) + mekanisme
- `experiments/E219_map_divergence/` — divergensi peta, INT-1, kontrol R2-F

**Perencanaan & korespondensi:**
- `papers/P2_settlement_model/revision_ammo/JCAA_R1_RESPONSE_PLAN_20260727.md` — triase 17 item + rencana v2
- `papers/P2_settlement_model/revision_ammo/anticipated_critiques.md` — **BASI, jangan dipakai mentah**
- `docs/correspondence/EMAIL_VERHAGEN_JCAA_DISCLOSURE_DRAFT_20260727.md` — **DITAHAN**

**Pelacakan:** `docs/WORKSTATE.md`, `docs/JOURNAL.md` (2026-07-27 entri 1–5)

---

## 7. Yang SENGAJA tidak dikerjakan

- **Email ke editor** — ditahan atas instruksi PI.
- **E219 dua-tahap "suitable but absent"** (permintaan R2-C) — larut setelah klaim taphonomic dicabut;
  tidak ada lagi klaim prediksi situs untuk didekomposisi.
- **Naskah v0.2** — menunggu persetujuan co-author dan keputusan lingkup editor. Menulis 30 halaman
  sebelum tahu apakah editor mau revisi atau submit baru = pemborosan.
- **Commit repo** — perlu izin PI.

---

## 8. Catatan untuk sesi berikutnya

Kalau konteks hilang: baca `docs/WORKSTATE.md` bagian 2026-07-27 (5) lebih dulu, lalu dokumen 3 paket ini.
Rantai buktinya E217 → E218 → E218b → E219 dan tiap tahap punya pre-registrasi serta hasil mentahnya di
`results/`.

**Jangan** mulai menulis naskah v0.2 sebelum blok A dan B di §3 tuntas.

**Ingat forcing function ME#19:** kendala mengikat proyek ini adalah NON-EXPOSURE, bukan kekurangan rigor.
Kerja hari ini menambah rigor secara besar-besaran; ia **tidak** menambah exposure. Tiga aksi eksternal
lama (balasan Verberne, Zenodo D1+D2, balasan Lamqaddam) tetap belum tuntas dan tetap milik Pak Amien.
