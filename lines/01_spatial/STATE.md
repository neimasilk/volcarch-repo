# STATE — Line 01 SPATIAL

**Updated:** 2026-08-11 · **Temperature:** 🟢 COOLING — P2 **TERKIRIM**, 9 hari sebelum tenggat

> ## ✅ P2 v0.2 DIKIRIM KE JCAA — 2026-08-11
>
> Status portal Round 1 kini: **"Submission has been resubmitted for another review round."**
> Tenggat 2026-08-20 terpenuhi dengan sisa 9 hari. **Tidak ada lagi item P2 yang menunggu PI.**
>
> **Yang masuk ke portal** (grid Revisions, submission #280):
>
> | ID | Nama di portal | Komponen | Berkas sumber |
> |---|---|---|---|
> | 9970 | Manuscript (revised) | Manuscript | `submission_jcaa_v0.2.pdf` (29 hal) |
> | 9971 | Supplementary Tables S1-S6 | Supplementary file (for review) | `supplementary_tables_v0.2.pdf` (5 hal) |
> | 9972 | Response to Reviewers | Response to reviewers | `revision_ammo/RESPONSE_TO_REVIEWERS_v0.2_UPLOAD.pdf` (8 hal) |
>
> **Verifikasi pasca-unggah (bukan asumsi).** Salinan server diunduh kembali dan dibandingkan dengan
> berkas lokal: naskah **29 hal, teks identik, SHA1 sama**. Portal menulis ulang wadah PDF dua berkas
> (Producer `MiKTeX pdfTeX` → `mPDF 8.3.1`, ukuran byte berubah) tetapi **isi tidak tersentuh**;
> surat balasan (xelatex) lolos tanpa diubah sama sekali. Pola ini mengikuti komponen, bukan berkas.
>
> **Catatan Review Discussion terkirim** ke Verhagen + Gonzalez-Perez (2026-08-11 02:11 BST).
>
> **Temuan baru yang belum tertutup — metadata portal.** Judul dan abstrak di rekaman JCAA **masih
> versi lama** ("Tautology-Free…"; abstrak yang menyatakan *pseudo-absence realism is the dominant
> lever* dan *tautology suite conditional pass* — semuanya sudah ditarik v0.2). Penulis **tidak bisa
> memperbaikinya**: field menerima ketikan tetapi tombol Save mati, dan setelah reload suntingan
> hilang. Diuji langsung, bukan disimpulkan. Sudah diminta ke editor lewat catatan di atas.
> **Jika editor tidak merespons, ini perlu ditagih** — klaim yang ditarik jangan sampai hidup di basis
> data jurnal, apalagi terbawa ke produksi.

> 📄 **Review menyeluruh pra-submit SELESAI (2026-08-10).** Naskah v0.2 lulus. Review menemukan
> **13 masalah yang lolos dari SIG 9/9 GREEN**, 4 blocking — semuanya sudah diperbaiki. Rinciannya:
> `docs/HANDOFF_20260810.md`.
>
> **Empat blocker itu bukan angka**, dan itulah sebabnya G1 tidak menangkapnya: (1) AI Disclosure
> menyangkal kontribusi AI pada research design, dibantah oleh pra-registrasi di repo **publik** yang
> ditunjuk Data Availability sendiri; (2) Tabel suplemen S3–S6 dijanjikan tapi tidak ada; (3) surat
> balasan masih berkepala "DRAFT prepared by Claude Code / NOT SENT / sign-off pending"; (4) klaim
> lisensi MIT tanpa file `LICENSE`. **Usul G11 untuk SIG: gerbang klaim non-numerik.**
>
> Perbaikan substantif terpenting: **6 rujukan ENM diverifikasi Crossref lalu ditambahkan**
> (lobo2008, hijmans2012, barve2011, warren2011, radosavljevic2014, guillera2015) dan §1.3 ditulis
> ulang untuk mengakui prior art lebih dulu — ini menutup kritik R1 "not entirely novel" yang
> sebelumnya masih terbuka lebar terhadap klaim baru. Dua miscitation dibuang. **Tabel 4 dikoreksi**:
> gap dihitung terhadap seed-average (0.751), bukan best run (0.768) → +0.122 menjadi **+0.105**.
>
> **Status: naskah (29 hal, kompilasi bersih) + suplemen (5 hal, 6 tabel) + surat balasan SIAP.
> Tersisa hanya `git push` + upload portal — keduanya PI.**

---

## Hard deadline

**P2 resubmission to JCAA: 2026-08-20 — no extension will be requested.** PI decision 2026-08-03: every
`[RUN]` item is done, dissolved, or out of scope, so the remaining revision is **writing**, and asking
for more time on run-related grounds would have given the editor a reason that is not true. Asking the
scope question ("revision or new submission?") was also dropped — it risks inviting the answer "new
submission", which would discard the only non-reject this project has had in 14 months. Withdrawn draft
and full reasoning: `docs/correspondence/EMAIL_VERHAGEN_EXTENSION_REQUEST_20260803.md`.

**Consequence: v0.2 is a revision of #280, and the corrections go in the Response to Reviewers.**

---

## Blocked on PI (nothing downstream can move)

| # | Item | Since | Status |
|---|---|---|---|
| A | **Confirm authorship with the human Go Frendi.** The new manuscript reaches the *opposite* conclusion from the one he signed in March. `review_package_20260727/05_*` is Claude's analysis of his likely position, **not a signature.** | 2026-07-27 | ✅ **RESOLVED 2026-08-05.** PI confirmed Go Frendi is OK with the reversed claim set. The Authors' Contributions sentence (`submission_jcaa_v0.2.tex:573`) is now factually true. No further action. |
| B | ~~Send the Verhagen email~~ | 2026-07-27 | **CLOSED 2026-08-03 — withdrawn, no email will be sent.** |
| C | ~~Decide scope: revision vs new submission~~ | 2026-07-27 | **CLOSED — revision of #280.** |
| D | ~~Permission to commit~~ | 2026-07-27 | **CLOSED — committed.** |
| E | **v0.2 title** — candidates in `revision_ammo/JCAA_R1_RESPONSE_PLAN_20260727.md` §3. Candidates resting on "selection picks the worst design" are dead (K5); those resting on **evaluation incomparability** survive. | 2026-07-27 | ✅ **RESOLVED 2026-08-05 — kandidat 3 terkonfirmasi** ("An Evaluation Artefact in Presence–Background Archaeological Modelling…"). Naskah tidak berubah. |

---

## Next actions for Claude (in order)

- [x] **B′ — apply K1–K3 to the claim set.** ✅ 2026-08-03 → doc 10 (K5/K6/K7/G1c added same day).
- [x] Block H — **SIG G1 blind re-derivation.** ✅ 61 checks 2026-08-03; **re-run 2026-08-05: 62
      checks, 58 OK, 4 mismatch** (persis klaim lama yang ditarik K5/K6/K7/G1c) + **A7** baru
      (0.706 common-bg, lolos).
- [x] **E224 — K4 confirmation run.** ✅ 2026-08-03 — FAILED; TGB null dilaporkan **unexplained** (§3.5).
- [x] Block D — **R2 covariate table.** ✅ Table 1 (roles) + Table 2 (inclusion) di v0.2 (§2.1, §2.3).
- [x] Block G — **reviewer response letter.** ✅ diselaraskan 2026-08-05 ke naskah final: semua
      `[NEEDS v0.2]` diselesaikan, R2-H menyatakan 7 figur v0.1 yang dihapus, penomoran E218/E219
      disamakan (E219 part C kini di naskah §3.8).
- [x] Block F — **new figures.** ✅ 2026-08-05: fig14 artefact, fig15 dose-response, fig16 robust/
      contingent map, fig17 stabilisasi — semua dari file hasil mentah via `build_v02_figures.py`.
- [x] Block E — **old figures refresh.** ✅ fig10 di-redraw dengan 13 pusat kanonik (INT-1); fig3
      di-restate sebagai ladder under examination; prefix caption "Figure N."/“Table N.” manual dihapus.
- [x] **Manuscript v0.2 prose + perbaikan.** ✅ **26 pp, kompilasi bersih, nol overfull.** S1 (klaim
      level Tabel 4 ditambal, A7), S2 (Test 1/3 didefinisikan di §2.4), S3 (latar arkeologi East Java),
      S4 (abstrak 216 kata, satu angka headline +0.042), ENM lit (5 sitasi terverifikasi, `[NEEDS
      CITATION]` hilang).
- [x] Block I — **cross-model review (G9).** ✅ 2026-08-05 — subagent adversarial (diminta menolak):
      **tak ada klaim tertolak, tak ada mismatch angka**; 3 frasa dikencangkan (Limitation 3 agregat,
      scope "seed-ensembled", scope AI-disclosure) + inflasi home-court dinyatakan spesifik hybrid
      (A8/A9, check lolos).
- [x] **SIG re-sign-off final.** ✅ 2026-08-05 → `SIG_signoff.md` = **🟢 GO pada integritas naskah**
      (9/9 gerbang hijau).
- [x] **Review menyeluruh pra-submit (permintaan PI, handoff 5 Agt §2).** ✅ 2026-08-10 — 13 temuan,
      4 blocking, semuanya ditutup hari ini. Naskah 29 hal kompilasi bersih; suplemen S1–S6 dibuat
      (`build_supplement.py` → `supplementary_tables_v0.2.pdf`); 6 rujukan ENM diverifikasi Crossref;
      Tabel 4 dikoreksi ke seed-average; AI Disclosure ditulis ulang jujur; `LICENSE` dibuat; laporan
      reviewer rahasia berhenti dipublikasikan. **G1 final: 64 check, 4 mismatch** (= K5/K6/K7/G1c
      yang memang ditarik) → `SIG_G1_VERIFICATION_20260810.md`.

## Submit — SELESAI 2026-08-11

- [x] **`git push`** (19 commit). ✅ `origin/main` sinkron. Pernyataan Data Availability kini benar.
      Efek samping penting: tiga berkas rahasia (`02_LAPORAN_REVIEWER.md` + dua dok review
      beratas-nama Go Frendi) **berhenti dipublikasikan** — sebelum push, ketiganya masih terbuka di
      GitHub, di repo yang naskahnya sendiri tunjuk ke editor. Riwayat lama masih memuatnya; keputusan
      PI tetap **jangan** rewrite history.
- [x] **Upload portal JCAA** ✅ 3 berkas, komponen benar, terverifikasi byte-per-byte.
      Surat balasan siap-unggah dibuat sebagai `RESPONSE_TO_REVIEWERS_v0.2_UPLOAD.md/.pdf`: blok
      komentar internal dibuang, penanda "(PI decision…)" dihapus, tanda tangan ditambahkan.
      ⚠ Saat konversi, percobaan pertama **diam-diam menghilangkan** `≥ ≤ ≈ ≠ β ρ` (font tanpa
      glyph) — "k ≥ 7" jadi "k  7". Diperbaiki dengan Times New Roman + xelatex, lalu diverifikasi
      simbol per simbol. **Pelajaran: selalu cek glyph setelah md→PDF.**
- [x] **APC waiver £593** ✅ diangkat di surat balasan (keputusan PI 2026-08-11), bukan email terpisah.

## Tersisa — menunggu jurnal

- [ ] **Metadata portal**: minta editor mengganti judul + abstrak (lihat kotak di atas). Sudah diminta
      2026-08-11; **tagih jika belum berubah saat kontak berikutnya.**
- [ ] Balasan editor / ronde review kedua. Tidak ada aksi kita sampai itu datang.

## Deliberately NOT doing

Additional synthetic regimes · second-region replication → both declared **future work** in the
manuscript. E219 two-stage "suitable but absent" (R2-C) → dissolved when the taphonomic claim was
withdrawn.

---

## Other papers in this line

- **P17** (ArchCalc 365) — under review. No action. Do not touch the manuscript; it is live and
  double-blind.
- **P11** — retarget to SPAFA is **queued behind the [07_career](../07_career/) exposure actions** by
  PI decision. Do not start it as an alternative to P2 work.
- **D2** — Zenodo upload is a [07_career](../07_career/) item.

## Inbox (found while working, not yet triaged)

- `docs/experiment_index.json` covers only **84 of 214** experiment directories, so the
  experiment→paper mapping is stale. Re-run `tools/scan_experiments.py` and add a `line` field.
  Cheap, and it is what makes this whole layer self-maintaining.
