# HANDOFF — Session 22 (2026-04-23)

**Duration:** Full work session
**Mode:** Execution — HKI product build + E211 Phase 1 pipeline run + PhD track continuation
**Trigger:** Pak Amien: "lanjutkan pekerjaan, ada pekerjaan baru: 1 produk HKI, lanjutkan yang condong ke UvA"

---

## 1. Deliverables produced this session

### A. VOC-ArchNLP v1.0.0 — HKI Product (NEW)

Python package di `tools/voc_archnlp/` untuk didaftarkan sebagai Hak Cipta Program Komputer ke DJKI.

**File package:**
- `__init__.py` — metadata (Mukhlis Amien, ORCID, CC BY 4.0)
- `extractor.py` — ArchaeologicalMentionExtractor [KOMPONEN BARU]: 6 entity types (MONUMENT, GRAVE, RUIN, ARTIFACT, INSCRIPTION, DEPTH), konversi voet/el/palm/duim → meter, output CSV/JSON
- `pipeline.py` — VOCArchPipeline: orchestrator 4-stage
- `cli.py` — Unified CLI: `python -m voc_archnlp [download|preprocess|normalize|extract|run]`
- `__main__.py`, `setup.py`, `requirements.txt`

**Dokumen HKI (`docs/HKI/`):**
- `DESKRIPSI_PROGRAM.md` — isian formulir DJKI (Bahasa Indonesia, lengkap)
- `MANUAL_PENGGUNA.md` — user manual
- `ARSITEKTUR_SISTEM.md` — diagram + penjelasan teknis
- `PANDUAN_PENDAFTARAN_DJKI.md` — step-by-step daftar di e-hakcipta.dgip.go.id (Rp 400K perseorangan atau via LPPM)

---

### B. E211 Phase 1 — COMPLETE

Pipeline VOC-ArchNLP dijalankan penuh pada 500 file GLOBALISE (146 juta kata).

**Hasil:**

| Stage | Output |
|---|---|
| Preprocess | 548,929 paragraf, 145,971,146 kata |
| Normalize | 1,000 file normalized (500 clean + 500 paras), colonial Dutch → modern Dutch |
| Extract (pre-normalize) | **33,930 candidate mentions** |
| Extract (post-normalize) | 33,931 (delta: +1 dari tjandi→candi) |

**Output files di `results/E211_voc_mentions/`:**
- `voc_archaeological_mentions.csv` — full extraction (33,930 rows)
- `voc_mentions_java_filtered.csv` — geographic filter Java/Indonesia (14,626 rows)
- `voc_mentions_high_precision.csv` — MONUMENT+INSCRIPTION+Java (871 rows)
- `voc_mentions_normalized.csv` — post-normalize extraction (33,931 rows)
- `annotation_sample_v1.csv` — 65-sentence annotation sample untuk Phase 2
- `ANNOTATION_GUIDE_v1.md` — panduan annotasi

**Temuan kritis (lihat `experiments/E211_voc_dagregister_nlp/FINDINGS_v1_20260423.md`):**
- `oudheden` (kosakata arkeologi Belanda) = **0 occurrences** dalam 500 file
- False positive utama: `pagode` = mata uang emas (bukan bangunan), `arca` = Latin/Portugis untuk peti (bukan arca Hindu), `opschrift` = label dokumen (bukan prasasti)
- Estimasi precision: **<15%** — keyword matching tidak cukup untuk VOC dagregisters
- Ini **negative result bermakna**: memvalidasi bahwa Delpher/OV (1854–1949) lebih kaya signal arkeologi daripada dagregisters era awal (1600s–1700s)
- Normalisasi tidak mengubah precision — masalah domain-semantik, bukan ortografis
- Geographic bias: Batavia (407), Banten (107) — interior Java (Trowulan, Singosari) = 0

**Phase 2 requires:** annotasi 65 kalimat (annotation_sample_v1.csv, ~2 jam) → estimasi precision → language detection filter → NER fine-tuning

---

### C. PhD Track — UvA

- **Lamqaddam (UvA):** Pak Amien konfirmasi sudah aman (email terkirim). Status: chat dijadwalkan, BPI support letter in progress.
- **Vossen email:** Diperbarui di `docs/drafts/email_vossen_vu_globalise.md` — BPI Dosen dihapus (umur 48 likely expired), VOC-ArchNLP HKI ditambahkan, experiment count 207+. **Siap kirim setelah review Pak Amien.**

---

### D. Scholarship Research

Riset komprehensif selesai (thread terpisah). Summary di `docs/correspondence/BEASISWA_RESEARCH_20260423.md`.

**Kesimpulan:** Semua beasiswa pemerintah Indonesia TERTUTUP di umur 48 (LPDP maks 47, BKI maks 45, Beasiswa Unggulan maks 47). Yang terbuka:
- **Dutch promovendus model** — PhD di Belanda = karyawan, gaji €2.618–3.333/bln, tidak ada batas usia
- **MSCA Doctoral Networks** — no age limit, ~€3.800/bln, next positions 2027
- **NWO PhDs in Humanities** — no age limit, round 2027
- **Fulbright** — no age limit tapi USA only

---

## 2. State at end of session

| Item | Status |
|---|---|
| 6 papers under review | Unchanged (P2-JCAA, P7-Antiquity, P8-OL, P11-Archipel, P17-ArchCalc) |
| P0 v0.4 | Pending submission (checklist di `papers/P0_invisible_civilization/SUBMISSION_CHECKLIST.md`) |
| P1 | Phase 0 fallow di `papers/MASTERPIECE/` |
| E209 | AUC 0.844, landscape mosaic fix still pending (task #30) |
| E211 | Phase 1 COMPLETE, Phase 2 pending annotation |
| PhD Verberne | Proposal sent 2026-04-16, waiting |
| PhD Lamqaddam | Chat scheduled, AMAN |
| PhD Vossen | Email ready, belum terkirim |
| PhD Cohen | Apply Dec 2026 |
| HKI VOC-ArchNLP | Package built, docs ready, belum daftar ke DJKI |

---

## 3. Decisions/actions untuk Pak Amien

| Prioritas | Item | File |
|---|---|---|
| Segera | Kirim Vossen email | `docs/drafts/email_vossen_vu_globalise.md` |
| Segera | Daftar HKI ke DJKI | `docs/HKI/PANDUAN_PENDAFTARAN_DJKI.md` |
| 1–2 minggu | Annotate 65 kalimat | `results/E211_voc_mentions/annotation_sample_v1.csv` |
| Tunggu | Konfirmasi batas usia BPI Dosen 2026 | Cek lpdp.kemenkeu.go.id |
| Tunggu | Review P0 submission checklist | `papers/P0_invisible_civilization/SUBMISSION_CHECKLIST.md` |

---

## 4. Untuk Claude di session berikutnya

Baca HANDOFF ini dulu. State 2026-04-23: VOC-ArchNLP v1.0.0 sudah dibuat sebagai produk HKI (`tools/voc_archnlp/`), E211 Phase 1 sudah dieksekusi (33,930 candidates, temuan utama: `oudheden` = 0, precision <15%, ini negative result bermakna), Lamqaddam AMAN per Pak Amien, Vossen email siap kirim, scholarship research selesai (semua beasiswa Indonesia expired di umur 48, jalur terbaik = Dutch promovendus model + MSCA). Annotation sample 65 kalimat sudah siap di `results/E211_voc_mentions/annotation_sample_v1.csv` — ini adalah gate untuk Phase 2. Jangan jalankan AI review lagi (masih saturation per Session 21). Jangan polish P0 tanpa permintaan spesifik. P1 = Phase 0 fallow. Mulai session berikutnya dengan menunggu keputusan Pak Amien untuk: (a) kirim Vossen email, (b) daftar HKI, (c) annotate 65 kalimat.

---

*HANDOFF produced 2026-04-23 end-of-session-22.*
