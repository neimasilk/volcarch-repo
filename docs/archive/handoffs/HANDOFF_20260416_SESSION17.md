# HANDOFF: Session 17 — Mata Elang #14 + PhD Sent + Core Stack (2026-04-16)

**Dari:** Claude (sesi 17)
**Untuk:** Sesi berikutnya
**Durasi:** ~6 jam (pipeline mode)

---

## RINGKASAN 30 DETIK

Sesi otonom terpanjang sejauh ini. PhD proposal **SENT** ke Verberne (2 fix kritis). Mata Elang #14 = kritik struktural terdalam (9 blind spot, 4 fatal risk). **7 eksperimen baru** (E201-E207, total 207). Strategi pivot: **no more paper submissions** — paper = amunisi PhD. **Riset dalam diam.** Core stack untuk PhD pipeline dibangun: GLOBALISE downloader, VOC preprocessor, spelling normalizer, 6.26 juta kata corpus siap NER. ArcheoBERTje (model Verberne) dijalankan pada data kita → gap 60% terkuantifikasi.

---

## KEPUTUSAN STRATEGIS SESI INI

### 1. PhD Proposal SENT ke Verberne
- v0.2 (2 fix: ±1.2 dihapus dari RQ4, E075/E083 conflation dipisah)
- Cover email: `docs/correspondence/phd_proposal/COVER_EMAIL_VERBERNE.md`
- **WAIT for response** (~1-2 minggu)

### 2. No More Paper Submissions
- 6 paper under review sudah cukup untuk credibility
- Paper menjadi **amunisi PhD**, bukan goal sendiri
- Riset dalam diam — tidak go-public, tidak submit baru
- Flexibility terjaga untuk pivot ke Cohen/Verberne/lainnya

### 3. Vossen Email DITAHAN
- Tunggu clarity Verberne dulu (~1-2 minggu)
- Kalau Verberne positif → Vossen approach sebagai co-promotor
- Kalau Verberne diam/tolak → Vossen jadi track alternatif

### 4. Core Stack = PhD-Agnostic Foundation
- Pipeline yang berguna untuk KEDUA track (Verberne NLP atau Cohen structured prediction)
- GLOBALISE data, preprocessing, normalizer — semua supervisor-independent

---

## DELIVERABLES

### Mata Elang #14
- **File:** `docs/research_notes/MATA_ELANG_14_2026_04_16.md`
- 4 fatal risks, 9 structural blind spots, 10 experiment recommendations
- Meta-insight: project = "quantitative absence detector" → harus flip ke "reconstructing presence"
- 10 new ideas registered: I-137 to I-146

### Experiments (7 baru: E201-E207)

| ID | Temuan | Status | PhD? |
|----|--------|--------|:--:|
| E201 | Philippines 55-65% open-air, gap LARGER | SUCCESS | Revision support material |
| E202 | 30m DEM can't detect candi (sub-pixel) | INCONCLUSIVE | LiDAR planning |
| E203 | Genome: 5th evidence channel, Java aDNA blank | SUCCESS | Independent evidence |
| E204 | Bronze drums: ~40 in volcanic Java | SUCCESS | "Selective survival" reframe |
| E205 | Wayang: 20-30% sempalan, punakawan = indigenous deities | SUCCESS | Living evidence |
| **E206** | **ArcheoBERTje gap: 60% entity types missing** | **SUCCESS** | **PhD core** |
| **E207** | **GLOBALISE VOC pilot: 6,893 files, CC0, 55% performance drop** | **SUCCESS** | **PhD core** |

### Core Stack (PhD Pipeline Foundation)

| Component | File | Status |
|-----------|------|--------|
| GLOBALISE downloader | `tools/globalise_pipeline/download_globalise.py` | DONE |
| VOC preprocessor | `tools/globalise_pipeline/preprocess_voc.py` | DONE |
| Spelling normalizer | `tools/globalise_pipeline/normalize_colonial_dutch.py` | DONE (10/10 tests) |
| Corpus (50 files) | `data/raw/globalise_voc/` (39 MB) | DONE |
| Preprocessed corpus | `data/processed/globalise_voc/` | DONE (34,545 paragraphs, 6.26M words) |

### Key Numbers
- **207 experiments** total (E001-E207, E180 skipped)
- **6.26 juta kata** VOC corpus preprocessed and ready for NER
- **6,893** GLOBALISE inventory numbers available (we have 50)
- **ArcheoBERTje gap:** 3 missing entity types, 55% performance drop on VOC vs OV
- **50+ colonial place names** in normalizer dictionary

---

## YANG BELUM SELESAI

| Item | Status | Next |
|------|--------|------|
| Gold-standard NER annotations (500 sentences) | NOT STARTED | Select sentences → annotate with 7 entity types |
| Colonial place-name gazetteer prototype | NOT STARTED | Extract E091 names → map to modern GIS |
| Verberne response | WAITING | ~1-2 minggu |
| Cohen formal application | Dec 2026 | Prepare closer to deadline |
| Vossen email | ON HOLD | Tunggu Verberne |
| IELTS | NOT SCHEDULED | Mid-2026 |
| 6 papers under review | WAITING | P1, P2, P7, P8, P11, P17 |
| Castillo email | READY TO SEND | Independent of PhD |

---

## SCORECARD

| Metrik | Awal Sesi | Akhir Sesi |
|--------|-----------|------------|
| Experiments | 200 | **207** (+7) |
| PhD proposal | Ready | **SENT** |
| Core stack tools | 0 | **3** (downloader, preprocessor, normalizer) |
| VOC corpus | 0 words | **6.26M words** (50 files) |
| ArcheoBERTje evaluated | No | **Yes** (gap quantified: 60%) |
| GLOBALISE accessed | No | **Yes** (API, 50 files downloaded) |
| Mata Elang | #13 | **#14** (deepest critique) |
| New ideas | I-136 | **I-146** (+10) |
| Strategy | Submit papers | **Riset dalam diam** (PhD evidence base) |

---

## FILES CHANGED/CREATED THIS SESSION

### New experiments
- `experiments/E201_philippines_deep_comparison/` (README + results)
- `experiments/E202_dem_depression_detection/` (README + script + results + figures)
- `experiments/E203_genome_population_structure/` (README)
- `experiments/E204_bronze_drum_distribution/` (README)
- `experiments/E205_wayang_indigenous_layer/` (README)
- `experiments/E206_archeobert_colonial_gap/` (README + script + results)
- `experiments/E207_globalise_voc_pilot/` (README + script + results + data)

### Tools
- `tools/globalise_pipeline/download_globalise.py`
- `tools/globalise_pipeline/preprocess_voc.py`
- `tools/globalise_pipeline/normalize_colonial_dutch.py`
- `tools/globalise_pipeline/globalise_file_index.json` (6,893 file IDs cached)

### Data
- `data/raw/globalise_voc/` (50 VOC transcription files, 39 MB)
- `data/processed/globalise_voc/` (preprocessed: 34,545 paragraphs, 6.26M words)

### Documents
- `docs/HANDOFF_20260416_SESSION17.md` (this file)
- `docs/research_notes/MATA_ELANG_14_2026_04_16.md`
- `docs/correspondence/phd_proposal/COVER_EMAIL_VERBERNE.md`
- `docs/correspondence/phd_proposal/PhD_Proposal_Amien_Leiden_v0.1.tex` (fixed: ±1.2, E075/E083)
- `docs/correspondence/phd_proposal/PhD_Proposal_Amien_Leiden_v0.1.pdf` (recompiled)
- `docs/correspondence/phd_proposal/CLAIM_AUDIT_TRAIL.md` (updated: ±1.2 source found, v0.2 fixes)
- `docs/JOURNAL.md` (updated: Session 17 entry)
- `docs/IDEA_REGISTRY.md` (updated: I-137 to I-146)

