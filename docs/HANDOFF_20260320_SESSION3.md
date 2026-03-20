# HANDOFF — Mudik Session 3 (2026-03-20)

## Apa yang Dikerjakan

Sesi otonom: 3 sesi sebelumnya (Session 1-2) menghasilkan E115 + structural critique + pre-mortem. Session 3 melanjutkan dalam mode fully autonomous.

### Eksperimen Baru (E116-E119)

| ID | Judul | Hasil Kunci |
|----|-------|-------------|
| **E116** | Testable Predictions | 20 GPR survey di E080 targets → expect 2.5 temuan, 95% CI [0,6], P(zero)=7%. Framework IS falsifiable. Biaya: $40K-100K. |
| **E117** | Archaeological Onset Analysis | Detection horizon: surface survey hanya menjangkau ~1900 CE pada 4mm/yr. Semua situs pra-400 CE di Jawa ada di gua, teras sungai, atau pantai. Zero open-air di interior vulkanik. |
| **E118** | Information Gain | 3.5× efisiensi pencarian, 29% reduksi entropi, hemat $16.7K per temuan pertama. Survey deficit = masalah terbesar, volcanic context = solusi terbaik. |
| **E119** | Synthesis Figure | Data JSON untuk satu figur yang menceritakan seluruh cerita VOLCARCH: diagonal kedalaman × horizon deteksi × situs dikenal. Render dengan matplotlib post-mudik. |

### Tools & Dokumen

| Item | Lokasi | Fungsi |
|------|--------|--------|
| Auto-sync checker | `tools/check_doc_sync.py` | Cek konsistensi jumlah eksperimen di 6 dokumen. Exit code 0/1. Jalankan sebelum commit. |
| Falsifiability package | `papers/P1.../revision_ammo/FALSIFIABILITY_PACKAGE.md` | Paragraf siap-pakai untuk reviewer yang tanya "is this falsifiable?" |
| README | `README.md` | Diperbarui — sekarang mencerminkan 120 eksperimen, tabel hasil, status paper. |
| TRIGGER_MAP | `docs/TRIGGER_MAP.md` | Diperbarui — status rejection + 5 rules dari rejection pattern analysis. |

### Keputusan Penting

1. **Michelson-Morley framing** — Proyek ini seperti eksperimen Michelson-Morley: nilai ada di METHOD + PREDICTIONS, bukan di discovery. Null result (GPR tidak menemukan apa-apa) tetap kontribusi ilmiah. Kedua outcome adalah kontribusi.

2. **Bias acknowledged** — Diskusi jujur tentang kemungkinan bias: "mungkin memang tidak ada peradaban di interior vulkanik Jawa pra-400 CE." Kesimpulan: still go, dengan kemungkinan pivot. Framework tetap valuable sebagai methodology paper.

3. **Failed experiment rescue** — Analisis E024/E039/E081: semuanya sudah implicitly rescued oleh eksperimen lanjutan (E083, E103, E110). Tidak perlu eksperimen baru untuk rescue.

## Status Saat Ini

- **120 eksperimen** (E001-E119 + E095)
- **3 paper under review** (P2-JCAA, P7-Antiquity, P8-OL)
- **P1 EGQSJ fully ready** — tinggal register di editor.copernicus.org dan upload
- **Falsifiability package lengkap:** E115 (robustness) → E116 (predictions) → E117 (detection horizon) → E118 (practical value) → E119 (synthesis figure)

## Post-Mudik Action Items (Urutan Prioritas)

1. **P1 submit EGQSJ** — register editor.copernicus.org → upload `submission_egqsj_v1.0.tex` → submit
2. **Verify JCAA APC** — P2 charges £300-450. Cek apakah waiver sudah applied
3. **Render E119 synthesis figure** — `pip install matplotlib` → render dari JSON data
4. **P11 review manual** → submit Wacana
5. **P17 review manual** → submit Archeologia e Calcolatori
6. **D1+D2 → Zenodo** (30 menit per paper, gratis)

## File yang Dimodifikasi

```
NEW:
  experiments/E116_testable_predictions/     (script + results + README)
  experiments/E117_archaeological_onset/     (script + results + README)
  experiments/E118_information_gain/         (script + results + README)
  experiments/E119_synthesis_figure/         (script + results + README)
  tools/check_doc_sync.py
  papers/P1.../revision_ammo/FALSIFIABILITY_PACKAGE.md
  docs/HANDOFF_20260320_SESSION3.md          (this file)

MODIFIED:
  README.md
  docs/WORKSTATE.md, JOURNAL.md, EXPERIMENT_INDEX.md
  docs/L1_CONSTITUTION.md, L2_STRATEGY.md, L3_EXECUTION.md, EVAL.md
  docs/TRIGGER_MAP.md
```

## Git Log

```
7439b43 feat: E119 synthesis figure + 120 experiments milestone
63345c6 docs: update README, WORKSTATE, TRIGGER_MAP — session 3 wrap-up
0885414 feat: E117-E118, falsifiability package, auto-sync checker — mudik session 3
49a2ad9 feat: E115-E116, pre-mortem analysis, rejection patterns, cascade robustness — mudik session 2-3
c3b38f2 feat: P1 EGQSJ fully ready + structural critique + Diamond OA targets — mudik session
```
