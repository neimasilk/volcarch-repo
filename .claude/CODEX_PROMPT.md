# Prompt for Codex — VOLCARCH Project Continuation

Copy-paste prompt ini ke Codex. Handoff document ada di `.claude/HANDOFF_2026-02-24.md`.

---

## PROMPT

```
Kamu melanjutkan project riset VOLCARCH (Volcanic Taphonomic Bias in Indonesian Archaeological Records). Ini bukan software project — ini research repo, output-nya paper dan model.

PERTAMA, baca file-file berikut SEBELUM melakukan apapun:
1. CLAUDE.md (instruksi utama)
2. .claude/HANDOFF_2026-02-24.md (handoff dari sesi sebelumnya — BACA INI PALING PENTING)
3. docs/L3_EXECUTION.md (task aktif saat ini)
4. docs/EVAL.md (metrik evaluasi)

KONTEKS SINGKAT:
- Kita sedang membangun settlement suitability model (Paper 2 / H3 test)
- E007 (5 terrain features) → AUC=0.659 (BELOW MVR 0.75)
- E008 (+ river distance) → AUC=0.695 (BELOW MVR, tapi improving)
- Challenge 1 (tautology test) consistently PASSES — model tautology-free
- AUC naik monoton setiap tambah fitur: 0.659 → 0.695 → ???

TUGAS SEKARANG — E009:
1. Download soil data dari SoilGrids (clay + silt fraction) untuk Jawa Timur
   - Bounding box: lon 111-115, lat -9 to -6.5
   - Bulk GeoTIFF dari https://files.isric.org/soilgrids/latest/data/
   - ATAU gunakan REST API: https://rest.isric.org/soilgrids/v2.0/properties/query
   - Resample ke grid DEM yang sama (EPSG:32749, 30m pixel size)
   - Simpan sebagai data/processed/dem/jatim_clay.tif dan jatim_silt.tif
2. Buat experiments/E009_settlement_model_v3/ (README.md + script)
   - Copy pattern dari experiments/E008_settlement_model_v2/01_settlement_model_v2.py
   - Tambah clay dan silt ke FEAT_COLS
   - Run spatial block CV yang sama (5 folds, ~50km blocks)
3. Evaluasi:
   - AUC > 0.75 → Paper 2 GO (tulis di README sebagai SUCCESS)
   - AUC 0.65-0.75 → REVISIT (coba Target-Group Background pseudo-absences)
   - AUC < 0.65 → KILL SIGNAL
   - Challenge 1 HARUS tetap PASS (rho < 0.3)
4. Update docs/JOURNAL.md (append entry baru, JANGAN edit entry lama)
5. Update docs/L3_EXECUTION.md dengan hasil

ATURAN PENTING:
- Jangan fabricate data
- Semua experiment di-log di JOURNAL.md
- Failed experiments TIDAK dihapus, didokumentasi
- Challenge 1 harus selalu pass — JANGAN masukkan volcanic proximity sebagai feature
- Python 3.10+, gunakan geopandas, rasterio, scikit-learn, xgboost
- Platform: Windows 11, shell bash, Python via `py` command
- Working directory: C:\Users\Mukhlis Amien\Documents\volcarch-repo

Kalau SoilGrids download gagal atau terlalu besar, alternatif:
- Download hanya clay content (skip silt)
- Atau gunakan HWSD v2 (FAO Harmonized World Soil Database) — download GeoTIFF dari https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v2/en/
- Atau skip soil dan langsung coba Target-Group Background (Path B dari handoff)

Jika AUC > 0.75 tercapai, lanjut ke:
- Update E009 README dengan "SUCCESS"
- Mulai outline Paper 2 di papers/P2_settlement_model/outline.md
- Update L3_EXECUTION.md dan JOURNAL.md
```
