# VOLCARCH Handoff - 2026-02-24 (Night)

## Context Snapshot
- User instruction terakhir: **jangan push ke GitHub dulu**.
- Branch aktif: `main`
- Status branch: `main...origin/main [ahead 1]`
- Commit lokal terbaru yang belum dipush: `8c93743` (`chore: push full project snapshot (data, experiments, papers)`).
- Commit remote terakhir: `a595a03` pada `origin/main`.

## Kenapa Push Terasa Berat
- Ada banyak artefak besar di repo (DEM `.tif` ratusan MB per file, HTML suitability map ~80+ MB per file).
- `.tif` sudah di-track via Git LFS (`.gitattributes: *.tif filter=lfs diff=lfs merge=lfs -text`), tapi upload LFS tetap besar dan lama.
- File HTML besar belum masuk LFS, sehingga ukuran push tetap tinggi.

## Pekerjaan Sesi Ini (Sudah Selesai, Belum Dipush)
1. LaTeX toolchain:
- MiKTeX terpasang via winget (`MiKTeX.MiKTeX 25.12`).

2. Manuskrip LaTeX interdisipliner:
- `papers/P2_settlement_model/submission_remote_sensing_v0.2.tex`
- `papers/P2_settlement_model/submission_remote_sensing_v0.2.pdf` (berhasil compile).

3. Visual baru (ilustrasi/diagram):
- `papers/P2_settlement_model/build_interdisciplinary_visuals.py`
- `papers/P2_settlement_model/figures/fig1_interdisciplinary_framework.png`
- `papers/P2_settlement_model/figures/fig8_pipeline_overview.png`
- `papers/P2_settlement_model/figures/fig9_interpretation_bridge.png`

4. Update dokumen tracking:
- `docs/L3_EXECUTION.md`
- `docs/JOURNAL.md`
- `papers/P2_settlement_model/submission_checklist.md`
- `papers/P2_settlement_model/remote_sensing_template_map.md`

## Working Tree Saat Diserahterimakan
Modified:
- `docs/JOURNAL.md`
- `docs/L3_EXECUTION.md`
- `papers/P2_settlement_model/remote_sensing_template_map.md`
- `papers/P2_settlement_model/submission_checklist.md`

Untracked:
- `papers/P2_settlement_model/build_interdisciplinary_visuals.py`
- `papers/P2_settlement_model/figures/fig1_interdisciplinary_framework.png`
- `papers/P2_settlement_model/figures/fig8_pipeline_overview.png`
- `papers/P2_settlement_model/figures/fig9_interpretation_bridge.png`
- `papers/P2_settlement_model/submission_remote_sensing_v0.2.tex`
- `papers/P2_settlement_model/submission_remote_sensing_v0.2.pdf`

## Catatan Operasional Besok
- Jangan push otomatis; minta konfirmasi user dulu.
- Kalau lanjut kerja naskah: basis utama sekarang `papers/P2_settlement_model/submission_remote_sensing_v0.2.tex`.
- Compile command yang valid:
```powershell
& "C:\Users\Mukhlis Amien\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode -halt-on-error papers/P2_settlement_model/submission_remote_sensing_v0.2.tex
```
- Regenerate visual:
```powershell
py papers/P2_settlement_model/build_interdisciplinary_visuals.py
```

## Rekomendasi Lanjutan (Setelah User Konfirmasi)
1. Rapikan strategi ukuran repo sebelum push:
- Opsi A: pertahankan full snapshot + LFS upload (lama, tapi lengkap).
- Opsi B: pindahkan artefak besar non-esensial (mis. HTML map besar) ke release artifact/external storage, lalu commit yang lebih ringan.
2. Baru setelah user setuju, lakukan commit terstruktur dan push.
