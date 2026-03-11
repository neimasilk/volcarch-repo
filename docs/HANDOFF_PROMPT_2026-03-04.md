# VOLCARCH — Handoff Prompt (2026-03-04)

Copy-paste the section below (between the `---` markers) into a new Claude Code session.

---

## Prompt

Kamu melanjutkan pekerjaan pada **VOLCARCH** — research project tentang volcanic taphonomic bias pada rekaman arkeologi Indonesia.

### Konteks Proyek
- Repo: `volcarch-repo` — baca `CLAUDE.md` di root untuk aturan umum
- Baca `docs/L1_CONSTITUTION.md` → `L2_STRATEGY.md` → `L3_EXECUTION.md` untuk konteks riset
- **Baca `docs/HANDOFF_2026-03-04.md`** untuk detail lengkap session terakhir

### Status Per 2026-03-04

**Dashboard:**
- Deployment package sudah di-push ke https://github.com/neimasilk/volcarch-dashboard
- BELUM di-deploy ke Streamlit Cloud (perlu manual deploy di share.streamlit.io)
- Lokal: `python -m streamlit run tools/dashboard.py` atau `cd deploy/volcarch-dashboard && python -m streamlit run app.py`

**Paper 1 — Taphonomic Framework:**
- Target jurnal: **Internet Archaeology** (FREE, Scopus Q1 Archaeology)
- File: `papers/P1_taphonomic_framework/submission_intarch_v0.1.tex` (21 halaman, compile clean)
- Compile: `pdflatex → bibtex → pdflatex → pdflatex`
- BELUM submit — perlu: ORCID, cover letter/proposal, submit via form atau email editor@intarch.ac.uk
- Submit URL: https://uni-york.formstack.com/forms/iaproposal

**Paper 2 — Settlement Suitability Model:**
- Target jurnal: **Journal of Remote Sensing** (SPJ/AAAS, FREE sampai 2027, IF 6.8, Scopus Q1)
- File: `papers/P2_settlement_model/submission_jrs_v0.1.tex` (17 halaman, compile clean)
- Compile: `pdflatex → biber → pdflatex → pdflatex` (pakai biber, BUKAN bibtex)
- BELUM submit — perlu: cover letter (wajib mention AI use), submit via ScholarOne
- Submit URL: https://spj.science.org/journal/remotesensing

**Experiments:** E013-E017 complete. E017 (tephra POC) FAILED — Paper 3 butuh per-volcano calibration.

### Pending Actions (prioritas)
1. Deploy dashboard ke Streamlit Cloud → dapatkan URL
2. Register ORCID (jika belum) → update di Paper 1
3. Buat cover letter untuk kedua jurnal
4. Submit Paper 1 (proposal ke Internet Archaeology)
5. Submit Paper 2 (full submission ke J. Remote Sensing)

### Tugas Selanjutnya
[ISI TUGAS SPESIFIK DI SINI — contoh di bawah]

**Contoh tugas:**
- Buatkan cover letter untuk Paper 1 (Internet Archaeology)
- Buatkan cover letter untuk Paper 2 (Journal of Remote Sensing, harus mention AI use)
- Deploy dashboard dan update URL di kedua paper
- Review final Paper 2 sebelum submit
- Update L3_EXECUTION.md dan JOURNAL.md dengan progress hari ini

---

## Catatan Penting untuk Session Baru

1. **Selalu baca CLAUDE.md** dulu — ada aturan research integrity.
2. **Paper 2 pakai `biber`** (bukan `bibtex`) — compile: `pdflatex → biber → pdflatex → pdflatex`.
3. **Paper 1 pakai `bibtex`** — compile: `pdflatex → bibtex → pdflatex → pdflatex`.
4. **Dashboard deploy** pakai `deploy/volcarch-dashboard/` (bukan `tools/dashboard.py`) — sudah ada di repo terpisah.
5. **AI disclosure wajib** di Paper 2 cover letter (kebijakan SPJ).
6. **ORCID placeholder** di Paper 1 perlu diganti dengan ORCID asli sebelum submit.
7. **Jangan modify raw data** di `data/raw/`.
8. **Catat perubahan** di `docs/JOURNAL.md` (append-only log).
