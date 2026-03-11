# VOLCARCH Dashboard — Handoff Prompt

Copy-paste the section below (between the `---` markers) into a new Claude Code session to resume dashboard work.

---

## Prompt

Kamu melanjutkan pekerjaan pada **VOLCARCH Interactive Dashboard** — Streamlit app untuk eksplorasi hasil model settlement suitability (Paper 2).

### Konteks Proyek
- Repo: `volcarch-repo` — research project tentang volcanic taphonomic bias pada rekaman arkeologi Indonesia
- Baca `CLAUDE.md` di root repo untuk aturan umum
- Baca `docs/L1_CONSTITUTION.md` → `L2_STRATEGY.md` → `L3_EXECUTION.md` untuk konteks riset

### Status Dashboard (2026-03-03)
Dashboard **SUDAH BERFUNGSI**. Semua 4 tab verified via Playwright headless test.

**Files utama:**
- `tools/dashboard.py` — Streamlit app (~454 lines, single file)
- `tools/precompute_dashboard_data.py` — Script precompute data (~608 lines)
- `tools/test_dashboard.py` — Playwright test script
- `data/processed/dashboard/` — 10 pre-computed files (~5MB total)
- `tools/DASHBOARD_HANDOFF.md` — Dokumentasi lengkap arsitektur, data flow, bug fix history

**Teknologi:**
- Streamlit 1.54.0 + streamlit-folium 0.26.2 + Folium 0.20.0
- XGBoost model (AUC=0.768, TSS=0.507)
- Playwright 1.57.0 untuk browser testing
- PENTING: Pakai `python -m streamlit run` (bukan `streamlit run`)

**4 Tabs:**
1. **Peta Interaktif** — Folium map, 3 switchable layers (suitability heatmap, zone overlay, burial depth), site + volcano markers
2. **Analisis SHAP** — Beeswarm + bar chart PNGs, summary table
3. **Klasifikasi Zona** — Zone legend (A/B/C/E), statistics, Dwarapala validation
4. **Validasi Model** — AUC progression chart (E007-E013), tautology test, temporal validation (E014)

**Bug yang sudah di-fix:**
- `MarshallComponentException: Object of type function is not JSON serializable` — root cause: streamlit 1.30.0 tidak support `on_change` callback untuk custom components. Fix: upgrade ke 1.54.0 + gradient keys string→float.

### Cara Mulai
1. Baca `tools/DASHBOARD_HANDOFF.md` untuk detail arsitektur dan data flow
2. Jalankan dashboard: `python -m streamlit run tools/dashboard.py`
3. Jalankan Playwright test: `python tools/test_dashboard.py`
4. Screenshots tersimpan di `tools/screenshots/` (tab1-tab4)

### Tugas Selanjutnya
[ISI TUGAS SPESIFIK DI SINI — contoh di bawah]

**Contoh tugas yang bisa dilakukan:**
- Tambah fitur download/export untuk grid predictions atau site list
- Tambah tab baru untuk Paper 3 (tephra modeling)
- Improve zone layer performance (ganti CircleMarkers → GeoJSON)
- Tambah filter/search situs arkeologi di peta
- Buat deployment config (Docker, Streamlit Cloud)
- Tambah bahasa Indonesia penuh untuk semua label dan deskripsi

---

## Catatan Penting untuk Session Baru

1. **Selalu baca CLAUDE.md** dulu — ada aturan research integrity, experiment protocol, dll.
2. **Jangan modify raw data** di `data/raw/` — hanya ubah `data/processed/`.
3. **Catat perubahan** di `docs/JOURNAL.md` (append-only log).
4. **Kalau re-precompute**, pastikan output zone counts tetap match: A=15,217 / B=1,093 / C=48 / E=49,074.
5. **Kill stale servers** sebelum test: cek `netstat -ano | grep :8502` dan `taskkill /F /PID <pid>`.
6. **Browser bilingual** — semua label pakai format "Indonesian / English".
