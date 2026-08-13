SIG sign-off — P11 "Temples Without Villages" — 2026-08-11 — run by Claude (orbit review, retarget ke SPAFA)
Draf: `draft_v0.6_spafa.tex` (14 halaman, kompilasi bersih, 0 undefined ref)

G1 re-derivation: [GREEN] — angka kanonik di-re-derive **buta** dari `E031/results/canonical30/` pada 2026-08-11: mean bearing **297.9°**, Rayleigh **p=1.217×10⁻⁹**, W **47.9%** (68/142), E **9.2%** (13/142), Zone A **45.1%** (dari `alignment_summary_canonical30.json`). Semua cocok persis dengan `CANONICAL_INVENTORY_CORRECTIONS_20260610.md`.
⚠ Verifikasi tersisa: **E069 survey-control (β=−0.477, p=0.0015, baris ~259)** belum di-re-derive pada inventori kanonik 30 — angka ini tidak masuk daftar re-derivasi 2026-06-10. Re-run sebelum submit.
G2 domain-sanity: [GREEN] — "candi mengelompok di sisi barat-barat-laut gunung terdekat" dan "candi lebih dekat gunung daripada inskripsi" konsisten dengan de Groot (Merapi/Penanggungan), Penanggungan 73 candi di flank barat, dan letusan/angin barat. Pertanyaan kunci "apakah Penanggungan mendominasi?" dijawab (pola bertahan setelah dikeluarkan, p=0.0009).
G3 canonical data: [GREEN] — teks v0.6 memakai angka terkoreksi + kalimat inventori 30-puncak (baris ~80). Figur **fig1+fig2 diregenerasi dari canonical30** (2026-08-11; `generate_figures.py` di-update; fig1 diverifikasi visual: 298°/p=1.2e-9/47.9%/9.2%). fig3–5 tidak direferensikan di naskah (E032 seasonality FDR-casualty tidak ikut terkirim — benar).
G4 circularity: [GREEN] — Zone A dipakai deskriptif; kontras candi↔inskripsi memakai dua kategori independen (lokasi candi vs lokasi inskripsi), bukan variabel yang sama.
G5 equifinality: [GREEN] — kontrol defisit survei ada (E069, baris ~257–260) + pernyataan falsifiability via GPR (baris ~277–278: "if systematic geophysical survey finds nothing at predicted locations, the model would be refuted"). ⚠ E069 harus di-verify kanonik (G1 di atas).
G6 counter-evidence: [GREEN] — kanal yang bisa menyangkal dinyatakan eksplisit (GPR di lokasi prediksi). Bias preservasi arahnya menguatkan (burial menghapus candi dekat gunung → klaster sebenarnya ter-counedown), bukan mengancam. E214 (palinologi) tidak relevan langsung ke klaim kedekatan candi↔gunung.
G7 reproducibility: [GREEN] — `python generate_figures.py` meregenerasi semua figur dari canonical30; `pdflatex draft_v0.6_spafa.tex` kompilasi 14 halaman bersih. (Naskah memakai footnote inline, bukan bibtex — sengaja.)
G8 overstatement: [PARTIAL] — baris ~104 "all were either continuously visible or rediscovered" bisa dilunakkan menjadi "the 142 compiled candi"; baris ~266 "73% temple bias / only 5 settlements" tersumber (E129) tapi sebutkan n=391. Selebihnya bersih.
G9 cross-model: [NOT RUN] — belum ada review lintas-model untuk v0.6. JALANKAN sebelum submit (`tools/critical_reviewer_prompt.md` di DeepSeek).
G10 human independent review: [N/A] — notulen pendek; direkomendasikan jika retarget menjadi artikel penuh (opsional).

Downgrades made: inventori 16→30 menggeser angka tanpa mengubah arah kesimpulan (semua menguat atau tetap signifikan): 279°→298° (WNW), 47.2%→47.9%, kuadran timur "fewer than 4%"→"under 10%" (9.2%), Rayleigh 3.4e-8→1.2e-9, Zone A 42.3%→45.1%, overrep 17.9×→19.1×, gap candi↔inskripsi 9.2→6.1 km, MW p 5.2e-8→2.8e-7.

DECISION: **CONDITIONAL GO** — syarat sebelum kirim:
1. Re-derive E069 survey-control pada inventori kanonik (G1 tersisa). → ✅ **DONE 2026-08-13** — β = −0.831, p = 2.9×10⁻⁷ (menguat); `RESULTS_CANONICAL30_20260813.md`.
2. Jalankan G9 cross-model pada v0.6. → ✅ **DONE 2026-08-13** — `external_reviews/G9_CROSS_MODEL_20260813.md`: tidak ada fabrikasi, klaim inti selamat, 10 temuan presisi — **semua 10 diperbaiki** di v0.6 + v0.7 (mekanisme angin dikoreksi klimatologis, seksi 929 M di-re-derive kanonik 58/91/48/87%, Liangan 4–6 m, n=175, 13.1%, 2–9 m).
3. Cek format SPAFA Journal. → ✅ **VERIFIED 2026-08-13** — lihat `SPAFA_SUBMISSION_PREP.md` §3 (no-APC verbatim, Word/.RTF, dual-language, Harvard author-date, Figure Form, AI policy).
4. Soften G8 baris ~104 + verifikasi baris ~266. → ✅ **DONE** — kalimat ditulis ulang jujur (Sambisari 1966/Kimpulan 2009); 277/391 = 70.8% (73.1% komposit), terverifikasi E129.
5. Kompilasi final + konversi format sesuai SPAFA; portal submission = PI. → ✅ **SIAP** — `draft_v0.7_spafa.tex` (Harvard author-date, en-dash, per cent, Acknowledgements+AI disclosure, dual-language ID, References 29 entri semua-penulis via Crossref) → PDF 14 hal bersih + `spafa_assets/P11_submission_v0.7.docx` (template SPAFA) + Figure Form draf + cover letter.

**FINAL: 🟢 GO** — tinggal aksi PI: (a) review DOCX + terjemahan ID, (b) isi+tandatangani Figure Submission Form, (c) daftar di portal spafajournal.org dan submit (target ≤ 2026-08-20).
