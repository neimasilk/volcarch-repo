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

---

## 2026-10-01 — RE-CHECK v0.8 (after the E069 sign correction): 🔴 **NO-GO**

**Trigger:** E069/ADV-3 sign misread (ledger C023); v0.8 removed the claim. Two Opus verifiers then
re-checked v0.8 (workflow `p11-v08-sig-recheck`); the orchestrator re-derived the quantitative
blockers independently (`scratchpad/p11_blocker_check.py`, numbers below are from that re-run).

**Condition 1 of 2026-08-11 (E069 survey control): VOID.** The coefficient is on distance; it shows a
surplus of recorded sites near volcanoes. Removed in v0.8.

**Blockers (verified):**
1. **The island-wide western-flank claim does not survive excluding Penanggungan** on the canonical
   30-volcano inventory. All 142: Rayleigh p=5.9×10⁻¹⁰, mean bearing 297.9°. Penanggungan (n=69):
   p=1.4×10⁻²⁰, 62.3% in the western quadrant. **Remaining 73: R=0.136, p=0.26, mean bearing 227°,
   34.2% west — no significant clustering.** The text (tex 152–153: "the remaining … still show
   significant clustering, p=0.0009") quotes a pre-canonical number. Figure 2 already shows n=69 against a
   caption of 73.
2. **Duplicated candi coordinates:** the 142 rows of `E031/results/candi_volcano_pairs.csv` have only
   **103 distinct coordinates** (39 rows repeat another row's exact position). Every candi statistic
   counts some temples more than once. The coordinates come from the OSM/Wikipedia compilation, not from
   "published surveys, heritage registries, and standard catalogues" (tex 116).
3. **The Word file has no abstract in either language** (paragraphs: title → authors → affiliation →
   corresponding author → Keywords → Judul → Kata kunci → Introduction). SPAFA requires both. This was
   already true of v0.7, and the 2026-08-13 FINAL GO missed it.
4. **E153 co-location support** relies on the same compiled list; most of its 108 "non-temple sites" are
   not settlements, and its null uses uniform points in a box that includes sea. It is robust to removing
   modern features (6.78 → 6.22–6.61 km, C029) but should be stated as a rough check, not as a
   candi-settlement test.
5. **"71 per cent temple bias in East Java's archaeological database" footnote:** the 391-entry list is a
   compiled Wikipedia/Wikidata list (not East Java only), and its five "settlements" are not settlements
   (verifier report; to re-check against E129's data).
6. **SIG record certified v0.7** on results that no longer hold (G2 relied on p=0.0009; G5 on E069).

**SHOULD_FIX (from the verifier):** tex 272 "What this means" and 277 "what survives" still lean on the
removed conclusion; tex 85/297 assert erasure as established; tex 267 keeps the F9 "convergence" framing;
tex 97–98 calls depth ÷ assumed age a "measured sedimentation rate" (C024); AI-disclosure dates should
run to October 2026; running heads still read "Title of article".

**P17 cross-check (same candi list):** with the 103 unique coordinates the P17 contrast survives —
candi median 16.6 km vs inscriptions 27.6 km (was 14.5), gap 11.0 km (was 13.1), Mann–Whitney p=1.3×10⁻⁵
(was 1.1×10⁻⁷). The numbers shift; the conclusion holds.

**Decision: 🔴 NO-GO for SPAFA.** Do not submit v0.7 or v0.8. Needed before the next SIG: de-duplicate
the candi list; re-derive E031/E065/E105/E153 on it; downgrade the island-wide western-flank claim (it is
a Penanggungan pattern, partly through Trowulan temples assigned to Penanggungan); restore both abstracts
in the docx; re-run the SIG including **direction/sign checks** (lesson C023). Reframing the paper around
Penanggungan is a PI decision.
