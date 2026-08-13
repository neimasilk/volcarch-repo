# P11 → SPAFA Journal — Submission Prep (2026-08-11)

**Status:** SIAP-KIRIM secara konten (SIG CONDITIONAL GO). Portal submission = PI.
**Draf:** `draft_v0.6_spafa.tex` → `draft_v0.6_spafa.pdf` (14 halaman, kompilasi bersih).
**Riwayat:** tolak 2× (Cornell *Indonesia* — scope; *Archipel* — editorial). Temuan inti selamat
(candi–pemukiman gap 6.78 km, 80.6% <10 km, p<1e-6, kebal inventori). Retarget: **SPAFA Journal**.

---

## 1. Yang SUDAH dikerjakan sesi ini

| Item | Status |
|---|---|
| Angka kanonik diterapkan ke teks (baris 116/121/139/171/172): 298°·p=1.2e-9·47.9%·9.2%·45.1%·19.1×·6.1 km·p=2.8e-7 | ✅ |
| Kalimat inventori 30-puncak ditambahkan (§Volcanic Landscape) | ✅ |
| **Re-derivasi buta (G1)** dari `canonical30/`: semua angka cocok persis (297.9°, p=1.217e-9, W47.9%, E9.2%) | ✅ |
| **Figur fig1+fig2 diregenerasi** dari canonical30 (sebelumnya hardcode 279°/47.2%/3.5%/3.4e-8 — inkonsisten dengan teks; pasti tertangkap reviewer) | ✅ |
| Toolchain `generate_figures.py` diperbaiki (fig4 boxplot API matplotlib; semua 5 figur ter-regenerasi) | ✅ |
| Kompilasi final 14 halaman, 0 undefined reference | ✅ |
| SIG sign-off jujur | ✅ `SIG_signoff.md` |

## 2. SIG sign-off ringkas

**CONDITIONAL GO** — 5 syarat: (1) re-derive E069 survey-control pada inventori kanonik · (2) G9
cross-model pada v0.6 · (3) cek format SPAFA · (4) soften G8 baris ~104 + verifikasi baris ~266 ·
(5) konversi format + submit = PI. Detail: `SIG_signoff.md`.

## 3. Yang BELUM (tindakan tersisa, urut prioritas) — status 2026-08-13

1. ~~**Cek format SPAFA Journal**~~ ✅ **VERIFIED 2026-08-13** dari situs resmi (spafajournal.org →
   Submissions → Author Guidelines, "Last updated: 31 December 2025"). Ringkasan yang mengikat:
   - **NO APC** (verbatim: "The SPAFA Journal has no article processing charges (APCs) or any other
     charges.") — syarat zero-APC terpenuhi.
   - **Format: Microsoft Word / OpenOffice / .RTF** (bukan PDF/LaTeX) — konversi wajib. Template:
     SPJ Template 2026 Edition (diunduh: `spafa_assets/SPJ_Template_2026.docx`).
   - **Dual-language:** judul + abstrak + kata kunci wajib dalam bahasa Inggris DAN bahasa Asia
     Tenggara terkait (→ Bahasa Indonesia; draf ada di `SPAFA_DUAL_LANGUAGE_20260813.md`).
   - **Sitasi: Harvard author-date** — "(Binford 1983: 6)"; catatan kaki HANYA untuk diskusi, bukan
     sitasi. ⚠ Naskah kita memakai footnote Chicago penuh → **konversi sitasi wajib** sebelum submit.
   - **Figure Submission Form** wajib untuk setiap naskah berfigur (form diunduh:
     `spafa_assets/SPAFA_Figure_Submission_Form.docx`; draf isian di `SPAFA_FIGURE_FORM_DRAFT_20260813.md`).
   - Figur ≥240 dpi, disisipkan di dalam teks, caption dengan "Source: ..." — figur kita 300 dpi ✓
     (generate_figures.py `savefig.dpi: 300`), Source = "by the authors".
   - **AI policy:** AI boleh untuk editing bahasa + analisis data, WAJIB dideklarasi di bagian
     **Acknowledgments** dengan nama+versi tool, tanggal pemakaian, dan deskripsi singkat. AI-generated
     images dilarang (figur kita plot matplotlib dari data — aman, nyatakan di deklarasi).
   - Gaya: ejaan Inggris (British disukai), **tanpa em dash** (pakai en dash), "per cent" dieja di
     prosa, CE/BCE, 12pt single-spaced, angka satu-sembilan dieja.
   - Submit via portal OJS spafajournal.org (Login/Register) — aksi PI.
2. ~~**Re-derive E069**~~ ✅ **DONE 2026-08-13** — `adv3_survey_intensity_canonical30.py`: β = **−0.831**,
   p = **2.9×10⁻⁷** (quasi-LR) — **menguat** dari β=−0.477/p=0.0015. Catatan:
   `experiments/E069_adversarial_comparanda/adv3_survey_intensity/RESULTS_CANONICAL30_20260813.md`.
   Teks + footnote naskah baris ~259 diperbarui; kompilasi bersih, glyph OK.
3. **G9 cross-model** — ⏳ berjalan (subagent adversarial diprogram menolak, 2026-08-13); hasil →
   `external_reviews/G9_CROSS_MODEL_20260813.md`.
4. ~~**G8**~~ ✅ **DONE 2026-08-13** — baris ~104 dilunakkan ("the 142 compiled candi"); baris ~266
   diverifikasi ke sumber E129: 277/391 = **70.8%** (bukan 73%); 73.1% = candi+arca+prasasti.
   Teks + footnote dikoreksi.
5. **Konversi + submit** — tersisa: (a) konversi sitasi footnote→Harvard author-date + reformat
   bibliography; (b) terjemahan dual-language dimasukkan; (c) deklarasi AI → Acknowledgments dengan
   versi+tanggal; (d) gaya SPAFA (en-dash, per cent, ejaan); (e) ekspor .docx via pandoc dari template;
   (f) isi Figure Submission Form + tanda tangan PI; (g) submit portal = PI.

## 4. Cover letter (SPAFA)

```markdown
# Cover Letter — SPAFA Journal

**To:** The Editors, SPAFA Journal (SEAMEO Regional Centre for Archaeology and Fine Arts)
**From:** Mukhlis Amien, Universitas Bhinneka Nusantara, Malang, Indonesia
**Re:** Manuscript submission — "Temples Without Villages: Candi and the Hidden Settlement Geography of Volcanic Java"

Dear Editors,

We submit for your consideration the attached manuscript, "Temples Without Villages: Candi and the
Hidden Settlement Geography of Volcanic Java."

The paper argues that Java's surviving Hindu-Buddhist temples (candi) serve as spatial markers of a
settlement landscape that volcanic sedimentation has rendered invisible to archaeological survey.
Through analysis of 142 candi and 170 georeferenced inscriptions, we show that temples cluster on the
west-northwest flanks of the island's volcanoes — precisely the zones where burial by pyroclastic
deposits is most intense — while their orientation follows religious canon, demonstrating deliberate
landscape reading by their builders. Inscriptions sit on average ~6 km farther from volcanoes than
temples, marking a separate administrative geography. The 2008 discovery of the buried village of
Liangan validates this framework.

We believe this paper suits SPAFA Journal because it addresses a core Southeast Asian archaeological
question — the taphonomy of the region's volcanic landscapes and what it hides — with a method
(spatial statistics of standing monuments) directly relevant to heritage practice across the
archipelago.

The manuscript is approximately 3,200 words with 2 figures. It has not been published elsewhere and
is not under consideration at any other journal. All data and computational scripts are publicly
available (https://github.com/neimasilk/volcarch-repo).

**AI Disclosure:** AI tools (Claude, Anthropic) were used for spatial geocoding, statistical
computation, and manuscript drafting assistance. Research design, hypothesis formation, data
interpretation, and all conclusions are the authors' work.

Thank you for considering this submission.

Sincerely,
Mukhlis Amien
Lab Data Sains, Universitas Bhinneka Nusantara
Jl. Bunga Andong Selatan No.73, Malang 65141, Indonesia
amien@ubhinus.ac.id | ORCID: 0000-0002-1848-167X
```

*(Simpan ke `cover_letter_spafa.md`; perbarui jumlah kata saat format final.)*

## 5. Catatan strategis

- Temuan inti P11 **kebal inventori** (koreksi menguatkan) — ini argumen utama yang dipakai di
  abstract/cover: stabilitas terhadap koreksi kanonik adalah kekuatan, bukan kelemahan.
- Jika SPAFA menolak cepat (editorial), urutan cadangan tetap: Wacana (jika isu terbuka cocok) →
  PCI Archaeology. Jangan membuka line baru.
