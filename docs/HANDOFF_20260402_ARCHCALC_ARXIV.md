# HANDOFF: arXiv Published + ArchCalc Formatting (2026-04-02)

**Dari:** Claude (sesi 10)
**Untuk:** Pak Amien (besok)
**Durasi:** ~1 jam

---

## RINGKASAN 30 DETIK

P8 arXiv terbit (2604.00023). P17 ArchCalc submission package selesai diformat ulang setelah audit ketat terhadap paper ArchCalc yang sudah terbit — beberapa aturan di website berbeda dengan praktek di jurnal. Tabel di-rebuild manual karena pandoc hancurkan formatnya.

---

## MILESTONE: P8 arXiv PUBLISHED

- **arXiv:2604.00023** — http://arxiv.org/abs/2604.00023
- Kategori cs.CL, CC BY 4.0. Muncul di mailing 2026-04-02.
- Paper password untuk Go Frendi claim ownership: `ze47x`
- VOLCARCH sekarang punya preprint di 2 platform: Zenodo (P1) + arXiv (P8)

## P17 ArchCalc — COMPLIANCE AUDIT + REFORMAT

### Temuan Kritis dari Audit (website vs. paper asli berbeda!)

| Issue | Website bilang | Paper asli (35.1/36.1) | Yang kita fix |
|-------|---------------|----------------------|---------------|
| Paragraph numbering | "enumerate hierarchically" | **TIDAK ADA** nomor paragraf | Dihapus |
| Dashes | "em dash" | **En-dash dengan spasi ( – )** | `---` → ` -- ` |
| Figure captions | tidak detail | **"Fig. N –"** bukan "Figure N." | Diganti semua |
| Figure refs in-text | tidak detail | **"Fig."** bukan "Figure" | Diganti semua |

### File Submission (4 file, semua di `archcalc_submission/`)

1. `P17_manuscript_formatted.docx` — teks tanpa gambar/bibliografi, tabel di-rebuild manual
2. `P17_bibliography.docx` — 31 referensi, Harvard format, hanging indent
3. `P17_figures.zip` — 5 JPG 300dpi
4. `P17_figure_captions.docx` — format "Fig. N --"

### Compliance Audit: ALL PASS

- Em-dashes: 0 (converted)
- Figure refs: "Fig." throughout
- Footnotes: 0
- Anonymization: clean (no author name/VOLCARCH)
- Abstract: 198 words (max 200)
- Spelling: British English consistent
- Figures+Tables: 7 (max 10)
- Word count: ~5,340 (max 6,000)

### Fix Tambahan

- Spelling: 2× `civilization` → `civilisation` (standardisasi British)
- Abstract: trimmed 201→198 words
- Tabel 1 & 2: di-rebuild dari nol pakai python-docx (pandoc pecahkan booktabs)

### STATUS: Pak Amien perlu cek

- [ ] Buka `P17_manuscript_formatted.docx` di Word — **cek tabel sudah rapi?**
- [ ] Baca sekilas — final proofread
- [ ] Create account: https://submission.archcalc.cnr.it/
- [ ] Upload 4 file

---

## LAIN-LAIN

- **Email standardization**: VERIFIED clean. 0 old emails. Go-public UNBLOCKED.
- **Semua tracking docs updated**: JOURNAL, WORKSTATE, MEMORY, SUBMISSION_CHECKLIST, preprint_submission_guide

## YANG BELUM SELESAI (standing items)

| Item | Status | Siapa |
|------|--------|-------|
| P17 verify + upload ArchCalc | File ready, perlu cek | Pak Amien |
| P11 review → submit Archipel | v0.4 ready, cover letter ready | Pak Amien email |
| GitHub go public | UNBLOCKED (emails clean) | Pak Amien: Settings |
| JCAA APC waiver email | Draft ready | Pak Amien kirim |
| Go Frendi arXiv claim | Password ze47x | Pak Amien share |
| Zenodo deposit E171 | Metadata ready | Manual upload |

---

## SCRIPTS

- `archcalc_submission/format_for_archcalc.py` — heading styles, caption format, compliance audit
- `archcalc_submission/fix_tables.py` — rebuild tabel dari pandoc yang kacau

---

*"Satu paper di arxiv, satu lagi tinggal upload. Langkah kecil, tapi maju."*
