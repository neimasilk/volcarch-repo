# Zenodo deposit — fields to paste (prepared 2026-10-07; the PI logs in, Claude fills the form, the PI presses Publish)

**Upload type:** Dataset
**Files:** `p8_forms_all.csv`, `p8_candidates.csv`, `p8_makasar_uncoded_meanings.csv`, `p8_tolaki_pmp_lookalikes.csv`, `p8_cell_examples.csv`, `README.md` (this folder; the README is the column description)
**Title:** Uncoded basic vocabulary in six Sulawesi word lists of the Austronesian Basic Vocabulary Database: form-level data release
**Authors:** Amien, Mukhlis (Universitas Bhinneka Nusantara; ORCID 0000-0002-1848-167X) · Gunawan, Go Frendi (Universitas Bhinneka Nusantara)
**Description (plain text):**
Form-level data for the article "Uncoded basic vocabulary in six Sulawesi word lists: What the absence of a cognate-set assignment in the Austronesian Basic Vocabulary Database does and does not show" (Oceanic Linguistics, manuscript OL-03-2026-11, revised version). One row for each of the 1,357 forms of six ABVD word lists (Muna 27, Bugis 48, Makasar 166, Wolio 192, Tae'/Sa'dan Toraja 226, Tolaki 674; lexibank/abvd repository state 917c5a5, 7 October 2025): ABVD identifiers, the form as recorded, ABVD cognate-set numbers and loan flag, the label (1 = no cognate-set number), out-of-sample scores of a 25-input classifier (form and meaning inputs), of a 26-input variant with a list-identity input, and of the model trained on the other five lists, the cell of the label-by-score table, and the nearest same-meaning look-alike form in other Sulawesi lists and, for Tolaki, in Bungku-Tolaki lists (a mechanical spelling screen, not a cognate judgement). A second file holds the 438 uncoded forms as a subset; three smaller files hold the 70 uncoded Makasar meanings with the Bugis and Sa'dan Toraja forms and the PMP entry, the twelve uncoded Tolaki forms that resemble the PMP entry by the screen, and the 92 example rows behind the article's Table 7. README.md describes every column, the classifier, the look-alike screen and the lists compared. Forms, cognate-set numbers, loan flags and names are copied from ABVD (Greenhill, Blust & Gray 2008; CC BY 4.0); please credit ABVD and the list compilers when reusing. No form has been assessed etymologically by a specialist.
**Keywords:** Austronesian Basic Vocabulary Database; Sulawesi languages; cognate coding; basic vocabulary; Makasar; Tolaki; Bugis; Muna; Wolio; Sa'dan Toraja
**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
**Version:** 1.0
**Language:** English (metadata); forms in six Sulawesi languages
**Related identifiers:** is supplement to — the journal article (add the article DOI when known); is derived from — ABVD, https://abvd.eva.mpg.de and https://github.com/lexibank/abvd (commit 917c5a5); code — https://github.com/neimasilk/volcarch-repo (folders experiments/E227–E231)
**Funding:** none
**PUBLISHED 2026-10-07 by the PI (Claude filled the form in the Playwright window): record https://zenodo.org/records/23202245 — DOI 10.5281/zenodo.23202245, concept DOI 10.5281/zenodo.23202244.**
**After publishing (done):** copy the DOI into `P8_revision_v0.2.md` (data statement: replace `[DOI]`), rebuild the Word files, and record the DOI in the release README section 0, `docs/WORKSTATE.md` and the line STATE.

---

## 2026-10-08, afternoon — the description file was corrected INSIDE the published record (same DOI); no version 1.1

**Why.** The `README.md` in the record published on 2026-10-07 was the working draft (downloaded from the public record and
read on 2026-10-08: 20,773 bytes, md5 `3783944aac3876486d11026b010853f7`): an internal status note at the top ("completed for
the PI's reading before deposit … Still the PI's: read the whole file; fill the two placeholders …"), the heading "Description
of the two data files", and unfilled fields — `[title of the revised article]`, "to be confirmed by both authors",
`[date of deposit]`, `[year]`, `[title above]`, `https://doi.org/[DOI]`, `[repository URL]` (twice), "Makasar (or Makassar — the
authors' spelling decision)". The five CSV files were right. The local copy's line "this copy is the one in the deposit except
for this status line" was not true of three more lines.

**What was done** (PI logged in and said that publishing on Zenodo was fine; Claude drove the browser, about 13:45 WIB).
Zenodo offers two ways to change files: a new version, or — "to correct minor errors" — unlocking the files of a record
within 30 days of publication. This is a correction of the description only, so the second way was used: *Edit* → *Edit files*
→ *Edit published files*; checklist answered "No" to "I want to update the files with a new version" and the box "I will not
modify files that supplement findings/results of an already published work" ticked (true: only the description changes, and
the article is not published); `README.md` deleted and this folder's `README.md` uploaded; in the description the first
sentence now names the article by its current title, and a last sentence was added ("The description file (README.md) was
corrected on 2026-10-08; the data files are unchanged."); *Publish*.

**Result, checked without login through the public API** (`zenodo.org/api/records/23202245`, 13:46 WIB): DOI
**10.5281/zenodo.23202245** unchanged (concept DOI 10.5281/zenodo.23202244), version 1.0, publication date 2026-10-07;
`README.md` 20,356 bytes, md5 `3952faf28995e9222ff71895af01edb8`, **byte-identical to this folder's file**, no placeholder
left; the five CSV files carry the same md5 as before; description updated. The draft README is no longer downloadable from
the record (a copy of it is in the git history of this folder: the version of commit `92c29bd` plus the status line).

**Which DOI is cited.** The version DOI 10.5281/zenodo.23202245, as in the manuscript the PI approved on 2026-10-07 (the
named build and the cover letter; the anonymised article file redacts it). A plan to cite the concept DOI, made while a
version 1.1 was expected, was dropped the same afternoon.

**Before the correction the 21 local commits were pushed** (GitHub head `010b1f5`), so the folders `experiments/E227` … `E231`
that the README and the record's description point to are public.
