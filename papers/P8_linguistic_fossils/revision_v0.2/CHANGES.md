# revision_v0.2 — what was changed against `draft_v0.1_anonymous.tex`

**2026-10-05.** Only substitutions that reviewer 1 dictated word for word, plus two path fixes because this copy sits one folder deeper.
No number, no claim, no sentence structure was touched. The file is **not** a corrected manuscript: see `../REVISION_WORKPLAN.md` §5–§6 for what is still wrong in it.

| Reviewer point | Old | New | Occurrences |
|---|---|---|---|
| R1-1 (abstract, tex 38) | `forms that resist reconstruction to any proto-form and show` | `forms that resist reconstruction to higher-level proto-forms and show` | 1 |
| R1-1 (tex 56) | `basic vocabulary resists reconstruction to any proto-form:` | `basic vocabulary resists reconstruction to higher-level proto-forms:` | 1 |
| R1-4 (tex 65) | `including Celebic, South Sulawesi, and Muna--Buton,` | `including South Sulawesi, Bungku--Tolaki, and Muna--Buton,` | 1 |
| R1-5 (tex 73, 598) | `consistent with parallel independent innovations rather than` | `consistent with independent innovations rather than` | 2 |
| R1-5 (heading, tex 509) | `\subsection{Parallel innovation, not shared substrate}` | `\subsection{Independent innovation, not shared substrate}` | 1 |
| R1-5 (tex 517) | `appear to represent parallel independent innovations:` | `appear to represent independent innovations:` | 1 |
| R1-5 (AI disclosure, tex 614) | `the parallel-innovation alternative` | `the independent-innovation alternative` | 1 |
| R1-6 (tex 87) | `These languages represent three subgroups---Muna--Buton (Muna, Wolio), South Sulawesi (Bugis, Makassar), Celebic (Toraja-Sa'dan), and Southeast Sulawesi (Tolaki)---` | `These languages represent four primary subgroups---Muna--Buton (Muna), Wotu--Wolio (Wolio), South Sulawesi (Bugis, Makassar, Sa'dan Toraja), and Bungku--Tolaki (Tolaki)---` | 1 |
| R1-6 name order (tex 86, 237, 298, 405) | `Toraja-Sa'dan` | `Sa'dan Toraja` | 4 |
| R1-6 no abbreviation (tex 486) | `Bol.\ Mongondow` | `Bolaang Mongondow` | 1 |
| R1-6 no abbreviation (Table 5, tex 468) | `Bol.Mong.` | `Bolaang Mongondow` | 1 |
| path only (figures) | `../../experiments/` | `../../../experiments/` | 4 |
| path only (bibliography) | `\bibliography{references}` | `\bibliography{../references}` | 1 |

**Deliberately not changed** (decisions or prose of the PI):
- *Makassar* → *Makasar* (reviewer's preference; ABVD writes *Makassar*) — every occurrence.
- `any proto-language` at tex 60 and 540: a different sentence (the traditional method); the reviewer named two places.
- R1-3 (number of primary subgroups, tex 57), R1-7 ("under-documentation", tex 114), R1-8 (digraph list, tex 348), R1-10 and R1-12 (terms), every R2 item: need new wording or new numbers.
- Labels inside the four figures (`Toraja-Sadan`, `Bol.Mongondow`, internal codes): the figures have to be regenerated.
- Table 5 overflows further with the full name; it is to be rebuilt or dropped (decision D5).

## 2026-10-06 - bibliography corrections (working copy `references.bib` in this folder)

Path-only change in `p8_revision_v0.2.tex`: `\bibliography{../references}` -> `\bibliography{references}`, so that the tex file reads the corrected copy in this folder. The original `../references.bib` is untouched. Nothing else in the tex file changed.

Corrections follow `REFERENCE_CHECK_20261005.md` Part B (existence and bibliographic data only). Citation keys are unchanged. Entries not listed here are unchanged, except that a comment line (starting with %) was added above lundberg2017.

| key | what changed | evidence |
|---|---|---|
| ross2005 | `@article` in *Oceanic Linguistics* 44(2): 343-380 -> `@incollection` in *Papuan Pasts* (Pacific Linguistics 572), editors Pawley, Attenborough, Golson, Hide, pp. 15-65, Canberra | REFERENCE_CHECK_20261005.md Part B |
| bellwood1995 | `@article` with the book title as `journal` -> `@incollection` (editors Bellwood, Fox, Tryon; booktitle *The Austronesians*; 1995 print imprint, Canberra); pages 96-111 kept but UNVERIFIED; comment above the entry gives the 2006 ANU E Press alternative (pp. 103-118, DOI 10.22459/A.09.2006.05); PI to choose one edition | REFERENCE_CHECK_20261005.md Part B |
| swadesh1955 | `@inproceedings` with journal as `booktitle` -> `@article`, journal IJAL; DOI 10.1086/464321 added | REFERENCE_CHECK_20261005.md Part B |
| thurgood1999 | `@article` with `journal` -> `@book` with `series` = Oceanic Linguistics Special Publication, number 28; address Honolulu added | REFERENCE_CHECK_20261005.md Part B |
| casparis1975 | publisher "E. J. Brill / Lembaga Ilmu Pengetahuan Indonesia", place "Leiden / Jakarta" -> publisher Brill, place Leiden | REFERENCE_CHECK_20261005.md Part B |
| mcelhanon1970 | `@article` with `journal` -> `@book` with `series` = Pacific Linguistics, Series B, number 16; publisher Research School of Pacific and Asian Studies, The Australian National University | REFERENCE_CHECK_20261005.md Part B |
| list2012 | `@article` with proceedings title as `journal` -> `@inproceedings` with `booktitle`; publisher Association for Computational Linguistics, Avignon, France added | REFERENCE_CHECK_20261005.md Part B |
| anderson2018 | `@inproceedings` with `booktitle` -> `@article` with `journal` = Yearbook of the Poznan Linguistic Meeting | REFERENCE_CHECK_20261005.md Part B |
| mead2005 | entry deleted (NOT FOUND; no such chapter in *Papuan Pasts*; not cited in the draft) | REFERENCE_CHECK_20261005.md Part B |
| vandenBerg1996 | entry deleted (NOT FOUND; no such chapter in the Atlas; not cited in the draft) | REFERENCE_CHECK_20261005.md Part B |
| lundberg2017 | entry unchanged; comment added: page range 4765-4774 UNVERIFIED | REFERENCE_CHECK_20261005.md Part B |

Open for the PI: list2018 is bibliographically correct but is cited in the text for a use it does not fit (CLICS2 is a colexification database, not a phylogenetic tool); wording is the PI's. Bellwood 1995 edition choice (see above).
