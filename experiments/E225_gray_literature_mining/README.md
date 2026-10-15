# E225: Gray-Literature Mining — Forgotten Pre-400 CE Records

**Status:** SUCCESS (pilot, phase 1) — verdict **LANJUT** under the pre-registered rule, confirmed by hand (2026-10-15).
The automatic record count is NOT usable as is (see the controller check). Phase 2 needs its own pre-registration; not started.
**Line:** 05_archival_nlp · **Paper:** erasure-archive channel ("taphonomy of records"; see orbit review 2026-10-15)
**Hypothesis (amended §7):** Indonesian-language archaeology journals hold numeric dates for pre-400 CE human activity, with
quotes and pages, at a rate of ≥ 1 radiometric record per 100 readable documents (or ≥ 20 verified records in the pilot).
**Method:** frozen OAI-PMH frame (5 BRIN journals, 1,676 records) → arm A (150 random, seed 20261015) + arm B (81 articles whose
title/abstract mention dating) → PDF → text (PyMuPDF) → page prefilter → DeepSeek JSON extraction → anti-hallucination guard
(quote must occur verbatim, age digits inside the quote) → tiers A/B/C → decision LANJUT / PARSIAL / GAGAL.
**Cost:** ≤ $10 (DeepSeek); usage per call in `results/llm_usage.csv`.

## Record of the run (2026-10-15)

| Step | Result | File |
|---|---|---|
| Pre-run amendment committed | `ea4124c` (before any download) | `DESIGN.md` §7 |
| Scripts + frame committed | `56d905d` (before any extraction); frame sha256 `acd50b68…` | `01`–`05_*.py`, `results/frame_oai_20261015.*` |
| Frame | 1,676 records: Berkala Arkeologi 828, Amerta 467, Naditira Widya 153, Kalpataru 125, Purbawidya 103 (deleted OAI records skipped) | `results/frame_oai_20261015.csv` |
| Arms | A 150; B 81 (all frame records outside A that match the dating regex — fewer than the cap of 150) | `results/pilot_arms.csv` |
| Downloads | 231 of 231 PDFs | `results/download_log.csv` |
| Text | 196 readable (≥ 1,000 characters); 35 without a text layer (scans; OCR not run in the pilot) | `results/text_log.csv` |

## Deviations from the design (technical, decided before any extraction)

1. **PyMuPDF instead of pypdf** for the text layer (pypdf is not installed; PyMuPDF is). No effect on the rules.
2. **Download fix:** Amerta and Kalpataru serve 35 bytes of HTML (`<div style="display:none;">`) before `%PDF-`; the first pass
   rejected those 60 files. The prefix is now dropped and logged ("ok (dropped 35-byte prefix)"). The arms did not change
   (seeded). ⚠ An injected hidden `<div>` on a journal server can be a sign of site tampering; the files are only read as text.

3. **One model output truncated:** `berkalaarkeologi_5355__0.json` (arm B, Gunungwingko) stopped at the 8,192-token output
   limit and is not valid JSON; its records are lost to the count (listed as `llm_files_unparsed`). Not re-run.

## Result (step 5, frozen scripts `56d905d`; `results/e225_summary.json`)

| Outcome | Arm A (random, 120 readable of 150) | Arm B (targeted, 76 readable of 81) |
|---|---|---|
| Records extracted | 208 | 670 |
| O4 rejected by the guard (quote not verbatim or age digits not in the quote) | 47 | 136 |
| Counted pre-400 CE (verified, archaeological, tier A or B) | 77 | 269 |
| of which tier A (radiometric) | 19 | 94 |
| O1 tier A per 100 readable documents | **15.83** (threshold 1) | 123.68 (not a rate; targeted) |
| Documents with ≥ 1 counted record (Wilson 95%) | 19 / 120 (10.4–23.4%) | 34 / 76 |
| O3 counted with a location (coordinates or district/regency) | 53 | 187 |

O2, distinct counted records (site × lab code × age × unit) across A+B: **314**. Verdict by the frozen rule: **LANJUT**.

## Controller check (DESIGN §7; `06_controller_check.py`, `results/controller_sample.csv`, `results/controller_judgments.csv`)

**The 10 random verified records (seed 20261015), against the rendered PDF pages.** All 10 quotes and numbers match the
PDF; the extraction does not invent text. Classification is where it fails: of the 6 that were counted, **3 are wrong**:
a Saka year ("1070 S (1148 M)") read as 1070 BCE; a Hijri year ("143 A.H. (760–61 A.D.)") read as 143 CE; and radiocarbon
ranges ending 440 and 560 CE counted as pre-400. Two more are correct but outside Indonesia (Watom Island, Papua New
Guinea; Upper Swan, Australia). One is a cited Indonesian date (Minanga Sipakko). One true pre-400 date was missed
(Gua Niah "44 kya": unit not in the schema).

Because of that, two more sets were checked by hand against the text (added after the first check; post hoc):

- **All 19 tier-A records counted in arm A** (the rate criterion rests on them): 8 are the authors' own dates from
  Indonesian sites, 4 are Indonesian dates cited from earlier work, 4 are outside Indonesia (Vanuatu, Papua New Guinea,
  Sarawak ×2), 2 are wrong (a laboratory pretreatment test with no site; the contested Gunung Padang "5,200 SM" reported
  as another team's claim, without error), and 1 is a duplicate. **12 valid Indonesian tier-A records in 4 of 120 readable
  documents = 10.0 per 100 documents** (documents: 3.3%, Wilson 95% 1.3–8.3%). The threshold of 1 per 100 is passed
  even if only documents are counted. **LANJUT holds.**
- **30 random counted records from both arms** (to see how much of O2 = 314 holds up): 17 / 30 are correct Indonesian
  pre-400 records (57%, Wilson 39–73%), and 2 of them are the authors' own measurements. 7 / 30 are outside Indonesia
  (Australia, Sarawak, Israel, Syria), 6 / 30 are wrong (three Saka years and one Hijri year read as BCE/CE, a 2000–1000 BP
  range scored by its old end, and a negative sentence), and 1 is a duplicate. **O2 = 314 is an upper bound, not a register
  size.** Most of the correct records are well-known dates cited from international publications.

**Stricter re-reading of tier A** (`results/e225_summary_strict.json`). 05 is looser than DESIGN §7 in two places: it
accepts a radiometric BC/AD date with no error, and it trusts a calibrated range the model may have computed itself. With
both closed: arm A 16 tier-A records, 13.33 per 100; A+B 308 distinct; verdict still LANJUT. Seven records drop out, all
for "radiometric without error or stated range".

## What the pilot found (descriptive, not a test)

Radiometric records from Indonesian sites whose young end falls in the window that matters for the core question (between
800 BCE and 600 CE). Attribution was checked by hand for each one listed here.

From the excavating team's own article:
- Gorua I (Tobelo, Halmahera, Maluku Utara; Ono et al., excavated 2012–13): TKa-15813/-15814/-15815, about 2190 BP,
  calibrated 360–160 BCE
- Kuta Bataguh (Kapuas, Kalimantan Tengah; Sunarningsih et al.): Handil Gardu 1840 ± 30 BP, calibrated 125–256 CE (other
  samples 320–411 and 356–524 CE)
- Gua Cililin 1 (Kotabaru, Kalimantan Selatan; arm A): an iron-smelting workshop at 1720–1560 cal BP. By arithmetic that
  is 230–390 CE; the authors print "90–230 Masehi". It is pre-400 either way.
- Loyang Ujung Karang (Aceh Tengah): 2590 ± 120 BP (bone)
- Leang Jarie (Maros; arm A): layer 1b, 2850–2750 cal BP

Cited in a review:
- Emok Tum (Oksibil, Papua): Beta-518106 2140 ± 30 BP and Lyon-17268 1850 ± 30 BP, from Vanhaeren et al. 2018

None comes from the volcanic interior of Central or East Java. This is consistent with the project's earlier picture, but
it is not evidence for any account of it.

Whether these dates already appear in the international compilations is exactly the question for phase 2. Nothing here
shows that they are new.

## Cost

189 calls; 937,220 prompt tokens (913,028 cache miss, 24,192 hit) and 160,738 completion tokens (`results/llm_usage.csv`).
The run did not log the served model. A one-token call afterwards showed that the alias `deepseek-chat` is served as
`deepseek-flash` (V4.1-Flash). At its published peak rates ($0.30 / M miss, $0.006 / M hit, $1.20 / M output) the run
cost about **$0.47**; even at V4-Pro peak rates it would be ≤ $1.84. Account balance after the run: $6.11. Cap: $10.

## Conclusion

The channel is real. Indonesian archaeology journals carry numeric, mostly radiometric, dates for pre-400 CE human activity
at about 10 valid tier-A records per 100 random articles. Some of them are the excavators' own dates, from Kalimantan,
Maluku Utara, Aceh and South Sulawesi.

The automatic pipeline, however, is not a register. The verbatim-quote guard stops invented text, but it does not stop:
misread calendar eras; ranges scored by their old end; dates from outside Indonesia; and the same date cited again across
articles.

**Inputs for a phase-2 pre-registration** (none of this is decided; the PI decides whether phase 2 runs):

1. Schema:
   - Saka and Hijri eras with fixed conversions;
   - start and end for every stated range, counting only the young end;
   - `kya`/`ka` units;
   - a country field and an Indonesia filter;
   - a flag for the authors' own date versus a cited one.
2. De-duplication by lab code, read by hand where printed codes disagree (Wk-7598 vs Wk-47598).
3. OCR for the 35 scanned documents (15% of the pilot).
4. Log the served model and set a larger output limit.
5. A hand-checked precision sample whose size is fixed in advance.
6. Coverage comparison against the *JIPA* cave review, the Cochrane et al. 2021 supplement and Bronson & Glover. That
   comparison is what decides whether any of this is "new".

**Line:** 05_archival_nlp (records channel); bridge to 01_spatial (dated points with locations) and 06_thesis (the ~400 CE
boundary question).
