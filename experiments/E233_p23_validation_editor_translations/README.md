# E233: P23 validation against the editors' published translations

**Status:** SUCCESS — PASS by the pre-registered rule (κ 0.84; strict name error 5.4 %, 95 % CI 2.8–9.9 %), 2026-10-19
**Line:** 05_archival_nlp (primary), 04_language_text · **Paper:** P23 (Naskah A) · **Source:** E226 frame step

**Hypothesis:** the E226 village frame (two LLM coders + LLM adjudication) reads the village nouns of 8th–10th c.
charters the way the DHARMA editors read them in their published translations: κ(Y vs not-Y) ≥ 0.70 and at most 15 %
of the frame's village names wrong (thresholds from `papers/P23_sima_village_toponyms/validation/PROTOCOL.md`).

**Why:** P23's validation waited for an epigrapher from 2026-10-02 to 2026-10-19. PI 2026-10-19: a human expert is
never a blocker; pivot. The human judgement now comes from readings experts already published: the translations by
Griffiths, Kusumo Dewanti, Sastrawan, Bastiawan, Gomes, Nastiti (Masahar) and Stutterheim 1940 (Kurungan) of the same
editions the coders read.

**Method:** `DESIGN.md` (frozen before any comparison).
- Scope: the 30 frame inscriptions with a DHARMA translation, 216 of 595 occurrences.
- Two blind LLM readers transcribe how the translator rendered each occurrence (Y/N/U + name + verbatim quote).
- A verbatim guard checks every quote and name; the orchestrator adjudicates the readers' disagreements with quotes.
- Then κ, name accuracy, error taxonomy, strict error share with a Wilson interval, and a representativeness check.

**Data used:** DHARMA TEI snapshot `experiments/E023_ritual_screening/data/dharma/xml/` (CC BY 4.0, DHARMA project);
E226 `frame/coder_input.csv` and `frame/final_occurrences.csv`.

## Files

| File | What |
|---|---|
| `DESIGN.md` | pre-registration |
| `scripts/01_packets.py` | builds `packets/` (translation + occurrences, no frame codes; asserts E226 contexts reproduced) |
| `packets/` | 30 reader packets, `*.translation.txt` (guard text), `index.csv` |
| `reader/PROMPT.md` | reader instructions, verbatim |
| `reader/R1_A.csv` … `R2_B.csv` | the two readers' transcriptions (Sonnet agents, blind) |
| `reader/adjudication_todo.csv`, `reader/adjudication.csv` | the 2 reader disagreements and their resolution |
| `scripts/02_reference.py` | verbatim guard, reader agreement → `results/reference.csv`, `results/reader_agreement.json` |
| `scripts/03_compare.py` | frame vs reference → `results/disagreements_todo.csv`, `results/e233_metrics.json` |
| `results/error_taxonomy.csv` | orchestrator's reading of all 48 disagreements, one reason per row |

## Result (2026-10-19)

**PASS.** On the occurrences the DHARMA editors translated, the machine reading of E226 agrees with the editors'
reading.

**Reference quality.**
- The two blind readers agreed on 214 of 216 occurrences (κ 0.977, Y/N/U).
- All 161 names that both read as Y matched.
- The 2 differences were R1 writing three names in one cell (guard failure). Both were resolved from the quote
  (`reader/adjudication.csv`).
- Reference: Y 163, N 36, U 17. The U rows are passages not translated: Kayumwungan (verses only), Kiringan side 1v,
  lacunae.

| Metric (DESIGN §4) | Value |
|---|---|
| **M1** κ, frame Y vs not-Y against the reference (n = 199; 17 U excluded) | **0.839** (agreement 95.5 %) |
| — frame Y / ref Y · frame Y / ref N · frame not-Y / ref Y · both not-Y | 161 · 7 · 2 · 29 |
| M1 on the random-100 rows in scope (n = 37) | κ 1.00 |
| **M2** names, both Y (161) | 122 exact key · 3 boundary (rule) · 36 other |
| **M3** taxonomy of the 48 disagreements | 37 spelling · 7 generic read as village · 2 boundary · 2 missed village |
| **M4** strict share of frame-Y names wrong (n = 168) | **9 = 5.4 %**, Wilson 95 % 2.8–9.9 % |
| M4 lenient (boundary not counted) | 7 = 4.2 % (2.0–8.4 %) |
| Representativeness: coder A–B agreement inside vs outside the 30 inscriptions | 94.9 % vs 97.6 %, Fisher p = 0.097 → extrapolable by the rule |

**What the errors are.**
- **Spelling (37, not errors).** The frame keeps the edition's spelling, as the codebook asks. The translators normalise
  it: anusvāra *ṁ* → *ṅ* (24 rows), gemination, *v/b*, and Dutch *oe/ng/w* in Stutterheim's Kurungan. The E226 key keeps
  *m* and *n* apart, so the rule could not see these.
- **Generic noun read as a named village (7).**
  - Dalinan#003 and Sangsang#003 are the same formula: *kinon sumusuka ikanaṁ vanua: vuara kuṭi i X* ("ordered to
    demarcate the village(s): there was a kuṭī at X"). The frame attached the kuṭī's place to the noun.
  - Humanding#001: "the neighbouring villages: from Guntur …".
  - TajiGunung#010: "the village upstream of the monastery of Taṅkilan".
  - TajiGunung#018, #019, #027: *hulu vanua*, which this translator renders "the upstream village". This reading is
    contested: other DHARMA translators write *Hulu Vanua* as a name (WanuaTengahIII, Sindoro). It is still counted
    against the frame.
  - In 4 of the 7 (Dalinan, Hujuṅ Galuh, Guntur, Taṅkilan), the attached name is a real place in the translation,
    attached to the wrong noun. The other 3 are the contested *hulu vanua*.
- **Boundary (2).** *tasikk anak* (should be Tasik) and *kinaliṁ* (the translator reads *Kinaliṅ Kabini-hajyan*).
- **Missed (2).** Lintakan#012: the frame's '?' was the right name. WanuaTengahIII#004: *saṁ vanua tṅaḥ* was read as a
  title; the translator reads "from Vanua Tṅah".

**Sensitivity (computed after the result; the decision does not change).**
- Two rule-flagged "boundary" rows were reclassified as spelling by the orchestrator: Kiringan#005 *kavuṅ-kavuṁ*, and
  Lintakan#014 *sumuḍaḥ* (the translator drops the visarga). Counting them as wrong, as the mechanical rule would: 11 =
  6.5 % (3.7–11.3 %).
- Also counting the sandhi form *vuatan nəmmas* (translator: *Vuatan Mas*) as wrong: 12 = 7.1 % (4.1–12.1 %).
- **Worst case**, every frame-Y row whose passage the editors did not translate (13) counted wrong: 22 / 181 = 12.2 %
  (8.2–17.7 %). The point estimate is still under 15 %.
- The contested *hulu vanua* counted right instead: 6 = 3.6 %.

**What this means for P23.**
- The two-LLM frame is checked against human expert readings, not only against itself.
- Its errors are formula-level and fixable: the forward-pointing *kinon sumusuka ikanaṁ vanua:* clause, generic
  plurals before a list, and title vs place in *saṁ vanua tṅaḥ*.
- The dataset needs a **normalised-name column** (editors' conventions, e.g. *ṁ* → *ṅ*) beside the edition form, so
  that names can be compared and matched.
- **Extrapolation** to the 379 untranslated occurrences rests on one assumption: that they are no harder than this
  subset. The proxy supports it. The translated subset had *more* coder disagreement (5.1 % vs 2.4 %), though not
  significantly.

**Limits, stated as in DESIGN §6.**
- Not independent of the edition: the translators edited the text the coders read.
- The coder models may have seen these public translations in training. This is not testable here.
- The subset is not random.
- Every AI step is named:
  - LLM coders and an LLM adjudicator (E226);
  - LLM readers (Sonnet, two, blind);
  - the orchestrator (Claude) for the 2 reader adjudications and the 48 taxonomy rows, each with its reason.
- The human judgement is the translators': Griffiths, Kusumo Dewanti, Sastrawan, Bastiawan, Gomes, Nastiti, Stutterheim.

**Deviations.**
- One reader (R2, group B) ran `git status`, which listed the other reader's file name (R1_B.csv). It reports that it
  did not open the file. Its two differences from R1 in group B are the two adjudicated rows.
- Two readers wrote helper scripts in the session scratchpad, outside the repo. Neither changes any input.
- `02_reference.py` and `03_compare.py` were not in the pre-registration commit. They were written while the readers
  were running, before any reader output existed, and follow DESIGN §3–5. Only the ṁ/ṅ blind spot of the name key
  showed up afterwards; it was handled by the pre-registered (c) reading, not by changing the key.
