# E233 — P23 validation against the editors' published translations: pre-registration

**Frozen:** 2026-10-19, committed before any reference reading exists and before any comparison with the frame.
**Lines:** 05_archival_nlp (primary), 04_language_text. **Paper:** P23 (Naskah A). **Source:** E226 frame step.
**Why this experiment:** P23's validation (`papers/P23_sima_village_toponyms/validation/PROTOCOL.md`) waited for an
epigrapher to code 100 occurrences. None has answered since 2026-10-02. The PI ruled on 2026-10-19 that a human
expert is never a blocker ("AI penuh = fitur, bukan bug … kalau perlu ahli, pivot"). This is the pivot: the human
judgement comes from **readings experts have already published** — the translations the DHARMA editors made of the
same editions the LLM coders read — instead of a newly recruited coder.

**What has been looked at before freezing (so nothing below was tuned to it):** which of the 73 frame inscriptions
carry a translation (30, with 216 of the 595 occurrences; 40 of those 216 are in the random-100 sample of
`PROTOCOL.md`); the translators named in the TEI headers; the paragraph line references; the frame's Y count per
inscription. **No translation has been compared with any frame code or name.**

## 1. Question

Do the frame codes (`experiments/E226_t7_detection_calibration/frame/final_occurrences.csv`, two LLM coders +
LLM adjudication) read the village nouns the way the editors of these inscriptions read them in their published
translations? Two parts, as in `PROTOCOL.md`: (a) is a specific village named at this occurrence (Y vs not-Y);
(b) if both say Y, is it the same name with the same boundaries.

## 2. Reference

- **Source:** `<div type="translation">` of the same local DHARMA snapshot the frame was built from
  (`experiments/E023_ritual_screening/data/dharma/xml/`, March 2026). The live repository (commit `19d0d3f`,
  2026-09-25) was checked: it adds no translation for these inscriptions, so the snapshot is used.
- **Scope:** the 30 frame inscriptions whose longest translation div has > 300 characters → 216 occurrences.
  Translators per TEI header: Arlo Griffiths (most), with Tyassanti Kusumo Dewanti, Wayan Jarrah Sastrawan, Eko
  Bastiawan, Mekhola Gomes, Titi Surti Nastiti (Masahar); Kurungan is Stutterheim 1940 (Dutch).
- **Not in scope:** the other 43 inscriptions (379 occurrences) have no translation in DHARMA. Stage 2 (optional,
  §7) may add other published translations; it is not part of this decision.

## 3. Reading the reference (the only AI step on the reference side)

The AI does not judge Old Javanese here. It reports **what the translator wrote**.
- **Packets** (`scripts/01_packets.py` → `packets/`): per inscription, the translation (paragraphs with their line
  references), and the inscription's occurrences in text order with occurrence id, edition line (`lb`) and the same
  ±12-token context the coders saw. **No frame code, name or note is in a packet.** The script asserts that its
  tokenisation reproduces `frame/coder_input.csv` exactly (same occurrence count and context per inscription).
- **Two readers**, independent LLM agents (Sonnet), blind to each other and to the frame. Instructions in
  `reader/PROMPT.md` (written before the run, kept verbatim). Per occurrence:
  - `ref_code` **Y** — the translation renders a specific named place at this point (e.g. "native of X",
    "the village of X", "the elders of X", "the *sīma* of X"); **N** — rendered without a new place name
    (villagers, all the villages, the village elder + a person, "this village" referring back, a count, land);
    **U** — the passage is not translated, is in a lacuna or summarised, or cannot be located.
  - `ref_name` — the place name exactly as spelled in the translation (Y only).
  - `quote` — the verbatim span of the translation that shows the rendering (≤ 250 characters).
- **Guard** (`scripts/02_reference.py`): `quote` must occur in the translation (whitespace-normalised); `ref_name`
  must occur in `quote` (case-insensitive). A row failing the guard becomes U (reason logged).
- **Reconciling the readers:** where R1 and R2 differ in code, or in name key, the orchestrator (Claude) decides
  from the translation text and writes the reason and the quote (`reader/adjudication.csv`). Reader agreement
  (κ on Y/N/U, name agreement) is reported. Final → `results/reference.csv`.

## 4. Comparison (`scripts/03_compare.py`) — metrics reported whatever they are

- **Name key:** E226's `key()` (lowercase, diacritics stripped, w→v, non-alphanumerics removed), plus for the
  Dutch translation only oe→u, dj→j, tj→c.
- **M1** Cohen's κ and % agreement, frame Y vs not-Y (N or ?) against reference Y vs N, over occurrences with
  reference ≠ U. Number of U reported. Also on the 40 random-100 rows in scope.
- **M2** Name accuracy on rows both Y: (a) exact key match; (b) **boundary** — one key is a prefix or suffix of the
  other, or the token sets nest; (c) other mismatch. (c) rows are read by the orchestrator and split into
  **spelling** (the translator normalised a spelling of the same word, shown by the edition context) and **wrong**
  (a different word). Each (c) decision is written with the edition context and the quote.
- **M3** Error taxonomy over all disagreements (codes as in `PROTOCOL.md`): title read as village; person read as
  village; generic read as village; missed village; boundary; wrong name; and **reference doubtful** (the
  translation paraphrases or skips in a way that does not settle the reading). Written per row.
- **M4** Share of frame-Y occurrences that are **wrong** = reference N + boundary + wrong name (strict; spelling
  counts as right), with a Wilson 95 % interval. Lenient variant (boundary not counted) reported beside it.
- **Representativeness check:** coder A–B raw agreement (Y vs not-Y) inside vs outside the 30 translated
  inscriptions, Fisher exact test. If agreement differs at p < 0.05, the result is reported as **not**
  extrapolable to the 379 untranslated occurrences.

## 5. Decision rule (thresholds unchanged from `PROTOCOL.md`)

- **FAIL** if κ(Y) < 0.70 **or** strict M4 > 15 %: the list is not published as a resource until the frame is
  re-coded against references (or by a human); P23 then reports the LLM-coding failure as its result.
- **PASS** otherwise, stated as: "on the 216 occurrences the editors translated, the machine reading agrees with
  the editors' reading at κ = …; strict error … % (95 % CI …)". Extrapolation to the untranslated part only if
  the representativeness check passes, and always with the assumption written out.
- Either way the numbers go into P23's appendix with the AI steps named (LLM coders, LLM readers, LLM adjudicator;
  the human judgement is the translators').

## 6. Threats (stated now)

1. **Not independent of the edition.** The translators edited the same text the coders read. This measures
   whether the machine reads the edition as its editors do — which is the claim P23 needs — not a fresh reading.
2. **Training contamination.** DHARMA is public on GitHub; the coder models may have seen these translations.
   That would inflate agreement on exactly this subset. Not testable here; stated in the paper.
3. **Selection.** Translated inscriptions are not a random sample (the representativeness check addresses one
   proxy, difficulty as coder disagreement).
4. **Translation ≠ coding.** Translators sometimes paraphrase lists or omit repeated "native of" phrases; such
   rows become U or "reference doubtful", never counted against the frame.
5. **The readers are LLMs.** Mitigated by two blind readers, the verbatim guard, and adjudication with quotes —
   every reference row can be checked by anyone against the published translation.

## 7. Optional stage 2 (not part of the §5 decision)

Other published translations of the 43 untranslated inscriptions (e.g. Sarkar 1971–72; Indonesian editions), if
openly accessible, read the same way and reported separately. A human epigrapher's coding of the random-100
sheet, if one ever arrives, is compared with both the frame and this reference (human–reference agreement).
