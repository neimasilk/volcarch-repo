# E233: P23 validation against the editors' published translations

**Status:** PRE-REGISTERED (DESIGN.md, 2026-10-19) — readers not yet run
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

## Result

Not run yet.
