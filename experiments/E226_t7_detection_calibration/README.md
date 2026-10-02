# E226 — T7 detection calibration (villages named in 8th–10th c. inscriptions vs recorded settlements)

**Status:** IN PROGRESS (started 2026-10-02). **Lines:** 05_archival_nlp (primary), 06_thesis.
**Pre-registration:** `DESIGN.md` (frozen 2026-10-02 before any register or matching; amendments in §8).
**Matching is BLOCKED** until the PI fills N_ref and m in `DESIGN.md` §5.

## Hypothesis

If even villages known to have existed (named in 8th–10th c. *sīma* charters) are rarely matched by a
recorded settlement deposit of that period, then the absence of recorded pre-400 CE settlements in interior
volcanic Central Java is uninformative about population (against reading it as H6, low density). If they
are matched often (S/|G| − d₀ ≥ 0.25, |G| ≥ 30), the "villages are invisible" argument fails.

## Method (steps; see DESIGN.md)

| Step | What | Script / file | State |
|---|---|---|---|
| 1a | Village-noun occurrences with KWIC, 701–1000 CE, edition text | `scripts/01_build_candidates.py` → `frame/candidates_kwic.csv`, `frame/inscriptions_701_1000.csv` | ✅ 595 occurrences in 73 of 111 inscriptions |
| 1b | Two independent coders + adjudication → frame V_r | `frame/CODEBOOK.md`, `frame/coder_A/B.csv`, `scripts/02_reconcile.py` → `results/t7_village_frame.csv` | running |
| 2a | Modern desa gazetteer (Kepmendagri 2025) | `scripts/03_gazetteer.py` → `data/processed/gazetteer/desa_kedu_mataram_buffer_2025.csv` | ✅ 3,802 desa |
| 2b | Identification tiers I1/I2/I3 | — | pending |
| 3 | Settlement register, built blind to the frame | `register/PROMPT.md` → `register/t7_settlement_register_candidates.csv` | running |
| 4 | Matching, d, d₀, N₉₅, decision | — | **blocked on PI (N_ref, m)** |

## Data used

- DHARMA TEI editions, local snapshot `experiments/E023_ritual_screening/data/dharma/xml/` (268 files) and the
  IDENK metadata xlsx (dating, findspot kabupaten).
- E082 canonical30 geocode labels (region fallback only; known placeholders, C032).
- Kepmendagri 300.2.2-2430/2025 desa list via cahyadsn/wilayah (`data/sources.md`).

## Notes so far

- **Desk-scan count corrected:** the 2026-10-01 scan counted 440 village-noun tokens; it missed the *b*-spelling
  (*banuA*, *banva*), which adds 155 → **595 occurrences** (2 are *baṅun* "build", left for the coders to reject).
  Inscriptions with ≥1 village noun: 69 → 73. (Amendment A1.)
- Region is the inscription's findspot region, not the villages' location (threat 8).

## Result / Conclusion

Pending.
