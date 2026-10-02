# P23 human validation protocol (fixed 2026-10-02, before any human code exists)

**Why:** the frame was coded by two LLM agents (κ 0.90 between them) and adjudicated by an LLM. Agreement between
two models says nothing about whether they are right. P23's central quality claim rests on this check.

- **Sample:** 100 of 595 occurrences, simple random, `random.Random(23)` → `sample_ids.csv`; sheet
  `lembar_validasi_100_kemunculan.xlsx` shows context only (blind to AI codes and the attached list).
- **Coder:** one epigrapher (Dr Titi Surti Nastiti if she agrees; else another epigrapher). A second human, if
  available, codes the same 100 for human–human κ.
- **Reference for comparison:** `experiments/E226_t7_detection_calibration/frame/final_occurrences.csv` (post-adjudication).
- **Metrics (reported whatever they are):**
  1. Cohen's κ and % agreement, human vs final, on Y vs not-Y (and 3-category).
  2. Name accuracy on rows both code Y: exact match after the spelling key; boundary errors (too long/short)
     counted separately.
  3. Error taxonomy: title read as village, person read as village, generic read as village, missed village,
     boundary.
  4. 95% CI (Wilson) for the share of frame names that are wrong, extrapolated with stated assumptions.
- **Decision rule for the paper:** if human–final κ(Y) < 0.70 or > 15% of Y names are wrong, the list is not
  published as a resource until the whole frame is re-coded by a human; the paper then reports the LLM-coding
  failure as its result (still publishable as a method lesson).
