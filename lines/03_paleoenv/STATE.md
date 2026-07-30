# STATE — Line 03 PALEOENV

**Updated:** 2026-07-30 · **Temperature:** 🧊 BLOCKED on a human, but has real parallel work available

---

## The blocker

**No palynologist / paleobotanist co-author.** This gates SIG G2 + G10, and therefore gates the full
E216 manuscript, the cross-model review (G9), the Zenodo deposit (G7), and submission — in that order,
per `experiments/E216_paleoecological_interferometer/SUBMISSION_CHECKLIST.md`.

**Candidates:** Castillo (a draft email already exists for E215 — one email can cover both channels) ·
ITB or UGM geology/geography contacts. **This is a PI action.**

---

## Next actions for Claude (available now, parallel to the co-author search)

- [ ] **WS-A / A5 — write the full prose manuscript** for *Vegetation History and Archaeobotany* from
      `results/PAPER_DRAFT_OUTLINE.md`. The outline is already honest and caveat-first, so this is
      drafting, not re-deciding. **This does not need to wait for the co-author** — a co-author
      reviews prose; they should not be asked to wait for it to be written.
- [ ] **K4-style check on E216:** is there any parameter under which `n_cores_resolving_heartland` > 0?
      The sweep says no across 27 points; state the boundary of that grid explicitly in the manuscript
      so a reviewer cannot ask "but what if RPP were larger?"
- [ ] Reproducibility check on `code/e216_sensitivity_sweep.py` from a clean run (mini-G1 blind
      recompute — instructions are already in `zenodo_upload/`).

## Do NOT do

- ❌ Cross-model review (G9), Zenodo upload (G7), or any submission step — all sit **after** the
  co-author in the checklist.
- ❌ Start [06_thesis](../06_thesis/) WS-B (the Masterpiece reframe around detection-power) before this
  line's manuscript exists. The Fable plan is explicit: *one flagship finished beats five
  half-done*, and WS-A must close before WS-B opens.
- ❌ Reframe E214 into support for the thesis. If an analysis trends that way, escalate it to orbit
  mode rather than absorbing it.

## Blocked / external

| Item | Blocker | Owner |
|---|---|---|
| Palynologist co-author | outreach not sent | **PI** |
| Dieng calibration re-derivation | raw data paywalled | external — document as a limitation, do not work around it |
| E215 Castillo email | draft exists, unsent | **PI** |

## Inbox

- E216's detection-power framing is the strongest available spine for
  [06_thesis](../06_thesis/) P0 (WS-B). Recorded there; not to be acted on from here.
- `docs/research_notes/PHYTOLITH_VOLCANIC_PRESERVATION.md` and `LIANGAN_VALIDATION_CASE.md` belong to
  this line and are not yet cited by E215.
