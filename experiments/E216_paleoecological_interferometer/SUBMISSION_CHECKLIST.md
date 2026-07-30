# E216 — Submission Checklist

**Purpose:** separate what Sonnet/Claude can execute autonomously from what requires
Pak Amien's action or judgment, so nothing gets silently skipped or silently attempted
without authorization. Per `docs/research_notes/FABLE_STRATEGIC_PLAN_20260707.md` WS-A.

---

## ✅ DONE (Sonnet-executable, completed 2026-07-07)

- [x] A1 — Fixed Defect 1 (coverage ≠ resolution) in `code/e216_detection_function.py`
- [x] A2 — Fixed Defect 2 (deterministic P(detect) → parameter sensitivity sweep) via
      new `code/e216_sensitivity_sweep.py`
- [x] A3 — Fixed Defect 3 (positive control overstated "CONFIRMED" → labeled
      qualitative import; NO-GO branch disclosed)
- [x] A4 — Fixed Defect 4 (missing-core headline hid a failing corner) → full corner
      table + caveat now in `results/missing_core_spec.json` and the abstract
- [x] Regenerated all output files (`OUTCOME.json`, `missing_core_spec.json`,
      `missing_core_corner_table.csv`, `sensitivity_*.csv/json`) from corrected code
- [x] Rewrote `results/PAPER_DRAFT_OUTLINE.md` to reflect honest numbers throughout
- [x] Updated `README.md` status header with fix log

## 🔒 HUMAN-GATED (Pak Amien only — do NOT attempt to execute these autonomously)

### 1. Palynologist / paleobotanist co-author (SIG G2, G10 — REQUIRED, non-negotiable)
This is not optional polish. SIG G10 requires independent domain review before any
paper reaches flagship status, and G2 (domain-sanity) needs someone who can judge
whether the RPP ranges, RSAP radii, and detection-threshold assumptions used here are
defensible palynological practice — Claude cannot self-certify this.
- Candidate contacts: Castillo (already drafted for E215, `docs/drafts/email_castillo_phytolith.md`
  — could plausibly cover both channels), or an Indonesian Quaternary palynologist via
  ITB/UGM geology departments.
- **Action:** Pak Amien identifies and contacts a candidate. Nothing else in this
  checklist should proceed to actual submission before this is secured.

### 2. Cross-model review (SIG G9)
Run a skeptical cross-model review (DeepSeek/Gemini/whichever is available) on the
**updated** `PAPER_DRAFT_OUTLINE.md` — the fixed version, not the pre-2026-07-07 one
Opus already reviewed. Opus reviewed the *execution*; no model has yet reviewed the
*fixes*. This should happen after co-author review, not instead of it.
- **Action:** Pak Amien (or Claude, once authorized) runs this after Step 1.

### 3. Zenodo deposit (SIG G7)
Package `code/`, `data/`, `results/` (including the new sensitivity sweep outputs) for
Zenodo deposit — satisfies reproducibility gate. This CAN be prepared by Sonnet (zip
structure, README, citation metadata) but the actual upload requires Pak Amien's Zenodo
account.
- **Action:** Sonnet can prepare the package on request; Pak Amien uploads.

### 4. Manuscript prose write-up
`PAPER_DRAFT_OUTLINE.md` is a structured outline with draft abstract, not a submittable
manuscript. Converting it to full VH&A-formatted prose (LaTeX or Word, per journal
template) is a substantial writing task — can be done by Sonnet on request, but should
happen AFTER Step 1 (co-author), since a palynologist's input may change methodological
framing, not just add a byline.
- **Action:** Sonnet drafts full prose on request, ideally after co-author feedback.

### 5. Actual journal submission
Never autonomous. Requires Pak Amien's account, cover letter sign-off, and final
approval of all claims.

---

## Sequencing (do not skip ahead)

```
1. Co-author secured (G2/G10)         ← BLOCKS everything below
2. Full manuscript prose drafted       (Sonnet, on request, informed by co-author)
3. Cross-model review (G9)             (on the CURRENT fixed draft, not the old one)
4. Zenodo package + upload (G7)
5. Submit (Pak Amien)
```

**Do not let this jump the queue ahead of the ME#19 forcing function.** WS-A hardening
(above, DONE) is preparation work — it does not discharge the binding non-exposure
constraint. The three external actions (Verberne reply → Zenodo D1/D2 → Lamqaddam
reply) remain priority-zero and are entirely separate from this checklist.
