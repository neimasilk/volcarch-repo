# STATE — Line 02 TAPHONOMY

**Updated:** 2026-08-13 · **Temperature:** ⚠ WARM — no deadline, but owes a debt to every other line

---

## The one job

**WS-E — the integrity sweep. P17 arm DONE (08-03, koreksi SENT 08-11); P11 arm DONE (08-11,
v0.6 canonical30). Remaining: P1, P5, P8, manifesto.**

Enumerate every headline number in P1, P2, P5, P8, P11, P17, and `docs/drafts/manifesto.md` that
depends on volcano positions or the site inventory, then **blind re-derive each one** against
`data/processed/dashboard/volcanoes_java_full.csv` (30 volcanoes).

Why this is the priority: it is **mechanical, needs no PI decision, and unblocks others.** Line 01
needs it for SIG G1 on P2 v0.2; P11's retarget is gated on the canonical corrections; P17 is live at
a journal with numbers derived from the 7-volcano file.

Precedent to copy: `papers/P11_volcanic_informedness/revision_ammo/CANONICAL_INVENTORY_CORRECTIONS_20260610.md`
and the E031/E082 re-runs of 2026-06-10 (both **survived** — this sweep is expected to be survivable,
not fatal).

Output: one table per paper — *claim · old number · re-derived number · verdict (survives / restate /
withdraw)*.

---

## Progress — 2026-08-03

**P17 arm of WS-E: DONE.** `papers/P17_two_javas/revision_ammo/WSE_CANONICAL_INVENTORY_20260803.md`
(+ `verify_p17_numbers.py`, `p17_inventory_comparison.csv`).

- ✅ **The core claim survives and strengthens.** Canonical 30-volcano medians 14.5 vs 27.6 km, gap
  13.1 km, Mann-Whitney p = 1.5 × 10⁻⁷. Court-zone concentration **1.86× → 2.70×** — the published
  number *understates* the paper's own effect.
- ❌ **The methods statement does not describe the computation.** Neither published median is
  reproducible from the stated 10-volcano list (it gives 15.4 / 28.2 km). Consistent with the E104
  rebuild note that the original used **9 volcanoes for candi and 15 for inscriptions** — two rulers
  for the two groups being compared.
- ❌ n = 176 inscriptions published; the paper's own Java filter gives **174**.
- ✅ **P17 correction note SENT 2026-08-11** (portal queryId 162, verified landed) — while the paper
  is still under review.
- Also fixed: `e104_court_zone.json` had `candi: 0` throughout its distribution block (original run
  never populated it) — a canonical block is now appended.
- Gotcha for any future name-matching: the canonical file spells Sindoro **"Sundoro"** (GVP form).
  Prefix matching silently drops it.

**P2 arm: INT-1 confirmed closed** while working line 01 — E219 recomputed Test 1 on the canonical
inventory (ρ −0.281, 13 centres in bounds), verdict unchanged. But the *published* ρ = −0.163 does not
reproduce even on the old 7-volcano list (5-seed re-run gives −0.243): a single-instance value. That is
this line's defect showing up as seed instability, and it is now disclosed in the P2 response letter.

## Next actions for Claude

- [x] **P17 correction note to the ArchCalc editor** ✅ **SENT 2026-08-11** (queryId 162, verified
      landed). Nothing further until the journal responds.
- [ ] **WS-E, remaining papers:** P1, P11, P5, P8, `docs/drafts/manifesto.md`. P11 has a head start
      (`revision_ammo/CANONICAL_INVENTORY_CORRECTIONS_20260610.md`); P1 is the one that gates a
      submission.
- [x] **P7 correction notice** — 🅿 **PARKED by PI 2026-08-11**: Authorea/Wiley froze self-service
      editing of existing preprints; PI chose to leave the preprint as-is. Draft for revival:
      `docs/correspondence/EMAIL_AUTHOREA_SUPPORT_P7_CORRECTION_DRAFT_20260811.md`.
- [ ] **P1 → JASREP submission prep.** v2.0 is rewritten but has **not** passed SIG, and it contains
      spatial numbers touched by the defect. Sequence: WS-E on P1 → SIG → PI GO.
- [ ] Fold `E213` (aggradation/exposure asymmetry) into the P1 v2.0 exposure-window argument — it is
      the most recent mechanism evidence in this line and is not yet cited there.

## Blocked / external

| Item | Blocker |
|---|---|
| P1 submission to JASREP | needs WS-E + SIG + PI GO |
| ~~P7 correction notice posting~~ | 🅿 parked by PI 2026-08-11 (draft kept for revival) |
| Depth-data expansion (ADV-2's real fix) | needs borehole/excavation depth records — no accessible source yet. Standing item in `docs/TRIGGER_MAP.md`. |

## Inbox

- E107 resolved ADV-5 (Mon-Khmer substrate) and **upgraded E027**, but that upgrade is a
  [04_language_text](../04_language_text/) fact — check it is reflected there.
- The exposure/karst reframe (E178) is arguably a stronger paper than P1 v2.0 in its current form.
  Not an action — a note for the next orbit-mode review. ME#19 forcing function discharged 2026-08-11;
  new-paper status = PI call.
