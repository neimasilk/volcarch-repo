# Line 03 — PALEOENV (Paleo-Environmental Falsification)

> **Question:** Is there a paleo-environmental measurement that could **falsify** the thesis — and has
> it been made?

**Recommended model:** Opus. **Effort:** high. **Track:** B (curiosity, slow — deliberately not
deadline-driven).

---

## Scope

Pollen, phytolith, starch, and charcoal proxies for human land-use before 400 CE; the detection
physics of those proxies (RSAP, catchment dilution, deposition rates); and the design of measurements
that could settle the question. Reviewer community: **palaeoecology / archaeobotany**
(*Vegetation History and Archaeobotany*, *Quaternary Science Reviews*, *The Holocene*).

**Out of scope:** everything predictive (→ [01_spatial](../01_spatial/)) and everything about burial
rates (→ [02_taphonomy](../02_taphonomy/)).

---

## Why this line matters more than its size suggests

**This is the only line in the project that has produced real disconfirmation.**

`E214` (2026-06-08) **partially refutes** the strong pre-400 CE thesis. Before it, the project's
honest count of counter-evidence was zero across 214 experiments — a documented red flag
(memory `feedback_confirmation_architecture`, ME#17). E214 forced the thesis to be downgraded to a
*dispersed, low-visibility society*, and it is why `L1_CONSTITUTION.md` has a pending amendment.

**Protect this.** The standing instruction is: *do not let this line be reframed into confirmation.*
If an analysis here starts supporting the thesis, that is the result that needs the hardest scrutiny,
not the least.

### The strategic reframe this line embodies

From `docs/research_notes/FABLE_STRATEGIC_PLAN_20260707.md`: stop arguing *"the invisible civilization
existed"* (unfalsifiable — the equifinality trap that sank P7). Argue instead: ***"here is how you
would decide it, and here is the one missing measurement, priced at USD 8–15k."***
E216 is that argument made concrete, and is the designated **flagship** (WS-A).

---

## Experiments

| Experiment | What it is | Status |
|---|---|---|
| `E214_palynology_anthropogenic_signal` | Anthropogenic pollen signal before 400 CE | ⭐ **counter-evidence** — thesis downgraded |
| `E215_phytolith_starch_gap` | Phytolith/starch record gap | draft email to Castillo exists |
| `E216_paleoecological_interferometer` | Michelson–Morley-style falsification design: can Java's existing pollen-core network resolve a heartland clearing signal? | **HARDENED** (2026-07-07), *not* submission-ready |

### E216 — the finding, stated honestly

**OUTCOME-3, "The Decisive Missing Core."** Java's existing core network **cannot** resolve the
signal, for **two independent reasons** (not one):

1. **Resolution gap.** Network P(detect) = **0.000 at all 27 points** of the RPP_NAP × threshold ×
   alpha sensitivity grid. This is structural geometry, **not** a parameter artefact — the sweep
   strengthens the conclusion rather than weakening it. Core J6 *covers* the heartland geometrically
   but *dilutes* it (marine catchment): `n_cores_covering_heartland`=1 vs
   `n_cores_resolving_heartland`=0.
2. **Calibration not extractable.** The Dieng positive control is **QUALITATIVE ONLY — NOT
   re-derived** (raw data paywalled). The pre-registered branch was therefore a **NO-GO** hit, and
   `go_no_go_branch` in `results/OUTCOME.json` says so.

**The caveat that must stay in the opening sentences, not the appendix:** the "a core at Kedu gives
P=1.0" claim holds only in the optimistic corner. Full corner table
(`results/missing_core_corner_table.csv`): floor+uniform **12.6pp FAILS**; floor+clustered 34.5pp;
central+uniform 21.9pp; central+clustered 48.8pp. **The conservative corner fails in 85% of its own
parameter grid.**

Files: `SUBMISSION_CHECKLIST.md` (separates Claude-executable from human-gated, **in order**) ·
`results/PAPER_DRAFT_OUTLINE.md` (rewritten, caveat-first) · `code/e216_detection_function.py` ·
`code/e216_sensitivity_sweep.py` · `figures/fig1_network_rsap_map.png`,
`fig2_detection_power.png` (Fig 2 now shows the uniform-clearing line beside the clustered one, so
the caveat is visual) · `zenodo_upload/` (skeleton only — upload is human-gated) ·
`OPUS_REVIEW_20260625.md` (the 4 defects, all fixed and re-run, not just re-worded).

---

## Line rules

1. **Follow `SUBMISSION_CHECKLIST.md` order.** Do **not** jump to cross-model review (G9) or the
   Zenodo upload before the co-author exists. The order is deliberate.
2. **A palynologist co-author is a hard SIG requirement (G2/G10), not a nicety.** This line makes
   claims about pollen detection physics that no one here can peer-check.
3. **Caveats lead.** Any abstract, figure caption, or email out of this line states the conservative
   corner up front. This is the line whose credibility comes from candour.
4. **Never re-run a defect fix in text only.** All four E216 defects were fixed *and the code
   re-executed*, regenerating results. Keep that standard.
5. **Hardened ≠ submission-ready.** Do not describe E216 as ready to anyone outside the project.
