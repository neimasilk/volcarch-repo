# SIG sign-off — P2 / JCAA #280 revision v0.2 — 2026-08-03 — run by Claude Opus 5

**DECISION: 🔴 NO-GO.**
**Dominant blocker: the manuscript does not exist.** `submission_jcaa_v0.2.tex` has not been written.
Everything that feeds a manuscript is ready; the manuscript is not.

Four gates cannot even be evaluated until there is prose to evaluate (G2, G7, G8, G9). That is not a
technicality — G8 is a grep over the manuscript, G7 requires the figures, and G9 is a review *of the
paper*. A gate that cannot be run is not a pass.

Re-run this sign-off in full when v0.2 exists. Deadline: **2026-08-20 (17 days)**, no extension.

---

## Gate readout

| Gate | State | Basis |
|---|---|---|
| **G1** re-derivation (blind recompute) | 🟡 **GREEN on inputs, unproven on the manuscript** | 61 headline numbers recomputed from per-run files, `*_outcome.json` never read: `revision_ammo/SIG_G1_VERIFICATION_20260803.md`. The 4 standing mismatches are claims we withdrew (K5, K6, K7, G1c). **But G1 asks whether the *paper's* numbers match, and there is no paper.** Turns GREEN only after v0.2 is written from doc 10 and the script is re-run against it. |
| **G2** domain-sanity | 🔴 **NOT RUN** | The 5 domain questions have never been posed to the *reframed* paper. The old ones tested a taphonomic claim that no longer exists. Needs new questions for a methods paper (e.g. "would an SDM practitioner accept that a common evaluation background is the right comparison?"). |
| **G3** canonical data | 🟡 **GREEN in the analysis, unproven in the text** | INT-1 closed: E219 recomputed Test 1 on the canonical inventory (13 centres in bounds, ρ = −0.281). The manuscript must actually *carry* −0.281 and the 13-centre list. Until the text exists, this is an intention, not a pass. |
| **G4** circularity | 🟢 **GREEN** | The paper's subject *is* a circularity: a design scored against negatives it selected for itself. It is named, measured (60/60 own-background inflation) and put in the title position rather than defended. This is the gate the paper passes best. |
| **G5** equifinality | 🟡 **GREEN if the prose keeps its discipline** | The TGB null now has **no** endorsed explanation: we proposed one (K4), pre-registered a test, and it failed (E224). Doc 10 §K-E requires the manuscript to say *unexplained*. If v0.2 quietly restores a causal story, this flips RED. |
| **G6** counter-evidence foregrounded | 🟢 **GREEN — strongest gate in the file** | The paper is the disconfirmation of its own published claim; E224 disconfirms our own replacement explanation; the response letter discloses INT-1, INT-4, a non-reproducing ρ, and K1–K7 unprompted. There is no "no counter-evidence" claim anywhere. |
| **G7** reproducibility | 🔴 **RED** | "A single script regenerates every figure and number." The number half exists (`verify_headline_numbers.py`). **The figure half does not** — blocks E and F are untouched; no v0.2 figure has been generated. |
| **G8** overstatement scan | 🔴 **CANNOT RUN** | It is a grep over the manuscript. The banned-phrase list is prepared from K5/K6/K7 — "picks the worst", "always", "~10× faster", "monotone to the end of the dial", "2–5.6×" — but nothing can be scanned yet. |
| **G9** cross-model critical review | 🔴 **NOT DONE** | Block I. Requires a finished draft to attack. |
| **G10** human independent review | ⚪ **N/A** | Required for P0/masterpiece only; recommended, not binding, here. |

**Score: 2 GREEN · 3 conditional · 4 RED-or-unrunnable · 1 N/A.** One RED = NO-GO, and there are four.

---

## What is genuinely finished

- **Claim set** (`review_package_20260727/10_SET_KLAIM_TERKOREKSI.md`) — authoritative, K1–K7 applied.
- **Response to Reviewers** — all 17 items, plus 6 unprompted disclosures. Draft, needs section
  cross-references once the manuscript has sections.
- **Reviewer-facing tables** — covariate inclusion matrix + analytical roles (R2-D, R2-E).
- **The integrity machinery itself** — a re-runnable blind-derivation script, which is what G1 wants.
- **Every `[RUN]` item** — E217–E224. No computation is outstanding.

## What remains, in the order it has to happen

| # | Work | Gates it unlocks | Rough size |
|---|---|---|---|
| 1 | **v0.2 prose** — new framing, §1 taxonomy, ENM literature, heritage-management section, jargon pass, abstract, title | G2, G8, and the manuscript half of G1/G3/G5 | the bulk of the remaining effort |
| 2 | **Figures** — 5 new (dial dose-response, selection panel, reported-vs-truth, robust/contingent map, seed stabilisation) + 3 refreshed (Fig 1, 4, 5 per R2-H) | **G7** | substantial |
| 3 | **G2 five questions** + **G8 grep** | G2, G8 | ~1 hour once prose exists |
| 4 | **G9 cross-model review** prompted to reject | G9 | ~half a day |
| 5 | **Re-run G1** against the finished manuscript | G1 | minutes |
| 6 | Re-run this sign-off and commit the result | — | — |

## Non-SIG blockers that also gate submission

- **Go Frendi's sign-off on the reversed conclusion.** He knows the conclusion flipped; he has not
  approved the new claim set. This gates *submission*, not drafting — drafting can start now.
- **Title.** Candidates resting on "selection picks the worst design" died with K5.
- **APC waiver** (£593) still unresolved since 2026-04-07, against an absolute zero-APC constraint.

## Honest read on the 17 days

Feasible, but only if the prose starts immediately and nothing else is inserted ahead of it. The
earlier "3–4 weeks" estimate covered blocks C–G; D, G and H are now done, which removes a real share of
it. What is left — a reframed 30-page manuscript plus eight figures — is still the largest single piece
of work in the revision, and it has not started.

The failure mode to watch is not running out of time. It is arriving at 19 August with a manuscript and
skipping gates 2, 8 and 9 because the deadline is tomorrow. **If that trade presents itself, the
protocol's own answer is to request the extension then, with a true reason** — which is exactly the
condition the 3 August withdrawal note named as legitimate.

---

*Run against `docs/SUBMISSION_INTEGRITY_GATE.md` (G1–G10). Re-run required before upload; this file is
superseded by the next sign-off, not amended.*
