# SUBMISSION INTEGRITY GATE (SIG) — VOLCARCH

**Status:** ACTIVE protocol, adopted 2026-06-08 after P7/Antiquity rejection + palynology counter-evidence.
**Authority:** This gate is BINDING. No manuscript is submitted (or resubmitted/revised) until it passes. Applies to P0, P2, P8, P11, P17 revisions, and everything after.
**Principle:** *Integritas akademisi* — sebuah klaim tidak boleh dikirim sebelum data DAN argumentasinya kuat. **Simple is better, fail fast, pivot early; santai dalam waktu, serius dalam standar ilmiah.**

---

## How it works
- Each gate G1–G10 returns **GREEN** (pass), **RED** (fail), or **N/A**.
- **One RED = NO-GO.** You may only turn RED→GREEN by *fixing the problem* or *downgrading the claim to match the evidence* — **never by rewording** (see Banned Move).
- A short **sign-off block** (date, who ran it, gate states, downgrades made) is filled per submission and committed to the paper folder as `SIG_signoff.md`.
- The gate is cheap. Running it on P7 *before* submission would have caught the fatal error in one hour. Running it after cost a peer-reviewed rejection.

## The Banned Move
> **You may NOT answer a central, valid critique with a wording change.**
Structural/central problems (artifact, circularity, equifinality, sampling-on-dependent-variable, missing counter-evidence) are fixed only by **new data** or **a weaker claim**. If you catch yourself writing a defensive paragraph for a Validity≥1 / Centrality=2 critique → STOP and downgrade the claim. (Lesson: P1 v4.0 "sampling on DV" could not be reworded away; P7 distance artifact could not.)

---

## The Gates

### G1 — Re-derivation (blind recompute)
**Check:** Every headline number is recomputed *from raw data, by a fresh script or agent that has NOT seen the paper.* If the blind number ≠ the paper number → RED.
**Why:** caught the P7 volcano artifact (90–170 km → 33–53 km); cleared P17 (14.6/27.6 km reproduced exactly). Reviewing prose never catches this.
**Failure code:** F2 (shared/contaminated-substrate artifact).

### G2 — Domain-sanity
**Check:** Answer 5 basic domain-fact questions a field expert would ask about the central claim. (e.g. "Is Sangiran really far from any volcano?" → No, it's in the Solo basin by Lawu.) Any wrong answer that the paper depends on → RED.
**Why:** "Sangiran 169 km from a volcano" and "all four known deep-time sites" (Java is one of Earth's richest *H. erectus* regions) both fail this instantly.
**Failure code:** F1 (domain-fact error), F6 (overstatement: "all", "only", "first").

### G3 — Canonical data (single source of truth)
**Check:** Every reference input comes from the project's canonical file. No ad-hoc/duplicated inputs. Volcano inventory = `data/processed/dashboard/volcanoes_java_full.csv` (30) — NOT the legacy 7-list, NOT the 9- or 15-lists. Coordinates of any new site are VERIFIED against a citable source before use.
**Why:** the project had 3 different volcano lists (7/9/15) → recurring artifacts.
**Failure code:** F2.

### G4 — Circularity audit
**Check:** Is any variable used to *define* a category and then "discovered" as a property of that category? Is the model trained on the same outcome it then predicts (sampling on the dependent variable)? If yes and it is load-bearing → RED.
**Why:** Zone B is defined via distance-dependent Pyle burial, so "Zone B is closer to volcanoes" is partly tautological; suitability model trained on known (Zone-A) sites.
**Failure code:** F3 (circular construction).

### G5 — Equifinality
**Check:** List ≥3 competing explanations for the central claim (e.g. for "no sites" → burial / never-settled / unsurveyed / fluvial erosion / small population). State which the paper can and cannot distinguish. If the claim asserts one cause without ruling out the others → downgrade to a detection/observation statement → otherwise RED.
**Why:** the deepest, recurring reviewer critique (Antiquity R2 pt5). Absence is not erasure.
**Failure code:** F4 (equifinal interpretation).

### G6 — Counter-evidence foregrounded
**Check:** Has ≥1 *independent* channel been tested that could refute the claim? Is contrary evidence cited honestly in the text (not omitted)? A claim with zero attempted disconfirmation → RED. **The phrase "no counter-evidence" is itself a RED flag** unless you have actively hunted for it.
**Why:** the palynology channel (E214) partially REFUTES a large pre-400 CE population — this MUST appear in P0, not be buried. The old "SLR found 0 counter-evidence" was a confirmation-architecture artifact, now falsified.
**Failure code:** R1 (confirmation-seeking).

### G7 — Reproducibility
**Check:** A single script regenerates every figure and number from raw data, and it runs. Saved result files actually contain the claimed results.
**Why:** E104's saved output had `candi: 0` — the headline wasn't even stored. Non-reproducible = untrustworthy.
**Failure code:** F7 (non-reproducible).

### G8 — Overstatement scan
**Check:** Grep the manuscript for `all / only / none / 0 / always / proven / first / unprecedented / certainly`. Verify each instance against the data; soften or substantiate.
**Failure code:** F6.

### G9 — Cross-model critical review
**Check:** ≥1 critical review by a different model (DeepSeek/Gemini) prompted to REJECT. Each Validity≥1/Centrality=2 critique is either fixed-with-data or the claim is downgraded. (Existing practice — keep it, but note G1–G6 catch what prose-review cannot.)
**Failure code:** F2–F6 residuals.

### G10 — Human domain independent review (recommended, ≤ $100)
**Check (for the flagship / masterpiece):** one human geoarchaeologist/archaeologist audits the central claims. Optional for minor notes; **required before P0/masterpiece submission.**
**Why:** the missing role in the collaboration (ME#17 §5). A domain human catches F1/F4 the AI+author pair miss.

### G11 — Non-numeric claims gate (added 2026-08-11)
**Check:** every **non-numeric / qualitative / causal** claim (e.g. "because", "despite", "implies",
"reflects") has a testable implication, OR is explicitly downgraded to a description/observation.
Grep the manuscript for causal connectors and verify each has a cited basis.
**Why:** the numeric pipeline is gated (G1–G3) but causal prose is not; K4 in P2 was a *beautiful*
causal story that pre-registration had to kill. Prose carries the same artifact risk as numbers.
**Failure code:** F6 (overstatement) / F4 (equifinal causal claim).

### G12 — Post-submission re-download gate (added 2026-08-11)
**Check:** after ANY upload to a journal/Zenodo portal, **download the file back and compare** (byte
hash / SHA1; for PDFs check producer + glyphs). If the server copy ≠ local copy → contact the editor
BEFORE it is processed.
**Why:** JCAA rewrites some article components (Producer `MiKTeX pdfTeX` → `mPDF 8.3.1`); Zenodo
metadata can silently land empty (D1 lesson, 2026-08-11); portals carry stale titles/abstracts
(P2 metadata). The portal is not a trusted pipe — verify ground truth via `GET`/re-download.
**Failure code:** F8-adjacent (external-side integrity).

---

## Failure taxonomy (for classifying any caught problem)
F1 domain-fact · F2 shared-substrate artifact · F3 circular construction · F4 equifinal interpretation · F5 scope-creep · F6 overstatement · F7 non-reproducible · R1 confirmation-seeking.

**Added ME#19 (2026-06-10) — process/strategy failure modes (not paper-internal, but they drop the program):**
- **F8 non-exposure / controlled isolation** — detect: "In the last 30 days, did any artifact reach an external judge (journal/supervisor/reviewer) for a binding decision?" If no while internal output is high → RED. The AI–PI loop optimizes for internal coherence and lacks finality; only external review supplies it.
- **F9 correlated-channel convergence** — detect: for every "N independent channels converge" claim, name the shared latent variable (landscape prior, elite-textual bias, population-continuity assumption); if one exists, discount the convergence. Multiplying likelihoods with shared hidden variables inflates confidence without truth.
- **F10 interpretive elasticity** — detect: "What result would have updated the central frame *against* the thesis?" If no such result is specifiable, the frame is not a theory and may not be cited as evidence/prior. Applies to the manifesto.

## Worked examples (this session)
- **P7 (Antiquity):** G1 RED (distance artifact), G2 RED ("all four", Sangiran), G4 RED (Zone B circular), G5 RED (equifinality). → correctly NO-GO. Should never have been submitted.
- **P17 (ArchCalc, live):** G7 RED (candi:0 non-reproducible), G3 RED (3 inventories) — BUT G1 GREEN on re-derivation (14.5/27.6 km survives). → result sound; fix methods + add reproducible script at revision; no withdrawal.
- **P0 (masterpiece):** G6 now forces inclusion of the palynology partial-refutation (E214). The thesis must be downgraded from "large hidden civilization erased" to a falsifiable weaker form.

## Sign-off template (copy to `<paper>/SIG_signoff.md` before submit)
```
SIG sign-off — <paper> — <date> — run by <name/agent>
G1 re-derivation: [GREEN/RED]  notes:
G2 domain-sanity: [ ]  5 Qs + answers:
G3 canonical data: [ ]
G4 circularity: [ ]
G5 equifinality: [ ]  competing causes listed + which distinguished:
G6 counter-evidence: [ ]  independent channel(s) tested:
G7 reproducibility: [ ]  regen script:
G8 overstatement: [ ]
G9 cross-model: [ ]  reviewer + verdict:
G10 human independent review: [GREEN/RED/N/A]
Downgrades made to pass: <list>
DECISION: GO / NO-GO
```
