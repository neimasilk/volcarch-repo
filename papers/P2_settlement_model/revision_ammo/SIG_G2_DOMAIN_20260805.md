# SIG G2 — Domain-sanity check on the reframed v0.2 manuscript

**Date:** 2026-08-05 · **Run by:** Claude (against the final prose of `submission_jcaa_v0.2.tex`)
**Gate:** G2 of `docs/SUBMISSION_INTEGRITY_GATE.md` — five domain questions posed to the *reframed*
methods paper (the old G2 questions tested a taphonomic claim that no longer exists).

Five questions a computational-archaeology / SDM practitioner would plausibly ask, answered against the
manuscript text. **Result: all five pass on the manuscript's own disclosures; no change required.**

---

## Q1. Is "hold the evaluation background fixed" the right comparison, or just one choice among many?

An SDM practitioner would ask whether scoring every design on a *uniform* background is the correct
identification strategy, given that the common background is itself a modelling choice.

**Answer.** The paper's claim is deliberately about *fixing* the background, not about *uniform* being
the true one. The E218 matrix (§3.2) shows the hybrid design loses on **uniform, target-group AND
stratified** evaluation backgrounds (0/3 on each) and wins only on its own (3/3). So the artefact
disappears under *any* held-fixed common background; the conclusion does not rest on the specific
choice. The abstract and §4.1 state the requirement as "a common evaluation background held fixed
across designs", which is the correct, background-agnostic framing.

## Q2. Does "no interior optimum" generalise beyond the two dials that were swept?

A practitioner might ask whether the "maximum at the edge" property is general or an artefact of the
specific `hard_frac` grid.

**Answer.** Scoped honestly. The claim-set B1 and the manuscript state it for the sweeps performed —
synthetic (monotone) and real (monotone from 0.1, one dip −0.0071 at 0.0→0.1) — and the sentence now
carries the scope explicitly: *"its maximum always sits at the edge of whatever dial is swept, in both
the synthetic and real sweeps."* No claim is made for un-swept dials or other design dimensions.

## Q3. Is the "selection criterion" a real part of the workflow, or a reconstruction?

The E007–E013 progression was iterative development, not a formal selection run. A critic could say the
paper treats "the criterion" (AUC on own background) as if it were a formal selection rule.

**Answer.** The manuscript is explicit about the operating point: *"We stopped at hard_frac = 0.30
because our grid stopped there, not because the criterion said to"* (§3.3), and *"the criterion does
not pick the truth-best design in 29/60 cases but the differences are essentially zero"* (K-B4). The
criterion is presented as a property of the reported metric — that it never points inward — not as a
claim that the authors mechanically selected on it. This is a mild framing risk, but it is disclosed at
the exact point where a reader would over-read it.

## Q4. Are four synthetic worlds with n≈300–500 enough to support the claims made?

A practitioner would want to know the transfer limits of the synthetic results.

**Answer.** Fully disclosed in §4.6 (Limitations): item 3 states *"Four synthetic regimes are not a
universal result... no background design beat uniform on truth. This is not a refutation of target-group
background in general; it is a result for the regimes and sample sizes tested."* Item 1 discloses the
calibration gap (the criterion picks hybrid on real data but random on synthetic at the operating
point), which limits transferring E222 back to the real case. Item 4 discloses the +0.03 detection
floor. No overclaim survives these.

## Q5. Is reporting the TGB null as "unexplained" a defensible endpoint?

A practitioner might ask whether an unexplained null, after one failed manipulation (E224), carries any
information.

**Answer.** Yes, and this is the gate's strongest pass. The manuscript (§3.5) reports the null as
unexplained, discloses the failed pre-registered test (E224), discloses the limitation of its own test
(`road_dist` correlates +0.49 with `river_dist`), and explicitly resists re-interpreting the secondary
metric that did move (map Jaccard was the pre-registered primary; it said no). That is the K4
discipline the SIG itself demands.

---

**Verdict: G2 GREEN.** No domain question produces a manuscript change. The reframed paper's claims are
scoped to their evidence in each case, and the two weakest spots (Q3 framing, Q4 transfer) are
disclosed at the point of use rather than hidden.
