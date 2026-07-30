# E218 / E219 — Design (pre-registration)

**Written 2026-07-27, before running.** Driver: E217 refuted P2's central claim. Before that refutation is
carried into a reframed manuscript — or into an email to the editor — it has to survive the same
adversarial treatment we applied to the paper. PI instruction: *"ga perlu email dulu, kita coba pikirkan
baik-baik, desain experimen baru kalau diperlukan"* — deadline 2026-08-20 leaves room to do this properly.

**Governing principle:** E217 is currently a demolition. A demolition alone is not a paper and not a safe
thing to tell an editor. E218 asks whether the demolition holds; E219 asks what stands in its place.

---

## E218 — Is the evaluation artefact real, and what drives it?

### Threats to the E217 conclusion, and the test for each

| # | Threat | Why it could invalidate E217 | Test |
|---|---|---|---|
| T1 | **Choice of common evaluation background.** E217 used uniform-random over the frame. | If flatness of the ladder depends on that one choice, the conclusion is contingent, not general. | Repeat under **4 fixed evaluation backgrounds**: uniform, TGB-drawn, hybrid-drawn, spatially stratified (regular lattice). |
| T2 | **Only 5 seeds.** | The paper itself used 20. We must not hold the refutation to a weaker standard than the claim it refutes. | **20 seeds.** |
| T3 | **Only AUC.** | AUC is the disputed instrument; using it alone to condemn itself is circular. | Add **TSS** and the **continuous Boyce index** (Hirzel et al. 2006) — the presence-only metric recommended precisely when absences are unreliable. |
| T4 | **One block size (0.45° ≈ 50 km).** | The paper reports sensitivity at 40/50/60 km; the artefact test should too. | Repeat at **~40 / ~50 / ~60 km**. |
| T5 | **Decimated (~300 m) sampling lattice.** | Decimation is my deviation from E013, not the paper's. | Re-run one configuration at **DECIMATE=5** (~150 m) and confirm the conclusion is unchanged. |
| T6 | **"Dissimilarity causes inflation" is asserted, not shown.** | E217 observed hybrid背景 sits further from presences (zdist 0.623 vs 0.503) and inferred mechanism. Correlation, not demonstration. | **Dissimilarity sweep** (below). |

### T1 is the decisive test, and its prediction is sharp

If the artefact is real, then **the hybrid design should win only when it is evaluated against
hybrid-like negatives.** Concretely:

| Evaluation background | Prediction if E217 is right | Prediction if the paper was right |
|---|---|---|
| uniform | ladder flat | hybrid wins |
| TGB-drawn | ladder flat | hybrid wins |
| **hybrid-drawn** | **hybrid wins — this is the artefact made visible** | hybrid wins |
| spatially stratified | ladder flat | hybrid wins |

A design that wins only on its own turf is not a better model. If instead hybrid wins under *all four*
fixed evaluation backgrounds, **E217 is wrong and the paper's claim survives** — and that outcome is
recorded here in advance.

### T6 — the dissimilarity sweep (mechanism, not just observation)

Construct backgrounds with **controlled** mean environmental dissimilarity from the presences (target mean
zdist swept across roughly 0.5 → 4.0, holding sample size and spatial extent fixed). For each:

- measure AUC scored on that background itself (`auc_own`);
- measure AUC scored on the fixed uniform evaluation background (`auc_common`);
- regress `inflation = auc_own − auc_common` on background mean zdist.

**Pre-registered reading:** if inflation rises monotonically with background dissimilarity while
`auc_common` stays flat, the artefact has a quantified mechanism and the finding becomes a *law* rather
than an anecdote — "AUC inflation is a function of how far you place your negatives". If inflation is
unrelated to dissimilarity, my proposed mechanism is wrong and the E217 result needs a different
explanation before anything is published.

---

## E219 — What survives? Does background design change the map rather than the score?

E218 can only tell us the AUC ladder is an artefact. It cannot tell us pseudo-absence design is
*worthless* — and asserting that would be its own overclaim. Target-group background was never justified
by AUC in the first place; its rationale (Phillips et al. 2009) is **correcting sampling bias in the
presences**, which is a claim about *where* the model predicts, not how well it discriminates.

**Hypothesis:** background design materially changes the predicted suitability surface even when it does
not change any discrimination metric.

**Tests:**
1. Train each design, predict over the full frame, compare surfaces — Spearman correlation between designs,
   and overlap of the top-decile ("survey priority") cells. Low overlap with identical AUC would be the
   headline.
2. Characterise the disagreement: does TGB/hybrid shift predicted suitability **away from road-accessible
   cells**, i.e. does it do the bias correction it is supposed to do? Regress the per-cell prediction
   difference on road distance.
3. **Secondary — answers Reviewer 2's R2-F directly:** terrain-matched comparison (elevation × slope × TRI ×
   TWI) of volcanic vs non-volcanic uplands (Southern Mountains karst, Kendeng limestone), comparing
   predicted suitability and observed site density. Kept as a subordinate analysis because the reframed
   paper withdraws the taphonomic claim R2-F was aimed at — but showing it beats asserting it.

**Pre-registered readings:**

| Outcome | Consequence |
|---|---|
| Maps differ substantially, disagreement tracks road distance | **Constructive replacement claim:** choose background design on bias-correction grounds; AUC cannot adjudicate. Paper has a positive contribution, not only a demolition. |
| Maps are near-identical | Background design does little of anything here. Paper narrows to the artefact finding alone, and says so. |
| Maps differ but disagreement is unrelated to road distance | TGB is changing predictions for reasons other than its stated rationale — report as an open problem, do not dress it up. |

---

## Sequencing

E218 first: everything downstream depends on whether the refutation holds. E219 only if E218 confirms.
No email to the editor until both have reported.
