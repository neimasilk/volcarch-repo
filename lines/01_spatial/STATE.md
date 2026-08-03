# STATE — Line 01 SPATIAL

**Updated:** 2026-07-30 · **Temperature:** 🔥 HOT — hard deadline in **21 days**

---

## Hard deadline

**P2 resubmission to JCAA: 2026-08-20 — no extension will be requested.** PI decision 2026-08-03: every
`[RUN]` item is done, dissolved, or out of scope, so the remaining revision is **writing**, and asking
for more time on run-related grounds would have given the editor a reason that is not true. Asking the
scope question ("revision or new submission?") was also dropped — it risks inviting the answer "new
submission", which would discard the only non-reject this project has had in 14 months. Withdrawn draft
and full reasoning: `docs/correspondence/EMAIL_VERHAGEN_EXTENSION_REQUEST_20260803.md`.

**Consequence: v0.2 is a revision of #280, and the corrections go in the Response to Reviewers.**

---

## Blocked on PI (nothing downstream can move)

| # | Item | Since | Status |
|---|---|---|---|
| A | **Confirm authorship with the human Go Frendi.** The new manuscript reaches the *opposite* conclusion from the one he signed in March. `review_package_20260727/05_*` is Claude's analysis of his likely position, **not a signature.** | 2026-07-27 | PI confirmed on 2026-08-03 that Go Frendi **knows** the conclusion reversed. An explicit sign-off on the v0.2 claim set is still outstanding — send him `10_SET_KLAIM_TERKOREKSI.md`. |
| B | ~~Send the Verhagen email~~ | 2026-07-27 | **CLOSED 2026-08-03 — withdrawn, no email will be sent.** |
| C | ~~Decide scope: revision vs new submission~~ | 2026-07-27 | **CLOSED — revision of #280.** |
| D | ~~Permission to commit~~ | 2026-07-27 | **CLOSED — committed.** |
| E | **v0.2 title** — candidates in `revision_ammo/JCAA_R1_RESPONSE_PLAN_20260727.md` §3. Candidates resting on "selection picks the worst design" are dead (K5); those resting on **evaluation incomparability** survive. | 2026-07-27 | open |

---

## Next actions for Claude (in order)

- [x] **B′ — apply K1–K3 to the claim set.** ✅ 2026-08-03 →
      `review_package_20260727/10_SET_KLAIM_TERKOREKSI.md`. The blind re-derivation found **three more**
      defects (K5 "picks the worst" is false, K6 monotonicity, K7 density 1.9× not 2×) plus **G1c**
      (published ρ = −0.163 does not reproduce; 5-seed re-run gives −0.243).
- [x] Block H — **SIG G1 blind re-derivation.** ✅ 61 checks, script
      `revision_ammo/verify_headline_numbers.py`, report `revision_ammo/SIG_G1_VERIFICATION_20260803.md`.
      Re-run it before upload; the 4 expected mismatches are the withdrawn claims K5/K6/K7/G1c.
- [ ] **E224 — K4 confirmation run.** Pre-registered; add `road_dist` to the feature set and repeat
      E222's P3. Converts the TGB null from an unexplained result into a tested condition.
- [ ] Block D — **R2 covariate table**: per-experiment covariates + analytic role.
- [ ] Block G — **reviewer response letter**, 17 items.
- [ ] Block F — **new figures**: material ready in `E220_*`, `E222_*`, `E221_priority_sets_*.npz`.
- [ ] Block E — **old figures** (Fig 1, 4, 5) refresh.
- [ ] Block I — **cross-model review (G9).**
- [ ] **Manuscript v0.2 prose** — scope and deadline are now settled, and the claim set is closed, so
      the only remaining gate is the co-author sign-off (A). Draft from **doc 10**, never from doc 08 §3.

## Deliberately NOT doing

Additional synthetic regimes · second-region replication → both declared **future work** in the
manuscript. E219 two-stage "suitable but absent" (R2-C) → dissolved when the taphonomic claim was
withdrawn.

---

## Other papers in this line

- **P17** (ArchCalc 365) — under review. No action. Do not touch the manuscript; it is live and
  double-blind.
- **P11** — retarget to SPAFA is **queued behind the [07_career](../07_career/) exposure actions** by
  PI decision. Do not start it as an alternative to P2 work.
- **D2** — Zenodo upload is a [07_career](../07_career/) item.

## Inbox (found while working, not yet triaged)

- `docs/experiment_index.json` covers only **84 of 214** experiment directories, so the
  experiment→paper mapping is stale. Re-run `tools/scan_experiments.py` and add a `line` field.
  Cheap, and it is what makes this whole layer self-maintaining.
