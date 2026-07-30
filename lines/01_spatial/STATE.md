# STATE — Line 01 SPATIAL

**Updated:** 2026-07-30 · **Temperature:** 🔥 HOT — hard deadline in **21 days**

---

## Hard deadline

**P2 resubmission to JCAA: 2026-08-20.** Remaining work was estimated at 3–4 weeks *before* the K1–K3
corrections were added. **An extension request to 2026-09-30 is the standing recommendation** and is
part of the held Verhagen email.

---

## Blocked on PI (nothing downstream can move)

| # | Item | Since |
|---|---|---|
| A | **Confirm authorship with the human Go Frendi.** The new manuscript reaches the *opposite* conclusion from the one he signed in March. `review_package_20260727/05_*` is Claude's analysis of his likely position, **not a signature.** | 2026-07-27 |
| B | **Send the Verhagen email** (disclosure + extension request). Draft exists, status **HELD**. Needs updating with E217–E223 **and K1–K3**. | 2026-07-27 |
| C | **Decide scope:** revision of #280 vs. new submission. Determines whether v0.2 is a revision or a new paper. | 2026-07-27 |
| D | **Permission to commit.** ~11 modified + 19 untracked files, including all of E217–E223 and the review package. **Zero commits made on 2026-07-27.** | 2026-07-27 |
| E | **v0.2 title** — candidates in `revision_ammo/JCAA_R1_RESPONSE_PLAN_20260727.md` §3. The "tautology-free" claim in the current title must be downgraded. | 2026-07-27 |

---

## Next actions for Claude (in order)

- [ ] **B′ — apply K1–K3 to the claim set.** *Claim corrections, not new experiments.* This is the
      only unblocked item that matters, and it must precede any prose. See `../CLAUDE.md` for the
      table. Output: a corrected claim set doc, then update `03_TEMUAN_REVISI.md`.
- [ ] **K4 confirmation run** (cheap, optional but valuable): add `road_dist` to the feature set,
      re-run E222 P3. Converts the TGB null from an unexplained result into a tested condition.
- [ ] Block D — **R2 covariate table**: per-experiment covariates + analytic role. Not started.
- [ ] Block F — **new figures**: material ready in `E220_*`, `E222_*`,
      `E221_priority_sets_*.npz`. Not started.
- [ ] Block E — **old figures** (Fig 1, 4, 5) refresh. Not started.
- [ ] Block G — **reviewer response letter**, 17 items. Not started.
- [ ] Block H — **SIG G1 blind re-derivation** of every headline number incl. E217–E223. Partial.
- [ ] Block I — **cross-model review (G9).** Not started.
- [ ] **Manuscript v0.2 prose** — 🛑 do NOT start until A, B/C, and B′ are closed. Writing 30 pages
      before knowing whether the editor wants a revision or a new submission is waste.

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
