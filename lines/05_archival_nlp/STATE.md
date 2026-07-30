# STATE — Line 05 ARCHIVAL NLP

**Updated:** 2026-07-30 · **Temperature:** 🔧 Tooling complete, corpus run not authorised

---

## Current position

The pipeline exists, is registered as an HKI product, and has **not been pointed at the corpus.**
500 dagregister files are downloaded. E211 Phase 1 has been ready since **2026-04-23** and is waiting
on one PI decision.

This is the cheapest large result available anywhere in the project: the instrument is built, the data
is on disk, and no external human is required.

---

## Blocked on PI

| # | Item | Since |
|---|---|---|
| 1 | **Approve the E211 run** on the 500 downloaded files. | 2026-04-23 |
| 2 | **DJKI HKI submission** — 4 registration documents are ready in `docs/HKI/`. Filing is a PI action. | 2026-04-23 |
| 3 | **D1 → JOAD** submission incl. the APC-waiver request in the cover letter. | — |

---

## Next actions for Claude

- [ ] **Pre-write the E211 evaluation protocol** *before* the run is authorised: annotated held-out
      set, entity types, metrics, and the selection rule — declared in advance. Line 01's refutation
      is the reason this comes first, not after.
- [ ] Dry-run VOC-ArchNLP end-to-end on a **10-file sample** to prove the four modules chain cleanly
      and to size the full run. A smoke test is not the corpus run and does not need approval.
- [ ] Prepare D1's Zenodo fallback package, so a JOAD waiver refusal does not block the DOI (see
      `docs/ZENODO_UPLOAD_GUIDE_20260610.md`).
- [ ] Check whether the SCC/GDPR question actually blocks anything in the planned E211 output. If the
      output is entities + counts only, it does not — write that down so it stops being a vague
      worry.

## Do NOT do

- ❌ Run E211 on all 500 files without approval.
- ❌ Commit or publish extracted Delpher/KB full text.
- ❌ Add a fifth module to VOC-ArchNLP. It is registered at v1.0.0 with four; scope growth here is how
  the corpus run keeps getting postponed.

## Inbox

- `E128` (OV depth) is cited as independent depth evidence by
  [02_taphonomy](../02_taphonomy/) — if WS-E touches it, that line owns the re-derivation.
- This line is the technical basis of all four PhD approaches (Verberne, Cohen, Vossen, UvA). A
  completed E211 run would materially strengthen every one of them — which is an argument for
  unblocking it, tracked in [07_career](../07_career/).
