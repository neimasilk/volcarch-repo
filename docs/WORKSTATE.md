# WORKSTATE — Orbit Dashboard

**Updated:** 2026-08-10 · **This file is short by design. Keep it that way.**

> **In FOCUS MODE (cwd inside `lines/<nn>_*/`) you should not be reading this.** Read that line's
> `CLAUDE.md` + `STATE.md`. This file is the **orbit** view: what the whole portfolio is doing, what is
> overdue, and what only the PI can unblock.
>
> Per-line work queues live in `lines/*/STATE.md` and are **authoritative** for their line. This file
> routes; it does not duplicate. If they disagree, the line's `STATE.md` wins.
>
> **Session log ≠ work contract.** Narrative goes to `docs/JOURNAL.md`. The 871-line append-log this
> file used to be is preserved verbatim at `docs/archive/WORKSTATE_LOG_thru_20260727.md`.
>
> **Latest handoff: `docs/HANDOFF_20260810.md`** (older ones in `docs/archive/handoffs/`). The
> comprehensive pre-submission review the PI asked for **ran on 10 Aug and is closed**: 13 findings
> that SIG's 9/9 GREEN had missed, 4 of them blocking, all now fixed. **P2 is submission-ready and
> still NOT submitted — the only remaining steps are `git push` and the portal upload, both PI.**
> For the P2
> claim set, the authoritative document is
> `papers/P2_settlement_model/review_package_20260727/10_SET_KLAIM_TERKOREKSI.md` (3 Aug, +A7–A9 on
> 5 Aug) — it supersedes doc 08 §3 and carries K1–K7. Docs 07 and 08 contain numbers we have since
> withdrawn.
>
> ⚠ **Git: on `main`, the whole 2026-08-03 session is committed and unpushed** (`git rev-list --count origin/main..main`). Push awaits the PI.

---

## 1. 🚨 EXPOSURE LEDGER — read before anything else

The binding constraint on this project is **non-exposure, not rigor** (ME#19, 2026-06-10). Original
forcing-function deadline: **~2026-06-24 — passed.** All three items are send-ready. **PI only.**

| # | Action | Ready since | Days waiting |
|---|---|---|---|
| 1 | Send **Verberne** reply v4 (Leiden PhD — she asked 2 questions and is waiting) | 2026-06-09 | **~55** |
| 2 | Upload **D1 + D2 to Zenodo** → paste 2 DOIs | 2026-06-10 | **~54** |
| 3 | Send **Lamqaddam** reply (UvA — offered a BPI Dosen support letter) | 2026-04-23 | **~102** |
| 4 | **NEW —** send the **P17 correction note** to the ArchCalc editor (paper is under review now) | 2026-08-03 | 0 |
| 5 | **NEW —** post the **P7 preprint correction notice** | 2026-06-04 (defect known) | **~60** |

Detail + drafts: [`lines/07_career/STATE.md`](../lines/07_career/STATE.md).

**Scorecard: 0 acceptances · 7 rejections · 3 under review · 224 experiments.**
The 2026-07-27 session added world-class rigor and **zero** exposure; 2026-07-30 added infrastructure
and zero exposure; **2026-08-03 added a claim set, a response letter, a failed experiment and two
correction drafts — and zero exposure.** Every one of those was correct work. None of it moved the
constraint. Three of the five things now waiting on the PI are *send* actions, and two of them
(P17 correction, P7 notice) have a clock on them.

---

## 2. ⏰ Clock

| Deadline | Item |
|---|---|
| **2026-08-20** (**10 days**) | **P2 resubmission to JCAA — no extension will be requested** (PI decision 2026-08-03). **Manuscript v0.2 complete and reviewed (10 Aug):** 29 pp clean compile, supplementary tables S1–S6 generated, 6 ENM references added after the pre-submission review, Table 4 estimator corrected, AI disclosure rewritten. G1 final: 64 checks / 4 mismatch (the deliberately withdrawn claims). **Remaining: `git push` + portal upload — both PI.** |
| Dec 2026 | Edinburgh PhD application window (entry Oct 2027) |

---

## 3. Line status

| # | Line | Temp | Next action | Owner |
|---|---|---|---|---|
| **01** | [spatial](../lines/01_spatial/STATE.md) | 🔥 HOT | **Pre-submission review CLOSED (10 Aug): 13 findings past SIG, 4 blocking, all fixed.** Manuscript 29 pp + supplement 5 pp + response letter all ready. Claude's work on P2 is **done**. | **PI: push + upload** |
| **02** | [taphonomy](../lines/02_taphonomy/STATE.md) | ⚠ WARM | **WS-E: P17 arm DONE** (core claim survives and strengthens; methods sentence + 3 numbers need correcting **with a live journal**). Next: P1, P11, P5, P8, manifesto | Claude; editor note = PI |
| **03** | [paleoenv](../lines/03_paleoenv/STATE.md) | 🧊 BLOCKED | Write the E216 prose manuscript (parallel to co-author search) | Claude; co-author = PI |
| **04** | [language_text](../lines/04_language_text/STATE.md) | ⏳ WAITING | **P5 rewrite** → *Asian Ethnology* (overdue since ~June) | Claude |
| **05** | [archival_nlp](../lines/05_archival_nlp/STATE.md) | 🔧 READY | Pre-write E211 eval protocol; 10-file smoke test | Claude; run authorisation = PI |
| **06** | [thesis](../lines/06_thesis/STATE.md) | 🛑 FALLOW | Nothing. Subtract-only. | — |
| **07** | [career](../lines/07_career/STATE.md) | 🚨 BOTTLENECK | The three items in §1 | **PI** |
| — | `volcarch-genetics` (external repo) | — | cite as external; see `COMPANION_REPOS.md` | — |

---

## 4. Decisions waiting on the PI

**P2 / JCAA (1 left of 5):** ~~Verhagen email~~ withdrawn 3 Aug · ~~scope~~ settled (revision of #280) ·
~~commit permission~~ done · ~~**Go Frendi sign-off**~~ ✅ **RESOLVED 2026-08-05** (PI confirmed he is OK
with the reversed claim set; Authors' Contributions sentence now true) · ~~**v0.2 title**~~ ✅
**RESOLVED 2026-08-05 — kandidat 3** (no manuscript change).
**P17 / ArchCalc #365 — NEW, time-sensitive:** send the correction note to the editor while the paper
is still under review. Draft: `docs/correspondence/EMAIL_ARCHCALC_P17_CORRECTION_DRAFT_20260803.md`.
**P7:** post the preprint correction notice — overdue since 2026-06-04, costs one login. Draft:
`papers/P7_TOM/CORRECTION_NOTICE_DRAFT_20260803.md`.
**Other:** authorise the E211 corpus run (since 2026-04-23) · file DJKI HKI (4 docs ready) ·
palynologist co-author outreach for E216 · L1 amendment (adopt *"peradaban vulkanik"*) · send the
Vossen/VU email · **push the 2026-08-03 commits to GitHub**.

---

## 5. Portfolio

| Paper | Line | Status |
|---|---|---|
| **P2** Settlement model | 01 | 🔥 **R&R** JCAA #280, deadline 2026-08-20, no extension. Core claim self-refuted; reframed around the artefact. **v0.2 complete and pre-submission-reviewed (10 Aug):** `submission_jcaa_v0.2.tex` 29 pp clean compile; `supplementary_tables_v0.2.pdf` (6 tables, generated from raw results); response letter upload-ready; 6 ENM references added (Crossref-verified) closing R1's "not novel" exposure; Table 4 gaps corrected to the seed-average (+0.122 → +0.105); AI disclosure rewritten to match the public record; `LICENSE` added; confidential reviewer report un-published. **G1 10 Aug: 64 checks / 4 mismatch** (the withdrawn claims). **Remaining: push + upload (PI).** |
| **P17** Two Javas | 01 | ⏳ under review — ArchCalc #365, Diamond OA. Best odds. ⚠ **WS-E (3 Aug): finding survives and strengthens (court concentration 1.86× → 2.70×), but the methods sentence does not describe the computation and 3 numbers are wrong.** Correction note to the editor is drafted and waiting. |
| **P8** Linguistic fossils | 04 | ⏳ under review — *Oceanic Linguistics* OL-03-2026-11. arXiv:2604.00023. |
| **P11** Volcanic informedness | 01 | rejected 2× (both editorial). Core finding survives. Retarget SPAFA — queued behind §1. |
| **P1** Taphonomic framework | 02 | rejected 2×. v2.0 rewritten → JASREP. Needs WS-E + SIG. |
| **P5** Volcanic ritual clock | 04 | rejected (BKI) → *Asian Ethnology*. **Rewrite overdue.** |
| **P9** Peripheral conservatism | 04 | rejected (JSEAS). HOLD → DHQ. |
| **P16** Textual archaeology | 04 | 🅿 PARKED — convergence refuted (E090 v7). Unpark conditions in `PARKED.md`. |
| **P0** / MASTERPIECE | 06 | fallow. WS-B reframe queued behind WS-A. |
| **D1** / **D2** | 05 / 02 | drafts ready; Zenodo upload = §1 item 2. |
| **P7** TOM | 02 | ☠ dead — peer-rejected; preprint needs a correction notice. |
| **P3, P14** | 02, 04 | discontinued. **P18** HOLD. **P15** dissolved into P5. |

---

## 6. Orbit-mode rituals

**Mata Elang** (weekly strategic review) — criticism matrix {confidence × reversibility}. Scan
`docs/TRIGGER_MAP.md` for newly unblocked ideas; update maturity levels in `docs/IDEA_REGISTRY.md`.
Records: `docs/research_notes/MATA_ELANG_*.md` (through #19).

> ⚠ **ME#19 STOP-LIST IS ACTIVE.** No new Mata Elang / audit / skeptical-review sprint, no new papers,
> no new lines, until **exposure happens** (§1). This dashboard opens with the ledger and not with
> `IDEA_REGISTRY.md` for exactly this reason: stepping out one level must not become the way to avoid
> sending an email.

**Where ideas are kept safe** (so that parking something is not losing it):
`docs/IDEA_REGISTRY.md` (I-NNN, SPARK→PAPER) · `docs/TRIGGER_MAP.md` (if X unblocks, what opens) ·
`papers/*/PARKED.md` (unpark conditions) · `docs/drafts/` (incubating) ·
`docs/research_notes/*_LEAD_*.md` (live leads).

**Binding gates:** `docs/SUBMISSION_INTEGRITY_GATE.md` (G1–G10, pre-submit GO/NO-GO) ·
`docs/EVAL.md` · **F9** don't count correlated channels · **F10** don't cite the manifesto.

---

## 7. Housekeeping

- ✅ **2026-07-30:** experiment index regenerated — **84 → 213** entries, and **all 213 local
  experiments now carry a `lines` field.** Assignments are an explicit, auditable `LINE_MAP` in
  `tools/scan_experiments.py`; the script prints an **UNMAPPED** block if a new experiment is not
  added. Per-line lists: `docs/EXPERIMENT_INDEX.md` §"By Line of Inquiry".
- ✅ **2026-07-30:** `volcarch-genetics` moved out to `D:\documents\volcarch-genetics` (sibling, not
  nested — it was polluting this repo's `git status`). The apparent E203 contradiction was an **empty
  `results/` husk**, now deleted; E053 + E203 are canonical in that repo. `COMPANION_REPOS.md` and the
  genetics README both corrected.
- `.claude/` holds stale Feb 2026 handoffs and CODEX prompts mixed with `settings.local.json`.
