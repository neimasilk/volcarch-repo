# WORKSTATE — Orbit Dashboard

**Updated:** 2026-08-11 · **This file is short by design. Keep it that way.**

> # 🎉 P2 DIKIRIM KE JCAA — 2026-08-11
>
> **Setelah 14 bulan dan 224 eksperimen, eksposur terjadi.** Naskah + suplemen + surat balasan
> terunggah ke portal #280, komponen benar, 9 hari sebelum tenggat. Status Round 1: *"Submission has
> been resubmitted for another review round."* 19 commit ter-push. Rincian dan verifikasi:
> [`lines/01_spatial/STATE.md`](../lines/01_spatial/STATE.md).
>
> **Ini menutup satu-satunya item paling lama di §2 dan mengosongkan §4 untuk P2.** Yang tersisa di
> §1 adalah tepatnya jenis aksi yang sama: sesuatu yang sudah siap, tinggal dikirim.

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
> **Latest handoff: `docs/HANDOFF_20260811.md`** (older ones in `docs/archive/handoffs/`). It records
> the submission itself; `HANDOFF_20260810.md` before it recorded the pre-submission review that
> cleared the way (13 findings past SIG's 9/9 GREEN, 4 blocking, all fixed). **P2 is SUBMITTED.**
> For the P2
> claim set, the authoritative document is
> `papers/P2_settlement_model/review_package_20260727/10_SET_KLAIM_TERKOREKSI.md` (3 Aug, +A7–A9 on
> 5 Aug) — it supersedes doc 08 §3 and carries K1–K7. Docs 07 and 08 contain numbers we have since
> withdrawn.
>
> ✅ **Git: `main` is pushed and in sync with `origin/main` (2026-08-11).** This is what makes the
> manuscript's Data Availability statement true, and it is also what stopped the confidential JCAA
> reviewer report from being publicly browsable.

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
27 Jul added world-class rigor and **zero** exposure; 30 Jul added infrastructure and zero exposure;
3 Aug added a claim set, a response letter, a failed experiment and two correction drafts — and zero
exposure. **11 Aug broke that run: P2 went out.** The thing that finally moved it was not more rigor;
it was one session spent driving the portal instead of improving the manuscript.
**Items 1–3 and 5 below are in exactly the same state P2 was in on 10 August: finished, and unsent.**

---

## 2. ⏰ Clock

| Deadline | Item |
|---|---|
| ~~2026-08-20~~ | ~~P2 resubmission to JCAA~~ ✅ **MET 2026-08-11, nine days early.** Nothing owed to JCAA now; the next move is theirs. |
| Dec 2026 | Edinburgh PhD application window (entry Oct 2027) |

---

## 3. Line status

| # | Line | Temp | Next action | Owner |
|---|---|---|---|---|
| **01** | [spatial](../lines/01_spatial/STATE.md) | 🟢 COOLING | **P2 SENT 11 Aug** — 3 files uploaded, verified byte-for-byte against the server copies, editor notified. Open: portal metadata still shows the withdrawn title/abstract and authors cannot edit it — requested from the editor, **chase if unanswered**. Next line work: P11 retarget, still queued behind §1. | journal |
| **02** | [taphonomy](../lines/02_taphonomy/STATE.md) | ⚠ WARM | **WS-E: P17 arm DONE** (core claim survives and strengthens; methods sentence + 3 numbers need correcting **with a live journal**). Next: P1, P11, P5, P8, manifesto | Claude; editor note = PI |
| **03** | [paleoenv](../lines/03_paleoenv/STATE.md) | 🧊 BLOCKED | Write the E216 prose manuscript (parallel to co-author search) | Claude; co-author = PI |
| **04** | [language_text](../lines/04_language_text/STATE.md) | ⏳ WAITING | **P5 rewrite** → *Asian Ethnology* (overdue since ~June) | Claude |
| **05** | [archival_nlp](../lines/05_archival_nlp/STATE.md) | 🔧 READY | Pre-write E211 eval protocol; 10-file smoke test | Claude; run authorisation = PI |
| **06** | [thesis](../lines/06_thesis/STATE.md) | 🛑 FALLOW | Nothing. Subtract-only. | — |
| **07** | [career](../lines/07_career/STATE.md) | 🚨 BOTTLENECK | The three items in §1 | **PI** |
| — | `volcarch-genetics` (external repo) | — | cite as external; see `COMPANION_REPOS.md` | — |

---

## 4. Decisions waiting on the PI

**P2 / JCAA — ✅ NOTHING LEFT.** All five decisions closed and the paper is submitted (11 Aug).
The APC waiver was raised inside the response letter (PI decision 11 Aug) rather than as a separate
email. One thing is now owed *by the journal*: the record still carries the withdrawn title and
abstract, and the author-side form cannot save changes — asked for in the Review Discussion, chase if
it goes quiet.
**P17 / ArchCalc #365 — NEW, time-sensitive:** send the correction note to the editor while the paper
is still under review. Draft: `docs/correspondence/EMAIL_ARCHCALC_P17_CORRECTION_DRAFT_20260803.md`.
**P7:** post the preprint correction notice — overdue since 2026-06-04, costs one login. Draft:
`papers/P7_TOM/CORRECTION_NOTICE_DRAFT_20260803.md`.
**Other:** authorise the E211 corpus run (since 2026-04-23) · file DJKI HKI (4 docs ready) ·
palynologist co-author outreach for E216 · L1 amendment (adopt *"peradaban vulkanik"*) · send the
Vossen/VU email · ~~push the commits to GitHub~~ ✅ done 2026-08-11.

---

## 5. Portfolio

| Paper | Line | Status |
|---|---|---|
| **P2** Settlement model | 01 | ⏳ **RESUBMITTED 2026-08-11** — JCAA #280 revision v0.2, nine days inside the deadline. Core claim self-refuted and replaced; the paper is now about the artefact. Uploaded: manuscript (29 pp), Supplementary Tables S1–S6 (5 pp), Response to Reviewers (8 pp, "Response to reviewers" component as the editor required). Server copies re-downloaded and checked: text identical, same SHA1. Editor notified; APC waiver raised in the letter. ⚠ Portal metadata still shows the old title/abstract — author cannot edit, requested from the editor. |
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
