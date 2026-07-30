# CLAUDE.md — Instructions for Claude Code

## Project: VOLCARCH
**Volcanic Taphonomic Bias in Indonesian Archaeological Records**

This is a **research repository**, not a software project. The "product" is knowledge (papers,
models, maps), not an application. Applications are side-products that serve the research.

It began as one question — *was there a Nusantara society before 400 CE, and did volcanism erase its
record?* — and has since branched into several distinct method-communities, plus a career/PhD track.
**The branching is intentional and fine.** `lines/` is how it is kept navigable.

---

## 1. Work out which MODE you are in BEFORE reading anything else

**Check your working directory.**

### 🔬 FOCUS MODE — cwd is inside `lines/<nn>_<name>/`

That line's `CLAUDE.md` is your contract. Its `STATE.md` is your work queue. Read those two, then
only what they point at.

**Do NOT** read other lines' `STATE.md`, other lines' manuscripts, the full `docs/JOURNAL.md`, or
`docs/WORKSTATE.md`. If the task genuinely needs a second line, **say so and stop** — crossing lines
is an orbit-mode decision, not yours to make silently.

Canonical content stays at the repo root (`papers/`, `experiments/`, `data/`, `tools/`). A line
folder holds only its contract, its state, and pointers — nothing is duplicated. Launch so you can
reach both, while keeping search scoped to the line:

```
cd lines/01_spatial && claude --add-dir ../..     # or run /add-dir ../.. in-session
```

### 🛰 ORBIT MODE — cwd is the repo root

Read **`docs/WORKSTATE.md`** first. It is the orbit dashboard: overdue external actions, the status
of every line, decisions waiting on the PI. Then `lines/README.md` for the line map.

Orbit mode is for what no single line can do: portfolio review, **Mata Elang**, hunting for topics,
retargeting rejected papers, unparking ideas, and accounting for the forcing function. It is **not**
for doing a line's work — enter the line for that.

> ⚠ **The documented failure mode of this project is using orbit mode as an escape hatch:** step out
> one level, find an interesting new topic, and don't send the email that is months overdue. 223
> experiments, 0 acceptances, 7 rejections. The binding constraint is **non-exposure, not rigor**
> (ME#19, memory `feedback_non_exposure`). This is why `WORKSTATE.md` opens with the exposure ledger
> and not with `IDEA_REGISTRY.md`.

---

## 2. Binding rules — apply in BOTH modes

### Research Integrity
- **Never fabricate data.** If data is unavailable, document the gap.
- **Always record what you tried**, even if it failed. Append to `docs/JOURNAL.md`.
- **Cite sources.** Every dataset, every number, every claim needs a traceable source.
- **Uncertainty is expected.** Use confidence intervals, not false precision.
- **Submission Integrity Gate (BINDING).** Before submitting/resubmitting ANY manuscript, pass
  `docs/SUBMISSION_INTEGRITY_GATE.md` (GO/NO-GO, gates G1–G10). **Never answer a central, valid
  critique by rewording** — fix the data or downgrade the claim. Re-derive every headline number
  blind from raw data. Adopted 2026-06-08 after the P7/Antiquity rejection + E214 counter-evidence.
- **F9:** do not count "N converging channels" as strength — the channels are correlated.
- **F10:** do not cite `docs/drafts/manifesto.md` as evidence. It is a claim, not a source.

### Experiment Protocol
- Numbered directory: `experiments/ENNN_short_name/`. **Numbering is global and flat** — never
  per-line, never recycled. Next free number: check `ls -d experiments/E*` (currently through E223).
- Every experiment has a `README.md` with hypothesis, method, data used, result, conclusion, and
  status (SUCCESS / FAILED / INCONCLUSIVE / REVISIT). Pre-register the design in `DESIGN.md` when
  the result could go either way — E217–E223 are the model to copy.
- Failed experiments are NOT deleted. They are documented and tagged FAILED.
- Revisiting a failed experiment creates a NEW experiment (e.g. `E005_revisit_E002_...`).
- Tag which line(s) an experiment serves in its README. An experiment may serve several — that is
  normal and is why `experiments/` is a shared flat pool, not partitioned by line.

### Code Style
- Python 3.10+, prefer scripts over notebooks for reproducibility.
- `requirements.txt` / `pyproject.toml` for dependencies.
- Prefer well-known libraries: geopandas, rasterio, scikit-learn, xgboost, folium.
- Comment with *why*, not *what*.
- Windows cp1252: use a UTF-8 wrapper.

### When Unsure
- Domain expertise you lack confidence in (archaeology, geology) → **flag it** in JOURNAL.md and
  suggest a domain expert.
- A task that could take the project in a fundamentally new direction → **ask first**.
- A result that contradicts the core hypothesis → **document it honestly** and flag for review.
  **Do not suppress inconvenient results.** E214 and E217–E223 are the project's most valuable
  work precisely because they are disconfirming.

### Session Continuity
- **Session start:** in focus mode read the line's `STATE.md`; in orbit mode read
  `docs/WORKSTATE.md`. Continue in-progress items before starting anything new.
- **After compaction:** re-read the same file to re-anchor.
- **Session end (MANDATORY):** update the `STATE.md` of every line you touched, and
  `docs/WORKSTATE.md` if a line's headline status or a PI decision changed. Append to `JOURNAL.md`.
- **Rule:** never let work disappear between sessions. In-progress ⇒ it is written down.
- Handoff docs: only the **current** one lives in `docs/`. Older ones → `docs/archive/handoffs/`.

### inBox Protocol
`inBox/` is a drop zone for new material added by the researcher between sessions.
- At session start, read everything in `inBox/`, identify it, and route it: drafts → `docs/drafts/`
  (+ entry in its README); data → `data/raw|processed/` (+ entry in `data/sources.md`); literature →
  the owning line's paper folder or `docs/bibliography/`; code → `tools/` or the experiment folder.
- **Also record which line it belongs to** in that line's `STATE.md`.
- After processing, `inBox/` must be **empty**. Log every routed item in `JOURNAL.md`.

### Ideas — never discarded
- `docs/IDEA_REGISTRY.md` — every idea gets an `I-NNN` and a maturity (SPARK → HYPOTHESIS →
  TESTABLE → READY → EXPERIMENT → RESULT → PAPER). **Retired papers ≠ retired ideas.**
- `docs/TRIGGER_MAP.md` — reverse blocker index: "if X happens, what becomes possible?"
- Parked papers keep a `PARKED.md` in their folder stating the **unpark conditions**
  (`papers/P16_computational_textual_archaeology/PARKED.md` is the template).
- Serendipity: found something for another line while working this one? Tag it in JOURNAL as
  `[BRIDGE → <line>, I-NNN]` and add it to that line's `STATE.md` inbox. Do not chase it now.
- `docs/drafts/` incubates paper ideas. A draft goes active only on: **testable hypothesis +
  accessible data + executable methodology.** *"Santai dalam waktu, serius dalam metode."*

---

## 3. Repo layout

```
volcarch-repo/
├── CLAUDE.md              ← you are here (both modes)
├── lines/                 ← ★ THE ENTRY LAYER. One folder per line of inquiry.
│   ├── README.md           ← line map (orbit-mode index)
│   └── NN_name/{CLAUDE.md, STATE.md}
├── docs/
│   ├── WORKSTATE.md        ← ★ orbit dashboard (short, by design)
│   ├── L1_CONSTITUTION.md  ← core hypotheses & ethics. Owned by line 06.
│   ├── L2_STRATEGY.md      ← phase/methodology. Background; stale since 2026-03-30.
│   ├── L3_EXECUTION.md     ← superseded by line STATE.md files. Background only.
│   ├── EVAL.md             ← how success is measured
│   ├── SUBMISSION_INTEGRITY_GATE.md  ← BINDING pre-submit gate
│   ├── JOURNAL.md          ← append-only log (8.8k lines — grep it, don't read it)
│   ├── IDEA_REGISTRY.md · TRIGGER_MAP.md · COMPANION_REPOS.md
│   ├── HANDOFF_<latest>.md ← current handoff only
│   ├── archive/            ← superseded handoffs + WORKSTATE logs
│   ├── correspondence/ · research_notes/ · bibliography/ · funding/ · HKI/ · drafts/
├── experiments/ENNN_*/     ← flat, global, SHARED across lines (2.2 GB)
├── papers/PNN_*/           ← manuscripts, canonical (383 MB)
├── data/{raw,processed}/   ← SHARED (7.9 GB). raw/ is never modified.
├── tools/                  ← shared scripts + voc_archnlp + globalise_pipeline + dashboard
├── maps/ · models/ · results/ · deploy/
└── inBox/                  ← drop zone; must be empty after processing
```

**External:** the molecular/population-data channel lives in the **`volcarch-genetics`** repo — see
`docs/COMPANION_REPOS.md`. Cite it as external evidence with commit/DOI pinning. It is separate for
**model-compatibility** reasons, not organisational ones; do not use it as a precedent for splitting
other lines out of this repo.

---

## 4. Current status
→ **Focus mode:** the line's own `STATE.md`.
→ **Orbit mode:** `docs/WORKSTATE.md`, then the latest `docs/HANDOFF_*.md`.
