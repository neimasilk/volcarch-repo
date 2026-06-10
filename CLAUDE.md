# CLAUDE.md — Instructions for Claude Code

## Project: VOLCARCH
**Volcanic Taphonomic Bias in Indonesian Archaeological Records**

This is a **research repository**, not a software project. The "product" is knowledge (papers, models, maps), not an application. Applications are side-products that serve the research.

## How to Navigate This Repo

**Always read in this order before starting any task:**

0. `docs/WORKSTATE.md` — **Read this FIRST, always.** Current in-progress work, blocked items, and session prompt. This is your work contract — continue unfinished items before starting new work.
1. `docs/L1_CONSTITUTION.md` — The "UUD". Core hypotheses, philosophy, ethics. Almost never changes. Read this to understand *why* this project exists.
2. `docs/L2_STRATEGY.md` — Current research phase and active papers. Changes per phase. Read this to understand *what we are working on now*.
3. `docs/L3_EXECUTION.md` — Active tasks and experiments. Changes frequently. Read this to understand *what to do next*.
4. `docs/EVAL.md` — Evaluation criteria and validation protocol. Read this to understand *how we measure success*.
5. `docs/JOURNAL.md` — Append-only research log. Read recent entries to understand *what has been tried and what happened*.
6. `data/schema.md` — Data format definitions. Read when working with datasets.

## Repo Structure

```
volcarch/
├── CLAUDE.md                  ← You are here
├── README.md                  ← Public-facing project description
├── docs/
│   ├── L1_CONSTITUTION.md     ← Layer 1: Core hypotheses & philosophy (stable)
│   ├── L2_STRATEGY.md         ← Layer 2: Current phase & methodology (per-phase)
│   ├── L3_EXECUTION.md        ← Layer 3: Active tasks & experiments (per-week)
│   ├── EVAL.md                ← Evaluation criteria & validation protocol (stable-ish)
│   └── JOURNAL.md             ← Research log: decisions, results, failures (append-only)
├── data/
│   ├── raw/                   ← Original downloaded data (never modify)
│   ├── processed/             ← Cleaned/transformed data
│   ├── schema.md              ← Data format definitions
│   └── sources.md             ← Data provenance documentation
├── experiments/
│   ├── E001_site_density_vs_volcanic_proximity/
│   │   ├── README.md          ← Hypothesis, method, result, conclusion
│   │   ├── ...code files...
│   │   └── results/
│   ├── E002_.../
│   └── ...
├── models/                    ← Trained ML models + configs
├── maps/                      ← Generated probability maps, visualizations
├── papers/
│   ├── P1_taphonomic_framework/
│   ├── P2_settlement_model/
│   └── P3_burial_depth/
├── tools/                     ← Shared utility scripts (scraping, GIS, etc.)
└── inBox/                     ← Drop zone for new materials (see protocol below)
```

## Rules for Claude Code

### Research Integrity
- **Never fabricate data.** If data is unavailable, document the gap.
- **Always record what you tried**, even if it failed. Append to JOURNAL.md.
- **Cite sources.** Every dataset, every number, every claim needs a traceable source.
- **Uncertainty is expected.** Use confidence intervals, not false precision.
- **Submission Integrity Gate (BINDING).** Before submitting/resubmitting ANY manuscript, pass `docs/SUBMISSION_INTEGRITY_GATE.md` (GO/NO-GO, gates G1–G10). Never answer a central, valid critique by rewording — fix the data or downgrade the claim. Re-derive every headline number blind from raw data. Adopted 2026-06-08 after P7/Antiquity rejection + palynology counter-evidence (E214).

### Experiment Protocol
- Every experiment gets a numbered directory: `E001_`, `E002_`, etc.
- Every experiment directory has a `README.md` with: hypothesis, method, data used, result, conclusion, and status (SUCCESS / FAILED / INCONCLUSIVE / REVISIT).
- Failed experiments are NOT deleted. They are documented and tagged FAILED.
- If revisiting a failed experiment, create a new experiment (e.g., `E005_revisit_E002_...`).

### Code Style
- Python 3.10+, prefer scripts over notebooks for reproducibility.
- Use `requirements.txt` or `pyproject.toml` for dependencies.
- Prefer well-known libraries: geopandas, rasterio, scikit-learn, xgboost, folium.
- Comment with *why*, not *what*.

### When Unsure
- If a task involves domain expertise (archaeology, geology) that you lack confidence in, **flag it** in JOURNAL.md and suggest consulting a domain expert.
- If a task could take the project in a fundamentally new direction, **ask first** rather than executing.
- If an experiment result contradicts the core hypothesis (L1), **document it honestly** and flag for review. Do not suppress inconvenient results.

### inBox Protocol
The `inBox/` folder is a **drop zone** for new materials (drafts, data files, references, etc.) added by the researcher between sessions.

**Rules:**
- Anything new goes into `inBox/` first.
- At the start of each session, Claude reads everything in `inBox/`, determines what it is, and routes it:
  - **Draft papers/ideas** → `docs/drafts/` (with entry in `docs/drafts/README.md`)
  - **Data files** → `data/raw/` or `data/processed/` (with entry in `data/sources.md`)
  - **References/literature** → relevant paper folder or `docs/`
  - **Code/tools** → `tools/` or relevant experiment folder
- After processing, `inBox/` must be **empty**. Nothing lives there permanently.
- Every item processed from `inBox/` gets logged in `docs/JOURNAL.md`.

### Draft Papers Pipeline
- `docs/drafts/` holds incubating paper ideas — frameworks discussed but not yet backed by data/experiments.
- See `docs/drafts/README.md` for the catalog with priorities and status.
- A draft enters active development only after passing the gate: **testable hypothesis + accessible data + executable methodology**.
- Filosofi: *"Santai dalam waktu, serius dalam metode."*

### Exploration Mode & Idea Preservation
- **`docs/IDEA_REGISTRY.md`** — Master catalog of ALL research ideas with maturity levels (SPARK → PAPER). Every idea gets an ID. Retired papers ≠ retired ideas.
- **`docs/TRIGGER_MAP.md`** — Reverse blocker index: "If X happens, what becomes possible?" Scan during Mata Elang reviews.
- **Serendipity tagging:** When working on Paper X and discovering something for Paper Y, tag it in JOURNAL: `[BRIDGE → PY, I-NNN]`
- During exploration sessions, new ideas go to IDEA_REGISTRY with appropriate maturity level. Never discard — everything gets an ID.
- Mata Elang weekly reviews: scan TRIGGER_MAP for newly unblocked ideas, update IDEA_REGISTRY maturity levels.

### Session Continuity Protocol
- **Session start:** Read `docs/WORKSTATE.md` FIRST. Continue in-progress items before starting anything new. WORKSTATE is a work contract, not background reading.
- **After context compaction:** If you lose context mid-session, re-read `docs/WORKSTATE.md` to re-anchor on what you were doing.
- **Session end:** Update `docs/WORKSTATE.md` (MANDATORY). Move completed items out, update next actions, record any new blockers. This is how future sessions know what to continue.
- **Rule:** Never let work disappear between sessions. If something is in progress, it must be in WORKSTATE.md.

## Current Status
→ See `docs/WORKSTATE.md` for immediate work contract (what to continue NOW).
→ See `docs/L3_EXECUTION.md` for full task list and context.
