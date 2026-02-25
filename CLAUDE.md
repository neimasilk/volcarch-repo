# CLAUDE.md — Instructions for Claude Code

## Project: VOLCARCH
**Volcanic Taphonomic Bias in Indonesian Archaeological Records**

This is a **research repository**, not a software project. The "product" is knowledge (papers, models, maps), not an application. Applications are side-products that serve the research.

## How to Navigate This Repo

**Always read in this order before starting any task:**

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
└── tools/                     ← Shared utility scripts (scraping, GIS, etc.)
```

## Rules for Claude Code

### Research Integrity
- **Never fabricate data.** If data is unavailable, document the gap.
- **Always record what you tried**, even if it failed. Append to JOURNAL.md.
- **Cite sources.** Every dataset, every number, every claim needs a traceable source.
- **Uncertainty is expected.** Use confidence intervals, not false precision.

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

## Current Status
→ See `docs/L3_EXECUTION.md` for what to work on now.
