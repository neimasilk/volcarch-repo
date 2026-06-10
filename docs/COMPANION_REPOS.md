# Companion Repositories

Some evidence channels live in **separate repositories** to keep this repo focused
and to keep its session-start context within the newest model's topic-classifier
budget (see memory `feedback_clean_vocabulary`). The science is unchanged — only the
physical location differs. Cite these as external evidence, with commit/DOI pinning.

## `volcarch-genetics` (molecular / population-data channel)

Split out from `genetics/` on 2026-06-10. Holds the molecular-population evidence
channel (published-data reinterpretation; no wet-lab work).

| Item in this repo (now external) | Lives in companion repo | What it is |
|---|---|---|
| `experiments/E053_*` | `experiments/E053_*` | Molecular-preservation gap in Island SE Asia (literature synthesis) |
| `experiments/E203_*` | `experiments/E203_*` | Indonesian population-structure meta-analysis (5th evidence channel) |
| `docs/bibliography/05_paleogenomics/` | `bibliography_paleogenomics/` | Subfield SLR summary |
| — | `working_note_ancient_dna.md` | Working note: pre-Austronesian Java |

**Traceability:** experiments E055 (synthesis) and E214 (palynology) reference E053
as an evidence leg; the result statements remain valid and point here.

**Why separate:** this channel uses different data, methods, and a different observer
community than the spatial/NLP/epigraphic work. Treating it as a cited external
study is more honest than embedding it, and keeps this repo's session-start context
clean for the default model.

> When you lift `genetics/` out: move it to a *sibling* directory (e.g.
> `D:\documents\volcarch-genetics`), not nested inside this repo, then `git init`
> there. Nesting would re-surface its paths in this repo's `git status`.
