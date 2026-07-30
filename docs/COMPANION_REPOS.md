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

**Why separate:** the operative reason is **model compatibility** — the newest/most
capable model declines biology topics, and this channel is genuinely interesting, so
it is kept where a session can be opened against it deliberately (Opus). Secondarily,
the channel uses different data, methods, and observer community than the
spatial/NLP/epigraphic work, so citing it as an external study is more honest than
embedding it — and it keeps this repo's session-start context clean.

> ⚠ **This is not a precedent for splitting other topics out of this repo.** Focus and
> context are handled by scoped folders inside the repo — see `lines/README.md`. Only a
> model-refusal justifies a separate repo, and this is the only such channel.

## Location — resolved 2026-07-30

`D:\documents\volcarch-genetics` — a **sibling** directory, as this document instructed.
It had been sitting *nested* inside `volcarch-repo`, which re-surfaced its paths in this
repo's `git status`; it has now been moved out. Its own git history is intact
(`5c7304c init volcarch-genetics: …`).

**E053 and E203 are CANONICAL there, not mirrors.** Both were moved out on 2026-06-10.
What remained here until 2026-07-30 was an **empty `experiments/E203_*/results/` husk**
(zero files, untracked), which made it look as though E203 had never left. The husk is
deleted. Do **not** re-create either directory in this repo.

⚠ `volcarch-genetics/README.md` still describes its `experiments/` as a "reading copy
(mirror), canonical stays in `experiments/`" — that text predates the split and is now
wrong. Corrected in that repo on 2026-07-30.
