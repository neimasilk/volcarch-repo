# lines/ — Jalur Penelitian (Lines of Inquiry)

**This folder is the entry layer.** Each subfolder is one *line of inquiry*: a durable question with
its own method, its own literature, its own reviewer community, and its own venue list.

```
cd lines/01_spatial && claude --add-dir ../..      # focus mode
cd .. && claude                                    # orbit mode (see ../docs/WORKSTATE.md)
```

---

## Why lines, and not papers or folders-of-experiments

| Candidate unit | Why it was rejected |
|---|---|
| **One folder per paper** | Papers are volatile — 7 rejected, 1 parked, several retargeted. E216 has no paper folder at all. And "one level up" from `papers/P2/` is just a paper list, not a vantage point. |
| **Move experiments into topic folders** | `experiments/` is 2.2 GB and `data/` is 7.9 GB, both shared. 16 experiments serve more than one paper. Partitioning would break relative paths in ~214 READMEs, LaTeX figures, and the dashboard — for navigational gain only. |
| **One repo per topic** | Cross-line review (Mata Elang) becomes impossible, and cross-references rot. Only justified for **model-compatibility** — the single reason `volcarch-genetics` is external. |

So: **a line folder owns no data and no manuscripts.** It owns a contract, a state file, and
pointers. Nothing is duplicated; canonical content stays at the repo root. Renaming or merging a line
costs two files, not a migration.

---

## The lines

| # | Line | Question | Papers | Exp | State |
|---|---|---|---|---|---|
| **01** | [`01_spatial`](01_spatial/) | Where were the settlements, and can a model find them? | P2, P17, P11 | 78 | 🟢 COOLING — P2 resubmitted 2026-08-11; P11→SPAFA |
| **02** | [`02_taphonomy`](02_taphonomy/) | Does volcanism actually destroy or hide the record? | P1, D2, ~~P7~~, ~~P3~~ | 46 | ⚠ carrying an unrepaired data defect |
| **03** | [`03_paleoenv`](03_paleoenv/) | Can a paleo-environmental measurement *falsify* the thesis? | (E216 → VHA) | 3 | 🧊 blocked on palynologist co-author |
| **04** | [`04_language_text`](04_language_text/) | What do language and texts preserve of the substrate? | P8, P9, P5, P19, ~~P16~~ | 62 | ⏳ P8 under review; P5 needs rewrite |
| **05** | [`05_archival_nlp`](05_archival_nlp/) | What do colonial archives record, and can NLP extract it? | D1, P21 (+ HKI product) | 14 | 🔧 tooling done, pipeline unrun |
| **06** | [`06_thesis`](06_thesis/) | The original question. Synthesis. | P0/MASTERPIECE, P18 | 27 | 🛑 **fallow — subtract-only** |
| **07** | [`07_career`](07_career/) | PhD, funding, exposure, HKI. *Not a research line.* | — | 0 | ✅ exposure ledger **empty** (2026-08-11) |
| — | `volcarch-genetics` (external) | Molecular/population evidence | — | 2 | separate repo at `D:\documents\volcarch-genetics` — see `docs/COMPANION_REPOS.md` |

Strikethrough = discontinued or parked; the folder and its record stay.
**All 214 local experiments are mapped** (E001–E224; 10 numbers never created, E053/E203 external).
Counts sum above 214 because 16 experiments serve two lines.
Authoritative per-line lists: `docs/EXPERIMENT_INDEX.md` §"By Line of Inquiry".

---

## Rules for this layer

1. **A line folder never holds canonical content.** No manuscripts, no CSVs, no code. Pointers only.
   If you catch yourself copying a file into a line folder, stop — link it instead.
2. **Experiment numbering stays global and flat.** `E224` is the next one regardless of line.
3. **An experiment may belong to several lines.** Add it to `LINE_MAP` in
   `tools/scan_experiments.py` (primary line first) and re-run the script — it prints an
   **UNMAPPED** block if you forget, which is what stops this layer from going stale the way
   `experiment_index.json` did (it sat at 84 of 214 for months). Never duplicate the directory.
4. **One line at a time.** Crossing lines mid-session is how 214 experiments happened. If a line's
   work requires another line, write it into that line's `STATE.md` inbox and return to orbit.
5. **Adding a line** is cheap (two files) but is an orbit-mode decision; per ME#19 it sits on the
   stop-list unless the PI lifts it (exposure happened 2026-08-11 — status is a PI call).
   Prefer an `I-NNN` in `docs/IDEA_REGISTRY.md`.
6. **Each line declares its recommended model** in its `CLAUDE.md`. This repo has already been bitten
   once by model/topic mismatch (`volcarch-genetics`); the declaration makes it explicit.
