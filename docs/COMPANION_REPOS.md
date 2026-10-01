# Companion Repositories

**Current state (2026-10-01): there are no companion repositories.** Every evidence channel lives in
this repo. This file is kept as the record of the one time a channel was split out, and why it came
back.

## `volcarch-genetics` (population-evidence channel) — RE-MERGED 2026-10-01

| Period | Where it lived |
|---|---|
| until 2026-06-10 | in this repo (`experiments/E053_*`, `experiments/E203_*`, `docs/bibliography/05_paleogenomics/`) |
| 2026-06-10 → 2026-07-30 | separate repo, nested inside this one (split commit `14a2fc2`) |
| 2026-07-30 → 2026-09 | separate repo, sibling at `D:\documents\volcarch-genetics` |
| **since 2026-10-01** | **back in this repo**, at the original paths — PI moved it back in and asked for it to be re-integrated |

**Why it was split:** model compatibility only. A topic classifier on one model mis-flagged the
project's session-start context as a biology topic, and this channel carried most of the
biology-domain vocabulary. The science never changed; only the location did. The PI paused the
research for ~7 weeks (mid-Aug → 2026-10-01) over this mis-flag.

**Why it came back:** the channel is a sub-part of the original question (who lived in Java before
~400 CE, and why their record is thin), not a separate topic. Keeping it outside made the evidence
map incomplete and made traceability depend on a repo with no remote.

**What was restored, and how (2026-10-01):**

| Item | Restored to | Note |
|---|---|---|
| E053 (molecular-preservation gap in Island SE Asia) | `experiments/E053_adna_taphonomic_gap/` | byte-identical to the pre-split version → path history is continuous |
| E203 (Indonesian population-structure meta-analysis) | `experiments/E203_genome_population_structure/` | byte-identical to the pre-split version |
| Subfield literature summary | `docs/bibliography/05_paleogenomics/` | companion version kept (it dropped the pointer front-matter and restored the original wording) |
| Working note (2026-03-07) | `docs/drafts/working_note_ancient_dna.md` | full text replaces the 443-byte pointer stub |
| Companion README | `docs/archive/volcarch-genetics_README_20260730.md` | historical record |
| Companion git history (2 commits: `5c7304c` init, `703fa18` README fix; no remote) | tag **`archive/volcarch-genetics-20260730`** in this repo | `git worktree add <dir> archive/volcarch-genetics-20260730` recreates it exactly |

**Line assignment:** E053 → `02_taphonomy` (primary) + `06_thesis`; E203 → `06_thesis`
(`tools/scan_experiments.py` LINE_MAP).

## Rules that still apply

- **F9 applies to this channel like any other.** It is correlated with the rest (same site
  inventory, same "absence" logic); never count it as an independent converging channel.
- **Known integrity flags on its results** are open, not settled: E053 did not survive the E154 FDR
  audit (casualty), and the manifesto §3 already downgraded the "molecular-preservation trap as
  positive evidence" argument as absence-of-evidence. Re-check before any manuscript cites them.
- **Session-start vocabulary convention** (memory `feedback_clean_vocabulary`): the navigation files
  read at session start stay in plain pointer language; experiment and literature bodies keep the
  precise domain terms. Real science is not euphemised.
- If a model's classifier mis-fires on this content again, the channel for that is `/feedback`, not
  another split.
