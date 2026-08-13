#!/usr/bin/env python3
"""
Check that experiment counts are consistent across all documentation files.
Solves the B3 structural risk: document drift when experiment counts are
manually updated in multiple places.

Usage: python tools/check_doc_sync.py
Exit code: 0 if consistent, 1 if mismatches found

KANARI v2 (2026-08-13, kritik sistem putaran 2 / C013):
- LIVE targets must equal the on-disk directory count (exit 1 otherwise).
- HISTORICAL targets (docs marked stale/background by CLAUDE.md, or dated
  "as of" claims) are printed as warnings only — they must not block.
- Wired as a session-start ritual: docs/CLAUDE.md §2 (Session Continuity).
  A red canary means: fix doc drift BEFORE starting new work.
- TODO (next increment, per CRITIQUE_SYSTEM_DESIGN_20260813 R9):
  pointer checks — every "current/canonical" file reference must exist on
  disk; contract statements about an experiment's status must match the
  experiment's own README/results.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent

# Live claims: the stated count is the current truth and must equal disk.
LIVE_TARGETS = {
    "docs/EXPERIMENT_INDEX.md": r"\*\*Total:\*\*\s+(\d+)\s+experiments",
    "docs/WORKSTATE.md": r"Scorecard:.*?(\d+)\s+experiments\*\*",
    "docs/drafts/manifesto.md": r"\*\*Per [^*]*·\s+(\d+)\s+eksperimen",
    "lines/README.md": r"\*\*All\s+(\d+)\s+local experiments are mapped",
}

# Historical/dated claims: printed for visibility, never block.
HISTORICAL_TARGETS = {
    "docs/L2_STRATEGY.md": r"(\d+)\s+experiments",
    "docs/L3_EXECUTION.md": r"(\d+)-experiment",
    "docs/EVAL.md": r"Across\s+(\d+)\s+experiments",
    "docs/L1_CONSTITUTION.md": r"(\d+)\s+experiments\s+depend",
}


def count_experiment_dirs():
    """Count actual experiment directories."""
    exp_dir = REPO / "experiments"
    if not exp_dir.exists():
        return 0
    dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("E")]
    return len(dirs)


def extract_counts(targets):
    """Extract stated experiment counts from each doc file."""
    results = {}
    for relpath, pattern in targets.items():
        fpath = REPO / relpath
        if not fpath.exists():
            results[relpath] = ("MISSING", None)
            continue
        text = fpath.read_text(encoding="utf-8")
        match = re.search(pattern, text)
        if match:
            results[relpath] = ("OK", int(match.group(1)))
        else:
            results[relpath] = ("NO_MATCH", None)
    return results


def main():
    actual_dirs = count_experiment_dirs()
    live = extract_counts(LIVE_TARGETS)
    historical = extract_counts(HISTORICAL_TARGETS)

    print(f"Experiment directories on disk: {actual_dirs}")
    print(f"{'File':<42} {'Status':<10} {'Count':>7}")
    print("-" * 65)

    failures = []

    for relpath, (status, count) in live.items():
        label = f"{count}" if count is not None else status
        print(f"{relpath:<42} {status:<10} {label:>7}")
        if count is None:
            failures.append(f"{relpath}: pattern no longer matches (NO_MATCH/MISSING)")
        elif count != actual_dirs:
            failures.append(f"{relpath}: states {count}, disk has {actual_dirs}")

    for relpath, (status, count) in historical.items():
        label = f"{count}" if count is not None else status
        mark = " (historical — ignored)" if count else ""
        print(f"{relpath:<42} {status:<10} {label:>7}{mark}")

    print("-" * 65)

    if failures:
        print("SYNC MISMATCH — fix doc drift BEFORE new work:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"SYNC OK: all live docs agree with disk ({actual_dirs}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
