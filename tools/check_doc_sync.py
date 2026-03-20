#!/usr/bin/env python3
"""
Check that experiment counts are consistent across all documentation files.
Solves the B3 structural risk: document drift when experiment counts are
manually updated in multiple places.

Usage: python tools/check_doc_sync.py
Exit code: 0 if consistent, 1 if mismatches found
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent

# Files that contain experiment counts and the patterns to match
TARGETS = {
    "docs/EXPERIMENT_INDEX.md": r"\*\*Total:\*\*\s+(\d+)\s+experiments",
    "docs/WORKSTATE.md": r"\*\*(\d+)\s+experiments\*\*",
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


def extract_counts():
    """Extract stated experiment counts from each doc file."""
    results = {}
    for relpath, pattern in TARGETS.items():
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
    counts = extract_counts()

    print(f"Experiment directories on disk: {actual_dirs}")
    print(f"{'File':<40} {'Status':<12} {'Count'}")
    print("-" * 65)

    values = set()
    issues = []

    for relpath, (status, count) in counts.items():
        label = f"{count}" if count else status
        print(f"{relpath:<40} {status:<12} {label}")
        if count is not None:
            values.add(count)

    print("-" * 65)

    if len(values) == 0:
        print("ERROR: No counts found in any file.")
        return 1
    elif len(values) == 1:
        val = values.pop()
        print(f"SYNC OK: All docs agree on {val} experiments.")
        return 0
    else:
        print(f"SYNC MISMATCH: Found different counts: {sorted(values)}")
        print("Run document sync to fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
