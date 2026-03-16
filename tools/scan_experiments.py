#!/usr/bin/env python3
"""
Scan all experiment READMEs and generate EXPERIMENT_INDEX.md

Usage: python tools/scan_experiments.py
Output: docs/EXPERIMENT_INDEX.md (regenerated)

This is the SINGLE maintenance tool for the experiment framework.
Run it whenever experiments are added or statuses change.
"""

import os
import re
import json
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
OUTPUT_MD = os.path.join(REPO_ROOT, "docs", "EXPERIMENT_INDEX.md")
OUTPUT_JSON = os.path.join(REPO_ROOT, "docs", "experiment_index.json")


def extract_from_readme(readme_path):
    """Parse a README.md and extract key fields."""
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    info = {}

    # Title: first # heading
    m = re.search(r"^#\s+(.+)", content, re.MULTILINE)
    if m:
        raw_title = m.group(1).strip()
        # Remove E-number prefix if present
        raw_title = re.sub(r"^E\d+[:\s—–-]+\s*", "", raw_title)
        info["title"] = raw_title[:80]

    # Status
    m = re.search(r"\*\*Status:\*\*\s*(.+?)(?:\n|\*\*)", content)
    if m:
        status_raw = m.group(1).strip().rstrip("*")
        # Normalize
        status_upper = status_raw.upper()
        if "SUCCESS" in status_upper:
            info["status"] = "SUCCESS"
        elif "FAILED" in status_upper:
            info["status"] = "FAILED"
        elif "INCONCLUSIVE" in status_upper:
            info["status"] = "INCONCLUSIVE"
        elif "MIXED" in status_upper:
            info["status"] = "MIXED"
        elif "INFO" in status_upper and "NEG" in status_upper:
            info["status"] = "INFO NEG"
        elif "CONDITIONAL" in status_upper:
            info["status"] = "CONDITIONAL"
        elif "PARTIAL" in status_upper:
            info["status"] = "PARTIAL"
        elif "REVISIT" in status_upper:
            info["status"] = "REVISIT"
        elif "IN PROGRESS" in status_upper or "RUNNING" in status_upper:
            info["status"] = "IN PROGRESS"
        else:
            info["status"] = status_raw[:20]

    # Layer
    layers = set()
    for lm in re.finditer(r"L([1-6])", content[:500]):
        layers.add(f"L{lm.group(1)}")
    if layers:
        info["layers"] = sorted(layers)

    # Paper
    papers = set()
    for pm in re.finditer(r"\bP(\d+)\b", content[:500]):
        papers.add(f"P{pm.group(1)}")
    # Also check for D1/D2
    if re.search(r"\bD[12]\b", content[:500]):
        for dm in re.finditer(r"\b(D[12])\b", content[:500]):
            papers.add(dm.group(1))
    if papers:
        info["papers"] = sorted(papers)

    # Date
    m = re.search(r"\*\*Date:\*\*\s*(\d{4}-\d{2}-\d{2})", content)
    if m:
        info["date"] = m.group(1)

    # Key metric: look for p-values, AUC, rho, etc.
    metrics = []
    for pattern in [
        r"(?:AUC|auc)[=:]\s*([\d.]+)",
        r"p[=<]\s*([\d.e-]+)",
        r"(?:rho|ρ)[=:]\s*([+-]?[\d.]+)",
        r"z[=:]\s*([\d.]+)",
        r"F1[=:]\s*([\d.]+)"
    ]:
        m = re.search(pattern, content)
        if m:
            metrics.append(m.group(0))
    if metrics:
        info["key_metric"] = "; ".join(metrics[:2])

    return info


def scan_all_experiments():
    """Scan all experiment directories."""
    experiments = []

    if not os.path.exists(EXPERIMENTS_DIR):
        print(f"ERROR: {EXPERIMENTS_DIR} not found")
        return experiments

    for entry in sorted(os.listdir(EXPERIMENTS_DIR)):
        full_path = os.path.join(EXPERIMENTS_DIR, entry)
        if not os.path.isdir(full_path):
            continue

        # Extract E-number
        m = re.match(r"(E\d+)", entry)
        if not m:
            continue

        eid = m.group(1)
        readme_path = os.path.join(full_path, "README.md")

        exp = {
            "id": eid,
            "dir": entry,
            "has_readme": os.path.exists(readme_path)
        }

        if exp["has_readme"]:
            info = extract_from_readme(readme_path)
            if info:
                exp.update(info)

        experiments.append(exp)

    return experiments


def generate_index(experiments):
    """Generate the markdown index."""
    lines = []
    lines.append("# Experiment Index")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Total:** {len(experiments)} experiments")
    lines.append(f"**Regenerate:** `python tools/scan_experiments.py`")
    lines.append("")

    # Status summary
    from collections import Counter
    statuses = Counter(e.get("status", "UNKNOWN") for e in experiments)
    lines.append("## Status Summary")
    lines.append("")
    for s, c in sorted(statuses.items(), key=lambda x: -x[1]):
        lines.append(f"- **{s}:** {c}")
    lines.append("")

    # Main table
    lines.append("## All Experiments")
    lines.append("")
    lines.append("| ID | Title | Status | Layer | Paper | Key Metric |")
    lines.append("|-----|-------|--------|-------|-------|------------|")

    for e in experiments:
        eid = e["id"]
        title = e.get("title", e["dir"].replace(eid + "_", "").replace("_", " "))[:50]
        status = e.get("status", "NO README" if not e["has_readme"] else "?")
        layers = ",".join(e.get("layers", [""]))
        papers = ",".join(e.get("papers", [""]))
        metric = e.get("key_metric", "")[:40]

        lines.append(f"| {eid} | {title} | {status} | {layers} | {papers} | {metric} |")

    lines.append("")

    # Revisit candidates
    lines.append("## Revisit Candidates")
    lines.append("")
    lines.append("Experiments that failed or were inconclusive but could be revisited with new data/methods.")
    lines.append("")

    revisitable = [e for e in experiments if e.get("status") in
                   ("FAILED", "INCONCLUSIVE", "INFO NEG", "CONDITIONAL", "MIXED", "PARTIAL", "REVISIT")]
    if revisitable:
        lines.append("| ID | Title | Status | Why Revisitable |")
        lines.append("|-----|-------|--------|-----------------|")
        for e in revisitable:
            eid = e["id"]
            title = e.get("title", e["dir"])[:40]
            status = e.get("status", "?")
            lines.append(f"| {eid} | {title} | {status} | *(check README)* |")
    else:
        lines.append("*(none)*")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Auto-generated by `tools/scan_experiments.py`. Do not edit manually — changes will be overwritten.*")

    return "\n".join(lines)


def main():
    print("Scanning experiments...")
    experiments = scan_all_experiments()
    print(f"Found {len(experiments)} experiments")

    # Generate markdown
    md = generate_index(experiments)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Written: {OUTPUT_MD}")

    # Generate JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False)
    print(f"Written: {OUTPUT_JSON}")

    # Stats
    from collections import Counter
    statuses = Counter(e.get("status", "UNKNOWN") for e in experiments)
    print(f"\nStatus distribution:")
    for s, c in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")

    no_readme = [e for e in experiments if not e["has_readme"]]
    if no_readme:
        print(f"\nMISSING READMEs ({len(no_readme)}):")
        for e in no_readme:
            print(f"  {e['dir']}")


if __name__ == "__main__":
    main()
