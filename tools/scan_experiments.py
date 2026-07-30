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

# ---------------------------------------------------------------------------
# LINE ASSIGNMENT (lines/ navigation layer — see lines/README.md)
#
# Explicit and auditable ON PURPOSE. A regex/keyword guess would silently
# mis-file experiments, and a stale mapping is exactly the failure this layer
# exists to prevent. An experiment MAY belong to several lines — the first
# entry is its primary line. Never partition experiments/ on disk.
#
# WHEN YOU ADD AN EXPERIMENT: add it here too. The script exits non-zero-ish
# loud (prints an UNMAPPED block) if you forget.
# ---------------------------------------------------------------------------
LINES = {
    "01_spatial":       "Predictive modelling & site distribution",
    "02_taphonomy":     "Burial, erosion, exposure",
    "03_paleoenv":      "Paleo-environmental falsification",
    "04_language_text": "Language & text",
    "05_archival_nlp":  "Colonial archives & NLP",
    "06_thesis":        "Original question / synthesis",
    "07_career":        "Career & exposure (no experiments)",
}
EXTERNAL = "external:volcarch-genetics"

LINE_MAP = {
    # --- 01 spatial ---------------------------------------------------------
    "E003": ["01_spatial"], "E004": ["01_spatial"], "E005": ["01_spatial"],
    "E006": ["01_spatial"], "E007": ["01_spatial"], "E008": ["01_spatial"],
    "E009": ["01_spatial"], "E010": ["01_spatial"], "E011": ["01_spatial"],
    "E012": ["01_spatial"], "E013": ["01_spatial"], "E014": ["01_spatial"],
    "E015": ["01_spatial"], "E016": ["01_spatial"], "E019": ["01_spatial"],
    "E031": ["01_spatial"], "E059": ["01_spatial"], "E065": ["01_spatial"],
    "E066": ["01_spatial"], "E076": ["01_spatial"], "E080": ["01_spatial"],
    "E097": ["01_spatial"], "E100": ["01_spatial"], "E103": ["01_spatial"],
    "E104": ["01_spatial"], "E106": ["01_spatial"], "E108": ["01_spatial"],
    "E110": ["01_spatial"], "E115": ["01_spatial"], "E116": ["01_spatial"],
    "E118": ["01_spatial"], "E120": ["01_spatial"], "E121": ["01_spatial"],
    "E122": ["01_spatial"], "E124": ["01_spatial"], "E129": ["01_spatial"],
    "E139": ["01_spatial"], "E151": ["01_spatial"], "E152": ["01_spatial"],
    "E153": ["01_spatial"], "E155": ["01_spatial"], "E159": ["01_spatial"],
    "E163": ["01_spatial"], "E167": ["01_spatial"], "E171": ["01_spatial"],
    "E172": ["01_spatial"], "E175": ["01_spatial"], "E176": ["01_spatial"],
    "E183": ["01_spatial"], "E184": ["01_spatial"], "E185": ["01_spatial"],
    "E187": ["01_spatial"], "E189": ["01_spatial"], "E190": ["01_spatial"],
    "E191": ["01_spatial"], "E192": ["01_spatial"], "E194": ["01_spatial"],
    "E196": ["01_spatial"], "E202": ["01_spatial"], "E209": ["01_spatial"],
    "E210": ["01_spatial"], "E217": ["01_spatial"], "E218": ["01_spatial"],
    "E219": ["01_spatial"], "E220": ["01_spatial"], "E221": ["01_spatial"],
    "E222": ["01_spatial"], "E223": ["01_spatial"],
    # --- 02 taphonomy -------------------------------------------------------
    "E002": ["02_taphonomy"], "E017": ["02_taphonomy"], "E018": ["02_taphonomy"],
    "E020": ["02_taphonomy"], "E024": ["02_taphonomy"], "E052": ["02_taphonomy"],
    "E069": ["02_taphonomy"], "E075": ["02_taphonomy"], "E081": ["02_taphonomy"],
    "E083": ["02_taphonomy"], "E086": ["02_taphonomy"], "E092": ["02_taphonomy"],
    "E101": ["02_taphonomy"], "E117": ["02_taphonomy"], "E123": ["02_taphonomy"],
    "E132": ["02_taphonomy"], "E135": ["02_taphonomy"], "E137": ["02_taphonomy"],
    "E138": ["02_taphonomy"], "E140": ["02_taphonomy"], "E148": ["02_taphonomy"],
    "E156": ["02_taphonomy"], "E157": ["02_taphonomy"], "E161": ["02_taphonomy"],
    "E170": ["02_taphonomy"], "E173": ["02_taphonomy"], "E177": ["02_taphonomy"],
    "E178": ["02_taphonomy"], "E188": ["02_taphonomy"], "E193": ["02_taphonomy"],
    "E201": ["02_taphonomy"], "E213": ["02_taphonomy"],
    # --- 03 paleoenv --------------------------------------------------------
    "E214": ["03_paleoenv"], "E215": ["03_paleoenv"], "E216": ["03_paleoenv"],
    # --- 04 language & text -------------------------------------------------
    "E022": ["04_language_text"], "E023": ["04_language_text"],
    "E025": ["04_language_text"], "E026": ["04_language_text"],
    "E027": ["04_language_text"], "E028": ["04_language_text"],
    "E029": ["04_language_text"], "E030": ["04_language_text"],
    "E032": ["04_language_text"], "E033": ["04_language_text"],
    "E034": ["04_language_text"], "E035": ["04_language_text"],
    "E036": ["04_language_text"], "E037": ["04_language_text"],
    "E038": ["04_language_text"], "E039": ["04_language_text"],
    "E040": ["04_language_text"], "E041": ["04_language_text"],
    "E042": ["04_language_text"], "E043": ["04_language_text"],
    "E044": ["04_language_text"], "E049": ["04_language_text"],
    "E050": ["04_language_text"], "E051": ["04_language_text"],
    "E054": ["04_language_text"], "E056": ["04_language_text"],
    "E057": ["04_language_text"], "E058": ["04_language_text"],
    "E061": ["04_language_text"], "E063": ["04_language_text"],
    "E067": ["04_language_text"], "E074": ["04_language_text"],
    "E088": ["04_language_text"], "E089": ["04_language_text"],
    "E090": ["04_language_text"], "E094": ["04_language_text"],
    "E095": ["04_language_text"], "E096": ["04_language_text"],
    "E102": ["04_language_text"], "E105": ["04_language_text"],
    "E111": ["04_language_text"], "E112": ["04_language_text"],
    "E113": ["04_language_text"], "E114": ["04_language_text"],
    "E130": ["04_language_text"], "E131": ["04_language_text"],
    "E134": ["04_language_text"], "E146": ["04_language_text"],
    "E147": ["04_language_text"], "E150": ["04_language_text"],
    "E160": ["04_language_text"], "E165": ["04_language_text"],
    "E169": ["04_language_text"], "E181": ["04_language_text"],
    "E186": ["04_language_text"], "E198": ["04_language_text"],
    "E205": ["04_language_text"], "E208": ["04_language_text"],
    # --- 05 archival NLP ----------------------------------------------------
    "E070": ["05_archival_nlp"], "E091": ["05_archival_nlp"],
    "E093": ["05_archival_nlp"], "E098": ["05_archival_nlp"],
    "E125": ["05_archival_nlp"], "E141": ["05_archival_nlp"],
    "E142": ["05_archival_nlp"], "E143": ["05_archival_nlp"],
    "E200": ["05_archival_nlp"], "E206": ["05_archival_nlp"],
    "E207": ["05_archival_nlp"], "E211": ["05_archival_nlp"],
    # --- 06 thesis / synthesis ---------------------------------------------
    "E048": ["06_thesis"], "E055": ["06_thesis"], "E060": ["06_thesis"],
    "E062": ["06_thesis"], "E064": ["06_thesis"], "E068": ["06_thesis"],
    "E071": ["06_thesis"], "E073": ["06_thesis"], "E078": ["06_thesis"],
    "E079": ["06_thesis"], "E099": ["06_thesis"], "E119": ["06_thesis"],
    "E127": ["06_thesis"], "E133": ["06_thesis"], "E136": ["06_thesis"],
    "E144": ["06_thesis"], "E145": ["06_thesis"], "E149": ["06_thesis"],
    "E154": ["06_thesis"], "E158": ["06_thesis"], "E162": ["06_thesis"],
    "E168": ["06_thesis"], "E174": ["06_thesis"], "E199": ["06_thesis"],
    # --- cross-line (primary first) ----------------------------------------
    "E001": ["02_taphonomy", "01_spatial"],
    "E082": ["04_language_text", "01_spatial"],
    "E084": ["02_taphonomy", "01_spatial"],
    "E085": ["04_language_text", "02_taphonomy"],
    "E087": ["04_language_text", "02_taphonomy"],
    "E107": ["04_language_text", "02_taphonomy"],
    "E109": ["01_spatial", "02_taphonomy"],
    "E126": ["02_taphonomy", "01_spatial"],
    "E128": ["02_taphonomy", "05_archival_nlp"],
    "E164": ["02_taphonomy", "06_thesis"],
    "E166": ["02_taphonomy", "01_spatial"],
    "E179": ["06_thesis", "01_spatial"],
    "E182": ["01_spatial", "02_taphonomy"],
    "E195": ["02_taphonomy", "01_spatial"],
    "E197": ["02_taphonomy", "05_archival_nlp"],
    "E204": ["02_taphonomy", "06_thesis"],
    # --- external (companion repo; no local directory) ----------------------
    "E053": [EXTERNAL],
    "E203": [EXTERNAL],
}


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

        # Line assignment (lines/ navigation layer). Primary line first.
        exp["lines"] = LINE_MAP.get(eid, [])

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

    # Per-line breakdown (lines/ navigation layer)
    lines.append("## By Line of Inquiry")
    lines.append("")
    lines.append("Navigation layer: `lines/<name>/`. See `lines/README.md`. An experiment may serve")
    lines.append("several lines — it is listed under each, and its **primary** line is listed first in")
    lines.append("the table below. `experiments/` itself stays flat and shared; it is never partitioned.")
    lines.append("")
    for lname, ldesc in LINES.items():
        members = [e["id"] for e in experiments if lname in e.get("lines", [])]
        primary = [e["id"] for e in experiments if e.get("lines", [None])[:1] == [lname]]
        lines.append(f"### `{lname}` — {ldesc}")
        lines.append("")
        if members:
            lines.append(f"**{len(members)}** experiments ({len(primary)} primary): "
                         + " · ".join(members))
        else:
            lines.append("*(no experiments — this line's work is not experimental)*")
        lines.append("")

    ext = [e["id"] for e in experiments if EXTERNAL in e.get("lines", [])]
    lines.append(f"### `{EXTERNAL}`")
    lines.append("")
    lines.append("Canonical in the companion repo `D:\\documents\\volcarch-genetics` — **no local")
    lines.append("directory**, by design (see `docs/COMPANION_REPOS.md`). Cite as external evidence.")
    lines.append("")
    lines.append("E053 · E203" if not ext else " · ".join(ext))
    lines.append("")

    unmapped = [e["id"] for e in experiments if not e.get("lines")]
    if unmapped:
        lines.append("### ⚠ UNMAPPED — add to `LINE_MAP` in `tools/scan_experiments.py`")
        lines.append("")
        lines.append(" · ".join(unmapped))
        lines.append("")

    # Main table
    lines.append("## All Experiments")
    lines.append("")
    lines.append("| ID | Title | Status | Line | Layer | Paper | Key Metric |")
    lines.append("|-----|-------|--------|------|-------|-------|------------|")

    for e in experiments:
        eid = e["id"]
        title = e.get("title", e["dir"].replace(eid + "_", "").replace("_", " "))[:50]
        status = e.get("status", "NO README" if not e["has_readme"] else "?")
        elines = ",".join(l.replace("external:", "ext:") for l in e.get("lines", [])) or "**?**"
        layers = ",".join(e.get("layers", [""]))
        papers = ",".join(e.get("papers", [""]))
        metric = e.get("key_metric", "")[:40]

        lines.append(f"| {eid} | {title} | {status} | {elines} | {layers} | {papers} | {metric} |")

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

    # Line coverage — the check that keeps lines/ from going stale
    print("\nLine coverage:")
    for lname in LINES:
        n = sum(1 for e in experiments if lname in e.get("lines", []))
        print(f"  {lname}: {n}")

    unmapped = [e["id"] for e in experiments if not e.get("lines")]
    if unmapped:
        print(f"\n*** UNMAPPED ({len(unmapped)}) — add these to LINE_MAP in this file: ***")
        print("  " + " ".join(unmapped))
    else:
        print(f"\nAll {len(experiments)} local experiments mapped to a line. OK")

    # Stale entries: mapped IDs with no directory (expected only for EXTERNAL)
    local_ids = {e["id"] for e in experiments}
    ghosts = [k for k, v in LINE_MAP.items()
              if k not in local_ids and EXTERNAL not in v]
    if ghosts:
        print(f"\nWARNING: LINE_MAP references IDs with no directory: {' '.join(ghosts)}")


if __name__ == "__main__":
    main()
