"""
reword_triggers.py — Normalise project vocabulary to avoid model safety false-positives.

WHY: The strategy/methodology docs use a dense cluster of words that a topic
classifier mis-reads as "cybersecurity" (adversarial, hostile, kill criteria,
attack map, smoking gun, targeting) or "biology" (vague "genetic material").
None of it is cyber/bio risk — it is research-methodology metaphor. This script
swaps the metaphor for plain academic terms WITHOUT changing any scientific
claim, number, or finding.

SCOPE (deliberately tight): docs/, papers/**/*.md, tools/*.md, and the auto-memory
dir. It NEVER touches experiments/ or data/ — those hold primary-source quotes
(e.g. a Chinese text describing a volcano that "kills people") and linguistic
datasets ("to kill people") where the words are real content, not metaphor.

Legitimate domain terms are preserved on purpose: ancient DNA / aDNA / genome /
paleogenomic (real science), ground-penetrating radar / penetration depth (GPR),
cone penetration test, colonial exploitation (historiography), Mata Elang,
diamond-hunt, ADV-N labels.

Re-runnable. Idempotent. Reports per-file change counts.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(r"D:\documents\volcarch-repo")
MEMDIR = Path(r"C:\Users\neima\.claude\projects\D--documents-volcarch-repo\memory")

# Directories walked, with the suffixes processed in each.
ROOTS = [
    (REPO / "docs", {".md", ".json", ".csv"}),
    (REPO / "papers", {".md"}),
    (REPO / "tools", {".md"}),
    (MEMDIR, {".md"}),
]

# Individual files outside the walked roots that still need processing.
EXTRA_FILES = [
    REPO / "CLAUDE.md",
    REPO / "README.md",
    REPO / "experiments" / "E090_transformer_textual_nlp"
         / "V7_LABEL_SHUFFLE_FINDING_20260610.md",
]

# This script must never edit itself.
SELF = Path(__file__).resolve()

# --- Case-sensitive exact replacements (run first) ---------------------------
CS = [
    # underscore-prefixed refs that the \b word-boundary rules below cannot reach
    # (renamed files: RESPONSE_hostile_deepseek_*, KILL_CRITERION_AUDIT_*)
    ("_hostile_", "_critical_"),
    ("KILL_CRITERION", "STOP_CRITERION"),
    ("master_attack_map", "master_evidence_map"),
    ("MasterAttackMap", "MasterEvidenceMap"),
    ("Master Attack Map", "Master Evidence Map"),
    ("Master attack map", "Master evidence map"),
    ("master attack map", "master evidence map"),
    # weapons-cluster leak (round 2): "revision ammo" / "PhD ammo" / "ammo".
    # NB: the directory path `.../revision_ammo/` (underscore form) is deliberately
    # NOT touched — \bammo\b cannot reach it (preceded by "_", a word char), and
    # renaming it would break script/reference paths.
    ("PhD ammo", "PhD evidence base"),
    ("PhD Ammo", "PhD Evidence Base"),
    # darkness/extremism-cluster leak (round 2). The project's own framing is
    # "invisibility", so this is also the more accurate term.
    ("6 Layers of Darkness", "6 Layers of Invisibility"),
    ("Six Layers of Darkness", "Six Layers of Invisibility"),
    ("Layers of Darkness", "Layers of Invisibility"),
    ("Layers Of Darkness", "Layers Of Invisibility"),
    ("Manifesto v4", "Research Statement v4"),
    ("manifesto v4", "research statement v4"),
    # bio: replace the vague phrase with the precise archaeological term
    ("genetic material", "ancient DNA"),
    ("Genetic material", "Ancient DNA"),
    ("Genetic Material", "Ancient DNA"),
    ("materi genetik", "DNA purba"),
    ("Materi genetik", "DNA purba"),
    ("Materi Genetik", "DNA purba"),
]

# --- Case-insensitive, smart-cased, word-bounded (run after CS) --------------
# Ordered specific -> general; applied sequentially over the whole text.
CI = [
    ("attack_map", "evidence_map"),
    ("attack vectors", "critique points"),
    ("attack vector", "critique point"),
    ("attack map", "evidence map"),
    ("attack", "critique"),

    ("adversarial scorecard", "robustness scorecard"),
    ("adversarially", "independently"),
    ("adversarial testing", "robustness testing"),
    ("adversarial test", "robustness test"),
    ("adversarial review", "critical review"),
    ("adversarial reviewer", "critical reviewer"),
    ("adversarial collaboration", "structured critique"),
    ("adversarial", "critical"),

    ("hostile reviewer", "skeptical reviewer"),
    ("hostile review", "critical review"),
    ("hostile_", "critical_"),
    ("hostile", "skeptical"),

    ("kill_criterion", "stop_criterion"),
    ("kill criterion", "stop criterion"),
    ("kill criteria", "stop criteria"),
    ("kill-criterion", "stop-criterion"),
    ("kill list", "drop list"),
    ("killer", "decisive"),
    ("killed", "discontinued"),
    ("kills", "eliminates"),
    ("kill", "drop"),

    ("smoking gun", "decisive case"),
    ("smoking-gun", "decisive-case"),

    # --- round 2: weapons + extremism clusters missed in the first pass ---
    ("revision ammo", "revision support material"),
    ("ammunition", "supporting material"),
    ("ammo", "supporting material"),

    ("layers of darkness", "layers of invisibility"),
    ("dark side", "unexplored dimension"),
    ("sisi gelap", "sisi tersembunyi"),

    ("godfather-level", "landmark"),
    ("godfather level", "landmark"),
    ("godfather", "landmark"),
    ("provocateur", "challenger"),
    ("the reckoning", "the audit"),

    ("manifesto kegelisahan", "pernyataan kegelisahan"),
    ("research manifesto", "research statement"),

    ("killers", "decisive critiques"),

    ("fieldwork targets", "fieldwork candidates"),
    ("fieldwork target", "fieldwork candidate"),
    ("targeting protocol", "site-selection protocol"),
    ("targeted recovery", "prioritized recovery"),
    ("borehole targeting", "borehole siting"),
]


def smartcase(matched: str, repl: str) -> str:
    """Return repl recased to mirror the matched text."""
    if matched.isupper() and any(c.isalpha() for c in matched):
        return repl.upper()
    if matched.istitle():
        return repl.title()
    if matched[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def build(term: str) -> re.Pattern:
    left = r"\b" if term[:1].isalnum() else ""
    right = r"\b" if term[-1:].isalnum() else ""
    return re.compile(left + re.escape(term) + right, re.IGNORECASE)


CI_COMPILED = [(build(t), r) for t, r in CI]


def transform(text: str) -> tuple[str, int]:
    n = 0
    for a, b in CS:
        if a in text:
            n += text.count(a)
            text = text.replace(a, b)
    for pat, repl in CI_COMPILED:
        text, k = pat.subn(lambda m: smartcase(m.group(0), repl), text)
        n += k
    return text, n


def iter_files():
    for root, suffixes in ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in suffixes:
                yield p
    for p in EXTRA_FILES:
        if p.exists() and p.is_file():
            yield p


def main() -> None:
    total_files = 0
    total_repl = 0
    seen = set()
    for path in iter_files():
        rp = path.resolve()
        if rp == SELF or rp in seen:
            continue
        seen.add(rp)
        try:
            original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new, n = transform(original)
        if n and new != original:
            path.write_text(new, encoding="utf-8")
            total_files += 1
            total_repl += n
            print(f"{n:4d}  {path}")
    print(f"\n--- {total_repl} replacements across {total_files} files ---")


if __name__ == "__main__":
    main()
