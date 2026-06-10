"""
reword_triggers_tex.py — Narrow vocabulary pass for LaTeX manuscripts.

Manuscripts use legitimate archaeology domain terms that MUST be preserved:
"fieldwork targets", "targeted excavation/coring/GPR survey", "targeted
fieldwork", "kill-signal territory" (an AUC discontinuation term). So this pass
is deliberately MUCH narrower than reword_triggers.py — it only touches the few
words that read as security metaphor with zero domain meaning:

  adversarial  -> robustness        (always "adversarial regression/test" here)
  smoking gun  -> decisive case      (rhetorical heading)
  hostile      -> critical           (only in %% LaTeX comments)
  kill criteria/criterion -> stop ...(framework-level discontinuation criteria)
  killer       -> decisive           (only in %% comments)

It does NOT touch target/targeted/targeting/kill-signal — those are real terms.
Scope: papers/**/*.tex only.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(r"D:\documents\volcarch-repo")
PAPERS = REPO / "papers"

CI = [
    ("adversarial review", "critical review"),
    ("adversarial reviewer", "critical reviewer"),
    ("adversarial regression", "robustness regression"),
    ("adversarial", "robustness"),
    ("smoking gun", "decisive case"),
    ("smoking-gun", "decisive-case"),
    ("hostile review", "critical review"),
    ("hostile", "critical"),
    ("kill criterion", "stop criterion"),
    ("kill criteria", "stop criteria"),
    ("killer", "decisive"),
]


def smartcase(matched: str, repl: str) -> str:
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
    for pat, repl in CI_COMPILED:
        text, k = pat.subn(lambda m: smartcase(m.group(0), repl), text)
        n += k
    return text, n


def main() -> None:
    total_files = 0
    total_repl = 0
    for path in PAPERS.rglob("*.tex"):
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
    print(f"\n--- {total_repl} replacements across {total_files} .tex files ---")


if __name__ == "__main__":
    main()
