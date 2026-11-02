"""Word count of the v0.3 abstract, per sentence (JCAA limit 300; it is over).

    python papers/P2_settlement_model/tools_count_abstract.py
"""
import re
from pathlib import Path

TEX = Path(__file__).resolve().parent / "submission_jcaa_v0.3.tex"
src = TEX.read_text(encoding="utf-8")
body = src.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
body = re.sub(r"(?m)%.*$", "", body)                      # comments
body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body)  # commands (\noindent, \textbf)
body = re.sub(r"[${}]", "", body)
body = body.replace("~", " ").replace("--", "-")

words = body.split()
print(f"TOTAL words: {len(words)}  (limit 300; over by {len(words) - 300})")
print()
for i, s in enumerate(re.split(r"(?<=[.?])\s+", " ".join(body.split())), 1):
    if s.strip():
        print(f"{len(s.split()):4d}  {s[:110]}")
