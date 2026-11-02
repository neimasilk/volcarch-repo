"""Word count for the JCAA limit (11,500 incl. references), by the method that produced the 26 Oct figures.

Body = Introduction .. end of Conclusions, floats (figure/table) removed, LaTeX commands stripped.
References = words in references_jcaa.tex.
    python papers/P2_settlement_model/tools_count_words.py [FILE.tex]
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
tex = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "submission_jcaa_v0.3.tex"
src = tex.read_text(encoding="utf-8")

body = src.split(r"\section{Introduction}")[1].split(r"\section*{Supplementary Materials}")[0]


def clean(text):
    text = re.sub(r"(?m)(?<!\\)%.*$", "", text)                       # comments
    text = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", " ", text, flags=re.S)
    text = re.sub(r"\\(label|ref|cite[a-z]*|includegraphics)(\[[^\]]*\])?(\{[^}]*\})*", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)          # remaining commands
    text = re.sub(r"[${}]", " ", text)
    text = text.replace("~", " ").replace("--", "-")
    return text


body_words = len(clean(body).split())
refs = len(clean((HERE / "references_jcaa.tex").read_text(encoding="utf-8")).split())
print(f"{tex.name}: body {body_words:,} + references {refs:,} = {body_words + refs:,}  (limit 11,500)")
