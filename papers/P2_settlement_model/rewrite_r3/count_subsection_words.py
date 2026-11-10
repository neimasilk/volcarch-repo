# Count words in the three rewritten Results subsections, v0.3 vs v0.4 (tables and figures excluded).
import re
import sys


def subsection_words(path, titles):
    text = open(path, encoding="utf-8").read()
    lines = ["" if l.lstrip().startswith("%") else l for l in text.split("\n")]
    text = "\n".join(lines)
    out = {}
    parts = re.split(r"\\subsection\{([^}]*)\}", text)
    # parts: [pre, title1, body1, title2, body2, ...]
    for i in range(1, len(parts), 2):
        title, body = parts[i], parts[i + 1]
        body = re.split(r"\\subsection\{", body)[0]
        if title in titles:
            body = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
            body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", body)
            body = body.replace("$", " ").replace("~", " ")
            out[title] = len(body.split())
    return out


TITLES = [
    "The reported ladder",
    "The ladder does not survive a common evaluation background",
    "The presence records and a cleaned subset",
]

for path in ("submission_jcaa_v0.3.tex", "submission_jcaa_v0.4.tex"):
    w = subsection_words(path, TITLES)
    total = sum(w.values())
    print(f"{path}: total {total}")
    for t in TITLES:
        print(f"   {w.get(t, 0):5d}  {t[:55]}")
