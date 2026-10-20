"""Build submission_jcaa_v0.3.tex from v0.2 with mechanical changes only (round-2 revision, 2026-10-20).

Prose is the PI's (G16); this script changes nothing a reader would call writing:
  1. section signs -> "Section~\\ref{...}" (R1-5), including the two hard-coded numbers;
  2. Figures 1 and 5 -> the v03 files from build_v03_figures.py (R1-7, R1-8, R2-2);
  3. the source-header comments that narrate the revision process -> a neutral header; the unused \\figtodo macro removed.
Run:  python papers/P2_settlement_model/make_v03_mechanical.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "submission_jcaa_v0.3.tex"
# v0.3 is the PI's working file from 2026-10-21 on: never regenerate it over his edits.
if OUT.exists() and "--force" not in sys.argv:
    sys.exit(f"{OUT.name} exists; refusing to overwrite (the PI may be writing in it). Use --force only on a fresh copy.")
src = (HERE / "submission_jcaa_v0.2.tex").read_text(encoding="utf-8")
lines = src.split("\n")
log = []

# --- 3. header: everything above \documentclass is comment narrative ---------------------------------------------
dc = next(i for i, l in enumerate(lines) if l.startswith("\\documentclass"))
assert all(l.startswith("%") or not l.strip() for l in lines[:dc]), "non-comment line above \\documentclass"
header = [
    "% P2 -- Journal of Computer Applications in Archaeology #280 -- v0.3 working file (round-2 revision).",
    "% Base: v0.2 + mechanical changes only (make_v03_mechanical.py; REVISION_R2_WRITING_PACK.md section 7).",
    "% Prose: the PI (gate G16). Triage: REVISION_JCAA_R2_20261019.md.",
    "",
]
log.append(f"header: {dc} comment lines replaced by {len(header)}")
lines = header + lines[dc:]
text = "\n".join(lines)

# \figtodo: defined for placeholder figures in v0.2; drop it if nothing uses it.
uses = len(re.findall(r"\\figtodo\{", text))          # the definition reads "\figtodo}[1]{", so it is not counted
if uses == 0:
    text, n = re.subn(r"% Placeholder for figures not yet built.*?\\end\{center\}\}\n", "", text, flags=re.S)
    log.append(f"\\figtodo macro removed ({n} block; 0 uses)")
else:
    log.append(f"\\figtodo kept: {uses} uses")

# --- 1. section references -------------------------------------------------------------------------------------
# Number the sections in order to resolve the two hard-coded section signs.
nums, sec, sub = {}, 0, 0
body_start = text.index("\\begin{document}")
for m in re.finditer(r"\\(section|subsection)\{([^}]*)\}(?:\s*\\label\{([^}]+)\})?", text[body_start:]):
    if m.group(1) == "section":
        sec += 1; sub = 0; key = f"{sec}"
    else:
        sub += 1; key = f"{sec}.{sub}"
    nums[key] = (m.group(2), m.group(3))
for k in ("3.2", "4"):
    log.append(f"section {k} = {nums.get(k)}")

def hard(m):
    k = m.group(1).rstrip(".")
    title, label = nums[k]
    tail = "." if m.group(1).endswith(".") else ""
    return (f"Section~\\ref{{{label}}}" if label else f"Section~{k}") + tail

text, n_hard = re.subn(r"\\S(\d+(?:\.\d+)?\.?)(?![\d])", hard, text)
text, n_ref = re.subn(r"\\S\\ref\{", r"Section~\\ref{", text)
text, n_sref = re.subn(r"\\S~\\ref\{", r"Section~\\ref{", text)
log.append(f"section signs: {n_ref + n_sref} \\S\\ref -> Section~\\ref; {n_hard} hard-coded")
assert "\\S" not in re.sub(r"%.*", "", text), "a section sign is left in the text"

# --- 2. figures ----------------------------------------------------------------------------------------------------
for old, new in (("fig10_study_area_map.png", "fig01_study_area_v03.png"),
                 ("fig16_priority_map.png", "fig05_priority_map_v03.png")):
    assert text.count(old) == 1, old
    text = text.replace(old, new)
    log.append(f"figure: {old} -> {new}")
# The new Figure 1 carries a right-hand column (inset, legend, scale); at 0.9 text width its labels are too small.
text = text.replace("[width=0.9\\textwidth]{fig01_study_area_v03.png}", "[width=\\textwidth]{fig01_study_area_v03.png}")
text = text.replace("[width=0.85\\textwidth]{fig05_priority_map_v03.png}", "[width=\\textwidth]{fig05_priority_map_v03.png}")
log.append("figures 1 and 5: full text width")

OUT.write_text(text, encoding="utf-8")
print("\n".join(log))
sys.exit(0)
