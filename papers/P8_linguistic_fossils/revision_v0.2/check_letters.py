"""Checks of the two letters against the article file they describe (run after every edit of a letter or the article).

usage: python -X utf8 check_letters.py

1. Every number in the letters is looked up in the revised article (text of the anonymised Word file) and, failing
   that, in the submitted LaTeX source; what is in neither must be on the short list of things that cannot be there
   (dates, the manuscript number, the DOI, section numbers, counts made for the letter). Presence is necessary, not
   sufficient: the claim-by-claim reading is recorded in PACKAGE_R1_20261008.md, section 1.
2. Every "Section n.m" in the letters names a heading of the article; the heading is printed beside it.
3. Words the response says are gone or no longer used are counted in the article text.
4. The 38 reviewer points (R1 opening remark, R1-1..13, R2-1..24) are each answered once, in order.
5. The response (which may go to the reviewers) is checked for names, affiliation, repository and DOI.
Exit code 1 if any check fails.
"""
import io, re, subprocess, sys, zipfile
from pathlib import Path
import docx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).resolve().parent
ART = HERE / "P8_revision_v0.3_anonymous.docx"
TEX = HERE.parent / "draft_v0.1_anonymous.tex"
LETTERS = {"response": HERE / "RESPONSE_TO_REVIEWERS_v0.4.docx", "cover": HERE / "COVER_LETTER_v0.4.docx"}
fail = []


def plain(path):
    return subprocess.run(["pandoc", str(path), "-t", "plain", "--wrap=none"], capture_output=True, check=True).stdout.decode("utf-8")


norm = lambda s: s.replace(" ", " ").replace(" ", " ").replace(",", "")
art, tex = plain(ART), TEX.read_text(encoding="utf-8")
art_n, tex_n = norm(art), norm(tex)

# headings of the article with their numbers (the template's heading styles number themselves)
heads, a, b = {}, 0, 0
for p in docx.Document(str(ART)).paragraphs:
    if p.style.name == "OL heading A-level" and p.text.strip():
        a += 1; b = 0; heads[str(a)] = p.text.strip()
    elif p.style.name == "OL heading B-level" and p.text.strip():
        b += 1; heads[f"{a}.{b}"] = p.text.strip()

# 1 ------------------------------------------------------------------------------------------- numbers
# not in either text by their nature: manuscript number and dates; the DOI; the reviewer's own figure (30.1, quoted from
# the report); 356 = the sum of the residual column of the submitted Table 1 (26+49+32+68+67+114); pages 491/492 of Mills
ALLOW = {"03", "2026", "7", "8", "10.5281", "23202245", "30.1", "356", "492", "1989", "2001"}
tok = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")
print("1. numbers")
for name, path in LETTERS.items():
    text = plain(path)
    n = miss = 0
    for para in [p for p in re.split(r"\n\s*\n", text) if p.strip()]:
        para_n = norm(para)
        for m in tok.finditer(para_n):
            t = m.group(0).rstrip(",")
            n += 1
            pat = r"(?<![\d.])" + re.escape(t) + r"(?![\d])"
            where = "article" if re.search(pat, art_n) else "submitted" if re.search(pat, tex_n) else None
            pre = para_n[max(0, m.start() - 9): m.start()]
            if where is None and re.search(r"Sections? (\d\.\d and )?$", pre):
                where = "section number"
            if where is None and t not in ALLOW:
                miss += 1
                print(f"   NOT FOUND [{t}] in {name}: …{para_n[max(0, m.start() - 60): m.end() + 40]}…")
    print(f"   {name}: {n} numeric tokens, {miss} unexplained")
    if miss:
        fail.append(f"numbers in {name}")

# 2 ------------------------------------------------------------------------------------------- sections
print("2. section references")
for name, path in LETTERS.items():
    text = plain(path)
    text = re.sub(r"(?i)submitted Section \d(?:\.\d)?", "", text)   # a section of the submitted version, said so in the letter
    refs = sorted(set(re.findall(r"Sections? (\d(?:\.\d)?)", text)) | set(re.findall(r"Sections? \d(?:\.\d)? and (\d(?:\.\d)?)", text)))
    bad = [r for r in refs if r not in heads]
    print(f"   {name}: " + "; ".join(f"{r} = {heads.get(r, '??')}" for r in refs))
    if bad:
        fail.append(f"section references in {name}: {bad}")

# 3 ------------------------------------------------------------------------------------------- words said to be gone
print("3. words the response says are gone from the article (count in the article text)")
GONE = ["robust", "ablat", "parallel", "fricative", "Mongondow", "false positive", "false-positive", "unlabeled", "false alarm",
        "primary subgroup", "Swadesh-100", "DBSCAN", "E022", "E027", "E028", "beeswarm", "under-documented", "fingerprint of"]
for w in GONE:
    k = len(re.findall(re.escape(w), art, flags=re.I))
    print(f"   {w!r}: {k}")
    if k:
        fail.append(f"'{w}' occurs {k}x in the article")
for w, expect in (("positive", 3), ("under-documentation", 2), ("fingerprint", 1), ("candidate", 2)):
    k = len(re.findall(re.escape(w), art, flags=re.I))
    print(f"   {w!r}: {k} (expected {expect}: see the response, R1-12 / R1-7 / R1-10, and Sections 3.3, 3.5)")
    if k != expect:
        fail.append(f"'{w}' occurs {k}x, expected {expect}")

# 4 ------------------------------------------------------------------------------------------- reviewer points
print("4. reviewer points")
resp = plain(LETTERS["response"])
labels = ["Opening remark"] + [f"R1-{i}" for i in range(1, 14)] + [f"R2-{i}" for i in range(1, 25)]
pos = []
for lab in labels:
    hits = [m.start() for m in re.finditer(r"(?m)^\s*-\s+" + re.escape(lab) + r" —", resp)]
    if len(hits) != 1:
        fail.append(f"label {lab}: {len(hits)} bullets")
    pos.append(hits[0] if hits else -1)
print(f"   {sum(p >= 0 for p in pos)} of {len(labels)} points have exactly one bullet; in order: {pos == sorted(pos)}")
if pos != sorted(pos):
    fail.append("reviewer points out of order")
partc = resp.split("Part C.")[1]
print("   Part C bullets:", len(re.findall(r"(?m)^\s*-\s+\S", partc)))

# 5 ------------------------------------------------------------------------------------------- anonymity
print("5. anonymity of the response")
r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "tracked_build" / "check_anonymity.py"), str(LETTERS["response"])],
                   capture_output=True)
print("   " + r.stdout.decode("utf-8", errors="replace").strip().replace("\n", "\n   "))
if r.returncode:
    fail.append("anonymity of the response")

print("\nRESULT:", "all checks passed" if not fail else "FAILED: " + "; ".join(fail))
sys.exit(1 if fail else 0)
