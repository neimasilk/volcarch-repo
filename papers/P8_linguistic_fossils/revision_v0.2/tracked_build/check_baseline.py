import re, sys, docx, fitz
base = docx.Document(sys.argv[1])
paras = [p.text for p in base.paragraphs if p.text.strip() and not re.match(r"^[A-Z][^.]{2,40}, [A-Z][^.]*\. (?:19|20)\d\d", p.text)]
cells = []
for t in base.tables:
    for r in t.rows:
        for c in r.cells: cells.append(c.text)
print("paragraphs", len(paras), "tables", len(base.tables), "cells", len(cells))
pdf = fitz.open(r"D:\documents\volcarch-repo\papers\P8_linguistic_fossils\draft_v0.1_anonymous.pdf")
pt = " ".join(pg.get_text() for pg in pdf)
def norm(s):
    s = s.replace("\u2019","'").replace("\u2018","'").replace("\u201c",'"').replace("\u201d",'"').replace("\u2013","-").replace("\u2014","-").replace("\u00ad","").replace("-\n","")
    return re.sub(r"[^a-z0-9]+"," ",s.lower()).strip()
P = norm(pt)
words = P.split()
sh = set(" ".join(words[i:i+6]) for i in range(len(words)-5))
miss = []
tot = 0; hit = 0
for t in paras:
    w = norm(t).split()
    for i in range(0, max(0,len(w)-5)):
        tot += 1
        if " ".join(w[i:i+6]) in sh: hit += 1
        else: miss.append(t[:100])
print("6-gram coverage of baseline in submitted PDF: %d/%d = %.3f" % (hit, tot, hit/tot))
from collections import Counter
for t,c in Counter(miss).most_common(25): print(c, "|", t)
# citation / macro leftovers
alltext = "\n".join(paras)
for pat in [r"\[@", r"\textipa", r"\[a-zA-Z]+\{", r"\$", r"\?\?", r"\(\?\)", r"\{\}"]:
    m = re.findall(pat, alltext)
    print(pat, len(m))

