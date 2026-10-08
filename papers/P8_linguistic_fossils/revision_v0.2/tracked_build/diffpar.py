import re, sys, docx, fitz, difflib
base = docx.Document(sys.argv[1])
pdf = fitz.open(r"D:\documents\volcarch-repo\papers\P8_linguistic_fossils\draft_v0.1_anonymous.pdf")
pt = " ".join(pg.get_text() for pg in pdf)
def norm(s):
    s = s.replace("\u2019","'").replace("\u2018","'").replace("\u201c",'"').replace("\u201d",'"').replace("\u2013","-").replace("\u2014","-").replace("\u00ad","").replace("\u00a0"," ")
    s = re.sub(r"(\w)-\s+(\w)", r"\1-\2", s)   # line-break hyphen
    return re.sub(r"\s+"," ",s).strip()
P = norm(pt)
for key in sys.argv[2:]:
    for p in base.paragraphs:
        if p.text.startswith(key):
            t = norm(p.text)
            # locate by first 40 chars of paragraph in PDF
            i = P.find(t[:40])
            seg = P[i:i+len(t)+60] if i>=0 else "(start not found)"
            sm = difflib.SequenceMatcher(None, t, seg, autojunk=False)
            print("== ", key, "ratio %.3f" % sm.ratio())
            for tag,a1,a2,b1,b2 in sm.get_opcodes():
                if tag!="equal" and (a2-a1>2 or b2-b1>2):
                    print("  ", tag, repr(t[max(0,a1-15):a2+15]), "->", repr(seg[max(0,b1-15):b2+15]))
            break
