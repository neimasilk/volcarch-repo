"""Anonymity check of a Word file that goes to the reviewers: every XML part of the .docx is searched for the
authors' names, affiliation, e-mail, account and repository names and the data DOI; the document properties, the
revision authors and any date attribute are printed. Exit code 1 if anything is found.

usage: python -X utf8 check_anonymity.py FILE.docx [FILE.docx ...]
"""
import re, sys, zipfile

NEEDLES = ["amien", "mukhlis", "gunawan", "frendi", "bhinneka", "ubhinus", "stiki", "neimasilk", "neima", "volcarch",
           "github.com", "zenodo.2", "10.5281", "orcid", "malang"]
bad_total = 0
for path in sys.argv[1:]:
    z = zipfile.ZipFile(path)
    hits, authors, dates = {}, {}, 0
    for name in z.namelist():
        if not (name.endswith(".xml") or name.endswith(".rels")):
            continue
        s = z.read(name).decode("utf-8", errors="replace")
        low = s.lower()
        for n in NEEDLES:
            k = low.count(n)
            if k:
                hits[(name, n)] = k
        for a in re.findall(r'w:author="([^"]*)"', s):
            authors[a] = authors.get(a, 0) + 1
        dates += len(re.findall(r'w:date="|w16du:dateUtc="', s))
    core = z.read("docProps/core.xml").decode("utf-8") if "docProps/core.xml" in z.namelist() else ""
    app = z.read("docProps/app.xml").decode("utf-8") if "docProps/app.xml" in z.namelist() else ""
    props = {k: (re.search(rf"<{k}[^>]*>(.*?)</{k}>", core, re.S).group(1) if re.search(rf"<{k}[^>]*>(.*?)</{k}>", core, re.S) else "")
             for k in ("dc:creator", "cp:lastModifiedBy", "dc:title", "dc:description", "cp:keywords", "dc:subject")}
    company = re.search(r"<Company>(.*?)</Company>", app, re.S)
    custom = "docProps/custom.xml" in z.namelist() and "<property" in z.read("docProps/custom.xml").decode("utf-8", errors="replace")
    ok = not hits and set(authors) <= {"Author", "Authors"} and dates == 0 and not props["dc:creator"] and not props["cp:lastModifiedBy"] \
        and not custom
    # app.xml "Company" is not a failure by itself: the journal's own template carries "Linguistics RSPAS ANU" there
    # (it is in ol_base.docx); a company string naming the authors would be caught by the needles above.
    bad_total += 0 if ok else 1
    print(("OK   " if ok else "FAIL ") + path.replace("\\", "/").split("/")[-1])
    print("   needles found:", hits or "none")
    print("   revision authors:", authors or "none", "| date attributes:", dates)
    print("   properties:", props, "| company:", company.group(1) if company else "", "| custom properties:", custom)
sys.exit(1 if bad_total else 0)
