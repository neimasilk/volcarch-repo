"""1) strip the w:date attributes that Word writes as a placeholder (1900-..., hour 29) after
      'remove date and time' -- the attribute is optional, so the file is valid without it;
   2) QA of the tracked file in Word: Accept All must reproduce the revised article text and
      Reject All the converted submitted text (the defining property of a correct comparison).

usage: python strip_dates_and_qa.py TRACKED_RAW.docx BASELINE.docx REVISED.docx OUT.docx
"""
import sys, os, re, zipfile, shutil, difflib, tempfile
import pythoncom
import win32com.client as wc

raw, base, rev, out = [os.path.abspath(a) for a in sys.argv[1:5]]

# 1) strip dates ---------------------------------------------------------------------------------
n_attr = 0
with zipfile.ZipFile(raw) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename.endswith(".xml") and item.filename.startswith("word/"):
            s = data.decode("utf-8")
            s, k = re.subn(r'\s+w:date="[^"]*"', "", s)
            n_attr += k
            s, k = re.subn(r'\s+w16du:dateUtc="[^"]*"', "", s)
            n_attr += k
            data = s.encode("utf-8")
        zout.writestr(item, data)
print("date attributes removed:", n_attr)

# 2) QA ------------------------------------------------------------------------------------------
def norm(t):
    t = t.replace("\x07", " ").replace("\r", " ").replace("\x0b", " ").replace("\xa0", " ")
    t = re.sub(r"[​­]", "", t)
    return re.sub(r"\s+", " ", t).strip()

pythoncom.CoInitialize()
app = wc.DispatchEx("Word.Application")
app.Visible = False
app.DisplayAlerts = 0
tmp = tempfile.mkdtemp()
try:
    def text_of(path):
        d = app.Documents.Open(path, ConfirmConversions=False, ReadOnly=True, AddToRecentFiles=False, Visible=False)
        t = d.Content.Text
        d.Close(False)
        return norm(t)

    t_rev = text_of(rev)
    t_base = text_of(base)

    for mode in ("accept", "reject"):
        cp = os.path.join(tmp, mode + ".docx")
        shutil.copy(out, cp)
        d = app.Documents.Open(cp, ConfirmConversions=False, ReadOnly=False, AddToRecentFiles=False, Visible=False)
        d.TrackRevisions = False
        n = d.Revisions.Count
        if mode == "accept":
            d.Revisions.AcceptAll()
        else:
            d.Revisions.RejectAll()
        t = norm(d.Content.Text)
        d.Close(False)
        target = t_rev if mode == "accept" else t_base
        name = "revised article" if mode == "accept" else "converted submitted text"
        sm = difflib.SequenceMatcher(None, t, target, autojunk=False)
        # quick_ratio first, ratio is O(n^2) on 100k chars -> use word level
        wa, wb = t.split(" "), target.split(" ")
        smw = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
        print("%s-all (%d revisions) vs %s: word-level ratio %.5f; words %d vs %d" % (mode, n, name, smw.ratio(), len(wa), len(wb)))
        shown = 0
        for tag, a1, a2, b1, b2 in smw.get_opcodes():
            if tag != "equal":
                print("   ", tag, "|", " ".join(wa[a1:a2])[:120], "|=>|", " ".join(wb[b1:b2])[:120])
                shown += 1
                if shown >= 12:
                    print("    ..."); break
finally:
    app.Quit(False)
    shutil.rmtree(tmp, ignore_errors=True)
