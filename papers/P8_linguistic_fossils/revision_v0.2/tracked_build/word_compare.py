"""Compare the converted SUBMITTED text (baseline) with the revised anonymous article through Word,
producing a tracked-changes document whose revisions carry a neutral author label.

usage: python word_compare.py BASELINE.docx REVISED.docx OUT.docx

Anonymity measures (the editor asked for an anonymised revision):
  * Compare(AuthorName="Authors", RemovePersonalInformation=True)
  * Application.UserName / UserInitials set to neutral values for the session
  * the result is saved with RemovePersonalInformation on and document properties cleared
Verification of the saved file is a separate step (zip inspection), not done here.
"""
import sys, os, time
import pythoncom
import win32com.client as wc

base, rev, out = [os.path.abspath(a) for a in sys.argv[1:4]]
pythoncom.CoInitialize()
app = wc.DispatchEx("Word.Application")
app.Visible = False
app.DisplayAlerts = 0  # wdAlertsNone
old_user, old_init = app.UserName, app.UserInitials
app.UserName = "Authors"
app.UserInitials = "AU"
try:
    d_base = app.Documents.Open(base, ConfirmConversions=False, ReadOnly=True, AddToRecentFiles=False, Visible=False)
    t0 = time.time()
    # Compare(Name, AuthorName, CompareTarget, DetectFormatChanges, IgnoreAllComparisonWarnings,
    #         AddToRecentFiles, RemovePersonalInformation, RemoveDateAndTime)
    # CompareTarget 2 = wdCompareTargetNew (result goes to a new document)
    d_base.Compare(rev, "Authors", 2, False, True, False, True, True)
    result = app.ActiveDocument
    print("compare done in %.1fs; revisions: %d" % (time.time() - t0, result.Revisions.Count))
    authors = {}
    for r in result.Revisions:
        authors[r.Author] = authors.get(r.Author, 0) + 1
    print("revision authors:", authors)
    # neutral document properties
    for name in ("Author", "Last Author", "Manager", "Company", "Comments", "Keywords", "Subject"):
        try:
            result.BuiltInDocumentProperties(name).Value = ""
        except Exception as e:
            print("property", name, "->", e)
    try:
        result.BuiltInDocumentProperties("Title").Value = "Tracked changes: submitted text compared with the revised text"
    except Exception as e:
        print("title ->", e)
    # pandoc stored the submitted abstract as a custom property; drop all custom properties
    try:
        props = result.CustomDocumentProperties
        for i in range(props.Count, 0, -1):
            props(i).Delete()
    except Exception as e:
        print('custom properties ->', e)
    result.RemovePersonalInformation = True
    result.RemoveDocumentInformation(4)  # wdRDIRemovePersonalInformation
    result.SaveAs2(out, 16)  # wdFormatXMLDocument (.docx)
    print("saved", out)
    result.Close(False)
    d_base.Close(False)
finally:
    app.UserName, app.UserInitials = old_user, old_init
    app.Quit(False)
