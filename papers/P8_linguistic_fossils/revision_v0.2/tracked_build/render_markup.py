"""Export the tracked-changes document to PDF *with* markup so that it can be looked at.
usage: python render_markup.py TRACKED.docx OUT.pdf"""
import sys, os
import pythoncom
import win32com.client as wc

src, out = [os.path.abspath(a) for a in sys.argv[1:3]]
pythoncom.CoInitialize()
app = wc.DispatchEx("Word.Application")
app.Visible = False
app.DisplayAlerts = 0
try:
    d = app.Documents.Open(src, ConfirmConversions=False, ReadOnly=True, AddToRecentFiles=False, Visible=False)
    try:
        v = app.ActiveWindow.View
        v.ShowRevisionsAndComments = True
        v.RevisionsView = 0      # wdRevisionsViewFinal (shows markup)
        v.MarkupMode = 2         # wdBalloonRevisions / inline fallback below
    except Exception as e:
        print("view ->", e)
    # ExportAsFixedFormat(OutputFileName, ExportFormat=17 pdf, OpenAfterExport, OptimizeFor, Range, From, To,
    #                     Item=7 wdExportDocumentWithMarkup, ...)
    d.ExportAsFixedFormat(out, 17, False, 0, 0, 1, 1, 7)
    print("pages:", d.ComputeStatistics(2))
    d.Close(False)
finally:
    app.Quit(False)
