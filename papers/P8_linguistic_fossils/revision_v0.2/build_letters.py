"""Build the two letters of the P8 resubmission as Word files (and PDF reading copies) from their Markdown sources.

    COVER_LETTER_v0.4.md            -> .docx / .pdf   signed; portal slot "Author Cover Letter" (editors only)
    RESPONSE_TO_REVIEWERS_v0.4.md   -> .docx / .pdf   no names; portal slot "Revision Summary" (may go to the reviewers)

usage: python -X utf8 build_letters.py            (Word must be installed: docx2pdf drives it for the PDF)

pandoc writes the .docx with a reference document made here (letter_reference.docx: Times New Roman, plain black
headings) instead of pandoc's default look. The document properties are cleared, so that the response carries no
author field. Checks of the result are in check_letters.py.
"""
import subprocess, sys, zipfile, shutil, re
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor

HERE = Path(__file__).resolve().parent
REF = HERE / "letter_reference.docx"
VERSION = "v0.4"
JOBS = [("COVER_LETTER", "Cover letter — revised manuscript OL-03-2026-11"),
        ("RESPONSE_TO_REVIEWERS", "Response to the reviewers — manuscript OL-03-2026-11")]


def make_reference():
    raw = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], capture_output=True, check=True).stdout
    REF.write_bytes(raw)
    d = Document(str(REF))
    black = RGBColor(0, 0, 0)
    sizes = {"Title": 14, "Heading 1": 12, "Heading 2": 11, "Heading 3": 11}
    for st in d.styles:
        try:
            f = st.font
        except AttributeError:
            continue
        if st.type is None or f is None:
            continue
        f.name = "Times New Roman"
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts")
        if rf is not None:   # every script slot, and no theme font overriding the name
            w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            for a in ("ascii", "hAnsi", "cs", "eastAsia"):
                rf.set(w + a, "Times New Roman")
            for a in ("asciiTheme", "hAnsiTheme", "cstheme", "eastAsiaTheme"):
                if rf.get(w + a) is not None:
                    del rf.attrib[w + a]
        if st.name in sizes:
            f.size = Pt(sizes[st.name]); f.bold = True; f.italic = False; f.color.rgb = black
        elif st.name in ("Normal", "Body Text", "First Paragraph", "Compact"):
            f.size = Pt(11); f.color.rgb = black
    d.save(str(REF))


def clear_properties(path, title):
    d = Document(str(path))
    cp = d.core_properties
    cp.author = ""; cp.last_modified_by = ""; cp.comments = ""; cp.keywords = ""; cp.subject = ""; cp.title = title
    d.save(str(path))
    # python-docx leaves docProps/custom.xml and app.xml as pandoc wrote them; drop an empty custom part's content check
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        custom = z.read("docProps/custom.xml").decode("utf-8") if "docProps/custom.xml" in names else ""
    assert "<property" not in custom, "pandoc wrote a custom document property: " + custom[:200]


def main():
    make_reference()
    from docx2pdf import convert
    for stem, title in JOBS:
        md = HERE / f"{stem}_{VERSION}.md"
        out = HERE / f"{stem}_{VERSION}.docx"
        subprocess.run(["pandoc", str(md), "-f", "markdown+smart", "-t", "docx", "--reference-doc", str(REF), "-o", str(out)], check=True)
        clear_properties(out, title)
        convert(str(out), str(out.with_suffix(".pdf")))
        print("built", out.name, out.stat().st_size, "bytes;", out.with_suffix(".pdf").name)


if __name__ == "__main__":
    main()
