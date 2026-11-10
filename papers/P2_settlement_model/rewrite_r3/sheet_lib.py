# Shared layout helpers for the P2 author-rewrite writing sheets (session 2 onwards).
# Same look as build_writing_sheet.py (session 1), which is left as it was when it produced LEMBAR_TULIS_S1.
# Content rule (C093): bullets, numbers and questions only -- no prose the PI could copy.

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

CENTER = WD_ALIGN_PARAGRAPH.CENTER


def _shade(el, hexcolor):
    pr = el.get_or_add_tcPr() if el.tag.endswith("}tc") else el.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    pr.append(shd)


class Sheet:
    def __init__(self):
        self.doc = doc = Document()
        sec = doc.sections[0]
        sec.top_margin = sec.bottom_margin = Cm(1.8)
        sec.left_margin = sec.right_margin = Cm(2.0)
        st = doc.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(10.5)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        st.paragraph_format.space_after = Pt(4)
        fp = sec.footer.paragraphs[0]
        fp.alignment = CENTER
        run = fp.add_run()
        f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
        f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
        run._r.append(f1); run._r.append(it); run._r.append(f2)

    def cover(self, title, subtitle):
        p = self.doc.add_paragraph(); p.alignment = CENTER
        r = p.add_run(title); r.bold = True; r.font.size = Pt(22)
        r.font.color.rgb = RGBColor.from_string("1F3864")
        p = self.doc.add_paragraph(); p.alignment = CENTER
        r = p.add_run(subtitle); r.font.size = Pt(11.5); r.italic = True
        self.para("")

    def h(self, text, level=1, color="1F3864"):
        p = self.doc.add_heading(text, level=level)
        if getattr(self, "_new_page", False):
            p.paragraph_format.page_break_before = True
            self._new_page = False
        for r in p.runs:
            r.font.color.rgb = RGBColor.from_string(color)
        return p

    def para(self, text, bold=False, italic=False, size=10.5, space_before=0, align=None):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(space_before)
        if align:
            p.alignment = align
        r = p.add_run(text); r.bold = bold; r.italic = italic; r.font.size = Pt(size)
        return p

    def bullets(self, *items, style="List Bullet"):
        for it in items:
            p = self.doc.add_paragraph(style=style)
            p.paragraph_format.space_after = Pt(2)
            if it.startswith("**") and "**" in it[2:]:
                lead, rest = it[2:].split("**", 1)
                p.add_run(lead).bold = True
                p.add_run(rest)
            else:
                p.add_run(it)

    def table(self, rows, widths=None, header=True, size=9.5):
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(rows):
            for j, cell_text in enumerate(row):
                cell = t.cell(i, j); cell.text = ""
                p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(1)
                r = p.add_run(str(cell_text)); r.font.size = Pt(size)
                if header and i == 0:
                    r.bold = True
                    _shade(cell._tc, "D9E2F3")
        if widths:
            for row in t.rows:
                for j, w in enumerate(widths):
                    row.cells[j].width = Cm(w)
        return t

    def writebox(self, lines=8, label="TULIS DI SINI", fill="F7F7F7"):
        p = self.doc.add_paragraph()
        r = p.add_run(label); r.bold = True; r.italic = True; r.font.size = Pt(9)
        r.font.color.rgb = RGBColor.from_string("7F7F7F")
        t = self.doc.add_table(rows=1, cols=1); t.style = "Table Grid"
        cell = t.cell(0, 0); _shade(cell._tc, fill); cell.text = ""
        for _ in range(lines):
            cp = cell.add_paragraph(); cp.paragraph_format.space_after = Pt(2)
            cp.add_run("").font.size = Pt(10.5)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def task_card(self, no, heading, words, questions, detail):
        t = self.doc.add_table(rows=1, cols=1); t.style = "Table Grid"
        cell = t.cell(0, 0); _shade(cell._tc, "EEF3FA"); cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(f"{no}. {heading}   ·   {words}"); r.bold = True; r.font.size = Pt(11)
        r.font.color.rgb = RGBColor.from_string("1F3864")
        p2 = cell.add_paragraph(); p2.paragraph_format.space_before = Pt(3)
        rr = p2.add_run("Tulis prosa yang menjawab pertanyaan-pertanyaan ini:"); rr.italic = True; rr.font.size = Pt(9.5)
        for q in questions:
            qp = cell.add_paragraph(); qp.paragraph_format.space_after = Pt(1)
            qp.paragraph_format.left_indent = Cm(0.5)
            qp.add_run("• " + q).font.size = Pt(10)
        p3 = cell.add_paragraph(); p3.paragraph_format.space_before = Pt(3)
        r3 = p3.add_run(f"Angka dan sumbernya: {detail}"); r3.font.size = Pt(9)
        r3.font.color.rgb = RGBColor.from_string("7F7F7F")
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def page_break(self):
        # A separate break paragraph spills onto an empty page when the page above is exactly full;
        # putting page_break_before on the next heading avoids that.
        self._new_page = True

    def save(self, out):
        self.doc.save(out)
        print("written:", out)
