"""One PDF for the PI to read before Submit: the note BACA_DULU_P8_<date>.md, the cover letter, the response to the
reviewers and the redrawn figures.

usage: python -X utf8 build_reading_pdf.py      (run after build_letters.py; Word is needed for the note's PDF)
output: BACA_DULU_P8_20261012.pdf   (the copy of 2026-10-08 was built by the version of this script in git history)
"""
import subprocess
from pathlib import Path
import fitz
from docx2pdf import convert

HERE = Path(__file__).resolve().parent
FIG = HERE.parents[2] / "experiments" / "E229_p8_revision_tables" / "results"
DATE, LV = "20261012", "v0.5"
NOTE = HERE / f"BACA_DULU_P8_{DATE}.md"
OUT = HERE / f"BACA_DULU_P8_{DATE}.pdf"
tmp_docx, tmp_pdf = HERE / "_baca_note.docx", HERE / "_baca_note.pdf"

subprocess.run(["pandoc", str(NOTE), "-f", "markdown+smart", "-t", "docx", "--reference-doc", str(HERE / "letter_reference.docx"),
                "-o", str(tmp_docx)], check=True)
convert(str(tmp_docx), str(tmp_pdf))

out = fitz.open()
parts = [tmp_pdf, HERE / f"COVER_LETTER_{LV}.pdf", HERE / f"RESPONSE_TO_REVIEWERS_{LV}.pdf"]
for p in parts:
    with fitz.open(str(p)) as d:
        out.insert_pdf(d)
# figures: one A4 page each, the PNG scaled to fit under a one-line heading
FIGS = (("F2_workflow.png", "Gambar 2 (digambar ulang: 'profile' dihitung langsung, bukan keluaran classifier)"),
        ("F1_input_importance.png", "Gambar 1 (dirapikan: jarak baris label; tiga label)"))
for name, title in FIGS:
    page = out.new_page(width=595, height=842)
    page.insert_text((56, 60), title, fontsize=11, fontname="helv")
    pix = fitz.Pixmap(str(FIG / name))
    box_w, box_h = 595 - 112, 842 - 150
    scale = min(box_w / pix.width, box_h / pix.height, 312 / pix.width * 1.5)   # at most 1.5 x the printed width
    w, h = pix.width * scale, pix.height * scale
    x0 = (595 - w) / 2
    page.insert_image(fitz.Rect(x0, 80, x0 + w, 80 + h), filename=str(FIG / name))
toc = [[1, "Halaman baca: yang berubah dan keputusan", 1]]
n = fitz.open(str(tmp_pdf)).page_count
toc.append([1, f"Surat pengantar {LV}", n + 1]); n += fitz.open(str(parts[1])).page_count
toc.append([1, f"Tanggapan untuk reviewer {LV}", n + 1]); n += fitz.open(str(parts[2])).page_count
toc.append([1, "Gambar 2 dan 1", n + 1])
out.set_toc(toc)
out.save(str(OUT), deflate=True)
print("written", OUT.name, "pages:", out.page_count, "| bytes:", OUT.stat().st_size, "| contents:", [(t[1], t[2]) for t in toc])
out.close()
tmp_docx.unlink(); tmp_pdf.unlink()
