"""E225 step 3 — text layer of each downloaded PDF, page by page (DESIGN §7; PyMuPDF instead of pypdf, see README).

usage: python -X utf8 03_extract_text.py
in:    results/download_log.csv, data/raw/e225_corpus/pdf/
out:   data/raw/e225_corpus/text/<journal>_<id>.json   {"file", "pages": [...]}   (git-ignored)
       results/text_log.csv   (file, pages, chars, readable)
A document is "readable" when its text layer has at least 1,000 characters; OCR is not run in the pilot.
"""
import csv, json
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PDF_DIR = REPO / "data" / "raw" / "e225_corpus" / "pdf"
TXT_DIR = REPO / "data" / "raw" / "e225_corpus" / "text"
TXT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted({r["file"] for r in csv.DictReader((HERE / "results" / "download_log.csv").open(encoding="utf-8")) if r["file"]})
rows = []
for name in files:
    try:
        doc = fitz.open(PDF_DIR / name)
        pages = [p.get_text("text") for p in doc]
    except Exception as e:  # a broken PDF is logged, not skipped silently
        rows.append({"file": name, "pages": 0, "chars": 0, "readable": 0, "error": type(e).__name__})
        continue
    chars = sum(len(p) for p in pages)
    (TXT_DIR / (Path(name).stem + ".json")).write_text(json.dumps({"file": name, "pages": pages}, ensure_ascii=False), encoding="utf-8")
    rows.append({"file": name, "pages": len(pages), "chars": chars, "readable": int(chars >= 1000), "error": ""})

with (HERE / "results" / "text_log.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "PDFs;", sum(r["readable"] for r in rows), "readable;", sum(1 for r in rows if r["error"]), "errors")
