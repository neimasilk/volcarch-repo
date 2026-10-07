"""Build the P8 revision as a Word file in the Oceanic Linguistics template from the Markdown source.

usage: python build_docx.py [--anon]

Source: P8_revision_v0.2.md (this folder). Output: P8_revision_v0.2.docx, or P8_revision_v0.2_anonymous.docx with --anon
(author block and repository URL replaced by placeholders). Tables are filled from the result files of E228/E229/E231 at
build time, so a number in a table can never drift from its CSV. References are formatted by pandoc with the Unified
Style Sheet for Linguistics (CSL file beside this script) from the two .bib files, for the keys listed in CITED:.

Markdown conventions (kept deliberately small):
  TITLE: / RUNNINGHEAD: / AUTHOR: name | affiliation | e-mail  (repeatable) / KEYWORDS: / CITED: key, key, ...
  ABSTRACT: followed by the abstract paragraph (ends at the first blank line)
  # A-level heading        ## B-level heading   (the OL styles number themselves; write no numbers)
  [[TABLE:T1|Caption]]     table from the result files; following lines beginning with † ‡ § are table notes
  [[FIGURE:1|Caption]]     caption + position marker (the image file is uploaded separately, as the press asks)
  [[REFERENCES]]           the formatted reference list
  [[UNNUMBERED:Heading]]   an unnumbered heading (data availability, AI declaration, acknowledgments)
  *italic*  **bold**       inline; a reconstruction's asterisk is written \\* (e.g. \\*zalan) so it is not read as italics
"""
import csv, re, subprocess, sys, tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_COLOR_INDEX

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E228 = REPO / "experiments/E228_p8_revision_analyses/results"
E229 = REPO / "experiments/E229_p8_revision_tables/results"
E231 = REPO / "experiments/E231_p8_what_coded_means/results"
ANON = "--anon" in sys.argv
SRC = HERE / "P8_revision_v0.2.md"
OUT = HERE / ("P8_revision_v0.2_anonymous.docx" if ANON else "P8_revision_v0.2.docx")
BASE = HERE / "ol_base.docx"          # the journal's .dotx with its content type changed to a document (see VENUE.md)
CSL = HERE / "unified-style-sheet-for-linguistics.csl"
BIBS = [HERE / "references.bib", HERE / "references_reviewer_supplied.bib"]

# ----------------------------------------------------------------------------- helpers
def rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))

INLINE = re.compile(r"(\*\*.+?\*\*|\*(?!\s).+?(?<!\s)\*)")

def add_runs(p, text, size=None, highlight=None):
    text = text.replace("\\*", "\x00")           # protected asterisk (reconstructions)
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            r = p.add_run(piece[2:-2].replace("\x00", "*")); r.bold = True
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            r = p.add_run(piece[1:-1].replace("\x00", "*")); r.italic = True
        else:
            r = p.add_run(piece.replace("\x00", "*"))
        if size: r.font.size = Pt(size)
        if highlight: r.font.highlight_color = highlight
    return p

def para(d, text, style, **kw):
    p = d.add_paragraph(style=style)
    return add_runs(p, text, **kw)

def table(d, caption, header, rows, notes=()):
    para(d, caption, "OL caption")
    t = d.add_table(rows=1, cols=len(header)); t.style = d.styles["Table Grid"]
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; p = c.paragraphs[0]; p.style = d.styles["OL table column heading"]
        r = p.add_run(h); r.bold = True; r.font.size = Pt(8)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; p = cells[i].paragraphs[0]; p.style = d.styles["OL table contents"]
            add_runs(p, str(v), size=8)
    for n in notes:
        para(d, n, "OL text", size=8)
    para(d, "", "OL 3pt separator")

def rnd(x, nd):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

def fmt3(x): return rnd(x, 3)

def fmtd(x, nd=3):
    v = rnd(x, nd)
    return v if v.startswith('-') or float(v) == 0 else '+' + v

# ----------------------------------------------------------------------------- tables from the result files
NAME = {"Muna": "Muna", "Bugis": "Bugis", "Makassar": "Makasar", "Wolio": "Wolio", "Toraja-Sadan": "Sa'dan Toraja", "Tolaki": "Tolaki"}

def T1():
    t1 = rd(E229 / "T1_label_by_list.csv")
    return (["List", "Forms", "Coded (n)", "Coded (%)", "Uncoded (n)", "Uncoded (%)"],
            [[r["list"], r["forms"], r["coded_n"], f"{float(r['coded_pct']):.1f}", r["candidate_n"], f"{float(r['candidate_pct']):.1f}"] for r in t1])

def T2():
    ci = {(r["list"], r["class"]): r for r in rd(E231 / "C_retention_intervals.csv")}
    def cell(lst, cls):
        r = ci[(lst, cls)]; return f"{rnd(100*int(r['n'])/int(r['base']), 1)} [{rnd(100*float(r['ci_lo']), 1)}, {rnd(100*float(r['ci_hi']), 1)}] ({r['n']})"
    rows = []
    for key in ("Makassar", "Bugis", "Toraja-Sadan", "Wolio", "Muna", "Tolaki"):
        rows.append([NAME[key], ci[(key, "uncoded")]["base"], cell(key, "retained_from_PMP"), cell(key, "coded_not_PMP"), cell(key, "uncoded")])
    return (["List", "Meanings with a PMP entry", "In the PMP entry's set, % [95% CI] (n)", "Coded in another set, % [95% CI] (n)", "No cognate set, % [95% CI] (n)"], rows)

def T3():
    t2 = rd(E229 / "T2_cv_performance.csv"); lab = {"FM25": "form + meaning (25)", "F17": "form only (17)"}
    rows = [[lab[r["input_set"]], r["classifier"], fmt3(r["auc_mean"]), fmt3(r["accuracy_mean"]), fmt3(r["accuracy_always_coded"]),
             fmt3(r["precision_candidate_mean"]), fmt3(r["recall_candidate_mean"]), fmt3(r["f1_candidate_mean"])] for r in t2]
    sd = "; ".join(f"{lab[r['input_set']]}, {r['classifier']}: {fmt3(r['auc_sd_seed_means'])} / {fmt3(r['auc_sd_folds'])}" for r in t2)
    return (["Inputs", "Classifier", "AUC", "Accuracy", "Always coded", "Precision", "Recall", "F1"], rows, sd)

def T4():
    t3 = rd(E229 / "T3_lolo.csv"); rows = []
    for r in [x for x in t3 if x["input_set"] == "FM25"]:
        f = next(x for x in t3 if x["input_set"] == "F17" and x["list_key"] == r["list_key"])
        if r["list_key"] == "SUMMARY":
            rows.append(["Mean (SD) over the six lists", "", "", f"{fmt3(r['auc'])} ({fmt3(r['auc_sd_over_lists'])})", "", f"{fmt3(f['auc'])} ({fmt3(f['auc_sd_over_lists'])})", "", ""])
        else:
            rows.append([r["list"], r["n"], r["n_candidates"], fmt3(r["auc"]), fmt3(r["accuracy"]), fmt3(f["auc"]), fmt3(f["accuracy"]), fmt3(r["accuracy_majority_answer"])])
    return (["Held-out list", "n", "Uncoded", "AUC (F+M)", "Acc. (F+M)", "AUC (F)", "Acc. (F)", "Majority"], rows)

def T5():
    p3 = {r["contrast"]: r for r in rd(E231 / "P3_length_conditioned.csv") if r["strata"] == "list x number of letters"}
    eq = {"has_glottal": "written glottal mark", "sem_ACTION": "action meaning", "has_prefix_like": "onset string",
          "has_nasal_cluster": "nasal input as coded", "n_consonant_clusters": ">= 1 consonant-letter cluster"}
    rows = []
    for r in rd(E229 / "T6_profile.csv"):
        if r["type"] == "0/1":
            a, b = f"{100*float(r['mean_candidate']):.1f}%", f"{100*float(r['mean_coded']):.1f}%"; et = "odds ratio ‡"
        else:
            a, b = f"{float(r['mean_candidate']):.2f}", f"{float(r['mean_coded']):.2f}"; et = "mean difference †"
        eff = float(r["effect"]); sign = "+" if (r["type"] != "0/1" and eff > 0) else ""
        if r["property_key"] in eq:
            q = p3[eq[r["property_key"]]]; eqv = f"{rnd(q['odds_ratio'], 2)} [{rnd(q['ci_lo'], 2)}, {rnd(q['ci_hi'], 2)}]"
        else:
            eqv = "—"
        rows.append([r["property"], a, b, r["n_lists_same_sign_as_pooled"], et, f"{sign}{rnd(eff, 2)} [{rnd(r['ci_lo'], 2)}, {rnd(r['ci_hi'], 2)}]", eqv])
    return (["Property", "Uncoded", "Coded", "Same sign (of 6)", "Effect type", "Effect [95% CI]", "At equal letter count §"], rows)

def T6():
    rows = [[r["list"], r["n"], r["candidate_and_profile"], r["candidate_and_no_profile"], r["coded_and_profile"], r["coded_and_no_profile"], f"{float(r['kappa']):.2f}"] for r in rd(E229 / "T5_cells_by_list.csv")]
    return (["List", "n", "Uncoded, profile", "Uncoded, no profile", "Coded, profile", "Coded, no profile", "Kappa"], rows)

def T7():
    rows, seen = [], {}
    for r in rd(E229 / "T7_examples_R1-11.csv"):
        if r["scope"] != "all six": continue
        k = r["cell"]; seen[k] = seen.get(k, 0)
        if seen[k] >= 2: continue
        seen[k] += 1
        look = f"{r['sulawesi_lookalike_form']} ({r['sulawesi_lookalike_list']})" if r["sulawesi_lookalike_form"] else "—"
        rows.append([k.replace("candidate", "uncoded"), NAME[r["language"]], r["concept"], r["form"] + (" ‡" if r["abvd_loan_flag"] == "1" else ""),
                     r["cognacy"] or "—", (r["pmp_forms"] or "—").replace("*", "\\*"), f"{float(r['p_candidate_oof25']):.2f}", look])
    return (["Cell", "List", "Meaning", "Form (ABVD)", "Set", "PMP (ABVD)", "Score", "Nearest look-alike (list)"], rows)

def T8():
    names = {"V0_as_published": "As in the sources (ʔ or apostrophe)", "V1_q": "Mark written as q", "V2_k": "Mark written as k",
             "V3_unwritten": "Mark left unwritten", "V4_geminate": "Pre-glottalised consonant written as a geminate",
             "V5_all_as_glottal_letter": "Apostrophe written as ʔ (changes no input)", "V6_feature_removed": "Glottal input removed"}
    g = rd(E228 / "TABLE_R1-9_glottal_conventions.csv"); base_cv = float(fmt3(g[0]["cv_auc_25"])); base_lo = float(fmt3(g[0]["lolo_mean_25"]))
    rows = [[names[r["convention"]], r["forms_changed"], fmt3(r["cv_auc_25"]), fmtd(float(fmt3(r["cv_auc_25"])) - base_cv), fmt3(r["lolo_mean_25"]), fmtd(float(fmt3(r["lolo_mean_25"])) - base_lo)]
            for r in g]
    return (["Convention", "Forms changed", "AUC, cross-validated", "Δ", "AUC, held-out list (mean)", "Δ"], rows)

TABLES = {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5, "T6": T6, "T7": T7, "T8": T8}

# ----------------------------------------------------------------------------- references
def references(keys):
    md = "---\nbibliography:\n" + "".join(f"  - {b.as_posix()}\n" for b in BIBS) + "nocite: |\n  " + ", ".join(f"@{k}" for k in keys) + "\n---\n\n# References\n"
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "refs.md"; src.write_text(md, encoding="utf-8")
        out = subprocess.run(["pandoc", str(src), "--citeproc", "--csl", str(CSL), "-t", "plain", "--wrap=none"],
                             capture_output=True, text=True, encoding="utf-8")
        if out.returncode != 0:
            raise SystemExit(out.stderr)
    lines = [l.strip() for l in out.stdout.splitlines() if l.strip() and l.strip() != "References"]
    if out.stderr.strip():
        print("pandoc:", out.stderr.strip()[:500])
    return lines

# ----------------------------------------------------------------------------- build
def main():
    text = SRC.read_text(encoding="utf-8")
    meta = {"AUTHOR": []}
    body_lines = []
    abstract, in_abs = [], False
    for line in text.splitlines():
        m = re.match(r"^(TITLE|RUNNINGHEAD|AUTHOR|KEYWORDS|CITED|ABSTRACT):\s*(.*)$", line)
        if m and not body_lines:
            k, v = m.group(1), m.group(2).strip()
            if k == "AUTHOR": meta["AUTHOR"].append(v)
            elif k == "ABSTRACT": in_abs = True; abstract.append(v) if v else None
            else: meta[k] = v
            continue
        if in_abs:
            if line.strip() == "": in_abs = False
            else: abstract.append(line.strip())
            continue
        body_lines.append(line)
    keys = [k.strip() for k in re.split(r"[,\s]+", meta.get("CITED", "")) if k.strip()]

    d = Document(BASE)
    body = d.element.body
    for el in list(body):
        if not el.tag.endswith("}sectPr"):
            body.remove(el)

    para(d, meta["TITLE"], "OL article title")
    if ANON:
        para(d, "[Authors anonymised for review]", "OL author name")
        para(d, "[Affiliations anonymised for review]", "OL author affiliation")
    else:
        for a in meta["AUTHOR"]:
            parts = [x.strip() for x in a.split("|")]
            para(d, parts[0], "OL author name")
            if len(parts) > 1: para(d, parts[1], "OL author affiliation")
    para(d, " ".join(abstract), "OL abstract")
    if meta.get("KEYWORDS"): para(d, "Keywords: " + meta["KEYWORDS"], "OL abstract")

    first_after_heading = True
    i = 0
    paragraph_buf = []
    def flush():
        nonlocal paragraph_buf, first_after_heading
        if paragraph_buf:
            t = " ".join(paragraph_buf).strip()
            if ANON:
                t = re.sub(r"https://github\.com/\S+", "[repository URL redacted for review]", t)
                t = t.replace("https://doi.org/[DOI]", "[DOI redacted for review]")
            para(d, t, "OL text" if first_after_heading else "OL text indented")
            first_after_heading = False
            paragraph_buf = []
    while i < len(body_lines):
        line = body_lines[i]
        s = line.strip()
        if s == "":
            flush(); i += 1; continue
        if s.startswith("## "):
            flush(); para(d, s[3:].rstrip(".") + ".", "OL heading B-level"); first_after_heading = True; i += 1; continue
        if s.startswith("# "):
            flush(); para(d, s[2:].rstrip(".") + ".", "OL heading A-level"); first_after_heading = True; i += 1; continue
        m = re.match(r"^\[\[UNNUMBERED:(.+?)\]\]$", s)
        if m:
            flush(); para(d, m.group(1).upper(), "OL references heading"); first_after_heading = True; i += 1; continue
        m = re.match(r"^\[\[TABLE:(\w+)\|(.+?)\]\]$", s)
        if m:
            flush(); name, cap = m.group(1), m.group(2)
            notes = []
            j = i + 1
            while j < len(body_lines) and (body_lines[j].strip()[:1] in ("†", "‡", "§") or body_lines[j].strip().startswith("Note.")):
                notes.append(body_lines[j].strip()); j += 1
            spec = TABLES[name]()
            if name == "T3":
                header, rows, sd = spec; notes = notes + ["Standard deviation of the AUC over the ten repetition means / over the fifty test sets: " + sd + "."]
            else:
                header, rows = spec
            table(d, cap, header, rows, notes)
            first_after_heading = False
            i = j; continue
        m = re.match(r"^\[\[FIGURE:(\d+)\|(.+?)\]\]$", s)
        if m:
            flush(); para(d, m.group(2), "OL caption")
            p = para(d, f"[Figure {m.group(1)} about here — separate file]", "OL text");
            for r in p.runs: r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            para(d, "", "OL 3pt separator"); first_after_heading = False; i += 1; continue
        if s == "[[REFERENCES]]":
            flush(); para(d, "REFERENCES", "OL references heading")
            for entry in references(keys):
                e = entry
                if ANON: e = re.sub(r"https://github\.com/\S+", "[repository URL redacted]", e)
                para(d, e, "OL reference list")
            i += 1; continue
        paragraph_buf.append(s); i += 1
    flush()
    d.save(OUT)
    words = len(re.findall(r"\w+", " ".join(abstract) + " " + " ".join(l for l in body_lines if not l.startswith("[["))))
    print(f"saved {OUT.name}: paragraphs {len(d.paragraphs)}, tables {len(d.tables)}, abstract words {len(' '.join(abstract).split())}, body words about {words}, cited keys {len(keys)}")

if __name__ == "__main__":
    main()
