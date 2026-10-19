"""E233 step 1: reader packets — the editors' translation plus the occurrences to locate in it, no frame codes.

Rules are frozen in ../DESIGN.md §2–3. The occurrence contexts are rebuilt with E226's own tokenisation (plus line
markers) and asserted equal to E226 frame/coder_input.csv, so the readers see exactly what the coders saw.
"""
import csv
import importlib.util
import os
import re
import sys

from lxml import etree

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXP, "..", ".."))
E226 = os.path.join(ROOT, "experiments", "E226_t7_detection_calibration")
XML = os.path.join(ROOT, "experiments", "E023_ritual_screening", "data", "dharma", "xml")
OUT = os.path.join(EXP, "packets")
TEI = "{http://www.tei-c.org/ns/1.0}"
NS = {"t": "http://www.tei-c.org/ns/1.0"}
MIN_CHARS = 300  # DESIGN §2: longest translation div > 300 characters

spec = importlib.util.spec_from_file_location("e226", os.path.join(E226, "scripts", "01_build_candidates.py"))
e226 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e226)


def walk_marked(e, out, lines):
    # e226.walk, plus a marker token at each line break so every occurrence gets its edition line
    for ch in e:
        if not isinstance(ch.tag, str):
            if ch.tail:
                out.append(ch.tail)
            continue
        ln = etree.QName(ch).localname
        if ln in e226.DROP:
            pass
        elif ln == "lb" and ch.get("break") == "no":
            if ch.tail:
                out.append(ch.tail.lstrip())
            continue
        elif ln == "lb":
            lines.append(ch.get("n") or "?")
            out.append(f" ⟦LB{len(lines) - 1}⟧ ")
        elif ln in ("pb", "milestone"):
            out.append(" ")
        else:
            if ch.text:
                out.append(ch.text)
            walk_marked(ch, out, lines)
        if ch.tail:
            out.append(ch.tail)


def edition_tokens(root):
    out, lines = [], []
    for d in root.iter(TEI + "div"):
        if d.get("type") == "edition":
            out.append(d.text or "")
            walk_marked(d, out, lines)
            out.append(" \n ")
    toks, line_of, cur = [], [], ""
    for t in e226.TOK.findall(re.sub(r"\s+", " ", "".join(out))):
        m = re.fullmatch(r"⟦LB(\d+)⟧", t)
        if m:
            cur = lines[int(m.group(1))]
            continue
        toks.append(t)
        line_of.append(cur)
    return toks, line_of


def drop(el):
    parent, prev = el.getparent(), el.getprevious()
    if el.tail:
        if prev is not None:
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


def translation(root):
    divs = root.findall(f".//{TEI}div[@type='translation']")
    if not divs:
        return None, None
    size = lambda d: len(" ".join("".join(d.itertext()).split()))
    best = max(divs, key=size)
    if size(best) <= MIN_CHARS:  # scope test on the div as published, before notes are dropped
        return None, None
    for n in list(best.iter(TEI + "note")):  # translator's footnotes dropped, as notes are dropped from the edition
        drop(n)
    paras = []
    for i, p in enumerate(best.iter(TEI + "p"), 1):
        txt = " ".join("".join(p.itertext()).split())
        if txt:
            paras.append(f"[P{i}{' | n=' + p.get('n') if p.get('n') else ''}] {txt}")
    lang = best.get("{http://www.w3.org/XML/1998/namespace}lang") or "eng (not declared)"
    return "\n\n".join(paras), lang


def main():
    coder_in = {r["occ_id"]: r for r in csv.DictReader(open(os.path.join(E226, "frame", "coder_input.csv"),
                                                             encoding="utf-8"))}
    by_ins = {}
    for oid in coder_in:
        by_ins.setdefault(oid.split("#")[0], []).append(oid)
    os.makedirs(OUT, exist_ok=True)
    index = []
    for ins in sorted(by_ins):
        root = etree.parse(os.path.join(XML, f"DHARMA_{ins}.xml")).getroot()
        tr, lang = translation(root)
        if tr is None:
            continue
        toks, line_of = edition_tokens(root)
        # same tokens as E226 → same occurrences and contexts
        assert toks == e226.TOK.findall(re.sub(r"\s+", " ", e226.edition_text(root))), ins
        nt = [e226.norm(t) for t in toks]
        occ, n_v = [], 0
        for i, t in enumerate(nt):
            if not e226.VILLAGE.match(t):
                continue
            n_v += 1
            oid = f"{ins}#{n_v:03d}"
            row = dict(occ_id=oid, line=line_of[i], kwic_left=" ".join(toks[max(0, i - 12):i]), noun=toks[i],
                       kwic_right=" ".join(toks[i + 1:i + 13]))
            c = coder_in[oid]
            assert (row["kwic_left"], row["noun"], row["kwic_right"]) == (c["kwic_left"], c["noun"], c["kwic_right"]), oid
            occ.append(row)
        assert n_v == len(by_ins[ins]), ins
        title = " ".join("".join(root.find(f".//{TEI}title").itertext()).split())
        resp = "; ".join(" ".join("".join(r.itertext()).split())
                         for r in root.iterfind(f".//{TEI}titleStmt/{TEI}respStmt"))
        with open(os.path.join(OUT, f"{ins}.translation.txt"), "w", encoding="utf-8") as fh:
            fh.write(tr)
        md = [f"# {ins} — {title}", "", f"**Responsibility (TEI header):** {resp or 'not stated'}",
              f"**Translation language:** {lang}", "",
              "## Translation (verbatim from the DHARMA edition; paragraph number and line reference in [ ])", "",
              tr, "", "## Occurrences to locate in the translation (edition text order)", "",
              "| occ_id | edition line | context before | NOUN | context after |", "|---|---|---|---|---|"]
        md += [f"| {o['occ_id']} | {o['line']} | {o['kwic_left']} | **{o['noun']}** | {o['kwic_right']} |" for o in occ]
        with open(os.path.join(OUT, f"{ins}.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(md) + "\n")
        index.append(dict(inscription=ins, n_occ=len(occ), translation_chars=len(tr), lang=lang, resp=resp))
    with open(os.path.join(OUT, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(len(index), "packets;", sum(r["n_occ"] for r in index), "occurrences;",
          sum(r["translation_chars"] for r in index), "translation characters")


if __name__ == "__main__":
    main()
