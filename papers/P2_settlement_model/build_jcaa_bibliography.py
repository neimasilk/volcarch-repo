"""JCAA Harvard reference list for P2 (round-2 revision; VENUE.md A10). 2026-10-20.

natbib + plainnat prints "(Castiello and Tonini, 2021)" and full first names. JCAA asks for
"(Castiello & Tonini 2021)", "(Comer et al. 2023)" for four or more authors, and a list in the form
    Surname, Initials. Year. Title. Journal, vol(issue): pages. DOI: 10.xxx
This script writes that list as a natbib-compatible thebibliography (references_jcaa.tex) from references.bib,
for the keys the manuscript actually cites, so the in-text form follows from the \\bibitem labels.

    python build_jcaa_bibliography.py submission_jcaa_v0.3.tex            # write references_jcaa.tex
    python build_jcaa_bibliography.py submission_jcaa_v0.3.tex --apply    # also switch the .tex to it (idempotent)
    python build_jcaa_bibliography.py submission_jcaa_v0.3.tex --report   # print the list as plain text
Re-run after every change to the citations: the list holds only cited keys.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIB = HERE / "references.bib"
OUT = HERE / "references_jcaa.tex"


# ---------------------------------------------------------------- parsing
def parse_bib(text):
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        etype, key = m.group(1).lower(), m.group(2)
        i, depth = text.index("{", m.start()), 0
        j = i
        while True:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                break
            j += 1
        body = text[m.end():j]
        fields, k = {}, 0
        while True:
            fm = re.compile(r"\s*(\w+)\s*=\s*").match(body, k)
            if not fm:
                break
            name, k = fm.group(1).lower(), fm.end()
            if body[k] == "{":
                d, s = 0, k
                while True:
                    d += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                    if d == 0:
                        break
                val = body[s + 1:k - 1]
            elif body[k] == '"':
                e = body.index('"', k + 1)
                val, k = body[k + 1:e], e + 1
            else:
                vm = re.compile(r"[^,\s]+").match(body, k)
                val, k = vm.group(0), vm.end()
            fields[name] = re.sub(r"\s+", " ", val).strip()
            cm = re.compile(r"\s*,").match(body, k)
            if not cm:
                break
            k = cm.end()
        entries[key] = (etype, fields)
    return entries


def cited_keys(tex):
    keys = []
    body = "\n".join(l for l in tex.split("\n") if not l.lstrip().startswith("%"))
    for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", body):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


# ---------------------------------------------------------------- formatting
def unprotect(s):
    """Drop case-protecting braces ({R}andom -> Random) but keep accent groups such as {\\'e}."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\{([^{}\\]*)\}", r"\1", s)
    return s


def split_authors(field):
    out = []
    for a in re.split(r"\s+and\s+", field):
        a = a.strip()
        if a.startswith("{") and a.endswith("}"):          # corporate author
            out.append((unprotect(a[1:-1]), ""))
        elif "," in a:
            last, first = a.split(",", 1)
            out.append((last.strip(), first.strip()))
        else:
            parts = a.split()
            out.append((parts[-1], " ".join(parts[:-1])))
    return out


def initials(first):
    """'Jacob A.' -> 'JA'; 'Maria Elena' -> 'ME'; 'Jean-Pierre' -> 'J-P'; 'Jacob McC. M.' -> 'JMcCM'."""
    if not first:
        return ""
    out = []
    for tok in first.replace("~", " ").split():
        sub = []
        for part in tok.split("-"):
            p = part.strip(".")
            if not p:
                continue
            if p.startswith("Mc"):
                sub.append(p.rstrip("."))
                continue
            m = re.match(r"(\{?\\.\{?\w\}?\}?|\{?\w)", p)         # keep an accent macro with its letter
            sub.append(m.group(1).lstrip("{") if m else p[0])
        out.append("-".join(sub))
    return "".join(out)


def surname_tex(last):
    return unprotect(last)


def list_names(auths):
    names = [f"{surname_tex(l)}, {initials(f)}" if f else surname_tex(l) for l, f in auths]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def cite_label(auths):
    s = [surname_tex(l) for l, _ in auths]
    if len(s) == 1:
        return s[0]
    if len(s) <= 3:
        return ", ".join(s[:-1]) + " \\& " + s[-1]
    return s[0] + " et~al."


def end(s):
    s = s.rstrip()
    return s if s.endswith((".", "?", "!")) else s + "."


def pages(f, etype):
    p = f.get("pages", "")
    if not p:
        return ""
    p = p.replace("--", "–").replace("-", "–")
    if etype == "article" and "–" not in p:
        return f"Article {p}"                    # article numbers are not page ranges
    return p


def doi(f):
    return f" DOI: {f['doi']}" if f.get("doi") else ""


def format_entry(etype, f):
    auths = split_authors(f.get("author", f.get("editor", "")))
    head = f"{list_names(auths)}. {f.get('year', 'n.d.')}{{suffix}}. "
    title = unprotect(f.get("title", ""))
    if etype == "article":
        vol = f.get("volume", "")
        iss = f"({f['number'].replace('--', '–')})" if f.get("number") else ""
        pg = pages(f, etype)
        loc = ": ".join(x for x in (f"{vol}{iss}", pg) if x)      # no volume (e.g. PaleoAnthropology) -> pages only
        body = f"{end(title)} \\textit{{{unprotect(f.get('journal', ''))}}}" + (f", {loc}" if loc else "") + "."
    elif etype in ("incollection", "inproceedings", "inbook"):
        eds = split_authors(f["editor"]) if f.get("editor") else []
        ed = (list_names(eds) + (" (eds.) " if len(eds) > 1 else " (ed.) ")) if eds else ""
        book = unprotect(f.get("booktitle", ""))
        place = ": ".join(x for x in (f.get("address", ""), unprotect(f.get("publisher", ""))) if x)
        pg = pages(f, etype)
        body = f"{end(title)} In: {ed}\\textit{{{book}}}." + (f" {place}" if place else "") + (f", pp. {pg}" if pg else "") + "."
    elif etype == "book":
        place = ": ".join(x for x in (f.get("address", ""), unprotect(f.get("publisher", ""))) if x)
        body = f"\\textit{{{end(title)}}}" + (f" {place}." if place else "")
    else:
        extra = unprotect(f.get("howpublished", f.get("note", f.get("publisher", ""))))
        body = end(title) + (f" {end(extra)}" if extra else "")
    out = head + body.replace("..", ".") + doi(f)
    return out.replace("{\\'i}", "{\\'{\\i}}")              # accented dotless i (Dudík, Jiménez)


def sort_key(item):
    key, (etype, f) = item
    a = split_authors(f.get("author", f.get("editor", "")))
    plain = lambda s: re.sub(r"[^a-z ]", "", re.sub(r"\\.", "", s.lower()))
    return ([plain(l) for l, _ in a], f.get("year", ""), plain(f.get("title", "")))


def build(tex_path):
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    keys = cited_keys(Path(tex_path).read_text(encoding="utf-8"))
    missing = [k for k in keys if k not in entries]
    if missing:
        sys.exit(f"cited but not in references.bib: {missing}")
    items = sorted(((k, entries[k]) for k in keys), key=sort_key)
    # Same in-text label and year -> 2013a, 2013b (JCAA A10).
    seen = {}
    for k, (et, f) in items:
        seen.setdefault((cite_label(split_authors(f.get("author", f.get("editor", "")))), f.get("year")), []).append(k)
    suffix = {k: "abcdefgh"[i] for ks in seen.values() if len(ks) > 1 for i, k in enumerate(ks)}
    # Ragged right: long DOIs and author lists otherwise stretch the word spacing of justified lines.
    lines = ["% Generated by build_jcaa_bibliography.py from references.bib -- do not edit by hand.",
             "\\begingroup\\raggedright", "\\begin{thebibliography}{%d}" % len(items), ""]
    plain = []
    for k, (et, f) in items:
        auths = split_authors(f.get("author", f.get("editor", "")))
        lab = cite_label(auths)
        yr = f.get("year", "n.d.") + suffix.get(k, "")
        text = format_entry(et, f).replace("{suffix}", suffix.get(k, ""))
        lines += [f"\\bibitem[{lab}({yr})]{{{k}}}", text, ""]
        plain.append(text)
    lines += ["\\end{thebibliography}", "\\endgroup"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return items, plain


def apply(tex_path):
    p = Path(tex_path)
    t = p.read_text(encoding="utf-8")
    t2 = t.replace("\\usepackage[round]{natbib}", "\\usepackage[round,sort]{natbib}\n\\setcitestyle{aysep={},yysep={;}}")
    t2 = t2.replace("\\bibliographystyle{plainnat}\n", "")
    t2 = t2.replace("\\bibliography{references}", "\\input{references_jcaa.tex}")
    if t2 != t:
        p.write_text(t2, encoding="utf-8")
    return t2 != t


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    tex = sys.argv[1]
    items, plain = build(tex)
    print(f"{len(items)} cited entries -> {OUT.name}")
    if "--apply" in sys.argv:
        print("tex switched to references_jcaa.tex" if apply(tex) else "tex already uses references_jcaa.tex")
    if "--report" in sys.argv:
        for (k, (et, _)), s in zip(items, plain):
            print(f"[{k} | {et}] {s}")
