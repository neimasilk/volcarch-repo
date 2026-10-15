"""Reference list in the style of Oceanic Linguistics (Chicago 15th ed., author-date, as the journal's template gives it).

Why this file exists (2026-10-12): until v0.3 the list was set by pandoc with the Unified Style Sheet for Linguistics, which is
not the journal's style ("Oceanic Linguistics 32(1). 1-44. https://doi.org/..." instead of "Oceanic Linguistics 32:1-44.").
The rules implemented here are those of the template (OL-template-1.dotx, sections "References", "Inclusive numbers") and of the
copy editor's stylesheet of 2017:
  journal article   Family, Given, and Given Family. YYYY. Title. Journal Title vol:pp-pp.      (no issue: pagination continuous)
  chapter           ... YYYY. Title. In Book title, ed. by Given Family, pp-pp. Series N. Place: Publisher.
  book              ... YYYY. Title. Rev. ed. Place: Publisher.
  thesis            ... YYYY. Title. PhD thesis, School.   /   PhD dissertation, School.   (USA)
  page ranges abbreviated the Chicago way (101-8, 125-28, 498-510); en-dash; "and", serial comma; no "pp."; no DOI
  a repeated author designation is replaced by three em-dashes

Data come from the two .bib files through pandoc (`-t csljson`), so no bibliographic fact is typed here; only form is.

usage:  python -X utf8 ol_refs.py              prints the list for the keys in CITED: of the current article source
        import ol_refs; ol_refs.reference_list(keys, bibs, markup=False) -> list of strings (in order);
        markup=True marks italics with asterisks for build_docx.py
"""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIBS = [HERE / "references.bib", HERE / "references_reviewer_supplied.bib"]
EM3 = "———"

# pandoc lower-cases series names with the titles; the journal wants series in title capitals.
SERIES = {
    "pacific linguistics": "Pacific Linguistics",
    "nusa: linguistic studies of indonesian and other languages in indonesia":
        "NUSA: Linguistic Studies of Indonesian and Other Languages in Indonesia",
    "oceanic linguistics special publication": "Oceanic Linguistics Special Publication",
    "pacific linguistics, series b": "Pacific Linguistics, Series B",
}
# "PhD dissertation" for schools in the USA, "PhD thesis" otherwise (template, list of examples)
US_SCHOOLS = ("University of Michigan", "University of Hawai", "Yale", "Cornell", "Stanford", "MIT")
FIXES = [("Hawai'i", "Hawai‘i"), ("Hawai’i", "Hawai‘i")]


def load(bibs=BIBS):
    out = {}
    for b in bibs:
        r = subprocess.run(["pandoc", str(b), "-f", "bibtex", "-t", "csljson"], capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise SystemExit(r.stderr)
        for e in json.loads(r.stdout):
            out[e["id"]] = e
    return out


def clean(s):
    s = re.sub(r"</?span[^>]*>", "", s or "")
    s = re.sub(r"</?(i|b|sup|sub)>", "", s)
    return s.strip()


def family(n):
    if "literal" in n:
        return n["literal"]
    part = n.get("non-dropping-particle") or n.get("dropping-particle") or ""
    return (part + " " + n["family"]).strip()


def name(n, inverted):
    if "literal" in n:
        return n["literal"]
    return f"{family(n)}, {n['given']}" if inverted else f"{n['given']} {family(n)}"


def names(lst, invert_first=True):
    """Chicago: more than ten authors -> the first seven and 'et al.' (also when the .bib ends in 'and others')."""
    etal = any(n.get("literal") == "others" for n in lst)
    lst = [n for n in lst if n.get("literal") != "others"]
    if etal or len(lst) > 10:
        lst, etal = lst[:7], True
    parts = [name(n, invert_first and i == 0) for i, n in enumerate(lst)]
    if etal:
        return ", ".join(parts) + ", et al."
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        # the comma before "and" follows an inverted first name (template: "Blust, Robert, and Stephen Trussel")
        return f"{parts[0]}, and {parts[1]}" if invert_first else f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def pages(p):
    """Chicago 15, 9.64, as the template lists it: 46-48, 100-104, 101-8, 1103-4, 125-28, 498-510."""
    p = (p or "").replace("–", "-").replace("--", "-")
    m = re.fullmatch(r"(\d+)-(\d+)", p)
    if not m:
        return p
    a, b = m.group(1), m.group(2)
    ia = int(a)
    if ia < 100 or ia % 100 == 0 or len(a) != len(b):
        return f"{a}–{b}"
    k = 0
    while k < len(a) and a[k] == b[k]:
        k += 1
    changed = b[k:]
    if ia % 100 < 10:                       # 101-109, 1103-1104: the changed part only
        return f"{a}–{changed}"
    if len(changed) < 2:                    # 110-199: two digits, or more as needed
        changed = b[-2:]
    return f"{a}–{changed}"


def edition(e):
    ed = clean(e.get("edition", ""))
    if not ed:
        return ""
    if ed.lower() in ("revised", "rev.", "revised edition"):
        return " Rev. ed."
    if ed.isdigit():
        return f" {ed}{ {'1': 'st', '2': 'nd', '3': 'rd'}.get(ed, 'th') } ed."
    return f" {ed} ed."


def series(e):
    t = clean(e.get("collection-title", ""))
    if not t:
        return ""
    t = SERIES.get(t.lower(), t)
    num = e.get("collection-number") or (e.get("volume") if e["type"] == "chapter" else "") or ""
    return f" {t} {num}.".replace(" .", ".") if num else f" {t}."


def imprint(e):
    place, pub = clean(e.get("publisher-place", "")), clean(e.get("publisher", ""))
    if place and pub:
        return f" {place}: {pub}."
    return f" {pub}." if pub else ""


def author_string(e):
    return names(e.get("author") or e.get("editor") or [])


def body(e, markup=False):
    """Everything after 'Author. '. With markup, the titles of journals, books and collections are wrapped in asterisks
    (italics, as in the template's reference list) and a literal asterisk (a reconstruction in a title) is written \*."""
    esc = (lambda x: x.replace("*", r"\*")) if markup else (lambda x: x)
    it = (lambda x: "*" + esc(x) + "*") if markup else (lambda x: x)
    year = e["issued"]["date-parts"][0][0]
    title = esc(clean(e["title"]))
    t = e["type"]
    if t == "article-journal":
        vol, loc = str(e.get("volume", "")), pages(e.get("page", ""))
        # the submitted .bib (used only for the baseline of the tracked copy) has entries without a volume
        tail = f" {vol}:{loc}" if vol and loc else (f" {vol}{loc}" if vol or loc else "")
        s = f"{year}. {title}. {it(clean(e['container-title']))}{tail}."
    elif t == "book":
        s = f"{year}. {it(clean(e['title']))}.{edition(e)}{series(e)}{imprint(e)}"
    elif t in ("chapter", "paper-conference"):
        s = f"{year}. {title}. In {it(clean(e['container-title']))}"
        if e.get("editor"):
            s += ", ed. by " + names(e["editor"], invert_first=False)
        if e.get("page"):
            s += ", " + pages(e["page"])
        s += "." + series(e) + imprint(e)
    elif t == "thesis":
        school = clean(e.get("publisher", ""))
        kind = "PhD dissertation" if school.startswith(US_SCHOOLS) else "PhD thesis"
        s = f"{year}. {title}. {kind}, {school}."
    else:
        raise SystemExit(f"type not handled: {t} ({e['id']})")
    for a, b in FIXES:
        s = s.replace(a, b)
    return s


def sort_key(e):
    lst = [n for n in (e.get("author") or e.get("editor")) if n.get("literal") != "others"]
    first = lst[0]
    # a single-author work precedes a multi-author work that begins with the same family name (Chicago 16.103 ff.)
    return (family(first).lower(), 0 if len(lst) == 1 else 1,
            [family(n).lower() for n in lst[1:]], e["issued"]["date-parts"][0][0], clean(e["title"]).lower())


def reference_list(keys, bibs=BIBS, markup=False):
    db = load(bibs)
    missing = [k for k in keys if k not in db]
    if missing:
        raise SystemExit(f"keys not in the .bib files: {missing}")
    out, prev = [], None
    for e in sorted((db[k] for k in keys), key=sort_key):
        a = author_string(e)
        head = EM3 if a == prev else a
        prev = a
        sep = " " if head.endswith(".") else ". "
        out.append(head + sep + body(e, markup))
    return out


if __name__ == "__main__":
    src = sorted(HERE.glob("P8_revision_v0.*.md"))[-1] if len(sys.argv) < 2 else Path(sys.argv[1])
    m = re.search(r"^CITED:\s*(.*)$", src.read_text(encoding="utf-8"), re.M)
    keys = [k.strip() for k in re.split(r"[,\s]+", m.group(1)) if k.strip()]
    print(f"# {src.name}: {len(keys)} keys")
    for line in reference_list(keys):
        print(line)
    # self-test of the page rule against the template's own examples
    for a, want in [("46-48", "46–48"), ("100-104", "100–104"), ("101-108", "101–8"), ("1103-1104", "1103–4"),
                    ("125-128", "125–28"), ("498-510", "498–510"), ("205-209", "205–9"), ("359-409", "359–409"),
                    ("133-152", "133–52"), ("131-134", "131–34"), ("269-296", "269–96"), ("435-490", "435–90")]:
        assert pages(a) == want, (a, pages(a), want)
    print("page-rule self-test: OK (12 examples from the template)")
