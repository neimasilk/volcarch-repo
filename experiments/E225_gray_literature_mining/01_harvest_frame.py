"""E225 step 1 — harvest the corpus frame (DESIGN §7): every OAI-PMH record (oai_dc) of five BRIN archaeology journals.

usage: python -X utf8 01_harvest_frame.py
out:   results/frame_oai_20261015.csv (one row per record, frame order = journal order below, then OAI order)
       results/frame_oai_20261015.sha256
The frame is frozen once written: later steps read it and never re-harvest.
"""
import csv, hashlib, html, re, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "frame_oai_20261015.csv"
JOURNALS = ["berkalaarkeologi", "amerta", "nw", "kalpataru", "purbawidya"]   # DESIGN §7; nw = Naditira Widya
BASE = "https://ejournal.brin.go.id/{j}/oai"
UA = {"User-Agent": "Mozilla/5.0 (research; VOLCARCH E225; contact amien@ubhinus.ac.id)"}


def get(url):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "replace")
        except Exception as e:  # transient network errors: back off and retry
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def field(rec, name):
    vals = [html.unescape(re.sub(r"\s+", " ", v)).strip() for v in re.findall(rf"<dc:{name}[^>]*>(.*?)</dc:{name}>", rec, re.S)]
    return vals


rows = []
for j in JOURNALS:
    url = BASE.format(j=j) + "?verb=ListRecords&metadataPrefix=oai_dc"
    n_j = 0
    while url:
        x = get(url)
        for rec in x.split("<record>")[1:]:
            if 'status="deleted"' in rec.split("</header>")[0]:
                continue
            ident = re.search(r"<identifier>(.*?)</identifier>", rec).group(1)
            types = field(rec, "type")
            urls = [v for v in field(rec, "identifier") if v.startswith("http")]
            rows.append({
                "journal": j,
                "oai_id": ident,
                "article_url": next((u for u in urls if "/article/view/" in u), urls[0] if urls else ""),
                "title": " | ".join(field(rec, "title")),
                "abstract": " | ".join(field(rec, "description")),
                "date": (field(rec, "date") or [""])[0],
                "source": " | ".join(field(rec, "source")),
                "creators": "; ".join(field(rec, "creator")),
                "type": ";".join(types),
                "language": ";".join(field(rec, "language")),
            })
            n_j += 1
        tok = re.search(r"<resumptionToken[^>]*>([^<]+)</resumptionToken>", x)
        url = (BASE.format(j=j) + "?verb=ListRecords&resumptionToken=" + tok.group(1)) if tok else None
        time.sleep(1)
    print(j, n_j, "records")

OUT.parent.mkdir(exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
(OUT.with_suffix(".sha256")).write_text(digest + "  " + OUT.name + "\n", encoding="utf-8")
print("frame rows:", len(rows), "sha256:", digest)
