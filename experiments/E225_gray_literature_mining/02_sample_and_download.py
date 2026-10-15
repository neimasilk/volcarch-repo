"""E225 step 2 — draw the two pilot arms from the frozen frame and download their PDF galleys (DESIGN §7).

usage: python -X utf8 02_sample_and_download.py
in:    results/frame_oai_20261015.csv (checked against its .sha256 before anything else)
out:   results/pilot_arms.csv            (arm, frame_row, journal, article_id, title, ...)
       results/download_log.csv          (one row per document: galleys tried, status, bytes, local file)
       data/raw/e225_corpus/pdf/<journal>_<article_id>.pdf   (git-ignored)
"""
import csv, hashlib, random, re, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FRAME = HERE / "results" / "frame_oai_20261015.csv"
PDF_DIR = REPO / "data" / "raw" / "e225_corpus" / "pdf"
UA = {"User-Agent": "Mozilla/5.0 (research; VOLCARCH E225; contact amien@ubhinus.ac.id)"}
SEED, N_A, N_B_MAX = 20261015, 150, 150
DATING = re.compile(r"radiokarbon|radiocarbon|C-?14|carbon dating|pertanggalan|dating|\bBP\b|Before Present|termoluminesen|OSL|AMS",
                    re.I)

want = FRAME.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
assert hashlib.sha256(FRAME.read_bytes()).hexdigest() == want, "frame changed since it was frozen"
frame = list(csv.DictReader(FRAME.open(encoding="utf-8")))

rng = random.Random(SEED)
arm_a = sorted(rng.sample(range(len(frame)), N_A))
a_set = set(arm_a)
arm_b = [i for i, r in enumerate(frame) if i not in a_set and DATING.search(r["title"] + " " + r["abstract"])][:N_B_MAX]
n_b_all = sum(1 for i, r in enumerate(frame) if i not in a_set and DATING.search(r["title"] + " " + r["abstract"]))
print(f"arm A {len(arm_a)}; arm B {len(arm_b)} (of {n_b_all} matching the dating regex outside A)")


def get(url, binary=False):
    for attempt in range(3):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90)
            data = r.read()
            return r.headers.get("Content-Type", ""), (data if binary else data.decode("utf-8", "replace"))
        except Exception as e:
            if attempt == 2:
                return "ERR " + type(e).__name__ + " " + str(e)[:80], b"" if binary else ""
            time.sleep(4 * (attempt + 1))


PDF_DIR.mkdir(parents=True, exist_ok=True)
arms_rows, log_rows = [], []
for arm, idxs in (("A", arm_a), ("B", arm_b)):
    for i in idxs:
        r = frame[i]
        url = r["article_url"]
        m = re.search(r"/([^/]+)/article/view/(\d+)", url)
        art_id = m.group(2) if m else ""
        arms_rows.append({"arm": arm, "frame_row": i, "journal": r["journal"], "article_id": art_id, "title": r["title"][:200],
                          "date": r["date"], "source": r["source"][:120], "type": r["type"]})
        local = PDF_DIR / f"{r['journal']}_{art_id}.pdf"
        status, nbytes, galleys_tried = "", 0, []
        if local.exists() and local.read_bytes()[:5] == b"%PDF-":
            status, nbytes = "cached", local.stat().st_size
        elif not art_id:
            status = "no article id"
        else:
            ctype, page = get(url)
            galleys = list(dict.fromkeys(re.findall(rf"/article/view/{art_id}/(\d+)", page)))
            if not galleys:
                status = "no galley" if not ctype.startswith("ERR") else ctype
            for g in galleys:
                galleys_tried.append(g)
                ctype, data = get(f"https://ejournal.brin.go.id/{r['journal']}/article/download/{art_id}/{g}", binary=True)
                if data[:5] == b"%PDF-":
                    local.write_bytes(data)
                    status, nbytes = "ok", len(data)
                    break
                status = f"not pdf ({ctype[:40]})"
            time.sleep(1)
        log_rows.append({"arm": arm, "frame_row": i, "journal": r["journal"], "article_id": art_id, "status": status, "bytes": nbytes,
                         "galleys_tried": " ".join(galleys_tried), "file": local.name if status in ("ok", "cached") else ""})
    print(arm, "done:", sum(1 for x in log_rows if x["arm"] == arm and x["status"] in ("ok", "cached")), "PDFs of", len(idxs))

for name, rows in (("pilot_arms.csv", arms_rows), ("download_log.csv", log_rows)):
    with (HERE / "results" / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print("statuses:", {s: sum(1 for x in log_rows if x["status"] == s) for s in sorted({x["status"] for x in log_rows})})
