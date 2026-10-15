"""E225 step 4 — page-level prefilter for numeric dates, then DeepSeek extraction on the pages that pass (DESIGN §7).

usage: python -X utf8 04_llm_extract.py [--limit N]
in:    results/text_log.csv, data/raw/e225_corpus/text/*.json, repo .env (DEEPSEEK_API; never printed)
out:   results/prefilter_log.csv              (file, pages, pages_matched)
       results/llm_raw/<file>__<chunk>.json   (raw model output, kept for audit)
       results/llm_usage.csv                  (tokens per call, for the cost record)
Only pages with a prefilter match, plus one page on each side, are sent; each chunk ≤ ~24k characters with [[PAGE n]] markers.
"""
import csv, json, re, sys, time
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TXT_DIR = REPO / "data" / "raw" / "e225_corpus" / "text"
RAW = HERE / "results" / "llm_raw"
RAW.mkdir(parents=True, exist_ok=True)
MODEL, CHUNK = "deepseek-chat", 24000

DATE_RX = re.compile(
    r"\d[\d.,]*\s*(?:±|\+/-|\+-|±)\s*\d+"                                   # 2450 ± 60
    r"|\d[\d.,]*\s*(?:cal\.?\s*)?(?:B\.?\s?P\.?|BP|BC|AD|SM|S\.M\.|sebelum Masehi|kya|ka\b|tahun yang lalu|tahun lalu)"
    r"|abad\s+ke-?\s*\d+\s*(?:SM|S\.M\.|sebelum|M\b|Masehi)"
    r"|\b(?:Wk|Beta|ANU|ANUA|OZ[A-Z]?|NZA|SANU|GrN|GrA|Gr|Poz|UGa|BGS|IAEA|SUA|KIA|AA|UCIAMS|D-AMS|PLD|Pta|SNU|TKa|CAMS|Lv|Ua)\s*-\s*\d{2,6}",
    re.I)

SYSTEM = """You extract numeric dates from Indonesian archaeology journal articles (Indonesian or English).
Return JSON: {"dates": [ ... ]}. One object per distinct dated sample or dated statement that refers to an archaeological
context in the text (an occupation layer, artifact, burial, human remains, structure, or site occupation). Do NOT include
dates of purely geological or palaeoenvironmental samples unless the text ties them to human activity; do NOT include
publication years or the years in literature citations. Include dates of any age (old or young); classification is done later.
Fields (use null when the text does not state it; never infer a value that is not written):
 site, admin_location (village/district/regency/province as written), coordinates_as_stated, method (e.g. C14, AMS, TL,
 OSL, U-series, stated/typological), lab_code, age_value (number as written), age_error (number), age_unit (one of: BP,
 cal BP, BC, AD, SM, M, years ago, century BC, century AD), is_calibrated (true/false/null), cal_range_start_ce,
 cal_range_end_ce (integers, BCE negative, only if the text states a calibrated range), material, context (short),
 context_is_archaeological (true/false), page (the [[PAGE n]] number where it appears),
 quote (a VERBATIM excerpt of 10-40 words copied exactly from the text, containing the number).
If there are no such dates, return {"dates": []}."""


def load_key():
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("DEEPSEEK_API") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("DEEPSEEK_API missing in .env")


def call(key, text):
    payload = {"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    for attempt in range(4):
        try:
            r = requests.post("https://api.deepseek.com/v1/chat/completions", json=payload, timeout=300,
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"], j.get("usage", {})
        except Exception as e:
            if attempt == 3:
                return json.dumps({"error": type(e).__name__ + " " + str(e)[:200]}), {}
            time.sleep(10 * (attempt + 1))


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    key = load_key()
    readable = [r["file"] for r in csv.DictReader((HERE / "results" / "text_log.csv").open(encoding="utf-8")) if r["readable"] == "1"]
    pre_rows, usage_rows, n_calls = [], [], 0
    for name in readable[:limit]:
        stem = Path(name).stem
        pages = json.loads((TXT_DIR / (stem + ".json")).read_text(encoding="utf-8"))["pages"]
        hit = [i for i, p in enumerate(pages) if DATE_RX.search(p)]
        pre_rows.append({"file": name, "pages": len(pages), "pages_matched": len(hit)})
        if not hit:
            continue
        keep = sorted({j for i in hit for j in (i - 1, i, i + 1) if 0 <= j < len(pages)})
        chunks, cur = [], ""
        for i in keep:
            block = f"\n[[PAGE {i + 1}]]\n{pages[i]}"
            if cur and len(cur) + len(block) > CHUNK:
                chunks.append(cur)
                cur = ""
            cur += block
        chunks.append(cur)
        for k, ch in enumerate(chunks):
            out = RAW / f"{stem}__{k}.json"
            if out.exists():
                continue
            content, usage = call(key, ch)
            out.write_text(content, encoding="utf-8")
            usage_rows.append({"file": name, "chunk": k, "chars": len(ch), **{u: usage.get(u, "") for u in
                               ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")}})
            n_calls += 1
    with (HERE / "results" / "prefilter_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "pages", "pages_matched"])
        w.writeheader()
        w.writerows(pre_rows)
    up = HERE / "results" / "llm_usage.csv"
    new = not up.exists()
    with up.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "chunk", "chars", "prompt_tokens", "completion_tokens",
                                          "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"])
        if new:
            w.writeheader()
        w.writerows(usage_rows)
    print(f"{len(pre_rows)} readable docs; {sum(1 for r in pre_rows if r['pages_matched'])} with date-like pages; {n_calls} calls")


if __name__ == "__main__":
    main()
