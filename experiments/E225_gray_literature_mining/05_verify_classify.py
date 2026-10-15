"""E225 step 5 — anti-hallucination guard, pre-400 CE tiers, outcomes O1-O4 and the decision (DESIGN §7).

usage: python -X utf8 05_verify_classify.py
in:    results/llm_raw/*.json, data/raw/e225_corpus/text/*.json, results/pilot_arms.csv, download_log.csv, text_log.csv
out:   results/e225_records.csv   (every extracted record with verified flag, tier, pre400 flag, reasons)
       results/e225_summary.json  (O1-O4 per arm and the verdict)
Rules (fixed in DESIGN §7 before the run):
  verified  = the quote, whitespace-normalised, occurs verbatim in the document text AND the age digits occur in the quote
  tier A    = numeric method (C14/AMS/TL/OSL/U-series) with value and error; pre-400 if a stated calibrated range ends < 400 CE,
              or (uncalibrated 14C) age - 2*error >= 1750 BP; near the threshold -> "near", not counted
  tier B    = a numeric date stated by the authors without a lab measurement (e.g. "500 SM", "abad ke-2 M"); pre-400 if < 400 CE
  tier C    = period words only -> never counted (the extraction asks for numbers, so C records should not occur)
Only records with context_is_archaeological = true are counted.
"""
import csv, json, math, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TXT_DIR = REPO / "data" / "raw" / "e225_corpus" / "text"
RAW = HERE / "results" / "llm_raw"
RADIOMETRIC = re.compile(r"c\s*-?\s*14|radio|ams|tl\b|thermo|termo|osl|lumin|u-?series|uranium|ar/ar|k-ar", re.I)


def ws(s):
    s = (s or "").replace("­", "").replace("-\n", "").replace("‑", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def num(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", s):        # 2.450 or 2,450 as a thousands separator
        s = re.sub(r"[.,]", "", s)
    s = s.replace(",", ".")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else None


def digits_in_quote(value, quote):
    if value is None:
        return False
    q = re.sub(r"[.,\s]", "", quote)
    v = str(int(value)) if float(value).is_integer() else str(value).replace(".", "")
    return v in q


def to_ce(value, unit):
    """A single stated date in CE (BCE negative), or None."""
    u = (unit or "").lower()
    if value is None:
        return None
    if u in ("bc", "sm"):
        return -value
    if u in ("ad", "m"):
        return value
    if u == "century bc":
        return -(value - 1) * 100            # end of the century BC, the latest year it can mean
    if u == "century ad":
        return value * 100                   # last year of the century AD
    return None


def classify(rec):
    method = str(rec.get("method") or "")
    unit = str(rec.get("age_unit") or "").strip()
    age, err = num(rec.get("age_value")), num(rec.get("age_error"))
    end = num(rec.get("cal_range_end_ce"))
    radiometric = bool(RADIOMETRIC.search(method)) or bool(rec.get("lab_code"))
    if radiometric and age is not None:
        tier = "A"
        if end is not None:
            pre = "yes" if end < 400 else "no"
            why = f"stated calibrated range ends {end:.0f} CE"
        elif unit.lower() in ("bp",) and err is not None:
            young = age - 2 * err
            pre = "yes" if young >= 1750 else ("near" if young >= 1450 else "no")
            why = f"uncalibrated {age:.0f}±{err:.0f} BP; age-2err = {young:.0f}"
        elif unit.lower() == "cal bp" and err is not None:
            young = age - 2 * err                     # 1550 cal BP = 400 CE: the 2-sigma young end must be older
            pre = "yes" if young > 1550 else ("near" if young > 1350 else "no")
            why = f"{age:.0f}±{err:.0f} cal BP; age-2err = {young:.0f}"
        elif unit.lower() in ("bp", "cal bp", "years ago") and err is None:
            pre, why = "near", "age without error (DESIGN §7: tier A needs value and error)"
        else:
            ce = to_ce(age, unit)
            pre = "unknown" if ce is None else ("yes" if ce < 400 else "no")
            why = f"radiometric, unit {unit or '?'}"
        return tier, pre, why
    ce = to_ce(age, unit)
    if ce is not None:
        return "B", ("yes" if ce < 400 else "no"), f"stated date {ce:.0f} CE"
    if age is not None and unit.lower() in ("bp", "cal bp", "years ago"):
        return "B", ("yes" if age >= 1650 else "no"), f"stated {age:.0f} {unit}"
    return "C", "unknown", "no usable number"


arms = {}
for r in csv.DictReader((HERE / "results" / "pilot_arms.csv").open(encoding="utf-8")):
    arms.setdefault(f"{r['journal']}_{r['article_id']}", set()).add(r["arm"])
readable = {Path(r["file"]).stem for r in csv.DictReader((HERE / "results" / "text_log.csv").open(encoding="utf-8")) if r["readable"] == "1"}

rows, parse_fail = [], []
for raw in sorted(RAW.glob("*.json")):
    stem, chunk = raw.stem.split("__")
    try:
        data = json.loads(raw.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        parse_fail.append(raw.name)
        continue
    if "error" in data:
        parse_fail.append(raw.name)
        continue
    text = ws(" ".join(json.loads((TXT_DIR / (stem + ".json")).read_text(encoding="utf-8"))["pages"]))
    for rec in data.get("dates", []):
        quote = ws(rec.get("quote"))
        age = num(rec.get("age_value"))
        in_text = bool(quote) and quote in text
        verified = in_text and digits_in_quote(age, quote)
        tier, pre, why = classify(rec)
        arch = rec.get("context_is_archaeological") in (True, "true", "True")
        rows.append({"doc": stem, "arms": "+".join(sorted(arms.get(stem, []))), "chunk": chunk,
                     "site": rec.get("site"), "admin_location": rec.get("admin_location"),
                     "coordinates_as_stated": rec.get("coordinates_as_stated"), "method": rec.get("method"),
                     "lab_code": rec.get("lab_code"), "age_value": rec.get("age_value"), "age_error": rec.get("age_error"),
                     "age_unit": rec.get("age_unit"), "cal_range_start_ce": rec.get("cal_range_start_ce"),
                     "cal_range_end_ce": rec.get("cal_range_end_ce"), "material": rec.get("material"),
                     "context": rec.get("context"), "archaeological": int(arch), "page": rec.get("page"),
                     "quote": rec.get("quote"), "quote_in_text": int(in_text), "verified": int(verified),
                     "tier": tier, "pre400": pre, "reason": why,
                     "counted": int(verified and arch and pre == "yes" and tier in ("A", "B"))})

with (HERE / "results" / "e225_records.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["doc"])
    w.writeheader()
    w.writerows(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def key(r):
    return (str(r["site"] or "").strip().lower(), str(r["lab_code"] or "").strip().lower(), str(r["age_value"]), str(r["age_unit"]))


summary = {"llm_files_unparsed": parse_fail}
for arm in ("A", "B"):
    docs = [d for d, a in arms.items() if arm in a and d in readable]
    recs = [r for r in rows if arm in r["arms"].split("+")]
    counted = [r for r in recs if r["counted"]]
    tierA = [r for r in counted if r["tier"].startswith("A")]
    docs_with = {r["doc"] for r in counted}
    summary[arm] = {
        "documents_in_arm": sum(1 for a in arms.values() if arm in a), "documents_readable": len(docs),
        "records_extracted": len(recs), "records_rejected_by_guard": sum(1 for r in recs if not r["verified"]),
        "records_counted_pre400": len(counted), "of_which_tier_A": len(tierA),
        "distinct_counted_pre400": len({key(r) for r in counted}),
        "tierA_per_100_readable_docs": round(100 * len(tierA) / len(docs), 2) if docs else None,
        "docs_with_ge1_counted": len(docs_with),
        "share_docs_with_ge1_counted_wilson95": wilson(len(docs_with), len(docs)),
        "counted_with_location": sum(1 for r in counted if (r["coordinates_as_stated"] or r["admin_location"])),
    }
all_counted = {key(r) for r in rows if r["counted"]}
summary["distinct_counted_pre400_A_plus_B"] = len(all_counted)
a = summary["A"]
if (a["tierA_per_100_readable_docs"] or 0) >= 1 or len(all_counted) >= 20:
    verdict = "LANJUT"
elif a["of_which_tier_A"] == 0 and len(all_counted) < 5:
    verdict = "GAGAL (informatif)"
else:
    verdict = "PARSIAL"
summary["verdict"] = verdict
(HERE / "results" / "e225_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
