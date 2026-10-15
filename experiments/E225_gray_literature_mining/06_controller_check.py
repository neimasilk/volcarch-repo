"""E225 step 6 — controller check (DESIGN §7: "the controller checks 10 random verified records against the PDF") and a
stricter re-reading of the tier-A rule, written after step 5 had run (post hoc; reported next to the frozen result, never
instead of it).

usage: python -X utf8 06_controller_check.py [--render OUTDIR]
in:    results/e225_records.csv, data/raw/e225_corpus/pdf/
out:   results/controller_sample.csv   (random10_verified: the 10 records of DESIGN §7, seed 20261015, both arms;
                                        armA_tierA_all and random30_counted: see below; judged by hand in controller_judgments.csv)
       results/e225_summary_strict.json
       OUTDIR/<n>_<doc>_p<page>.png    (page images for the visual check; kept outside the repo, the PDFs are not ours)

Why a strict variant: reading the step-5 output showed two places where 05_verify_classify.py is looser than DESIGN §7.
  1. §7 tier A = radiometric "with value and error"; 05 accepts a radiometric BC/AD date with no error and no range.
  2. §7 counts a calibrated range only when "stated by the authors"; 05 takes cal_range_end_ce from the model, which
     sometimes converts a single cal BP value to CE itself. Strict: the range needs a start and an end that differ, and the
     end must be written in the quote (as CE/BC digits or as cal BP digits).
Everything else is the frozen rule.
"""
import csv, json, random, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PDF_DIR = REPO / "data" / "raw" / "e225_corpus" / "pdf"
SEED = 20261015

rows = list(csv.DictReader((HERE / "results" / "e225_records.csv").open(encoding="utf-8")))
verified = [r for r in rows if r["verified"] == "1"]
sample = random.Random(SEED).sample(verified, 10)
# Two further sets, added after the 10-record check found 3 of its 5 counted records wrong: every tier-A record counted in
# arm A (the rate criterion rests on them) and 30 random counted records (to estimate how many of O2 hold up)
counted = [r for r in rows if r["counted"] == "1"]
sets = {"random10_verified": sample,
        "armA_tierA_all": [r for r in counted if r["tier"] == "A" and "A" in r["arms"].split("+")],
        "random30_counted": random.Random(SEED).sample(counted, 30)}
cols = ["set", "n", "doc", "arms", "page", "site", "admin_location", "method", "lab_code", "age_value", "age_error", "age_unit",
        "cal_range_start_ce", "cal_range_end_ce", "archaeological", "tier", "pre400", "counted", "quote"]
with (HERE / "results" / "controller_sample.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for name, recs in sets.items():
        for n, r in enumerate(recs, 1):
            w.writerow({"set": name, "n": n, **{c: r[c] for c in cols[2:]}})

if "--render" in sys.argv:
    import fitz
    out = Path(sys.argv[sys.argv.index("--render") + 1])
    out.mkdir(parents=True, exist_ok=True)
    for n, r in enumerate(sample, 1):
        doc = fitz.open(PDF_DIR / (r["doc"] + ".pdf"))
        p = int(float(r["page"])) - 1 if r["page"] else 0
        doc[p].get_pixmap(dpi=110).save(out / f"{n:02d}_{r['doc']}_p{p + 1}.png")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def digits(q):
    return re.sub(r"[.,\s]", "", q or "")


def strict(r):
    """Return (counted_strict, reason). Tier B and the guard are unchanged; only tier A is re-read."""
    if r["counted"] != "1":
        return 0, "not counted by the frozen rule"
    if r["tier"] != "A":
        return 1, "tier B, unchanged"
    age, err = num(r["age_value"]), num(r["age_error"])
    start, end = num(r["cal_range_start_ce"]), num(r["cal_range_end_ce"])
    q = digits(r["quote"])
    stated_range = (start is not None and end is not None and start != end
                    and (str(int(abs(end))) in q or str(int(1950 - end)) in q))
    if err is None and not stated_range:
        return 0, "radiometric without error or stated range"
    if end is not None and not stated_range and err is not None:
        # the model's range is not in the quote: fall back to the uncalibrated rule on value and error
        unit = r["age_unit"].lower()
        if unit == "bp":
            return int(age - 2 * err >= 1750), "range not stated; uncalibrated rule"
        if unit == "cal bp":
            return int(age - 2 * err > 1550), "range not stated; cal BP rule"
        return 0, "range not stated; unit not BP"
    return 1, "frozen rule holds"


arms_docs = {}
for r in csv.DictReader((HERE / "results" / "pilot_arms.csv").open(encoding="utf-8")):
    arms_docs.setdefault(r["arm"], set()).add(f"{r['journal']}_{r['article_id']}")
readable = {Path(r["file"]).stem for r in csv.DictReader((HERE / "results" / "text_log.csv").open(encoding="utf-8")) if r["readable"] == "1"}


def key(r):
    return (r["site"].strip().lower(), r["lab_code"].strip().lower(), r["age_value"], r["age_unit"])


res = {}
for r in rows:
    r["counted_strict"], r["strict_reason"] = strict(r)
for arm in ("A", "B"):
    recs = [r for r in rows if arm in r["arms"].split("+") and r["counted_strict"]]
    n_read = len(arms_docs[arm] & readable)
    tA = [r for r in recs if r["tier"] == "A"]
    res[arm] = {"documents_readable": n_read, "records_counted_strict": len(recs), "of_which_tier_A": len(tA),
                "tierA_per_100_readable_docs": round(100 * len(tA) / n_read, 2),
                "docs_with_ge1_tierA": len({r["doc"] for r in tA}),
                "dropped_from_frozen": sum(1 for r in rows if arm in r["arms"].split("+") and r["counted"] == "1" and not r["counted_strict"])}
res["distinct_counted_strict_A_plus_B"] = len({key(r) for r in rows if r["counted_strict"]})
a = res["A"]
res["verdict_strict"] = ("LANJUT" if a["tierA_per_100_readable_docs"] >= 1 or res["distinct_counted_strict_A_plus_B"] >= 20
                         else "GAGAL (informatif)" if a["of_which_tier_A"] == 0 and res["distinct_counted_strict_A_plus_B"] < 5
                         else "PARSIAL")
res["dropped_reasons"] = {}
for r in rows:
    if r["counted"] == "1" and not r["counted_strict"]:
        res["dropped_reasons"][r["strict_reason"]] = res["dropped_reasons"].get(r["strict_reason"], 0) + 1
(HERE / "results" / "e225_summary_strict.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
