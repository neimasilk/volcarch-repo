"""E233 step 2: verbatim guard on the two readers, reader agreement, and the final reference (DESIGN §3).

Exits with ADJUDICATION NEEDED (and writes reader/adjudication_todo.csv) until every R1/R2 disagreement has a row in
reader/adjudication.csv. The frame is not read here.
"""
import collections
import csv
import importlib.util
import json
import os
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
PK, RD, RES = (os.path.join(EXP, d) for d in ("packets", "reader", "results"))
ROOT = os.path.abspath(os.path.join(EXP, "..", ".."))
spec = importlib.util.spec_from_file_location(
    "e226", os.path.join(ROOT, "experiments", "E226_t7_detection_calibration", "scripts", "01_build_candidates.py"))
e226 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e226)


def ws(s):
    # whitespace-normalised (DESIGN §3); NFC so that a copied precomposed/decomposed letter is the same letter
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def name_key(name, ins):
    s = name or ""
    if ins == "INSIDENKKurungan":  # Dutch spelling of the 1940 translation (DESIGN §4)
        s = s.lower().replace("oe", "u").replace("dj", "j").replace("tj", "c")
    return e226.key(s)


def kappa(a, b):
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def load_packets():
    occ, tr = {}, {}
    for r in csv.DictReader(open(os.path.join(PK, "index.csv"), encoding="utf-8")):
        ins = r["inscription"]
        tr[ins] = ws(open(os.path.join(PK, f"{ins}.translation.txt"), encoding="utf-8").read())
        for line in open(os.path.join(PK, f"{ins}.md"), encoding="utf-8"):
            m = re.match(r"\| (INS[^ |]+#\d{3}) \|", line)
            if m:
                occ[m.group(1)] = ins
    return occ, tr


def load_reader(rid, occ, tr):
    rows = {}
    for g in ("A", "B"):
        for r in csv.DictReader(open(os.path.join(RD, f"{rid}_{g}.csv"), encoding="utf-8-sig")):
            oid = (r.get("occ_id") or "").strip()
            if oid in rows:
                raise SystemExit(f"{rid}: duplicate {oid}")
            rows[oid] = r
    extra = set(rows) - set(occ)
    if extra:
        raise SystemExit(f"{rid}: unknown occ_id {sorted(extra)[:5]}")
    out, log = {}, []
    for oid, ins in occ.items():
        r = rows.get(oid)
        if r is None:
            out[oid] = dict(code="U", name="", quote="", why="missing row")
            log.append((oid, "missing row"))
            continue
        code = (r.get("ref_code") or "").strip().upper()
        name, quote = ws(r.get("ref_name")), ws(r.get("quote"))
        why = ""
        if code not in ("Y", "N", "U"):
            why = f"bad code {code!r}"
        elif code in ("Y", "N") and (not quote or quote not in tr[ins]):
            why = "quote not in translation"
        elif code == "Y" and (not name or name.lower() not in quote.lower()):
            why = "name not in quote"
        if why:
            log.append((oid, why))
            code = "U"
        out[oid] = dict(code=code, name=name if code == "Y" else "", quote=quote, why=why,
                        note=ws(r.get("note")))
    return out, log


def main():
    occ, tr = load_packets()
    R = {}
    guard = {}
    for rid in ("R1", "R2"):
        R[rid], guard[rid] = load_reader(rid, occ, tr)
    ids = sorted(occ)
    k3, a3 = kappa([R["R1"][i]["code"] for i in ids], [R["R2"][i]["code"] for i in ids])
    both_y = [i for i in ids if R["R1"][i]["code"] == R["R2"][i]["code"] == "Y"]
    same_name = sum(name_key(R["R1"][i]["name"], occ[i]) == name_key(R["R2"][i]["name"], occ[i]) for i in both_y)
    todo = [i for i in ids if R["R1"][i]["code"] != R["R2"][i]["code"] or
            (i in both_y and name_key(R["R1"][i]["name"], occ[i]) != name_key(R["R2"][i]["name"], occ[i]))]
    stats = dict(n=len(ids), guard_failures={k: len(v) for k, v in guard.items()},
                 guard_log={k: v for k, v in guard.items()},
                 counts={k: dict(collections.Counter(R[k][i]["code"] for i in ids)) for k in R},
                 kappa_3cat_R1_R2=round(k3, 3), agree_3cat=round(a3, 3), both_Y=len(both_y),
                 both_Y_same_name=same_name, n_disagreements=len(todo))
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RD, "adjudication_todo.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["occ_id", "R1_code", "R1_name", "R1_quote", "R1_note", "R1_guard",
                    "R2_code", "R2_name", "R2_quote", "R2_note", "R2_guard"])
        for i in todo:
            w.writerow([i] + [R[k][i][f] for k in ("R1", "R2") for f in ("code", "name", "quote", "note", "why")])
    adj_path = os.path.join(RD, "adjudication.csv")
    adj = {r["occ_id"]: r for r in csv.DictReader(open(adj_path, encoding="utf-8"))} if os.path.exists(adj_path) else {}
    missing = [i for i in todo if i not in adj]
    print(json.dumps({k: v for k, v in stats.items() if k != "guard_log"}, indent=1, ensure_ascii=False))
    if missing:
        print(f"ADJUDICATION NEEDED: {len(missing)} of {len(todo)} rows → reader/adjudication_todo.csv")
        sys.exit(2)
    ref = []
    for i in ids:
        if i in adj:
            a = adj[i]
            code, name, quote = a["ref_code"].strip().upper(), ws(a["ref_name"]), ws(a["quote"])
            assert code in ("Y", "N", "U"), i
            if code in ("Y", "N"):
                assert quote in tr[occ[i]], f"adjudication quote not in translation: {i}"
            if code == "Y":
                assert name and name.lower() in quote.lower(), f"adjudication name not in quote: {i}"
            how, reason = "adjudicated", a["reason"]
        else:
            code, name, quote = R["R1"][i]["code"], R["R1"][i]["name"], R["R1"][i]["quote"]
            how, reason = "agreed", ""
        ref.append(dict(occ_id=i, inscription=occ[i], ref_code=code, ref_name=name, quote=quote, how=how,
                        reason=reason))
    with open(os.path.join(RES, "reference.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ref[0]))
        w.writeheader()
        w.writerows(ref)
    stats["reference_counts"] = dict(collections.Counter(r["ref_code"] for r in ref))
    with open(os.path.join(RES, "reader_agreement.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1, ensure_ascii=False)
    print("reference:", stats["reference_counts"])


if __name__ == "__main__":
    main()
