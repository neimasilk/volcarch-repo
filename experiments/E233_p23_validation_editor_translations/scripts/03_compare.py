"""E233 step 3: frame vs reference — M1–M4, representativeness check, decision (DESIGN §4–5).

Writes results/disagreements_todo.csv; the M3/M4 numbers need results/error_taxonomy.csv (orchestrator, one row per
disagreement, with the reason) and exit with TAXONOMY NEEDED until it covers every disagreement.
"""
import collections
import csv
import importlib.util
import json
import math
import os
import re
import sys

from scipy.stats import fisher_exact

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RES = os.path.join(EXP, "results")
ROOT = os.path.abspath(os.path.join(EXP, "..", ".."))
E226 = os.path.join(ROOT, "experiments", "E226_t7_detection_calibration")
SAMPLE = os.path.join(ROOT, "papers", "P23_sima_village_toponyms", "validation", "sample_ids.csv")
spec = importlib.util.spec_from_file_location("ref", os.path.join(HERE, "02_reference.py"))
ref_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref_mod)
key = lambda name, ins: ref_mod.name_key(name, ins)
CATS = {"title_as_village", "person_as_village", "generic_as_village", "missed_village", "boundary", "wrong_name",
        "spelling", "reference_doubtful"}


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def tokens(name, ins):
    return {key(t, ins) for t in re.split(r"[\s\-]+", name or "") if key(t, ins)}


def name_class(fn, rn, ins):
    a, b = key(fn, ins), key(rn, ins)
    if a == b:
        return "exact"
    ta, tb = tokens(fn, ins), tokens(rn, ins)
    if a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a) or (ta and tb and (ta <= tb or tb <= ta)):
        return "boundary"
    return "other"


def metrics(rows):
    use = [r for r in rows if r["ref"] in ("Y", "N")]
    if not use:
        return {}
    fy = [r["frame"] == "Y" for r in use]
    ry = [r["ref"] == "Y" for r in use]
    k, po = ref_mod.kappa(fy, ry)
    tab = collections.Counter((f, g) for f, g in zip(fy, ry))
    return dict(n=len(use), n_U_excluded=len(rows) - len(use), kappa_Y=round(k, 3), agree_Y=round(po, 3),
                frameY_refY=tab[(True, True)], frameY_refN=tab[(True, False)], frameNotY_refY=tab[(False, True)],
                frameNotY_refN=tab[(False, False)])


def main():
    ref = {r["occ_id"]: r for r in csv.DictReader(open(os.path.join(RES, "reference.csv"), encoding="utf-8"))}
    frame = {r["occ_id"]: r for r in csv.DictReader(open(os.path.join(E226, "frame", "final_occurrences.csv"),
                                                          encoding="utf-8"))}
    sample = {r["occ_id"] for r in csv.DictReader(open(SAMPLE, encoding="utf-8"))}
    rows = []
    for oid, r in sorted(ref.items()):
        f = frame[oid]
        row = dict(occ_id=oid, ins=r["inscription"], frame=f["code"], frame_name=f["name"], ref=r["ref_code"],
                   ref_name=r["ref_name"], quote=r["quote"], in_sample=oid in sample, name_class="")
        if row["frame"] == "Y" and row["ref"] == "Y":
            row["name_class"] = name_class(f["name"], r["ref_name"], r["inscription"])
        rows.append(row)
    m1 = metrics(rows)
    m1_sample = metrics([r for r in rows if r["in_sample"]])
    both_y = [r for r in rows if r["name_class"]]
    m2 = dict(collections.Counter(r["name_class"] for r in both_y), n_both_Y=len(both_y))
    # disagreements: code mismatch (U excluded) or both Y with a non-exact name
    dis = [r for r in rows if r["ref"] in ("Y", "N") and
           ((r["frame"] == "Y") != (r["ref"] == "Y") or r["name_class"] in ("boundary", "other"))]
    kw = {r["occ_id"]: r for r in csv.DictReader(open(os.path.join(E226, "frame", "coder_input.csv"), encoding="utf-8"))}
    with open(os.path.join(RES, "disagreements_todo.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["occ_id", "frame", "frame_name", "ref", "ref_name", "name_class", "quote", "kwic"])
        for r in dis:
            k = kw[r["occ_id"]]
            w.writerow([r["occ_id"], r["frame"], r["frame_name"], r["ref"], r["ref_name"], r["name_class"], r["quote"],
                        f"{k['kwic_left']} [{k['noun']}] {k['kwic_right']}"])
    # representativeness: coder A–B agreement (Y vs not-Y) inside vs outside the translated inscriptions
    A = {r["occ_id"]: r["village_named"] == "Y" for r in csv.DictReader(open(os.path.join(E226, "frame", "coder_A.csv"), encoding="utf-8"))}
    B = {r["occ_id"]: r["village_named"] == "Y" for r in csv.DictReader(open(os.path.join(E226, "frame", "coder_B.csv"), encoding="utf-8"))}
    ins_in = {r["ins"] for r in rows}
    t = [[0, 0], [0, 0]]  # [in, out] x [agree, disagree]
    for oid in A:
        t[0 if oid.split("#")[0] in ins_in else 1][0 if A[oid] == B[oid] else 1] += 1
    p_rep = fisher_exact(t)[1]
    rep = dict(in_agree=t[0][0], in_disagree=t[0][1], out_agree=t[1][0], out_disagree=t[1][1],
               in_rate=round(t[0][0] / sum(t[0]), 4), out_rate=round(t[1][0] / sum(t[1]), 4),
               fisher_p=round(p_rep, 4), extrapolable=bool(p_rep >= 0.05))
    out = dict(M1=m1, M1_random100_subset=m1_sample, M2=m2, n_disagreements=len(dis), representativeness=rep)
    tax_path = os.path.join(RES, "error_taxonomy.csv")
    tax = {r["occ_id"]: r for r in csv.DictReader(open(tax_path, encoding="utf-8"))} if os.path.exists(tax_path) else {}
    missing = [r["occ_id"] for r in dis if r["occ_id"] not in tax]
    if missing:
        print(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"TAXONOMY NEEDED: {len(missing)} of {len(dis)} disagreements → results/disagreements_todo.csv")
        sys.exit(2)
    bad = [o for o, r in tax.items() if r["category"] not in CATS]
    assert not bad, f"unknown categories: {bad}"
    cat = {r["occ_id"]: tax[r["occ_id"]]["category"] for r in dis}
    out["M3"] = dict(collections.Counter(cat.values()))
    # M4 strict: among frame-Y rows with a Y/N reference, wrong = reference N (unless the reference is doubtful),
    # boundary, wrong name. Spelling and reference-doubtful rows count as right.
    fy = [r for r in rows if r["frame"] == "Y" and r["ref"] in ("Y", "N")]
    wrong = [r for r in fy if cat.get(r["occ_id"]) in ("title_as_village", "person_as_village", "generic_as_village",
                                                      "boundary", "wrong_name")]
    wrong_len = [r for r in wrong if cat[r["occ_id"]] != "boundary"]
    out["M4"] = dict(n_frame_Y=len(fy), strict_wrong=len(wrong), strict_share=round(len(wrong) / len(fy), 4),
                     strict_wilson95=wilson(len(wrong), len(fy)), lenient_wrong=len(wrong_len),
                     lenient_share=round(len(wrong_len) / len(fy), 4), lenient_wilson95=wilson(len(wrong_len), len(fy)))
    fail = m1["kappa_Y"] < 0.70 or out["M4"]["strict_share"] > 0.15
    out["decision"] = "FAIL" if fail else "PASS"
    out["decision_rule"] = "FAIL if kappa_Y < 0.70 or strict M4 share > 0.15 (DESIGN §5)"
    with open(os.path.join(RES, "e233_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
