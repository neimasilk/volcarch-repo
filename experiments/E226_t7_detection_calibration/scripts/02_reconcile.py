"""E226 step 1b: two-coder agreement, adjudication queue, and the village frame V_r.

Run once with only coder_A/B present -> writes frame/adjudication_todo.csv and agreement stats.
Run again after frame/adjudication.csv exists -> writes frame/final_occurrences.csv and
results/t7_village_frame.csv.
"""
import collections
import csv
import json
import os
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.join(os.path.dirname(__file__), "..")
F = lambda *p: os.path.join(HERE, *p)
PRIMARY = {"Kedu", "Prambanan-Mataram", "Kedu/Prambanan (E082 area label)"}


def read(p):
    return {r["occ_id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def nkey(s):
    """Spelling-insensitive key: OJ diacritics stripped, w->v, b-/v- treated alike, doubled consonants folded."""
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("w", "v").replace("ə", "e")
    s = re.sub(r"[^a-z]", "", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def kappa(pairs, cats):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in cats) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def main():
    cand = read(F("frame", "candidates_kwic.csv"))
    A, B = read(F("frame", "coder_A.csv")), read(F("frame", "coder_B.csv"))
    assert set(A) == set(B) == set(cand), "coder files do not cover the candidate set exactly"
    ids = list(cand)
    k3, po3 = kappa([(A[i]["village_named"], B[i]["village_named"]) for i in ids], ["Y", "N", "?"])
    kb, pob = kappa([(A[i]["village_named"] == "Y", B[i]["village_named"] == "Y") for i in ids], [True, False])
    both = [i for i in ids if A[i]["village_named"] == B[i]["village_named"] == "Y"]
    same = [i for i in both if nkey(A[i]["name"]) == nkey(B[i]["name"])]
    na = {nkey(A[i]["name"]) for i in ids if A[i]["village_named"] == "Y"}
    nb = {nkey(B[i]["name"]) for i in ids if B[i]["village_named"] == "Y"}
    stats = dict(n=len(ids), kappa_3cat=round(k3, 3), agree_3cat=round(po3, 3),
                 kappa_Y_vs_notY=round(kb, 3), agree_Y=round(pob, 3),
                 A_counts=collections.Counter(A[i]["village_named"] for i in ids),
                 B_counts=collections.Counter(B[i]["village_named"] for i in ids),
                 both_Y=len(both), both_Y_same_name=len(same),
                 name_jaccard=round(len(na & nb) / len(na | nb), 3), names_A=len(na), names_B=len(nb))
    json.dump(stats, open(F("results", "coder_agreement.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, ensure_ascii=False))

    todo = [i for i in ids if not (A[i]["village_named"] == B[i]["village_named"] == "N") and i not in same]
    with open(F("frame", "adjudication_todo.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["occ_id", "kwic", "A_code", "A_name", "A_note", "B_code", "B_name", "B_note"])
        for i in todo:
            c = cand[i]
            w.writerow([i, f"{c['kwic_left']} [[{c['noun']}]] {c['kwic_right']}",
                        A[i]["village_named"], A[i]["name"], A[i]["note"],
                        B[i]["village_named"], B[i]["name"], B[i]["note"]])
    print("adjudication queue:", len(todo))

    adj_p = F("frame", "adjudication.csv")
    if not os.path.exists(adj_p):
        return
    adj = read(adj_p)
    missing = set(todo) - set(adj)
    assert not missing, f"adjudication missing {len(missing)} rows"
    final = []
    for i in ids:
        if i in adj:
            code, name, how = adj[i]["final_code"], adj[i]["final_name"], "adjudicated"
        else:
            code, name, how = A[i]["village_named"], A[i]["name"], "agreed"
        c = cand[i]
        final.append(dict(occ_id=i, inscription=c["inscription"], date=c["date"], date_src=c["date_src"],
                          region=c["region"], code=code, name=name.strip().lower(), key=nkey(name),
                          watak=A[i]["watak"] or B[i]["watak"], role=A[i]["role"] or B[i]["role"], how=how))
    with open(F("frame", "final_occurrences.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(final[0]))
        w.writeheader()
        w.writerows(final)
    vil = collections.defaultdict(lambda: dict(spellings=collections.Counter(), insc=set(), regions=set(),
                                               dates=set(), wataks=collections.Counter(), roles=collections.Counter(),
                                               title_dated_only=True))
    for r in final:
        if r["code"] != "Y" or not r["key"]:
            continue
        v = vil[r["key"]]
        v["spellings"][r["name"]] += 1
        v["insc"].add(r["inscription"])
        v["regions"].add(r["region"])
        v["dates"].add(r["date"])
        if r["watak"]:
            v["wataks"][r["watak"].lower()] += 1
        v["roles"][r["role"]] += 1
        if r["date_src"] != "title":
            v["title_dated_only"] = False
    out = []
    for k, v in sorted(vil.items()):
        out.append(dict(key=k, name=v["spellings"].most_common(1)[0][0],
                        spellings="; ".join(s for s, _ in v["spellings"].most_common()),
                        n_mentions=sum(v["spellings"].values()), n_inscriptions=len(v["insc"]),
                        inscriptions="; ".join(sorted(v["insc"])), dates="; ".join(sorted(v["dates"])),
                        regions="; ".join(sorted(v["regions"])), primary=bool(v["regions"] & PRIMARY),
                        title_dated_only=v["title_dated_only"],
                        wataks="; ".join(w for w, _ in v["wataks"].most_common()),
                        roles="; ".join(f"{r}:{n}" for r, n in v["roles"].most_common())))
    with open(F("results", "t7_village_frame.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("villages (distinct keys):", len(out), "| primary-region:", sum(o["primary"] for o in out),
          "| title-dated-only primary:", sum(o["primary"] and o["title_dated_only"] for o in out))


if __name__ == "__main__":
    main()
