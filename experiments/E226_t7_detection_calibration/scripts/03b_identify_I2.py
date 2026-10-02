"""E226 step 2b (I2 part): unique modern desa of the same name in the inscription's findspot kabupaten.

I1 (published epigrapher identifications) is compiled separately into frame/identifications_I1.csv and
overrides I2. Exact key match only: a looser match would trade identification failures for false
identifications, which inflate d. Dusun names are not in the gazetteer, so I2 is conservative.
"""
import collections
import csv
import os
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GAZ = os.path.join(ROOT, "data", "processed", "gazetteer", "desa_kedu_mataram_buffer_2025.csv")


def modern_key(s):
    s = (s or "").lower().replace("ṁ", "ng").replace("ṅ", "ng").replace("ñ", "ny").replace("r̥", "re").replace("l̥", "le")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("ə", "e").replace("w", "v")
    s = re.sub(r"[^a-z]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def kab_key(s):
    s = (s or "").lower().replace("kabupaten", "").replace("kota", "").replace("kab.", "")
    return re.sub(r"[^a-z]", "", s).replace("gunungkidul", "gunungkidul")


def main():
    gaz = collections.defaultdict(list)
    for r in csv.DictReader(open(GAZ, encoding="utf-8")):
        gaz[(kab_key(r["kabupaten"]), modern_key(r["desa"]))].append(r)
    insc = {r["inscription"]: r for r in csv.DictReader(open(os.path.join(HERE, "frame", "inscriptions_701_1000.csv"),
                                                               encoding="utf-8"))}
    frame = list(csv.DictReader(open(os.path.join(HERE, "results", "t7_village_frame.csv"), encoding="utf-8")))
    out = []
    for v in frame:
        mk = modern_key(v["name"])
        kabs = {kab_key(insc[i]["xlsx_kab"]) for i in v["inscriptions"].split("; ") if insc[i]["xlsx_kab"]}
        hits = [h for k in kabs for h in gaz.get((k, mk), [])]
        tier = "I2" if len(hits) == 1 else "I3"
        why = ("unique desa match" if len(hits) == 1 else
               f"{len(hits)} desa of that name" if hits else
               "findspot kabupaten unknown" if not kabs else "no desa of that name")
        h = hits[0] if len(hits) == 1 else {}
        out.append(dict(key=v["key"], name=v["name"], modern_key=mk, findspot_kab="; ".join(sorted(kabs)),
                        tier_I2=tier, reason=why, desa=h.get("desa", ""), kecamatan=h.get("kecamatan", ""),
                        kabupaten=h.get("kabupaten", ""), kode=h.get("kode", ""), primary=v["primary"]))
    p = os.path.join(HERE, "results", "t7_identification_I2.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    c = collections.Counter((o["primary"], o["tier_I2"]) for o in out)
    print("I2 by (primary, tier):", dict(c))
    print("reasons:", collections.Counter(o["reason"] for o in out).most_common())


if __name__ == "__main__":
    main()
