"""E226 step 4: match identified villages to the frozen settlement register; d, d0, bounds, N95, decision.

Refuses to run until the PI has written N_ref and m into DESIGN.md §5 (pre-registration). Inputs:
  results/t7_village_frame.csv            frame V (primary = Kedu/Prambanan inscriptions)
  frame/identified_coords.csv             key, tier (I1/I2), lat, lon, kabupaten  (step 2c)
  register/t7_settlement_register_candidates.csv  frozen register (sha256 in README)
"""
import csv
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
from scipy import integrate, stats
from scipy.special import betaln

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REG = os.path.join(HERE, "register", "t7_settlement_register_candidates.csv")
REG_SHA = "10c7d0e0173521cd13f94a51145dd0f419482155b78e10dcc01070d676388133"
ADM2 = os.path.join(ROOT, "data", "raw", "gazetteer", "geoBoundaries-IDN-ADM2_simplified.geojson")
RHOS = (0.5, 1.0, 2.0)
N_NULL = 1000


def read_design():
    txt = open(os.path.join(HERE, "DESIGN.md"), encoding="utf-8").read()
    n = re.search(r"\*\*N_ref = ([^*]+)\*\*", txt)
    m = re.search(r"\*\*m ∈ \{([^}]*)\}", txt)
    if not n or "_" in n.group(1) or not m or "_" in m.group(1):
        sys.exit("BLOCKED: N_ref and m are not filled in DESIGN.md §5 (PI decision, pre-registered).")
    nref = n.group(1).strip()
    ms = [float(x) for x in m.group(1).split(",")]
    return nref, ms


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def n95(S, n, m):
    """Smallest N with P(no detection among N villages) <= 0.05, d_pre = m*p, p ~ Beta(S+.5, n-S+.5)."""
    a, b = S + 0.5, n - S + 0.5
    for N in range(1, 200000):
        if m == 1.0:
            p0 = math.exp(betaln(a, b + N) - betaln(a, b))
        else:
            p0 = integrate.quad(lambda p: (1 - m * p) ** N * stats.beta.pdf(p, a, b), 0, 1, limit=200)[0]
        if p0 <= 0.05:
            return N
    return float("inf")


def main():
    nref_txt, ms = read_design()
    assert hashlib.sha256(open(REG, "rb").read()).hexdigest() == REG_SHA, "register changed after freezing"
    frame = [r for r in csv.DictReader(open(os.path.join(HERE, "results", "t7_village_frame.csv"), encoding="utf-8"))
             if r["primary"] == "True"]
    V = len(frame)
    nref = V if nref_txt.lower().startswith("|v") else int(nref_txt)
    ident = {r["key"]: r for r in csv.DictReader(open(os.path.join(HERE, "frame", "identified_coords.csv"),
                                                        encoding="utf-8")) if r["lat"]}
    G = [dict(v, **ident[v["key"]]) for v in frame if v["key"] in ident]
    reg = list(csv.DictReader(open(REG, encoding="utf-8-sig")))
    import geopandas as gpd
    from shapely.geometry import Point
    adm = gpd.read_file(ADM2).set_index("shapeName")
    rng = np.random.default_rng(226)
    out = {"V": V, "G": len(G), "N_ref": nref, "m": ms, "runs": []}
    for label, keep in (("YES", {"YES"}), ("YES+UNCLEAR", {"YES", "UNCLEAR"})):
        sites = [(float(r["lat"]), float(r["lon"])) for r in reg if r["qualifies"] in keep and r["lat"] and r["lon"]]
        for rho in RHOS:
            hit = [any(km(float(g["lat"]), float(g["lon"]), a, b) <= rho for a, b in sites) for g in G]
            S, n = sum(hit), len(G)
            # Chance-match null: each identified village replaced by a random point in its kabupaten.
            null = []
            for _ in range(N_NULL):
                c = 0
                for g in G:
                    poly = adm.loc[g["kabupaten"].replace("Kabupaten ", "")].geometry
                    x0, y0, x1, y1 = poly.bounds
                    while True:
                        lon, lat = rng.uniform(x0, x1), rng.uniform(y0, y1)
                        if poly.contains(Point(lon, lat)):
                            break
                    c += any(km(lat, lon, a, b) <= rho for a, b in sites)
                null.append(c / n if n else 0)
            d = S / n if n else float("nan")
            lo, hi = stats.beta.ppf([0.025, 0.975], S + 0.5, n - S + 0.5) if n else (float("nan"),) * 2
            run = dict(register=label, rho_km=rho, S=S, n=n, d=d, d_ci95=[lo, hi], d0=float(np.mean(null)),
                       d_minus_d0=d - float(np.mean(null)),
                       bounds=[S / V, (S + V - n) / V],
                       N95={str(m): n95(S, n, m) for m in sorted(set(ms) | {1.0})})
            mmin = min(ms)
            run["decision"] = ("UNINFORMATIVE" if run["N95"][str(mmin)] >= nref else
                               "SUPPORTS_H6" if run["N95"]["1.0"] < nref else "AMBIGUOUS")
            run["falsifier_H3_control_dead"] = bool(n >= 30 and run["d_minus_d0"] >= 0.25)
            out["runs"].append(run)
            print(json.dumps(run, default=str))
    json.dump(out, open(os.path.join(HERE, "results", "t7_match.json"), "w", encoding="utf-8"), indent=1, default=str)


if __name__ == "__main__":
    main()
