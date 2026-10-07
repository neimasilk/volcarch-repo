"""E231 P13 (post hoc, 2026-10-07): the written glottal mark per list, and the pooled odds ratio without Tolaki.

Why: the pooled Mantel-Haenszel odds ratio of 2.67 (E229 T6) includes the Tolaki stratum, in which all 29 marked forms
are candidates and no coded form carries a mark (an infinite stratum). The PI may print per-list figures, so the
numbers need a home in the experiment rather than in a scratch script. Definitions are those of table E / P3:
a glottal mark = U+0294 or an apostrophe in the ABVD value; letters = alphabetic characters other than U+0294.
Input: the release file of E228 (one row per form). Output: results/P13_glottal_by_list.csv, results/P13_summary.json.
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REL = HERE.parent / "E228_p8_revision_analyses" / "release" / "p8_forms_all.csv"
RES = HERE / "results"
RES.mkdir(exist_ok=True)


def letters(s):
    return sum(1 for ch in s if ch.isalpha() and ch not in "ʔ")


def mh(v, c, strata):
    """Mantel-Haenszel odds ratio with Robins-Breslow-Greenland interval (same function as 02_posthoc_checks.py)."""
    v, c, strata = np.asarray(v), np.asarray(c), np.asarray(strata)
    R = S = PR = PSQR = QS = 0.0
    for s in np.unique(strata):
        m = strata == s
        a = float(((v == 1) & (c == 1) & m).sum()); b = float(((v == 1) & (c == 0) & m).sum())
        cc = float(((v == 0) & (c == 1) & m).sum()); d = float(((v == 0) & (c == 0) & m).sum())
        n = a + b + cc + d
        if n == 0:
            continue
        r, s_ = a * d / n, b * cc / n
        R += r; S += s_
        PR += (a + d) / n * r; PSQR += (a + d) / n * s_ + (b + cc) / n * r; QS += (b + cc) / n * s_
    if R == 0 or S == 0:
        return (float("nan"),) * 3
    orr = R / S
    var = PR / (2 * R * R) + PSQR / (2 * R * S) + QS / (2 * S * S)
    se = math.sqrt(var)
    return orr, orr * math.exp(-1.959964 * se), orr * math.exp(1.959964 * se)


rows = list(csv.DictReader(open(REL, encoding="utf-8")))
name = {"Muna (Katobu-Tongkuno Dialect)": "Muna", "Buginese (Soppeng Dialect)": "Bugis", "Makassar": "Makasar",
        "Wolio": "Wolio", "Tae' (S.Toraja)": "Sa'dan Toraja", "Tolaki": "Tolaki"}
L = np.array([name[r["abvd_language_name"]] for r in rows])
g = np.array([1 if ("ʔ" in r["form_as_in_abvd"] or "'" in r["form_as_in_abvd"]) else 0 for r in rows])
c = np.array([int(r["candidate_no_cognate_set"]) for r in rows])
nl = np.array([letters(r["form_as_in_abvd"]) for r in rows])
strat_len = np.array([f"{a}|{b}" for a, b in zip(L, nl)])

out = []
for lst in ["Muna", "Bugis", "Makasar", "Wolio", "Sa'dan Toraja", "Tolaki"]:
    m = L == lst
    a = int(((g == 1) & (c == 1) & m).sum()); b = int(((g == 0) & (c == 1) & m).sum())
    cc = int(((g == 1) & (c == 0) & m).sum()); d = int(((g == 0) & (c == 0) & m).sum())
    orr = (a * d) / (b * cc) if b * cc else (float("inf") if a * d else float("nan"))
    out.append({"list": lst, "candidates": a + b, "candidates_with_mark": a, "coded": cc + d, "coded_with_mark": cc,
                "share_candidates_pct": round(100 * a / (a + b), 1), "share_coded_pct": round(100 * cc / (cc + d), 1),
                "odds_ratio": ("inf" if orr == float("inf") else ("" if orr != orr else round(orr, 2)))})
with open(RES / "P13_glottal_by_list.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

noT = L != "Tolaki"
ss = np.isin(L, ["Bugis", "Makasar", "Sa'dan Toraja"])
summary = {
    "anchor_E229_T6_mh_by_list_all": [round(x, 3) for x in mh(g, c, L)],
    "anchor_E231_P3_mh_by_list_x_letters_all": [round(x, 3) for x in mh(g, c, strat_len)],
    "mh_by_list_without_tolaki": [round(x, 3) for x in mh(g[noT], c[noT], L[noT])],
    "mh_by_list_x_letters_without_tolaki": [round(x, 3) for x in mh(g[noT], c[noT], strat_len[noT])],
    "mh_by_list_x_letters_three_south_sulawesi_lists": [round(x, 3) for x in mh(g[ss], c[ss], strat_len[ss])],
}
(RES / "P13_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
for r in out:
    print(r)
print(json.dumps(summary, indent=1))
a1, a2 = summary["anchor_E229_T6_mh_by_list_all"][0], summary["anchor_E231_P3_mh_by_list_x_letters_all"][0]
assert abs(a1 - 2.669) < 0.01 and abs(a2 - 2.60) < 0.01, "anchors to E229 T6 / E231 P3 do not reproduce"
print("anchors OK")
