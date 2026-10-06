"""
E231 -- POST HOC checks (2026-10-06, after the second adversarial read of the same day).

Not part of the design: every count here was asked for by a reader who had already seen the results of 01_tabulations.py
and who had produced rough versions of several of them. The orchestrator wrote this script without using the reader's
scratch scripts; it shares only the loader, the normaliser, the feature code and the distance of p8common.py.
Nothing here is a test of a pre-stated hypothesis. Outputs: results/P*.csv, results/P_posthoc.json, results/run_log_posthoc.txt

P1  Makasar: retention and pairwise sharing among meanings that ARE coded (the raw shares count every uncoded meaning
    as "not retained" / "not shared").
P2  Coded forms whose every cognate set is confined to lists of the SAME language (same Glottocode) -- e.g. the second
    Muna list of ABVD -- and breadth counted in languages instead of lists.
P3  The onset-string and nasal inputs with length held fixed (Mantel-Haenszel, strata = list x number of letters).
P4  Hyphenated forms (the only compiler-marked segmentation in the data): candidate share inside Bugis and Sa'dan Toraja.
P5  Where Tolaki stands among the 42 Bungku-Tolaki comparison lists; sensitivity of the PMP look-alike screen on forms
    that ABVD itself puts in the PMP entry's set.
P6  What the "Sulawesi box" of table A contains.
P7  Sa'dan Toraja: final glottal mark against final k; S5 split: size of the difference against its permutation SD;
    the candidate forms of the meanings that pull the S5 statistic down most.
"""
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
sys.path.insert(0, str(HERE.parent / "E228_p8_revision_analyses"))
import p8common as P  # noqa: E402

_log = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    _log.write(s + "\n")
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode())


OUT = {}

# ------------------------------------------------------------------ data
f = P.load_lists(P.TARGET)
fa = P.all_forms()
lt = pd.read_csv(P.ABVD / "languages.csv", dtype=str, keep_default_na=False)
GLOT = dict(zip(lt.ID, lt.Glottocode))
LNAME = dict(zip(lt.ID, lt.Name))
cls = pd.read_csv(RES / "A_list_classification.csv", dtype=str, keep_default_na=False)
SUL = set(cls.ID[cls.sulawesi.str.lower() == "true"])
assert len(f) == 1357 and int(f.candidate.sum()) == 438, (len(f), f.candidate.sum())
say("forms", len(f), "candidates", int(f.candidate.sum()))


def sets_of(pid, cog):
    """(meaning, set number) for every numeric token of the Cognacy field; a trailing '?' is dropped."""
    out = []
    for tok in cog.split(","):
        t = tok.strip().rstrip("?").strip()
        if t.isdigit():
            out.append((pid, t))
    return out


members = defaultdict(set)  # set -> ABVD list ids with a form in it
for lid, pid, cg in zip(fa.Language_ID, fa.Parameter_ID, fa.Cognacy):
    if cg.strip():
        for k in sets_of(pid, cg):
            members[k].add(lid)


def mh(v, c, strata):
    """Mantel-Haenszel odds ratio of candidate status for v = 1 against v = 0, Robins-Breslow-Greenland interval."""
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
        return float("nan"), float("nan"), float("nan")
    orr = R / S
    var = PR / (2 * R * R) + PSQR / (2 * R * S) + QS / (2 * S * S)
    se = math.sqrt(var)
    return orr, orr * math.exp(-1.959964 * se), orr * math.exp(1.959964 * se)


# ================================================================== P1
say("\n== P1 Makasar among coded meanings")
t = pd.read_csv(HERE.parent / "E228_p8_revision_analyses" / "results" / "TABLE_R1-2_retention_by_language.csv")
rows = []
for _, r in t.iterrows():
    coded = int(r.retained_from_PMP_n) + int(r.coded_not_PMP_etymon_n)
    n = int(r.meanings_compared_with_PMP)
    rows.append({"list": r.language, "meanings": n, "in_PMP_set": int(r.retained_from_PMP_n), "coded_meanings": coded,
                 "share_of_all": round(int(r.retained_from_PMP_n) / n, 4),
                 "share_of_coded": round(int(r.retained_from_PMP_n) / coded, 4),
                 "upper_bound_if_all_uncoded_were_in_set": round((int(r.retained_from_PMP_n) + int(r.no_cognate_set_n)) / n, 4)})
P1a = pd.DataFrame(rows)
say(P1a.to_string(index=False))
b = pd.read_csv(RES / "B_pairwise_cognate_sharing.csv")
b["share_among_both_coded"] = (b.n_share_set / b.n_both_have_coded_form).round(4)
P1b = b[["list_1", "list_2", "meanings_in_both", "n_share_set", "share_set", "n_both_have_coded_form", "share_among_both_coded"]]
say(P1b.to_string(index=False))
P1a.to_csv(RES / "P1_retention_among_coded.csv", index=False)
P1b.to_csv(RES / "P1_pairwise_among_both_coded.csv", index=False)

# ================================================================== P2
say("\n== P2 sets confined to lists of the same language")
co = f[f.coded == 1].copy()
rows, detail = [], []
for lid, name in P.TARGET.items():
    sub = co[co.Language_ID == lid]
    own = GLOT[lid]
    n_same = n_single = n_sul = n_sul_same = 0
    glot_breadth, list_breadth = [], []
    for fid, pid, cg, raw, concept in zip(sub.ID, sub.Parameter_ID, sub.Cognacy, sub.raw, sub.concept):
        ks = sets_of(pid, cg)
        if not ks:
            continue
        lists_all = [members[k] for k in ks]
        same = all(all(GLOT.get(x, "") == own and own != "" for x in m) for m in lists_all)
        single = all(m == {lid} for m in lists_all)
        widest = max(lists_all, key=len)
        sul = all(x in SUL for x in widest)
        gl = {GLOT.get(x, "") or ("list:" + x) for x in widest}
        glot_breadth.append(len(gl)); list_breadth.append(len(widest))
        n_same += same; n_single += single; n_sul += sul; n_sul_same += (sul and same)
        if same:
            detail.append({"list": name, "abvd_form_id": fid, "concept": concept, "form": raw, "cognacy": cg,
                           "other_lists_in_its_sets": "; ".join(sorted({LNAME.get(x, x) for m in lists_all for x in m if x != lid}))})
    tot = int((f.Language_ID == lid).sum()); cand = int(f.candidate[f.Language_ID == lid].sum())
    rows.append({"list": name, "forms": tot, "coded_forms": len(sub), "widest_set_sulawesi_only": n_sul,
                 "every_set_same_language_only": n_same, "of_which_no_other_list_at_all": n_single,
                 "sulawesi_only_and_same_language_only": n_sul_same,
                 "uncoded_share_as_published": round(cand / tot, 4),
                 "uncoded_share_if_same_language_sets_counted_uncoded": round((cand + n_same) / tot, 4),
                 "median_breadth_lists": float(np.median(list_breadth)), "median_breadth_languages": float(np.median(glot_breadth))})
P2 = pd.DataFrame(rows)
say(P2.to_string(index=False))
P2.to_csv(RES / "P2_same_language_sets.csv", index=False)
pd.DataFrame(detail).to_csv(RES / "P2_same_language_forms.csv", index=False)
say("lists sharing a Glottocode with a target list:",
    {n: sorted(LNAME[x] for x in GLOT if GLOT[x] == GLOT[l] and x != l) for l, n in P.TARGET.items()})

# ================================================================== P3
say("\n== P3 onset string and nasal input with length held fixed")
X = P.featurize(f.form, f.concept)
cand = f.candidate.values
lst = f.language.values


def letters(s):
    return sum(1 for ch in s if ch.isalpha() and ch not in "ʔ")


nlet = f.form.map(letters).values
def nasal_strict(s):
    # the strict definition of table E (01_tabulations.py), repeated here so that the anchor below must reproduce 2.05
    t = s.lower().replace("ng", "ŋ").replace("ny", "ɲ")
    return int(re.search("[mnŋɲ][pbtdcjkgq]", t) is not None)


ns = f.form.map(nasal_strict).values
pre = X.has_prefix_like.values
strat_list = lst
strat_len = np.array([f"{a}|{b}" for a, b in zip(lst, nlet)])
rows = []


def row(name, v, strata, mask=None, note=""):
    m = np.ones(len(f), bool) if mask is None else mask
    o, lo, hi = mh(np.asarray(v)[m], cand[m], np.asarray(strata)[m])
    rows.append({"contrast": name, "n_forms": int(m.sum()), "n_with": int(np.asarray(v)[m].sum()),
                 "odds_ratio": round(o, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "strata": note})


row("onset string (anchor: E231 E original 1.57)", pre, strat_list, note="list")
row("onset string and >= 6 letters (anchor: E231 E strict 2.31)", ((pre == 1) & (nlet >= 6)).astype(int), strat_list, note="list")
row("nasal strict (anchor: E231 E strict 2.05)", ns, strat_list, note="list")
row(">= 6 letters, alone", (nlet >= 6).astype(int), strat_list, note="list")
row("onset string, among forms of >= 6 letters", pre, strat_list, mask=nlet >= 6, note="list")
row("onset string, among forms of <= 5 letters", pre, strat_list, mask=nlet <= 5, note="list")
row("onset string", pre, strat_len, note="list x number of letters")
row("nasal strict", ns, strat_len, note="list x number of letters")
row("nasal input as coded", X.has_nasal_cluster.values, strat_len, note="list x number of letters")
row(">= 1 consonant-letter cluster", (X.n_consonant_clusters.values >= 1).astype(int), strat_len, note="list x number of letters")
row("written glottal mark", X.has_glottal.values, strat_len, note="list x number of letters")
row("action meaning", X.sem_ACTION.values, strat_len, note="list x number of letters")
P3 = pd.DataFrame(rows)
say(P3.to_string(index=False))
P3.to_csv(RES / "P3_length_conditioned.csv", index=False)

# ================================================================== P4
say("\n== P4 hyphenated forms")
hy = f.form.str.contains("-").values
rows = []
for name in ["Bugis", "Toraja-Sadan", "Muna", "Makassar", "Wolio", "Tolaki"]:
    m = lst == name
    a, n1 = int(cand[m & hy].sum()), int((m & hy).sum())
    c0, n0 = int(cand[m & ~hy].sum()), int((m & ~hy).sum())
    rows.append({"list": name, "hyphenated_forms": n1, "of_which_candidates": a,
                 "share": round(a / n1, 4) if n1 else None, "other_forms": n0, "of_which_candidates_other": c0,
                 "share_other": round(c0 / n0, 4)})
P4 = pd.DataFrame(rows)
say(P4.to_string(index=False))
o, lo, hi = mh(hy.astype(int), cand, strat_list)
say("hyphen alone, within lists: OR %.3f [%.3f, %.3f]; forms with a hyphen: %d" % (o, lo, hi, int(hy.sum())))
P4.to_csv(RES / "P4_hyphen_by_list.csv", index=False)
OUT["P4_hyphen_within_list_or"] = [round(o, 3), round(lo, 3), round(hi, 3)]

# ================================================================== P5
say("\n== P5 Tolaki among the comparison lists; sensitivity of the PMP screen")
bt = pd.read_csv(RES / "D_bt_lists.csv")
say("columns of D_bt_lists.csv:", list(bt.columns))
share_col = [c for c in bt.columns if "coded" in c.lower() and "share" in c.lower()][0]
grp_col = [c for c in bt.columns if c.lower() in ("group", "set", "role", "kind", "block")]
tol_share = float((f.coded[f.Language_ID == "674"]).mean())
comp = bt
if grp_col:
    say("groups:", dict(Counter(bt[grp_col[0]])))
    comp = bt[bt[grp_col[0]].astype(str).str.contains("comparison|42|other", case=False)]
say("rows used as comparison lists:", len(comp), "| Tolaki coded share: %.4f" % tol_share)
n_le = int((comp[share_col] <= tol_share).sum())
say("comparison lists at or below Tolaki:", n_le, "| 10th percentile %.4f | median %.4f" %
    (comp[share_col].quantile(0.10), comp[share_col].median()))
OUT["P5_comparison_lists"] = int(len(comp)); OUT["P5_lists_at_or_below_tolaki"] = n_le
OUT["P5_tolaki_coded_share"] = round(tol_share, 4)

pmp = fa[fa.Language_ID == "269"]
pmp_forms, pmp_sets = defaultdict(list), defaultdict(set)
for pid, val, frm, cg in zip(pmp.Parameter_ID, pmp.Value, pmp.Form, pmp.Cognacy):
    raw = val if val else frm
    n_ = P.norm(P.clean_form(raw))
    if len(n_) >= 2:
        pmp_forms[pid].append(n_)
    for k in sets_of(pid, cg):
        pmp_sets[pid].add(k)
tk = f[(f.Language_ID == "674") & (f.coded == 1)]
in_set = in_hit = out_set = out_hit = 0
for pid, cg, form in zip(tk.Parameter_ID, tk.Cognacy, tk.form):
    n_ = P.norm(form)
    if len(n_) < 2 or not pmp_forms.get(pid):
        continue
    hit = min(P.d_stem(n_, q) for q in pmp_forms[pid]) <= 0.34
    if set(sets_of(pid, cg)) & pmp_sets[pid]:
        in_set += 1; in_hit += hit
    else:
        out_set += 1; out_hit += hit
say("Tolaki coded forms in the PMP entry's set: %d, of which look-alikes of the PMP form: %d (%.1f %%)" % (in_set, in_hit, 100 * in_hit / in_set))
say("Tolaki coded forms in another set: %d, of which look-alikes of the PMP form: %d" % (out_set, out_hit))
OUT["P5_sensitivity_in_set"] = [int(in_hit), int(in_set)]; OUT["P5_other_set"] = [int(out_hit), int(out_set)]
d = pd.read_csv(RES / "D_pmp_lookalikes_by_list.csv")
say(d.to_string(index=False))

# ================================================================== P6
say("\n== P6 the Sulawesi box of table A")
sul = cls[cls.sulawesi.str.lower() == "true"].copy()
bt_ids = set(bt[[c for c in bt.columns if c.lower() in ("id", "list_id", "abvd_language_id", "language_id")][0]].astype(str)) if any(
    c.lower() in ("id", "list_id", "abvd_language_id", "language_id") for c in bt.columns) else set()
sul["kind"] = "other"
sul.loc[sul.ID.isin(bt_ids) | sul.ID.isin(["674", "780"]), "kind"] = "Bungku-Tolaki (comparison, dialect or proto list)"
sul.loc[sul.Name.str.contains("Bajo|Bajau|Sama", case=False), "kind"] = "Bajo / Sama-Bajaw"
sul.loc[sul.ID.isin(["48", "166", "226"]), "kind"] = "South Sulawesi target list"
say("lists counted as Sulawesi:", len(sul), dict(Counter(sul.kind)))
say("others:", "; ".join(sorted(sul.Name[sul.kind == "other"])))
no_coord = cls[(cls.has_coords.str.lower() != "true")]
say("lists without coordinates:", len(no_coord), "| of them proto:", int((no_coord.is_proto.str.lower() == "true").sum()))
sul[["ID", "Name", "Glottocode", "is_proto", "kind"]].to_csv(RES / "P6_sulawesi_box_lists.csv", index=False)
OUT["P6_sulawesi_lists"] = int(len(sul)); OUT["P6_kinds"] = dict(Counter(sul.kind))
OUT["P6_distinct_glottocodes_in_box"] = int(sul.Glottocode[sul.Glottocode != ""].nunique())

# ================================================================== P7
say("\n== P7 Sa'dan Toraja final segments; S5 split; influential meanings")
tor = f[f.language == "Toraja-Sadan"]
fin_glot = int(tor.form.str.contains(r"[ʔ']$").sum()); fin_k = int(tor.form.str.contains(r"k$").sum())
say("Sa'dan Toraja forms ending in a glottal mark: %d; ending in k: %d" % (fin_glot, fin_k))
OUT["P7_toraja_final_glottal_mark"] = fin_glot; OUT["P7_toraja_final_k"] = fin_k
s5 = json.load(open(HERE.parent / "E228_p8_revision_analyses" / "results" / "S5_permutation.json", encoding="utf-8"))
a, o_ = s5["b_pairs_inside_south_sulawesi"], s5["b_pairs_other"]
da, do = a["T_permutation_mean"] - a["T_observed"], o_["T_permutation_mean"] - o_["T_observed"]
se = math.sqrt(a["T_permutation_sd"] ** 2 + o_["T_permutation_sd"] ** 2)
say("excess inside South Sulawesi %.4f (SD %.4f); elsewhere %.4f (SD %.4f); difference %.4f = %.2f combined SD" %
    (da, a["T_permutation_sd"], do, o_["T_permutation_sd"], da - do, (da - do) / se))
OUT["P7_split"] = {"inside": round(da, 4), "elsewhere": round(do, 4), "difference": round(da - do, 4), "in_combined_sd": round((da - do) / se, 2)}
h = pd.read_csv(RES / "H_leave_one_concept_out.csv")
say("columns of H:", list(h.columns))
ccol = [c for c in h.columns if "concept" in c.lower() or "meaning" in c.lower()][0]
dcol = [c for c in h.columns if "change" in c.lower()][0]
top = h.sort_values(dcol, ascending=False).head(8)[ccol].tolist()
rows = []
for c in top:
    fm = f[(f.concept == c) & (f.candidate == 1)]
    forms = [f"{l}: {r}" for l, r in zip(fm.language, fm.raw)]
    n_m = int(fm.form.str.lower().str.match(r"^m[aeo]").sum())
    rows.append({"meaning": c, "candidate_forms": len(fm), "begin_with_ma_me_mo": n_m, "forms": "; ".join(forms)})
P7 = pd.DataFrame(rows)
say(P7.to_string(index=False))
P7.to_csv(RES / "P7_influential_meanings.csv", index=False)

# ================================================================== P8-P12 (asked for by the third read, same day)
# P8  the PMP look-alike screen: sensitivity on forms ABVD itself puts in the PMP entry's set, for all six lists
# P9  Makasar against Bugis and Sa'dan Toraja, per meaning: paired figures on meanings coded in both lists; what the
#     two relatives have for the meanings Makasar leaves uncoded; look-alikes of Makasar forms among the relatives' forms
# P10 coded share BY MEANING on the meanings each Bungku-Tolaki list shares with Tolaki 674 (table D / P5 count forms)
# P11 consonant-letter cluster with the glottal mark also held fixed; onset string with the REMAINING letters held fixed
# P12 the classifier with the 44 same-language Muna forms relabelled as candidates
say("\n== P8 sensitivity of the PMP look-alike screen, all six lists")


def sure_sets(pid, cog):
    out = set()
    for tok in cog.split(","):
        t = tok.strip()
        if t.isdigit():
            out.add((pid, t))
    return out


pmp_sure = defaultdict(set)
for pid, cg in zip(pmp.Parameter_ID, pmp.Cognacy):
    pmp_sure[pid] |= sure_sets(pid, cg)
rows = []
for lid, name in P.TARGET.items():
    sub_ = f[(f.Language_ID == lid) & (f.coded == 1)]
    a = b_ = 0
    for pid, cg, form in zip(sub_.Parameter_ID, sub_.Cognacy, sub_.form):
        n_ = P.norm(form)
        if len(n_) < 2 or not pmp_forms.get(pid) or not (set(sets_of(pid, cg)) & pmp_sets[pid]):
            continue
        b_ += 1
        a += min(P.d_stem(n_, q) for q in pmp_forms[pid]) <= 0.34
    rows.append({"list": name, "coded_forms_in_PMP_set": b_, "found_by_screen": int(a), "sensitivity": round(a / b_, 3)})
P8 = pd.DataFrame(rows)
say(P8.to_string(index=False))
P8.to_csv(RES / "P8_screen_sensitivity.csv", index=False)

say("\n== P9 Makasar and its two relatives, per meaning")
cls_m = {}
for lid, name in P.TARGET.items():
    g = f[f.Language_ID == lid]
    d_ = {}
    for pid, gg in g.groupby("Parameter_ID"):
        if not pmp_sure.get(pid):
            continue
        fs = set()
        for cg in gg.Cognacy:
            fs |= sure_sets(pid, cg)
        d_[pid] = "in_PMP_set" if fs & pmp_sure[pid] else ("coded_otherwise" if gg.coded.sum() > 0 else "uncoded")
    cls_m[name] = d_
cnt = {n: (sum(v == "in_PMP_set" for v in d_.values()), sum(v == "coded_otherwise" for v in d_.values()),
           sum(v == "uncoded" for v in d_.values()), len(d_)) for n, d_ in cls_m.items()}
say("anchor (E228 S2): Makassar", cnt["Makassar"], "Bugis", cnt["Bugis"], "Toraja-Sadan", cnt["Toraja-Sadan"])
assert cnt["Makassar"] == (79, 52, 70, 201) and cnt["Bugis"] == (97, 57, 47, 201) and cnt["Toraja-Sadan"] == (96, 66, 38, 200)


def mcnemar_exact(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


rows = []
for other in ["Bugis", "Toraja-Sadan"]:
    A, B_ = cls_m["Makassar"], cls_m[other]
    both = [m for m in A if m in B_ and A[m] != "uncoded" and B_[m] != "uncoded"]
    a_in = sum(A[m] == "in_PMP_set" for m in both); b_in = sum(B_[m] == "in_PMP_set" for m in both)
    b01 = sum(A[m] != "in_PMP_set" and B_[m] == "in_PMP_set" for m in both)
    b10 = sum(A[m] == "in_PMP_set" and B_[m] != "in_PMP_set" for m in both)
    unc = [m for m in A if A[m] == "uncoded" and m in B_]
    rows.append({"relative": other, "meanings_coded_in_both": len(both), "makasar_in_PMP_set": a_in, "relative_in_PMP_set": b_in,
                 "mcnemar_p": round(mcnemar_exact(b01, b10), 4),
                 "makasar_uncoded_meanings_also_in_relative_list": len(unc),
                 "of_which_relative_in_PMP_set": sum(B_[m] == "in_PMP_set" for m in unc),
                 "of_which_relative_coded_otherwise": sum(B_[m] == "coded_otherwise" for m in unc),
                 "of_which_relative_uncoded": sum(B_[m] == "uncoded" for m in unc)})
P9a = pd.DataFrame(rows)
say(P9a.to_string(index=False))
P9a.to_csv(RES / "P9_makasar_vs_relatives_by_meaning.csv", index=False)

# the meanings a specialist is asked to look at: Makasar uncoded, with what the two relatives have
rows = []
raw_by = defaultdict(list)
for l, pid, raw_, cg in zip(f.language, f.Parameter_ID, f.raw, f.Cognacy):
    raw_by[(l, pid)].append(raw_ + (f" [{cg}]" if cg.strip() else ""))
for m, c_ in sorted(cls_m["Makassar"].items()):
    if c_ != "uncoded":
        continue
    rows.append({"meaning": P.PNAME.get(m, m), "makasar_forms": "; ".join(raw_by[("Makassar", m)]),
                 "pmp_forms": "; ".join(sorted({(v if v else fr) for pid_, v, fr in zip(pmp.Parameter_ID, pmp.Value, pmp.Form) if pid_ == m})),
                 "bugis_class": cls_m["Bugis"].get(m, "not in list"), "bugis_forms": "; ".join(raw_by[("Bugis", m)]),
                 "toraja_class": cls_m["Toraja-Sadan"].get(m, "not in list"), "toraja_forms": "; ".join(raw_by[("Toraja-Sadan", m)])})
pd.DataFrame(rows).to_csv(RES / "P9_makasar_uncoded_meanings_for_specialist.csv", index=False)
say("written: P9_makasar_uncoded_meanings_for_specialist.csv (%d meanings; set numbers in brackets)" % len(rows))

rng = np.random.default_rng(231)
SS = ["Bugis", "Makassar", "Toraja-Sadan"]
normf = {i: P.norm(s) for i, s in zip(f.ID, f.form)}
by_lm = defaultdict(list)
for i, l, pid in zip(f.ID, f.language, f.Parameter_ID):
    if len(normf[i]) >= 2:
        by_lm[(l, pid)].append(normf[i])
pids_all = sorted(set(f.Parameter_ID))
rows = []
for L in SS:
    others = [x for x in SS if x != L]
    for label, flag in (("candidate", 1), ("coded", 0)):
        sub_ = f[(f.language == L) & (f.candidate == flag)]
        n = hit = 0
        chance = []
        for i, pid in zip(sub_.ID, sub_.Parameter_ID):
            a_ = normf[i]
            if len(a_) < 2:
                continue
            comp = [q for o in others for q in by_lm.get((o, pid), [])]
            if not comp:
                continue
            n += 1
            hit += min(P.d_stem(a_, q) for q in comp) <= 0.34
            draws = rng.choice([p_ for p_ in pids_all if p_ != pid], size=20, replace=False)
            ch = []
            for p_ in draws:
                c2 = [q for o in others for q in by_lm.get((o, p_), [])]
                if c2:
                    ch.append(min(P.d_stem(a_, q) for q in c2) <= 0.34)
            if ch:
                chance.append(np.mean(ch))
        rows.append({"list": L, "class": label, "forms_compared": n, "lookalike_in_other_two_SS_lists": int(hit),
                     "share": round(hit / n, 4), "chance_share": round(float(np.mean(chance)), 4)})
P9b = pd.DataFrame(rows)
say(P9b.to_string(index=False))
P9b.to_csv(RES / "P9_lookalikes_within_south_sulawesi.csv", index=False)

say("\n== P10 coded share by meaning, on the meanings shared with Tolaki 674")
tol = f[f.Language_ID == "674"]
tol_coded_by_m = tol.groupby("Parameter_ID").coded.max().to_dict()
rows = []
for _, r in bt.iterrows():
    lid = str(r["id"])
    g = fa[(fa.Language_ID == lid)]
    g = g[(g.Value != "") | (g.Form != "")]
    by_m = g.assign(c=(g.Cognacy.str.strip() != "").astype(int)).groupby("Parameter_ID").c.max().to_dict()
    shared = [m for m in by_m if m in tol_coded_by_m]
    if not shared:
        continue
    rows.append({"group": r["group"], "id": lid, "name": r["name"], "n_forms": int(r["n_forms"]),
                 "coded_share_by_form": round(float(r["coded_share"]), 4), "shared_meanings": len(shared),
                 "list_coded_share_by_meaning": round(float(np.mean([by_m[m] for m in shared])), 4),
                 "tolaki_coded_share_same_meanings": round(float(np.mean([tol_coded_by_m[m] for m in shared])), 4)})
P10 = pd.DataFrame(rows)
P10["list_at_or_below_tolaki"] = (P10.list_coded_share_by_meaning <= P10.tolaki_coded_share_same_meanings).astype(int)
P10.to_csv(RES / "P10_bt_coded_share_by_meaning.csv", index=False)
c42 = P10[P10.group == "comparison_list"]
say("comparison lists:", len(c42), "| at or below Tolaki on the same meanings:", int(c42.list_at_or_below_tolaki.sum()),
    "| median list %.4f | median Tolaki on the same meanings %.4f" % (c42.list_coded_share_by_meaning.median(), c42.tolaki_coded_share_same_meanings.median()))
say(P10[P10.group != "comparison_list"][["group", "name", "n_forms", "coded_share_by_form", "shared_meanings",
                                         "list_coded_share_by_meaning", "tolaki_coded_share_same_meanings"]].to_string(index=False))
OUT["P10"] = {"comparison_lists": int(len(c42)), "at_or_below_tolaki_by_meaning": int(c42.list_at_or_below_tolaki.sum()),
              "median_list_by_meaning": round(float(c42.list_coded_share_by_meaning.median()), 4),
              "median_tolaki_same_meanings": round(float(c42.tolaki_coded_share_same_meanings.median()), 4)}
# P10b (asked for by the last number trace): "by meaning" above means ANY form of the meaning is coded, so a list with
# many synonyms per meaning scores higher. Same comparison with the FIRST form of each meaning only.
tol_all = fa[(fa.Language_ID == "674") & ((fa.Value != "") | (fa.Form != ""))]
tol_first = tol_all.drop_duplicates("Parameter_ID", keep="first")
tol_first_c = dict(zip(tol_first.Parameter_ID, (tol_first.Cognacy.str.strip() != "").astype(int)))
rows = []
for _, r in bt.iterrows():
    lid = str(r["id"])
    g = fa[(fa.Language_ID == lid) & ((fa.Value != "") | (fa.Form != ""))]
    g1 = g.drop_duplicates("Parameter_ID", keep="first")
    c1 = dict(zip(g1.Parameter_ID, (g1.Cognacy.str.strip() != "").astype(int)))
    shared = [m for m in c1 if m in tol_first_c]
    if not shared:
        continue
    n_multi = int((g.groupby("Parameter_ID").size() > 1).sum())
    rows.append({"group": r["group"], "id": lid, "name": r["name"], "n_forms": len(g), "shared_meanings": len(shared),
                 "meanings_with_more_than_one_form": n_multi,
                 "list_coded_share_first_form": round(float(np.mean([c1[m] for m in shared])), 4),
                 "tolaki_coded_share_first_form_same_meanings": round(float(np.mean([tol_first_c[m] for m in shared])), 4)})
P10b = pd.DataFrame(rows)
P10b["list_at_or_below_tolaki"] = (P10b.list_coded_share_first_form <= P10b.tolaki_coded_share_first_form_same_meanings).astype(int)
P10b.to_csv(RES / "P10b_bt_coded_share_first_form.csv", index=False)
c42b = P10b[P10b.group == "comparison_list"]
say("first form per meaning — comparison lists:", len(c42b), "| at or below Tolaki:", int(c42b.list_at_or_below_tolaki.sum()),
    "| median list %.4f | median Tolaki on the same meanings %.4f" % (c42b.list_coded_share_first_form.median(),
                                                                     c42b.tolaki_coded_share_first_form_same_meanings.median()))
say(P10b[P10b.group != "comparison_list"][["group", "name", "n_forms", "shared_meanings", "meanings_with_more_than_one_form",
                                           "list_coded_share_first_form", "tolaki_coded_share_first_form_same_meanings"]].to_string(index=False))
OUT["P10b"] = {"at_or_below_tolaki_first_form": int(c42b.list_at_or_below_tolaki.sum()),
               "median_list_first_form": round(float(c42b.list_coded_share_first_form.median()), 4),
               "median_tolaki_first_form": round(float(c42b.tolaki_coded_share_first_form_same_meanings.median()), 4)}
sp = c42[["n_forms", "coded_share_by_form"]].rank().corr().iloc[0, 1]
say("Spearman, number of forms against coded share by form (42 lists): %.3f" % sp)
OUT["P10"]["spearman_forms_vs_share_by_form"] = round(float(sp), 3)

say("\n== P11 cluster with the glottal mark also held fixed; onset string with the remaining letters held fixed")
PREF = sorted({s.rstrip("-") for s in P.E027["AUSTRONESIAN_PREFIXES"]}, key=len, reverse=True)


def remaining_letters(s):
    low = s.lower()
    for pfx in PREF:
        if low.startswith(pfx):
            return letters(s) - letters(pfx)
    return letters(s)


rem = f.form.map(remaining_letters).values
rows = []
strat_len_glot = np.array([f"{a}|{b}|{c}" for a, b, c in zip(lst, nlet, X.has_glottal.values)])
strat_rem = np.array([f"{a}|{b}" for a, b in zip(lst, rem)])
row(">= 1 consonant-letter cluster", (X.n_consonant_clusters.values >= 1).astype(int), strat_len_glot, note="list x letters x glottal mark")
row("onset string", pre, strat_rem, note="list x letters after the matched string")
P11 = pd.DataFrame(rows)
say(P11.to_string(index=False))
P11.to_csv(RES / "P11_cluster_and_onset_other_conditionings.csv", index=False)

say("\n== P12 classifier with the 44 same-language Muna forms relabelled as candidates")
same_ids = set(pd.DataFrame(detail).query("list == 'Muna'").abvd_form_id)
y0 = f.coded.values.copy()
y1 = y0.copy(); y1[f.ID.isin(same_ids).values] = 0
F17 = P.PHON + P.INIT
rows = []
for lab, cols in (("form + meaning (25 inputs)", P.PURE25), ("form only (17 inputs)", F17)):
    a0, s0, _ = P.cv_auc(X[cols].values, y0)
    a1, s1, _ = P.cv_auc(X[cols].values, y1)
    rows.append({"inputs": lab, "cv_auc_label_as_published": round(a0, 4), "cv_auc_44_muna_forms_as_candidates": round(a1, 4),
                 "difference": round(a1 - a0, 4), "candidates_as_published": int((y0 == 0).sum()), "candidates_relabelled": int((y1 == 0).sum())})
P12 = pd.DataFrame(rows)
say(P12.to_string(index=False))
P12.to_csv(RES / "P12_label_sensitivity_muna.csv", index=False)
OUT["P8"] = P8.to_dict("records"); OUT["P9a"] = P9a.to_dict("records"); OUT["P9b"] = P9b.to_dict("records")
OUT["P11"] = P11.to_dict("records"); OUT["P12"] = P12.to_dict("records")

OUT["P1"] = P1a.to_dict("records"); OUT["P2"] = P2.to_dict("records"); OUT["P3"] = P3.to_dict("records")
(RES / "P_posthoc.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
(RES / "run_log_posthoc.txt").write_text(_log.getvalue(), encoding="utf-8")
say("\nwritten: P1_*.csv P2_*.csv P3_length_conditioned.csv P4_hyphen_by_list.csv P6_sulawesi_box_lists.csv "
    "P7_influential_meanings.csv P_posthoc.json run_log_posthoc.txt")
