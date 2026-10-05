"""
E228 — analyses for the P8 revision (pre-registered in DESIGN.md, frozen 2026-10-05).

Sections S1–S6 as in DESIGN.md. Nothing here changes E022–E042.
Run:  python experiments/E228_p8_revision_analyses/01_revision_analyses.py
"""
import io
import json
import random
import re
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

import p8common as P

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT, REL = HERE / "results", HERE / "release"
OUT.mkdir(exist_ok=True); REL.mkdir(exist_ok=True)
SEED = 228


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


f = P.load_lists(P.TARGET)
f["in_table1_set"] = P.table1_set(f)
y = f.coded.values
langs = f.language.values
F0 = P.featurize(f.form, f.concept, f.language)
assert len(f) == 1357 and int(f.candidate.sum()) == 438 and int(f.in_table1_set.sum()) == 356

# =============================================================================
hdr("S1  glottal-stop orthography (reviewer 1, point 9)")
# =============================================================================
MARK = "ʔ'"


def geminate(s):
    out = []
    for i, ch in enumerate(s):
        if ch in MARK:
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if nxt and nxt.isalpha() and nxt.lower() not in P.VOWELS:
                out.append(nxt)
        else:
            out.append(ch)
    return "".join(out)


CONV = {
    "V0_as_published": lambda s: s,
    "V1_q": lambda s: re.sub("[ʔ']", "q", s),
    "V2_k": lambda s: re.sub("[ʔ']", "k", s),
    "V3_unwritten": lambda s: re.sub("[ʔ']", "", s),
    "V4_geminate": geminate,
    "V5_all_as_glottal_letter": lambda s: s.replace("'", "ʔ"),
}
s1 = {}
for name, fn in CONV.items():
    forms_v = f.form.map(fn)
    forms_v = forms_v.where(forms_v != "", f.form)
    Fv = P.featurize(forms_v, f.concept, f.language)
    row = {"forms_changed": int((forms_v.values != f.form.values).sum()),
           "share_with_glottal_feature": float(Fv.has_glottal.mean())}
    for tag, cols in (("25", P.PURE25), ("26", P.ABL26)):
        a, sd, _ = P.cv_auc(Fv[cols].values, y)
        lo, _ = P.lolo_auc(Fv[cols].values, y, langs)
        row[f"cv_auc_{tag}"], row[f"cv_sd_{tag}"] = round(a, 4), round(sd, 4)
        row[f"lolo_mean_{tag}"] = round(float(np.mean(list(lo.values()))), 4)
        row[f"lolo_{tag}"] = {k: round(v, 3) for k, v in lo.items()}
    s1[name] = row
    print(name, {k: v for k, v in row.items() if not k.startswith("lolo_2")}, flush=True)
row = {"forms_changed": 0, "share_with_glottal_feature": None}
for tag, cols in (("25", P.PURE25), ("26", P.ABL26)):
    c2 = [c for c in cols if c != "has_glottal"]
    a, sd, _ = P.cv_auc(F0[c2].values, y)
    lo, _ = P.lolo_auc(F0[c2].values, y, langs)
    row[f"cv_auc_{tag}"], row[f"cv_sd_{tag}"] = round(a, 4), round(sd, 4)
    row[f"lolo_mean_{tag}"] = round(float(np.mean(list(lo.values()))), 4)
    row[f"lolo_{tag}"] = {k: round(v, 3) for k, v in lo.items()}
s1["V6_feature_removed"] = row
print("V6_feature_removed", {k: v for k, v in row.items() if not k.startswith("lolo_2")})
base = s1["V0_as_published"]
for k, v in s1.items():
    v["delta_cv_25"] = round(v["cv_auc_25"] - base["cv_auc_25"], 4)
    v["delta_cv_26"] = round(v["cv_auc_26"] - base["cv_auc_26"], 4)
    v["delta_lolo_25"] = round(v["lolo_mean_25"] - base["lolo_mean_25"], 4)
worst = max(abs(v["delta_cv_25"]) for k, v in s1.items() if k != "V0_as_published")
verdict1 = ("no dependence (<= 0.010)" if worst <= 0.010 else
            "dependent (> 0.020): robustness sentence withdrawn for this property" if worst > 0.020 else
            "partial dependence (0.010–0.020)")
s1["_decision"] = {"largest_abs_delta_cv_25": round(worst, 4), "rule_outcome": verdict1}
print("S1 decision:", s1["_decision"])
# the cluster feature treats the two glottal symbols differently — count it
cl_shift = (P.featurize(f.form.map(CONV["V5_all_as_glottal_letter"]), f.concept).n_consonant_clusters
            - F0.n_consonant_clusters)
s1["_note_cluster_feature"] = {
    "forms_whose_cluster_count_changes_when_apostrophe_is_written_as_glottal_letter":
        {L: int((cl_shift[langs == L] != 0).sum()) for L in sorted(set(langs))},
    "forms_where_glottal_letter_plus_consonant_is_counted_as_a_cluster":
        {L: int(((F0.n_consonant_clusters - P.featurize(f.form.map(CONV["V3_unwritten"]).where(
            f.form.map(CONV["V3_unwritten"]) != "", f.form), f.concept).n_consonant_clusters)[langs == L] > 0).sum())
         for L in sorted(set(langs))}}
print(s1["_note_cluster_feature"])
save("S1_glottal_conventions.json", s1)

# =============================================================================
hdr("S2  Makasar: retention from PMP vs share of candidates (reviewer 1, point 2)")
# =============================================================================
cog = pd.read_csv(P.ABVD / "cognates.csv", dtype=str, keep_default_na=False,
                  usecols=["Form_ID", "Cognateset_ID", "Doubt"])
sets_all = cog.groupby("Form_ID").Cognateset_ID.apply(set).to_dict()
sets_sure = cog[cog.Doubt == "false"].groupby("Form_ID").Cognateset_ID.apply(set).to_dict()
fa = P.all_forms()


def proto_sets(lid, table):
    d = defaultdict(set)
    for r in fa[fa.Language_ID == lid].itertuples():
        d[r.Parameter_ID] |= table.get(r.ID, set())
    return d


def retention(proto_id, table):
    ps = proto_sets(proto_id, table)
    rows = {}
    for L, g in f.groupby("language"):
        ret = codednot = unc = 0
        base_n = 0
        for pid, gg in g.groupby("Parameter_ID"):
            if not ps.get(pid):
                continue
            base_n += 1
            fs = [table.get(i, set()) for i in gg.ID]
            if any(s & ps[pid] for s in fs):
                ret += 1
            elif gg.coded.sum() > 0:
                codednot += 1
            else:
                unc += 1
        rows[L] = {"concepts_in_base": base_n, "retained": ret, "coded_not_proto": codednot, "uncoded": unc,
                   "pct_retained": round(100 * ret / base_n, 1), "pct_not_retained": round(100 * (1 - ret / base_n), 1),
                   "pct_coded_not_proto": round(100 * codednot / base_n, 1), "pct_uncoded": round(100 * unc / base_n, 1)}
    return rows


inconsistent = int(sum((r.Cognacy.strip() != "") != bool(sets_all.get(r.ID)) for r in f.itertuples()))
s2 = {"unit": "concept (meaning); base = concepts present in both the language list and the proto list",
      "forms_where_Cognacy_field_and_cognates_table_disagree": inconsistent,
      "PMP_sure": retention("269", sets_sure), "PMP_incl_doubtful": retention("269", sets_all),
      "PAn_sure": retention("280", sets_sure)}
# form-level and concept-level candidate shares, for the same table
s2["candidate_share"] = {
    L: {"forms": int(len(g)), "candidate_forms": int(g.candidate.sum()),
        "pct_forms_candidate": round(100 * g.candidate.mean(), 1),
        "pct_forms_table1_residual": round(100 * g.in_table1_set.mean(), 1),
        "concepts": int(g.concept.nunique()),
        "pct_concepts_all_forms_candidate": round(100 * (g.groupby("concept").coded.sum() == 0).mean(), 1)}
    for L, g in f.groupby("language")}
mk = s2["PMP_sure"]["Makassar"]
s2["_decision"] = {
    "makasar_pct_not_retained_from_PMP": mk["pct_not_retained"],
    "within_10_points_of_62": bool(abs(mk["pct_not_retained"] - 62) <= 10),
    "reading": "same kind of quantity as the published 62 %; the manuscript's figure is the narrower 'uncoded' class"
    if abs(mk["pct_not_retained"] - 62) <= 10 else "differs from the published figure by more than 10 points — report with reasons",
    "south_sulawesi_rows": {L: s2["PMP_sure"][L] for L in ("Makassar", "Bugis", "Toraja-Sadan")}}
print(pd.DataFrame(s2["PMP_sure"]).T.to_string())
print("incl. doubtful:\n", pd.DataFrame(s2["PMP_incl_doubtful"]).T[["pct_retained", "pct_coded_not_proto", "pct_uncoded"]].to_string())
print("PAn:\n", pd.DataFrame(s2["PAn_sure"]).T[["concepts_in_base", "pct_retained", "pct_coded_not_proto", "pct_uncoded"]].to_string())
print(pd.DataFrame(s2["candidate_share"]).T.to_string())
print("S2 decision:", s2["_decision"]["makasar_pct_not_retained_from_PMP"], s2["_decision"]["reading"])
save("S2_makasar_retention.json", s2)

# =============================================================================
hdr("S3  low-level inheritance / 'documentation gaps' (reviewer 1, points 1, 7, 13)")
# =============================================================================
langs_tab = pd.read_csv(P.ABVD / "languages.csv", dtype=str, keep_default_na=False)
langs_tab["lat"] = pd.to_numeric(langs_tab.Latitude, errors="coerce")
langs_tab["lon"] = pd.to_numeric(langs_tab.Longitude, errors="coerce")
LNAME = dict(zip(langs_tab.ID, langs_tab.Name))
GLOT = dict(zip(langs_tab.ID, langs_tab.Glottocode))
BT_OTHER = ["41"] + [str(i) for i in range(875, 909)] + [str(i) for i in range(914, 920)] + ["972"]
BT_OTHER = [i for i in BT_OTHER if i in LNAME]
TOLAKI_DIALECTS = [str(i) for i in range(909, 914)]
PBT = ["780"]


def comp_index(list_ids):
    """concept -> {normalised form: (list name, original form)}"""
    idx = defaultdict(dict)
    sub = fa[fa.Language_ID.isin(list_ids)]
    for r in sub.itertuples():
        v = r.Value if r.Value else r.Form
        n = P.norm(P.clean_form(v))
        if len(n) >= 2:
            idx[P.PNAME.get(r.Parameter_ID, r.Parameter_ID)].setdefault(n, (LNAME[r.Language_ID], v))
    return idx


def best_match(nform, concept, idx):
    best, who = 9.0, ("", "")
    for g, src in idx.get(concept, {}).items():
        d = P.d_stem(nform, g)
        if d < best:
            best, who = d, src
            if d == 0:
                break
    return best, who


def lookalike_table(sub, idx, k_chance, rng):
    concepts = [c for c in idx if idx[c]]
    rows = []
    for r in sub.itertuples():
        n = P.norm(r.form)
        if len(n) < 2:
            continue
        d, who = best_match(n, r.concept, idx)
        others = rng.sample([c for c in concepts if c != r.concept], k_chance)
        ch = [best_match(n, c, idx)[0] for c in others]
        rows.append({"form_id": r.ID, "language": r.language, "concept": r.concept, "form": r.form,
                     "candidate": r.candidate, "has_comparison": int(bool(idx.get(r.concept))),
                     "d": d if d < 9 else np.nan, "match_list": who[0], "match_form": who[1],
                     **{f"chance_{t}": float(np.mean([c <= t for c in ch])) for t in (0.25, 0.34, 0.50)}})
    return pd.DataFrame(rows)


def summarise(t):
    out = {}
    for lab, g in (("candidate", t[t.candidate == 1]), ("coded", t[t.candidate == 0])):
        g = g[g.has_comparison == 1]
        out[lab] = {"n_forms_with_comparison": int(len(g))}
        for thr in (0.25, 0.34, 0.50):
            obs, ch = float((g.d <= thr).mean()), float(g[f"chance_{thr}"].mean())
            out[lab][f"thr_{thr}"] = {"observed_pct": round(100 * obs, 1), "chance_pct": round(100 * ch, 1),
                                      "excess_points": round(100 * (obs - ch), 1)}
    return out


rng = random.Random(SEED)
tol = f[f.language == "Tolaki"]
idx_pbt, idx_bt, idx_dial = comp_index(PBT), comp_index(BT_OTHER), comp_index(TOLAKI_DIALECTS)
idx_low = comp_index(PBT + BT_OTHER)
t_pbt = lookalike_table(tol, idx_pbt, 50, rng)
t_bt = lookalike_table(tol, idx_bt, 50, rng)
t_low = lookalike_table(tol, idx_low, 50, rng)
t_dial = lookalike_table(tol, idx_dial, 50, rng)
s3 = {"S3a_tolaki": {"proto_bungku_tolaki": summarise(t_pbt), "other_bungku_tolaki_lists": summarise(t_bt),
                     "proto_or_other_BT (decision set)": summarise(t_low), "tolaki_dialect_lists": summarise(t_dial),
                     "n_other_BT_lists": len(BT_OTHER)}}
# is the proto form itself coded in ABVD where the Tolaki reflex is not?
pbt_forms = fa[fa.Language_ID == "780"].assign(concept=lambda d: d.Parameter_ID.map(P.PNAME))
pbt_coded_concepts = set(pbt_forms[pbt_forms.Cognacy.str.strip() != ""].concept)
cand_match = t_pbt[(t_pbt.candidate == 1) & (t_pbt.d <= 0.34)]
s3["S3a_tolaki"]["candidates_matching_PBT_at_0.34"] = {
    "n": int(len(cand_match)),
    "of_which_PBT_entry_is_cognate_coded_in_ABVD": int(cand_match.concept.isin(pbt_coded_concepts).sum())}
exc = s3["S3a_tolaki"]["proto_or_other_BT (decision set)"]["candidate"]["thr_0.34"]["excess_points"]
s3["S3a_tolaki"]["_decision"] = {
    "excess_points_candidates_vs_chance": exc,
    "rule_outcome": "substantial low-level inheritance among Tolaki candidates (>= 20 points)" if exc >= 20
    else "little low-level sharing detected by this screen (< 20 points)"}
print(json.dumps(s3["S3a_tolaki"], indent=1, ensure_ascii=False))

box = langs_tab[(langs_tab.lat > -6.6) & (langs_tab.lat < 2.0) & (langs_tab.lon > 118.5) & (langs_tab.lon < 125.6)]
box = box[~box.Name.str.startswith("Bajo")]
s3b, t_all = {}, []
for lid, L in P.TARGET.items():
    ids = [i for i in box.ID if GLOT[i] != GLOT[lid] and i != lid]
    idx = comp_index(ids)
    t = lookalike_table(f[(f.language == L)], idx, 10, rng)
    t_all.append(t)
    s3b[L] = {"n_comparison_lists": len(ids), **summarise(t)}
    print(L, len(ids), json.dumps(s3b[L], ensure_ascii=False), flush=True)
s3["S3b_six_languages_vs_other_sulawesi_lists"] = s3b
t_all = pd.concat(t_all, ignore_index=True)
save("S3_lookalikes.json", s3)
t_low.to_csv(OUT / "S3a_tolaki_vs_bungku_tolaki.csv", index=False, encoding="utf-8")
t_all.to_csv(OUT / "S3b_six_languages_vs_sulawesi.csv", index=False, encoding="utf-8")

# =============================================================================
hdr("S4  agreement between label and classifier, out of fold (R1 pts 11–12; R2)")
# =============================================================================
s4 = {}
auc25, sd25, oof25 = P.cv_auc(F0[P.PURE25].values, y)
auc26, sd26, oof26 = P.cv_auc(F0[P.ABL26].values, y)
lolo25, p_lolo25 = P.lolo_auc(F0[P.PURE25].values, y, langs)
f["p_cand_oof25"], f["p_cand_oof26"], f["p_cand_lolo25"] = oof25, oof26, p_lolo25


def cells(p):
    ml = (p >= 0.5).astype(int)
    c = np.where((f.candidate == 1) & (ml == 1), "candidate & profile",
                 np.where((f.candidate == 0) & (ml == 0), "coded & no profile",
                          np.where((f.candidate == 1) & (ml == 0), "candidate & no profile", "coded & profile")))
    return c, float(cohen_kappa_score(f.candidate, ml))


for tag, p in (("oof_25_features", oof25), ("oof_26_features", oof26), ("lolo_25_features", p_lolo25)):
    c, k = cells(p)
    s4[tag] = {"kappa": round(k, 3), "auc_of_averaged_probability": round(float(roc_auc_score(f.candidate, p)), 3),
               "cells": pd.Series(c).value_counts().to_dict(),
               "candidate_and_profile_by_language": pd.Series(langs[c == "candidate & profile"]).value_counts().to_dict(),
               "share_of_candidates_with_profile_pct": round(100 * float(((f.candidate == 1) & (p >= 0.5)).sum()) / 438, 1)}
    print(tag, json.dumps(s4[tag], ensure_ascii=False))
s4["cv_auc"] = {"25_features": [round(auc25, 4), round(sd25, 4)], "26_features": [round(auc26, 4), round(sd26, 4)],
                "lolo_25": {k: round(v, 3) for k, v in lolo25.items()}}
s4["published_in_sample"] = {"kappa": 0.611, "cells": {"CS": 266, "CA": 878, "RO": 172, "MO": 41}}
f["cell"], _ = cells(oof25)
f["dist_from_half"] = (f.p_cand_oof25 - 0.5).abs()
ex_rows = []
for cell, g in f.groupby("cell"):
    for scope, gg in [("all six", g)] + [(L, g[g.language == L]) for L in sorted(set(langs))]:
        for r in gg.sort_values("dist_from_half", ascending=False).head(5 if scope == "all six" else 3).itertuples():
            ex_rows.append({"cell": cell, "scope": scope, "language": r.language, "concept": r.concept, "form": r.raw,
                            "abvd_form_id": r.ID, "cognacy": r.Cognacy, "p_candidate_oof25": round(r.p_cand_oof25, 3),
                            "on_E022_15_concept_list": int(r.concept in P.PAN_LIST)})
pd.DataFrame(ex_rows).to_csv(OUT / "S4_cell_examples.csv", index=False, encoding="utf-8")
# the five concepts reviewer 2 asks about
five = ["One Hundred", "Fifty", "Twenty", "to stand", "to hit"]
s4["reviewer2_five_concepts"] = {
    c: {"candidate_forms": int(g.candidate.sum()), "of": int(len(g)),
        "languages_with_candidate_and_profile": int(g[(g.candidate == 1) & (g.p_cand_oof25 >= 0.5)].language.nunique()),
        "forms": "; ".join(f"{r.language[:3]} {r.raw}{'' if r.candidate else ' [coded]'}" for r in g.itertuples())}
    for c, g in f[f.concept.isin(five)].groupby("concept")}
cc = f[(f.candidate == 1) & (f.p_cand_oof25 >= 0.5)].groupby("concept").language.nunique()
s4["concepts_candidate_and_profile_in_4plus_languages"] = sorted(cc[cc >= 4].index)
print(json.dumps(s4["reviewer2_five_concepts"], indent=1, ensure_ascii=False))
print("4+ languages:", s4["concepts_candidate_and_profile_in_4plus_languages"])
save("S4_out_of_fold_agreement.json", s4)

# =============================================================================
hdr("S5  same-meaning vs different-meaning candidates across languages (permutation)")
# =============================================================================
nf = [P.norm(x) for x in f.form]
n = len(nf)
DM = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    a = nf[i]
    for j in range(i + 1, n):
        DM[i, j] = DM[j, i] = P.ned(a, nf[j])
lang_code = pd.factorize(f.language)[0]
conc_code = pd.factorize(f.concept)[0]
cross = lang_code[:, None] != lang_code[None, :]
iu = np.triu(np.ones((n, n), dtype=bool), 1)


def perm_test(mask, n_perm, pair_filter=None, seed=SEED):
    idx = np.where(mask)[0]
    D = DM[np.ix_(idx, idx)]
    pairs = cross[np.ix_(idx, idx)] & iu[np.ix_(idx, idx)]
    if pair_filter is not None:
        pairs &= pair_filter[np.ix_(idx, idx)]
    cc_, ll_ = conc_code[idx].copy(), lang_code[idx]
    same = (cc_[:, None] == cc_[None, :]) & pairs
    n_pairs = int(same.sum())
    if n_pairs == 0:
        return {"n_forms": int(len(idx)), "n_pairs": 0}
    T, N = float(D[same].mean()), int((D[same] <= 0.34).sum())
    rs = np.random.default_rng(seed)
    Tp, Np = np.empty(n_perm), np.empty(n_perm)
    groups = [np.where(ll_ == l)[0] for l in np.unique(ll_)]
    for k in range(n_perm):
        c2 = cc_.copy()
        for g in groups:
            c2[g] = cc_[rs.permutation(g)]
        s_ = (c2[:, None] == c2[None, :]) & pairs
        v = D[s_]
        Tp[k], Np[k] = v.mean(), (v <= 0.34).sum()
    return {"n_forms": int(len(idx)), "n_pairs": n_pairs, "T_observed": round(T, 4),
            "T_permutation_mean": round(float(Tp.mean()), 4), "T_permutation_sd": round(float(Tp.std()), 4),
            "p_T": round(float(((Tp <= T).sum() + 1) / (n_perm + 1)), 5),
            "lookalike_pairs_observed": N, "lookalike_pairs_expected": round(float(Np.mean()), 2),
            "p_N": round(float(((Np >= N).sum() + 1) / (n_perm + 1)), 5), "n_perm": n_perm}


cand = (f.candidate == 1).values
not_num = ~f.concept.isin(P.NUMERALS).values
not_pan = ~f.concept.isin(P.PAN_LIST).values
ss = f.language.isin(P.SOUTH_SULAWESI).values
both_ss = ss[:, None] & ss[None, :]
s5 = {
    "a_all_candidates": perm_test(cand, 10000),
    "b_candidates_without_numerals (PRIMARY)": perm_test(cand & not_num, 10000),
    "c_b_without_E022_15_concepts": perm_test(cand & not_num & not_pan, 10000),
    "d_candidates_with_oof_profile": perm_test(cand & (f.p_cand_oof25 >= 0.5).values, 10000),
    "d2_d_without_numerals": perm_test(cand & (f.p_cand_oof25 >= 0.5).values & not_num, 10000),
    "b_pairs_inside_south_sulawesi": perm_test(cand & not_num, 10000, pair_filter=both_ss),
    "b_pairs_other": perm_test(cand & not_num, 10000, pair_filter=~both_ss),
    "positive_control_coded_forms": perm_test(~cand, 1000),
}
for k, v in s5.items():
    print(k, v, flush=True)
pb = s5["b_candidates_without_numerals (PRIMARY)"]
s5["_decision"] = {"p_T_primary": pb["p_T"], "p_N_primary": pb["p_N"],
                   "rule_outcome": "negative result stands (p >= 0.05)" if pb["p_T"] >= 0.05 and pb["p_N"] >= 0.05
                   else "a shared component is detectable (p < 0.05): downgrade the statement and list the pairs"}
print("S5 decision:", s5["_decision"])
# observed look-alike pairs among candidates (all concepts), for specialists
idx = np.where(cand)[0]
pr = []
for a_, b_ in combinations(idx, 2):
    if lang_code[a_] != lang_code[b_] and conc_code[a_] == conc_code[b_] and DM[a_, b_] <= 0.34:
        pr.append({"concept": f.concept.iloc[a_], "language_1": f.language.iloc[a_], "form_1": f.raw.iloc[a_],
                   "language_2": f.language.iloc[b_], "form_2": f.raw.iloc[b_], "distance": round(float(DM[a_, b_]), 3),
                   "numeral": int(f.concept.iloc[a_] in P.NUMERALS), "on_E022_15_concept_list": int(f.concept.iloc[a_] in P.PAN_LIST),
                   "both_south_sulawesi": int(ss[a_] and ss[b_])})
pd.DataFrame(pr).sort_values(["concept", "distance"]).to_csv(OUT / "S5_candidate_lookalike_pairs.csv", index=False, encoding="utf-8")
s5["n_lookalike_pairs_listed"] = len(pr)
save("S5_permutation.json", s5)

# =============================================================================
hdr("S6  the 16 additional languages, no language-level input")
# =============================================================================
ex = pd.read_csv(P.EXP / "E027_ml_substrate_detection" / "results" / "expansion_summary.csv")
ex = ex[ex.group != "Original"]
exmap = {str(r.language_id): r.language for r in ex.itertuples()}
grp = dict(zip(ex.language, ex.group))
g16 = P.load_lists(exmap)
Fx = P.featurize(g16.form, g16.concept)
clf = P.xgb().fit(F0[P.PURE25].values.astype(float), y)
g16["p_cand"] = 1 - clf.predict_proba(Fx[P.PURE25].values.astype(float))[:, 1]
rows = []
for L, g in g16.groupby("language"):
    rows.append({"language": L, "group": grp[L], "n_forms": len(g), "pct_candidate_label": round(100 * g.candidate.mean(), 1),
                 "mean_p": round(float(g.p_cand.mean()), 3),
                 "mean_p_coded_forms_only": round(float(g[g.candidate == 0].p_cand.mean()), 3),
                 "pct_predicted_profile": round(100 * float((g.p_cand >= 0.5).mean()), 1),
                 "auc_vs_own_label": round(float(roc_auc_score(g.candidate, g.p_cand)), 3),
                 "published_mean_p": float(ex[ex.language == L].mean_p_substrate.iloc[0]),
                 "published_auc": float(ex[ex.language == L].auc.iloc[0])})
t6 = pd.DataFrame(rows).sort_values(["group", "language"])
print(t6.to_string(index=False))


def exact_perm(col):
    a = t6[t6.group == "Sulawesi"][col].values
    b = t6[t6.group == "W.Indonesian"][col].values
    allv = np.concatenate([a, b])
    obs = a.mean() - b.mean()
    cnt = tot = 0
    for comb in combinations(range(len(allv)), len(b)):
        m = np.zeros(len(allv), dtype=bool); m[list(comb)] = True
        d = allv[~m].mean() - allv[m].mean()
        cnt += abs(d) >= abs(obs) - 1e-12; tot += 1
    return {"sulawesi_mean": round(float(a.mean()), 3), "western_mean": round(float(b.mean()), 3),
            "difference": round(float(obs), 3), "p_exact_two_sided": round(cnt / tot, 4), "n_splits": tot}


s6 = {"per_language": t6.to_dict("records"), "contrast_mean_p": exact_perm("mean_p"),
      "contrast_mean_p_coded_forms_only": exact_perm("mean_p_coded_forms_only"),
      "contrast_pct_candidate_label": exact_perm("pct_candidate_label"),
      "n_languages_auc_ge_0.60": int((t6.auc_vs_own_label >= 0.60).sum()),
      "mean_auc_16": round(float(t6.auc_vs_own_label.mean()), 3)}
ok6 = s6["contrast_mean_p"]["p_exact_two_sided"] < 0.05 and s6["n_languages_auc_ge_0.60"] >= 12
s6["_decision"] = {"rule_outcome": "geographic-pattern sentence may stay" if ok6
                   else "geographic-pattern sentence not supported under the language-free model: remove from abstract"}
print(json.dumps({k: v for k, v in s6.items() if k != "per_language"}, indent=1))
save("S6_expansion_language_free.json", s6)
t6.to_csv(OUT / "S6_expansion_language_free.csv", index=False, encoding="utf-8")

# =============================================================================
hdr("release files (reviewer 1, point 13)")
# =============================================================================
ABVD_NAME = {lid: LNAME[lid] for lid in P.TARGET}
low = t_low.set_index("form_id")
sul = t_all.set_index("form_id")
rel = pd.DataFrame({
    "abvd_form_id": f.ID, "abvd_language_id": f.Language_ID, "abvd_language_name": f.Language_ID.map(ABVD_NAME),
    "concept": f.concept, "form_as_in_abvd": f.raw, "abvd_cognacy": f.Cognacy, "abvd_loan_flag": f.abvd_loan_flag,
    "candidate_no_cognate_set": f.candidate, "in_table1_set_of_submitted_version": f.in_table1_set,
    "concept_on_15_concept_list_of_submitted_version": f.concept.isin(P.PAN_LIST).astype(int),
    "numeral_concept": f.concept.isin(P.NUMERALS).astype(int),
    "p_profile_out_of_fold_25feat": f.p_cand_oof25.round(3), "p_profile_out_of_fold_26feat": f.p_cand_oof26.round(3),
    "p_profile_language_held_out_25feat": f.p_cand_lolo25.round(3), "cell": f.cell,
    "sulawesi_lookalike_distance": f.ID.map(sul.d).round(3), "sulawesi_lookalike_list": f.ID.map(sul.match_list),
    "sulawesi_lookalike_form": f.ID.map(sul.match_form),
    "bungku_tolaki_lookalike_distance": f.ID.map(low.d).round(3), "bungku_tolaki_lookalike_list": f.ID.map(low.match_list),
    "bungku_tolaki_lookalike_form": f.ID.map(low.match_form),
})
rel.to_csv(REL / "p8_forms_all.csv", index=False, encoding="utf-8")
rel[rel.candidate_no_cognate_set == 1].sort_values(["abvd_language_name", "concept"]).to_csv(
    REL / "p8_candidates.csv", index=False, encoding="utf-8")
print("release rows:", len(rel), int(rel.candidate_no_cognate_set.sum()),
      "| Tolaki candidates:", int(((rel.abvd_language_id == "674") & (rel.candidate_no_cognate_set == 1)).sum()),
      "| Tolaki in Table-1 set:", int(((rel.abvd_language_id == "674") & (rel.in_table1_set_of_submitted_version == 1)).sum()))
print("\nDONE")
