"""
E231 -- what "coded" covers, how the lists compare, stricter versions of four inputs (P8 revision).
DESIGN.md (2026-10-06) is binding; this script only counts and gives intervals, it tests no hypothesis.
Run:  PYTHONIOENCODING=utf-8 python experiments/E231_p8_what_coded_means/01_tabulations.py

Class-coding convention (as p8common / E229):
  y = 1 <=> CODED (any ABVD cognate assignment);  cand = 1 - y <=> CANDIDATE (empty Cognacy field).
  P(candidate) = out-of-fold mean of 1 - P(coded).  A positive difference / OR > 1 means "more often among candidates".
Order of work: rebuild everything the anchors need -> check anchors -> only then write tables (E229 pattern).
"""
import io
import json
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
# p8common is imported, never copied or edited.
sys.path.insert(0, str(EXP / "E228_p8_revision_analyses"))
import p8common as P  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
warnings.filterwarnings("ignore")


class Tee:
    """Everything printed also goes to results/run_log.txt (UTF-8; the console is cp1252)."""
    def __init__(self, path):
        self.con = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        self.fh = open(path, "w", encoding="utf-8")

    def write(self, s):
        self.con.write(s)
        self.fh.write(s)

    def flush(self):
        self.con.flush()
        self.fh.flush()


sys.stdout = Tee(OUT / "run_log.txt")
SEED = 231
NB = 2000


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def show(df, **kw):
    print(df.to_string(index=False, **kw), flush=True)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def mh_or(v, c, strata):
    """Mantel-Haenszel OR of CANDIDATE status for a 0/1 property, stratified (Robins-Breslow-Greenland variance).
    Same formula as E229 01_tables.py mh_or (which follows E227 section D), generalised to any strata."""
    v, c, strata = np.asarray(v), np.asarray(c), np.asarray(strata)
    R = S = P_R = P_S_Q_R = Q_S = 0.0
    for L in sorted(set(strata)):
        m = strata == L
        a = ((v[m] == 1) & (c[m] == 1)).sum(); b = ((v[m] == 1) & (c[m] == 0)).sum()
        cc = ((v[m] == 0) & (c[m] == 1)).sum(); d = ((v[m] == 0) & (c[m] == 0)).sum()
        n = a + b + cc + d
        if n == 0:
            continue
        r, s = a * d / n, b * cc / n
        p, q = (a + d) / n, (b + cc) / n
        R += r; S += s; P_R += p * r; P_S_Q_R += p * s + q * r; Q_S += q * s
    if R == 0 or S == 0:
        return (np.nan, np.nan, np.nan)
    orr = R / S
    var = P_R / (2 * R * R) + P_S_Q_R / (2 * R * S) + Q_S / (2 * S * S)
    return orr, float(np.exp(np.log(orr) - 1.96 * np.sqrt(var))), float(np.exp(np.log(orr) + 1.96 * np.sqrt(var)))


# ================================================================================================ data
f = P.load_lists(P.TARGET)
F0 = P.featurize(f.form, f.concept, f.language)
y = f.coded.values
cand = 1 - y
langs = f.language.values
LISTS = list(P.TARGET.values())
assert len(f) == 1357 and int(cand.sum()) == 438
fa = P.all_forms()

lt = pd.read_csv(P.ABVD / "languages.csv", dtype=str, keep_default_na=False)
lt["lat"] = pd.to_numeric(lt.Latitude, errors="coerce")
lt["lon"] = pd.to_numeric(lt.Longitude, errors="coerce")
LNAME, GLOT = dict(zip(lt.ID, lt.Name)), dict(zip(lt.ID, lt.Glottocode))

# ------------------------------------------------------------------ which lists count as "Sulawesi" (DESIGN section 1)
# Choice: box test (strict inequalities, as E228 S3b) for ordinary lists, no Bajo exclusion (the box defines
# "inside Sulawesi"; Bajo exclusion in E228 was only for the comparison set). Lists without coordinates are NOT
# Sulawesi. Proto-languages are outside unless their NAME says Sulawesi or Bungku(-Tolaki); a proto-list's stray
# coordinate (some carry one) is ignored. PMP (269) and PAn (280) are outside by name.
is_proto = lt.Name.str.startswith("Proto")
in_box = (lt.lat > -6.6) & (lt.lat < 2.0) & (lt.lon > 118.5) & (lt.lon < 125.6)
name_says_sul = lt.Name.str.contains("Sulawesi|Bungku")
lt["is_proto"] = is_proto
lt["has_coords"] = lt.lat.notna() & lt.lon.notna()
lt["sulawesi"] = np.where(is_proto, name_says_sul, in_box & lt.has_coords)
SUL = set(lt[lt.sulawesi].ID)
assert "269" not in SUL and "280" not in SUL and "780" in SUL
lt[["ID", "Name", "Glottocode", "Latitude", "Longitude", "is_proto", "has_coords", "sulawesi"]].to_csv(
    OUT / "A_list_classification.csv", index=False, encoding="utf-8")
hdr("which lists count as Sulawesi")
print("lists total", len(lt), "| non-proto lists without coordinates (not Sulawesi):",
      int((~lt.has_coords & ~lt.is_proto).sum()), "| proto-lists:", int(lt.is_proto.sum()),
      "| Sulawesi lists:", len(SUL))
print("proto-lists counted as Sulawesi by name:", lt[lt.is_proto & lt.sulawesi].Name.tolist())
print("proto-lists with a stray coordinate (ignored):", lt[lt.is_proto & lt.has_coords].Name.tolist())
print("six target lists in Sulawesi set:", {L: (lid in SUL) for lid, L in P.TARGET.items()})
print("non-proto lists without coordinates, first 10 by name:", lt[~lt.has_coords & ~lt.is_proto].Name.head(10).tolist())

# ------------------------------------------------------------------ cognate sets from the Cognacy field
# A set = (Parameter_ID, number). Cognacy is comma-separated; a number followed by '?' is doubtful. Tokens that are not
# numbers ('L', 'x20' ...) are not sets and are tallied. E228 used cognates.csv; the two agree in the six lists
# (checked before writing: only a naming difference 'ashes'/'ash' in the set label).
tok_re = re.compile(r"(\d+)(\?)?")
unparsed = Counter()
sets_all, sets_sure = {}, {}
setlangs_all, setlangs_sure = defaultdict(set), defaultdict(set)
for fid, lid, pid, cg in zip(fa.ID, fa.Language_ID, fa.Parameter_ID, fa.Cognacy):
    sa, ss_ = set(), set()
    for t in cg.split(","):
        t = t.strip()
        if not t:
            continue
        m = tok_re.fullmatch(t)
        if not m:
            unparsed[t if not t.isdigit() else "?"] += 1
            continue
        key = (pid, m.group(1))
        sa.add(key)
        if not m.group(2):
            ss_.add(key)
    if sa:
        sets_all[fid] = frozenset(sa)
        for k in sa:
            setlangs_all[k].add(lid)
    if ss_:
        sets_sure[fid] = frozenset(ss_)
        for k in ss_:
            setlangs_sure[k].add(lid)
print("non-numeric Cognacy tokens in all of ABVD (not sets):", dict(unparsed.most_common(8)),
      "| distinct sets:", len(setlangs_all))
_sul_only = {}


def set_sul_only(k):
    if k not in _sul_only:
        _sul_only[k] = all(l in SUL for l in setlangs_all[k])
    return _sul_only[k]


def form_breadth(fid, certain=False):
    """Widest set of the form: (breadth, sulawesi_only). Ties among widest sets: conservative, the form counts as
    Sulawesi-only only if every tied widest set is."""
    ss = (sets_sure if certain else sets_all).get(fid)
    if not ss:
        return (np.nan, np.nan)
    sl = setlangs_sure if certain else setlangs_all
    b = max(len(sl[k]) for k in ss)
    wid = [k for k in ss if len(sl[k]) == b]
    return (b, int(all(set_sul_only(k) for k in wid)))


# ---------------------------------------------------------------- S2 rebuilt (E228 definition), per meaning
def proto_sets(lid, table):
    d = defaultdict(set)
    for fid, pid in zip(fa[fa.Language_ID == lid].ID, fa[fa.Language_ID == lid].Parameter_ID):
        d[pid] |= table.get(fid, frozenset())
    return d


def classify(proto_id, table):
    """Returns {language: {pid: 'retained'|'coded_not_proto'|'uncoded'}}; base = meanings where the proto list has a set."""
    ps = proto_sets(proto_id, table)
    out = {}
    for L, g in f.groupby("language"):
        d = {}
        for pid, gg in g.groupby("Parameter_ID"):
            if not ps.get(pid):
                continue
            fs = [table.get(i, frozenset()) for i in gg.ID]
            if any(s & ps[pid] for s in fs):
                d[pid] = "retained"
            elif gg.coded.sum() > 0:
                d[pid] = "coded_not_proto"
            else:
                d[pid] = "uncoded"
        out[L] = d
    return out


S2 = classify("269", sets_sure)
S2_dbt = classify("269", sets_all)
S2_counts = {L: (sum(v == "retained" for v in d.values()), sum(v == "coded_not_proto" for v in d.values()),
                 sum(v == "uncoded" for v in d.values()), len(d)) for L, d in S2.items()}
hdr("S2 rebuilt (retained / coded-not-PMP / uncoded, base)")
print(S2_counts)

# ---------------------------------------------------------------- S5 primary set rebuilt
nf = {i: P.norm(s) for i, s in zip(f.ID, f.form)}
sel = f[(f.candidate == 1) & (~f.concept.isin(P.NUMERALS))]
pair_rows = []
for pid, g in sel.groupby("concept"):
    rows = list(g.itertuples())
    for a, b in combinations(rows, 2):
        if a.language != b.language:
            pair_rows.append((pid, P.ned(nf[a.ID], nf[b.ID]), a.language, b.language))
pairs = pd.DataFrame(pair_rows, columns=["concept", "d", "l1", "l2"])
T_obs = float(pairs.d.mean())
n_look = int((pairs.d <= 0.34).sum())
print("S5 primary:", len(pairs), "pairs, T =", round(T_obs, 4), "look-alike pairs", n_look)

# ---------------------------------------------------------------- loan flags
loan = f[f.abvd_loan_flag == 1]
print("loan-flagged forms:", len(loan), "of which uncoded:", int((loan.coded == 0).sum()))

# ================================================================================================ E (needed for anchors)
hdr("E  original and strict inputs, summarised like E229 T6")
MARKS = "ʔ'"


def letters(s):
    # alphabetic characters only; ʔ is category Lo so it is excluded by hand; apostrophes, hyphens, spaces,
    # '|' and combining marks are not alphabetic.
    return sum(1 for c in s if c.isalpha() and c != "ʔ")


def nasal_strict(s):
    t = s.lower().replace("ng", "ŋ").replace("ny", "ɲ")
    # 'ñ' (18 forms) is not in the DESIGN's nasal list and is left as it is.
    return int(re.search("[mnŋɲ][pbtdcjkgq]", t) is not None)


def redup_strict(s):
    fl = s.lower()
    for plen in (2, 3):
        for i in range(len(fl) - plen * 2 + 1):
            seq = fl[i:i + plen]
            # 'letter' sequence: every character alphabetic and not a glottal letter (reading of "two- or three-letter")
            if all(c.isalpha() and c != "ʔ" for c in seq) and seq == fl[i + plen:i + plen * 2]:
                return 1
    return 0


nlet = f.form.map(letters)
EV = pd.DataFrame({
    "nasal_original": F0.has_nasal_cluster, "nasal_strict": f.form.map(nasal_strict),
    "redup_original": F0.has_reduplication, "redup_strict": f.form.map(redup_strict),
    "contains_hyphen": f.form.map(lambda s: int("-" in s)),
    "prefix_original": F0.has_prefix_like, "prefix_strict": ((F0.has_prefix_like == 1) & (nlet >= 6)).astype(int),
    "prefix_original_le4_letters": ((F0.has_prefix_like == 1) & (nlet <= 4)).astype(int),
    "length_original": F0.form_length.astype(float), "length_letters": nlet.astype(float)})
EV_INFO = {
    "nasal_original": ("nasal sequence", "original", "0/1", "original NASAL_CLUSTERS substring rule"),
    "nasal_strict": ("nasal sequence", "strict", "0/1", "after ng->eng, ny->ny-palatal: nasal letter + stop letter"),
    "redup_original": ("reduplication", "original", "0/1", "original: hyphen OR repeated 2-3 character block"),
    "redup_strict": ("reduplication", "strict", "0/1", "repeated 2- or 3-letter block only, hyphen rule dropped"),
    "contains_hyphen": ("reduplication", "hyphen rule alone", "0/1", "form contains a hyphen"),
    "prefix_original": ("prefix-like onset", "original", "0/1", "original onset rule"),
    "prefix_strict": ("prefix-like onset", "strict", "0/1", "original onset rule and >= 6 letters"),
    "prefix_original_le4_letters": ("prefix-like onset", "tally", "0/1", "original onset rule flagged AND <= 4 letters"),
    "length_original": ("length", "original", "count", "number of characters of the form"),
    "length_letters": ("length", "letters only", "count", "alphabetic characters only (no glottal marks, hyphens, spaces)"),
}
COUNT_KEYS = ["length_original", "length_letters"]
# percentile bootstrap: 2,000 resamples of forms within list x label, seed 231, equal list weights (as E229, seed differs);
# same resamples used for the two count variables.
rng = np.random.default_rng(SEED)
boot = np.zeros((NB, len(COUNT_KEYS)))
for L in LISTS:
    for lab, sign in ((1, 1.0), (0, -1.0)):
        m = (langs == L) & (cand == lab)
        vals = EV.loc[m, COUNT_KEYS].values
        idx = rng.integers(0, len(vals), size=(NB, len(vals)))
        boot += sign * vals[idx].mean(axis=1) / len(LISTS)
e_rows = []
for key, (inp, var, typ, rule) in EV_INFO.items():
    v = EV[key].values.astype(float)
    r = {"input": inp, "variant": var, "key": key, "type": typ, "rule": rule,
         "n_cand_with": int(((v == 1) & (cand == 1)).sum()) if typ == "0/1" else np.nan,
         "n_coded_with": int(((v == 1) & (cand == 0)).sum()) if typ == "0/1" else np.nan,
         "mean_candidate": float(v[cand == 1].mean()), "mean_coded": float(v[cand == 0].mean())}
    r["pooled_difference"] = r["mean_candidate"] - r["mean_coded"]
    diffs = []
    for L in LISTS:
        m = langs == L
        d = float(v[m & (cand == 1)].mean() - v[m & (cand == 0)].mean())
        r[f"diff_{L}"] = d
        diffs.append(d)
    r["n_lists_same_sign_as_pooled"] = int(sum(np.sign(d) == np.sign(r["pooled_difference"]) and d != 0 for d in diffs))
    if typ == "count":
        j = COUNT_KEYS.index(key)
        pt = np.mean([v[(langs == L) & (cand == 1)].mean() - v[(langs == L) & (cand == 0)].mean() for L in LISTS])
        r.update(effect_type="stratified mean difference (candidate - coded), equal list weights, percentile bootstrap",
                 effect=float(pt), ci_lo=float(np.percentile(boot[:, j], 2.5)), ci_hi=float(np.percentile(boot[:, j], 97.5)))
    else:
        o, lo, hi = mh_or(v, cand, langs)
        r.update(effect_type="Mantel-Haenszel odds ratio of being a candidate, stratified by list", effect=o, ci_lo=lo, ci_hi=hi)
    e_rows.append(r)
E = pd.DataFrame(e_rows)
show(E[["key", "n_cand_with", "n_coded_with", "mean_candidate", "mean_coded", "n_lists_same_sign_as_pooled",
        "effect", "ci_lo", "ci_hi"]].round(3))
print("diff per list:")
show(E[["key"] + [f"diff_{L}" for L in LISTS]].round(3))

# ================================================================================================ G (needed for anchors)
hdr("G  out-of-fold P(candidate), FM25 and F17")
FM25, F17 = P.PURE25, P.PHON + P.INIT
auc25, sd25, oof25 = P.cv_auc(F0[FM25].values, y)
auc17, sd17, oof17 = P.cv_auc(F0[F17].values, y)
print("CV AUC FM25", round(auc25, 4), round(sd25, 4), "| F17", round(auc17, 4), round(sd17, 4), flush=True)

# ================================================================================================ anchors
hdr("ANCHORS (DESIGN section 3)")
anchors = []


def anchor(name, expected, got, tol=None, exact=False):
    ok = bool(expected == got) if exact else bool(abs(float(expected) - float(got)) <= tol)
    anchors.append({"name": name, "expected": expected if isinstance(expected, (str, list, dict, bool)) else float(expected),
                    "got": got if isinstance(got, (str, list, dict, bool)) else float(got), "ok": ok,
                    "tolerance": None if exact else tol})
    print(("OK   " if ok else "FAIL ") + name, "expected", expected, "got", got, flush=True)


anchor("n_forms", 1357, len(f), exact=True)
anchor("n_candidates", 438, int(cand.sum()), exact=True)
for L, exp in (("Makassar", (79, 52, 70, 201)), ("Bugis", (97, 57, 47, 201)), ("Toraja-Sadan", (96, 66, 38, 200))):
    anchor(f"S2 {L} retained/coded-not-PMP/uncoded/base", list(exp), list(S2_counts[L]), exact=True)
anchor("S5 primary pairs", 440, len(pairs), exact=True)
anchor("S5 primary T (4 decimals)", 0.8357, round(T_obs, 4), exact=True)
anchor("S5 primary look-alike pairs", 4, n_look, exact=True)
anchor("loan-flagged forms", 11, len(loan), exact=True)
anchor("loan-flagged forms that are uncoded", 6, int((loan.coded == 0).sum()), exact=True)
t6 = pd.read_csv(EXP / "E229_p8_revision_tables" / "results" / "T6_profile.csv", encoding="utf-8").set_index("property_key")
for key, t6key in (("nasal_original", "has_nasal_cluster"), ("redup_original", "has_reduplication"),
                   ("prefix_original", "has_prefix_like"), ("length_original", "form_length")):
    r = E[E.key == key].iloc[0]
    anchor(f"E229 T6 {t6key} mean_candidate", t6.loc[t6key, "mean_candidate"], r.mean_candidate, 0.001)
    anchor(f"E229 T6 {t6key} mean_coded", t6.loc[t6key, "mean_coded"], r.mean_coded, 0.001)
    anchor(f"E229 T6 {t6key} effect", t6.loc[t6key, "effect"], r.effect, 0.001)
    if r.type == "0/1":   # bootstrap CI of the count differs by seed (229 vs 231), so only the MH CIs are anchored
        anchor(f"E229 T6 {t6key} CI lo", t6.loc[t6key, "ci_lo"], r.ci_lo, 0.001)
        anchor(f"E229 T6 {t6key} CI hi", t6.loc[t6key, "ci_hi"], r.ci_hi, 0.001)
rel = pd.read_csv(EXP / "E228_p8_revision_analyses" / "release" / "p8_forms_all.csv", encoding="utf-8",
                  usecols=["abvd_form_id", "p_profile_out_of_fold_25feat"]).set_index("abvd_form_id").loc[f.ID.values]
anchor("FM25 oof P(candidate) max |diff| vs E228 release (<= 0.0006)", 0.0,
       float(np.abs(rel.p_profile_out_of_fold_25feat.values - oof25).max()), 0.0006)
n_fail = sum(not a["ok"] for a in anchors)
(OUT / "anchor_checks.json").write_text(json.dumps({"n_checks": len(anchors), "n_failed": n_fail, "checks": anchors},
                                                   indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n{len(anchors)} anchor checks, {n_fail} failed")
if n_fail:
    print("STOP: an anchor failed. Nothing is adjusted; no table was written. See results/anchor_checks.json.")
    sys.exit(1)


def save(df, name, nd=4):
    df.round(nd).to_csv(OUT / name, index=False, encoding="utf-8")


# ================================================================================================ A
hdr("A  what 'coded' covers")
coded = f[f.coded == 1].copy()
bd = [form_breadth(i) for i in coded.ID]
coded["breadth"] = [b[0] for b in bd]
coded["sul_only"] = [b[1] for b in bd]
bc = [form_breadth(i, certain=True) for i in coded.ID]
coded["breadth_certain"] = [b[0] for b in bc]
coded["doubtful_only"] = coded.breadth_certain.isna().astype(int)
coded["has_doubtful"] = [int(len(sets_all.get(i, ())) > len(sets_sure.get(i, ()))) for i in coded.ID]
print("coded forms with no parsable set number:", int(coded.breadth.isna().sum()))
a_rows = []
for L in LISTS + ["All six"]:
    g = coded if L == "All six" else coded[coded.language == L]
    gb = g[g.breadth.notna()]
    gc = g[g.breadth_certain.notna()]
    a_rows.append({"list": L, "n_coded": len(g), "n_with_set_number": len(gb), "median_breadth": float(gb.breadth.median()),
                   "share_breadth_le5": float((gb.breadth <= 5).mean()), "share_breadth_le20": float((gb.breadth <= 20).mean()),
                   "share_sulawesi_only": float(gb.sul_only.mean()), "share_loan_flagged": float(g.abvd_loan_flag.mean()),
                   "n_loan_flagged": int(g.abvd_loan_flag.sum()),
                   "n_with_doubtful_assignment": int(g.has_doubtful.sum()), "n_only_doubtful": int(g.doubtful_only.sum()),
                   "median_breadth_certain_only": float(gc.breadth_certain.median()),
                   "share_breadth_le5_certain_only": float((gc.breadth_certain <= 5).mean())})
A1 = pd.DataFrame(a_rows)
show(A1.round(3))
save(A1, "A_coded_breadth_by_list.csv")

# middle class per list, meaning level
mid_rows, mid_detail = [], []
for L in LISTS:
    g = f[f.language == L]
    for pid, cls in S2[L].items():
        if cls != "coded_not_proto":
            continue
        gg = g[(g.Parameter_ID == pid) & (g.coded == 1)]
        best = None
        allsul = True
        for i in gg.ID:
            for k in sets_all.get(i, ()):
                b = len(setlangs_all[k])
                allsul &= set_sul_only(k)
                if best is None or b > best[0]:
                    best = (b, [k])
                elif b == best[0]:
                    best[1].append(k)
        if best is None:
            continue
        mid_detail.append({"list": L, "pid": pid, "widest_breadth": best[0],
                           "widest_sul_only": int(all(set_sul_only(k) for k in best[1])), "all_sets_sul_only": int(allsul)})
MD = pd.DataFrame(mid_detail)
for L in LISTS + ["All six (meaning-list pairs)"]:
    g = MD if L.startswith("All") else MD[MD.list == L]
    mid_rows.append({"list": L, "n_middle_class_meanings": S2_counts[L][1] if not L.startswith("All") else
                     sum(S2_counts[x][1] for x in LISTS), "n_with_set": len(g),
                     "median_widest_breadth": float(g.widest_breadth.median()),
                     "share_widest_le5": float((g.widest_breadth <= 5).mean()),
                     "share_widest_sulawesi_only": float(g.widest_sul_only.mean()),
                     "share_all_sets_sulawesi_only": float(g.all_sets_sul_only.mean())})
A2 = pd.DataFrame(mid_rows)
show(A2.round(3))
save(A2, "A_middle_class_breadth.csv")

lo = loan.copy()
lo["parsed_sets"] = [";".join(f"{p.split('_', 1)[1]}-{n}" for p, n in sorted(sets_all.get(i, ()))) for i in lo.ID]
lo["breadth"] = [form_breadth(i)[0] for i in lo.ID]
lo["sulawesi_only"] = [form_breadth(i)[1] for i in lo.ID]
A3 = lo[["language", "concept", "raw", "ID", "Cognacy", "Loan", "coded", "parsed_sets", "breadth", "sulawesi_only"]].rename(
    columns={"raw": "form", "ID": "abvd_form_id", "Cognacy": "cognacy_field", "Loan": "loan_field"})
show(A3)
save(A3, "A_loan_flagged.csv")

# ================================================================================================ B
hdr("B  pairwise cognate-set sharing between the six lists")
b_rows = []
by = {L: {pid: g for pid, g in gg.groupby("Parameter_ID")} for L, gg in f.groupby("language")}


def has_set(ids, table):
    s = set()
    for i in ids:
        s |= table.get(i, frozenset())
    return s


for L1, L2 in combinations(LISTS, 2):
    shared = sorted(set(by[L1]) & set(by[L2]))
    k_all = k_sure = k_neither = k_bothcoded = 0
    for pid in shared:
        g1, g2 = by[L1][pid], by[L2][pid]
        if has_set(g1.ID, sets_all) & has_set(g2.ID, sets_all):
            k_all += 1
        if has_set(g1.ID, sets_sure) & has_set(g2.ID, sets_sure):
            k_sure += 1
        c1, c2 = g1.coded.sum() > 0, g2.coded.sum() > 0
        k_neither += int(not c1 and not c2)
        k_bothcoded += int(c1 and c2)
    n = len(shared)
    lo_, hi_ = wilson(k_all, n)
    lo2, hi2 = wilson(k_sure, n)
    b_rows.append({"list_1": L1, "list_2": L2, "meanings_in_both": n, "n_share_set": k_all, "share_set": k_all / n,
                   "ci_lo": lo_, "ci_hi": hi_, "n_share_set_certain_only": k_sure, "share_set_certain_only": k_sure / n,
                   "ci_lo_certain_only": lo2, "ci_hi_certain_only": hi2, "n_neither_has_coded_form": k_neither,
                   "share_neither_has_coded_form": k_neither / n, "n_both_have_coded_form": k_bothcoded})
B = pd.DataFrame(b_rows)
show(B.round(3))
save(B, "B_pairwise_cognate_sharing.csv")

# ================================================================================================ C
hdr("C  intervals and paired tests for the PMP table")
c_rows = []
for L in LISTS:
    n = S2_counts[L][3]
    for lab, k in zip(("retained_from_PMP", "coded_not_PMP", "uncoded"), S2_counts[L][:3]):
        lo_, hi_ = wilson(k, n)
        c_rows.append({"list": L, "class": lab, "n": k, "base": n, "share": k / n, "ci_lo": lo_, "ci_hi": hi_})
C1 = pd.DataFrame(c_rows)
show(C1.round(3))
save(C1, "C_retention_intervals.csv")
t_rows = []
for other in ("Bugis", "Toraja-Sadan"):
    both = sorted(set(S2["Makassar"]) & set(S2[other]))
    for label, cls in (("retained_from_PMP", "retained"), ("uncoded", "uncoded")):
        a = np.array([S2["Makassar"][p] == cls for p in both])
        b = np.array([S2[other][p] == cls for p in both])
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        t_rows.append({"comparison": f"Makassar vs {other}", "class": label, "meanings_in_both_bases": len(both),
                       "makassar_yes_other_no": n10, "makassar_no_other_yes": n01, "both_yes": int((a & b).sum()),
                       "both_no": int((~a & ~b).sum()), "makassar_share": a.mean(), "other_share": b.mean(),
                       "p_exact_two_sided_unadjusted": p})
C2 = pd.DataFrame(t_rows)
show(C2.round(4))
save(C2, "C_paired_tests.csv")

# ================================================================================================ D
hdr("D  Bungku-Tolaki comparison lists and PMP look-alikes")
BT42 = ["41"] + [str(i) for i in range(875, 909)] + [str(i) for i in range(914, 920)] + ["972"]
BT42 = [i for i in BT42 if i in LNAME]
DIAL = [str(i) for i in range(909, 914)]
assert len(BT42) == 42, len(BT42)
lt_i = lt.set_index("ID")
d_rows = []
for grp, ids in (("Proto-Bungku-Tolaki", ["780"]), ("comparison_list", BT42), ("tolaki_dialect", DIAL)):
    for i in ids:
        g = P.load_lists({i: LNAME[i]})
        d_rows.append({"group": grp, "id": i, "name": LNAME[i], "glottocode": GLOT[i], "author": lt_i.loc[i, "author"],
                       "source": lt_i.loc[i, "source"], "n_forms": len(g), "n_coded": int(g.coded.sum()),
                       "coded_share": float(g.coded.mean()) if len(g) else np.nan})
D1 = pd.DataFrame(d_rows)
save(D1, "D_bt_lists.csv")
d42, d5, dp = D1[D1.group == "comparison_list"], D1[D1.group == "tolaki_dialect"], D1[D1.group == "Proto-Bungku-Tolaki"].iloc[0]
d_summary = {
    "PBT": {"n_forms": int(dp.n_forms), "n_coded": int(dp.n_coded), "coded_share": float(dp.coded_share)},
    "comparison_42": {"n_lists": len(d42), "coded_share_min": float(d42.coded_share.min()),
                      "coded_share_median": float(d42.coded_share.median()), "coded_share_max": float(d42.coded_share.max()),
                      "n_distinct_glottocodes": int(d42.glottocode.nunique()), "n_distinct_author": int(d42.author.nunique()),
                      "n_distinct_source": int(d42.source.nunique()), "authors": sorted(set(d42.author)),
                      "sources": sorted(set(d42.source)), "total_forms": int(d42.n_forms.sum()),
                      "n_lists_coded_share_le_0.5": int((d42.coded_share <= 0.5).sum())},
    "tolaki_dialects_5": {"n_lists": len(d5), "coded_share_min": float(d5.coded_share.min()),
                          "coded_share_median": float(d5.coded_share.median()), "coded_share_max": float(d5.coded_share.max()),
                          "n_distinct_glottocodes": int(d5.glottocode.nunique()), "authors": sorted(set(d5.author)),
                          "sources": sorted(set(d5.source))},
    "tolaki_674_for_reference": {"coded_share": float(1 - cand[langs == "Tolaki"].mean())}}
(OUT / "D_summary.json").write_text(json.dumps(d_summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(d_summary, indent=1, ensure_ascii=False))
show(D1[["group", "id", "name", "glottocode", "author", "source", "n_forms", "coded_share"]].round(3))


# --- PMP look-alikes: same machinery as E228 S3 (copied from 01_revision_analyses.py: comp_index, best_match,
# lookalike_table, summarise), so a rebuild gives the same convention; only the comparison set and the seed differ.
def comp_index(list_ids):
    idx = defaultdict(dict)
    for r in fa[fa.Language_ID.isin(list_ids)].itertuples():
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


def lookalike_table(sub, idx, k_chance, rng_):
    concepts = [c for c in idx if idx[c]]
    rows = []
    for r in sub.itertuples():
        n = P.norm(r.form)
        if len(n) < 2:
            continue
        d, who = best_match(n, r.concept, idx)
        others = rng_.sample([c for c in concepts if c != r.concept], k_chance)
        ch = [best_match(n, c, idx)[0] for c in others]
        rows.append({"form_id": r.ID, "language": r.language, "concept": r.concept, "form": r.form, "raw": r.raw,
                     "candidate": r.candidate, "has_comparison": int(bool(idx.get(r.concept))),
                     "d": d if d < 9 else np.nan, "match_list": who[0], "match_form": who[1],
                     **{f"chance_{t}": float(np.mean([c <= t for c in ch])) for t in (0.25, 0.34)}})
    return pd.DataFrame(rows)


idx_pmp = comp_index(["269"])
dl_rows, tol_tab = [], None
for lid, L in P.TARGET.items():
    rng_ = random.Random(SEED)       # re-seeded per list so a list's chance rate does not depend on list order
    t = lookalike_table(f[f.language == L], idx_pmp, 50, rng_)
    if L == "Tolaki":
        tol_tab = t
    for lab, g in (("candidate", t[t.candidate == 1]), ("coded", t[t.candidate == 0])):
        g = g[g.has_comparison == 1]
        row = {"list": L, "class": lab, "n_forms_with_pmp_entry": len(g)}
        for thr in (0.34, 0.25):
            o, c_ = float((g.d <= thr).mean()), float(g[f"chance_{thr}"].mean())
            row.update({f"n_lookalike_{thr}": int((g.d <= thr).sum()), f"observed_{thr}": o, f"chance_{thr}": c_,
                        f"excess_points_{thr}": 100 * (o - c_)})
        dl_rows.append(row)
D2 = pd.DataFrame(dl_rows)
show(D2.round(3))
save(D2, "D_pmp_lookalikes_by_list.csv")
tt = tol_tab[(tol_tab.candidate == 1) & (tol_tab.d <= 0.34)].sort_values("d")
D3 = tt[["form_id", "raw", "concept", "match_form", "d"]].rename(
    columns={"form_id": "abvd_form_id", "raw": "tolaki_form", "concept": "meaning", "match_form": "pmp_form", "d": "distance"})
D3["distance_le_0.25"] = (D3.distance <= 0.25).astype(int)
D3["tolaki_form_is_loan_flagged"] = D3.abvd_form_id.map(dict(zip(f.ID, f.abvd_loan_flag)))
print("Tolaki candidates that are look-alikes of the PMP form of the same meaning (screen, not etymologies):")
show(D3.round(3))
save(D3, "D_tolaki_pmp_lookalikes.csv")

# ================================================================================================ E, write
save(E, "E_strict_variants.csv")

# ================================================================================================ F
hdr("F  final segment")
SS3 = ["Bugis", "Makassar", "Toraja-Sadan"]
last = f.form.map(lambda s: s[-1].lower())


def fcat(ch):
    if ch in MARKS:
        return "glottal_mark"
    if ch in P.VOWELS:
        return "vowel"
    return "other_consonant"


f["final_cat"] = last.map(fcat)
odd_final = f[(f.final_cat == "other_consonant") & ~last.map(lambda c: c.isalpha())]
print("forms whose last character is neither a letter nor a glottal mark (kept in 'other_consonant'):", len(odd_final),
      Counter(last[odd_final.index]))
f_rows = []
for scope, g in [(L, f[f.language == L]) for L in SS3] + [("pooled three lists", f[f.language.isin(SS3)])]:
    for cat in ("vowel", "glottal_mark", "other_consonant"):
        gg = g[g.final_cat == cat]
        lo_, hi_ = wilson(int(gg.candidate.sum()), len(gg))
        f_rows.append({"block": "final_class_by_list", "list": scope, "group_1": cat, "group_2": "", "n": len(gg),
                       "n_candidate": int(gg.candidate.sum()), "candidate_share": gg.candidate.mean() if len(gg) else np.nan,
                       "odds_ratio": np.nan, "ci_lo": lo_, "ci_hi": hi_})
g3 = f[f.language.isin(SS3) & (f.final_cat != "glottal_mark")]
o, lo_, hi_ = mh_or((g3.final_cat == "vowel").astype(int).values, g3.candidate.values, g3.language.values)
f_rows.append({"block": "MH_ends_in_vowel_vs_other_consonant_glottal_final_set_aside", "list": "three lists, stratified",
               "group_1": "vowel-final", "group_2": "other consonant-final", "n": len(g3), "n_candidate": int(g3.candidate.sum()),
               "candidate_share": g3.candidate.mean(), "odds_ratio": o, "ci_lo": lo_, "ci_hi": hi_})
# reference: the same OR with glottal-final forms kept as 'not vowel' (what the original input does)
g3b = f[f.language.isin(SS3)]
o2, lo2, hi2 = mh_or((g3b.final_cat == "vowel").astype(int).values, g3b.candidate.values, g3b.language.values)
f_rows.append({"block": "MH_ends_in_vowel_all_other_finals_together_reference", "list": "three lists, stratified",
               "group_1": "vowel-final", "group_2": "all non-vowel-final", "n": len(g3b), "n_candidate": int(g3b.candidate.sum()),
               "candidate_share": g3b.candidate.mean(), "odds_ratio": o2, "ci_lo": lo2, "ci_hi": hi2})
# final vowel x carries a glottal mark, all six lists
FV = ["a", "e", "i", "o", "u"]
v = f[f.final_cat == "vowel"].copy()
v["final_vowel"] = last[v.index].map(lambda c: c if c in FV else "other vowel")
v["carries_glottal_mark"] = F0.has_glottal[v.index].astype(int)
for L in LISTS:
    for car in (0, 1):
        for fv in FV + ["other vowel", "any vowel"]:
            gg = v[(v.language == L) & (v.carries_glottal_mark == car)]
            if fv != "any vowel":
                gg = gg[gg.final_vowel == fv]
            if len(gg) == 0:
                continue
            lo_, hi_ = wilson(int(gg.candidate.sum()), len(gg))
            f_rows.append({"block": "vowel_final_forms_by_final_vowel_and_glottal_mark", "list": L,
                           "group_1": fv, "group_2": "carries glottal mark" if car else "no glottal mark", "n": len(gg),
                           "n_candidate": int(gg.candidate.sum()), "candidate_share": gg.candidate.mean(), "odds_ratio": np.nan,
                           "ci_lo": lo_, "ci_hi": hi_})
F = pd.DataFrame(f_rows)
show(F[F.block != "vowel_final_forms_by_final_vowel_and_glottal_mark"].round(3))
show(F[(F.block == "vowel_final_forms_by_final_vowel_and_glottal_mark") & (F.group_1 == "any vowel")].round(3))
save(F, "F_final_segment.csv")
tk = f[(f.language == "Toraja-Sadan") & (last == "k")]
FK = tk[["ID", "concept", "raw", "Cognacy", "coded", "candidate"]].rename(columns={"ID": "abvd_form_id", "raw": "form"})
FK["label"] = np.where(FK.candidate == 1, "candidate", "coded")
FK["carries_glottal_mark_elsewhere"] = F0.has_glottal[tk.index].values
FK = FK.sort_values(["label", "concept"])
print("Sa'dan Toraja forms ending in k:", len(FK), "candidates", int(FK.candidate.sum()),
      "| Toraja forms ending in a glottal mark:", int(((f.language == "Toraja-Sadan") & (f.final_cat == "glottal_mark")).sum()))
show(FK)
save(FK, "F_tae_final_k.csv")

# ================================================================================================ G
hdr("G  intervals for the AUC")
rs_f = np.random.default_rng(SEED)
rs_m = np.random.default_rng(SEED)
pid_arr = f.Parameter_ID.values
uniq = np.unique(pid_arr)
by_p = [np.where(pid_arr == u)[0] for u in uniq]
n = len(f)
g_rows = []
for tag, oof, cvm, cvsd in (("FM25", oof25, auc25, sd25), ("F17", oof17, auc17, sd17)):
    # y_true = candidate, score = out-of-fold P(candidate)
    pooled = float(roc_auc_score(cand, oof))
    bf, bm = np.empty(NB), np.empty(NB)
    for k in range(NB):
        ix = rs_f.integers(0, n, n)
        bf[k] = roc_auc_score(cand[ix], oof[ix])
        ix = np.concatenate([by_p[j] for j in rs_m.integers(0, len(by_p), len(by_p))])
        bm[k] = roc_auc_score(cand[ix], oof[ix])
    g_rows.append({"inputs": tag, "scope": "pooled out-of-fold", "n_forms": n, "auc": pooled,
                   "ci_lo_resample_forms": np.percentile(bf, 2.5), "ci_hi_resample_forms": np.percentile(bf, 97.5),
                   "ci_lo_resample_meanings": np.percentile(bm, 2.5), "ci_hi_resample_meanings": np.percentile(bm, 97.5),
                   "published_protocol_cv_auc_mean": cvm, "published_protocol_cv_auc_sd_seed_means": cvsd})
    for L in LISTS:
        m = langs == L
        g_rows.append({"inputs": tag, "scope": f"out-of-fold inside {L}", "n_forms": int(m.sum()),
                       "auc": float(roc_auc_score(cand[m], oof[m]))})
G = pd.DataFrame(g_rows)
show(G.round(4))
save(G, "G_auc_intervals.csv")

# ================================================================================================ H
hdr("H  leave one concept out, S5 primary set")
N_all, S_all = len(pairs), float(pairs.d.sum())
h_rows = []
for c, g in pairs.groupby("concept"):
    rest = N_all - len(g)
    t_wo = (S_all - float(g.d.sum())) / rest if rest > 0 else np.nan
    h_rows.append({"meaning": c, "n_pairs": len(g), "mean_distance_of_its_pairs": float(g.d.mean()),
                   "min_pairwise_distance": float(g.d.min()), "n_lookalike_pairs_le_0.34": int((g.d <= 0.34).sum()),
                   "T_without_meaning": t_wo, "T_all": T_obs, "change_in_T": t_wo - T_obs})
H = pd.DataFrame(h_rows).sort_values("change_in_T", ascending=False).reset_index(drop=True)
H["rank_by_raise"] = H.index + 1
H["in_top_15"] = (H.rank_by_raise <= 15).astype(int)
print("T (all) =", round(T_obs, 4), "| pairs", N_all, "| meanings with pairs", len(H))
show(H.head(15).round(4))
print("smallest change:", round(float(H.change_in_T.min()), 5), "| largest:", round(float(H.change_in_T.max()), 5))
save(H, "H_leave_one_concept_out.csv", 5)

print("\nfiles:", sorted(p.name for p in OUT.iterdir()))
print("\nDONE")
