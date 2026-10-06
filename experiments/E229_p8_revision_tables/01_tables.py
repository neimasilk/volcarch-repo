"""
E229 -- data files behind the revised tables T1-T7 and Figure 1 of P8 (DESIGN.md, frozen 2026-10-06).

No hypothesis is tested. Every choice that could be tuned after seeing a number (threshold, classifier
settings, which examples are shown, how a property is summarised) is fixed in DESIGN.md.
Run:  PYTHONIOENCODING=utf-8 python experiments/E229_p8_revision_tables/01_tables.py

Class-coding convention used throughout (the trap named in DESIGN §3):
  y = 1  <=>  CODED (ABVD cognate set assigned)         -- as in p8common
  cand = 1 - y  <=>  CANDIDATE (empty Cognacy field)    -- the class of interest in every F1/precision/recall
  P(candidate) = 1 - P(coded);  "predicted candidate"  <=>  P(candidate) >= 0.5   (threshold fixed, never tuned)
All SDs are population SDs (ddof = 0), as in E227/E228 (this is what reproduces the 0.0066 / 0.0082 anchors).
"""
import io
import json
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# UTF-8 stdout first: the Windows console is cp1252 and the data contain IPA characters.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
# p8common is imported, never copied or edited (DESIGN §1).
sys.path.insert(0, str(EXP / "E228_p8_revision_analyses"))
import p8common as P  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = HERE / "results"
TEX = OUT / "tables_tex"
OUT.mkdir(exist_ok=True)
TEX.mkdir(exist_ok=True)
E227R = EXP / "E227_p8_g1_blind_rederivation" / "results"
E228R = EXP / "E228_p8_revision_analyses" / "results"
E228REL = EXP / "E228_p8_revision_analyses" / "release"


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


print("python", sys.version.split()[0], "| xgboost", xgboost.__version__, "| scikit-learn", sklearn.__version__,
      "| shap", shap.__version__, "| numpy", np.__version__, "| pandas", pd.__version__,
      "| matplotlib", matplotlib.__version__)

# ----------------------------------------------------------------------------------------------- data
LABELS = pd.read_csv(HERE / "labels.csv", dtype=str, keep_default_na=False, encoding="utf-8")
LAB = dict(zip(LABELS.key, LABELS.label))
LISTS = list(P.TARGET.values())          # Muna, Bugis, Makassar, Wolio, Toraja-Sadan, Tolaki (DESIGN order)

f = P.load_lists(P.TARGET)
F0 = P.featurize(f.form, f.concept, f.language)
y = f.coded.values                       # 1 = coded
cand = 1 - y                             # 1 = candidate
langs = f.language.values
assert len(f) == 1357 and int(cand.sum()) == 438 and (y == f.coded).all()

FM25 = P.PURE25
F17 = P.PHON + P.INIT
assert len(FM25) == 25 and len(F17) == 17

# --------------------------------------------------------------------------------------- classifiers
def make_xgb():
    return P.xgb()


def make_rf():
    # settings as run for the submitted version (E027; E227 rows C05-C09)
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced",
                                  random_state=42, n_jobs=-1)


def make_lr():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)


CLASSIFIERS = {"XGBoost": (make_xgb, False), "Random Forest": (make_rf, False),
               "Logistic regression": (make_lr, True)}


def metrics(y_te, p_coded):
    """All metrics with the CANDIDATE class as the positive class (DESIGN §3).
    p_coded = predicted P(coded); the candidate probability is its complement."""
    p_cand = 1.0 - p_coded
    pred_cand = (p_cand >= 0.5).astype(int)
    c_te = 1 - y_te
    return {
        # AUC does not depend on which class is called positive; computed on the coded probability
        "auc": roc_auc_score(y_te, p_coded),
        "accuracy": accuracy_score(c_te, pred_cand),
        # zero_division=0: a fold where no candidate is predicted gets precision 0 (counted in n_folds_no_cand_pred)
        "f1_candidate": f1_score(c_te, pred_cand, pos_label=1, zero_division=0),
        "precision_candidate": precision_score(c_te, pred_cand, pos_label=1, zero_division=0),
        "recall_candidate": recall_score(c_te, pred_cand, pos_label=1, zero_division=0),
        # coded class as positive: recode the predictions, not the labels
        "f1_coded": f1_score(y_te, 1 - pred_cand, pos_label=1, zero_division=0),
        "acc_always_coded": float(np.mean(y_te == 1)),
        "no_cand_pred": int(pred_cand.sum() == 0),
    }


def cv_run(X, yy, make, scale):
    """Stratified 5-fold x 10 seeds, random_state = 7*seed+13 (the published protocol).
    Returns per-fold metric table (50 rows) and the out-of-fold P(candidate) averaged over the 10 seeds."""
    X = np.asarray(X, dtype=float)
    rows, oof = [], np.zeros(len(yy))
    for seed in range(10):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 7 + 13)
        for k, (tr, te) in enumerate(skf.split(X, yy)):
            Xtr, Xte = X[tr], X[te]
            if scale:
                sc = StandardScaler().fit(Xtr)        # fitted on the training part only
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            clf = make().fit(Xtr, yy[tr])
            assert list(clf.classes_) == [0, 1]       # column 1 of predict_proba is P(coded)
            p_coded = clf.predict_proba(Xte)[:, 1]
            oof[te] += (1 - p_coded) / 10
            rows.append({"seed": seed, "fold": k, **metrics(yy[te], p_coded)})
    return pd.DataFrame(rows), oof


def lolo_run(X, make):
    """Leave one language out; threshold 0.5 on P(candidate)."""
    X = np.asarray(X, dtype=float)
    out = {}
    for L in LISTS:
        te = langs == L
        clf = make().fit(X[~te], y[~te])
        p_coded = clf.predict_proba(X[te])[:, 1]
        m = metrics(y[te], p_coded)
        share_coded = float(np.mean(y[te] == 1))
        out[L] = {"n": int(te.sum()), "n_candidates": int(cand[te].sum()), "auc": m["auc"],
                  "accuracy": m["accuracy"], "accuracy_majority_answer": max(share_coded, 1 - share_coded),
                  "accuracy_always_coded": share_coded, "f1_candidate": m["f1_candidate"]}
    return out


# ================================================================================================ compute
hdr("T2  cross-validated performance, 2 input sets x 3 classifiers (50 folds each)")
INPUT_SETS = {"FM25": FM25, "F17": F17}
t2_rows, oof_store = [], {}
fold_tables = {}
for iname, cols in INPUT_SETS.items():
    for cname, (mk, sc) in CLASSIFIERS.items():
        ft, oof = cv_run(F0[cols].values, y, mk, sc)
        fold_tables[(iname, cname)] = ft
        oof_store[(iname, cname)] = oof
        row = {"input_set": iname, "n_inputs": len(cols), "classifier": cname}
        for m in ("auc", "accuracy", "f1_candidate", "precision_candidate", "recall_candidate", "f1_coded"):
            seed_means = ft.groupby("seed")[m].mean()
            row[f"{m}_mean"] = ft[m].mean()
            row[f"{m}_sd_seed_means"] = float(np.std(seed_means.values))
            row[f"{m}_sd_folds"] = float(np.std(ft[m].values))
        row["accuracy_always_coded"] = ft.acc_always_coded.mean()
        row["n_folds_no_cand_pred"] = int(ft.no_cand_pred.sum())
        t2_rows.append(row)
        print(iname, cname, {k: round(v, 4) for k, v in row.items() if k.endswith("_mean")}, flush=True)
T2 = pd.DataFrame(t2_rows)

hdr("T3  leave-one-language-out, XGBoost")
t3_rows, lolo_res = [], {}
for iname, cols in INPUT_SETS.items():
    res = lolo_run(F0[cols].values, make_xgb)
    lolo_res[iname] = res
    for L, r in res.items():
        t3_rows.append({"input_set": iname, "list_key": L, "list": LAB[L], **r})
    aucs = np.array([r["auc"] for r in res.values()])
    t3_rows.append({"input_set": iname, "list_key": "SUMMARY", "list": "mean and SD over the six lists",
                    "n": int(sum(r["n"] for r in res.values())),
                    "n_candidates": int(sum(r["n_candidates"] for r in res.values())),
                    "auc": float(aucs.mean()), "auc_sd_over_lists": float(np.std(aucs)),
                    "n_lists_auc_ge_0.65": int((aucs >= 0.65).sum())})
    print(iname, {L: round(r["auc"], 3) for L, r in res.items()}, "mean", round(aucs.mean(), 4),
          ">=0.65:", int((aucs >= 0.65).sum()), flush=True)
T3 = pd.DataFrame(t3_rows)

hdr("T4  CV AUC with and without Tolaki, XGBoost")
t4_rows = []
no_tol = langs != "Tolaki"
for iname, cols in INPUT_SETS.items():
    ft6 = fold_tables[(iname, "XGBoost")]
    ft5, _ = cv_run(F0[cols].values[no_tol], y[no_tol], make_xgb, False)   # same protocol on the reduced set
    a6, a5 = ft6.auc.mean(), ft5.auc.mean()
    t4_rows.append({"input_set": iname, "n_forms_six": len(y), "auc_six_lists": a6,
                    "sd_seed_means_six": float(np.std(ft6.groupby("seed").auc.mean().values)),
                    "n_forms_five": int(no_tol.sum()), "n_candidates_five": int(cand[no_tol].sum()),
                    "auc_without_tolaki": a5,
                    "sd_seed_means_five": float(np.std(ft5.groupby("seed").auc.mean().values)),
                    "difference_without_minus_six": a5 - a6})
    print(iname, round(a6, 4), round(a5, 4), flush=True)
T4 = pd.DataFrame(t4_rows)

hdr("T5  label x profile cells (out-of-fold P(candidate) >= 0.5, FM25, XGBoost)")
oof25 = oof_store[("FM25", "XGBoost")]
ml = (oof25 >= 0.5).astype(int)           # 1 = profile (predicted candidate)
t5_rows = []
for L in LISTS + ["ALL"]:
    m = np.ones(len(y), dtype=bool) if L == "ALL" else (langs == L)
    c, p = cand[m] == 1, ml[m] == 1
    t5_rows.append({"list_key": L, "list": "All six lists" if L == "ALL" else LAB[L], "n": int(m.sum()),
                    "candidate_and_profile": int((c & p).sum()), "candidate_and_no_profile": int((c & ~p).sum()),
                    "coded_and_profile": int((~c & p).sum()), "coded_and_no_profile": int((~c & ~p).sum()),
                    "kappa": float(cohen_kappa_score(cand[m], ml[m]))})
T5 = pd.DataFrame(t5_rows)
print(T5.round(3).to_string(index=False))

hdr("T1  label by list")
t1_rows = []
for L in LISTS + ["ALL"]:
    m = np.ones(len(y), dtype=bool) if L == "ALL" else (langs == L)
    n, nc = int(m.sum()), int(cand[m].sum())
    t1_rows.append({"list_key": L, "list": "All six lists" if L == "ALL" else LAB[L], "forms": n,
                    "coded_n": n - nc, "coded_pct": round(100 * (n - nc) / n, 1),
                    "candidate_n": nc, "candidate_pct": round(100 * nc / n, 1)})
T1 = pd.DataFrame(t1_rows)
print(T1.to_string(index=False))

hdr("T6  profile of candidates vs coded forms")
PROPS = ["form_length", "n_vowels", "n_consonant_clusters", "has_glottal", "has_prefix_like", "has_nasal_cluster",
         "has_reduplication", "ends_in_vowel", "sem_ACTION", "is_core_vocab"]
COUNTS = ["form_length", "n_vowels", "n_consonant_clusters"]
Xp = F0[PROPS].astype(float)


def mh_or(col):
    """Mantel-Haenszel OR of CANDIDATE status for a 0/1 property, stratified by list (Robins-Breslow-Greenland
    variance) -- same formula as E227 section D so the two reproduce each other."""
    R = S = P_R = P_S_Q_R = Q_S = 0.0
    for L in LISTS:
        m = langs == L
        v, c = Xp[col].values[m], cand[m]
        a = ((v == 1) & (c == 1)).sum(); b = ((v == 1) & (c == 0)).sum()
        cc = ((v == 0) & (c == 1)).sum(); d = ((v == 0) & (c == 0)).sum()
        n = a + b + cc + d
        r, s = a * d / n, b * cc / n
        p, q = (a + d) / n, (b + cc) / n
        R += r; S += s; P_R += p * r; P_S_Q_R += p * s + q * r; Q_S += q * s
    orr = R / S
    var = P_R / (2 * R * R) + P_S_Q_R / (2 * R * S) + Q_S / (2 * S * S)
    return orr, float(np.exp(np.log(orr) - 1.96 * np.sqrt(var))), float(np.exp(np.log(orr) + 1.96 * np.sqrt(var)))


# percentile bootstrap, 2,000 resamples of forms within list x label, seed 229 (DESIGN §2).
# The stratified mean difference gives every list the same weight (the DESIGN does not fix a weighting; equal
# weights keep Tolaki, which holds 134 of the 438 candidates, from dominating).  Same resamples are used for the three counts.
rng = np.random.default_rng(229)
NB = 2000
boot = np.zeros((NB, len(COUNTS)))
for L in LISTS:
    for lab, sign in ((1, 1.0), (0, -1.0)):
        m = (langs == L) & (cand == lab)
        vals = Xp.loc[m, COUNTS].values
        idx = rng.integers(0, len(vals), size=(NB, len(vals)))
        boot += sign * vals[idx].mean(axis=1) / len(LISTS)      # (NB, 3)
strat_point = np.zeros(len(COUNTS))
for L in LISTS:
    m = langs == L
    strat_point += (Xp.loc[m & (cand == 1), COUNTS].mean().values - Xp.loc[m & (cand == 0), COUNTS].mean().values) / len(LISTS)

t6_rows = []
for col in PROPS:
    v = Xp[col].values
    r = {"property_key": col, "property": LAB[col], "type": "count" if col in COUNTS else "0/1",
         "mean_candidate": float(v[cand == 1].mean()), "mean_coded": float(v[cand == 0].mean())}
    r["pooled_difference"] = r["mean_candidate"] - r["mean_coded"]
    diffs = []
    for L in LISTS:
        m = langs == L
        d = float(v[m & (cand == 1)].mean() - v[m & (cand == 0)].mean())
        r[f"diff_{L}"] = d
        diffs.append(d)
    r["n_lists_same_sign_as_pooled"] = int(sum(np.sign(d) == np.sign(r["pooled_difference"]) and d != 0 for d in diffs))
    if col in COUNTS:
        j = COUNTS.index(col)
        r.update(effect_type="stratified mean difference (candidate - coded), equal list weights",
                 effect=float(strat_point[j]),
                 ci_lo=float(np.percentile(boot[:, j], 2.5)), ci_hi=float(np.percentile(boot[:, j], 97.5)))
    else:
        o, lo, hi = mh_or(col)
        r.update(effect_type="Mantel-Haenszel odds ratio of being a candidate, stratified by list",
                 effect=o, ci_lo=lo, ci_hi=hi)
    t6_rows.append(r)
T6 = pd.DataFrame(t6_rows)
print(T6[["property_key", "mean_candidate", "mean_coded", "n_lists_same_sign_as_pooled", "effect", "ci_lo", "ci_hi"]]
      .round(3).to_string(index=False))

hdr("F1  mean |SHAP| of the 25 inputs (XGBoost on all 1,357 forms)")
Xa = F0[FM25].values.astype(float)
clf_all = make_xgb().fit(Xa, y)
sv = shap.TreeExplainer(clf_all).shap_values(Xa)
sv = np.asarray(sv)
assert sv.shape == (1357, 25), sv.shape
r_coded = np.array([np.corrcoef(Xa[:, i], sv[:, i])[0, 1] if Xa[:, i].std() > 0 else np.nan for i in range(25)])
# the model's output is P(coded), so SHAP > 0 pushes toward CODED; flip the sign for the candidate direction
r_cand = -r_coded
F1 = pd.DataFrame({"feature": FM25, "label": [LAB[c] for c in FM25],
                   "group": ["written form" if c in F17 else "meaning" for c in FM25],
                   "mean_abs_shap": np.abs(sv).mean(axis=0),
                   "r_value_vs_shap_toward_candidate": r_cand})
F1["direction"] = ["·" if (not np.isfinite(r)) or abs(r) < 0.3 else ("↑" if r > 0 else "↓") for r in r_cand]
F1 = F1.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
F1["rank"] = F1.index + 1
print(F1.round(3).to_string(index=False))

# ---------------------------------------------------------------------------------------------- T7 inputs
hdr("T7  examples (the 92 rows of E228 S4_cell_examples.csv, unchanged) + extra columns")
ex = pd.read_csv(E228R / "S4_cell_examples.csv", dtype=str, keep_default_na=False, encoding="utf-8")
ex_orig_cols = list(ex.columns)
fi = f.set_index("ID")
fa = P.all_forms()
rel = pd.read_csv(E228REL / "p8_forms_all.csv", dtype=str, keep_default_na=False, encoding="utf-8").set_index("abvd_form_id")


def proto_entry(lid, pid):
    g = fa[(fa.Language_ID == lid) & (fa.Parameter_ID == pid)]
    forms = [(v if v != "" else fo) for v, fo in zip(g.Value, g.Form)]
    codes = [c.strip() if c.strip() != "" else "(none)" for c in g.Cognacy]
    return forms, codes, [c.strip() for c in g.Cognacy]


def code_set(s, certain_only):
    out = set()
    for t in s.split(","):
        t = t.strip()
        if t == "":
            continue
        if t.endswith("?"):
            if certain_only:
                continue
            t = t[:-1]
        out.add(t)
    return out


t7_rows = []
for r in ex.itertuples():
    fid = r.abvd_form_id
    pid = fi.loc[fid, "Parameter_ID"]
    own_code = fi.loc[fid, "Cognacy"]
    row = {c: getattr(r, c) for c in ex_orig_cols}
    row["abvd_loan_flag"] = int(fi.loc[fid, "abvd_loan_flag"])
    pmp_f, pmp_c, pmp_raw = proto_entry("269", pid)
    pan_f, pan_c, _ = proto_entry("280", pid)
    row["pmp_forms"] = " ; ".join(pmp_f)
    row["pmp_cognate_codes"] = " ; ".join(pmp_c)
    row["pan_forms"] = " ; ".join(pan_f)
    row["pan_cognate_codes"] = " ; ".join(pan_c)
    if pmp_f:
        for tag, certain in (("shares_code_with_pmp", False), ("shares_code_with_pmp_certain_only", True)):
            mine = code_set(own_code, certain)
            theirs = set().union(*[code_set(c, certain) for c in pmp_raw])
            row[tag] = int(bool(mine & theirs))
    else:
        row["shares_code_with_pmp"] = ""
        row["shares_code_with_pmp_certain_only"] = ""
    for c in ("sulawesi_lookalike_distance", "sulawesi_lookalike_list", "sulawesi_lookalike_form",
              "bungku_tolaki_lookalike_distance", "bungku_tolaki_lookalike_list", "bungku_tolaki_lookalike_form"):
        row[c] = rel.loc[fid, c]
    t7_rows.append(row)
T7 = pd.DataFrame(t7_rows)
print(len(T7), "rows;", "with PMP entry:", int((T7.pmp_forms != "").sum()), "| with PAn entry:", int((T7.pan_forms != "").sum()))

# =============================================================================================== anchors
hdr("ANCHORS (DESIGN §4)")
anchors = []


def anchor(name, expected, got, tol=None, exact=False):
    if exact:
        ok = bool(expected == got)
    else:
        ok = bool(abs(float(expected) - float(got)) <= tol)
    anchors.append({"name": name, "expected": expected if isinstance(expected, (str, list, dict)) else float(expected),
                    "got": got if isinstance(got, (str, list, dict)) else float(got), "ok": ok,
                    "tolerance": None if exact else tol})
    print(("OK   " if ok else "FAIL ") + name, "expected", expected, "got", got, flush=True)


anchor("n_forms", 1357, len(y), exact=True)
anchor("n_candidates", 438, int(cand.sum()), exact=True)
# DESIGN amendment A1: the anchor is keyed by list, not by position (the first run compared in the wrong order)
PER_LIST = {"Muna": 34, "Bugis": 62, "Makassar": 80, "Wolio": 83, "Toraja-Sadan": 45, "Tolaki": 134}
anchor("candidates_per_list (keyed by list; amendment A1)", PER_LIST,
       {L: int(cand[langs == L].sum()) for L in LISTS}, exact=True)
x25 = T2[(T2.input_set == "FM25") & (T2.classifier == "XGBoost")].iloc[0]
x17 = T2[(T2.input_set == "F17") & (T2.classifier == "XGBoost")].iloc[0]
anchor("cv_auc_FM25_xgb", 0.7265, x25.auc_mean, 0.0005)
anchor("cv_auc_sd_seed_means_FM25_xgb", 0.0066, x25.auc_sd_seed_means, 0.0005)
anchor("cv_auc_F17_xgb", 0.6717, x17.auc_mean, 0.0005)
anchor("cv_auc_sd_seed_means_F17_xgb", 0.0082, x17.auc_sd_seed_means, 0.0005)
s7 = json.loads((E228R / "S7_feature_groups.json").read_text(encoding="utf-8"))
for iname, key in (("FM25", "form_plus_meaning_25"), ("F17", "form_only_17")):
    r3 = T3[(T3.input_set == iname) & (T3.list_key == "SUMMARY")].iloc[0]
    anchor(f"lolo_mean_auc_{iname}", {"FM25": 0.7008, "F17": 0.6413}[iname], r3.auc, 0.0005)
    anchor(f"lolo_mean_auc_{iname}_vs_S7", s7[key]["lolo_mean_auc"], r3.auc, 0.0005)
    anchor(f"lolo_lists_ge_0.65_{iname}_vs_S7", s7[key]["lolo_languages_ge_0.65"], int(r3["n_lists_auc_ge_0.65"]), exact=True)
    for L in LISTS:
        got = float(T3[(T3.input_set == iname) & (T3.list_key == L)].auc.iloc[0])
        # S7 stores 3-decimal values: compare after rounding to 3 decimals
        anchor(f"lolo_auc_{iname}_{L}_vs_S7 (3 decimals)", s7[key]["lolo_per_language"][L], round(got, 3), exact=True)
s4 = json.loads((E228R / "S4_out_of_fold_agreement.json").read_text(encoding="utf-8"))["oof_25_features"]
tot = T5[T5.list_key == "ALL"].iloc[0]
anchor("cells [CP, CnoP, CodedP, CodednoP]", [172, 266, 105, 814],
       [int(tot.candidate_and_profile), int(tot.candidate_and_no_profile), int(tot.coded_and_profile),
        int(tot.coded_and_no_profile)], exact=True)
anchor("cells_vs_S4_json", [s4["cells"]["candidate & profile"], s4["cells"]["candidate & no profile"],
                            s4["cells"]["coded & profile"], s4["cells"]["coded & no profile"]],
       [int(tot.candidate_and_profile), int(tot.candidate_and_no_profile), int(tot.coded_and_profile),
        int(tot.coded_and_no_profile)], exact=True)
anchor("kappa (3 decimals)", 0.308, round(float(tot.kappa), 3), exact=True)
# extra, not in DESIGN §4: the stored per-form probability and cell of the E228 release file
rel_p = pd.read_csv(E228REL / "p8_forms_all.csv", encoding="utf-8", usecols=["abvd_form_id", "p_profile_out_of_fold_25feat", "cell"])
rel_p = rel_p.set_index("abvd_form_id").loc[f.ID.values]
anchor("EXTRA max |oof P(candidate) - E228 release| (<= 0.0006, release is rounded to 3 decimals)", 0.0,
       float(np.abs(rel_p.p_profile_out_of_fold_25feat.values - oof25).max()), 0.0006)
cell_mine = np.where((cand == 1) & (ml == 1), "candidate & profile", np.where((cand == 0) & (ml == 0), "coded & no profile",
                     np.where((cand == 1) & (ml == 0), "candidate & no profile", "coded & profile")))
anchor("EXTRA forms whose cell differs from the E228 release", 0, int((cell_mine != rel_p.cell.values).sum()), exact=True)
# SHAP
sh = pd.read_csv(E227R / "shap_pure25.csv")
anchor("shap_order_of_25_inputs", list(sh.feature), list(F1.feature), exact=True)
shm = F1.set_index("feature").mean_abs_shap
d_shap = float(max(abs(shm[r.feature] - r.mean_abs_shap) for r in sh.itertuples()))
anchor("shap_max_abs_difference_in_mean_abs_shap", 0.0, d_shap, 0.002)
# direction: E227 correlation is toward 'coded'; ours is toward candidate, so r must be its negative
d_r = float(np.nanmax([abs(F1.set_index("feature").r_value_vs_shap_toward_candidate[r.feature] + r.corr_value_shap)
                       for r in sh.itertuples()]))
anchor("EXTRA shap_direction_r_equals_minus_E227_r (max abs difference)", 0.0, d_r, 0.02)
# T6
dl = pd.read_csv(E227R / "direction_by_label.csv").set_index("property")
for col in PROPS:
    r = T6[T6.property_key == col].iloc[0]
    anchor(f"T6 {col} mean_candidate", dl.loc[col, "mean_candidate"], r.mean_candidate, 0.001)
    anchor(f"T6 {col} mean_coded", dl.loc[col, "mean_coded"], r.mean_coded, 0.001)
    if col not in COUNTS:
        anchor(f"T6 {col} MH OR", dl.loc[col, "MH_OR"], r.effect, 0.001)
        anchor(f"T6 {col} MH lo", dl.loc[col, "MH_lo"], r.ci_lo, 0.001)
        anchor(f"T6 {col} MH hi", dl.loc[col, "MH_hi"], r.ci_hi, 0.001)
# T7 unchanged
ex_back = T7[ex_orig_cols].reset_index(drop=True)
anchor("T7 rows and original columns identical to E228 S4_cell_examples.csv", True,
       bool(len(T7) == 92 and ex_back.equals(ex.reset_index(drop=True))), exact=True)

n_fail = sum(not a["ok"] for a in anchors)
(OUT / "anchor_checks.json").write_text(json.dumps({"n_checks": len(anchors), "n_failed": n_fail, "checks": anchors},
                                                   indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n{len(anchors)} anchor checks, {n_fail} failed")
if n_fail:
    print("STOP: an anchor failed. Nothing is adjusted; no table was written. See results/anchor_checks.json.")
    sys.exit(1)

# ================================================================================================ write
hdr("writing outputs")
T1.to_csv(OUT / "T1_label_by_list.csv", index=False, encoding="utf-8")
T2.to_csv(OUT / "T2_cv_performance.csv", index=False, encoding="utf-8")
T3.to_csv(OUT / "T3_lolo.csv", index=False, encoding="utf-8")
T4.to_csv(OUT / "T4_without_tolaki.csv", index=False, encoding="utf-8")
T5.to_csv(OUT / "T5_cells_by_list.csv", index=False, encoding="utf-8")
T6.to_csv(OUT / "T6_profile.csv", index=False, encoding="utf-8")
T7.to_csv(OUT / "T7_examples_R1-11.csv", index=False, encoding="utf-8")
F1.to_csv(OUT / "F1_input_importance.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------------------------------------- figure
plt.rcParams["font.family"] = "DejaVu Sans"     # has the glottal-stop glyph and the arrows
plt.rcParams["pdf.fonttype"] = 42
# Amendment A3: the journal's template limits a figure to 26 picas (312 pt = 4.33 in) and asks for .jpg/.tiff,
# so the figure is drawn at that width (type stays legible without rescaling) and also saved as TIFF.
FIG_W = 312 / 72
fig, ax = plt.subplots(figsize=(FIG_W, 7.2))
d = F1.iloc[::-1].reset_index(drop=True)         # largest bar at the top
fill = {"written form": "0.30", "meaning": "0.82"}
ax.barh(range(len(d)), d.mean_abs_shap, color=[fill[g] for g in d.group], edgecolor="black", linewidth=0.5, height=0.72)
ax.set_yticks(range(len(d)))
# long labels (amendment A8) are wrapped so that the plotting area keeps its width at 312 pt
ax.set_yticklabels([textwrap.fill(t, 31) for t in d.label], fontsize=6.5, linespacing=0.9)
xmax = d.mean_abs_shap.max()
for i, r in d.iterrows():
    ax.text(r.mean_abs_shap + xmax * 0.02, i, r.direction, va="center", ha="left", fontsize=8, fontweight="bold")
ax.set_xlim(0, xmax * 1.14)
# wrapped: at the journal width a one-line axis label runs off the figure (seen in the first rendering)
ax.set_xlabel(textwrap.fill(LAB["fig1_axis_x"], 44), fontsize=7)
ax.tick_params(axis="x", labelsize=7)
ax.tick_params(axis="y", length=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_ylim(-0.7, len(d) - 0.3)
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=fill["written form"], edgecolor="black", linewidth=0.5),
           plt.Rectangle((0, 0), 1, 1, facecolor=fill["meaning"], edgecolor="black", linewidth=0.5),
           plt.Line2D([], [], linestyle="none"), plt.Line2D([], [], linestyle="none"), plt.Line2D([], [], linestyle="none")]
# The legend is as wide as the plotting area at the journal's width, so anywhere inside the axes it covers bars or
# direction marks (seen in two renderings); it goes below the axis instead.
fig.tight_layout(pad=0.4, rect=[0, 0.105, 1, 1])
fig.legend(handles, [LAB["fig1_legend_form"], LAB["fig1_legend_meaning"], LAB["fig1_legend_up"], LAB["fig1_legend_down"],
                     LAB["fig1_legend_none"]],
           loc="lower center", fontsize=6.5, frameon=False, handlelength=1.3, borderaxespad=0.1, labelspacing=0.35)
fig.savefig(OUT / "F1_input_importance.png", dpi=600)
fig.savefig(OUT / "F1_input_importance.pdf")
fig.savefig(OUT / "F1_input_importance.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)

# ------------------------------------------------------------------------------------ LaTeX fragments
def tex_esc(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def nf(x, k=3):
    return "" if pd.isna(x) else f"{float(x):.{k}f}"


def tabular(colspec, header, body):
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for b in body:
        lines.append(b if b == r"\midrule" else " & ".join(b) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# every number below is read back from the CSV that was just written, never typed
c1, c2, c3, c4, c5, c6 = (pd.read_csv(OUT / n, encoding="utf-8", keep_default_na=False, na_values=[""]) for n in
                          ("T1_label_by_list.csv", "T2_cv_performance.csv", "T3_lolo.csv", "T4_without_tolaki.csv",
                           "T5_cells_by_list.csv", "T6_profile.csv"))
body = []
for r in c1.itertuples():
    if r.list_key == "ALL":
        body.append(r"\midrule")
    body.append([tex_esc(r.list), f"{r.forms:d}", f"{r.coded_n:d}", f"{r.coded_pct:.1f}", f"{r.candidate_n:d}",
                 f"{r.candidate_pct:.1f}"])
(TEX / "T1_label_by_list.tex").write_text(tabular("lrrrrr", ["List", "Forms", "Coded (n)", "Coded (\\%)", "Candidate (n)",
                                                           "Candidate (\\%)"], body), encoding="utf-8")

body = []
for r in c2.itertuples():
    body.append([r.input_set, tex_esc(r.classifier), f"{nf(r.auc_mean)} ({nf(r.auc_sd_seed_means)}; {nf(r.auc_sd_folds)})",
                 nf(r.accuracy_mean), nf(r.accuracy_always_coded), nf(r.f1_candidate_mean), nf(r.precision_candidate_mean),
                 nf(r.recall_candidate_mean), nf(r.f1_coded_mean)])
(TEX / "T2_cv_performance.tex").write_text(tabular(
    "llrrrrrrr", ["Inputs", "Classifier", "AUC (SD seeds; SD folds)", "Accuracy", "Always ``coded''", "F1 candidate",
                  "Precision candidate", "Recall candidate", "F1 coded"], body), encoding="utf-8")

body = []
for iname in ("FM25", "F17"):
    for r in c3[c3.input_set == iname].itertuples():
        if r.list_key == "SUMMARY":
            n_ge = int(c3.loc[(c3.input_set == iname) & (c3.list_key == "SUMMARY"), "n_lists_auc_ge_0.65"].iloc[0])
            body.append([iname, "Mean (SD) over lists", "", "", f"{nf(r.auc)} ({nf(r.auc_sd_over_lists)})", "", "", "",
                         f"{n_ge} of 6 lists $\\geq$ 0.65"])
        else:
            body.append([iname, tex_esc(r.list), f"{r.n:d}", f"{r.n_candidates:d}", nf(r.auc), nf(r.accuracy),
                         nf(r.accuracy_majority_answer), nf(r.accuracy_always_coded), nf(r.f1_candidate)])
    body.append(r"\midrule")
body = body[:-1]
(TEX / "T3_lolo.tex").write_text(tabular(
    "llrrrrrrl", ["Inputs", "Held-out list", "n", "Candidates", "AUC", "Accuracy", "Majority answer", "Always ``coded''",
                  "F1 candidate"], body), encoding="utf-8")

body = [[r.input_set, nf(r.auc_six_lists), nf(r.auc_without_tolaki), nf(r.difference_without_minus_six)]
        for r in c4.itertuples()]
(TEX / "T4_without_tolaki.tex").write_text(tabular(
    "lrrr", ["Inputs", "AUC, six lists", "AUC, without Tolaki", "Difference"], body), encoding="utf-8")

body = []
for r in c5.itertuples():
    if r.list_key == "ALL":
        body.append(r"\midrule")
    body.append([tex_esc(r.list), f"{r.candidate_and_profile:d}", f"{r.candidate_and_no_profile:d}",
                 f"{r.coded_and_profile:d}", f"{r.coded_and_no_profile:d}", nf(r.kappa)])
(TEX / "T5_cells_by_list.tex").write_text(tabular(
    "lrrrrr", ["List", "Candidate, profile", "Candidate, no profile", "Coded, profile", "Coded, no profile", "$\\kappa$"],
    body), encoding="utf-8")

body = []
for r in c6.itertuples():
    eff = f"{nf(r.effect, 2)} [{nf(r.ci_lo, 2)}, {nf(r.ci_hi, 2)}]"
    body.append([tex_esc(r.property), nf(r.mean_candidate, 2), nf(r.mean_coded, 2),
                 f"{r.n_lists_same_sign_as_pooled:d} of 6", "OR" if r.type == "0/1" else "diff.", eff])
(TEX / "T6_profile.tex").write_text(tabular(
    "lrrrlr", ["Property", "Candidates", "Coded", "Lists with same sign", "Effect", "Estimate [95\\% CI]"], body),
    encoding="utf-8")

print("files:", sorted(p.name for p in OUT.iterdir()), sorted(p.name for p in TEX.iterdir()))
print("\nDONE")
