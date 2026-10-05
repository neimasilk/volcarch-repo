"""
E227 script 01 — G1 / G1-bis audit of the P8 manuscript numbers
================================================================
Recomputes, from the raw ABVD CLDF snapshot, every number that the reviewed
manuscript (papers/P8_linguistic_fossils/draft_v0.1_anonymous.tex) states for the
six-language corpus, and compares it with (a) the manuscript and (b) the result
files stored by E022/E027/E028/E029/E041/E042.

Nothing in E022–E042 is modified. Method parameters that cannot be re-derived
(word lists, concept sets, prefix lists) are read out of the original scripts with
`ast`, so that a difference in the output is a difference in the data or in the
logic, not in a retyped constant. Feature extraction, labels, cross-validation and
the statistics are re-implemented here.

Run:  python experiments/E227_p8_g1_blind_rederivation/01_g1_audit.py
"""
import ast
import io
import json
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Windows console is cp1252; the forms contain IPA characters.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP = REPO / "experiments"
ABVD = EXP / "E022_linguistic_subtraction" / "data" / "abvd" / "cldf"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

TARGET = {"27": "Muna", "48": "Bugis", "166": "Makassar",
          "192": "Wolio", "226": "Toraja-Sadan", "674": "Tolaki"}

AUDIT = []  # rows of the claims table


def claim(cid, where, what, paper, recomputed, verdict, note=""):
    AUDIT.append({"id": cid, "tex_line": where, "claim": what, "manuscript": paper,
                  "recomputed": recomputed, "verdict": verdict, "note": note})
    print(f"[{verdict:<16}] {cid} (tex {where}) {what}: paper={paper} | recomputed={recomputed}"
          + (f" | {note}" if note else ""))


def consts_from(path, names):
    """Read top-level literal assignments from a script without executing it."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in names:
                try:
                    out[t.id] = ast.literal_eval(node.value)
                except ValueError:
                    # set("...") call and arithmetic like 1.0 - 0.282
                    out[t.id] = eval(compile(ast.Expression(node.value), str(path), "eval"),
                                     {"set": set})
    missing = set(names) - set(out)
    if missing:
        raise RuntimeError(f"{path}: constants not found: {missing}")
    return out


E022 = consts_from(EXP / "E022_linguistic_subtraction" / "enhanced_subtraction.py",
                   ["PAN_KNOWN", "SANSKRIT_PATTERNS", "ARABIC_PATTERNS", "MALAY_TRADE"])
E027 = consts_from(EXP / "E027_ml_substrate_detection" / "00_prepare_features.py",
                   ["LANG_COVERAGE", "SWADESH_100", "SEMANTIC_DOMAINS", "VOWELS",
                    "AUSTRONESIAN_PREFIXES", "NASAL_CLUSTERS"])
VOWELS = E027["VOWELS"]

# --------------------------------------------------------------------------
# A. corpus, units, labels
# --------------------------------------------------------------------------
print("=" * 78, "\nA. CORPUS, UNITS AND LABELS\n", "=" * 78, sep="")
params = pd.read_csv(ABVD / "parameters.csv", dtype=str, keep_default_na=False)
pname = dict(zip(params.ID, params.Name))
forms_all = pd.read_csv(ABVD / "forms.csv", dtype=str, keep_default_na=False)
f = forms_all[forms_all.Language_ID.isin(TARGET)].copy()
f["language"] = f.Language_ID.map(TARGET)
f["concept"] = f.Parameter_ID.map(pname)
f["raw"] = f.Value.where(f.Value != "", f.Form)


def clean_form(raw):
    s = re.sub(r"\[.*?\]\s*", "", raw.strip())
    return s.strip(" -,;.")


f["form"] = f.raw.map(clean_form)
n_empty = int((f.form == "").sum())
f = f[f.form != ""].copy()
f["has_cog"] = f.Cognacy.str.strip() != ""
f["loan_flag"] = ~f.Loan.str.strip().str.lower().isin(["", "false", "0"])

N = len(f)
claim("A01", "40,86", "total forms, six languages", 1357, N, "MATCH" if N == 1357 else "MISMATCH",
      f"{n_empty} forms empty after cleaning")
per_lang_n = f.groupby("language").size().to_dict()
paper_n = {"Muna": 219, "Bugis": 242, "Makassar": 217, "Wolio": 254, "Toraja-Sadan": 216, "Tolaki": 209}
claim("A02", "86", "forms per language", paper_n, per_lang_n,
      "MATCH" if paper_n == per_lang_n else "MISMATCH")
claim("A03", "89", "number of distinct concepts", 210, int(f.concept.nunique()),
      "MATCH" if f.concept.nunique() == 210 else "MISMATCH")

# G1-bis units: forms are not concepts
per_lang_concepts = f.groupby("language").concept.nunique().to_dict()
multi = {l: int((g.groupby("concept").size() > 1).sum()) for l, g in f.groupby("language")}
dup_forms = int(f.duplicated(["language", "concept", "form"]).sum())
claim("A04", "89", "UNITS: 'each form covers one of 210 concepts' — concepts covered per language",
      "210 implied", per_lang_concepts, "NOTE",
      f"concepts with >1 form per language: {multi}; exact duplicate (lang,concept,form) rows: {dup_forms}. "
      "All rates in the paper are per FORM, not per concept (meaning).")

# label 1: blank ABVD Cognacy field
n_nocog = int((~f.has_cog).sum())
per_lang_nocog = (~f.has_cog).groupby(f.language).sum().astype(int).to_dict()
claim("A05", "93", "forms with ABVD cognacy code ('Austronesian', n=919)", 919, int(f.has_cog.sum()),
      "MATCH" if f.has_cog.sum() == 919 else "MISMATCH")
claim("A06", "40,93", "forms without ABVD cognacy code (the '438 candidates')", 438, n_nocog,
      "MATCH" if n_nocog == 438 else "MISMATCH", f"per language {per_lang_nocog}")
claim("A07", "40", "abstract: '438 candidate forms (26.5% of the corpus)'", "26.5%",
      f"{100 * n_nocog / N:.1f}%", "MISMATCH",
      "438/1357 = 32.3%. 26.5% is the mean of six per-language rates of a DIFFERENT, smaller set (356 forms; see A10).")


def is_loanword(value, loan_set):
    fl = value.lower().strip()
    if len(fl) < 3:
        return False
    for loan in loan_set:
        if len(loan) < 3:
            continue
        if (fl == loan or fl.startswith(loan) or fl.endswith(loan)) and len(loan) / len(fl) >= 0.6:
            return True
    return False


# E022 'enhanced' subtraction is applied to the RAW value (as in the original run)
f["pat_sanskrit"] = f.raw.map(lambda v: is_loanword(v, E022["SANSKRIT_PATTERNS"]))
f["pat_arabic"] = f.raw.map(lambda v: is_loanword(v, E022["ARABIC_PATTERNS"]))
f["pat_malay"] = f.raw.map(lambda v: is_loanword(v, E022["MALAY_TRADE"]))
f["any_tag"] = f.has_cog | f.loan_flag | f.pat_sanskrit | f.pat_arabic | f.pat_malay
f["pan_rescued"] = (~f.any_tag) & f.concept.isin(E022["PAN_KNOWN"])
f["resid_enh"] = (~f.any_tag) & (~f.pan_rescued)

tab1 = f.groupby("language").agg(total=("form", "size"), has_cog=("has_cog", "sum"),
                                 pan_rescued=("pan_rescued", "sum"), residual=("resid_enh", "sum"))
tab1["pct_residual"] = (100 * tab1.residual / tab1.total).round(1)
tab1["pct_cov"] = (100 * tab1.has_cog / tab1.total).round(0).astype(int)
paper_tab1 = {"Muna": (219, 185, 26, 11.9), "Bugis": (242, 180, 49, 20.2), "Toraja-Sadan": (216, 171, 32, 14.8),
              "Wolio": (254, 171, 68, 26.8), "Makassar": (217, 137, 67, 30.9), "Tolaki": (209, 75, 114, 54.5)}
ok = all((int(tab1.loc[l, "total"]), int(tab1.loc[l, "has_cog"]), int(tab1.loc[l, "residual"]),
          float(tab1.loc[l, "pct_residual"])) == v for l, v in paper_tab1.items())
claim("A08", "235-240", "Table 1 rows (total, has cognacy, residual, % residual)", "as printed",
      tab1[["total", "has_cog", "residual", "pct_residual"]].to_dict("index"), "MATCH" if ok else "MISMATCH")
claim("A09", "110,242", "mean of per-language residual rates", "26.5%", f"{tab1.pct_residual.mean():.1f}%",
      "MATCH" if abs(tab1.pct_residual.mean() - 26.5) < 0.06 else "MISMATCH")
n_enh = int(f.resid_enh.sum())
claim("A10", "40,93,404", "LABEL SETS: Table 1 residuals vs the set used as ML label / 'rule-based residuals'",
      "one set implied (438)", f"Table 1 set = {n_enh} forms ({100 * n_enh / N:.1f}% of corpus); ML label set = {n_nocog}",
      "NOT-AS-DESCRIBED",
      "The paper describes one rule-based residual set. There are two: 356 (after loan filter + PAn list) in Table 1, "
      "and 438 (blank cognacy only) everywhere else: ML labels, LOLO table, consensus, clustering.")
claim("A11", "107,247", "forms 'rescued' by the PAn cross-check", 75, int(f.pan_rescued.sum()),
      "MATCH" if f.pan_rescued.sum() == 75 else "MISMATCH",
      f"per language {f.groupby('language').pan_rescued.sum().astype(int).to_dict()}")
claim("A12", "107", "number of PAn/PMP reconstructions in the list", 15, len(E022["PAN_KNOWN"]),
      "MATCH" if len(E022["PAN_KNOWN"]) == 15 else "MISMATCH")
claim("A13", "107", "METHOD: PAn cross-check compares forms with reconstructions", "form-level check implied",
      "concept-level filter", "NOT-AS-DESCRIBED",
      "Code rescues EVERY uncoded form whose MEANING is one of 15 concepts; the form itself is never compared with the "
      "reconstruction. Listed in results/pan_rescued_75.csv for a linguist to inspect.")
claim("A14", "93", "label description: forms 'rescued through PAn cross-checking were labeled Austronesian (n=919)'",
      "rescued forms inside the 919", "0 of 75 rescued forms are inside the 919", "MISMATCH",
      "919 = forms with an ABVD cognacy code only. The 75 'rescued' forms carry the ML label 'candidate'.")
loan_hits = f[(f.pat_sanskrit | f.pat_arabic | f.pat_malay | f.loan_flag)]
claim("A15", "106", "forms removed by the loanword layer (pattern lists + ABVD Loan flag)", "not stated",
      f"{len(loan_hits)} forms tagged; {int((loan_hits.has_cog == False).sum())} of them uncoded", "NOTE",
      "see results/loan_layer_hits.csv — pattern matches are string matches against Indonesian/Sanskrit words")
f[f.pan_rescued][["language", "concept", "form"]].assign(
    pan_form_in_list=lambda d: d.concept.map(E022["PAN_KNOWN"])).sort_values(["concept", "language"]).to_csv(
    OUT / "pan_rescued_75.csv", index=False, encoding="utf-8")
loan_hits[["language", "concept", "form", "has_cog", "loan_flag", "pat_sanskrit", "pat_arabic", "pat_malay"]].to_csv(
    OUT / "loan_layer_hits.csv", index=False, encoding="utf-8")

# Tier-1 concepts (residual in >= 5 languages) under the Table-1 definition
tier = f[f.resid_enh].groupby("concept").language.nunique()
tier1 = sorted(tier[tier >= 5].index)
paper_tier1 = sorted(["if", "to bite", "to tie up, fasten", "to cut, hack", "grass", "to throw", "they", "big"])
tier_forms = f[f.resid_enh].groupby("concept").size()
tier1_forms = sorted(tier_forms[tier_forms >= 5].index)
claim("A16", "111,248", "concepts residual 'in five or more of the six languages' (Table-1 definition)", paper_tier1, tier1,
      "MATCH" if tier1 == paper_tier1 else "MISMATCH",
      f"The paper's eight are concepts with >= 5 residual FORMS ({'same list' if tier1_forms == paper_tier1 else tier1_forms}); "
      "a language with two or three synonyms was counted two or three times. Languages per concept: "
      + str({c: int(tier.get(c, 0)) for c in paper_tier1}))
tier_nocog = f[~f.has_cog].groupby("concept").language.nunique()
claim("A17", "111", "same count under the ML-label definition (blank cognacy)", "8 (paper gives one list)",
      f"{int((tier_nocog >= 5).sum())}: {sorted(tier_nocog[tier_nocog >= 5].index)}", "NOTE")
claim("A18", "114,528", "Tolaki cognacy coverage", "36%", f"{tab1.loc['Tolaki', 'pct_cov']}%",
      "MATCH" if tab1.loc["Tolaki", "pct_cov"] == 36 else "MISMATCH")

# --------------------------------------------------------------------------
# B. features
# --------------------------------------------------------------------------
print("\n", "=" * 78, "\nB. FEATURES\n", "=" * 78, sep="")


def n_vowels(s):
    return sum(1 for c in s.lower() if c in VOWELS)


def n_clusters(s):
    count, consec, inside = 0, 0, False
    for c in s.lower():
        if c not in VOWELS and c.isalpha():
            consec += 1
            if consec == 2 and not inside:
                count += 1
                inside = True
        else:
            consec, inside = 0, False
    return count


def has_redup(s):
    if "-" in s:
        return 1
    fl = s.lower()
    for plen in (2, 3):
        for i in range(len(fl) - plen * 2 + 1):
            if fl[i:i + plen] == fl[i + plen:i + plen * 2]:
                return 1
    return 0


def domain(c):
    for d, cs in E027["SEMANTIC_DOMAINS"].items():
        if c in cs:
            return d
    return "OTHER"


lang_code = {n: i for i, n in enumerate(sorted(TARGET.values()))}
X = pd.DataFrame({
    "form_id": f.ID.values, "language": f.language.values, "concept": f.concept.values, "form": f.form.values,
    "label": f.has_cog.astype(int).values,           # 1 = has ABVD cognacy code
    "form_length": f.form.map(len).values,
    "n_vowels": f.form.map(n_vowels).values,
    "vowel_ratio": f.form.map(lambda s: round(n_vowels(s) / len(s), 4)).values,
    "ends_in_vowel": f.form.map(lambda s: int(s[-1].lower() in VOWELS)).values,
    "initial_char": f.form.map(lambda s: s[0].lower() if s[0].lower() in "mabtkps" else "other").values,
    "has_glottal": f.form.map(lambda s: int("ʔ" in s or "'" in s)).values,
    "has_nasal_cluster": f.form.map(lambda s: int(any(nc in s.lower() for nc in E027["NASAL_CLUSTERS"]))).values,
    "has_reduplication": f.form.map(has_redup).values,
    "n_consonant_clusters": f.form.map(n_clusters).values,
    "has_prefix_like": f.form.map(lambda s: int(s.lower().startswith(tuple(E027["AUSTRONESIAN_PREFIXES"])))).values,
    "semantic_domain": f.concept.map(domain).values,
    "is_core_vocab": f.concept.map(lambda c: int(c in E027["SWADESH_100"])).values,
    "language_id_encoded": f.language.map(lang_code).values,
    "language_cognacy_coverage": f.language.map(lambda l: round(E027["LANG_COVERAGE"][l], 4)).values,
})
stored = pd.read_csv(EXP / "E027_ml_substrate_detection" / "data" / "features_matrix.csv", encoding="utf-8",
                     keep_default_na=False)
m = X.merge(stored, on="form_id", suffixes=("", "_st"))
cmp_cols = ["label", "form", "form_length", "n_vowels", "vowel_ratio", "ends_in_vowel", "initial_char", "has_glottal",
            "has_nasal_cluster", "has_reduplication", "n_consonant_clusters", "has_prefix_like", "semantic_domain",
            "is_core_vocab", "language_id_encoded", "language_cognacy_coverage"]
diffs = {}
for c in cmp_cols:
    a, b = m[c], m[c + "_st"]
    nd = int((np.abs(a.astype(float) - b.astype(float)) > 1e-6).sum()) if a.dtype != object else int((a != b).sum())
    if nd:
        diffs[c] = nd
claim("B01", "—", "stored feature matrix reproducible from raw ABVD (rows matched / columns differing)",
      "n/a", f"{len(m)}/{len(stored)} rows matched; differing columns: {diffs or 'none'}",
      "MATCH" if len(m) == len(stored) == N and not diffs else "MISMATCH")

true_cov = (f.groupby("language").has_cog.mean()).round(3).to_dict()
claim("B02", "141", "feature 'language-level cognacy coverage rate (proportion of forms with ABVD cognate codes)'",
      true_cov, {k: round(v, 3) for k, v in E027["LANG_COVERAGE"].items()}, "NOT-AS-DESCRIBED",
      "Values are hard-coded as 1 - (an early residual rate quoted 'from memory'); e.g. Muna 0.718 vs true coverage 0.845. "
      "Rank order of languages also differs. Affects every model that keeps this feature (full Model B, SHAP figure, "
      "consensus, expansion). The headline 26-feature model drops it.")

ic = pd.get_dummies(X.initial_char, prefix="init").astype(int)
sd = pd.get_dummies(X.semantic_domain, prefix="sem").astype(int)
D = pd.concat([X, ic, sd], axis=1)
PHON = ["form_length", "n_vowels", "vowel_ratio", "ends_in_vowel", "has_glottal", "has_nasal_cluster",
        "has_reduplication", "n_consonant_clusters", "has_prefix_like"]
INIT, SEM = list(ic.columns), list(sd.columns)
FULL = PHON + INIT + ["is_core_vocab"] + SEM + ["language_id_encoded", "language_cognacy_coverage"]
ABL = PHON + INIT + ["is_core_vocab"] + SEM + ["language_id_encoded"]
PURE = PHON + INIT + ["is_core_vocab"] + SEM
claim("B03", "127,132-141,321", "feature counts: full / ablated / pure", "27 / 26 / 25",
      f"{len(FULL)} / {len(ABL)} / {len(PURE)}", "MATCH",
      f"composition = {len(PHON)} form features + {len(INIT)} initial-segment dummies + 1 core-list flag + {len(SEM)} "
      "semantic-domain dummies (+ language id, + coverage). The paper's breakdown '10 + 8 + 2 + 2' sums to 22 and counts "
      "the initial segment twice.")
claim("B04", "41", "abstract: '26 phonological and distributional features — excluding all cognacy data'",
      "phonological + distributional", f"{len(PHON) + len(INIT)} form-based, {1 + len(SEM)} semantic, 1 language identity",
      "NOT-AS-DESCRIBED",
      "'Distributional' is the paper's own name for the cognacy-derived features that Model B EXCLUDES (tex 126). "
      "8 of the 26 are semantic, 1 is the language's identity.")
y = D.label.values
langs = D.language.values


def xgb():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, reg_lambda=1.0,
                         scale_pos_weight=1.0, eval_metric="logloss", random_state=42, verbosity=0)


def cv(cols, make=xgb, scale=False, mask=None, pos_for_f1=1):
    Xa, ya = D[cols].values.astype(float), y
    if mask is not None:
        Xa, ya = Xa[mask], ya[mask]
    seed_auc, seed_f1, seed_acc, fold_auc = [], [], [], []
    for seed in range(10):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 7 + 13)
        a, f1s, acc = [], [], []
        for tr, te in skf.split(Xa, ya):
            Xtr, Xte = Xa[tr], Xa[te]
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            clf = make().fit(Xtr, ya[tr])
            p = clf.predict_proba(Xte)[:, 1]
            pred = clf.predict(Xte)
            a.append(roc_auc_score(ya[te], p))
            f1s.append(f1_score(ya[te], pred, pos_label=pos_for_f1))
            acc.append(accuracy_score(ya[te], pred))
        fold_auc += a
        seed_auc.append(np.mean(a)); seed_f1.append(np.mean(f1s)); seed_acc.append(np.mean(acc))
    return dict(auc=np.mean(seed_auc), auc_sd_seeds=np.std(seed_auc), auc_sd_folds=np.std(fold_auc),
                f1=np.mean(seed_f1), acc=np.mean(seed_acc))


def lolo(cols, make=xgb):
    Xa = D[cols].values.astype(float)
    res = {}
    for L in sorted(set(langs)):
        te = langs == L
        clf = make().fit(Xa[~te], y[~te])
        p = clf.predict_proba(Xa[te])[:, 1]
        pred = clf.predict(Xa[te])
        res[L] = dict(auc=roc_auc_score(y[te], p), f1_aus=f1_score(y[te], pred, pos_label=1),
                      f1_cand=f1_score(y[te], pred, pos_label=0), acc=accuracy_score(y[te], pred),
                      n=int(te.sum()), n_cand=int((y[te] == 0).sum()))
    return res


# --------------------------------------------------------------------------
# C. cross-validation, LOLO, ablation, sensitivity
# --------------------------------------------------------------------------
print("\n", "=" * 78, "\nC. MODELS\n", "=" * 78, sep="")
r_full, r_abl, r_pure = cv(FULL), cv(ABL), cv(PURE)


def close(a, b, tol=0.006):
    return abs(a - b) <= tol


claim("C01", "271,279,331", "Model B (27 feat.) XGBoost CV AUC", "0.760 ± 0.007",
      f"{r_full['auc']:.4f} ± {r_full['auc_sd_seeds']:.4f}", "MATCH" if close(r_full["auc"], 0.760) else "MISMATCH",
      f"± is the SD of 10 seed means; SD over the 50 folds = {r_full['auc_sd_folds']:.3f} (the paper says 50 splits)")
claim("C02", "41,332,497,594", "HEADLINE: ablated (26 feat.) CV AUC", "0.763 ± 0.007",
      f"{r_abl['auc']:.4f} ± {r_abl['auc_sd_seeds']:.4f}", "MATCH" if close(r_abl["auc"], 0.763) else "MISMATCH",
      f"SD over folds = {r_abl['auc_sd_folds']:.3f}")
claim("C03", "333", "'pure phonological' (25 feat.) CV AUC", "0.727 ± 0.007",
      f"{r_pure['auc']:.4f} ± {r_pure['auc_sd_seeds']:.4f}", "MATCH" if close(r_pure["auc"], 0.727) else "MISMATCH",
      "this, not the 26-feature model, is the model with no language information; it still has 8 semantic features")
claim("C04", "271", "Model B XGBoost F1 / accuracy", "0.822 / 0.741", f"{r_full['f1']:.3f} / {r_full['acc']:.3f}",
      "MATCH" if close(r_full["f1"], 0.822) and close(r_full["acc"], 0.741) else "MISMATCH")
f1_cand = cv(FULL, pos_for_f1=0)["f1"]
claim("C05", "265-273", "which class the F1 column scores", "not stated",
      f"F1 of the 'Austronesian' (majority) class; F1 of the candidate class = {f1_cand:.3f}", "NOT-AS-DESCRIBED",
      "In a paper about detecting non-mainstream forms the reader will take F1 = 0.82 as the detection score.")
maj = max(np.mean(y), 1 - np.mean(y))
claim("C06", "271", "accuracy vs the majority-class baseline", "0.741 (no baseline given)",
      f"always-'Austronesian' accuracy = {maj:.3f}", "NOTE", "accuracy 0.741 is ~6 points above doing nothing")
claim("C07", "153", "XGBoost 'scale_pos_weight adjusted for class imbalance'", "adjusted",
      "scale_pos_weight = 1.0 in every script", "MISMATCH", "no class weighting was applied to XGBoost")

rf = cv(FULL, make=lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=42,
                                                 class_weight="balanced", n_jobs=-1))
lr = cv(FULL, make=lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
        scale=True)
claim("C08", "272-273,280", "Model B RF / LR CV AUC", "0.762 / 0.747", f"{rf['auc']:.3f} / {lr['auc']:.3f}",
      "MATCH" if close(rf["auc"], 0.762) and close(lr["auc"], 0.747) else "MISMATCH")

DIST_NOTE = "Model A needs the cognate-set features; reproduced from the stored matrix"
A_cols = FULL + ["max_cognate_set_size", "n_cognate_sets", "concept_residual_rate", "concept_cross_lang_count"]
Dst = D.merge(stored[["form_id", "max_cognate_set_size", "n_cognate_sets", "concept_residual_rate",
                      "concept_cross_lang_count"]], on="form_id")
assert (Dst.form_id.values == D.form_id.values).all()
D[A_cols[-4:]] = Dst[A_cols[-4:]].values
rA = cv(A_cols)
claim("C09", "126,267", "Model A feature count and CV AUC", "31 / 1.000", f"{len(A_cols)} / {rA['auc']:.4f}",
      "MATCH" if len(A_cols) == 31 and rA["auc"] > 0.999 else "MISMATCH", DIST_NOTE)

L_full, L_abl, L_pure = lolo(FULL), lolo(ABL), lolo(PURE)
paper_lolo = {"Tolaki": (0.806, 0.530, 0.364, 209, 134), "Makassar": (0.747, 0.781, 0.659, 217, 80),
              "Bugis": (0.727, 0.799, 0.707, 242, 62), "Wolio": (0.697, 0.798, 0.669, 254, 83),
              "Toraja-Sadan": (0.696, 0.787, 0.690, 216, 45), "Muna": (0.618, 0.465, 0.402, 219, 34)}
ok = all(close(L_full[l]["auc"], v[0]) and close(L_full[l]["f1_aus"], v[1], 0.01) and close(L_full[l]["acc"], v[2], 0.01)
         and L_full[l]["n"] == v[3] and L_full[l]["n_cand"] == v[4] for l, v in paper_lolo.items())
claim("C10", "294-299", "Table 3 LOLO rows (AUC, F1, Acc, N, N_substr)", "as printed",
      {l: (round(v["auc"], 3), round(v["f1_aus"], 3), round(v["acc"], 3), v["n"], v["n_cand"]) for l, v in L_full.items()},
      "MATCH" if ok else "MISMATCH",
      "N_substr = forms with blank cognacy (438 set), not the Table 1 residuals")
aucs = [v["auc"] for v in L_full.values()]
claim("C11", "301", "LOLO mean AUC ± SD (27 feat.)", "0.715 ± 0.057", f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
      "MATCH" if close(np.mean(aucs), 0.715) else "MISMATCH")
claim("C12", "294,299", "LOLO accuracy for Tolaki and Muna", "0.364 / 0.402",
      f"{L_full['Tolaki']['acc']:.3f} / {L_full['Muna']['acc']:.3f}; majority baselines "
      f"{max(134, 75) / 209:.3f} / {185 / 219:.3f}", "NOTE",
      "Both far BELOW the majority baseline: with the language features the held-out language is thresholded wrongly. "
      "AUC (ranking) is unaffected; the text does not comment on these two accuracies.")
a2, a3 = [v["auc"] for v in L_abl.values()], [v["auc"] for v in L_pure.values()]
claim("C13", "332,338,498", "ablated LOLO mean AUC / languages >= 0.65", "0.722 / 6 of 6",
      f"{np.mean(a2):.3f} / {sum(a >= 0.65 for a in a2)} of 6", "MATCH" if close(np.mean(a2), 0.722) else "MISMATCH",
      "per language " + str({k: round(v["auc"], 3) for k, v in L_abl.items()}))
claim("C14", "333", "pure LOLO mean AUC / languages >= 0.65", "0.701 / 5 of 6",
      f"{np.mean(a3):.3f} / {sum(a >= 0.65 for a in a3)} of 6", "MATCH" if close(np.mean(a3), 0.701) else "MISMATCH")
claim("C15", "339", "Muna LOLO AUC: full -> ablated", "0.618 -> 0.679",
      f"{L_full['Muna']['auc']:.3f} -> {L_abl['Muna']['auc']:.3f}",
      "MATCH" if close(L_full["Muna"]["auc"], 0.618) and close(L_abl["Muna"]["auc"], 0.679) else "MISMATCH")
no_t = cv(FULL, mask=(langs != "Tolaki"))
claim("C16", "313,532", "CV AUC without Tolaki (27 feat.)", "0.698 (Δ = −0.062)",
      f"{no_t['auc']:.4f} (Δ = {no_t['auc'] - r_full['auc']:+.3f})", "MATCH" if close(no_t["auc"], 0.698) else "MISMATCH")
no_t_abl = cv(ABL, mask=(langs != "Tolaki"))
no_t_pure = cv(PURE, mask=(langs != "Tolaki"))
claim("C17", "313", "same sensitivity for the HEADLINE (26 feat.) and the pure (25 feat.) model", "not reported",
      f"26 feat.: {no_t_abl['auc']:.3f} (Δ = {no_t_abl['auc'] - r_abl['auc']:+.3f}); "
      f"25 feat.: {no_t_pure['auc']:.3f} (Δ = {no_t_pure['auc'] - r_pure['auc']:+.3f})", "NOTE",
      "the paper reports the Tolaki sensitivity for the 27-feature model only")
claim("C18", "313", "'remained above the 0.65 threshold'", "0.65 presented as a threshold", "no source", "UNSUPPORTED",
      "0.65 is the project's own go/no-go line (E027 PLAN), not a convention; R2 asks what range counts as reliable")

# --------------------------------------------------------------------------
# D. direction of every 'fingerprint' property (G1-bis item 2)
# --------------------------------------------------------------------------
print("\n", "=" * 78, "\nD. DIRECTION OF THE FINGERPRINT PROPERTIES\n", "=" * 78, sep="")
D["cand"] = 1 - D.label
D["sem_ACTION_"] = D.sem_ACTION


def mh_or(col):
    """Mantel-Haenszel odds ratio of candidate status for a binary property, stratified by language (RBG variance)."""
    R = S = P_R = P_S_Q_R = Q_S = 0.0
    for _, g in D.groupby("language"):
        a = ((g[col] == 1) & (g.cand == 1)).sum(); b = ((g[col] == 1) & (g.cand == 0)).sum()
        c = ((g[col] == 0) & (g.cand == 1)).sum(); d = ((g[col] == 0) & (g.cand == 0)).sum()
        n = a + b + c + d
        if n == 0:
            continue
        r, s = a * d / n, b * c / n
        p, q = (a + d) / n, (b + c) / n
        R += r; S += s; P_R += p * r; P_S_Q_R += p * s + q * r; Q_S += q * s
    if R == 0 or S == 0:
        return float("nan"), float("nan"), float("nan")
    orr = R / S
    var = P_R / (2 * R * R) + P_S_Q_R / (2 * R * S) + Q_S / (2 * S * S)
    lo, hi = np.exp(np.log(orr) - 1.96 * np.sqrt(var)), np.exp(np.log(orr) + 1.96 * np.sqrt(var))
    return orr, lo, hi


rows = []
for col in ["form_length", "n_vowels", "n_consonant_clusters", "has_glottal", "has_prefix_like", "has_nasal_cluster",
            "has_reduplication", "ends_in_vowel", "sem_ACTION", "is_core_vocab"]:
    r = {"property": col, "mean_candidate": D[D.cand == 1][col].mean(), "mean_coded": D[D.cand == 0][col].mean()}
    for L, g in D.groupby("language"):
        r[f"diff_{L}"] = g[g.cand == 1][col].mean() - g[g.cand == 0][col].mean()
    r["n_langs_positive"] = sum(r[f"diff_{L}"] > 0 for L in sorted(set(langs)))
    if set(D[col].unique()) <= {0, 1}:
        r["MH_OR"], r["MH_lo"], r["MH_hi"] = mh_or(col)
    rows.append(r)
dir_tab = pd.DataFrame(rows)
dir_tab.to_csv(OUT / "direction_by_label.csv", index=False, encoding="utf-8")
print(dir_tab.round(3).to_string())


def drow(col):
    return dir_tab[dir_tab.property == col].iloc[0]


p_ = drow("has_prefix_like")
claim("D01", "41,385,448,497", "'fewer canonical Austronesian prefixes' among non-mainstream forms", "fewer",
      f"prefix-like onset: candidates {100 * p_.mean_candidate:.1f}% vs coded {100 * p_.mean_coded:.1f}%; "
      f"within-language OR = {p_.MH_OR:.2f} [{p_.MH_lo:.2f}, {p_.MH_hi:.2f}]; higher in {p_.n_langs_positive} of 6 languages",
      "DIRECTION" if p_.mean_candidate > p_.mean_coded else "MATCH",
      "The stated direction is the reverse of the data. Mean |SHAP| of this feature is rank 23 of 27 (E02).")
g_ = drow("has_glottal")
claim("D02", "41,382,563", "'higher rates of glottal stops' among non-mainstream forms", "higher",
      f"candidates {100 * g_.mean_candidate:.1f}% vs coded {100 * g_.mean_coded:.1f}%; within-language OR = "
      f"{g_.MH_OR:.2f} [{g_.MH_lo:.2f}, {g_.MH_hi:.2f}]; higher in {g_.n_langs_positive} of 6 languages",
      "MATCH" if g_.MH_lo > 1 else "WEAKER", "per-language detail in results/glottal_by_language.csv")
c_ = drow("n_consonant_clusters")
claim("D03", "41,381", "'more consonant clusters'", "more",
      f"mean clusters: candidates {c_.mean_candidate:.2f} vs coded {c_.mean_coded:.2f}; higher in {c_.n_langs_positive} of 6",
      "MATCH" if c_.mean_candidate > c_.mean_coded else "DIRECTION",
      "counted on orthography: 'ng', 'ngk', 'mb' etc. count as clusters (see E05)")
def n_nuclei(s):
    cnt, inside = 0, False
    for ch in s.lower():
        if ch in VOWELS:
            if not inside:
                cnt += 1
            inside = True
        else:
            inside = False
    return cnt


D["n_nuclei"] = D.form.map(n_nuclei)
nuc_c, nuc_a = D[D.cand == 1].n_nuclei.mean(), D[D.cand == 0].n_nuclei.mean()
l_ = drow("form_length")
v_ = drow("n_vowels")
claim("D04", "41,379", "'longer forms': 2.57 vs 2.29 syllables", "2.57 vs 2.29",
      f"vowel nuclei: {nuc_c:.2f} vs {nuc_a:.2f}; vowel letters {v_.mean_candidate:.2f} vs {v_.mean_coded:.2f}; "
      f"characters {l_.mean_candidate:.2f} vs {l_.mean_coded:.2f}; longer in {l_.n_langs_positive} of 6",
      "MATCH" if close(nuc_c, 2.57, 0.006) and close(nuc_a, 2.29, 0.006) else "MISMATCH",
      "'syllable' = maximal run of vowel letters, so every vowel sequence (ae, oo, ia) counts as ONE syllable; "
      "in these languages most such sequences are two syllables, so both figures undercount")
a_ = drow("sem_ACTION")
claim("D05", "380,504", "action concepts over-represented among candidates", "over-represented",
      f"candidates {100 * a_.mean_candidate:.1f}% vs coded {100 * a_.mean_coded:.1f}%; OR = {a_.MH_OR:.2f} "
      f"[{a_.MH_lo:.2f}, {a_.MH_hi:.2f}]", "MATCH" if a_.MH_lo > 1 else "WEAKER")

gl = D.groupby("language").agg(n=("form", "size"), glottal=("has_glottal", "sum"),
                               apostrophe=("form", lambda s: int(s.str.contains("'").sum())),
                               ipa_glottal=("form", lambda s: int(s.str.contains("ʔ").sum())),
                               final_q=("form", lambda s: int(s.str.lower().str.endswith("q").sum())),
                               final_k=("form", lambda s: int(s.str.lower().str.endswith("k").sum())),
                               any_q=("form", lambda s: int(s.str.lower().str.contains("q").sum())))
gl["pct_glottal"] = (100 * gl.glottal / gl.n).round(1)
for L, g in D.groupby("language"):
    gl.loc[L, "glottal_rate_candidate"] = round(100 * g[g.cand == 1].has_glottal.mean(), 1)
    gl.loc[L, "glottal_rate_coded"] = round(100 * g[g.cand == 0].has_glottal.mean(), 1)
gl.to_csv(OUT / "glottal_by_language.csv", encoding="utf-8")
print(gl.to_string())
claim("D06", "135,382", "UNITS: how the six sources write the glottal stop", "'ʔ or orthographic apostrophe'",
      gl[["apostrophe", "ipa_glottal", "final_q", "final_k", "pct_glottal"]].to_dict("index"), "NOTE",
      "the marker rate differs several-fold between sources; reviewer 1's point 9 is about exactly this")

# --------------------------------------------------------------------------
# E. SHAP
# --------------------------------------------------------------------------
print("\n", "=" * 78, "\nE. SHAP\n", "=" * 78, sep="")
import shap  # noqa: E402


def shap_table(cols):
    Xa = D[cols].values.astype(float)
    clf = xgb().fit(Xa, y)
    sv = shap.TreeExplainer(clf).shap_values(Xa)
    t = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    # sign: does a HIGH feature value push the output toward class 1 (= has cognacy code)?
    t["corr_value_shap"] = [np.corrcoef(Xa[:, i], sv[:, i])[0, 1] if Xa[:, i].std() > 0 else np.nan
                            for i in range(len(cols))]
    t = t.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    t["rank"] = t.index + 1
    return t


S_full, S_abl, S_pure = shap_table(FULL), shap_table(ABL), shap_table(PURE)
S_full.to_csv(OUT / "shap_full27.csv", index=False); S_abl.to_csv(OUT / "shap_ablated26.csv", index=False)
S_pure.to_csv(OUT / "shap_pure25.csv", index=False)
top5 = [(r.feature, round(r.mean_abs_shap, 3)) for r in S_full.head(5).itertuples()]
paper5 = [("language_cognacy_coverage", 0.559), ("form_length", 0.378), ("sem_ACTION", 0.230),
          ("n_consonant_clusters", 0.190), ("has_glottal", 0.188)]
claim("E01", "378-382", "top-5 mean |SHAP| (27-feature model)", paper5, top5,
      "MATCH" if [a for a, _ in top5] == [a for a, _ in paper5] and all(close(a[1], b[1], 0.01) for a, b in zip(top5, paper5))
      else "MISMATCH")
pr = S_full[S_full.feature == "has_prefix_like"].iloc[0]
claim("E02", "385,497", "rank of the prefix feature in SHAP", "part of the fingerprint",
      f"rank {pr['rank']} of {len(FULL)} (mean |SHAP| {pr.mean_abs_shap:.3f})", "NOTE")
claim("E03", "366-383", "SHAP analysis is of the headline model", "implied",
      "SHAP figure and values are from the 27-feature model; headline AUC is from the 26-feature model",
      "NOT-AS-DESCRIBED", "top-5 of the 26-feature model: " + str([(r.feature, round(r.mean_abs_shap, 3))
                                                                   for r in S_abl.head(6).itertuples()]))
fl = S_full[S_full.feature == "form_length"].iloc[0]
claim("E04", "371", "Figure 1 caption: 'positive SHAP values push toward substrate classification'",
      "positive = substrate",
      f"model output = P(has cognacy code); corr(form_length, its SHAP) = {fl.corr_value_shap:+.2f}", "DIRECTION",
      "Positive SHAP pushes toward 'Austronesian'. The figure's own title says so ('Predicting Austronesian (label=1)'). "
      "The verbal conclusions (longer = candidate) are right; the caption is inverted.")
other = S_full[S_full.feature == "sem_OTHER"].iloc[0]
claim("E05", "375-383", "features ranked 6th-10th are not discussed", "five listed",
      f"sem_OTHER rank {other['rank']} ({other.mean_abs_shap:.3f}), then n_vowels, vowel_ratio, language id, core-list flag",
      "NOTE", f"{int((D.semantic_domain == 'OTHER').sum())} of {N} forms ({100 * (D.semantic_domain == 'OTHER').mean():.0f}%) "
              "fall in the residual domain 'OTHER'")
dom_src = "hand-written concept sets inside 00_prepare_features.py (SEMANTIC_DOMAINS, SWADESH_100)"
claim("E06", "90,139", "source of the semantic domains and of the 'Swadesh-100' subset (R2 asks)",
      "Swadesh (1955) cited", dom_src, "UNSUPPORTED",
      f"The 'Swadesh-100' set in the code has {len(E027['SWADESH_100'])} concepts, "
      f"{int(D.is_core_vocab.mean() * 100)}% of forms are flagged core. Neither list is taken from Concepticon/WOLD.")

# --------------------------------------------------------------------------
# F. cross-method 'consensus' as published (in-sample)
# --------------------------------------------------------------------------
print("\n", "=" * 78, "\nF. CONSENSUS (as published)\n", "=" * 78, sep="")
clf_full = xgb().fit(D[FULL].values.astype(float), y)
D["p_sub_insample"] = clf_full.predict_proba(D[FULL].values.astype(float))[:, 0]
D["ml_sub"] = (D.p_sub_insample >= 0.5).astype(int)
quad = np.where((D.cand == 1) & (D.ml_sub == 1), "CS", np.where((D.cand == 0) & (D.ml_sub == 0), "CA",
                np.where((D.cand == 1) & (D.ml_sub == 0), "RO", "MO")))
D["quadrant_insample"] = quad
qc = Counter(quad)
kap = cohen_kappa_score(D.cand, D.ml_sub)
claim("F01", "402", "quadrants CS / CA / RO / MO", "266 / 878 / 172 / 41",
      f"{qc['CS']} / {qc['CA']} / {qc['RO']} / {qc['MO']}",
      "MATCH" if (qc["CS"], qc["CA"], qc["RO"], qc["MO"]) == (266, 878, 172, 41) else "MISMATCH")
claim("F02", "43,401,597", "Cohen's kappa between the two methods", 0.611, round(kap, 3),
      "MATCH" if close(kap, 0.611) else "MISMATCH")
ins_auc = roc_auc_score(y, 1 - D.p_sub_insample)
claim("F03", "178-187,404,546", "INDEPENDENCE of the two 'methods'", "'confirmed independently by ML'",
      f"ML probabilities are IN-SAMPLE: the model was fitted to the rule label on these same 1,357 forms "
      f"(in-sample AUC {ins_auc:.3f} vs cross-validated {r_full['auc']:.3f})", "NOT-AS-DESCRIBED",
      "kappa = 0.61 measures how well the model memorised its own training label, not agreement between two methods. "
      "The out-of-fold version is pre-registered in E228.")
claim("F04", "404", "'266 CS forms = 60.7% of all rule-based residuals'", "60.7%", f"{100 * qc['CS'] / n_nocog:.1f}% of {n_nocog}",
      "MATCH" if close(100 * qc["CS"] / n_nocog, 60.7, 0.06) else "MISMATCH",
      "denominator is the 438 blank-cognacy set, not Table 1's 356")
cs = D[D.quadrant_insample == "CS"]
claim("F05", "405", "CS forms per language", {"Tolaki": 121, "Makassar": 48, "Bugis": 36, "Wolio": 36, "Toraja-Sadan": 15, "Muna": 10},
      cs.language.value_counts().to_dict(),
      "MATCH" if cs.language.value_counts().to_dict() == {"Tolaki": 121, "Makassar": 48, "Bugis": 36, "Wolio": 36,
                                                          "Toraja-Sadan": 15, "Muna": 10} else "MISMATCH")
claim("F06", "405 vs 240", "INTERNAL: Tolaki CS forms (121) exceed Tolaki residuals in Table 1 (114)", "121 > 114",
      f"{int((cs.language == 'Tolaki').sum())} CS vs {int(tab1.loc['Tolaki', 'residual'])} Table-1 residuals", "MISMATCH",
      "a 'consensus of both methods' cannot be larger than one of them; consequence of the two label sets (A10)")
claim("F07", "407", "CS by semantic domain", "ACTION 117, GRAMMAR 40, QUALITY 38, NATURE 20, NUMBER 20, BODY 12",
      cs.semantic_domain.value_counts().to_dict(), "MATCH" if cs.semantic_domain.value_counts().get("ACTION") == 117 else "MISMATCH",
      "OTHER (19) is omitted from the sentence; the six listed sum to 247 of 266")
base_action = D.sem_ACTION.mean()
claim("F08", "407,504", "'predominance of action verbs' (44.0% of CS)", "44.0%",
      f"CS {100 * cs.sem_ACTION.mean():.1f}% vs whole corpus {100 * base_action:.1f}%", "NOTE",
      "no base rate is given in the paper; sem_ACTION is also an input feature of the model that defines CS")
ro, mo, ca = D[D.quadrant_insample == "RO"], D[D.quadrant_insample == "MO"], D[D.quadrant_insample == "CA"]
claim("F09", "411", "RO vs CS: length / glottal / clusters", "5.35 vs 6.58 / 10.5% vs 32.0% / 0.35 vs 0.56",
      f"{ro.form_length.mean():.2f} vs {cs.form_length.mean():.2f} / {100 * ro.has_glottal.mean():.1f}% vs "
      f"{100 * cs.has_glottal.mean():.1f}% / {ro.n_consonant_clusters.mean():.2f} vs {cs.n_consonant_clusters.mean():.2f}",
      "MATCH" if close(ro.form_length.mean(), 5.35, 0.02) and close(cs.form_length.mean(), 6.58, 0.02) else "MISMATCH",
      "these are the model's own input features: the split is by construction, not a finding")
claim("F10", "414", "MO: length / glottal / clusters (vs CA)", "6.63 / 26.8% / 0.73 vs 0.30",
      f"{mo.form_length.mean():.2f} / {100 * mo.has_glottal.mean():.1f}% / {mo.n_consonant_clusters.mean():.2f} vs "
      f"{ca.n_consonant_clusters.mean():.2f}", "MATCH" if close(mo.form_length.mean(), 6.63, 0.02) else "MISMATCH")
claim("F11", "563", "'substrate candidates 32.0% glottal vs consensus Austronesian forms 10.5%'", "CA = 10.5%",
      f"CA = {100 * ca.has_glottal.mean():.1f}%; 10.5% is the RO value", "MISMATCH", "wrong group named in §4.5")
cs_conc = cs.groupby("concept").language.nunique()
four = sorted(cs_conc[cs_conc >= 4].index)
claim("F12", "417", "concepts that are CS in >= 4 languages", ["Fifty", "One Hundred", "Twenty", "to hit", "to stand"], four,
      "MATCH" if four == ["Fifty", "One Hundred", "Twenty", "to hit", "to stand"] else "MISMATCH")
in_pan = [c for c in four if c in E022["PAN_KNOWN"]]
claim("F13", "417 vs 107", "INTERNAL: are those concepts on the paper's own PAn list?", "not mentioned",
      f"{in_pan} are on the 15-concept PAn list, i.e. the rule-based method had classed them as inherited", "MISMATCH",
      "R2 asks why these five appear. Answer: the 'consensus' was computed on the label that ignores the PAn step; "
      "the other three are decimal compounds (tex 440-445).")
rank = D[D.cand == 1].sort_values("p_sub_insample", ascending=False).head(50)
t50 = (rank.semantic_domain.value_counts() / 50 * 100).round(0).to_dict()
claim("F14", "380", "top-50 candidates: 46% action, 26% quality, 16% grammatical", "46 / 26 / 16",
      f"{t50.get('ACTION', 0):.0f} / {t50.get('QUALITY', 0):.0f} / {t50.get('GRAMMAR', 0):.0f}; languages in top-50: "
      f"{rank.language.value_counts().to_dict()}",
      "MATCH" if (t50.get("ACTION"), t50.get("QUALITY"), t50.get("GRAMMAR")) == (46.0, 26.0, 16.0) else "CHECK",
      "ranking is by in-sample probability of the 27-feature model")

# --------------------------------------------------------------------------
# G. clustering as published
# --------------------------------------------------------------------------
print("\n", "=" * 78, "\nG. CLUSTERING (as published)\n", "=" * 78, sep="")


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca_ in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca_ != cb)))
        prev = cur
    return prev[-1]


def nlev(a, b):
    return 0.0 if not a and not b else lev(a, b) / max(len(a), len(b))


cs_sorted = cs.sort_values("p_sub_insample", ascending=False)
cforms = [s.lower().strip() for s in cs_sorted.form]
n = len(cforms)
DM = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        DM[i, j] = DM[j, i] = nlev(cforms[i], cforms[j])
Z = linkage(squareform(DM), method="ward")
sil = {k: silhouette_score(DM, fcluster(Z, t=k, criterion="maxclust"), metric="precomputed") for k in range(5, 31)}
best_k = max(sil, key=sil.get)
# same analysis in the row order of the stored E028 file (what E029 actually read)
st_cs = pd.read_csv(EXP / "E028_substrate_consensus" / "results" / "consensus_substrates.csv", encoding="utf-8",
                    keep_default_na=False)
same_set = sorted(zip(st_cs.language, st_cs.concept, st_cs.form)) == sorted(zip(cs.language, cs.concept, cs.form))
sf = [x.lower().strip() for x in st_cs.form]
DM2 = np.zeros((len(sf), len(sf)))
for i in range(len(sf)):
    for j in range(i + 1, len(sf)):
        DM2[i, j] = DM2[j, i] = nlev(sf[i], sf[j])
Z2 = linkage(squareform(DM2), method="ward")
sil2 = {k: silhouette_score(DM2, fcluster(Z2, t=k, criterion="maxclust"), metric="precomputed") for k in range(5, 31)}
bk2 = max(sil2, key=sil2.get)
rng = np.random.default_rng(0)
perm_best = []
for _ in range(20):
    o = rng.permutation(n)
    Zp = linkage(squareform(DM[np.ix_(o, o)]), method="ward")
    sp = {k: silhouette_score(DM[np.ix_(o, o)], fcluster(Zp, t=k, criterion="maxclust"), metric="precomputed")
          for k in range(5, 31)}
    perm_best.append(max(sp.values()))
claim("G01", "425", "Ward clustering: best k and silhouette", "k = 30, 0.114",
      f"stored row order: k = {bk2}, {sil2[bk2]:.3f} (same 266 forms: {same_set}); other row orders of the same forms: "
      f"best silhouette {min(perm_best):.3f}–{max(perm_best):.3f} over 20 shuffles",
      "MATCH" if bk2 == 30 and close(sil2[bk2], 0.114) else "MISMATCH",
      "the value depends on the order of the rows (ties in edit distance); the conclusion 'near zero' does not")
sil_ext = {k: silhouette_score(DM, fcluster(Z, t=k, criterion="maxclust"), metric="precomputed") for k in (40, 60, 80, 100, 133)}
claim("G02", "207,425", "'optimal k = 30'", "optimal", f"k = 30 is the top of the searched range; beyond it: "
      + ", ".join(f"k={k}: {v:.3f}" for k, v in sil_ext.items()), "NOT-AS-DESCRIBED",
      "silhouette is still rising at the boundary, so 30 is not an optimum (R2 asks why 5–30). Ward linkage also "
      "assumes Euclidean distances; normalised edit distance is not Euclidean.")
db = DBSCAN(eps=0.3, min_samples=3, metric="precomputed").fit_predict(DM)
ncl, nnoise = len(set(db)) - (1 if -1 in db else 0), int((db == -1).sum())
claim("G03", "426", "DBSCAN eps=0.3, min_samples=3: clusters / forms clustered / noise", "4 / 14 / 94.7%",
      f"{ncl} / {n - nnoise} / {100 * nnoise / n:.1f}%", "MATCH" if (ncl, n - nnoise) == (4, 14) else "MISMATCH",
      "'best configuration' = highest silhouette over the 5% of forms that were not noise")
conc_forms = defaultdict(dict)
for r in cs_sorted.itertuples():
    conc_forms[r.concept][r.language] = r.form.lower().strip()   # last form per language wins, as in E029
multi3 = {c: d for c, d in conc_forms.items() if len(d) >= 3}
obs = {c: float(np.mean([nlev(a, b) for i, a in enumerate(list(d.values())) for b in list(d.values())[i + 1:]]))
       for c, d in multi3.items()}
obs_mean = float(np.mean(list(obs.values())))
claim("G04", "213,430", "concepts that are CS in >= 3 languages / mean cross-language distance", "20 / 0.769",
      f"{len(multi3)} / {obs_mean:.3f}", "MATCH" if len(multi3) == 20 and close(obs_mean, 0.769, 0.01) else "MISMATCH")
claim("G05", "442", "distances for 'Fifty' and 'Twenty'", "0.204 / 0.292",
      f"{obs.get('Fifty', float('nan')):.3f} / {obs.get('Twenty', float('nan')):.3f}",
      "MATCH" if close(obs.get("Fifty", 9), 0.204, 0.01) and close(obs.get("Twenty", 9), 0.292, 0.01) else "MISMATCH")
# null as designed in E029: one random concept, 3 random languages, first-listed form of each
by_c = defaultdict(lambda: defaultdict(list))
for r in f.itertuples():
    by_c[r.concept][r.language].append(r.raw)
pool = [(c, [l for l in TARGET.values() if d.get(l)], d) for c, d in by_c.items()]
pool = [p for p in pool if len(p[1]) >= 3]
ps, null_means, null_sds = [], [], []
for seed in range(20):
    rng = random.Random(seed)
    nd = []
    for _ in range(1000):
        c, av, d = rng.choice(pool)
        sel = rng.sample(av, 3)
        fs = [d[l][0].lower().strip() for l in sel]
        nd.append(np.mean([nlev(fs[0], fs[1]), nlev(fs[0], fs[2]), nlev(fs[1], fs[2])]))
    ps.append(np.mean(np.array(nd) <= obs_mean)); null_means.append(np.mean(nd)); null_sds.append(np.std(nd))
claim("G06", "44,430-431", "null mean ± SD and one-tailed p", "0.677 ± 0.226, p = 0.569",
      f"{np.mean(null_means):.3f} ± {np.mean(null_sds):.3f}, p = {np.mean(ps):.3f} (range {min(ps):.3f}–{max(ps):.3f} over 20 seeds)",
      "MATCH" if close(np.mean(ps), 0.569, 0.04) else "MISMATCH")
coded_share = float(np.mean([f[(f.concept == c)].has_cog.mean() for c, _, _ in pool]))
claim("G07", "213-214,431", "DESIGN of the test behind p = 0.569", "'substrate forms vs random vocabulary'",
      f"null = same-concept forms in 3 languages, {100 * coded_share:.0f}% of which are cognate-coded; statistic = mean of 20 "
      "concepts compared with a distribution of single-concept draws", "NOT-AS-DESCRIBED",
      "Two problems: (1) the null is ordinary same-meaning vocabulary, which is mostly cognate, so the test asks whether "
      "candidates resemble each other MORE than known cognates do; (2) a mean of 20 is compared with the spread of single "
      "draws. The conclusion (no detectable shared forms) may well survive, but p = 0.569 does not measure it. "
      "A concept-shuffle permutation test is pre-registered in E228.")
num_in_20 = sorted(c for c in multi3 if c in E027["SEMANTIC_DOMAINS"]["NUMBER"])
pan_in_20 = sorted(c for c in multi3 if c in E022["PAN_KNOWN"])
claim("G08", "430,440", "composition of the 20 concepts", "two numeral compounds discussed",
      f"numerals: {num_in_20}; on the paper's PAn list: {pan_in_20}", "NOTE")

# --------------------------------------------------------------------------
# H. expansion table (from the stored per-language file)
# --------------------------------------------------------------------------
print("\n", "=" * 78, "\nH. EXPANSION (stored results)\n", "=" * 78, sep="")
ex = pd.read_csv(EXP / "E027_ml_substrate_detection" / "results" / "expansion_summary.csv")
gm = ex.groupby("group").agg(n=("language", "size"), rule=("rule_residual_rate", "mean"),
                             ml=("ml_substrate_rate", "mean"), auc=("auc", "mean"), p=("mean_p_substrate", "mean"))
print(gm.round(4).to_string())
paper_gm = {"Original": (6, 28.0, 23.1, 0.890, 0.326), "Sulawesi": (8, 49.9, 62.4, 0.685, 0.606),
            "W.Indonesian": (6, 28.4, 35.3, 0.634, 0.393), "E.Indonesian": (2, 34.9, 51.9, 0.661, 0.520)}
ok = all(int(gm.loc[g, "n"]) == v[0] and close(100 * gm.loc[g, "rule"], v[1], 0.06) and close(100 * gm.loc[g, "ml"], v[2], 0.06)
         and close(gm.loc[g, "auc"], v[3], 0.001) and close(gm.loc[g, "p"], v[4], 0.001) for g, v in paper_gm.items())
claim("H01", "467-470", "Table 5 group means", "as printed", gm.round(3).to_dict("index"), "MATCH" if ok else "MISMATCH")
orig = ex[ex.group == "Original"].set_index("language")
true_rule = {l: round(per_lang_nocog[l] / per_lang_n[l], 3) for l in per_lang_n}
claim("H02", "467", "'Original 6' mean rule-based rate", "28.0%",
      f"stored per-language rates {orig.rule_residual_rate.to_dict()} vs rates of the label actually used {true_rule} "
      f"(mean {100 * np.mean(list(true_rule.values())):.1f}%)", "MISMATCH",
      "four of the six stored rates are hard-coded numbers that match neither Table 1 nor the label; the paper thus "
      "gives 26.5%, 28.0% and 32.3% for the same six languages")
claim("H03", "467,482", "'Original 6' AUC 0.890 compared with expansion AUC 0.663", "0.890 vs 0.663",
      "0.890 is in-sample (model scored on its own training forms); the like-for-like figure is the LOLO mean "
      f"{np.mean(aucs):.3f}", "NOT-AS-DESCRIBED")
exp_only = ex[ex.group != "Original"]
claim("H04", "482", "mean AUC over the 16 expansion languages", 0.663, round(exp_only.auc.mean(), 3),
      "MATCH" if close(exp_only.auc.mean(), 0.663, 0.001) else "MISMATCH",
      f"range {exp_only.auc.min():.3f}–{exp_only.auc.max():.3f}; {int((exp_only.auc < 0.60).sum())} of 16 below 0.60, "
      f"{int((exp_only.auc < 0.65).sum())} of 16 below the paper's own 0.65 line")
sul, wes = ex[ex.group == "Sulawesi"].mean_p_substrate, ex[ex.group == "W.Indonesian"].mean_p_substrate
u = mannwhitneyu(sul, wes, alternative="two-sided")
claim("H05", "45,483", "'Sulawesi significantly higher' (0.606 vs 0.393)", "significantly",
      f"no test in the code (a fixed 0.03 cut-off); Mann-Whitney on the stored values: U = {u.statistic:.0f}, p = {u.pvalue:.3f} (8 vs 6 languages)",
      "UNSUPPORTED")
rho = spearmanr(1 - exp_only.rule_residual_rate, exp_only.mean_p_substrate)
claim("H06", "456,483", "CIRCULARITY: language-level inputs given to the model for the 16 new languages", "not described",
      f"coverage feature = each new language's own share of coded forms; language id = 3 for all (3 = Tolaki's code). "
      f"Spearman rho(coverage, mean P_sub) across the 16 = {rho.statistic:+.2f} (p = {rho.pvalue:.4f})",
      "NOT-AS-DESCRIBED",
      "Established from the code (03_expansion_validation.py, lines 492-505), not from the correlation, which would "
      "also arise from a real signal. The model that scored the new languages was given each language's own rate of "
      "coded forms and the identity code of the language with the most candidates. The 'geographic pattern' has to be "
      "recomputed with the 25-feature model, which has no language-level input (E228, pre-registered).")
bm = ex[ex.language == "Bol.Mongondow"].iloc[0]; go = ex[ex.language == "Gorontalo"].iloc[0]
claim("H07", "468,486", "Bolaang Mongondow 9.2% / Gorontalo 84.2% predicted", "9.2% / 84.2%",
      f"{100 * bm.ml_substrate_rate:.1f}% (AUC {bm.auc:.3f}) / {100 * go.ml_substrate_rate:.1f}% (AUC {go.auc:.3f})",
      "MATCH", "both languages have the two lowest AUCs of the table (near chance); the text explains Mongondow by "
               "'Gorontalic retention' while Gorontalo itself is the opposite outlier — domain check needed (PI)")

# --------------------------------------------------------------------------
# I. IPA / syllable robustness inputs, and the Muna 'dh' question (reviewer 1, point 8)
# --------------------------------------------------------------------------
print("\n", "=" * 78, "\nI. DIGRAPHS\n", "=" * 78, sep="")
E041 = consts_from(EXP / "E041_ipa_validation" / "01_ipa_approximation.py", ["UNIVERSAL_DIGRAPHS", "LANG_DIGRAPHS"])


def to_ipa(form, lang):
    r = form.lower()
    for dg, ipa in E041["LANG_DIGRAPHS"].get(lang, []):
        r = r.replace(dg, ipa)
    for dg, ipa in E041["UNIVERSAL_DIGRAPHS"]:
        r = r.replace(dg, ipa)
    return r


D["ipa"] = [to_ipa(a, b) for a, b in zip(D.form, D.language)]
chg = D.ipa != D.form.str.lower()
claim("I01", "349", "forms changed by digraph conversion / Muna", "75 (5.5%) / 54 (24.7%)",
      f"{int(chg.sum())} ({100 * chg.mean():.1f}%) / {int(chg[D.language == 'Muna'].sum())} "
      f"({100 * chg[D.language == 'Muna'].mean():.1f}%)",
      "MATCH" if int(chg.sum()) == 75 and int(chg[D.language == "Muna"].sum()) == 54 else "MISMATCH",
      f"per language {chg.groupby(D.language).sum().astype(int).to_dict()}")
claim("I02", "348", "digraph list in the text: ng, ny (all); gh, bh (Muna)", "4 digraphs",
      f"code converts {[d for d, _ in E041['UNIVERSAL_DIGRAPHS']]} for all, "
      f"{ {k: [d for d, _ in v] for k, v in E041['LANG_DIGRAPHS'].items() if v} }", "MISMATCH",
      "Muna dh WAS converted (to a dental fricative symbol) but is missing from the sentence; Wolio gh is also unlisted")
dig = {}
for dg in ["dh", "gh", "bh", "ng", "ny"]:
    mm = D[(D.language == "Muna") & D.form.str.lower().str.contains(dg)]
    dig[dg] = (len(mm), int(mm.cand.sum()))
claim("I03", "348", "REVIEWER 1 POINT 8: Muna forms containing dh / gh / bh (total, of which uncoded)", "dh not mentioned",
      {k: f"{v[0]} forms, {v[1]} uncoded" for k, v in dig.items()}, "NOTE",
      "dh forms: " + "; ".join(f"{r.form} '{r.concept}'" for r in D[(D.language == 'Muna') & D.form.str.lower().str.contains('dh')].itertuples()))
ng_as_cluster = int(D.form.str.lower().str.contains("ng").sum())
claim("I04", "135,381", "UNITS: digraph 'ng' counted as a consonant cluster and as a nasal cluster", "not stated",
      f"{ng_as_cluster} forms ({100 * ng_as_cluster / N:.0f}%) contain 'ng'; in the orthographic features each counts as CC",
      "NOTE", "the cluster feature partly measures how often a language has the velar nasal")

# --------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------
audit = pd.DataFrame(AUDIT)
audit.to_csv(OUT / "claims_audit.csv", index=False, encoding="utf-8")
D.drop(columns=["sem_ACTION_"]).to_csv(OUT / "forms_recomputed.csv", index=False, encoding="utf-8")
summary = {"n_claims": len(audit), "by_verdict": audit.verdict.value_counts().to_dict(),
           "abvd_snapshot": "lexibank/abvd CLDF, git 917c5a5 (2025-10-07)",
           "versions": {"xgboost": __import__("xgboost").__version__, "sklearn": __import__("sklearn").__version__,
                        "shap": shap.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
(OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\n", "=" * 78, sep="")
print(json.dumps(summary, indent=2))
