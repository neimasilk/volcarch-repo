"""
E230 -- the digraph ("IPA") and length checks of P8, re-derived (Part A) and re-run for the revision models (Part B).
Pre-registration: DESIGN.md (frozen 2026-10-06). Decision rules of its section 3 are applied literally.

Run:  PYTHONIOENCODING=utf-8 python experiments/E230_p8_revisit_E041_E042_digraph_length/01_digraph_length.py

Conventions
- y = 1 <=> CODED (ABVD cognate set assigned), as in p8common; the candidate class is 1 - y.
- Part A re-implements E041 / E042 from their code (their scripts are NOT run; their result files are NOT touched).
  Constants (digraph maps, vowel sets, prefix tuples) are read from those scripts with ast (P.consts_from) so that
  nothing is retyped; the functions are ported line by line.
- Part B uses p8common (imported, never copied) with the published protocol: 5-fold x 10 seeds, random_state = 7*seed+13,
  XGBoost = P.xgb(), LOLO. Folds depend only on y, so a variant and the baseline see the same folds (paired seeds).
- SDs of ten seed values are written with ddof=1 AND ddof=0; the noise flag of DESIGN s3 is reported for both
  (DESIGN does not say which SD; the ddof=1 flag is the one used in decisions.json, the other is a sensitivity column).
"""
import ast
import csv
import io
import json
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)
EXP = HERE.parent

# UTF-8 stdout (Windows console is cp1252; forms contain IPA symbols), tee-d into run_log.txt
_log = open(RES / "run_log.txt", "w", encoding="utf-8")


class Tee:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def write(self, s):
        self.a.write(s)
        self.b.write(s)
        return len(s)

    def flush(self):
        self.a.flush()
        self.b.flush()


sys.stdout = Tee(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True), _log)
warnings.filterwarnings("ignore")

sys.path.insert(0, str(EXP / "E228_p8_revision_analyses"))
import p8common as P  # noqa: E402  (imported, never copied or edited)
import sklearn  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

T0 = time.time()


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


print("python", sys.version.split()[0], "| xgboost", xgb.__version__, "| scikit-learn", sklearn.__version__,
      "| numpy", np.__version__, "| pandas", pd.__version__)

# ------------------------------------------------------------------------------------------------ constants
E041_PATH = EXP / "E041_ipa_validation" / "01_ipa_approximation.py"
E042_PATH = EXP / "E042_syllable_validation" / "01_syllable_count.py"
C41 = P.consts_from(E041_PATH, ["UNIVERSAL_DIGRAPHS", "LANG_DIGRAPHS", "VOWELS", "AUSTRONESIAN_PREFIXES"])
C42 = P.consts_from(E042_PATH, ["VOWELS"])
V41, V42 = C41["VOWELS"], C42["VOWELS"]
PRE41 = C41["AUSTRONESIAN_PREFIXES"]
E27MATRIX = EXP / "E027_ml_substrate_detection" / "data" / "features_matrix.csv"
STORED41 = json.loads((EXP / "E041_ipa_validation" / "results" / "ipa_validation_summary.json").read_text(encoding="utf-8"))
STORED42 = json.loads((EXP / "E042_syllable_validation" / "results" / "syllable_validation_summary.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ digraphs
def to_ipa_e041(form, language):
    """E041 orthographic_to_ipa: lower-case, language-specific digraphs first, then the universal ones."""
    r = form.lower()
    for dg, sym in C41["LANG_DIGRAPHS"].get(language, []):
        r = r.replace(dg, sym)
    for dg, sym in C41["UNIVERSAL_DIGRAPHS"]:
        r = r.replace(dg, sym)
    return r


def digraph_counts(forms, langs):
    """Sequential application as the code does it: occurrences replaced per list per digraph."""
    rows = []
    allD = ["gh", "bh", "dh", "ng", "ny"]
    for L in sorted(set(langs)):
        fl = [fm.lower() for fm, l in zip(forms, langs) if l == L]
        applied = [d for d, _ in C41["LANG_DIGRAPHS"].get(L, [])] + [d for d, _ in C41["UNIVERSAL_DIGRAPHS"]]
        # sequential: count in the string as it is when that digraph's turn comes
        occ = {d: 0 for d in allD}
        formsw = {d: 0 for d in allD}
        cur = list(fl)
        for dg, sym in C41["LANG_DIGRAPHS"].get(L, []) + C41["UNIVERSAL_DIGRAPHS"]:
            for i, s in enumerate(cur):
                n = s.count(dg)
                if n:
                    occ[dg] += n
                    formsw[dg] += 1
                cur[i] = s.replace(dg, sym)
        n_changed = sum(1 for a, b in zip(fl, cur) if a != b)
        rows.append({"level": "list", "list": L, "digraph": "ALL", "applied_in_E041_code": "", "n_forms": len(fl),
                     "n_forms_changed": n_changed, "pct_forms_changed": round(100 * n_changed / len(fl), 2),
                     "n_forms_containing_raw": "", "n_forms_replaced": "", "n_occurrences_replaced": ""})
        for d in allD:
            rows.append({"level": "digraph", "list": L, "digraph": d, "applied_in_E041_code": d in applied,
                         "n_forms": len(fl), "n_forms_changed": "", "pct_forms_changed": "",
                         "n_forms_containing_raw": sum(1 for s in fl if d in s),
                         "n_forms_replaced": formsw[d] if d in applied else 0,
                         "n_occurrences_replaced": occ[d] if d in applied else 0})
    return pd.DataFrame(rows)


# ================================================================================================ B data
hdr("DATA (p8common loader; y = 1 coded)")
f = P.load_lists(P.TARGET)
y = f.coded.values
cand = 1 - y
langs = f.language.values
LISTS = list(P.TARGET.values())
assert len(f) == 1357 and int(cand.sum()) == 438
forms = list(f.form)
F0 = P.featurize(f.form, f.concept, f.language)
FM25 = P.PURE25
F17 = P.PHON + P.INIT
assert len(FM25) == 25 and len(F17) == 17
MODELS = {"FM25": FM25, "F17": F17}

conv = [to_ipa_e041(a, b) for a, b in zip(forms, langs)]
changed = np.array([c != a.lower() for c, a in zip(conv, forms)])
print("forms changed by the E041 conversion:", int(changed.sum()), "per list:",
      {L: int(changed[langs == L].sum()) for L in LISTS})
F_D1 = P.featurize(conv, f.concept, f.language)
F_D2 = F_D1.copy()
for c in P.INIT:                                   # D2: initial-letter class from the ORIGINAL spelling
    F_D2[c] = F0[c].values
vowel_groups = np.array([max(1, sum(1 for i, c in enumerate(s.lower())
                                    if c in V42 and (i == 0 or s.lower()[i - 1] not in V42))) for s in forms])


def variant_X(vid, cols):
    """Inputs of one variant for one model, as a DataFrame (names kept for traceability)."""
    if vid == "V0":
        return F0[cols].copy()
    if vid == "D1":
        return F_D1[cols].copy()
    if vid == "D2":
        return F_D2[cols].copy()
    X = F0[cols].copy()
    if vid == "L1":
        X["form_length"] = vowel_groups
    elif vid == "L2":
        X = X.drop(columns=["form_length"])
    elif vid == "L3":
        X = X.drop(columns=["form_length", "n_vowels"])
    else:
        raise ValueError(vid)
    return X


VARIANTS = ["V0", "D1", "D2", "L1", "L2", "L3"]


def cv_seed_means(X, yy):
    """Published protocol: per-seed mean of the five fold AUCs (folds from random_state = 7*seed+13)."""
    X = np.asarray(X, dtype=float)
    out = []
    for seed in range(10):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 7 + 13)
        fold = []
        for tr, te in skf.split(X, yy):
            p = P.xgb().fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
            fold.append(roc_auc_score(yy[te], p))
        out.append(float(np.mean(fold)))
    return np.array(out)


def lolo_lists(X):
    res, _ = P.lolo_auc(np.asarray(X, dtype=float), y, langs)
    return {L: res[L] for L in LISTS}


# ================================================================================================ anchors
hdr("ANCHORS (DESIGN s4) -- V0 and D1 counts; the script stops if one fails")
RESULTS = {}      # (variant, model) -> dict(seed_means, cv, lolo dict)
for m, cols in MODELS.items():
    X = variant_X("V0", cols)
    sm = cv_seed_means(X, y)
    lo = lolo_lists(X)
    RESULTS[("V0", m)] = {"seed_means": sm, "cv": float(sm.mean()), "lolo": lo, "lolo_mean": float(np.mean(list(lo.values())))}
    print(f"V0 {m}: CV {sm.mean():.4f}  LOLO mean {np.mean(list(lo.values())):.4f}  ({time.time() - T0:.0f}s)", flush=True)

dh_muna = sorted(fm for fm, l in zip(forms, langs) if l == "Muna" and "dh" in fm.lower())
anchors = []


def anchor(name, expected, got, tol=None):
    ok = (abs(got - expected) <= tol) if tol is not None else (got == expected)
    anchors.append({"name": name, "expected": expected if not isinstance(expected, float) else round(expected, 4),
                    "got": got if not isinstance(got, float) else round(got, 6), "ok": bool(ok)})


anchor("V0 FM25 CV AUC", 0.7265, RESULTS[("V0", "FM25")]["cv"], 0.0005)
anchor("V0 F17 CV AUC", 0.6717, RESULTS[("V0", "F17")]["cv"], 0.0005)
anchor("V0 FM25 LOLO mean", 0.7008, RESULTS[("V0", "FM25")]["lolo_mean"], 0.0005)
anchor("V0 F17 LOLO mean", 0.6413, RESULTS[("V0", "F17")]["lolo_mean"], 0.0005)
anchor("D1 forms changed (total)", 75, int(changed.sum()))
anchor("D1 forms changed Muna", 54, int(changed[langs == "Muna"].sum()))
anchor("D1 forms changed Tolaki", 20, int(changed[langs == "Tolaki"].sum()))
anchor("D1 forms changed Tae' (Toraja-Sadan)", 1, int(changed[langs == "Toraja-Sadan"].sum()))
anchor("Muna forms containing dh (count)", 2, len(dh_muna))
anchors.append({"name": "Muna dh forms (E227 I03: akaradhaa, idho)", "expected": ["akaradhaa", "idho"],
                "got": dh_muna, "ok": dh_muna == ["akaradhaa", "idho"]})
(RES / "anchor_checks.json").write_text(json.dumps({"all_ok": all(a["ok"] for a in anchors), "anchors": anchors},
                                                   indent=2, ensure_ascii=False), encoding="utf-8")
for a in anchors:
    print(("OK   " if a["ok"] else "FAIL ") + f"{a['name']}: expected {a['expected']} got {a['got']}")
if not all(a["ok"] for a in anchors):
    print("\nANCHOR FAILED -- STOP. Nothing adjusted.")
    sys.exit(2)

# ================================================================================================ PART A
hdr("PART A -- re-implementation of E041 and E042 (their scripts are not run)")
rows = list(csv.DictReader(open(E27MATRIX, encoding="utf-8")))
yM = np.array([int(r["label"]) for r in rows])
langM = np.array([r["language"] for r in rows])
print("E027 features_matrix.csv:", len(rows), "rows; label==1:", int(yM.sum()), "; languages:", sorted(set(langM)))

SEM_D = {"ACTION": 0, "BODY": 1, "GRAMMAR": 2, "NATURE": 3, "NUMBER": 4, "OTHER": 5, "QUALITY": 6}
INIT_D = {"m": 0, "a": 1, "b": 2, "t": 3, "k": 4, "p": 5, "s": 6, "other": 7}


# ---- E041 feature functions (ported line by line; vowel set and prefix tuple read from the script)
def e_count_vowels(s):
    return sum(1 for c in s.lower() if c in V41)


def e_vowel_ratio(s):
    return 0.0 if len(s) == 0 else round(e_count_vowels(s) / len(s), 4)


def e_ends_in_vowel(s):
    return 0 if not s else (1 if s[-1].lower() in V41 else 0)


def e_initial_class(s):
    if not s:
        return "other"
    c = s[0].lower()
    return c if c in ("m", "a", "b", "t", "k", "p", "s") else "other"


def e_has_glottal(s):
    return 1 if ("ʔ" in s or "'" in s) else 0


def e_has_nasal_cluster(s):
    fl = s.lower()
    nasal_ipa = ["ŋ", "ɲ", "m", "n"]
    stops = set("ptckbdgq")
    for i in range(len(fl) - 1):
        if fl[i] in "mn" or fl[i] in nasal_ipa:
            if fl[i + 1] in stops:
                return 1
    for nc in ("mb", "nd", "nj", "mp", "nk", "nt", "nc"):
        if nc in fl:
            return 1
    return 0


def e_has_redup(s):
    if "-" in s:
        return 1
    fl = s.lower()
    for plen in (2, 3):
        for i in range(len(fl) - plen * 2 + 1):
            if fl[i:i + plen] == fl[i + plen:i + plen * 2]:
                return 1
    return 0


def e_n_clusters(s):
    count, inc, consec = 0, False, 0
    for c in s.lower():
        if c not in V41 and c.isalpha():
            consec += 1
            if consec == 2 and not inc:
                count += 1
                inc = True
        else:
            consec, inc = 0, False
    return count


def e_prefix(s):
    fl = s.lower()
    return 1 if any(fl.startswith(p) for p in PRE41) else 0


def e041_vec(r, form_for_feats, fl, ncc, ic_form, hybrid_tail=None):
    """Feature vector in the order of E041 make_feature_vector (26 inputs incl. language_id_encoded).
    form_for_feats = the string on which n_vowels ... has_prefix_like are computed."""
    s = form_for_feats
    ic = [0] * 8
    ic[INIT_D.get(e_initial_class(ic_form), 7)] = 1
    sd = [0] * 7
    sd[SEM_D.get(r["semantic_domain"], 5)] = 1
    return [fl, e_count_vowels(s), e_vowel_ratio(s), e_ends_in_vowel(s), *ic, e_has_glottal(s), e_has_nasal_cluster(s),
            e_has_redup(s), ncc, e_prefix(s), *sd, int(r["is_core_vocab"]), int(r["language_id_encoded"])]


X41_ipa, X41_ortho, X41_pure = [], [], []
n_changed41 = 0
for r in rows:
    fo, lg = r["form"], r["language"]
    fi = to_ipa_e041(fo, lg)
    n_changed41 += int(fi != fo.lower())
    fl_o, ncc_o = int(r["form_length"]), int(r["n_consonant_clusters"])
    X41_ipa.append(e041_vec(r, fi, len(fi), e_n_clusters(fi), fi))
    # E041 "ortho" matrix: ONLY length, cluster count and initial letter come from the original spelling;
    # n_vowels, vowel_ratio, ends_in_vowel, glottal, nasal cluster, reduplication and prefix still come from the
    # converted string (this is what make_feature_vector(use_ipa=False) does).
    X41_ortho.append(e041_vec(r, fi, fl_o, ncc_o, fo))
    # not in E041: every input from the original spelling (a clean orthographic baseline with E041's own functions)
    X41_pure.append(e041_vec(r, fo, len(fo), e_n_clusters(fo), fo))
X41_ipa, X41_ortho, X41_pure = (np.array(a, dtype=float) for a in (X41_ipa, X41_ortho, X41_pure))
print("E041 re-implementation: forms changed", n_changed41, "| matrix", X41_ipa.shape)


def xgb_old(seed):
    """E041/E042 parameter set, literally (use_label_encoder is ignored by xgboost >= 2)."""
    return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                             use_label_encoder=False, random_state=seed, verbosity=0)


def cv_generic(X, yy, split_rs, model_rs):
    """Returns the 50 fold AUCs (seed-major) under a stated protocol."""
    aucs = []
    for seed in range(10):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_rs(seed))
        for tr, te in skf.split(X, yy):
            p = xgb_old(model_rs(seed)).fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(yy[te], p))
    return np.array(aucs)


def lolo_generic(X, yy, lg, model_seed=42):
    out = {}
    for L in sorted(set(lg)):
        te = lg == L
        p = xgb_old(model_seed).fit(X[~te], yy[~te]).predict_proba(X[te])[:, 1]
        out[L] = float(roc_auc_score(yy[te], p))
    return out


PROT_OLD = (lambda s: s, lambda s: s)              # E041/E042: folds random_state=seed, model random_state=seed
PROT_PUB = (lambda s: s * 7 + 13, lambda s: 42)    # published: folds 7*seed+13, model random_state=42

A = {}
for nm, X in [("e041_ipa", X41_ipa), ("e041_ortho", X41_ortho), ("e041_pure_ortho", X41_pure)]:
    a = cv_generic(X, yM, *PROT_OLD)
    lo = lolo_generic(X, yM, langM)
    A[nm] = {"cv": float(a.mean()), "cv_sd": float(a.std()), "lolo": lo, "lolo_mean": float(np.mean(list(lo.values())))}
    print(f"{nm:<16} CV {a.mean():.4f} (SD {a.std():.4f})  LOLO mean {A[nm]['lolo_mean']:.4f}  "
          f"{ {k: round(v, 4) for k, v in lo.items()} }  ({time.time() - T0:.0f}s)", flush=True)

# ---- E042
VOW42 = V42


def syl42(s):
    c, inv = 0, False
    for ch in s.lower():
        if ch in VOW42:
            if not inv:
                c += 1
                inv = True
        else:
            inv = False
    return max(c, 1)


def e042_vec(r, length_mode):
    ic = [0] * 8
    ic[INIT_D.get(r.get("initial_char", "other"), 7)] = 1     # E042 takes the initial letter from the stored matrix column
    sd = [0] * 7
    sd[SEM_D.get(r["semantic_domain"], 5)] = 1
    base = [int(r["n_vowels"]), float(r["vowel_ratio"]), int(r["ends_in_vowel"]), *ic, int(r["has_glottal"]),
            int(r["has_nasal_cluster"]), int(r["has_reduplication"]), int(r["n_consonant_clusters"]),
            int(r["has_prefix_like"]), *sd, int(r["is_core_vocab"]), int(r["language_id_encoded"])]
    cl, sy = int(r["form_length"]), syl42(r["form"])
    if length_mode == "char":
        return [cl] + base
    if length_mode == "syllable":
        return [sy] + base
    if length_mode == "both":
        return [cl, sy] + base
    return base                                                 # no_length


corr = float(np.corrcoef([syl42(r["form"]) for r in rows], [int(r["form_length"]) for r in rows])[0, 1])
print("E042 correlation syllable ~ char:", round(corr, 4), "(stored", STORED42["char_syllable_correlation"], ")")
for nm, mode in [("char_length", "char"), ("syllable_count", "syllable"), ("both", "both"), ("no_length", None)]:
    X = np.array([e042_vec(r, mode) for r in rows], dtype=float)
    a = cv_generic(X, yM, *PROT_OLD)
    lo = lolo_generic(X, yM, langM)
    A["e042_" + nm] = {"cv": float(a.mean()), "cv_sd": float(a.std()), "lolo": lo, "lolo_mean": float(np.mean(list(lo.values()))),
                       "n_features": X.shape[1]}
    print(f"e042_{nm:<15} ({X.shape[1]} inputs) CV {a.mean():.4f}  LOLO mean {A['e042_' + nm]['lolo_mean']:.4f}  "
          f"({time.time() - T0:.0f}s)", flush=True)
    if nm == "char_length":
        X42_char = X

# ---- decomposition: which inputs vs which protocol explain the difference from the published models
print("\nDecomposition (same inputs, other protocol; same protocol, other inputs)")
DEC = {}
a = cv_generic(X41_ortho, yM, *PROT_PUB)
DEC["e041_ortho_inputs_published_protocol"] = float(a.mean())
a = cv_generic(X42_char, yM, *PROT_PUB)
DEC["e042_char_inputs_published_protocol"] = float(a.mean())
ABL26 = F0[P.ABL26].values.astype(float)
DEC["ABL26_p8common_published_protocol"] = float(cv_generic(ABL26, y, *PROT_PUB).mean())
DEC["ABL26_p8common_E041_protocol"] = float(cv_generic(ABL26, y, *PROT_OLD).mean())
DEC["FM25_p8common_E041_protocol"] = float(cv_generic(F0[FM25].values.astype(float), y, *PROT_OLD).mean())
for k, v in DEC.items():
    print(f"  {k}: {v:.4f}")

# ------------------------------------------------------------------------------------ A_audit.csv
INP41 = ("E027 features_matrix.csv, 1357 forms; 26 inputs = 25 form/meaning inputs + language_id_encoded (the 'ablated' "
         "26-input model with a language-level input); 'ortho' column = hybrid (length, cluster count, initial letter from the "
         "original spelling, the other six form inputs from the CONVERTED string); 'ipa' column = all form inputs from the converted string")
PROT41 = ("5-fold x 10 seeds, fold split random_state = seed (0..9), model random_state = seed; 'CV AUC' = mean of the 50 fold AUCs; "
          "LOLO = model random_state 42, held-out language's id code unseen in training")
INP42 = ("E027 features_matrix.csv, 1357 forms; 26 inputs (char_length) = 25 form/meaning inputs + language_id_encoded; "
         "all form inputs taken from the stored matrix (original spelling); initial letter from the matrix column")
r4 = lambda v: round(float(v), 4)


def verdict_for(reproduced_ok, described_ok):
    if not reproduced_ok:
        return "MISMATCH"
    return "MATCH" if described_ok else "NOT-AS-DESCRIBED"


aud = []


def add(i, src, item, ms, stored, repro, inputs, protocol, reproduces, described, note, tol=0.0006):
    d = None if (stored is None or repro is None) else abs(repro - stored)
    aud.append({"id": i, "source": src, "item": item, "manuscript_value": ms, "stored_value_in_E04x_json": stored,
                "reproduced_value": None if repro is None else round(repro, 4),
                "abs_diff_vs_stored": None if d is None else round(d, 4),
                "reproduced_within_tol": reproduces, "inputs": inputs, "protocol": protocol,
                "verdict": verdict_for(reproduces, described), "note": note})


e41o, e41i = A["e041_ortho"], A["e041_ipa"]
S41 = STORED41
tolrep = lambda a, b: abs(a - b) <= 0.0006
described_note = ("The manuscript calls the object 'Model B' retrained on IPA forms; the reproduced number belongs to the 26-input "
                  "model WITH a language-id input, other folds than the published protocol, and (baseline column) a hybrid input set. "
                  "Not a model of Tables 2-4.")
add("A01", "E041", "CV AUC, orthographic baseline", 0.772, S41["cv_results"]["ortho"]["auc_mean"], e41o["cv"], INP41, PROT41,
    tolrep(e41o["cv"], S41["cv_results"]["ortho"]["auc_mean"]), False,
    described_note + f" Same inputs under the published folds: {DEC['e041_ortho_inputs_published_protocol']:.4f}; clean orthographic inputs "
    f"(all from the original spelling, E041 functions, E041 protocol): {A['e041_pure_ortho']['cv']:.4f}; 25 inputs of Table 4 (FM25) with E041 protocol: "
    f"{DEC['FM25_p8common_E041_protocol']:.4f}; 26 inputs (ABL26, p8common features) with E041 protocol: {DEC['ABL26_p8common_E041_protocol']:.4f}, "
    f"with the published protocol: {DEC['ABL26_p8common_published_protocol']:.4f} (Table 4 prints 0.763).")
add("A02", "E041", "CV AUC, IPA", 0.774, S41["cv_results"]["ipa"]["auc_mean"], e41i["cv"], INP41, PROT41,
    tolrep(e41i["cv"], S41["cv_results"]["ipa"]["auc_mean"]), False, described_note)
add("A03", "E041", "CV delta IPA - ortho", 0.002, round(S41["cv_results"]["ipa"]["auc_mean"] - S41["cv_results"]["ortho"]["auc_mean"], 4),
    e41i["cv"] - e41o["cv"], INP41, PROT41, tolrep(e41i["cv"] - e41o["cv"], S41["cv_results"]["ipa"]["auc_mean"] - S41["cv_results"]["ortho"]["auc_mean"]),
    False, "Delta of two numbers from the hybrid-baseline comparison; the 'orthographic' side is not a pure orthographic model. " +
    f"Delta with clean orthographic inputs: {e41i['cv'] - A['e041_pure_ortho']['cv']:+.4f}.")
add("A04", "E041", "LOLO mean AUC, orthographic baseline", 0.724, S41["lolo_results"]["ortho_mean_auc"], e41o["lolo_mean"], INP41, PROT41,
    tolrep(e41o["lolo_mean"], S41["lolo_results"]["ortho_mean_auc"]), False,
    described_note + f" Clean orthographic inputs: LOLO mean {A['e041_pure_ortho']['lolo_mean']:.4f}.")
add("A05", "E041", "LOLO mean AUC, IPA", 0.733, S41["lolo_results"]["ipa_mean_auc"], e41i["lolo_mean"], INP41, PROT41,
    tolrep(e41i["lolo_mean"], S41["lolo_results"]["ipa_mean_auc"]), False, described_note)
add("A06", "E041", "LOLO delta IPA - ortho", 0.009, round(S41["lolo_results"]["ipa_mean_auc"] - S41["lolo_results"]["ortho_mean_auc"], 4),
    e41i["lolo_mean"] - e41o["lolo_mean"], INP41, PROT41,
    tolrep(e41i["lolo_mean"] - e41o["lolo_mean"], S41["lolo_results"]["ipa_mean_auc"] - S41["lolo_results"]["ortho_mean_auc"]), False,
    f"Delta with clean orthographic inputs: {e41i['lolo_mean'] - A['e041_pure_ortho']['lolo_mean']:+.4f}.")
ge65_o = sum(1 for v in e41o["lolo"].values() if v >= 0.65)
ge65_i = sum(1 for v in e41i["lolo"].values() if v >= 0.65)
add("A07", "E041", "All six LOLO languages >= 0.65 under IPA", "6/6", float(S41["lolo_results"]["ipa_ge65"]), float(ge65_i), INP41, PROT41,
    ge65_i == S41["lolo_results"]["ipa_ge65"], False,
    f"Reproduced {ge65_i}/6 (baseline {ge65_o}/6). Lowest IPA list AUC {min(e41i['lolo'].values()):.4f}. "
    "0.65 is the manuscript's own threshold; the claim is true for this model, which is not the one in Tables 2-4.")
mu = e41i["lolo"]["Muna"] - e41o["lolo"]["Muna"]
dd = {L: e41i["lolo"][L] - e41o["lolo"][L] for L in e41i["lolo"]}
add("A08", "E041", "Muna LOLO delta (IPA - ortho)", 0.042, round(S41["lolo_results"]["ipa"]["Muna"]["auc"] - S41["lolo_results"]["ortho"]["Muna"]["auc"], 4),
    mu, INP41, PROT41, tolrep(mu, S41["lolo_results"]["ipa"]["Muna"]["auc"] - S41["lolo_results"]["ortho"]["Muna"]["auc"]), False,
    "Per-list deltas of this reproduction: " + ", ".join(f"{k} {v:+.4f}" for k, v in dd.items()) +
    f". With clean orthographic inputs as the baseline the Muna delta is {e41i['lolo']['Muna'] - A['e041_pure_ortho']['lolo']['Muna']:+.4f} "
    f"(per-list: " + ", ".join(f"{L} {e41i['lolo'][L] - A['e041_pure_ortho']['lolo'][L]:+.4f}" for L in e41i['lolo']) + "). "
    "One held-out list, one model seed (random_state 42), no interval.")
aud.append({"id": "A16", "source": "E041", "item": "inference: 'orthographic digraphs were adding noise rather than signal' (from the Muna delta)",
            "manuscript_value": "text", "stored_value_in_E04x_json": None, "reproduced_value": None, "abs_diff_vs_stored": None,
            "reproduced_within_tol": None, "inputs": INP41, "protocol": PROT41, "verdict": "UNSUPPORTED",
            "note": "Rests on one held-out list, one model seed and a hybrid baseline; no test or interval; see Part B / decisions.json R4."})
aud.append({"id": "A17", "source": "E041+E042", "item": "inference: 'the model detects phonological patterns, not orthographic artifacts'",
            "manuscript_value": "text", "stored_value_in_E04x_json": None, "reproduced_value": None, "abs_diff_vs_stored": None,
            "reproduced_within_tol": None, "inputs": INP41, "protocol": PROT41, "verdict": "UNSUPPORTED",
            "note": "An unchanged AUC under a change of ~5% of the strings (digraphs) or of one input (length) cannot show absence of "
                    "orthographic dependence; E228 S1 shows one orthographic dependence (glottal convention); E230 Part B L3 shows the length "
                    "information survives in the vowel count."})
S42 = STORED42
INP42n = INP42
PROT42 = PROT41
add("A09", "E042", "CV AUC, character-count baseline", 0.768, S42["cv_results"]["char_length"]["auc_mean"], A["e042_char_length"]["cv"], INP42n, PROT42,
    tolrep(A["e042_char_length"]["cv"], S42["cv_results"]["char_length"]["auc_mean"]), False,
    "Same 26-input model as the 'ablated' row of the submitted Table 4 (0.763), but with folds random_state = seed instead of 7*seed+13. "
    f"Same inputs under the published folds: {DEC['e042_char_inputs_published_protocol']:.4f}. " + "Not a model of Tables 2-4 under the published protocol.")
add("A10", "E042", "CV AUC, syllable count replaces character count", 0.769, S42["cv_results"]["syllable_count"]["auc_mean"],
    A["e042_syllable_count"]["cv"], INP42n, PROT42, tolrep(A["e042_syllable_count"]["cv"], S42["cv_results"]["syllable_count"]["auc_mean"]), False,
    "'Syllable' = number of maximal vowel-letter runs (E042 count_syllables), not a syllabification.")
add("A11", "E042", "CV delta syllable - char", "<0.001", round(S42["cv_results"]["syllable_count"]["auc_mean"] - S42["cv_results"]["char_length"]["auc_mean"], 4),
    A["e042_syllable_count"]["cv"] - A["e042_char_length"]["cv"], INP42n, PROT42,
    tolrep(A["e042_syllable_count"]["cv"] - A["e042_char_length"]["cv"], S42["cv_results"]["syllable_count"]["auc_mean"] - S42["cv_results"]["char_length"]["auc_mean"]), False,
    "Stored delta +0.0006; no variance estimate of the delta.")
add("A12", "E042", "LOLO mean, character-count baseline", 0.722, S42["lolo_means"]["char_length"], A["e042_char_length"]["lolo_mean"], INP42n, PROT42,
    tolrep(A["e042_char_length"]["lolo_mean"], S42["lolo_means"]["char_length"]), True,
    "LOLO has no fold protocol; the 26-input model is the one of the 'ablated' row of Table 4 (0.722 there): MATCH is on the number and the model.")
add("A13", "E042", "LOLO mean, syllable count replaces character count", 0.728, S42["lolo_means"]["syllable_count"],
    A["e042_syllable_count"]["lolo_mean"], INP42n, PROT42, tolrep(A["e042_syllable_count"]["lolo_mean"], S42["lolo_means"]["syllable_count"]), True,
    "Per list the stored values move by up to " + f"{max(abs(S42['lolo_results']['syllable_count'][L] - S42['lolo_results']['char_length'][L]) for L in S42['lolo_results']['char_length']):.3f}"
    " (Makassar +0.023, Muna -0.026): the mean hides list-level movements of the order of 0.02-0.03.")
add("A14", "E042", "'no length' CV AUC (form_length removed; n_vowels, vowel_ratio stay)", 0.769, S42["cv_results"]["no_length"]["auc_mean"],
    A["e042_no_length"]["cv"], INP42n, PROT42, tolrep(A["e042_no_length"]["cv"], S42["cv_results"]["no_length"]["auc_mean"]), False,
    "Only form_length is removed; the vowel count (and the ratio, which is vowel count over length) remain, so the length information "
    "is still in the model. The sentence 'does not depend on form length at all' is not tested by this variant (see Part B L3).")
add("A15", "E042", "'no length' LOLO mean", 0.732, S42["lolo_means"]["no_length"], A["e042_no_length"]["lolo_mean"], INP42n, PROT42,
    tolrep(A["e042_no_length"]["lolo_mean"], S42["lolo_means"]["no_length"]), False, "Same remark as A14.")
pd.DataFrame(aud).to_csv(RES / "A_audit.csv", index=False, encoding="utf-8")
print("\nA_audit.csv:", len(aud), "rows; reproduced within 0.0006 of the stored value:",
      sum(1 for r in aud if r["reproduced_within_tol"]), "/", len(aud))
for r in aud:
    print(f"  {r['id']} {r['item'][:58]:<58} ms {str(r['manuscript_value']):<7} stored {r['stored_value_in_E04x_json']} "
          f"repro {r['reproduced_value']}  diff {r['abs_diff_vs_stored']}  {r['verdict']}")

# ================================================================================================ PART B
hdr("PART B -- variants x models, published protocol")
for vid in VARIANTS[1:]:
    for m, cols in MODELS.items():
        X = variant_X(vid, cols)
        sm = cv_seed_means(X, y)
        lo = lolo_lists(X)
        RESULTS[(vid, m)] = {"seed_means": sm, "cv": float(sm.mean()), "lolo": lo, "lolo_mean": float(np.mean(list(lo.values()))),
                             "n_inputs": X.shape[1]}
        print(f"{vid} {m}: n_inputs {X.shape[1]} CV {sm.mean():.4f}  LOLO {RESULTS[(vid, m)]['lolo_mean']:.4f}  ({time.time() - T0:.0f}s)", flush=True)
for m, cols in MODELS.items():
    RESULTS[("V0", m)]["n_inputs"] = len(cols)


def band(d):
    a = abs(d)
    return "no dependence shown (|d|<=0.010)" if a <= 0.010 else ("depends (|d|>0.020)" if a > 0.020 else "partial (0.010<|d|<=0.020)")


brow, lrow = [], []
for vid in VARIANTS:
    for m in MODELS:
        r, b = RESULTS[(vid, m)], RESULTS[("V0", m)]
        diffs = r["seed_means"] - b["seed_means"]
        mean_d = float(diffs.mean())
        sd1, sd0 = float(diffs.std(ddof=1)), float(diffs.std(ddof=0))
        thr1, thr0 = 2 * sd1 / np.sqrt(10), 2 * sd0 / np.sqrt(10)
        base = vid == "V0"
        r["delta"] = r["cv"] - b["cv"]
        r["diff_mean"], r["sd1"], r["sd0"] = mean_d, sd1, sd0
        r["dist1"] = None if base else bool(abs(mean_d) >= thr1)
        r["dist0"] = None if base else bool(abs(mean_d) >= thr0)
        r["band_cv"] = "baseline" if base else band(r["delta"])
        r["lolo_delta"] = r["lolo_mean"] - b["lolo_mean"]
        brow.append({"variant": vid, "model": m, "n_inputs": r["n_inputs"], "cv_auc": round(r["cv"], 4),
                     "cv_delta_vs_V0": round(r["delta"], 4), "paired_diff_mean": round(mean_d, 5),
                     "paired_diff_sd_ddof1": round(sd1, 5), "paired_diff_sd_ddof0": round(sd0, 5),
                     "noise_threshold_2sd_over_sqrt10_ddof1": round(thr1, 5),
                     "distinguishable_from_zero_ddof1": r["dist1"], "distinguishable_from_zero_ddof0": r["dist0"],
                     "paired_diffs_10_seeds": " ".join(f"{x:+.4f}" for x in diffs),
                     "lolo_mean": round(r["lolo_mean"], 4), "lolo_delta_vs_V0": round(r["lolo_delta"], 4),
                     "band_on_cv_delta": r["band_cv"],
                     "band_on_lolo_delta_informative": "baseline" if base else band(r["lolo_delta"])})
        for L in LISTS:
            lrow.append({"list": L, "variant": vid, "model": m, "auc": round(r["lolo"][L], 4),
                         "delta_vs_V0": round(r["lolo"][L] - b["lolo"][L], 4)})
pd.DataFrame(brow).to_csv(RES / "B_variants.csv", index=False, encoding="utf-8")
pd.DataFrame(lrow).to_csv(RES / "B_lolo_by_list.csv", index=False, encoding="utf-8")
dc = digraph_counts(forms, list(langs))
dc.to_csv(RES / "B_forms_changed.csv", index=False, encoding="utf-8")

hdr("B_variants (CV AUC, delta, paired seed difference, flag)")
print(pd.DataFrame(brow)[["variant", "model", "cv_auc", "cv_delta_vs_V0", "paired_diff_mean", "paired_diff_sd_ddof1",
                          "distinguishable_from_zero_ddof1", "distinguishable_from_zero_ddof0", "lolo_mean", "lolo_delta_vs_V0",
                          "band_on_cv_delta"]].to_string(index=False))

# ================================================================================================ decisions
hdr("DECISIONS (DESIGN s3, applied literally)")
BAND_TEXT = {"no dependence shown (|d|<=0.010)": "NO DEPENDENCE SHOWN at this resolution: the text may say that discrimination is unchanged "
                                                 "under this conversion and nothing more (not that the model 'detects phonology rather than orthography')",
             "partial (0.010<|d|<=0.020)": "PARTIAL DEPENDENCE",
             "depends (|d|>0.020)": "MODEL DEPENDS on this convention: robustness sentence withdrawn for it"}
dec = {"noise_rule": "a delta is 'not distinguishable from zero' when |mean paired seed difference| < 2*SD/sqrt(10); SD ddof=1 used here "
                     "(ddof=0 flag alongside in B_variants.csv)",
       "R1_bands_on_CV_delta": {}}
for vid in VARIANTS[1:]:
    for m in MODELS:
        r = RESULTS[(vid, m)]
        dec["R1_bands_on_CV_delta"][f"{vid}_{m}"] = {
            "cv_delta": round(r["delta"], 4), "band": r["band_cv"], "outcome": BAND_TEXT[r["band_cv"]],
            "paired_mean": round(r["diff_mean"], 5), "paired_sd_ddof1": round(r["sd1"], 5),
            "distinguishable_from_zero_ddof1": r["dist1"], "distinguishable_from_zero_ddof0": r["dist0"],
            "lolo_delta": round(r["lolo_delta"], 4)}
# R2: the digraph robustness sentence (tex 352-359 / abstract): D1 and D2 read per model
dec["R2_digraph_robustness_sentence"] = {
    f"{vid}_{m}": {"cv_delta": round(RESULTS[(vid, m)]["delta"], 4), "band": RESULTS[(vid, m)]["band_cv"],
                   "outcome": BAND_TEXT[RESULTS[(vid, m)]["band_cv"]]} for vid in ("D1", "D2") for m in MODELS}
dec["R2_digraph_robustness_sentence"]["verdict_robustness_sentence_withdrawn_for_digraphs"] = bool(
    any(RESULTS[(vid, m)]["band_cv"].startswith("depends") for vid in ("D1", "D2") for m in MODELS))
dec["R2_digraph_robustness_sentence"]["note"] = ("whatever the band, 'the model detects phonological patterns rather than orthographic artifacts' is "
                                                 "not supported by an unchanged AUC under a ~5% string change (E228 S1 already shows one orthographic dependence)")
# R3: 'does not depend on form length at all' tested by L3 (CV delta), per model
r3 = {}
for m in MODELS:
    d3, d2, d1 = RESULTS[("L3", m)]["delta"], RESULTS[("L2", m)]["delta"], RESULTS[("L1", m)]["delta"]
    r3[m] = {"L3_cv_delta": round(d3, 4), "L2_cv_delta": round(d2, 4), "L1_cv_delta": round(d1, 4),
             "L3_le_minus_0.010": bool(d3 <= -0.010),
             "L2_unchanged_but_L3_not": bool(abs(d2) <= 0.010 and d3 <= -0.010),
             "L3_distinguishable_from_zero_ddof1": RESULTS[("L3", m)]["dist1"]}
w = any(r3[m]["L3_le_minus_0.010"] for m in MODELS)
dec["R3_length_sentence_tex359_449"] = {
    "per_model": r3, "sentence_withdrawn": bool(w),
    "outcome": ("WITHDRAWN: delta(L3) <= -0.010 in " + ", ".join(m for m in MODELS if r3[m]["L3_le_minus_0.010"]) if w else
                "NOT withdrawn by L3: delta(L3) > -0.010 in both models (this is 'no dependence shown at this resolution', not a proof of independence)"),
    "L2_unchanged_but_L3_not_models": [m for m in MODELS if r3[m]["L2_unchanged_but_L3_not"]],
    "note_if_any_L2_unchanged_but_L3_not": "length information was still in the model through the vowel count"}
# R4: Muna largest positive LOLO delta in both models
r4d = {}
for vid in ("D1", "D2"):
    for m in MODELS:
        dl = {L: RESULTS[(vid, m)]["lolo"][L] - RESULTS[("V0", m)]["lolo"][L] for L in LISTS}
        top = max(dl, key=dl.get)
        r4d[f"{vid}_{m}"] = {"lolo_delta_by_list": {L: round(v, 4) for L, v in dl.items()}, "largest_positive_list": top if dl[top] > 0 else None,
                             "muna_is_largest_positive": bool(top == "Muna" and dl["Muna"] > 0), "muna_delta": round(dl["Muna"], 4),
                             "muna_rank_among_six": int(1 + sum(1 for L in LISTS if dl[L] > dl["Muna"]))}
cond_all4 = all(v["muna_is_largest_positive"] for v in r4d.values())
cond_D1 = all(r4d[f"D1_{m}"]["muna_is_largest_positive"] for m in MODELS)
dec["R4_muna_largest_improvement_tex354"] = {
    "per_variant_model": r4d, "muna_largest_positive_in_both_models_under_D1": bool(cond_D1),
    "muna_largest_positive_under_D1_and_D2_both_models": bool(cond_all4),
    "sentence_withdrawn_strict_reading_all_four": bool(not cond_all4),
    "sentence_withdrawn_D1_only_reading": bool(not cond_D1),
    "note": "DESIGN s3 says 'under D1 and D2 in both models'; the strict reading (all four combinations) is the literal one, the D1-only reading is shown for comparison",
    "no_test_no_interval": "one list, one held-out run per model"}
# R5: the digraph list the code applies, with counts
lst = {}
for L in LISTS:
    sub = dc[(dc.level == "digraph") & (dc.list == L) & (dc.applied_in_E041_code == True)]  # noqa: E712
    lst[L] = {"digraphs_applied": [{"digraph": r.digraph, "forms_replaced": int(r.n_forms_replaced),
                                    "occurrences": int(r.n_occurrences_replaced)} for r in sub.itertuples()],
              "forms_changed": int(dc[(dc.level == "list") & (dc.list == L)].n_forms_changed.iloc[0]),
              "forms_total": int(dc[(dc.level == "list") & (dc.list == L)].n_forms.iloc[0])}
dec["R5_digraph_list_applied_by_the_code"] = {"mapping": {"universal": C41["UNIVERSAL_DIGRAPHS"], "by_list": C41["LANG_DIGRAPHS"]},
                                              "counts_by_list": lst, "forms_changed_total": int(changed.sum())}
dec["domain_flag"] = "symbols are placeholders; whether Muna bh/dh/gh are fricatives, implosives or dental stops is for a specialist (DESIGN s6)"
(RES / "decisions.json").write_text(json.dumps(dec, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in dec.items() if k.startswith("R2") or k.startswith("R3")}, indent=1, ensure_ascii=False)[:3500])
print("\nR4 Muna:", {k: (v["muna_delta"], v["muna_rank_among_six"], v["muna_is_largest_positive"]) for k, v in r4d.items()})
print("done in", round(time.time() - T0), "s")
_log.flush()
