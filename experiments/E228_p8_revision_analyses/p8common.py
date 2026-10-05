"""
Shared loading, feature extraction and model code for E228.

Feature definitions and constants are those of E027 (read from its script with `ast`,
never retyped). E227 showed that this extraction reproduces the stored E027 matrix
exactly (claims_audit.csv, row B01).
"""
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent.parent
EXP = REPO / "experiments"
ABVD = EXP / "E022_linguistic_subtraction" / "data" / "abvd" / "cldf"

TARGET = {"27": "Muna", "48": "Bugis", "166": "Makassar",
          "192": "Wolio", "226": "Toraja-Sadan", "674": "Tolaki"}
SOUTH_SULAWESI = {"Bugis", "Makassar", "Toraja-Sadan"}


def consts_from(path, names):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in names:
                try:
                    out[t.id] = ast.literal_eval(node.value)
                except ValueError:
                    out[t.id] = eval(compile(ast.Expression(node.value), str(path), "eval"), {"set": set})
    missing = set(names) - set(out)
    if missing:
        raise RuntimeError(f"{path}: constants not found: {missing}")
    return out


E022 = consts_from(EXP / "E022_linguistic_subtraction" / "enhanced_subtraction.py",
                   ["PAN_KNOWN", "SANSKRIT_PATTERNS", "ARABIC_PATTERNS", "MALAY_TRADE"])
E027 = consts_from(EXP / "E027_ml_substrate_detection" / "00_prepare_features.py",
                   ["SWADESH_100", "SEMANTIC_DOMAINS", "VOWELS", "AUSTRONESIAN_PREFIXES", "NASAL_CLUSTERS"])
VOWELS = E027["VOWELS"]
NUMERALS = set(E027["SEMANTIC_DOMAINS"]["NUMBER"])
PAN_LIST = set(E022["PAN_KNOWN"])

_params = pd.read_csv(ABVD / "parameters.csv", dtype=str, keep_default_na=False)
PNAME = dict(zip(_params.ID, _params.Name))
_forms_cache = None


def all_forms():
    global _forms_cache
    if _forms_cache is None:
        _forms_cache = pd.read_csv(ABVD / "forms.csv", dtype=str, keep_default_na=False,
                                   usecols=["ID", "Language_ID", "Parameter_ID", "Value", "Form", "Cognacy", "Loan"])
    return _forms_cache


def clean_form(raw):
    s = re.sub(r"\[.*?\]\s*", "", raw.strip())
    return s.strip(" -,;.")


def load_lists(lang_map):
    """lang_map: {ABVD language id: short name}. Returns one row per non-empty form."""
    fa = all_forms()
    f = fa[fa.Language_ID.isin(lang_map)].copy()
    f["language"] = f.Language_ID.map(lang_map)
    f["concept"] = f.Parameter_ID.map(PNAME)
    f["raw"] = f.Value.where(f.Value != "", f.Form)
    f["form"] = f.raw.map(clean_form)
    f = f[f.form != ""].copy()
    f["coded"] = (f.Cognacy.str.strip() != "").astype(int)
    f["candidate"] = 1 - f.coded
    f["abvd_loan_flag"] = (~f.Loan.str.strip().str.lower().isin(["", "false", "0"])).astype(int)
    return f.reset_index(drop=True)


def _is_loanword(value, loan_set):
    fl = value.lower().strip()
    if len(fl) < 3:
        return False
    return any(len(l) >= 3 and (fl == l or fl.startswith(l) or fl.endswith(l)) and len(l) / len(fl) >= 0.6
               for l in loan_set)


def table1_set(f):
    """The 356-form residual set of the manuscript's Table 1 (E022 'enhanced'), as a 0/1 column."""
    pat = f.raw.map(lambda v: _is_loanword(v, E022["SANSKRIT_PATTERNS"]) or _is_loanword(v, E022["ARABIC_PATTERNS"])
                    or _is_loanword(v, E022["MALAY_TRADE"]))
    tagged = (f.coded == 1) | (f.abvd_loan_flag == 1) | pat
    return ((~tagged) & (~f.concept.isin(PAN_LIST))).astype(int)


# ---------------------------------------------------------------- features
def _n_vowels(s):
    return sum(1 for c in s.lower() if c in VOWELS)


def _n_clusters(s):
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


def _redup(s):
    if "-" in s:
        return 1
    fl = s.lower()
    for plen in (2, 3):
        for i in range(len(fl) - plen * 2 + 1):
            if fl[i:i + plen] == fl[i + plen:i + plen * 2]:
                return 1
    return 0


def _domain(c):
    for d, cs in E027["SEMANTIC_DOMAINS"].items():
        if c in cs:
            return d
    return "OTHER"


PHON = ["form_length", "n_vowels", "vowel_ratio", "ends_in_vowel", "has_glottal", "has_nasal_cluster",
        "has_reduplication", "n_consonant_clusters", "has_prefix_like"]
INIT = ["init_a", "init_b", "init_k", "init_m", "init_other", "init_p", "init_s", "init_t"]
SEM = ["sem_ACTION", "sem_BODY", "sem_GRAMMAR", "sem_NATURE", "sem_NUMBER", "sem_OTHER", "sem_QUALITY"]
PURE25 = PHON + INIT + ["is_core_vocab"] + SEM
ABL26 = PURE25 + ["language_id_encoded"]
LANG_CODE = {n: i for i, n in enumerate(sorted(TARGET.values()))}


def featurize(forms, concepts, languages=None):
    forms = pd.Series(list(forms))
    concepts = pd.Series(list(concepts))
    X = pd.DataFrame({
        "form_length": forms.map(len),
        "n_vowels": forms.map(_n_vowels),
        "vowel_ratio": forms.map(lambda s: round(_n_vowels(s) / len(s), 4) if len(s) else 0.0),
        "ends_in_vowel": forms.map(lambda s: int(bool(s) and s[-1].lower() in VOWELS)),
        "has_glottal": forms.map(lambda s: int("ʔ" in s or "'" in s)),
        "has_nasal_cluster": forms.map(lambda s: int(any(nc in s.lower() for nc in E027["NASAL_CLUSTERS"]))),
        "has_reduplication": forms.map(_redup),
        "n_consonant_clusters": forms.map(_n_clusters),
        "has_prefix_like": forms.map(lambda s: int(s.lower().startswith(tuple(E027["AUSTRONESIAN_PREFIXES"])))),
        "is_core_vocab": concepts.map(lambda c: int(c in E027["SWADESH_100"])),
    })
    init = forms.map(lambda s: s[0].lower() if s and s[0].lower() in "mabtkps" else "other")
    for c in INIT:
        X[c] = (init == c[5:]).astype(int)
    dom = concepts.map(_domain)
    for c in SEM:
        X[c] = (dom == c[4:]).astype(int)
    X["semantic_domain"] = dom
    if languages is not None:
        X["language_id_encoded"] = pd.Series(list(languages)).map(LANG_CODE)
    return X


# ---------------------------------------------------------------- models
def xgb():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, reg_lambda=1.0,
                         scale_pos_weight=1.0, eval_metric="logloss", random_state=42, verbosity=0)


def cv_auc(X, y):
    """Published protocol. y = 1 for coded forms. Returns mean AUC, SD over seed means, and out-of-fold P(candidate)."""
    X = np.asarray(X, dtype=float)
    oof = np.zeros(len(y))
    seed_auc = []
    for seed in range(10):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 7 + 13)
        fold = []
        for tr, te in skf.split(X, y):
            p = xgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            fold.append(roc_auc_score(y[te], p))
            oof[te] += (1 - p) / 10
        seed_auc.append(np.mean(fold))
    return float(np.mean(seed_auc)), float(np.std(seed_auc)), oof


def lolo_auc(X, y, langs):
    X = np.asarray(X, dtype=float)
    langs = np.asarray(langs)
    res, p_all = {}, np.zeros(len(y))
    for L in sorted(set(langs)):
        te = langs == L
        p = xgb().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        res[L] = float(roc_auc_score(y[te], p))
        p_all[te] = 1 - p
    return res, p_all


# ---------------------------------------------------------------- string distance
def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]


def ned(a, b):
    if not a and not b:
        return 0.0
    return lev(a, b) / max(len(a), len(b))


def norm(s):
    s = re.sub(r"[\[\(].*?[\]\)]", "", s.lower())
    return re.sub(r"[\*\s\-'ʔ’`.,;/]", "", s)


_memo = {}


def d_stem(a, b):
    """Stem-tolerant distance (DESIGN §4): up to three leading characters may be ignored on either side."""
    key = (a, b) if a <= b else (b, a)
    if key in _memo:
        return _memo[key]
    best = ned(a, b)
    for i in range(4):
        if len(a) - i < 3:
            break
        for j in range(4):
            if len(b) - j < 3:
                break
            if i or j:
                d = ned(a[i:], b[j:])
                if d < best:
                    best = d
    _memo[key] = best
    return best
