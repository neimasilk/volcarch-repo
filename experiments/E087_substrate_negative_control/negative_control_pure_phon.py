"""
E087c: Substrate Detector Negative Control — PURE PHONOLOGY ONLY
================================================================
The definitive test. Removes ALL language-level features:
  - NO language_cognacy_coverage
  - NO language_id_encoded

Only word-level phonological features remain:
  form_length, n_vowels, vowel_ratio, ends_in_vowel, has_glottal,
  has_nasal_cluster, has_reduplication, n_consonant_clusters, has_prefix_like,
  initial_char (one-hot), semantic_domain (one-hot), is_core_vocab

This is the purest test: can the PHONOLOGICAL SHAPE of individual words
distinguish substrate from non-substrate, without knowing which language
the word comes from?
"""
import csv
import io
import json
import re
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).parent.parent.parent
ABVD = REPO / "experiments" / "E022_linguistic_subtraction" / "data" / "abvd" / "cldf"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# Language sets
ORIGINAL_LANGS = {"27": "Muna", "48": "Bugis", "166": "Makassar",
                  "192": "Wolio", "226": "Toraja-Sadan", "674": "Tolaki"}
CONTROL1_LANGS = {"277": "Tagalog", "153": "Cebuano"}
CONTROL2_LANGS = {"233": "Malay", "172": "Minangkabau"}
CONTROL3_LANGS = {"20": "Javanese", "284": "Sundanese"}
CONTROL4_LANGS = {"277": "Tagalog", "33": "Kapampangan"}
CONTROL5_LANGS = {"28": "Iban", "233": "Malay"}
CONTROL6_LANGS = {"648": "Acehnese", "188": "Toba_Batak"}

VOWELS = set("aeiouəɛɨɔæøüöäåãẽĩõũâêîôûàèìòùáéíóú")
AUSTRONESIAN_PREFIXES = (
    "ma-", "me-", "mo-", "pa-", "ka-", "ta-", "na-", "po-",
    "ma", "me", "mo", "pa", "ka", "ta", "na", "po",
    "maŋ", "meŋ", "moŋ", "paŋ", "aŋ", "mak-", "mat-",
)
NASAL_CLUSTERS = ("ng", "mb", "nd", "nj", "mp", "nk", "ŋg", "ŋk", "nc", "nt", "ŋ")

SWADESH_100 = {
    "hand", "leg/foot", "to walk", "road/path", "to come", "to turn", "to swim",
    "skin", "back", "belly", "bone", "intestines", "liver", "breast", "shoulder",
    "blood", "head", "neck", "hair", "nose", "mouth", "tooth", "tongue",
    "to laugh", "to cry", "to vomit", "to eat", "to drink", "to bite",
    "to see", "to hear", "to sleep", "to lie down", "to sit", "to stand",
    "person/human being", "man/male", "woman/female", "child", "husband", "wife",
    "mother", "father", "house", "name", "to say", "rope", "to sew", "needle",
    "to hunt", "to hit", "to steal", "to kill", "to die, be dead", "to live, be alive",
    "to cut, hack", "stick/wood", "to split", "sharp", "dull, blunt",
    "to work", "to plant", "to choose", "to grow", "to swell",
    "to squeeze", "to hold", "to dig", "to buy", "to open, uncover",
    "to pound, beat", "to throw", "to fall", "to fly",
    "dog", "bird", "egg", "feather", "fish", "louse", "mosquito", "rat",
    "meat/flesh", "fat/grease", "tail", "snake", "worm (earthworm)",
    "tree", "leaf", "root", "flower", "fruit", "grass",
    "earth/soil", "stone", "sand", "water", "to flow", "sea", "salt",
    "lake", "woods/forest", "sky", "moon", "star", "cloud", "fog",
    "rain", "thunder", "lightning", "wind", "to blow",
    "warm", "cold", "dry", "wet", "heavy",
    "fire", "to burn", "smoke", "ashes",
    "black", "white", "red", "yellow", "green",
    "small", "big", "short", "long", "thin", "thick", "narrow", "wide",
    "painful, sick", "shy, ashamed", "old", "new", "good", "bad, evil",
    "correct, true", "night", "day", "year",
    "when?", "to hide", "to climb", "at", "in, inside", "above", "below",
    "this", "that", "near", "far", "where?", "I", "thou", "he/she",
    "we (inclusive)", "you", "they", "what?", "who?", "other", "all",
    "and", "if", "how?", "no, not", "to count",
    "One", "Two", "Three", "Four", "Five",
}

SEMANTIC_DOMAINS = {
    "BODY": {"hand", "leg/foot", "back", "belly", "bone", "intestines", "liver",
             "breast", "shoulder", "blood", "head", "neck", "hair", "nose",
             "mouth", "tooth", "tongue", "skin", "ear", "eye", "feather",
             "wing", "tail", "fat/grease", "meat/flesh", "egg"},
    "NATURE": {"earth/soil", "stone", "sand", "water", "sea", "salt", "lake",
               "woods/forest", "sky", "moon", "star", "cloud", "fog", "rain",
               "thunder", "lightning", "wind", "fire", "smoke", "ashes",
               "tree", "leaf", "root", "flower", "fruit", "grass",
               "road/path", "dust", "dirty"},
    "ACTION": {"to walk", "to come", "to turn", "to swim", "to breathe",
               "to sniff, smell", "to laugh", "to cry", "to vomit", "to spit",
               "to chew", "to eat", "to cook", "to drink", "to bite",
               "to see", "to hear", "to sleep", "to lie down", "to sit",
               "to stand", "to say", "to sew", "to hunt", "to hit",
               "to steal", "to kill", "to die, be dead", "to live, be alive",
               "to cut, hack", "to split", "to work", "to plant", "to choose",
               "to grow", "to swell", "to squeeze", "to hold", "to dig",
               "to buy", "to open, uncover", "to pound, beat", "to throw",
               "to fall", "to fly", "to flow", "to blow", "to burn",
               "to hide", "to climb", "to count", "to think", "to fear",
               "to yawn", "to dream", "to scratch", "to stab, pierce",
               "to tie up, fasten", "to know, be knowledgeable", "to turn"},
    "QUALITY": {"warm", "cold", "dry", "wet", "heavy", "black", "white",
                "red", "yellow", "green", "small", "big", "short", "long",
                "thin", "thick", "narrow", "wide", "painful, sick",
                "shy, ashamed", "old", "new", "good", "bad, evil",
                "correct, true", "sharp", "dull, blunt", "rotten", "dirty"},
    "NUMBER": {"One", "Two", "Three", "Four", "Five", "Six", "Seven",
               "Eight", "Nine", "Ten", "Twenty", "Fifty", "One Hundred",
               "One Thousand"},
    "GRAMMAR": {"this", "that", "near", "far", "where?", "I", "thou",
                "he/she", "we (inclusive)", "you", "they", "what?", "who?",
                "other", "all", "and", "if", "how?", "no, not", "when?",
                "in, inside", "above", "below", "at"},
}


def clean_form(raw):
    s = raw.strip()
    s = re.sub(r'\[.*?\]\s*', '', s)
    s = s.strip(" -,;.")
    return s


def classify_domain(concept):
    for domain, concepts in SEMANTIC_DOMAINS.items():
        if concept in concepts:
            return domain
    return "OTHER"


def compute_features_pure_phon(form, concept):
    """Word-level phonological features ONLY. No language info whatsoever."""
    fl = len(form)
    nv = sum(1 for c in form.lower() if c in VOWELS)
    vr = round(nv / fl, 4) if fl > 0 else 0.0
    eiv = 1 if (form and form[-1].lower() in VOWELS) else 0

    c0 = form[0].lower() if form else "other"
    ic = c0 if c0 in ('m', 'a', 'b', 't', 'k', 'p', 's') else "other"

    hg = 1 if ("ʔ" in form or "'" in form) else 0

    fl_lower = form.lower()
    hnc = 0
    for nc in NASAL_CLUSTERS:
        if nc in fl_lower:
            hnc = 1
            break

    hr = 0
    if "-" in form:
        hr = 1
    else:
        for plen in (2, 3):
            for i in range(len(fl_lower) - plen * 2 + 1):
                chunk = fl_lower[i:i + plen]
                if chunk == fl_lower[i + plen:i + plen * 2]:
                    hr = 1
                    break
            if hr:
                break

    ncc = 0
    in_cluster = False
    consec = 0
    for c in fl_lower:
        if c not in VOWELS and c.isalpha():
            consec += 1
            if consec == 2 and not in_cluster:
                ncc += 1
                in_cluster = True
        else:
            consec = 0
            in_cluster = False

    hpl = 0
    for prefix in AUSTRONESIAN_PREFIXES:
        if fl_lower.startswith(prefix):
            hpl = 1
            break

    sd = classify_domain(concept)
    icv = 1 if concept in SWADESH_100 else 0

    return {
        "form_length": fl,
        "n_vowels": nv,
        "vowel_ratio": vr,
        "ends_in_vowel": eiv,
        "initial_char": ic,
        "has_glottal": hg,
        "has_nasal_cluster": hnc,
        "has_reduplication": hr,
        "n_consonant_clusters": ncc,
        "has_prefix_like": hpl,
        "semantic_domain": sd,
        "is_core_vocab": icv,
    }


def load_abvd_data():
    params = {}
    with open(ABVD / "parameters.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params[row["ID"]] = row["Name"]

    cognate_info = defaultdict(list)
    with open(ABVD / "cognates.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cognate_info[row["Form_ID"]].append(row["Cognateset_ID"])

    all_forms = []
    with open(ABVD / "forms.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_forms.append(row)

    return params, cognate_info, all_forms


def build_features_pure_phon(target_langs, params, cognate_info, all_forms):
    lang_forms = [r for r in all_forms if r["Language_ID"] in target_langs]

    rows = []
    for r in lang_forms:
        lid = r["Language_ID"]
        lang_name = target_langs[lid]
        form_id = r["ID"]
        concept = params.get(r["Parameter_ID"], "?")
        raw_value = r.get("Value", "") or r.get("Form", "")
        form_value = clean_form(raw_value)

        if not form_value or len(form_value) < 1:
            continue

        cognacy = r.get("Cognacy", "").strip()
        cog_sets = cognate_info.get(form_id, [])
        if cognacy or cog_sets:
            label = 1
        else:
            label = 0

        feats = compute_features_pure_phon(form_value, concept)
        feats["label"] = label
        feats["language"] = lang_name
        feats["form"] = form_value
        feats["concept"] = concept
        rows.append(feats)

    return rows


def rows_to_Xy(rows):
    df = pd.DataFrame(rows)
    ic_dummies = pd.get_dummies(df["initial_char"], prefix="init")
    df = pd.concat([df, ic_dummies], axis=1)
    sd_dummies = pd.get_dummies(df["semantic_domain"], prefix="sem")
    df = pd.concat([df, sd_dummies], axis=1)

    phon_cols = [
        "form_length", "n_vowels", "vowel_ratio", "ends_in_vowel",
        "has_glottal", "has_nasal_cluster", "has_reduplication",
        "n_consonant_clusters", "has_prefix_like",
    ]
    init_cols = [c for c in df.columns if c.startswith("init_")]
    sem_cols = [c for c in df.columns if c.startswith("sem_")]
    meta_cols = ["is_core_vocab"]

    # NO language features at all
    model_cols = phon_cols + init_cols + meta_cols + sem_cols
    X = df[model_cols].values.astype(float)
    y = df["label"].values
    return X, y, model_cols


def evaluate_cv_auc(X, y, n_seeds=10, n_folds=5):
    if sum(y == 0) < 5 or sum(y == 1) < 5:
        return 0.50, 0.0

    seed_aucs = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed * 7 + 13)
        fold_aucs = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = RandomForestClassifier(
                n_estimators=500, min_samples_leaf=5,
                random_state=42, class_weight="balanced", n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            if len(np.unique(y_test)) < 2:
                continue
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_prob))
        if fold_aucs:
            seed_aucs.append(np.mean(fold_aucs))

    if not seed_aucs:
        return 0.50, 0.0
    return np.mean(seed_aucs), np.std(seed_aucs)


def run_random_label_permutation(X, substrate_rate, n_iter=200, n_folds=5):
    """Permutation test: random labels at given substrate rate."""
    rng = np.random.RandomState(2026)
    aucs = []
    n = X.shape[0]
    for i in range(n_iter):
        y_rand = np.ones(n, dtype=int)
        n_sub = int(n * substrate_rate)
        idx = rng.choice(n, n_sub, replace=False)
        y_rand[idx] = 0

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=rng.randint(0, 100000))
        fold_aucs = []
        for train_idx, test_idx in skf.split(X, y_rand):
            clf = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=5,
                random_state=None, class_weight="balanced", n_jobs=-1,
            )
            clf.fit(X[train_idx], y_rand[train_idx])
            if len(np.unique(y_rand[test_idx])) < 2:
                continue
            y_prob = clf.predict_proba(X[test_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_rand[test_idx], y_prob))
        if fold_aucs:
            aucs.append(np.mean(fold_aucs))
    return np.array(aucs)


def main():
    print("=" * 70)
    print("E087c: NEGATIVE CONTROL — PURE PHONOLOGY ONLY")
    print("  No language_cognacy_coverage, no language_id_encoded")
    print("  ONLY word-level phonological features")
    print("=" * 70)
    print()

    params, cognate_info, all_forms = load_abvd_data()
    print(f"Loaded {len(params)} concepts, {len(all_forms)} forms\n")

    test_sets = [
        ("E027 Original (Sulawesi 6)", ORIGINAL_LANGS),
        ("C1: Tagalog + Cebuano", CONTROL1_LANGS),
        ("C2: Malay + Minangkabau", CONTROL2_LANGS),
        ("C3: Javanese + Sundanese", CONTROL3_LANGS),
        ("C4: Tagalog + Kapampangan", CONTROL4_LANGS),
        ("C5: Iban + Malay", CONTROL5_LANGS),
        ("C6: Acehnese + Toba Batak", CONTROL6_LANGS),
    ]

    results = {}
    print(f"{'Test':<50} {'N':>5} {'Sub%':>6} {'AUC':>8} {'Std':>8}")
    print("-" * 80)

    for name, langs in test_sets:
        rows = build_features_pure_phon(langs, params, cognate_info, all_forms)
        X, y, cols = rows_to_Xy(rows)
        n_sub = sum(y == 0)
        n_tot = len(y)
        pct = round(100 * n_sub / n_tot, 1)

        if n_sub < 10:
            auc, std = 0.50, 0.0
            note = f"(only {n_sub} residuals)"
        else:
            auc, std = evaluate_cv_auc(X, y)
            note = ""

        print(f"{name:<50} {n_tot:>5} {pct:>5.1f}% {auc:>8.4f} {std:>8.4f}  {note}")

        results[name] = {
            "auc": round(float(auc), 4),
            "std": round(float(std), 4),
            "n_forms": int(n_tot),
            "n_substrate": int(n_sub),
            "substrate_rate": round(float(pct), 1),
        }

    # Permutation test on original Sulawesi data
    print("\n--- Permutation test on Sulawesi pure-phonology ---")
    rows_orig = build_features_pure_phon(ORIGINAL_LANGS, params, cognate_info, all_forms)
    X_orig, y_orig, _ = rows_to_Xy(rows_orig)
    orig_auc = results["E027 Original (Sulawesi 6)"]["auc"]

    print("  Running 200 random-label permutations...")
    perm_aucs = run_random_label_permutation(
        X_orig, substrate_rate=sum(y_orig == 0) / len(y_orig), n_iter=200
    )
    perm_mean = np.mean(perm_aucs)
    perm_std = np.std(perm_aucs)
    perm_p = np.mean(perm_aucs >= orig_auc)
    perm_z = (orig_auc - perm_mean) / perm_std if perm_std > 0 else float('inf')

    print(f"  Observed AUC (pure phon): {orig_auc:.4f}")
    print(f"  Permuted mean:            {perm_mean:.4f}")
    print(f"  Permuted std:             {perm_std:.4f}")
    print(f"  Z-score:                  {perm_z:.2f}")
    print(f"  P-value:                  {perm_p:.4f}")
    print(f"  Verdict:                  {'PASS' if perm_p < 0.05 else 'FAIL'}")

    print()

    # ============================================================
    # GRAND COMPARISON TABLE
    # ============================================================
    print("=" * 70)
    print("GRAND COMPARISON: Feature Set Ablation x Language Pair")
    print("=" * 70)
    print()

    # Compare all three feature sets
    # (values from previous runs for with-lcov and no-lcov variants)
    with_lcov_aucs = {
        "E027 Original (Sulawesi 6)": 0.7610,
        "C1: Tagalog + Cebuano": 0.6107,
        "C2: Malay + Minangkabau": 0.6896,
        "C3: Javanese + Sundanese": 0.6505,
        "C4: Tagalog + Kapampangan": 0.6849,
        "C5: Iban + Malay": 0.7939,
    }

    no_lcov_aucs = {
        "E027 Original (Sulawesi 6)": 0.7657,
        "C1: Tagalog + Cebuano": 0.5968,
        "C2: Malay + Minangkabau": 0.6914,
        "C3: Javanese + Sundanese": 0.6473,
        "C4: Tagalog + Kapampangan": 0.6805,
        "C5: Iban + Malay": 0.7881,
    }

    print(f"  {'Test':<40} {'Full':>8} {'No lcov':>8} {'PurePhon':>8}")
    print(f"  {'-'*68}")
    for name in ["E027 Original (Sulawesi 6)", "C1: Tagalog + Cebuano",
                  "C2: Malay + Minangkabau", "C3: Javanese + Sundanese",
                  "C4: Tagalog + Kapampangan", "C5: Iban + Malay"]:
        wl = with_lcov_aucs.get(name, "-")
        nl = no_lcov_aucs.get(name, "-")
        pp = results.get(name, {}).get("auc", "-")
        wl_s = f"{wl:.4f}" if isinstance(wl, float) else wl
        nl_s = f"{nl:.4f}" if isinstance(nl, float) else nl
        pp_s = f"{pp:.4f}" if isinstance(pp, float) else pp
        print(f"  {name:<40} {wl_s:>8} {nl_s:>8} {pp_s:>8}")

    print()

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    c1_auc = results["C1: Tagalog + Cebuano"]["auc"]
    c2_auc = results["C2: Malay + Minangkabau"]["auc"]
    c5_auc = results["C5: Iban + Malay"]["auc"]
    orig_pp = results["E027 Original (Sulawesi 6)"]["auc"]

    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print()

    # Key question: does E027 detect substrate SPECIFICALLY in Sulawesi,
    # or does it detect generic phonological patterns in any language pair?
    print(f"  Key question: Is Sulawesi AUC ({orig_pp:.3f}) significantly")
    print(f"  higher than the best non-substrate control?")
    print()

    best_ctrl = max(c1_auc, c2_auc)
    best_ctrl_name = "C1" if c1_auc >= c2_auc else "C2"
    gap = orig_pp - best_ctrl

    print(f"  Sulawesi pure-phonology AUC:          {orig_pp:.4f}")
    print(f"  Best non-substrate control ({best_ctrl_name}):     {best_ctrl:.4f}")
    print(f"  Gap (Sulawesi - best control):         {gap:+.4f}")
    print(f"  Iban+Malay (C5, coverage mismatch):    {c5_auc:.4f}")
    print(f"  Permutation p-value (Sulawesi):        {perm_p:.4f}")
    print()

    # C5 (Iban+Malay) is problematic because Iban has GENUINELY different
    # phonological characteristics from Malay — different orthographic
    # conventions, more consonant clusters, different prefix patterns.
    # This is not "substrate" but "phylogenetic divergence within Malayic".
    # The key comparison is C1 and C2 (truly closely related, same subgroup).

    if orig_pp > 0.65 and c1_auc < 0.60 and c2_auc < 0.65 and perm_p < 0.05:
        verdict = "CONDITIONAL PASS"
        expl = (
            f"Pure phonological features detect substrate in Sulawesi (AUC={orig_pp:.3f}, "
            f"p={perm_p:.4f}) but NOT in closely related non-substrate pairs "
            f"(C1={c1_auc:.3f}, C2={c2_auc:.3f}). However, C5 (Iban+Malay, AUC={c5_auc:.3f}) "
            f"shows that orthographic/phonological differences between ANY divergent pair "
            f"can inflate AUC. The gap of {gap:+.3f} between Sulawesi and C2 is small. "
            f"The detector captures REAL phonological differences but cannot definitively "
            f"distinguish 'substrate' from 'phylogenetic divergence'. "
            f"Use with explicit caveat in P8."
        )
    elif orig_pp > 0.70 and c1_auc < 0.55 and c2_auc < 0.60:
        verdict = "PASS"
        expl = (
            f"Strong pure-phonology signal in Sulawesi ({orig_pp:.3f}) "
            f"with near-chance controls (C1={c1_auc:.3f}, C2={c2_auc:.3f}). "
            f"Detector is specific."
        )
    elif orig_pp < 0.60:
        verdict = "FAIL — NO PHONOLOGICAL SIGNAL"
        expl = (
            f"Pure phonological features alone produce AUC={orig_pp:.3f} in Sulawesi, "
            f"barely above chance. The E027 AUC=0.762 was driven by "
            f"language-level features (language_cognacy_coverage, language_id), "
            f"not word-level phonology."
        )
    elif c2_auc > 0.65 or c5_auc > 0.75:
        verdict = "GREY ZONE — SPECIFICITY UNCERTAIN"
        expl = (
            f"Sulawesi AUC ({orig_pp:.3f}) is above chance but control pairs also "
            f"show signal (C2={c2_auc:.3f}, C5={c5_auc:.3f}). The detector picks up "
            f"GENERIC phonological variation between any pair of languages with "
            f"different ABVD coverage, not substrate-specific patterns. "
            f"The Sulawesi signal may be partly genuine but is confounded by "
            f"the fact that Sulawesi languages are phonologically more diverse "
            f"than the Philippine/Malayic pairs used as controls."
        )
    else:
        verdict = "CONDITIONAL PASS"
        expl = (
            f"Sulawesi ({orig_pp:.3f}) > controls (C1={c1_auc:.3f}, C2={c2_auc:.3f}) "
            f"but the gap is modest. Interpret E027 with caution."
        )

    print(f"  VERDICT: {verdict}")
    print(f"  {expl}")

    # Save
    output = {
        "experiment": "E087c_pure_phonology",
        "date": "2026-03-16",
        "feature_set": "Pure phonology only (no language features)",
        "n_features_used": int(X_orig.shape[1]),
        "results": results,
        "permutation_test": {
            "observed_auc": round(float(orig_auc), 4),
            "permuted_mean": round(float(perm_mean), 4),
            "permuted_std": round(float(perm_std), 4),
            "z_score": round(float(perm_z), 2),
            "p_value": round(float(perm_p), 4),
        },
        "comparison_table": {
            name: {
                "full_model": round(with_lcov_aucs.get(name, 0), 4),
                "no_lcov": round(no_lcov_aucs.get(name, 0), 4),
                "pure_phon": results.get(name, {}).get("auc", 0),
            }
            for name in ["E027 Original (Sulawesi 6)", "C1: Tagalog + Cebuano",
                          "C2: Malay + Minangkabau", "C5: Iban + Malay"]
        },
        "verdict": verdict,
        "explanation": expl,
    }

    with open(OUT / "negative_control_pure_phon.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUT / 'negative_control_pure_phon.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
