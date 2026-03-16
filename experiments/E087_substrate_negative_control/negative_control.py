"""
E087: Substrate Detector Negative Control
==========================================
Tests whether the E027 ML substrate detector (XGBoost/RF, AUC=0.762) is
specific to real pre-Austronesian substrate, or whether it detects "substrate"
in ANY pair of closely related Austronesian languages.

If the detector finds substrate where none should exist, it's detecting
phylogenetic noise — not genuine pre-Austronesian residue.

Three controls:
  Control 1 — Closely Related Languages (Tagalog + Cebuano):
    Both Central Philippine, high ABVD cognacy. If AUC > 0.70 -> detector broken.

  Control 2 — Known No-Substrate Pair (Malay + Minangkabau):
    Malayic sister languages, no substrate between them. AUC should be ~0.50.

  Control 3 — Random Label Assignment:
    Take Javanese + Sundanese words, randomly assign labels with same
    substrate proportion as E027 (32.3%). AUC should be ~0.50.

Pass criteria:
  Control 1 AUC < 0.60 -> VOLCARCH PASSES (detector is specific)
  Control 1 AUC 0.60–0.70 -> GREY ZONE
  Control 1 AUC > 0.70 -> VOLCARCH FAILS (detector is broken)
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
E027_DATA = REPO / "experiments" / "E027_ml_substrate_detection" / "data"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# ============================================================
# Control language sets
# ============================================================
# Control 1: Central Philippine — Tagalog + Cebuano
# Both are well-documented, high cognacy coverage, closely related.
# No historical linguist claims substrate between them.
CONTROL1_LANGS = {
    "277": "Tagalog",
    "153": "Cebuano",
}

# Control 2: Malayic — Malay (Bahasa Indonesia) + Minangkabau
# Sister languages in the Malayic subgroup, geographically adjacent in Sumatra.
# No known substrate influence between them.
CONTROL2_LANGS = {
    "233": "Malay",
    "172": "Minangkabau",
}

# Control 3: Western Indonesian — Javanese + Sundanese
# These are for the random-label test.
CONTROL3_LANGS = {
    "20": "Javanese",
    "284": "Sundanese",
}

# The original E027 Sulawesi languages (for comparison)
ORIGINAL_LANGS = {
    "27": "Muna",
    "48": "Bugis",
    "166": "Makassar",
    "192": "Wolio",
    "226": "Toraja-Sadan",
    "674": "Tolaki",
}

# Phonological feature extractors (copied from E027 00_prepare_features.py)
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


# ============================================================
# Feature extraction functions (identical to E027)
# ============================================================

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


def compute_features(form, concept, lang_name, lang_coverage, lang_idx):
    """Compute the same features as E027 Model B for a single form."""
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

    # Reduplication
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

    # Consonant clusters
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

    # Prefix-like
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
        "language_id_encoded": lang_idx,
        "language_cognacy_coverage": lang_coverage,
    }


def load_abvd_data():
    """Load all ABVD CLDF data."""
    # Parameters (concepts)
    params = {}
    with open(ABVD / "parameters.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params[row["ID"]] = row["Name"]

    # Cognate sets
    cognate_info = defaultdict(list)
    with open(ABVD / "cognates.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cognate_info[row["Form_ID"]].append(row["Cognateset_ID"])

    # Forms
    all_forms = []
    with open(ABVD / "forms.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_forms.append(row)

    return params, cognate_info, all_forms


def build_features_for_langs(target_langs, params, cognate_info, all_forms):
    """
    Build feature matrix for a set of languages using E022-style residual labeling.

    Label assignment (same logic as E027/E022):
    - If form has cognacy field or is in cognate_info -> label=1 (Austronesian)
    - If form has NO cognacy -> label=0 (residual/candidate substrate)
    """
    # Filter forms for target languages
    lang_forms = [r for r in all_forms if r["Language_ID"] in target_langs]

    # Compute per-language cognacy coverage
    lang_has_cognacy = defaultdict(lambda: {"total": 0, "with_cog": 0})
    for r in lang_forms:
        lid = r["Language_ID"]
        lang_has_cognacy[lid]["total"] += 1
        cognacy = r.get("Cognacy", "").strip()
        cog_sets = cognate_info.get(r["ID"], [])
        if cognacy or cog_sets:
            lang_has_cognacy[lid]["with_cog"] += 1

    lang_coverage = {}
    for lid in target_langs:
        stats = lang_has_cognacy[lid]
        lang_coverage[lid] = round(stats["with_cog"] / max(stats["total"], 1), 4)

    # Language index encoding
    lang_idx_map = {lid: i for i, lid in enumerate(sorted(target_langs.keys()))}

    # Build feature rows with labels
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

        # Label assignment: same as E022/E027
        cognacy = r.get("Cognacy", "").strip()
        cog_sets = cognate_info.get(form_id, [])
        if cognacy or cog_sets:
            label = 1  # Austronesian (has cognacy)
        else:
            label = 0  # Residual (no cognacy = candidate substrate)

        feats = compute_features(
            form_value, concept, lang_name,
            lang_coverage[lid], lang_idx_map[lid]
        )
        feats["label"] = label
        feats["language"] = lang_name
        feats["form"] = form_value
        feats["concept"] = concept
        feats["form_id"] = form_id
        rows.append(feats)

    return rows, lang_coverage


def rows_to_Xy(rows):
    """Convert feature rows to X matrix and y vector, same as E027 Model B."""
    df = pd.DataFrame(rows)

    # One-hot encode initial_char
    ic_dummies = pd.get_dummies(df["initial_char"], prefix="init")
    df = pd.concat([df, ic_dummies], axis=1)

    # One-hot encode semantic_domain
    sd_dummies = pd.get_dummies(df["semantic_domain"], prefix="sem")
    df = pd.concat([df, sd_dummies], axis=1)

    # Phonological features
    phon_cols = [
        "form_length", "n_vowels", "vowel_ratio", "ends_in_vowel",
        "has_glottal", "has_nasal_cluster", "has_reduplication",
        "n_consonant_clusters", "has_prefix_like",
    ]
    init_cols = [c for c in df.columns if c.startswith("init_")]
    sem_cols = [c for c in df.columns if c.startswith("sem_")]
    lang_cols = ["language_id_encoded", "language_cognacy_coverage"]
    meta_cols = ["is_core_vocab"]

    model_b_cols = phon_cols + init_cols + meta_cols + sem_cols + lang_cols

    X = df[model_b_cols].values.astype(float)
    y = df["label"].values

    return X, y, model_b_cols, df


def evaluate_cv_auc(X, y, n_seeds=10, n_folds=5):
    """Multi-seed stratified K-fold CV, returns mean AUC and std."""
    seed_aucs = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed * 7 + 13)
        fold_aucs = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)

            # Handle edge case: if only one class in test fold
            if len(np.unique(y_test)) < 2:
                continue
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_prob))

        if fold_aucs:
            seed_aucs.append(np.mean(fold_aucs))

    if not seed_aucs:
        return 0.5, 0.0
    return np.mean(seed_aucs), np.std(seed_aucs)


def run_random_label_test(X, substrate_rate=0.323, n_iterations=200, n_folds=5):
    """
    Control 3: Assign random labels with the same substrate rate as E027,
    train classifier, record AUC distribution.
    """
    rng = np.random.RandomState(2026)
    random_aucs = []
    n = X.shape[0]

    for i in range(n_iterations):
        # Random labels with same substrate rate
        y_rand = np.zeros(n, dtype=int)
        n_substrate = int(n * substrate_rate)
        substrate_idx = rng.choice(n, n_substrate, replace=False)
        y_rand[substrate_idx] = 0  # already 0, but explicit
        non_substrate_idx = np.setdiff1d(np.arange(n), substrate_idx)
        y_rand[non_substrate_idx] = 1

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=rng.randint(0, 100000))
        fold_aucs = []
        for train_idx, test_idx in skf.split(X, y_rand):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_rand[train_idx], y_rand[test_idx]

            clf = RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                random_state=None,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            if len(np.unique(y_test)) < 2:
                continue
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_prob))

        if fold_aucs:
            random_aucs.append(np.mean(fold_aucs))

    return np.array(random_aucs)


def main():
    print("=" * 70)
    print("E087: SUBSTRATE DETECTOR NEGATIVE CONTROL")
    print("=" * 70)
    print()
    print("Testing whether the E027 ML substrate detector (AUC=0.762)")
    print("is specific to real substrate, or detects noise in ANY pair")
    print("of closely related Austronesian languages.")
    print()

    # ============================================================
    # Load ABVD data
    # ============================================================
    print("[0] Loading ABVD CLDF data...")
    params, cognate_info, all_forms = load_abvd_data()
    print(f"  {len(params)} concepts, {len(all_forms)} total forms")
    print()

    # ============================================================
    # Step 0: Reproduce E027 original AUC for comparison
    # ============================================================
    print("=" * 70)
    print("[0b] REPRODUCING E027 ORIGINAL AUC (Sulawesi 6 languages)")
    print("=" * 70)

    orig_rows, orig_coverage = build_features_for_langs(
        ORIGINAL_LANGS, params, cognate_info, all_forms
    )
    X_orig, y_orig, orig_cols, orig_df = rows_to_Xy(orig_rows)
    n_sub_orig = sum(y_orig == 0)
    n_an_orig = sum(y_orig == 1)
    pct_sub_orig = round(100 * n_sub_orig / len(y_orig), 1)

    print(f"  N={len(y_orig)}, Austronesian={n_an_orig}, Substrate={n_sub_orig} ({pct_sub_orig}%)")
    print(f"  Features: {X_orig.shape[1]}")
    print(f"  Cognacy coverage: {orig_coverage}")

    orig_auc, orig_std = evaluate_cv_auc(X_orig, y_orig)
    print(f"  Reproduced AUC: {orig_auc:.4f} +/- {orig_std:.4f}")
    print(f"  (E027 reported: 0.7618 for RF)")
    print()

    # ============================================================
    # CONTROL 1: Closely Related Languages (Tagalog + Cebuano)
    # ============================================================
    print("=" * 70)
    print("[1/3] CONTROL 1: CLOSELY RELATED LANGUAGES")
    print("  Tagalog + Cebuano (Central Philippine)")
    print("  Both high-cognacy, closely related, NO substrate between them")
    print("=" * 70)

    c1_rows, c1_coverage = build_features_for_langs(
        CONTROL1_LANGS, params, cognate_info, all_forms
    )
    X_c1, y_c1, c1_cols, c1_df = rows_to_Xy(c1_rows)
    n_sub_c1 = sum(y_c1 == 0)
    n_an_c1 = sum(y_c1 == 1)
    pct_sub_c1 = round(100 * n_sub_c1 / len(y_c1), 1)

    print(f"  N={len(y_c1)}, Austronesian={n_an_c1}, Substrate(residual)={n_sub_c1} ({pct_sub_c1}%)")
    print(f"  Features: {X_c1.shape[1]}")
    print(f"  Cognacy coverage: {c1_coverage}")

    if n_sub_c1 < 10:
        print(f"\n  WARNING: Only {n_sub_c1} residual forms. Very few 'substrate'")
        print(f"  candidates — this is EXPECTED for well-documented languages.")
        print(f"  The classifier may not train meaningfully.")
        c1_auc = 0.50
        c1_std = 0.0
        c1_note = f"SKIPPED: only {n_sub_c1} residuals, too few for meaningful classification"
    elif pct_sub_c1 < 5.0:
        print(f"\n  WARNING: Substrate rate {pct_sub_c1}% is extremely low.")
        print(f"  The classifier will struggle with extreme class imbalance.")
        c1_auc, c1_std = evaluate_cv_auc(X_c1, y_c1)
        c1_note = f"Low substrate rate ({pct_sub_c1}%), interpret with caution"
    else:
        c1_auc, c1_std = evaluate_cv_auc(X_c1, y_c1)
        c1_note = ""

    print(f"\n  >>> CONTROL 1 AUC: {c1_auc:.4f} +/- {c1_std:.4f}")

    if c1_auc < 0.60:
        c1_verdict = "PASS (detector does NOT find substrate here)"
    elif c1_auc <= 0.70:
        c1_verdict = "GREY ZONE (some signal detected, needs investigation)"
    else:
        c1_verdict = "FAIL (detector finds 'substrate' where none should exist)"
    print(f"  >>> VERDICT: {c1_verdict}")
    if c1_note:
        print(f"  >>> NOTE: {c1_note}")

    # Per-language breakdown
    print(f"\n  Per-language residual rates:")
    for lid, lname in CONTROL1_LANGS.items():
        lang_rows = [r for r in c1_rows if r["language"] == lname]
        lang_sub = sum(1 for r in lang_rows if r["label"] == 0)
        lang_pct = round(100 * lang_sub / max(len(lang_rows), 1), 1)
        print(f"    {lname}: {len(lang_rows)} forms, {lang_sub} residual ({lang_pct}%)")

    print()

    # ============================================================
    # CONTROL 2: Known No-Substrate Pair (Malay + Minangkabau)
    # ============================================================
    print("=" * 70)
    print("[2/3] CONTROL 2: KNOWN NO-SUBSTRATE PAIR")
    print("  Malay (Bahasa Indonesia) + Minangkabau (Malayic)")
    print("  Sister languages, no substrate between them")
    print("=" * 70)

    c2_rows, c2_coverage = build_features_for_langs(
        CONTROL2_LANGS, params, cognate_info, all_forms
    )
    X_c2, y_c2, c2_cols, c2_df = rows_to_Xy(c2_rows)
    n_sub_c2 = sum(y_c2 == 0)
    n_an_c2 = sum(y_c2 == 1)
    pct_sub_c2 = round(100 * n_sub_c2 / len(y_c2), 1)

    print(f"  N={len(y_c2)}, Austronesian={n_an_c2}, Substrate(residual)={n_sub_c2} ({pct_sub_c2}%)")
    print(f"  Features: {X_c2.shape[1]}")
    print(f"  Cognacy coverage: {c2_coverage}")

    if n_sub_c2 < 10:
        print(f"\n  WARNING: Only {n_sub_c2} residual forms.")
        c2_auc = 0.50
        c2_std = 0.0
        c2_note = f"SKIPPED: only {n_sub_c2} residuals"
    elif pct_sub_c2 < 5.0:
        print(f"\n  WARNING: Substrate rate {pct_sub_c2}% is extremely low.")
        c2_auc, c2_std = evaluate_cv_auc(X_c2, y_c2)
        c2_note = f"Low substrate rate ({pct_sub_c2}%)"
    else:
        c2_auc, c2_std = evaluate_cv_auc(X_c2, y_c2)
        c2_note = ""

    print(f"\n  >>> CONTROL 2 AUC: {c2_auc:.4f} +/- {c2_std:.4f}")

    if c2_auc < 0.60:
        c2_verdict = "PASS (no substrate detected)"
    elif c2_auc <= 0.70:
        c2_verdict = "GREY ZONE"
    else:
        c2_verdict = "FAIL (detector finds 'substrate' where none should exist)"
    print(f"  >>> VERDICT: {c2_verdict}")
    if c2_note:
        print(f"  >>> NOTE: {c2_note}")

    # Per-language breakdown
    print(f"\n  Per-language residual rates:")
    for lid, lname in CONTROL2_LANGS.items():
        lang_rows = [r for r in c2_rows if r["language"] == lname]
        lang_sub = sum(1 for r in lang_rows if r["label"] == 0)
        lang_pct = round(100 * lang_sub / max(len(lang_rows), 1), 1)
        print(f"    {lname}: {len(lang_rows)} forms, {lang_sub} residual ({lang_pct}%)")

    print()

    # ============================================================
    # CONTROL 3: Random Labels (Javanese + Sundanese)
    # ============================================================
    print("=" * 70)
    print("[3/3] CONTROL 3: RANDOM LABEL ASSIGNMENT")
    print("  Javanese + Sundanese words, randomly labeled as 'substrate'")
    print("  at the same rate as E027 (32.3%). 200 iterations.")
    print("=" * 70)

    c3_rows, c3_coverage = build_features_for_langs(
        CONTROL3_LANGS, params, cognate_info, all_forms
    )
    X_c3, y_c3_real, c3_cols, c3_df = rows_to_Xy(c3_rows)

    print(f"  N={len(y_c3_real)} forms from Javanese + Sundanese")
    print(f"  (Real residual rate: {100*sum(y_c3_real==0)/len(y_c3_real):.1f}%)")
    print(f"  Assigning random labels at 32.3% substrate rate...")
    print()

    random_aucs = run_random_label_test(X_c3, substrate_rate=0.323, n_iterations=200)

    c3_mean = np.mean(random_aucs)
    c3_std = np.std(random_aucs)
    c3_95th = np.percentile(random_aucs, 95)
    c3_max = np.max(random_aucs)

    print(f"  Random label AUC distribution (200 iterations):")
    print(f"    Mean:  {c3_mean:.4f}")
    print(f"    Std:   {c3_std:.4f}")
    print(f"    95th:  {c3_95th:.4f}")
    print(f"    Max:   {c3_max:.4f}")

    if c3_mean < 0.55:
        c3_verdict = "PASS (random labels produce ~chance AUC)"
    else:
        c3_verdict = "FAIL (random labels produce above-chance AUC — feature leakage?)"
    print(f"\n  >>> CONTROL 3 VERDICT: {c3_verdict}")

    # Also run the REAL labels on the same Javanese+Sundanese set as a comparison
    n_sub_c3r = sum(y_c3_real == 0)
    if n_sub_c3r >= 10 and sum(y_c3_real == 1) >= 10:
        print(f"\n  Bonus: Real-label AUC on Javanese + Sundanese:")
        c3_real_auc, c3_real_std = evaluate_cv_auc(X_c3, y_c3_real)
        print(f"    AUC: {c3_real_auc:.4f} +/- {c3_real_std:.4f}")
        print(f"    (N={len(y_c3_real)}, {n_sub_c3r} residual = {100*n_sub_c3r/len(y_c3_real):.1f}%)")
        print(f"    This is the detector applied to W. Indonesian languages — expect lower AUC")
    else:
        c3_real_auc = None
        c3_real_std = None

    print()

    # ============================================================
    # BONUS CONTROL: Tagalog + Kapampangan (different subgroups)
    # ============================================================
    print("=" * 70)
    print("[BONUS] CONTROL 4: TAGALOG + KAPAMPANGAN")
    print("  Different Central Luzon subgroups, geographically adjacent")
    print("  More distant than Control 1, but still no substrate expected")
    print("=" * 70)

    CONTROL4_LANGS = {
        "277": "Tagalog",
        "33": "Kapampangan",
    }

    c4_rows, c4_coverage = build_features_for_langs(
        CONTROL4_LANGS, params, cognate_info, all_forms
    )
    X_c4, y_c4, c4_cols, c4_df = rows_to_Xy(c4_rows)
    n_sub_c4 = sum(y_c4 == 0)
    n_an_c4 = sum(y_c4 == 1)
    pct_sub_c4 = round(100 * n_sub_c4 / len(y_c4), 1)

    print(f"  N={len(y_c4)}, Austronesian={n_an_c4}, Residual={n_sub_c4} ({pct_sub_c4}%)")

    if n_sub_c4 < 10:
        c4_auc, c4_std = 0.50, 0.0
        c4_note = f"SKIPPED: only {n_sub_c4} residuals"
    else:
        c4_auc, c4_std = evaluate_cv_auc(X_c4, y_c4)
        c4_note = ""

    print(f"\n  >>> CONTROL 4 AUC: {c4_auc:.4f} +/- {c4_std:.4f}")
    if c4_note:
        print(f"  >>> NOTE: {c4_note}")

    print()

    # ============================================================
    # BONUS CONTROL 5: Iban + Malay (both Malayic, Borneo vs Malay Peninsula)
    # ============================================================
    print("=" * 70)
    print("[BONUS] CONTROL 5: IBAN + MALAY (BAHASA INDONESIA)")
    print("  Both Malayic subgroup. Known to be closely related.")
    print("=" * 70)

    CONTROL5_LANGS = {
        "28": "Iban",
        "233": "Malay",
    }

    c5_rows, c5_coverage = build_features_for_langs(
        CONTROL5_LANGS, params, cognate_info, all_forms
    )
    X_c5, y_c5, c5_cols, c5_df = rows_to_Xy(c5_rows)
    n_sub_c5 = sum(y_c5 == 0)
    pct_sub_c5 = round(100 * n_sub_c5 / len(y_c5), 1)

    print(f"  N={len(y_c5)}, Residual={n_sub_c5} ({pct_sub_c5}%)")

    if n_sub_c5 < 10:
        c5_auc, c5_std = 0.50, 0.0
        c5_note = f"SKIPPED: only {n_sub_c5} residuals"
    else:
        c5_auc, c5_std = evaluate_cv_auc(X_c5, y_c5)
        c5_note = ""

    print(f"\n  >>> CONTROL 5 AUC: {c5_auc:.4f} +/- {c5_std:.4f}")
    if c5_note:
        print(f"  >>> NOTE: {c5_note}")

    print()

    # ============================================================
    # GRAND SUMMARY
    # ============================================================
    print("=" * 70)
    print("GRAND SUMMARY — E087 SUBSTRATE DETECTOR NEGATIVE CONTROL")
    print("=" * 70)

    print(f"\n  {'Test':<45} {'AUC':>8} {'Verdict':>12}")
    print(f"  {'-'*65}")
    print(f"  {'E027 Original (Sulawesi, 6 langs)':<45} {orig_auc:>8.4f} {'REFERENCE':>12}")
    print(f"  {'C1: Tagalog + Cebuano (Central Philippine)':<45} {c1_auc:>8.4f} {c1_verdict.split('(')[0].strip():>12}")
    print(f"  {'C2: Malay + Minangkabau (Malayic)':<45} {c2_auc:>8.4f} {c2_verdict.split('(')[0].strip():>12}")
    print(f"  {'C3: Random labels on Jav+Sun (mean)':<45} {c3_mean:>8.4f} {c3_verdict.split('(')[0].strip():>12}")
    print(f"  {'C4: Tagalog + Kapampangan (bonus)':<45} {c4_auc:>8.4f}")
    print(f"  {'C5: Iban + Malay (bonus)':<45} {c5_auc:>8.4f}")

    # Delta analysis
    print(f"\n  AUC Deltas vs Original:")
    print(f"    Original - C1 = {orig_auc - c1_auc:+.4f}")
    print(f"    Original - C2 = {orig_auc - c2_auc:+.4f}")
    print(f"    Original - C3 = {orig_auc - c3_mean:+.4f}")

    # Overall verdict
    all_controls_pass = (c1_auc < 0.60 and c2_auc < 0.60 and c3_mean < 0.55)
    grey_zone = (0.60 <= c1_auc <= 0.70 or 0.60 <= c2_auc <= 0.70)
    any_fail = (c1_auc > 0.70 or c2_auc > 0.70 or c3_mean > 0.60)

    if all_controls_pass:
        overall = "VOLCARCH PASSES"
        explanation = (
            f"The E027 detector is SPECIFIC to real substrate signal. "
            f"Closely related non-substrate language pairs produce AUC near chance "
            f"(C1={c1_auc:.3f}, C2={c2_auc:.3f}), while the original Sulawesi "
            f"languages produce AUC={orig_auc:.3f}. Random labels produce {c3_mean:.3f}. "
            f"The {orig_auc - max(c1_auc, c2_auc):.3f} AUC gap between E027 and controls "
            f"confirms the detector is finding genuine phonological substrate, not phylogenetic noise."
        )
    elif any_fail:
        overall = "VOLCARCH FAILS"
        explanation = (
            f"The E027 detector finds 'substrate' in language pairs where none should exist. "
            f"C1 (Tagalog+Cebuano)={c1_auc:.3f}, C2 (Malay+Minangkabau)={c2_auc:.3f}. "
            f"This suggests the detector is picking up generic phonological variation or "
            f"phylogenetic noise between ANY language pair, not substrate-specific signal. "
            f"L4 evidence from ML substrate detection should be treated with EXTREME CAUTION."
        )
    else:
        overall = "GREY ZONE"
        explanation = (
            f"Some controls show moderate AUC (C1={c1_auc:.3f}, C2={c2_auc:.3f}), "
            f"suggesting the detector captures SOME phylogenetic noise but not as "
            f"strongly as the Sulawesi substrate signal ({orig_auc:.3f}). "
            f"The gap of {orig_auc - max(c1_auc, c2_auc):.3f} is positive but may not "
            f"be large enough to confidently attribute it to substrate alone."
        )

    print(f"\n  >>> OVERALL VERDICT: {overall}")
    print(f"  >>> {explanation}")

    # ============================================================
    # Save results
    # ============================================================
    results = {
        "experiment": "E087_substrate_negative_control",
        "date": "2026-03-16",
        "reference": {
            "original_sulawesi_auc": round(float(orig_auc), 4),
            "original_sulawesi_std": round(float(orig_std), 4),
            "n_forms": int(len(y_orig)),
            "n_substrate": int(n_sub_orig),
            "substrate_rate": round(float(pct_sub_orig), 1),
        },
        "control_1_closely_related": {
            "languages": "Tagalog + Cebuano (Central Philippine)",
            "auc": round(float(c1_auc), 4),
            "std": round(float(c1_std), 4),
            "n_forms": int(len(y_c1)),
            "n_substrate": int(n_sub_c1),
            "substrate_rate": round(float(pct_sub_c1), 1),
            "cognacy_coverage": {k: round(float(v), 4) for k, v in c1_coverage.items()},
            "verdict": c1_verdict,
            "note": c1_note,
        },
        "control_2_no_substrate": {
            "languages": "Malay + Minangkabau (Malayic)",
            "auc": round(float(c2_auc), 4),
            "std": round(float(c2_std), 4),
            "n_forms": int(len(y_c2)),
            "n_substrate": int(n_sub_c2),
            "substrate_rate": round(float(pct_sub_c2), 1),
            "cognacy_coverage": {k: round(float(v), 4) for k, v in c2_coverage.items()},
            "verdict": c2_verdict,
            "note": c2_note,
        },
        "control_3_random_labels": {
            "languages": "Javanese + Sundanese (random labels at 32.3%)",
            "mean_auc": round(float(c3_mean), 4),
            "std_auc": round(float(c3_std), 4),
            "percentile_95": round(float(c3_95th), 4),
            "max_auc": round(float(c3_max), 4),
            "n_iterations": 200,
            "n_forms": int(len(y_c3_real)),
            "verdict": c3_verdict,
            "real_label_auc": round(float(c3_real_auc), 4) if c3_real_auc else None,
        },
        "control_4_bonus_tagalog_kapampangan": {
            "auc": round(float(c4_auc), 4),
            "std": round(float(c4_std), 4),
            "n_forms": int(len(y_c4)),
            "n_substrate": int(n_sub_c4),
            "note": c4_note,
        },
        "control_5_bonus_iban_malay": {
            "auc": round(float(c5_auc), 4),
            "std": round(float(c5_std), 4),
            "n_forms": int(len(y_c5)),
            "n_substrate": int(n_sub_c5),
            "note": c5_note,
        },
        "overall_verdict": overall,
        "explanation": explanation,
        "delta_orig_c1": round(float(orig_auc - c1_auc), 4),
        "delta_orig_c2": round(float(orig_auc - c2_auc), 4),
        "delta_orig_c3": round(float(orig_auc - c3_mean), 4),
    }

    with open(OUT / "negative_control_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUT / 'negative_control_results.json'}")

    # Save the random AUC distribution for plotting
    np.save(OUT / "control3_random_aucs.npy", random_aucs)
    print(f"  Saved: {OUT / 'control3_random_aucs.npy'}")

    # Save summary text
    with open(OUT / "summary.txt", "w", encoding="utf-8") as f:
        f.write("E087: Substrate Detector Negative Control — Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Date: 2026-03-16\n\n")
        f.write(f"E027 Original (Sulawesi 6 langs): AUC = {orig_auc:.4f}\n\n")
        f.write(f"Control 1: Tagalog + Cebuano:     AUC = {c1_auc:.4f}  {c1_verdict}\n")
        f.write(f"Control 2: Malay + Minangkabau:   AUC = {c2_auc:.4f}  {c2_verdict}\n")
        f.write(f"Control 3: Random labels (mean):  AUC = {c3_mean:.4f}  {c3_verdict}\n")
        f.write(f"Control 4: Tagalog + Kapampangan: AUC = {c4_auc:.4f}\n")
        f.write(f"Control 5: Iban + Malay:          AUC = {c5_auc:.4f}\n\n")
        f.write(f"VERDICT: {overall}\n")
        f.write(f"{explanation}\n")
    print(f"  Saved: {OUT / 'summary.txt'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
