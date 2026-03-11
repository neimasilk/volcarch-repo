"""
E027 Script 03: Expansion Validation
======================================
Apply the trained Model B (phonological-only XGBoost) to ADDITIONAL languages
beyond the original 6 Sulawesi languages. Tests whether the substrate detection
signal generalizes geographically and genetically.

Languages:
  - Sulawesi expansion: 8 additional Sulawesi languages
  - Western Indonesian comparison: 6 languages (Javanese, Balinese, etc.)
  - Tests geographic/genetic patterning of predicted substrate rates

Output:
  results/expansion_summary.csv
  results/expansion_report.txt
  results/expansion_barplot.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.metrics import roc_auc_score
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost not available. Using RandomForest.")

REPO = Path(__file__).parent.parent.parent
ABVD = REPO / "experiments" / "E022_linguistic_subtraction" / "data" / "abvd" / "cldf"
DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# ============================================================
# EXPANSION LANGUAGE SELECTION
# ============================================================
# Group A: Additional Sulawesi languages (>190 forms, not in original 6)
# Group B: Western Indonesian comparison languages
# Group C: Eastern Indonesian / NTT (geographic control)

EXPANSION_LANGS = {
    # Group A: Sulawesi expansion
    "2":    {"name": "Banggai",           "group": "Sulawesi",  "forms_expected": 208},
    "5":    {"name": "Bantik",            "group": "Sulawesi",  "forms_expected": 233},
    "12":   {"name": "Gorontalo",         "group": "Sulawesi",  "forms_expected": 215},
    "240":  {"name": "Bol.Mongondow",     "group": "Sulawesi",  "forms_expected": 228},
    "884":  {"name": "Kulisusu",          "group": "Sulawesi",  "forms_expected": 222},
    "748":  {"name": "Totoli",            "group": "Sulawesi",  "forms_expected": 227},
    "999":  {"name": "Uma",               "group": "Sulawesi",  "forms_expected": 242},
    "968":  {"name": "Kambowa",           "group": "Sulawesi",  "forms_expected": 211},
    # Group B: Western Indonesian comparison
    "1":    {"name": "Balinese",          "group": "W.Indonesian", "forms_expected": 294},
    "20":   {"name": "Javanese",          "group": "W.Indonesian", "forms_expected": 217},
    "233":  {"name": "Malay",             "group": "W.Indonesian", "forms_expected": 236},
    "284":  {"name": "Sundanese",         "group": "W.Indonesian", "forms_expected": 217},
    "185":  {"name": "Sasak",             "group": "W.Indonesian", "forms_expected": 267},
    "648":  {"name": "Acehnese",          "group": "W.Indonesian", "forms_expected": 237},
    # Group C: Eastern Indonesian (geographic control)
    "14":   {"name": "Bima",              "group": "E.Indonesian", "forms_expected": 246},
    "84":   {"name": "Manggarai",         "group": "E.Indonesian", "forms_expected": 235},
}

# Original 6 language IDs (for exclusion / reference)
ORIGINAL_LANG_IDS = {"27", "48", "166", "192", "226", "674"}

# ============================================================
# Feature engineering functions — COPIED from 00_prepare_features.py
# ============================================================

VOWELS = set("aeiouəɛɨɔæøüöäåãẽĩõũâêîôûàèìòùáéíóú")
AUSTRONESIAN_PREFIXES = ("ma-", "me-", "mo-", "pa-", "ka-", "ta-", "na-", "po-",
                         "ma", "me", "mo", "pa", "ka", "ta", "na", "po",
                         "maŋ", "meŋ", "moŋ", "paŋ", "aŋ", "mak-", "mat-")
NASAL_CLUSTERS = ("ng", "mb", "nd", "nj", "mp", "nk", "ŋg", "ŋk",
                  "nc", "nt", "ŋ")

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


def classify_domain(concept):
    for domain, concepts in SEMANTIC_DOMAINS.items():
        if concept in concepts:
            return domain
    return "OTHER"


def is_core_vocab(concept):
    return 1 if concept in SWADESH_100 else 0


def compute_form_length(form):
    return len(form)


def count_vowels(form):
    return sum(1 for c in form.lower() if c in VOWELS)


def vowel_ratio(form):
    if len(form) == 0:
        return 0.0
    return round(count_vowels(form) / len(form), 4)


def ends_in_vowel(form):
    if not form:
        return 0
    return 1 if form[-1].lower() in VOWELS else 0


def get_initial_class(form):
    if not form:
        return "other"
    c = form[0].lower()
    if c in ('m', 'a', 'b', 't', 'k', 'p', 's'):
        return c
    return "other"


def has_glottal(form):
    return 1 if ("ʔ" in form or "'" in form) else 0


def has_nasal_cluster(form):
    fl = form.lower()
    for nc in NASAL_CLUSTERS:
        if nc in fl:
            return 1
    return 0


def has_reduplication(form):
    if "-" in form:
        return 1
    fl = form.lower()
    for plen in (2, 3):
        for i in range(len(fl) - plen * 2 + 1):
            chunk = fl[i:i+plen]
            if chunk == fl[i+plen:i+plen*2]:
                return 1
    return 0


def count_consonant_clusters(form):
    fl = form.lower()
    count = 0
    in_cluster = False
    consec = 0
    for c in fl:
        if c not in VOWELS and c.isalpha():
            consec += 1
            if consec == 2 and not in_cluster:
                count += 1
                in_cluster = True
        else:
            consec = 0
            in_cluster = False
    return count


def has_prefix_like(form):
    fl = form.lower()
    for prefix in AUSTRONESIAN_PREFIXES:
        if fl.startswith(prefix):
            return 1
    return 0


def clean_form(raw):
    s = raw.strip()
    s = re.sub(r'\[.*?\]\s*', '', s)
    s = s.strip(" -,;.")
    return s


# ============================================================
# Model B feature column order — MUST match training
# ============================================================
PHON_FEATURES = [
    "form_length", "n_vowels", "vowel_ratio", "ends_in_vowel",
    "has_glottal", "has_nasal_cluster", "has_reduplication",
    "n_consonant_clusters", "has_prefix_like",
]
SEMANTIC_FEATURES_LIST = ["is_core_vocab"]
LANG_FEATURES = ["language_id_encoded", "language_cognacy_coverage"]


def main():
    print("=" * 70)
    print("E027 Script 03: Expansion Validation")
    print("=" * 70)

    # ============================================================
    # Step 1: Load and retrain Model B on original 6 languages
    # ============================================================
    print("\n[1/7] Loading original training data and retraining Model B...")

    train_df = pd.read_csv(DATA / "features_matrix.csv", encoding="utf-8")
    print(f"  Original training data: {len(train_df)} forms from {train_df['language'].nunique()} languages")

    # One-hot encode initial_char and semantic_domain — record the columns
    ic_dummies = pd.get_dummies(train_df["initial_char"], prefix="init")
    sd_dummies = pd.get_dummies(train_df["semantic_domain"], prefix="sem")
    train_df = pd.concat([train_df, ic_dummies, sd_dummies], axis=1)

    init_cols = sorted([c for c in train_df.columns if c.startswith("init_")])
    sem_cols = sorted([c for c in train_df.columns if c.startswith("sem_")])

    model_b_cols = PHON_FEATURES + init_cols + SEMANTIC_FEATURES_LIST + sem_cols + LANG_FEATURES
    print(f"  Model B features ({len(model_b_cols)}): {model_b_cols}")

    X_train = train_df[model_b_cols].values.astype(float)
    y_train = train_df["label"].values

    # Train Model B
    if HAS_XGB:
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            reg_lambda=1.0, eval_metric="logloss", random_state=42,
            use_label_encoder=False, verbosity=0,
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )

    clf.fit(X_train, y_train)
    print(f"  Model B trained ({type(clf).__name__})")

    # Verify on training data
    train_probs = clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_probs)
    print(f"  Training AUC (sanity check): {train_auc:.4f}")

    # ============================================================
    # Step 2: Load ABVD data for expansion languages
    # ============================================================
    print("\n[2/7] Loading ABVD data for expansion languages...")

    # Load parameters
    params = {}
    with open(ABVD / "parameters.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params[row["ID"]] = row["Name"]

    # Load cognate info
    cognate_info = defaultdict(list)
    cognate_set_sizes = defaultdict(int)
    with open(ABVD / "cognates.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            form_id = row["Form_ID"]
            cog_id = row["Cognateset_ID"]
            cognate_info[form_id].append(cog_id)
            cognate_set_sizes[cog_id] += 1

    # Load forms for expansion languages
    expansion_lang_ids = set(EXPANSION_LANGS.keys())
    exp_forms = []
    with open(ABVD / "forms.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Language_ID"] in expansion_lang_ids:
                exp_forms.append(row)

    print(f"  Loaded {len(exp_forms)} forms across {len(expansion_lang_ids)} languages")

    # Count forms per language
    forms_per_lang = defaultdict(int)
    for row in exp_forms:
        lid = row["Language_ID"]
        forms_per_lang[lid] += 1

    print(f"\n  {'ID':<6} {'Name':<20} {'Forms':>6}")
    print("  " + "-" * 35)
    for lid in sorted(expansion_lang_ids, key=lambda x: int(x)):
        info = EXPANSION_LANGS[lid]
        print(f"  {lid:<6} {info['name']:<20} {forms_per_lang.get(lid, 0):>6}")

    # ============================================================
    # Step 3: Assign labels (cognacy-based) and compute features
    # ============================================================
    print("\n[3/7] Assigning labels and computing features...")

    # For expansion languages, we use cognacy as ground truth:
    # - Has cognacy in ABVD (Cognacy field or cognates.csv) → Austronesian (label=1)
    # - No cognacy → Candidate substrate (label=0)

    lang_results = {}  # lid -> {stats}

    all_expansion_rows = []

    for lid in sorted(expansion_lang_ids, key=lambda x: int(x)):
        info = EXPANSION_LANGS[lid]
        lang_name = info["name"]
        lang_forms = [r for r in exp_forms if r["Language_ID"] == lid]

        n_cognate = 0
        n_residual = 0
        feature_rows = []

        for row in lang_forms:
            form_id = row["ID"]
            concept = params.get(row["Parameter_ID"], "?")
            raw_value = row.get("Value", "") or row.get("Form", "")
            form_value = clean_form(raw_value)

            if not form_value or len(form_value) < 1:
                continue

            # Label: check cognacy
            cognacy = row.get("Cognacy", "").strip()
            cog_sets = cognate_info.get(form_id, [])

            if cognacy or cog_sets:
                label = 1  # Austronesian
                n_cognate += 1
            else:
                label = 0  # Candidate substrate
                n_residual += 1

            # Compute features
            fl = compute_form_length(form_value)
            nv = count_vowels(form_value)
            vr = vowel_ratio(form_value)
            eiv = ends_in_vowel(form_value)
            ic = get_initial_class(form_value)
            hg = has_glottal(form_value)
            hnc = has_nasal_cluster(form_value)
            hr = has_reduplication(form_value)
            ncc = count_consonant_clusters(form_value)
            hpl = has_prefix_like(form_value)
            sd = classify_domain(concept)
            icv = is_core_vocab(concept)

            feature_rows.append({
                "form_id": form_id,
                "language": lang_name,
                "language_id": lid,
                "group": info["group"],
                "concept": concept,
                "form": form_value,
                "label": label,
                "form_length": fl,
                "n_vowels": nv,
                "vowel_ratio": round(vr, 4),
                "ends_in_vowel": eiv,
                "initial_char": ic,
                "has_glottal": hg,
                "has_nasal_cluster": hnc,
                "has_reduplication": hr,
                "n_consonant_clusters": ncc,
                "has_prefix_like": hpl,
                "semantic_domain": sd,
                "is_core_vocab": icv,
            })

        total = n_cognate + n_residual
        residual_rate = n_residual / total if total > 0 else 0
        cognacy_coverage = n_cognate / total if total > 0 else 0

        lang_results[lid] = {
            "name": lang_name,
            "group": info["group"],
            "n_total": total,
            "n_cognate": n_cognate,
            "n_residual": n_residual,
            "residual_rate": round(residual_rate, 4),
            "cognacy_coverage": round(cognacy_coverage, 4),
            "feature_rows": feature_rows,
        }

        all_expansion_rows.extend(feature_rows)

    print(f"  Total expansion forms: {len(all_expansion_rows)}")

    # ============================================================
    # Step 4: Build feature matrices for prediction
    # ============================================================
    print("\n[4/7] Building feature matrices for expansion languages...")

    # We need to match the EXACT one-hot columns from training
    # Training init_cols and sem_cols were derived from the training data
    # We need to create the same columns for expansion data

    exp_df = pd.DataFrame(all_expansion_rows)

    # One-hot encode initial_char — but ensure same columns as training
    ic_dummies_exp = pd.get_dummies(exp_df["initial_char"], prefix="init")
    sd_dummies_exp = pd.get_dummies(exp_df["semantic_domain"], prefix="sem")

    # Add missing columns (from training), drop extra columns (not in training)
    for col in init_cols:
        if col not in ic_dummies_exp.columns:
            ic_dummies_exp[col] = 0
    for col in sem_cols:
        if col not in sd_dummies_exp.columns:
            sd_dummies_exp[col] = 0

    # Keep only training columns
    ic_dummies_exp = ic_dummies_exp[[c for c in init_cols if c in ic_dummies_exp.columns]]
    sd_dummies_exp = sd_dummies_exp[[c for c in sem_cols if c in sd_dummies_exp.columns]]

    exp_df = pd.concat([exp_df.reset_index(drop=True),
                        ic_dummies_exp.reset_index(drop=True),
                        sd_dummies_exp.reset_index(drop=True)], axis=1)

    # For expansion languages, we need language_id_encoded and language_cognacy_coverage
    # These are language-level features. For new languages, we use:
    # - language_id_encoded: set to a neutral value (mean of training languages = 2.5)
    #   This is important: we DON'T want the model to rely on language identity
    #   since these are new languages not seen during training.
    # - language_cognacy_coverage: computed from the actual cognacy data for each language

    # Compute per-language coverage and add as column
    exp_df["language_cognacy_coverage"] = 0.0
    exp_df["language_id_encoded"] = 3  # neutral (median of 0-5 range from training)

    for lid, lr in lang_results.items():
        mask = exp_df["language_id"] == lid
        exp_df.loc[mask, "language_cognacy_coverage"] = lr["cognacy_coverage"]

    # Ensure all model_b_cols exist
    for col in model_b_cols:
        if col not in exp_df.columns:
            exp_df[col] = 0

    X_exp = exp_df[model_b_cols].values.astype(float)
    y_exp = exp_df["label"].values

    print(f"  Expansion feature matrix: {X_exp.shape}")
    print(f"  Labels: {sum(y_exp == 1)} Austronesian, {sum(y_exp == 0)} residual")

    # ============================================================
    # Step 5: Predict substrate probabilities
    # ============================================================
    print("\n[5/7] Predicting substrate probabilities...")

    probs = clf.predict_proba(X_exp)
    prob_substrate = probs[:, 0]  # P(substrate)
    predictions = clf.predict(X_exp)

    exp_df["p_substrate"] = prob_substrate
    exp_df["predicted_label"] = predictions

    # ============================================================
    # Step 6: Analyze results per language
    # ============================================================
    print("\n[6/7] Analyzing results per language...")

    summary_rows = []

    print(f"\n  {'Language':<18} {'Group':<14} {'N':>5} {'Rule%':>7} {'ML%':>7} {'AUC':>7} {'MeanP':>7}")
    print("  " + "-" * 70)

    for lid in sorted(lang_results.keys(), key=lambda x: int(x)):
        lr = lang_results[lid]
        mask = exp_df["language_id"] == lid
        lang_df = exp_df[mask]

        if len(lang_df) == 0:
            continue

        y_true = lang_df["label"].values
        p_sub = lang_df["p_substrate"].values
        preds = lang_df["predicted_label"].values

        # Rule-based residual rate
        rule_residual_rate = lr["residual_rate"]

        # ML predicted substrate rate (fraction predicted as label=0)
        ml_substrate_rate = sum(preds == 0) / len(preds) if len(preds) > 0 else 0

        # Mean P(substrate) for all forms
        mean_p_substrate = np.mean(p_sub)

        # AUC (only if both classes present)
        auc = None
        if len(set(y_true)) >= 2:
            try:
                # Note: p_substrate is P(class=0), but AUC needs P(positive class)
                # Our label convention: 1=Austronesian, 0=substrate
                # For AUC, predict P(Austronesian) = 1 - P(substrate) vs label
                auc = roc_auc_score(y_true, 1 - p_sub)
            except Exception:
                auc = None

        auc_str = f"{auc:.4f}" if auc is not None else "  N/A"

        print(f"  {lr['name']:<18} {lr['group']:<14} {lr['n_total']:>5} "
              f"{rule_residual_rate*100:>6.1f}% {ml_substrate_rate*100:>6.1f}% "
              f"{auc_str:>7} {mean_p_substrate:>6.3f}")

        summary_rows.append({
            "language": lr["name"],
            "language_id": lid,
            "group": lr["group"],
            "n_forms": lr["n_total"],
            "n_cognate": lr["n_cognate"],
            "n_residual": lr["n_residual"],
            "rule_residual_rate": round(rule_residual_rate, 4),
            "ml_substrate_rate": round(ml_substrate_rate, 4),
            "auc": round(auc, 4) if auc is not None else None,
            "mean_p_substrate": round(mean_p_substrate, 4),
        })

    # Add original 6 languages for comparison
    print(f"\n  --- Original 6 languages (from E027 training) ---")
    original_names = {
        "27": "Muna", "48": "Bugis", "166": "Makassar",
        "192": "Wolio", "226": "Toraja-Sadan", "674": "Tolaki",
    }
    original_residual_rates = {
        "Muna": 0.155, "Bugis": 0.223, "Makassar": 0.221,
        "Wolio": 0.253, "Toraja-Sadan": 0.184, "Tolaki": 0.641,
    }

    for lid, name in original_names.items():
        mask = train_df["language"] == name
        lang_train = train_df[mask]
        y_t = lang_train["label"].values
        X_t = lang_train[model_b_cols].values.astype(float)
        probs_t = clf.predict_proba(X_t)
        preds_t = clf.predict(X_t)
        p_sub_t = probs_t[:, 0]

        rule_rr = original_residual_rates.get(name, 0)
        ml_rr = sum(preds_t == 0) / len(preds_t) if len(preds_t) > 0 else 0
        mean_p_t = np.mean(p_sub_t)

        auc_t = None
        if len(set(y_t)) >= 2:
            try:
                auc_t = roc_auc_score(y_t, 1 - p_sub_t)
            except Exception:
                pass

        auc_str = f"{auc_t:.4f}" if auc_t is not None else "  N/A"
        print(f"  {name:<18} {'Original':<14} {len(lang_train):>5} "
              f"{rule_rr*100:>6.1f}% {ml_rr*100:>6.1f}% "
              f"{auc_str:>7} {mean_p_t:>6.3f}")

        summary_rows.append({
            "language": name,
            "language_id": lid,
            "group": "Original",
            "n_forms": len(lang_train),
            "n_cognate": int(sum(y_t == 1)),
            "n_residual": int(sum(y_t == 0)),
            "rule_residual_rate": round(rule_rr, 4),
            "ml_substrate_rate": round(ml_rr, 4),
            "auc": round(auc_t, 4) if auc_t is not None else None,
            "mean_p_substrate": round(mean_p_t, 4),
        })

    # ============================================================
    # Step 7: Group-level analysis and GO/NO-GO
    # ============================================================
    print("\n[7/7] Group-level analysis...")

    # Group means
    group_stats = defaultdict(lambda: {"ml_rates": [], "rule_rates": [], "aucs": [], "mean_ps": []})
    for sr in summary_rows:
        g = sr["group"]
        group_stats[g]["ml_rates"].append(sr["ml_substrate_rate"])
        group_stats[g]["rule_rates"].append(sr["rule_residual_rate"])
        if sr["auc"] is not None:
            group_stats[g]["aucs"].append(sr["auc"])
        group_stats[g]["mean_ps"].append(sr["mean_p_substrate"])

    print(f"\n  {'Group':<18} {'N langs':>7} {'Mean Rule%':>10} {'Mean ML%':>10} {'Mean AUC':>10} {'Mean P(sub)':>12}")
    print("  " + "-" * 70)
    for group in ["Original", "Sulawesi", "W.Indonesian", "E.Indonesian"]:
        gs = group_stats[group]
        n = len(gs["ml_rates"])
        if n == 0:
            continue
        mean_rule = np.mean(gs["rule_rates"])
        mean_ml = np.mean(gs["ml_rates"])
        mean_auc = np.mean(gs["aucs"]) if gs["aucs"] else None
        mean_p = np.mean(gs["mean_ps"])
        auc_str = f"{mean_auc:.4f}" if mean_auc is not None else "N/A"
        print(f"  {group:<18} {n:>7} {mean_rule*100:>9.1f}% {mean_ml*100:>9.1f}% {auc_str:>10} {mean_p:>11.4f}")

    # ============================================================
    # Analysis: Do patterns make linguistic sense?
    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: GEOGRAPHIC/GENETIC PATTERNING")
    print("=" * 70)

    sulawesi_ml = group_stats["Sulawesi"]["ml_rates"]
    western_ml = group_stats["W.Indonesian"]["ml_rates"]
    eastern_ml = group_stats["E.Indonesian"]["ml_rates"]
    original_ml = group_stats["Original"]["ml_rates"]

    sulawesi_p = group_stats["Sulawesi"]["mean_ps"]
    western_p = group_stats["W.Indonesian"]["mean_ps"]
    eastern_p = group_stats["E.Indonesian"]["mean_ps"]

    print(f"\n  Sulawesi expansion: mean ML substrate rate = {np.mean(sulawesi_ml)*100:.1f}% "
          f"(range: {np.min(sulawesi_ml)*100:.1f}%-{np.max(sulawesi_ml)*100:.1f}%)")
    print(f"  Original 6 Sulawesi: mean ML substrate rate = {np.mean(original_ml)*100:.1f}% "
          f"(range: {np.min(original_ml)*100:.1f}%-{np.max(original_ml)*100:.1f}%)")
    print(f"  Western Indonesian:  mean ML substrate rate = {np.mean(western_ml)*100:.1f}% "
          f"(range: {np.min(western_ml)*100:.1f}%-{np.max(western_ml)*100:.1f}%)")
    if eastern_ml:
        print(f"  Eastern Indonesian:  mean ML substrate rate = {np.mean(eastern_ml)*100:.1f}% "
              f"(range: {np.min(eastern_ml)*100:.1f}%-{np.max(eastern_ml)*100:.1f}%)")

    # Test: Is there a difference between groups?
    # Simple comparison: do Sulawesi languages have higher predicted substrate rates
    # than Western Indonesian?
    print(f"\n  Mean P(substrate) comparison:")
    print(f"    Sulawesi expansion: {np.mean(sulawesi_p):.4f}")
    print(f"    Western Indonesian: {np.mean(western_p):.4f}")
    if eastern_p:
        print(f"    Eastern Indonesian: {np.mean(eastern_p):.4f}")

    diff = np.mean(sulawesi_p) - np.mean(western_p)
    print(f"\n    Delta (Sulawesi - Western): {diff:+.4f}")

    # Interpretation
    print("\n  INTERPRETATION:")
    if diff > 0.03:
        print("    Sulawesi languages show HIGHER predicted substrate rates than Western")
        print("    Indonesian. This is consistent with Sulawesi having more pre-Austronesian")
        print("    substrate retention — possibly due to geographic isolation, later")
        print("    Austronesian arrival, or different substrate populations.")
        pattern_detected = True
    elif diff < -0.03:
        print("    Western Indonesian languages show HIGHER predicted substrate rates.")
        print("    This is UNEXPECTED but could reflect different orthographic conventions")
        print("    or contact-induced borrowing patterns in Western languages.")
        pattern_detected = True
    else:
        print("    No significant difference between Sulawesi and Western Indonesian.")
        print("    The phonological fingerprint may not differentiate by geography.")
        pattern_detected = False

    # AUC check: does the model generalize to new languages?
    expansion_aucs = group_stats["Sulawesi"]["aucs"] + group_stats["W.Indonesian"]["aucs"] + group_stats["E.Indonesian"]["aucs"]
    if expansion_aucs:
        mean_exp_auc = np.mean(expansion_aucs)
        print(f"\n  Mean AUC across expansion languages: {mean_exp_auc:.4f}")
        if mean_exp_auc >= 0.60:
            print("    Model generalizes to unseen languages (AUC >= 0.60)")
            generalizes = True
        else:
            print("    Model does NOT generalize well to unseen languages (AUC < 0.60)")
            generalizes = False
    else:
        generalizes = False
        mean_exp_auc = None

    # ============================================================
    # GO/NO-GO for expansion
    # ============================================================
    print("\n" + "=" * 70)
    print("GO/NO-GO VERDICT")
    print("=" * 70)

    go_criteria = []
    go_criteria.append(("Geographic patterning detected", pattern_detected))
    go_criteria.append(("Model generalizes (AUC >= 0.60)", generalizes))
    go_criteria.append(("Sulawesi expansion consistent with original",
                        abs(np.mean(sulawesi_ml) - np.mean(original_ml)) < 0.25))

    n_met = sum(1 for _, v in go_criteria if v)

    print()
    for criterion, met in go_criteria:
        status = "PASS" if met else "FAIL"
        print(f"  [{status}] {criterion}")

    if n_met >= 2:
        verdict = "GO"
        explanation = ("Model predictions show linguistically plausible patterns "
                       "across new languages. Expansion validates the substrate "
                       "detection approach.")
    elif n_met == 1:
        verdict = "CONDITIONAL GO"
        explanation = ("Partial validation. Some patterns detected but model "
                       "generalization is limited. Use with caution.")
    else:
        verdict = "NO-GO"
        explanation = ("Predictions show no geographic/genetic patterning. "
                       "The phonological fingerprint may be dataset-specific.")

    print(f"\n  >>> VERDICT: {verdict}")
    print(f"  >>> {explanation}")

    # ============================================================
    # Save results
    # ============================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save summary CSV
    summary_path = OUT / "expansion_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved: {summary_path}")

    # Save verdict
    verdict_exp = {
        "verdict": verdict,
        "explanation": explanation,
        "criteria": {c: v for c, v in go_criteria},
        "n_expansion_languages": len(EXPANSION_LANGS),
        "group_means": {
            g: {
                "mean_ml_substrate_rate": round(np.mean(gs["ml_rates"]), 4),
                "mean_rule_residual_rate": round(np.mean(gs["rule_rates"]), 4),
                "mean_auc": round(np.mean(gs["aucs"]), 4) if gs["aucs"] else None,
                "mean_p_substrate": round(np.mean(gs["mean_ps"]), 4),
                "n_languages": len(gs["ml_rates"]),
            }
            for g, gs in group_stats.items()
        },
        "mean_expansion_auc": round(mean_exp_auc, 4) if mean_exp_auc is not None else None,
    }
    with open(OUT / "expansion_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_exp, f, indent=2, default=str)
    print(f"  Saved: {OUT / 'expansion_verdict.json'}")

    # ============================================================
    # Generate barplot
    # ============================================================
    print("\n  Generating expansion barplot...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Sort summary rows by group then by ML rate
    group_order = {"Original": 0, "Sulawesi": 1, "E.Indonesian": 2, "W.Indonesian": 3}
    sorted_rows = sorted(summary_rows, key=lambda x: (group_order.get(x["group"], 99), -x["ml_substrate_rate"]))

    names = [sr["language"] for sr in sorted_rows]
    ml_rates = [sr["ml_substrate_rate"] * 100 for sr in sorted_rows]
    rule_rates = [sr["rule_residual_rate"] * 100 for sr in sorted_rows]
    groups = [sr["group"] for sr in sorted_rows]

    # Color by group
    color_map = {
        "Original": "#1f77b4",
        "Sulawesi": "#2ca02c",
        "W.Indonesian": "#d62728",
        "E.Indonesian": "#ff7f0e",
    }
    colors = [color_map.get(g, "#999999") for g in groups]

    # Panel 1: ML predicted substrate rate
    ax = axes[0]
    bars = ax.barh(range(len(names)), ml_rates, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("ML Predicted Substrate Rate (%)")
    ax.set_title("Model B Predicted Substrate Rate\n(Phonological Features Only)")
    ax.invert_yaxis()
    ax.axvline(x=np.mean([sr["ml_substrate_rate"]*100 for sr in sorted_rows if sr["group"] == "Original"]),
               color="#1f77b4", linestyle="--", alpha=0.5, label="Original mean")

    # Panel 2: Rule-based residual rate (comparison)
    ax = axes[1]
    bars = ax.barh(range(len(names)), rule_rates, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Rule-Based Residual Rate (%)")
    ax.set_title("ABVD Cognacy-Based Residual Rate\n(No ABVD Cognacy Assignment)")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", alpha=0.8, label="Original 6"),
        Patch(facecolor="#2ca02c", alpha=0.8, label="Sulawesi expansion"),
        Patch(facecolor="#d62728", alpha=0.8, label="W. Indonesian"),
        Patch(facecolor="#ff7f0e", alpha=0.8, label="E. Indonesian"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("E027 Expansion: ML Substrate Detection Across Indonesian Languages",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "expansion_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT / 'expansion_barplot.png'}")

    # ============================================================
    # Generate detailed report
    # ============================================================
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("E027 EXPANSION VALIDATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"Date: 2026-03-10")
    report_lines.append(f"Model: XGBoost Model B (phonological-only, {len(model_b_cols)} features)")
    report_lines.append(f"Trained on: 6 original Sulawesi languages ({len(train_df)} forms)")
    report_lines.append(f"Applied to: {len(EXPANSION_LANGS)} expansion languages ({len(all_expansion_rows)} forms)")
    report_lines.append("")
    report_lines.append("EXPANSION LANGUAGES:")
    report_lines.append(f"  Sulawesi: Banggai, Bantik, Gorontalo, Bol.Mongondow, Kulisusu, Totoli, Uma, Kambowa")
    report_lines.append(f"  Western Indonesian: Balinese, Javanese, Malay, Sundanese, Sasak, Acehnese")
    report_lines.append(f"  Eastern Indonesian: Bima, Manggarai")
    report_lines.append("")
    report_lines.append("RESULTS BY LANGUAGE:")
    report_lines.append(f"  {'Language':<18} {'Group':<14} {'N':>5} {'Rule%':>7} {'ML%':>7} {'AUC':>7}")
    report_lines.append("  " + "-" * 60)
    for sr in sorted_rows:
        auc_str = f"{sr['auc']:.4f}" if sr['auc'] is not None else "  N/A"
        report_lines.append(
            f"  {sr['language']:<18} {sr['group']:<14} {sr['n_forms']:>5} "
            f"{sr['rule_residual_rate']*100:>6.1f}% {sr['ml_substrate_rate']*100:>6.1f}% {auc_str:>7}"
        )
    report_lines.append("")
    report_lines.append("GROUP MEANS:")
    for group in ["Original", "Sulawesi", "W.Indonesian", "E.Indonesian"]:
        gs = group_stats[group]
        if not gs["ml_rates"]:
            continue
        report_lines.append(f"  {group:<18} ML rate={np.mean(gs['ml_rates'])*100:.1f}%  "
                           f"Rule rate={np.mean(gs['rule_rates'])*100:.1f}%  "
                           f"Mean P(sub)={np.mean(gs['mean_ps']):.4f}")
    report_lines.append("")
    report_lines.append(f"VERDICT: {verdict}")
    report_lines.append(f"  {explanation}")
    report_lines.append("")

    report_path = OUT / "expansion_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
