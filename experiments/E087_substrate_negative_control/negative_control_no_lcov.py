"""
E087b: Substrate Detector Negative Control — WITHOUT language_cognacy_coverage
==============================================================================
Re-runs the negative control tests but EXCLUDES the language_cognacy_coverage
feature, which is known to be semi-circular (E027 caveat #4, E085 test 4).

This is the FAIREST test: can PURELY PHONOLOGICAL features distinguish
substrate in Sulawesi but NOT in control pairs?

If yes -> the phonological fingerprint is real and specific
If no  -> the E027 AUC=0.762 is inflated by language_cognacy_coverage
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

# Same language sets as main script
ORIGINAL_LANGS = {"27": "Muna", "48": "Bugis", "166": "Makassar",
                  "192": "Wolio", "226": "Toraja-Sadan", "674": "Tolaki"}
CONTROL1_LANGS = {"277": "Tagalog", "153": "Cebuano"}
CONTROL2_LANGS = {"233": "Malay", "172": "Minangkabau"}
CONTROL3_LANGS = {"20": "Javanese", "284": "Sundanese"}
CONTROL4_LANGS = {"277": "Tagalog", "33": "Kapampangan"}
CONTROL5_LANGS = {"28": "Iban", "233": "Malay"}

# Also test a pair with KNOWN substrate: Acehnese (Mon-Khmer substrate documented)
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


def compute_features_no_lcov(form, concept, lang_idx):
    """
    Compute features WITHOUT language_cognacy_coverage.
    Keep language_id_encoded as a minimal language control.
    """
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
        "language_id_encoded": lang_idx,
        # NO language_cognacy_coverage
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


def build_features_no_lcov(target_langs, params, cognate_info, all_forms):
    lang_forms = [r for r in all_forms if r["Language_ID"] in target_langs]
    lang_idx_map = {lid: i for i, lid in enumerate(sorted(target_langs.keys()))}

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

        feats = compute_features_no_lcov(form_value, concept, lang_idx_map[lid])
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
    lang_cols = ["language_id_encoded"]  # NO language_cognacy_coverage
    meta_cols = ["is_core_vocab"]

    model_cols = phon_cols + init_cols + meta_cols + sem_cols + lang_cols
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


def main():
    print("=" * 70)
    print("E087b: NEGATIVE CONTROL — WITHOUT language_cognacy_coverage")
    print("  Testing PURELY PHONOLOGICAL signal")
    print("=" * 70)
    print()

    params, cognate_info, all_forms = load_abvd_data()
    print(f"Loaded {len(params)} concepts, {len(all_forms)} forms\n")

    test_sets = [
        ("E027 Original (Sulawesi 6)", ORIGINAL_LANGS),
        ("C1: Tagalog + Cebuano", CONTROL1_LANGS),
        ("C2: Malay + Minangkabau", CONTROL2_LANGS),
        ("C3: Javanese + Sundanese (real labels)", CONTROL3_LANGS),
        ("C4: Tagalog + Kapampangan", CONTROL4_LANGS),
        ("C5: Iban + Malay", CONTROL5_LANGS),
        ("C6: Acehnese + Toba Batak (positive ctrl)", CONTROL6_LANGS),
    ]

    results = {}
    print(f"{'Test':<50} {'N':>5} {'Substr%':>8} {'AUC':>8} {'Std':>8}")
    print("-" * 82)

    for name, langs in test_sets:
        rows = build_features_no_lcov(langs, params, cognate_info, all_forms)
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

        print(f"{name:<50} {n_tot:>5} {pct:>7.1f}% {auc:>8.4f} {std:>8.4f}  {note}")

        results[name] = {
            "auc": round(float(auc), 4),
            "std": round(float(std), 4),
            "n_forms": int(n_tot),
            "n_substrate": int(n_sub),
            "substrate_rate": round(float(pct), 1),
        }

    print()

    # Interpretation
    orig_auc = results["E027 Original (Sulawesi 6)"]["auc"]
    c1_auc = results["C1: Tagalog + Cebuano"]["auc"]
    c2_auc = results["C2: Malay + Minangkabau"]["auc"]
    c5_auc = results["C5: Iban + Malay"]["auc"]

    print("=" * 70)
    print("INTERPRETATION — No-LCOV Results")
    print("=" * 70)
    print()
    print(f"  Original Sulawesi AUC (no lcov): {orig_auc:.4f}")
    print(f"  C1 Tagalog+Cebuano (no lcov):    {c1_auc:.4f}  delta = {orig_auc - c1_auc:+.4f}")
    print(f"  C2 Malay+Minangkabau (no lcov):  {c2_auc:.4f}  delta = {orig_auc - c2_auc:+.4f}")
    print(f"  C5 Iban+Malay (no lcov):         {c5_auc:.4f}  delta = {orig_auc - c5_auc:+.4f}")
    print()

    # Compare with WITH-lcov results
    print("  COMPARISON: With vs Without language_cognacy_coverage:")
    print(f"  {'Test':<35} {'With lcov':>10} {'Without':>10} {'Drop':>8}")
    print(f"  {'-'*65}")
    with_lcov = {
        "E027 Original (Sulawesi 6)": 0.7610,
        "C1: Tagalog + Cebuano": 0.6107,
        "C2: Malay + Minangkabau": 0.6896,
        "C5: Iban + Malay": 0.7939,
    }
    for name, wl_auc in with_lcov.items():
        wo_auc = results[name]["auc"]
        drop = wo_auc - wl_auc
        print(f"  {name:<35} {wl_auc:>10.4f} {wo_auc:>10.4f} {drop:>+8.4f}")

    print()

    # Verdict
    all_ctrl_below_60 = all(
        results[n]["auc"] < 0.60
        for n in ["C1: Tagalog + Cebuano", "C2: Malay + Minangkabau", "C5: Iban + Malay"]
    )
    any_ctrl_above_70 = any(
        results[n]["auc"] > 0.70
        for n in ["C1: Tagalog + Cebuano", "C2: Malay + Minangkabau", "C5: Iban + Malay"]
    )

    if orig_auc > 0.65 and all_ctrl_below_60:
        verdict = "VOLCARCH PASSES"
        expl = (
            f"Without language_cognacy_coverage, the Sulawesi detector still achieves "
            f"AUC={orig_auc:.3f} while all controls are below 0.60. "
            f"The phonological fingerprint is REAL and SPECIFIC to Sulawesi substrate."
        )
    elif any_ctrl_above_70:
        verdict = "VOLCARCH FAILS"
        expl = (
            f"Even without lcov, a control pair produces AUC > 0.70. "
            f"The detector is not specific to substrate."
        )
    elif orig_auc < 0.60:
        verdict = "WEAK — PHONOLOGICAL SIGNAL INSUFFICIENT"
        expl = (
            f"Without language_cognacy_coverage, the Sulawesi AUC drops to {orig_auc:.3f}. "
            f"Much of the E027 claim rests on the semi-circular lcov feature. "
            f"The purely phonological signal is too weak to be convincing."
        )
    else:
        verdict = "CONDITIONAL PASS"
        expl = (
            f"Sulawesi AUC ({orig_auc:.3f}) is above controls, but some controls "
            f"(C2={c2_auc:.3f}, C5={c5_auc:.3f}) show moderate signal. "
            f"The gap is real but not large."
        )

    print(f"  VERDICT: {verdict}")
    print(f"  {expl}")

    # Save
    output = {
        "experiment": "E087b_no_lcov",
        "date": "2026-03-16",
        "feature_set": "Model B WITHOUT language_cognacy_coverage",
        "results": results,
        "verdict": verdict,
        "explanation": expl,
    }
    with open(OUT / "negative_control_no_lcov.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUT / 'negative_control_no_lcov.json'}")


if __name__ == "__main__":
    main()
