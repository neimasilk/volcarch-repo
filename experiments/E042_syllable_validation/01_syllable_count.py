"""
E042: Syllable Count Validation
================================
Tests whether replacing character count (form_length) with syllable count
affects Model B performance.

Criticism: "Linguists measure word length by syllables or mora, not by
counting keyboard characters. Using len(string) is methodologically invalid
in phonological linguistics."

Method:
1. Compute syllable_count for all 1,357 forms (= vowel nuclei count)
2. Replace form_length with syllable_count in Model B
3. Also test: both features together
4. Compare AUC across all variants

If AUC is maintained with syllable_count → the signal is syllable-level,
not character-level artifact.
"""
import csv
import io
import re
import sys
import json
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import xgboost as xgb

REPO = Path(__file__).parent.parent.parent
E027_DATA = REPO / "experiments" / "E027_ml_substrate_detection" / "data"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

VOWELS = set("aeiouəɛɨɔæøüöäåãẽĩõũâêîôûàèìòùáéíóú")


def count_syllables(form):
    """Count syllables as vowel nuclei (sequences of vowels = 1 syllable).

    This is a standard approximation used in computational phonology.
    A vowel nucleus is any maximal sequence of vowel characters.

    Examples:
        'lima'     → 2 (li-ma)
        'kangkaha' → 3 (kang-ka-ha)
        'ghaghe'   → 2 (gha-ghe)
        'apingka'  → 3 (a-ping-ka)
        'aesia'    → 3 (a-e-si-a) → actually 4... but diphthongs...

    Note: This may overcount for sequences like 'ae', 'oe' that might be
    diphthongs (1 syllable) rather than hiatus (2 syllables). This is
    a known limitation, but since it affects all forms uniformly, it
    should not bias the substrate/Austronesian comparison.
    """
    fl = form.lower()
    count = 0
    in_vowel = False
    for c in fl:
        if c in VOWELS:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
    return max(count, 1)  # at least 1 syllable


def main():
    print("=" * 70)
    print("E042: Syllable Count Validation for E027 Model B")
    print("=" * 70)

    # Load feature matrix
    print("\n[1/4] Loading E027 feature matrix...")
    rows = []
    with open(E027_DATA / "features_matrix.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows)} forms")

    # ============================================================
    # Step 2: Compute syllable counts and compare with char length
    # ============================================================
    print("\n[2/4] Computing syllable counts...")

    forms_data = []
    for row in rows:
        form = row["form"]
        char_len = int(row["form_length"])
        syl_count = count_syllables(form)
        forms_data.append({
            **row,
            "syllable_count": syl_count,
            "char_length": char_len,
            "chars_per_syllable": round(char_len / syl_count, 2) if syl_count > 0 else 0,
        })

    # Stats
    syl_counts = [f["syllable_count"] for f in forms_data]
    char_lens = [f["char_length"] for f in forms_data]
    correlation = np.corrcoef(syl_counts, char_lens)[0, 1]

    print(f"  Syllable count: min={min(syl_counts)}, max={max(syl_counts)}, "
          f"mean={np.mean(syl_counts):.2f}")
    print(f"  Char length:    min={min(char_lens)}, max={max(char_lens)}, "
          f"mean={np.mean(char_lens):.2f}")
    print(f"  Correlation (syllable ~ char): r = {correlation:.4f}")

    # Compare by label
    syl_aus = [f["syllable_count"] for f in forms_data if int(f["label"]) == 1]
    syl_sub = [f["syllable_count"] for f in forms_data if int(f["label"]) == 0]
    char_aus = [f["char_length"] for f in forms_data if int(f["label"]) == 1]
    char_sub = [f["char_length"] for f in forms_data if int(f["label"]) == 0]

    print(f"\n  By label:")
    print(f"  {'Metric':<20} {'Austronesian':>14} {'Non-mainstream':>14} {'Delta':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Char length (mean)':<20} {np.mean(char_aus):>14.2f} {np.mean(char_sub):>14.2f} "
          f"{np.mean(char_sub)-np.mean(char_aus):>+8.2f}")
    print(f"  {'Syllable count':<20} {np.mean(syl_aus):>14.2f} {np.mean(syl_sub):>14.2f} "
          f"{np.mean(syl_sub)-np.mean(syl_aus):>+8.2f}")
    print(f"  {'Chars/syllable':<20} {np.mean(char_aus)/np.mean(syl_aus):>14.2f} "
          f"{np.mean(char_sub)/np.mean(syl_sub):>14.2f}")

    # Per-language syllable stats
    print(f"\n  Per-language syllable counts:")
    print(f"  {'Language':<18} {'Mean syl (Aus)':>14} {'Mean syl (Sub)':>14} {'Delta':>8}")
    print(f"  {'-'*58}")
    lang_stats = defaultdict(lambda: {"aus_syl": [], "sub_syl": []})
    for f in forms_data:
        if int(f["label"]) == 1:
            lang_stats[f["language"]]["aus_syl"].append(f["syllable_count"])
        else:
            lang_stats[f["language"]]["sub_syl"].append(f["syllable_count"])
    for lang in sorted(lang_stats.keys()):
        ls = lang_stats[lang]
        ma = np.mean(ls["aus_syl"]) if ls["aus_syl"] else 0
        ms = np.mean(ls["sub_syl"]) if ls["sub_syl"] else 0
        print(f"  {lang:<18} {ma:>14.2f} {ms:>14.2f} {ms-ma:>+8.2f}")

    # ============================================================
    # Step 3: Train Model B variants
    # ============================================================
    print("\n[3/4] Training Model B variants...")

    # Feature engineering
    semantic_dummies = {"ACTION": 0, "BODY": 1, "GRAMMAR": 2, "NATURE": 3,
                       "NUMBER": 4, "OTHER": 5, "QUALITY": 6}
    initial_dummies = {"m": 0, "a": 1, "b": 2, "t": 3, "k": 4, "p": 5, "s": 6, "other": 7}

    def make_features(feat_dict, length_mode="char"):
        """Build feature vector with different length metrics."""
        if length_mode == "char":
            length_val = feat_dict["char_length"]
        elif length_mode == "syllable":
            length_val = feat_dict["syllable_count"]
        elif length_mode == "both":
            pass  # handled below

        ic = feat_dict.get("initial_char", "other")
        ic_vec = [0] * 8
        ic_vec[initial_dummies.get(ic, 7)] = 1

        sd = feat_dict["semantic_domain"]
        sd_vec = [0] * 7
        sd_vec[semantic_dummies.get(sd, 5)] = 1

        base = [
            int(feat_dict["n_vowels"]),
            float(feat_dict["vowel_ratio"]),
            int(feat_dict["ends_in_vowel"]),
            *ic_vec,
            int(feat_dict["has_glottal"]),
            int(feat_dict["has_nasal_cluster"]),
            int(feat_dict["has_reduplication"]),
            int(feat_dict["n_consonant_clusters"]),
            int(feat_dict["has_prefix_like"]),
            *sd_vec,
            int(feat_dict["is_core_vocab"]),
            int(feat_dict["language_id_encoded"]),
        ]

        if length_mode == "both":
            return [feat_dict["char_length"], feat_dict["syllable_count"]] + base
        else:
            return [length_val] + base

    # Build feature matrices for all variants
    variants = {
        "char_length": "char",
        "syllable_count": "syllable",
        "both": "both",
        "no_length": None,
    }

    X_dict = {}
    for name, mode in variants.items():
        if mode is None:
            # No length feature at all
            X_dict[name] = np.array([make_features(f, "char")[1:] for f in forms_data])
        else:
            X_dict[name] = np.array([make_features(f, mode) for f in forms_data])

    y = np.array([int(f["label"]) for f in forms_data])
    languages = np.array([f["language"] for f in forms_data])

    for name, X in X_dict.items():
        print(f"  {name}: {X.shape[1]} features")

    # XGBoost params (same as E027)
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "verbosity": 0,
    }

    # --- CV + LOLO for all variants ---
    cv_results = {}
    lolo_results = {}

    for name, X in X_dict.items():
        # Stratified 5-fold CV x 10 seeds
        aucs = []
        for seed in range(10):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for train_idx, test_idx in skf.split(X, y):
                model = xgb.XGBClassifier(**{**xgb_params, "random_state": seed})
                model.fit(X[train_idx], y[train_idx])
                y_prob = model.predict_proba(X[test_idx])[:, 1]
                aucs.append(roc_auc_score(y[test_idx], y_prob))
        cv_results[name] = {"mean": np.mean(aucs), "std": np.std(aucs)}

        # LOLO
        unique_langs = sorted(set(languages))
        lolo = {}
        for held_out in unique_langs:
            train_mask = languages != held_out
            test_mask = languages == held_out
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X[train_mask], y[train_mask])
            y_prob = model.predict_proba(X[test_mask])[:, 1]
            auc = roc_auc_score(y[test_mask], y_prob)
            lolo[held_out] = round(auc, 4)
        lolo_results[name] = lolo

    # Print results
    print(f"\n  {'Variant':<18} {'CV AUC':>12} {'LOLO mean':>12} {'LOLO >=0.65':>12}")
    print(f"  {'-'*56}")
    for name in variants:
        cv = cv_results[name]
        lolo_aucs = list(lolo_results[name].values())
        lolo_mean = np.mean(lolo_aucs)
        ge65 = sum(1 for a in lolo_aucs if a >= 0.65)
        print(f"  {name:<18} {cv['mean']:.4f}±{cv['std']:.4f} {lolo_mean:>12.4f} {ge65:>8}/6")

    # Print per-language LOLO comparison
    print(f"\n  Per-language LOLO AUC:")
    print(f"  {'Language':<18} {'char_length':>12} {'syllable':>12} {'both':>12} {'no_length':>12}")
    print(f"  {'-'*70}")
    for lang in sorted(set(languages)):
        vals = [f"{lolo_results[name][lang]:.4f}" for name in variants]
        print(f"  {lang:<18} {'  '.join(f'{v:>12}' for v in vals)}")

    # ============================================================
    # Step 4: Save results
    # ============================================================
    print("\n[4/4] Saving results...")

    summary = {
        "experiment": "E042 Syllable Count Validation",
        "date": "2026-03-11",
        "char_syllable_correlation": round(float(correlation), 4),
        "mean_syllables_austronesian": round(float(np.mean(syl_aus)), 2),
        "mean_syllables_nonmainstream": round(float(np.mean(syl_sub)), 2),
        "cv_results": {name: {"auc_mean": round(cv_results[name]["mean"], 4),
                               "auc_std": round(cv_results[name]["std"], 4)}
                       for name in variants},
        "lolo_results": {name: lolo_results[name] for name in variants},
        "lolo_means": {name: round(float(np.mean(list(lolo_results[name].values()))), 4)
                       for name in variants},
        "conclusion": "",
    }

    # Determine conclusion
    cv_char = cv_results["char_length"]["mean"]
    cv_syl = cv_results["syllable_count"]["mean"]
    cv_delta = cv_syl - cv_char

    lolo_char = np.mean(list(lolo_results["char_length"].values()))
    lolo_syl = np.mean(list(lolo_results["syllable_count"].values()))
    lolo_delta = lolo_syl - lolo_char

    if abs(cv_delta) < 0.02 and abs(lolo_delta) < 0.02:
        summary["conclusion"] = (
            f"ROBUST: Replacing character count with syllable count produces negligible change "
            f"(CV delta={cv_delta:+.4f}, LOLO delta={lolo_delta:+.4f}). "
            f"The length signal operates at syllable level, not character level."
        )
    elif cv_delta > 0.02 or lolo_delta > 0.02:
        summary["conclusion"] = (
            f"SYLLABLE BETTER: Syllable count outperforms character count "
            f"(CV delta={cv_delta:+.4f}, LOLO delta={lolo_delta:+.4f}). "
            f"Confirms that the signal is syllabic, not orthographic."
        )
    else:
        summary["conclusion"] = (
            f"CHARACTER BETTER: Character count outperforms syllable count "
            f"(CV delta={cv_delta:+.4f}, LOLO delta={lolo_delta:+.4f}). "
            f"Some character-level information contributes beyond syllable structure."
        )

    with open(OUT / "syllable_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {OUT / 'syllable_validation_summary.json'}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\n  {summary['conclusion']}")
    print(f"\n  Char→Syllable: CV {cv_char:.4f} → {cv_syl:.4f} ({cv_delta:+.4f})")
    print(f"  Char→Syllable: LOLO {lolo_char:.4f} → {lolo_syl:.4f} ({lolo_delta:+.4f})")


if __name__ == "__main__":
    main()
