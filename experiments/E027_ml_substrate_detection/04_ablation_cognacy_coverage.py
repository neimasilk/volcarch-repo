"""
E027 Script 04: Ablation — Remove language_cognacy_coverage confound
====================================================================
Red-team test: What happens to Model B when we remove the #1 SHAP feature
(language_cognacy_coverage), which is a language-level property correlated
with the labeling process itself?

Reports: CV AUC and LOLO AUC with and without the feature.
"""
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

PHON_FEATURES = [
    "form_length", "n_vowels", "vowel_ratio", "ends_in_vowel",
    "has_glottal", "has_nasal_cluster", "has_reduplication",
    "n_consonant_clusters", "has_prefix_like",
]

SEMANTIC_FEATURES = ["is_core_vocab"]


def load_and_prepare():
    df = pd.read_csv(DATA / "features_matrix.csv", encoding="utf-8")
    ic_dummies = pd.get_dummies(df["initial_char"], prefix="init")
    sd_dummies = pd.get_dummies(df["semantic_domain"], prefix="sem")
    df = pd.concat([df, ic_dummies, sd_dummies], axis=1)

    init_cols = [c for c in df.columns if c.startswith("init_")]
    sem_cols = [c for c in df.columns if c.startswith("sem_")]

    # Model B full (original): includes both language features
    full_cols = PHON_FEATURES + init_cols + SEMANTIC_FEATURES + sem_cols + [
        "language_id_encoded", "language_cognacy_coverage"
    ]

    # Model B-ablated: remove language_cognacy_coverage
    ablated_cols = PHON_FEATURES + init_cols + SEMANTIC_FEATURES + sem_cols + [
        "language_id_encoded"
    ]

    # Model B-pure: remove BOTH language features (purely phonological + semantic)
    pure_cols = PHON_FEATURES + init_cols + SEMANTIC_FEATURES + sem_cols

    y = df["label"].values
    languages = df["language"].values

    return df, y, languages, full_cols, ablated_cols, pure_cols


def run_cv(X, y, n_seeds=10, n_folds=5):
    aucs, f1s = [], []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed * 7 + 13)
        fold_aucs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                reg_lambda=1.0, scale_pos_weight=1.0,
                eval_metric="logloss", random_state=42,
                use_label_encoder=False, verbosity=0,
            )
            clf.fit(X[train_idx], y[train_idx])
            y_prob = clf.predict_proba(X[test_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y[test_idx], y_prob))
        aucs.append(np.mean(fold_aucs))
    return np.mean(aucs), np.std(aucs)


def run_lolo(X, y, languages):
    unique_langs = sorted(set(languages))
    lang_aucs = {}
    for held_out in unique_langs:
        test_mask = languages == held_out
        train_mask = ~test_mask
        if len(set(y[test_mask])) < 2:
            continue
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            reg_lambda=1.0, scale_pos_weight=1.0,
            eval_metric="logloss", random_state=42,
            use_label_encoder=False, verbosity=0,
        )
        clf.fit(X[train_mask], y[train_mask])
        y_prob = clf.predict_proba(X[test_mask])[:, 1]
        lang_aucs[held_out] = round(roc_auc_score(y[test_mask], y_prob), 4)

    mean_auc = np.mean(list(lang_aucs.values()))
    return lang_aucs, round(float(mean_auc), 4)


def main():
    print("=" * 70)
    print("E027 Ablation: language_cognacy_coverage confound test")
    print("=" * 70)

    df, y, languages, full_cols, ablated_cols, pure_cols = load_and_prepare()

    X_full = df[full_cols].values.astype(float)
    X_ablated = df[ablated_cols].values.astype(float)
    X_pure = df[pure_cols].values.astype(float)

    print(f"\nFull Model B:    {len(full_cols)} features")
    print(f"Ablated (-cov):  {len(ablated_cols)} features")
    print(f"Pure (no lang):  {len(pure_cols)} features")
    print(f"Labels: {sum(y==1)} Austronesian, {sum(y==0)} substrate")

    # --- CV ---
    print("\n--- Stratified 5-fold CV (×10 seeds) ---")
    auc_full, std_full = run_cv(X_full, y)
    print(f"  Full Model B:    AUC = {auc_full:.4f} ± {std_full:.4f}")

    auc_abl, std_abl = run_cv(X_ablated, y)
    print(f"  Ablated (-cov):  AUC = {auc_abl:.4f} ± {std_abl:.4f}  (delta = {auc_abl - auc_full:+.4f})")

    auc_pure, std_pure = run_cv(X_pure, y)
    print(f"  Pure (no lang):  AUC = {auc_pure:.4f} ± {std_pure:.4f}  (delta = {auc_pure - auc_full:+.4f})")

    # --- LOLO ---
    print("\n--- Leave-One-Language-Out ---")
    lolo_full, mean_full = run_lolo(X_full, y, languages)
    print(f"\n  Full Model B (mean AUC = {mean_full}):")
    for lang, auc in sorted(lolo_full.items()):
        print(f"    {lang:<18} AUC = {auc}")

    lolo_abl, mean_abl = run_lolo(X_ablated, y, languages)
    print(f"\n  Ablated -cov (mean AUC = {mean_abl}, delta = {mean_abl - mean_full:+.4f}):")
    for lang, auc in sorted(lolo_abl.items()):
        delta = auc - lolo_full.get(lang, 0)
        print(f"    {lang:<18} AUC = {auc}  ({delta:+.4f})")

    lolo_pure, mean_pure = run_lolo(X_pure, y, languages)
    print(f"\n  Pure no-lang (mean AUC = {mean_pure}, delta = {mean_pure - mean_full:+.4f}):")
    for lang, auc in sorted(lolo_pure.items()):
        delta = auc - lolo_full.get(lang, 0)
        print(f"    {lang:<18} AUC = {auc}  ({delta:+.4f})")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Variant':<25} {'CV AUC':<18} {'LOLO AUC':<18} {'LOLO ≥0.65'}")
    n_full = sum(1 for a in lolo_full.values() if a >= 0.65)
    n_abl = sum(1 for a in lolo_abl.values() if a >= 0.65)
    n_pure = sum(1 for a in lolo_pure.values() if a >= 0.65)
    print(f"  {'Full Model B':<25} {auc_full:.4f}±{std_full:.4f}    {mean_full:<18} {n_full}/6")
    print(f"  {'Ablated (-coverage)':<25} {auc_abl:.4f}±{std_abl:.4f}    {mean_abl:<18} {n_abl}/6")
    print(f"  {'Pure (no lang features)':<25} {auc_pure:.4f}±{std_pure:.4f}    {mean_pure:<18} {n_pure}/6")

    # Verdict
    print("\n  INTERPRETATION:")
    if auc_abl >= 0.70:
        print("  >>> Ablated model retains strong signal. Cognacy coverage is NOT driving the result.")
        print("  >>> The phonological fingerprint is robust.")
    elif auc_abl >= 0.65:
        print("  >>> Ablated model retains moderate signal. Some inflation from coverage, but core signal real.")
    else:
        print("  >>> WARNING: Large drop. Cognacy coverage may be driving most of the result.")

    # Save
    results = {
        "full": {"cv_auc": round(auc_full, 4), "cv_std": round(std_full, 4),
                 "lolo_mean": mean_full, "lolo_per_lang": lolo_full},
        "ablated_no_coverage": {"cv_auc": round(auc_abl, 4), "cv_std": round(std_abl, 4),
                                "lolo_mean": mean_abl, "lolo_per_lang": lolo_abl},
        "pure_no_lang_features": {"cv_auc": round(auc_pure, 4), "cv_std": round(std_pure, 4),
                                  "lolo_mean": mean_pure, "lolo_per_lang": lolo_pure},
    }
    with open(OUT / "ablation_cognacy_coverage.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT / 'ablation_cognacy_coverage.json'}")


if __name__ == "__main__":
    main()
