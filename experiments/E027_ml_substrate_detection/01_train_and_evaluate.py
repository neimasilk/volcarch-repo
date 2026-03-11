"""
E027 Script 01: Train and Evaluate ML Models
==============================================
Two model variants:
  Model A — Full features (all 18) — ranking tool, expected high AUC (circular)
  Model B — Phonological-only (10 phon + 2 semantic + 2 language) — scientific claim

Three classifiers: XGBoost, Random Forest, Logistic Regression
Validation: Stratified 5-fold CV (×10 seeds) + Leave-One-Language-Out (LOLO)

Output: results/cv_results.json, results/lolo_results.json
"""
import csv
import io
import json
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost not available. Using only RF and LR.")

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# Feature groups
PHON_FEATURES = [
    "form_length", "n_vowels", "vowel_ratio", "ends_in_vowel",
    "has_glottal", "has_nasal_cluster", "has_reduplication",
    "n_consonant_clusters", "has_prefix_like",
]
# initial_char is categorical — will be one-hot encoded

DIST_FEATURES = [
    "max_cognate_set_size", "n_cognate_sets",
    "concept_residual_rate", "concept_cross_lang_count",
]

SEMANTIC_FEATURES = [
    "is_core_vocab",
]
# semantic_domain is categorical — will be one-hot encoded

LANG_FEATURES = [
    "language_id_encoded", "language_cognacy_coverage",
]


def load_data():
    """Load feature matrix and prepare X, y arrays."""
    df = pd.read_csv(DATA / "features_matrix.csv", encoding="utf-8")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # One-hot encode initial_char
    ic_dummies = pd.get_dummies(df["initial_char"], prefix="init")
    df = pd.concat([df, ic_dummies], axis=1)

    # One-hot encode semantic_domain
    sd_dummies = pd.get_dummies(df["semantic_domain"], prefix="sem")
    df = pd.concat([df, sd_dummies], axis=1)

    # Define feature sets
    init_cols = [c for c in df.columns if c.startswith("init_")]
    sem_cols = [c for c in df.columns if c.startswith("sem_")]

    # Model B: phonological + semantic + language (NO distributional)
    model_b_cols = PHON_FEATURES + init_cols + SEMANTIC_FEATURES + sem_cols + LANG_FEATURES

    # Model A: all features
    model_a_cols = model_b_cols + DIST_FEATURES

    y = df["label"].values  # 1 = Austronesian, 0 = substrate
    languages = df["language"].values

    X_a = df[model_a_cols].values.astype(float)
    X_b = df[model_b_cols].values.astype(float)

    return df, X_a, X_b, y, languages, model_a_cols, model_b_cols


def get_classifiers():
    """Return dict of classifier constructors."""
    classifiers = {}
    if HAS_XGB:
        classifiers["XGBoost"] = lambda: XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            reg_lambda=1.0, scale_pos_weight=1.0,
            eval_metric="logloss", random_state=42,
            use_label_encoder=False, verbosity=0,
        )
    classifiers["RandomForest"] = lambda: RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )
    classifiers["LogisticRegression"] = lambda: LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced",
        max_iter=1000, random_state=42, solver="lbfgs",
    )
    return classifiers


def run_stratified_cv(X, y, n_seeds=10, n_folds=5):
    """Run stratified K-fold CV across multiple random seeds."""
    classifiers = get_classifiers()
    results = {}

    for clf_name, clf_fn in classifiers.items():
        seed_aucs = []
        seed_f1s = []
        seed_accs = []

        for seed in range(n_seeds):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed * 7 + 13)
            fold_aucs = []
            fold_f1s = []
            fold_accs = []

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Scale for LR
                if clf_name == "LogisticRegression":
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                clf = clf_fn()
                clf.fit(X_train, y_train)

                y_prob = clf.predict_proba(X_test)[:, 1]
                y_pred = clf.predict(X_test)

                fold_aucs.append(roc_auc_score(y_test, y_prob))
                fold_f1s.append(f1_score(y_test, y_pred))
                fold_accs.append(accuracy_score(y_test, y_pred))

            seed_aucs.append(np.mean(fold_aucs))
            seed_f1s.append(np.mean(fold_f1s))
            seed_accs.append(np.mean(fold_accs))

        results[clf_name] = {
            "auc_mean": round(float(np.mean(seed_aucs)), 4),
            "auc_std": round(float(np.std(seed_aucs)), 4),
            "f1_mean": round(float(np.mean(seed_f1s)), 4),
            "f1_std": round(float(np.std(seed_f1s)), 4),
            "acc_mean": round(float(np.mean(seed_accs)), 4),
            "acc_std": round(float(np.std(seed_accs)), 4),
        }

    return results


def run_lolo(X, y, languages):
    """Run Leave-One-Language-Out cross-validation."""
    classifiers = get_classifiers()
    unique_langs = sorted(set(languages))
    results = {}

    for clf_name, clf_fn in classifiers.items():
        lang_results = {}
        for held_out in unique_langs:
            test_mask = languages == held_out
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            # Skip if test set is homogeneous (only one class)
            if len(set(y_test)) < 2:
                lang_results[held_out] = {
                    "auc": None,
                    "f1": None,
                    "acc": round(float(accuracy_score(y_test, [1] * len(y_test))), 4),
                    "n_test": int(sum(test_mask)),
                    "note": "single class in test set",
                }
                continue

            if clf_name == "LogisticRegression":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            clf = clf_fn()
            clf.fit(X_train, y_train)

            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = clf.predict(X_test)

            lang_results[held_out] = {
                "auc": round(float(roc_auc_score(y_test, y_prob)), 4),
                "f1": round(float(f1_score(y_test, y_pred)), 4),
                "acc": round(float(accuracy_score(y_test, y_pred)), 4),
                "n_test": int(sum(test_mask)),
                "n_substrate": int(sum(y_test == 0)),
                "n_austronesian": int(sum(y_test == 1)),
            }

        # Compute mean AUC across languages with valid AUC
        valid_aucs = [v["auc"] for v in lang_results.values() if v["auc"] is not None]
        lang_results["_mean"] = {
            "auc_mean": round(float(np.mean(valid_aucs)), 4) if valid_aucs else None,
            "auc_std": round(float(np.std(valid_aucs)), 4) if valid_aucs else None,
            "n_langs_above_065": sum(1 for a in valid_aucs if a >= 0.65),
            "n_langs_valid": len(valid_aucs),
        }
        results[clf_name] = lang_results

    return results


def main():
    print("=" * 70)
    print("E027 Script 01: Train and Evaluate ML Models")
    print("=" * 70)

    df, X_a, X_b, y, languages, cols_a, cols_b = load_data()

    print(f"\nModel A features: {len(cols_a)} ({X_a.shape})")
    print(f"Model B features: {len(cols_b)} ({X_b.shape})")
    print(f"Labels: {sum(y==1)} Austronesian, {sum(y==0)} substrate")

    # ============================================================
    # Stratified 5-fold CV (×10 seeds)
    # ============================================================
    print("\n" + "=" * 70)
    print("STRATIFIED 5-FOLD CV (×10 seeds)")
    print("=" * 70)

    print("\n--- Model A (Full Features) ---")
    cv_a = run_stratified_cv(X_a, y, n_seeds=10, n_folds=5)
    for clf, res in cv_a.items():
        print(f"  {clf:<22} AUC={res['auc_mean']:.4f}±{res['auc_std']:.4f}  "
              f"F1={res['f1_mean']:.4f}±{res['f1_std']:.4f}  "
              f"Acc={res['acc_mean']:.4f}±{res['acc_std']:.4f}")

    print("\n--- Model B (Phonological-Only) ---")
    cv_b = run_stratified_cv(X_b, y, n_seeds=10, n_folds=5)
    for clf, res in cv_b.items():
        print(f"  {clf:<22} AUC={res['auc_mean']:.4f}±{res['auc_std']:.4f}  "
              f"F1={res['f1_mean']:.4f}±{res['f1_std']:.4f}  "
              f"Acc={res['acc_mean']:.4f}±{res['acc_std']:.4f}")

    # Save CV results
    cv_results = {"model_a": cv_a, "model_b": cv_b}
    with open(OUT / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)
    print(f"\nSaved: {OUT / 'cv_results.json'}")

    # ============================================================
    # Leave-One-Language-Out (LOLO)
    # ============================================================
    print("\n" + "=" * 70)
    print("LEAVE-ONE-LANGUAGE-OUT (LOLO)")
    print("=" * 70)

    print("\n--- Model A (Full Features) ---")
    lolo_a = run_lolo(X_a, y, languages)
    for clf_name, lang_results in lolo_a.items():
        print(f"\n  {clf_name}:")
        for lang, res in lang_results.items():
            if lang == "_mean":
                print(f"    {'MEAN':<18} AUC={res['auc_mean']:.4f}±{res['auc_std']:.4f}  "
                      f"({res['n_langs_above_065']}/{res['n_langs_valid']} langs ≥0.65)")
            else:
                auc_str = f"{res['auc']:.4f}" if res['auc'] is not None else "N/A"
                f1_str = f"{res['f1']:.4f}" if res.get('f1') is not None else "N/A"
                print(f"    {lang:<18} AUC={auc_str}  F1={f1_str}  "
                      f"n={res['n_test']}")

    print("\n--- Model B (Phonological-Only) ---")
    lolo_b = run_lolo(X_b, y, languages)
    for clf_name, lang_results in lolo_b.items():
        print(f"\n  {clf_name}:")
        for lang, res in lang_results.items():
            if lang == "_mean":
                print(f"    {'MEAN':<18} AUC={res['auc_mean']:.4f}±{res['auc_std']:.4f}  "
                      f"({res['n_langs_above_065']}/{res['n_langs_valid']} langs ≥0.65)")
            else:
                auc_str = f"{res['auc']:.4f}" if res['auc'] is not None else "N/A"
                f1_str = f"{res['f1']:.4f}" if res.get('f1') is not None else "N/A"
                print(f"    {lang:<18} AUC={auc_str}  F1={f1_str}  "
                      f"n={res['n_test']}")

    # Save LOLO results
    lolo_results = {"model_a": lolo_a, "model_b": lolo_b}
    with open(OUT / "lolo_results.json", "w", encoding="utf-8") as f:
        json.dump(lolo_results, f, indent=2)
    print(f"\nSaved: {OUT / 'lolo_results.json'}")

    # ============================================================
    # GO/NO-GO Assessment
    # ============================================================
    print("\n" + "=" * 70)
    print("GO/NO-GO ASSESSMENT")
    print("=" * 70)

    # Use best classifier for Model B (typically XGBoost)
    best_clf = "XGBoost" if HAS_XGB else "RandomForest"
    b_auc = cv_b[best_clf]["auc_mean"]
    b_lolo = lolo_b[best_clf]["_mean"]

    print(f"\nModel B ({best_clf}):")
    print(f"  CV AUC:        {b_auc:.4f}")
    print(f"  LOLO mean AUC: {b_lolo['auc_mean']:.4f}")
    print(f"  LOLO ≥0.65:    {b_lolo['n_langs_above_065']}/{b_lolo['n_langs_valid']} languages")

    if b_auc >= 0.75 and b_lolo['n_langs_above_065'] >= 4:
        verdict = "GO"
        explanation = "AUC ≥ 0.75 and LOLO ≥ 0.65 for 4+ languages. ML substrate detection is viable."
    elif b_auc >= 0.65:
        verdict = "CONDITIONAL GO"
        explanation = "AUC 0.65-0.75. ML validates E022 but adds limited new insight."
    else:
        verdict = "NO-GO"
        explanation = "AUC < 0.65. Phonological features cannot reliably distinguish substrate."

    print(f"\n  >>> VERDICT: {verdict}")
    print(f"  >>> {explanation}")

    # Save verdict
    verdict_data = {
        "verdict": verdict,
        "explanation": explanation,
        "model_b_best_classifier": best_clf,
        "cv_auc": b_auc,
        "lolo_auc_mean": b_lolo['auc_mean'],
        "lolo_langs_above_065": b_lolo['n_langs_above_065'],
    }
    with open(OUT / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_data, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
