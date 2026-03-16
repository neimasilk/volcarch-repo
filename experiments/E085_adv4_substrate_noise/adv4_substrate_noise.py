"""
E085: ADV-4 Substrate Noise Permutation Test
=============================================
Adversarial test for L4 (Cosmological Overwrite).

Tests whether the E027 ML substrate detection (XGBoost AUC=0.760)
reflects genuine phonological differences or is statistical noise.

Three tests:
  1. Label permutation: shuffle substrate/non-substrate labels 1000x,
     train classifier each time, record AUC distribution.
  2. Random feature baseline: replace real phonological features with
     random noise, keep real labels.
  3. Frequency-only baseline: use only word frequency (form_length as proxy)
     as a feature — no phonological information.

Pass criterion: empirical p < 0.05 (observed AUC in top 5% of permuted dist)
"""
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).parent.parent.parent
E027_DATA = REPO / "experiments" / "E027_ml_substrate_detection" / "data"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# ============================================================
# Configuration
# ============================================================
N_PERMUTATIONS = 1000
N_FOLDS = 5
OBSERVED_AUC_XGBOOST = 0.7599  # From E027 results
OBSERVED_AUC_RF = 0.7618       # From E027 results (we'll use RF for speed)

# Model B features from E027 (phonological-only, no distributional)
PHON_FEATURES = [
    "form_length", "n_vowels", "vowel_ratio", "ends_in_vowel",
    "has_glottal", "has_nasal_cluster", "has_reduplication",
    "n_consonant_clusters", "has_prefix_like",
]
SEMANTIC_FEATURES = ["is_core_vocab"]
LANG_FEATURES = ["language_id_encoded", "language_cognacy_coverage"]


def load_e027_data():
    """Load the E027 feature matrix and prepare Model B features."""
    df = pd.read_csv(E027_DATA / "features_matrix.csv", encoding="utf-8")
    print(f"  Loaded {len(df)} rows from E027 feature matrix")

    # One-hot encode initial_char (same as E027)
    ic_dummies = pd.get_dummies(df["initial_char"], prefix="init")
    df = pd.concat([df, ic_dummies], axis=1)

    # One-hot encode semantic_domain (same as E027)
    sd_dummies = pd.get_dummies(df["semantic_domain"], prefix="sem")
    df = pd.concat([df, sd_dummies], axis=1)

    init_cols = [c for c in df.columns if c.startswith("init_")]
    sem_cols = [c for c in df.columns if c.startswith("sem_")]

    # Model B columns (phonological + semantic + language, NO distributional)
    model_b_cols = PHON_FEATURES + init_cols + SEMANTIC_FEATURES + sem_cols + LANG_FEATURES

    X = df[model_b_cols].values.astype(float)
    y = df["label"].values  # 1 = Austronesian, 0 = substrate

    print(f"  Model B features: {len(model_b_cols)} columns")
    print(f"  Labels: {sum(y==1)} Austronesian, {sum(y==0)} substrate")
    print(f"  Class balance: {sum(y==0)/len(y)*100:.1f}% substrate")

    return df, X, y, model_b_cols


def evaluate_cv_auc(X, y, clf_fn, n_folds=5, seed=42):
    """Single run of stratified K-fold CV, returns mean AUC."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = clf_fn()
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, y_prob))
    return np.mean(aucs)


def evaluate_cv_auc_multiseed(X, y, clf_fn, n_seeds=10, n_folds=5):
    """Multi-seed stratified K-fold CV (same as E027), returns mean AUC."""
    seed_aucs = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed * 7 + 13)
        fold_aucs = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = clf_fn()
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_prob))
        seed_aucs.append(np.mean(fold_aucs))
    return np.mean(seed_aucs)


def get_rf():
    """Return a RandomForest classifier (faster than XGBoost for 1000 iterations)."""
    return RandomForestClassifier(
        n_estimators=200,  # Reduced from 500 for speed in permutation
        min_samples_leaf=5,
        random_state=None,  # Random for each permutation
        class_weight="balanced",
        n_jobs=-1,
    )


def get_lr():
    """Return a Logistic Regression classifier."""
    return LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced",
        max_iter=1000, solver="lbfgs",
    )


def main():
    print("=" * 70)
    print("E085: ADV-4 Substrate Noise Permutation Test")
    print("=" * 70)
    print()

    # ============================================================
    # Step 0: Load data and reproduce observed AUC
    # ============================================================
    print("[0/4] Loading E027 data and reproducing observed AUC...")
    df, X, y, feature_cols = load_e027_data()

    # Reproduce the observed AUC using RF with the same E027 settings
    # (to have our own baseline rather than just trusting the stored value)
    print("\n  Reproducing observed AUC with RandomForest (10 seeds x 5 folds)...")
    rf_full = lambda: RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )
    reproduced_auc = evaluate_cv_auc_multiseed(X, y, rf_full, n_seeds=10, n_folds=5)
    print(f"  Reproduced RF AUC: {reproduced_auc:.4f} (E027 reported: {OBSERVED_AUC_RF:.4f})")

    # Use our reproduced value as the observed AUC for the permutation test
    observed_auc = reproduced_auc
    print(f"\n  Using observed AUC = {observed_auc:.4f} for permutation test")

    # Also compute with Logistic Regression for comparison
    print("  Computing LR baseline...")
    lr_fn = lambda: LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced",
        max_iter=1000, solver="lbfgs", random_state=42,
    )
    # For LR we need to scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    observed_auc_lr = evaluate_cv_auc_multiseed(X_scaled, y, lr_fn, n_seeds=10, n_folds=5)
    print(f"  Reproduced LR AUC: {observed_auc_lr:.4f}")

    # ============================================================
    # Test 1: Label Permutation Test (primary)
    # ============================================================
    print("\n" + "=" * 70)
    print("[1/4] LABEL PERMUTATION TEST")
    print(f"  Shuffling labels {N_PERMUTATIONS} times, training RF each time...")
    print("=" * 70)

    rng = np.random.RandomState(2026)
    permuted_aucs = []

    for i in range(N_PERMUTATIONS):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Permutation {i+1}/{N_PERMUTATIONS}...")

        # Shuffle labels
        y_perm = rng.permutation(y)

        # Single-seed CV for speed (still robust with 5 folds)
        auc = evaluate_cv_auc(X, y_perm, get_rf, n_folds=N_FOLDS, seed=rng.randint(0, 100000))
        permuted_aucs.append(auc)

    permuted_aucs = np.array(permuted_aucs)

    # Statistics
    perm_mean = np.mean(permuted_aucs)
    perm_std = np.std(permuted_aucs)
    perm_95th = np.percentile(permuted_aucs, 95)
    perm_99th = np.percentile(permuted_aucs, 99)
    perm_max = np.max(permuted_aucs)

    # Empirical p-value: fraction of permuted AUCs >= observed
    p_value = np.mean(permuted_aucs >= observed_auc)

    # Z-score: how many SDs above the permuted mean
    z_score = (observed_auc - perm_mean) / perm_std if perm_std > 0 else float('inf')

    print(f"\n  --- Label Permutation Results ---")
    print(f"  Observed AUC:        {observed_auc:.4f}")
    print(f"  Permuted mean:       {perm_mean:.4f}")
    print(f"  Permuted std:        {perm_std:.4f}")
    print(f"  Permuted 95th pctl:  {perm_95th:.4f}")
    print(f"  Permuted 99th pctl:  {perm_99th:.4f}")
    print(f"  Permuted max:        {perm_max:.4f}")
    print(f"  Z-score:             {z_score:.2f}")
    print(f"  Empirical p-value:   {p_value:.4f}")
    print(f"  Verdict:             {'PASS' if p_value < 0.05 else 'FAIL'} (p < 0.05 required)")

    # ============================================================
    # Test 2: Random Feature Baseline
    # ============================================================
    print("\n" + "=" * 70)
    print("[2/4] RANDOM FEATURE BASELINE")
    print(f"  Replacing real features with random noise, keeping real labels...")
    print("=" * 70)

    N_RANDOM_RUNS = 100  # Fewer iterations needed since labels are fixed
    random_feature_aucs = []

    for i in range(N_RANDOM_RUNS):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Random features run {i+1}/{N_RANDOM_RUNS}...")

        # Generate random features with same shape as X
        X_random = rng.randn(X.shape[0], X.shape[1])

        auc = evaluate_cv_auc(X_random, y, get_rf, n_folds=N_FOLDS, seed=rng.randint(0, 100000))
        random_feature_aucs.append(auc)

    random_feature_aucs = np.array(random_feature_aucs)
    rf_mean = np.mean(random_feature_aucs)
    rf_std = np.std(random_feature_aucs)
    rf_max = np.max(random_feature_aucs)
    rf_p = np.mean(random_feature_aucs >= observed_auc)
    rf_z = (observed_auc - rf_mean) / rf_std if rf_std > 0 else float('inf')

    print(f"\n  --- Random Feature Results ---")
    print(f"  Observed AUC (real features):  {observed_auc:.4f}")
    print(f"  Random features mean AUC:      {rf_mean:.4f}")
    print(f"  Random features std:           {rf_std:.4f}")
    print(f"  Random features max:           {rf_max:.4f}")
    print(f"  Z-score:                       {rf_z:.2f}")
    print(f"  P(random >= observed):         {rf_p:.4f}")
    print(f"  Verdict:                       {'PASS' if rf_p < 0.05 else 'FAIL'}")

    # ============================================================
    # Test 3: Frequency-Only Baseline (form_length as proxy)
    # ============================================================
    print("\n" + "=" * 70)
    print("[3/4] FREQUENCY-ONLY BASELINE")
    print("  Using only form_length as a feature (no phonological info)...")
    print("=" * 70)

    # Extract form_length column only
    form_length_idx = feature_cols.index("form_length")
    X_freq = X[:, form_length_idx:form_length_idx+1]

    freq_auc = evaluate_cv_auc_multiseed(
        X_freq, y,
        lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, random_state=42,
            class_weight="balanced", n_jobs=-1,
        ),
        n_seeds=10, n_folds=5
    )

    # Also try LR with form_length only
    X_freq_scaled = StandardScaler().fit_transform(X_freq)
    freq_auc_lr = evaluate_cv_auc_multiseed(
        X_freq_scaled, y, lr_fn, n_seeds=10, n_folds=5
    )

    auc_lift = observed_auc - freq_auc

    print(f"\n  --- Frequency-Only Results ---")
    print(f"  Full model AUC (RF):          {observed_auc:.4f}")
    print(f"  form_length only AUC (RF):    {freq_auc:.4f}")
    print(f"  form_length only AUC (LR):    {freq_auc_lr:.4f}")
    print(f"  AUC lift from full model:     {auc_lift:+.4f}")
    print(f"  Verdict:                      {'PASS' if auc_lift > 0.05 else 'MARGINAL' if auc_lift > 0.02 else 'FAIL'}")
    print(f"  (PASS if lift > 0.05, meaning phonological features add >5% AUC)")

    # ============================================================
    # Test 4: Feature-importance sanity check — language_cognacy_coverage alone
    # ============================================================
    print("\n" + "=" * 70)
    print("[4/4] LANGUAGE COGNACY COVERAGE BASELINE")
    print("  Testing AUC using only the top SHAP feature (potential circularity)...")
    print("=" * 70)

    lcov_idx = feature_cols.index("language_cognacy_coverage")
    X_lcov = X[:, lcov_idx:lcov_idx+1]

    lcov_auc = evaluate_cv_auc_multiseed(
        X_lcov, y,
        lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, random_state=42,
            class_weight="balanced", n_jobs=-1,
        ),
        n_seeds=10, n_folds=5
    )

    # Without language_cognacy_coverage
    lcov_exclude_cols = [i for i, c in enumerate(feature_cols) if c != "language_cognacy_coverage"]
    X_no_lcov = X[:, lcov_exclude_cols]

    no_lcov_auc = evaluate_cv_auc_multiseed(
        X_no_lcov, y,
        lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, random_state=42,
            class_weight="balanced", n_jobs=-1,
        ),
        n_seeds=10, n_folds=5
    )

    lcov_contribution = observed_auc - no_lcov_auc

    print(f"\n  --- Cognacy Coverage Baseline ---")
    print(f"  Full model AUC:                    {observed_auc:.4f}")
    print(f"  language_cognacy_coverage only:     {lcov_auc:.4f}")
    print(f"  Without language_cognacy_coverage:  {no_lcov_auc:.4f}")
    print(f"  Contribution of lcov to AUC:        {lcov_contribution:+.4f}")
    print(f"  (Positive means removing lcov hurts. Large value = circularity concern)")

    # ============================================================
    # GRAND SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("GRAND SUMMARY — ADV-4 Substrate Noise Permutation Test")
    print("=" * 70)

    test1_pass = p_value < 0.05
    test2_pass = rf_p < 0.05
    test3_pass = auc_lift > 0.05
    test3_marginal = auc_lift > 0.02

    tests_passed = sum([test1_pass, test2_pass, test3_pass])

    print(f"\n  Test 1: Label Permutation      {'PASS' if test1_pass else 'FAIL'}  "
          f"(p={p_value:.4f}, z={z_score:.2f})")
    print(f"  Test 2: Random Features        {'PASS' if test2_pass else 'FAIL'}  "
          f"(p={rf_p:.4f}, z={rf_z:.2f})")
    print(f"  Test 3: Frequency-Only Lift    {'PASS' if test3_pass else 'MARGINAL' if test3_marginal else 'FAIL'}  "
          f"(lift={auc_lift:+.4f})")
    print(f"  Test 4: Circularity Check      lcov_only={lcov_auc:.4f}, "
          f"no_lcov={no_lcov_auc:.4f}")

    # Overall verdict
    if test1_pass and test2_pass and test3_pass:
        verdict = "PASS"
        explanation = (
            f"Substrate detection is NOT noise. Observed AUC ({observed_auc:.4f}) is "
            f"{z_score:.1f} SDs above permuted mean ({perm_mean:.4f}), "
            f"p={p_value:.4f}. Real features outperform random features. "
            f"Phonological features add +{auc_lift:.3f} AUC beyond form_length alone."
        )
    elif test1_pass and test2_pass:
        verdict = "CONDITIONAL PASS"
        explanation = (
            f"Substrate detection is significantly above noise (p={p_value:.4f}), "
            f"but much of the signal comes from form_length alone "
            f"(lift only {auc_lift:+.4f}). The phonological fingerprint is real "
            f"but thin."
        )
    elif test1_pass:
        verdict = "WEAK PASS"
        explanation = (
            f"Labels predict better than chance (p={p_value:.4f}), but the signal "
            f"may be driven by simple features rather than genuine phonological "
            f"substrate detection."
        )
    else:
        verdict = "FAIL"
        explanation = (
            f"Substrate detection is NOT significantly better than noise "
            f"(p={p_value:.4f}). The AUC=0.760 may be a statistical artifact. "
            f"L4 evidence from ML substrate detection is UNRELIABLE."
        )

    print(f"\n  >>> OVERALL VERDICT: {verdict}")
    print(f"  >>> {explanation}")

    # Circularity caveat
    if lcov_auc > 0.65:
        print(f"\n  CAVEAT: language_cognacy_coverage alone achieves AUC={lcov_auc:.4f}.")
        print(f"  This feature correlates with labeling process (languages with lower")
        print(f"  ABVD coverage have more residuals by definition). The purely")
        print(f"  phonological signal (without lcov) achieves AUC={no_lcov_auc:.4f}.")
        if no_lcov_auc < 0.70:
            print(f"  WARNING: Without lcov, AUC drops below 0.70. The claimed")
            print(f"  'phonological fingerprint' is partly driven by language-level")
            print(f"  coverage differences, not word-level phonology.")

    # ============================================================
    # Save results
    # ============================================================
    results = {
        "experiment": "E085_adv4_substrate_noise",
        "date": "2026-03-13",
        "observed_auc_rf": round(float(observed_auc), 4),
        "observed_auc_lr": round(float(observed_auc_lr), 4),
        "e027_reported_auc_xgboost": OBSERVED_AUC_XGBOOST,
        "e027_reported_auc_rf": OBSERVED_AUC_RF,
        "test1_label_permutation": {
            "n_permutations": N_PERMUTATIONS,
            "permuted_mean": round(float(perm_mean), 4),
            "permuted_std": round(float(perm_std), 4),
            "permuted_95th": round(float(perm_95th), 4),
            "permuted_99th": round(float(perm_99th), 4),
            "permuted_max": round(float(perm_max), 4),
            "z_score": round(float(z_score), 2),
            "empirical_p": round(float(p_value), 4),
            "pass": bool(test1_pass),
        },
        "test2_random_features": {
            "n_runs": N_RANDOM_RUNS,
            "random_mean_auc": round(float(rf_mean), 4),
            "random_std_auc": round(float(rf_std), 4),
            "random_max_auc": round(float(rf_max), 4),
            "z_score": round(float(rf_z), 2),
            "p_value": round(float(rf_p), 4),
            "pass": bool(test2_pass),
        },
        "test3_frequency_only": {
            "form_length_only_auc_rf": round(float(freq_auc), 4),
            "form_length_only_auc_lr": round(float(freq_auc_lr), 4),
            "auc_lift": round(float(auc_lift), 4),
            "pass": bool(test3_pass),
            "marginal": bool(test3_marginal),
        },
        "test4_circularity": {
            "lcov_only_auc": round(float(lcov_auc), 4),
            "no_lcov_auc": round(float(no_lcov_auc), 4),
            "lcov_contribution": round(float(lcov_contribution), 4),
        },
        "verdict": verdict,
        "explanation": explanation,
        "permuted_auc_distribution": [round(float(a), 4) for a in permuted_aucs],
    }

    with open(OUT / "adv4_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {OUT / 'adv4_results.json'}")

    # Also save a compact summary
    summary = {k: v for k, v in results.items() if k != "permuted_auc_distribution"}
    with open(OUT / "adv4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {OUT / 'adv4_summary.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
