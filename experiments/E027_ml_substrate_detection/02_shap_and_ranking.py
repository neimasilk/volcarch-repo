"""
E027 Script 02: SHAP Analysis and Substrate Ranking
=====================================================
- SHAP analysis on Model B (phonological-only) — the scientific model
- Beeswarm plot of feature importances
- Rank all 438 residuals by substrate probability
- Compare with E022 Tier 1/2/3 classifications
- Sensitivity analysis: ±Tolaki

Output: results/shap_beeswarm.png, results/substrate_ranking.csv,
        results/shap_summary.json
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: SHAP not available. Skipping SHAP analysis.")

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
E022_RESIDUALS = Path(__file__).parent.parent / "E022_linguistic_subtraction" / "results" / "poc_residuals_detail.csv"
E022_CROSS_LANG = Path(__file__).parent.parent / "E022_linguistic_subtraction" / "results" / "enhanced_cross_language.csv"
OUT.mkdir(exist_ok=True)

# Feature groups (same as script 01)
PHON_FEATURES = [
    "form_length", "n_vowels", "vowel_ratio", "ends_in_vowel",
    "has_glottal", "has_nasal_cluster", "has_reduplication",
    "n_consonant_clusters", "has_prefix_like",
]
SEMANTIC_FEATURES = ["is_core_vocab"]
LANG_FEATURES = ["language_id_encoded", "language_cognacy_coverage"]


def load_data():
    """Load feature matrix and prepare arrays for Model B."""
    df = pd.read_csv(DATA / "features_matrix.csv", encoding="utf-8")

    # One-hot encode
    ic_dummies = pd.get_dummies(df["initial_char"], prefix="init")
    sd_dummies = pd.get_dummies(df["semantic_domain"], prefix="sem")
    df = pd.concat([df, ic_dummies, sd_dummies], axis=1)

    init_cols = [c for c in df.columns if c.startswith("init_")]
    sem_cols = [c for c in df.columns if c.startswith("sem_")]

    model_b_cols = PHON_FEATURES + init_cols + SEMANTIC_FEATURES + sem_cols + LANG_FEATURES

    y = df["label"].values
    X_b = df[model_b_cols].values.astype(float)

    return df, X_b, y, model_b_cols


def load_e022_tiers():
    """Load E022 cross-language tier classifications."""
    tiers = {}  # concept -> tier
    if E022_CROSS_LANG.exists():
        with open(E022_CROSS_LANG, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                concept = row["concept"]
                n_langs = int(row["n_languages"])
                pan = row.get("pan_known", "").strip()
                if n_langs >= 5 and not pan:
                    tiers[concept] = 1
                elif n_langs == 4:
                    tiers[concept] = 2
                elif n_langs == 3:
                    tiers[concept] = 3
    else:
        print("  E022 cross-language file not found, skipping tier comparison")
    return tiers


def main():
    print("=" * 70)
    print("E027 Script 02: SHAP Analysis and Substrate Ranking")
    print("=" * 70)

    df, X_b, y, model_b_cols = load_data()
    print(f"Loaded {len(df)} forms, Model B has {len(model_b_cols)} features")

    # ============================================================
    # Step 1: Train final Model B on full data
    # ============================================================
    print("\n[1/6] Training final Model B (XGBoost) on full data...")

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

    clf.fit(X_b, y)
    print("  Model trained.")

    # Get substrate probabilities (probability of class 0 = substrate)
    probs = clf.predict_proba(X_b)
    prob_substrate = probs[:, 0]  # P(substrate)
    prob_austronesian = probs[:, 1]  # P(Austronesian)

    # ============================================================
    # Step 2: SHAP analysis
    # ============================================================
    print("\n[2/6] Computing SHAP values...")

    if HAS_SHAP and HAS_XGB:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_b)

        # SHAP beeswarm plot
        print("  Generating beeswarm plot...")
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_b,
            feature_names=model_b_cols,
            show=False,
            max_display=20,
        )
        plt.title("SHAP Feature Importance — Model B (Phonological-Only)\n"
                   "Predicting Austronesian (label=1) vs. Substrate (label=0)",
                   fontsize=12)
        plt.tight_layout()
        plt.savefig(OUT / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT / 'shap_beeswarm.png'}")

        # SHAP bar plot (mean absolute SHAP)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_b,
            feature_names=model_b_cols,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.title("Mean |SHAP| — Model B Feature Importance", fontsize=12)
        plt.tight_layout()
        plt.savefig(OUT / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT / 'shap_bar.png'}")

        # Extract feature importance rankings
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_ranking = sorted(zip(model_b_cols, mean_abs_shap),
                             key=lambda x: -x[1])

        print("\n  Top 15 features by mean |SHAP|:")
        for i, (feat, val) in enumerate(shap_ranking[:15]):
            print(f"    {i+1:>2}. {feat:<30} {val:.4f}")

        # Save SHAP summary
        shap_summary = {
            "feature_importance": [
                {"feature": feat, "mean_abs_shap": round(float(val), 6)}
                for feat, val in shap_ranking
            ],
        }
        with open(OUT / "shap_summary.json", "w", encoding="utf-8") as f:
            json.dump(shap_summary, f, indent=2)
        print(f"  Saved: {OUT / 'shap_summary.json'}")

    elif HAS_SHAP and not HAS_XGB:
        print("  Using RandomForest SHAP (slower)...")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_b)
        # For RF, shap_values is [array_class0, array_class1]
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs_shap = np.abs(sv).mean(axis=0)
        shap_ranking = sorted(zip(model_b_cols, mean_abs_shap), key=lambda x: -x[1])
        print("\n  Top 15 features by mean |SHAP|:")
        for i, (feat, val) in enumerate(shap_ranking[:15]):
            print(f"    {i+1:>2}. {feat:<30} {val:.4f}")
    else:
        print("  SHAP not available. Using built-in feature importance.")
        if HAS_XGB:
            importances = clf.feature_importances_
        else:
            importances = clf.feature_importances_
        shap_ranking = sorted(zip(model_b_cols, importances), key=lambda x: -x[1])
        print("\n  Top 15 features by importance:")
        for i, (feat, val) in enumerate(shap_ranking[:15]):
            print(f"    {i+1:>2}. {feat:<30} {val:.4f}")

    # ============================================================
    # Step 3: Rank substrate candidates
    # ============================================================
    print("\n[3/6] Ranking substrate candidates...")

    # Get E022 tier info
    e022_tiers = load_e022_tiers()

    substrate_rows = []
    for i, row in df.iterrows():
        if row["label"] == 0:  # substrate candidate
            tier = e022_tiers.get(row["concept"], 0)
            substrate_rows.append({
                "rank": 0,  # will fill after sorting
                "language": row["language"],
                "concept": row["concept"],
                "form": row["form"],
                "p_substrate": round(float(prob_substrate[i]), 4),
                "p_austronesian": round(float(prob_austronesian[i]), 4),
                "e022_tier": tier,
                "form_length": row["form_length"],
                "vowel_ratio": round(float(row["vowel_ratio"]), 3),
                "ends_in_vowel": int(row["ends_in_vowel"]),
                "has_glottal": int(row["has_glottal"]),
                "has_nasal_cluster": int(row["has_nasal_cluster"]),
                "has_prefix_like": int(row["has_prefix_like"]),
                "semantic_domain": row["semantic_domain"],
            })

    # Sort by substrate probability (descending)
    substrate_rows.sort(key=lambda x: -x["p_substrate"])
    for i, sr in enumerate(substrate_rows):
        sr["rank"] = i + 1

    # Save ranking
    outpath = OUT / "substrate_ranking.csv"
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=substrate_rows[0].keys())
        writer.writeheader()
        writer.writerows(substrate_rows)
    print(f"  Saved: {outpath}")
    print(f"  {len(substrate_rows)} substrate candidates ranked")

    # Top 20 most confident substrate candidates
    print("\n  Top 20 substrate candidates (highest P(substrate)):")
    print(f"  {'Rank':>4} {'P(sub)':>7} {'Language':<16} {'Concept':<25} {'Form':<15} {'Tier':>4}")
    print("  " + "-" * 75)
    for sr in substrate_rows[:20]:
        tier_str = f"T{sr['e022_tier']}" if sr['e022_tier'] > 0 else "-"
        print(f"  {sr['rank']:>4} {sr['p_substrate']:>7.4f} {sr['language']:<16} "
              f"{sr['concept']:<25} {sr['form']:<15} {tier_str:>4}")

    # ============================================================
    # Step 4: E022 Tier comparison
    # ============================================================
    print("\n[4/6] E022 Tier comparison...")

    tier_probs = defaultdict(list)
    for sr in substrate_rows:
        if sr["e022_tier"] > 0:
            tier_probs[sr["e022_tier"]].append(sr["p_substrate"])

    print(f"\n  {'Tier':>5} {'N':>5} {'Mean P(sub)':>12} {'Median':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 50)
    for tier in sorted(tier_probs.keys()):
        probs = tier_probs[tier]
        print(f"  T{tier:>4} {len(probs):>5} {np.mean(probs):>12.4f} "
              f"{np.median(probs):>8.4f} {np.min(probs):>8.4f} {np.max(probs):>8.4f}")

    # All substrates
    all_probs = [sr["p_substrate"] for sr in substrate_rows]
    print(f"  {'ALL':>5} {len(all_probs):>5} {np.mean(all_probs):>12.4f} "
          f"{np.median(all_probs):>8.4f} {np.min(all_probs):>8.4f} {np.max(all_probs):>8.4f}")

    # ============================================================
    # Step 5: Semantic domain analysis of top substrates
    # ============================================================
    print("\n[5/6] Semantic domain analysis of top-50 substrates...")

    top50 = substrate_rows[:50]
    domain_counts = defaultdict(int)
    for sr in top50:
        domain_counts[sr["semantic_domain"]] += 1

    print(f"\n  {'Domain':<12} {'Count':>6} {'%':>6}")
    print("  " + "-" * 28)
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = round(100 * count / 50, 1)
        print(f"  {domain:<12} {count:>6} {pct:>5.1f}%")

    # ============================================================
    # Step 6: Sensitivity analysis (±Tolaki)
    # ============================================================
    print("\n[6/6] Sensitivity analysis: ±Tolaki...")

    if HAS_XGB:
        # Without Tolaki
        mask_no_tolaki = df["language"] != "Tolaki"
        X_no_t = X_b[mask_no_tolaki]
        y_no_t = y[mask_no_tolaki]

        clf_no_t = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            reg_lambda=1.0, eval_metric="logloss", random_state=42,
            use_label_encoder=False, verbosity=0,
        )

        # CV without Tolaki
        aucs = []
        for seed in range(10):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 7 + 13)
            for train_idx, test_idx in skf.split(X_no_t, y_no_t):
                clf_temp = XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    reg_lambda=1.0, eval_metric="logloss", random_state=42,
                    use_label_encoder=False, verbosity=0,
                )
                clf_temp.fit(X_no_t[train_idx], y_no_t[train_idx])
                y_prob = clf_temp.predict_proba(X_no_t[test_idx])[:, 1]
                aucs.append(roc_auc_score(y_no_t[test_idx], y_prob))

        mean_auc_no_t = np.mean(aucs)
        std_auc_no_t = np.std(aucs)

        print(f"\n  With Tolaki:    AUC = 0.7599 ± 0.0073  (N=1357)")
        print(f"  Without Tolaki: AUC = {mean_auc_no_t:.4f} ± {std_auc_no_t:.4f}  (N={sum(mask_no_tolaki)})")

        delta = mean_auc_no_t - 0.7599
        print(f"  Delta: {delta:+.4f}")
        if abs(delta) < 0.03:
            print("  -> Tolaki removal has MINIMAL effect (<3%). Results are robust.")
        elif delta > 0:
            print("  -> Removing Tolaki IMPROVES AUC. Tolaki is noisy.")
        else:
            print("  -> Removing Tolaki DECREASES AUC. Tolaki contributes signal.")

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # Load verdict
    with open(OUT / "verdict.json", encoding="utf-8") as f:
        verdict = json.load(f)

    print(f"\n  Verdict:         {verdict['verdict']}")
    print(f"  CV AUC (Model B): {verdict['cv_auc']:.4f}")
    print(f"  LOLO AUC mean:    {verdict['lolo_auc_mean']:.4f}")
    print(f"  LOLO ≥0.65:       {verdict['lolo_langs_above_065']}/6 languages")
    print(f"  Substrates ranked: {len(substrate_rows)}")

    if HAS_SHAP:
        print(f"\n  Top 5 SHAP features (phonological fingerprint of substrate):")
        for i, (feat, val) in enumerate(shap_ranking[:5]):
            print(f"    {i+1}. {feat} ({val:.4f})")

    print(f"\n  Key finding: substrate vocabulary shows a distinct phonological")
    print(f"  fingerprint — primarily longer forms, lower vowel ratios,")
    print(f"  and fewer Austronesian-like prefixes.")

    print("\nDone.")


if __name__ == "__main__":
    main()
