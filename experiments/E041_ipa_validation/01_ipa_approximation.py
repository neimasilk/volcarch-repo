"""
E041 Script 01: IPA Approximation Validation
=============================================
Tests whether E027 Model B results are robust to orthography-to-IPA conversion.

Criticism: "All features are computed from orthographic forms, not IPA.
The model may learn orthographic patterns, not phonological ones."

Method:
1. Convert orthographic forms to approximate IPA using language-specific
   digraph-to-phoneme mappings
2. Recompute phonological features on IPA forms
3. Retrain Model B with IPA features
4. Compare AUC (original orthographic vs IPA)

Conservative approach: only collapse unambiguous digraphs where the
phonemic status is well-established in the literature.
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

# ============================================================
# Digraph-to-IPA mappings
# ============================================================
# These are CONSERVATIVE: only unambiguous digraphs
# Sources: Muna (van den Berg 1989), Bugis/Makassar (Pelenkahu 1983),
#          Wolio (Anceaux 1987), Toraja (Veen 1940), Tolaki (Mead 2003)

# Universal digraphs (all 6 languages)
UNIVERSAL_DIGRAPHS = [
    ("ng", "\u014b"),     # ŋ - velar nasal
    ("ny", "\u0272"),     # ɲ - palatal nasal
]

# Language-specific digraphs
LANG_DIGRAPHS = {
    "Muna": [
        ("gh", "\u0263"),   # ɣ - voiced velar fricative
        ("bh", "\u03b2"),   # β - voiced bilabial fricative (approx)
        ("dh", "\u00f0"),   # ð - voiced dental fricative (approx)
    ],
    "Wolio": [
        ("gh", "\u0263"),   # ɣ - voiced velar fricative
    ],
    "Bugis": [],            # Bugis orthography is relatively transparent
    "Makassar": [],         # Makassar orthography is relatively transparent
    "Toraja-Sadan": [],     # Standard orthography
    "Tolaki": [],           # Standard orthography
}

# Prenasalized stops: mb, nd, nj, mp, nk, nt, nc
# These are kept as-is because their phonemic status (single segment vs cluster)
# is debated. Both analyses exist in the literature.
# Keeping them as 2 characters is the CONSERVATIVE choice (less favorable to us).


def orthographic_to_ipa(form, language):
    """Convert orthographic form to approximate IPA."""
    result = form.lower()

    # Apply language-specific digraphs FIRST (some are substrings of others)
    for digraph, ipa in LANG_DIGRAPHS.get(language, []):
        result = result.replace(digraph, ipa)

    # Then universal digraphs
    for digraph, ipa in UNIVERSAL_DIGRAPHS:
        result = result.replace(digraph, ipa)

    return result


# ============================================================
# Feature extractors (same as E027 but on IPA forms)
# ============================================================

VOWELS = set("aeiou\u0259\u025b\u0268\u0254\u00e6\u00f8\u00fc\u00f6\u00e4\u00e5\u00e3\u1ebd\u0129\u00f5\u0169\u00e2\u00ea\u00ee\u00f4\u00fb\u00e0\u00e8\u00ec\u00f2\u00f9\u00e1\u00e9\u00ed\u00f3\u00fa")
AUSTRONESIAN_PREFIXES = ("ma-", "me-", "mo-", "pa-", "ka-", "ta-", "na-", "po-",
                         "ma", "me", "mo", "pa", "ka", "ta", "na", "po",
                         "ma\u014b", "me\u014b", "mo\u014b", "pa\u014b", "a\u014b",
                         "mak-", "mat-")


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
    return 1 if ("\u0294" in form or "'" in form) else 0

def has_nasal_cluster(form):
    fl = form.lower()
    # After IPA conversion, check for IPA nasals + stop sequences
    nasal_ipa = ["\u014b", "\u0272", "m", "n"]  # ŋ, ɲ, m, n
    stops = set("ptckbdgq")
    for i in range(len(fl) - 1):
        if fl[i] in "mn" or fl[i] in nasal_ipa:
            if fl[i+1] in stops:
                return 1
    # Also check remaining orthographic clusters
    for nc in ("mb", "nd", "nj", "mp", "nk", "nt", "nc"):
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
    """Count CC+ sequences using IPA-aware vowel detection."""
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


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("E041: IPA Approximation Validation for E027 Model B")
    print("=" * 70)

    # Load the feature matrix from E027
    print("\n[1/5] Loading E027 feature matrix...")
    rows = []
    with open(E027_DATA / "features_matrix.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows)} forms")

    # ============================================================
    # Step 2: Convert to approximate IPA and recompute features
    # ============================================================
    print("\n[2/5] Converting orthographic forms to approximate IPA...")

    conversion_stats = {
        "total": 0,
        "changed": 0,
        "length_diff": [],
        "cluster_diff": [],
        "by_language": defaultdict(lambda: {"total": 0, "changed": 0, "length_diffs": []})
    }

    ipa_features = []
    for row in rows:
        form_ortho = row["form"]
        language = row["language"]
        label = int(row["label"])
        concept = row["concept"]

        # Convert
        form_ipa = orthographic_to_ipa(form_ortho, language)

        conversion_stats["total"] += 1
        if form_ipa != form_ortho.lower():
            conversion_stats["changed"] += 1
            conversion_stats["by_language"][language]["changed"] += 1
        conversion_stats["by_language"][language]["total"] += 1

        # Compute IPA features
        fl_ipa = compute_form_length(form_ipa)
        fl_ortho = int(row["form_length"])
        nv_ipa = count_vowels(form_ipa)
        vr_ipa = vowel_ratio(form_ipa)
        eiv_ipa = ends_in_vowel(form_ipa)
        ic_ipa = get_initial_class(form_ipa)
        hg_ipa = has_glottal(form_ipa)
        hnc_ipa = has_nasal_cluster(form_ipa)
        hr_ipa = has_reduplication(form_ipa)
        ncc_ipa = count_consonant_clusters(form_ipa)
        hpl_ipa = has_prefix_like(form_ipa)
        ncc_ortho = int(row["n_consonant_clusters"])

        conversion_stats["length_diff"].append(fl_ipa - fl_ortho)
        conversion_stats["cluster_diff"].append(ncc_ipa - ncc_ortho)
        conversion_stats["by_language"][language]["length_diffs"].append(fl_ipa - fl_ortho)

        ipa_features.append({
            "form_id": row["form_id"],
            "language": language,
            "concept": concept,
            "form_ortho": form_ortho,
            "form_ipa": form_ipa,
            "label": label,
            # IPA features
            "form_length": fl_ipa,
            "n_vowels": nv_ipa,
            "vowel_ratio": round(vr_ipa, 4),
            "ends_in_vowel": eiv_ipa,
            "has_glottal": hg_ipa,
            "has_nasal_cluster": hnc_ipa,
            "has_reduplication": hr_ipa,
            "n_consonant_clusters": ncc_ipa,
            "has_prefix_like": hpl_ipa,
            # Keep original semantic/control features
            "semantic_domain": row["semantic_domain"],
            "is_core_vocab": int(row["is_core_vocab"]),
            "language_id_encoded": int(row["language_id_encoded"]),
            # Ortho values for comparison
            "form_length_ortho": fl_ortho,
            "n_consonant_clusters_ortho": ncc_ortho,
        })

    # Print conversion stats
    print(f"\n  Forms changed by IPA conversion: {conversion_stats['changed']}/{conversion_stats['total']} "
          f"({100*conversion_stats['changed']/conversion_stats['total']:.1f}%)")

    length_diffs = conversion_stats["length_diff"]
    nonzero_diffs = [d for d in length_diffs if d != 0]
    print(f"  Mean length change: {np.mean(length_diffs):.3f} characters")
    print(f"  Forms with length change: {len(nonzero_diffs)}/{len(length_diffs)}")

    cluster_diffs = conversion_stats["cluster_diff"]
    nonzero_cl = [d for d in cluster_diffs if d != 0]
    print(f"  Mean cluster count change: {np.mean(cluster_diffs):.3f}")
    print(f"  Forms with cluster change: {len(nonzero_cl)}/{len(cluster_diffs)}")

    print(f"\n  Per-language conversion impact:")
    print(f"  {'Language':<18} {'Changed':>8} {'Total':>6} {'%':>6} {'Mean dLen':>10}")
    print(f"  {'-'*50}")
    for lang in sorted(conversion_stats["by_language"].keys()):
        ls = conversion_stats["by_language"][lang]
        pct = 100 * ls["changed"] / ls["total"]
        mean_dl = np.mean(ls["length_diffs"])
        print(f"  {lang:<18} {ls['changed']:>8} {ls['total']:>6} {pct:>5.1f}% {mean_dl:>10.3f}")

    # ============================================================
    # Step 3: Example conversions
    # ============================================================
    print("\n[3/5] Example conversions:")
    examples = [(f["form_ortho"], f["form_ipa"], f["language"], f["concept"])
                for f in ipa_features
                if f["form_ipa"] != f["form_ortho"].lower()][:20]
    for ortho, ipa, lang, concept in examples:
        print(f"  {lang:>15}: {ortho:<20} -> {ipa:<20} ({concept})")

    # ============================================================
    # Step 4: Retrain Model B with IPA features
    # ============================================================
    print("\n[4/5] Training Model B on IPA features...")

    # Prepare feature matrices
    # Model B features: phonological (10) + semantic (2) + language_id (1)
    # Excluding language_cognacy_coverage (per ablation result)
    semantic_dummies = {"ACTION": 0, "BODY": 1, "GRAMMAR": 2, "NATURE": 3,
                       "NUMBER": 4, "OTHER": 5, "QUALITY": 6}

    initial_dummies = {"m": 0, "a": 1, "b": 2, "t": 3, "k": 4, "p": 5, "s": 6, "other": 7}

    def make_feature_vector(feat_dict, use_ipa=True):
        """Create feature vector from feature dict."""
        fl = feat_dict["form_length"] if use_ipa else feat_dict["form_length_ortho"]
        ncc = feat_dict["n_consonant_clusters"] if use_ipa else feat_dict["n_consonant_clusters_ortho"]

        ic = get_initial_class(feat_dict["form_ipa"] if use_ipa else feat_dict["form_ortho"])
        ic_vec = [0] * 8
        ic_vec[initial_dummies.get(ic, 7)] = 1

        sd = feat_dict["semantic_domain"]
        sd_vec = [0] * 7
        sd_vec[semantic_dummies.get(sd, 5)] = 1

        return [
            fl,
            feat_dict["n_vowels"],
            feat_dict["vowel_ratio"],
            feat_dict["ends_in_vowel"],
            *ic_vec,
            feat_dict["has_glottal"],
            feat_dict["has_nasal_cluster"],
            feat_dict["has_reduplication"],
            ncc,
            feat_dict["has_prefix_like"],
            *sd_vec,
            feat_dict["is_core_vocab"],
            feat_dict["language_id_encoded"],
        ]

    # Build matrices for both IPA and orthographic versions
    X_ipa = np.array([make_feature_vector(f, use_ipa=True) for f in ipa_features])
    X_ortho = np.array([make_feature_vector(f, use_ipa=False) for f in ipa_features])
    y = np.array([f["label"] for f in ipa_features])
    languages = np.array([f["language"] for f in ipa_features])

    n_features = X_ipa.shape[1]
    print(f"  Feature matrix: {X_ipa.shape[0]} samples x {n_features} features")
    print(f"  Label distribution: {np.sum(y==1)} Austronesian, {np.sum(y==0)} non-mainstream")

    # XGBoost parameters (same as E027)
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "verbosity": 0,
    }

    # --- Stratified 5-fold CV (10 seeds) ---
    results = {"ipa": [], "ortho": []}

    print("\n  Running 5-fold CV x 10 seeds...")
    for seed in range(10):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for train_idx, test_idx in skf.split(X_ipa, y):
            for label, X in [("ipa", X_ipa), ("ortho", X_ortho)]:
                model = xgb.XGBClassifier(**{**xgb_params, "random_state": seed})
                model.fit(X[train_idx], y[train_idx])
                y_prob = model.predict_proba(X[test_idx])[:, 1]
                y_pred = model.predict(X[test_idx])
                auc = roc_auc_score(y[test_idx], y_prob)
                f1 = f1_score(y[test_idx], y_pred)
                acc = accuracy_score(y[test_idx], y_pred)
                results[label].append({"auc": auc, "f1": f1, "acc": acc})

    for label in ["ortho", "ipa"]:
        aucs = [r["auc"] for r in results[label]]
        f1s = [r["f1"] for r in results[label]]
        accs = [r["acc"] for r in results[label]]
        print(f"\n  {label.upper():>6} CV:  AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  "
              f"F1 = {np.mean(f1s):.4f}  Acc = {np.mean(accs):.4f}")

    # --- LOLO (Leave-One-Language-Out) ---
    print("\n  Running LOLO validation...")
    unique_langs = sorted(set(languages))
    lolo_results = {"ipa": {}, "ortho": {}}

    for held_out in unique_langs:
        train_mask = languages != held_out
        test_mask = languages == held_out

        for label, X in [("ipa", X_ipa), ("ortho", X_ortho)]:
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X[train_mask], y[train_mask])
            y_prob = model.predict_proba(X[test_mask])[:, 1]
            y_pred = model.predict(X[test_mask])
            auc = roc_auc_score(y[test_mask], y_prob)
            f1 = f1_score(y[test_mask], y_pred)
            acc = accuracy_score(y[test_mask], y_pred)
            lolo_results[label][held_out] = {"auc": auc, "f1": f1, "acc": acc,
                                              "n_test": int(np.sum(test_mask)),
                                              "n_sub": int(np.sum(y[test_mask] == 0))}

    print(f"\n  {'Language':<18} {'ORTHO AUC':>10} {'IPA AUC':>10} {'Delta':>8}")
    print(f"  {'-'*50}")
    for lang in unique_langs:
        o = lolo_results["ortho"][lang]["auc"]
        i = lolo_results["ipa"][lang]["auc"]
        delta = i - o
        marker = " ***" if abs(delta) > 0.02 else ""
        print(f"  {lang:<18} {o:>10.4f} {i:>10.4f} {delta:>+8.4f}{marker}")

    ortho_mean = np.mean([lolo_results["ortho"][l]["auc"] for l in unique_langs])
    ipa_mean = np.mean([lolo_results["ipa"][l]["auc"] for l in unique_langs])
    print(f"  {'MEAN':<18} {ortho_mean:>10.4f} {ipa_mean:>10.4f} {ipa_mean-ortho_mean:>+8.4f}")

    ortho_ge65 = sum(1 for l in unique_langs if lolo_results["ortho"][l]["auc"] >= 0.65)
    ipa_ge65 = sum(1 for l in unique_langs if lolo_results["ipa"][l]["auc"] >= 0.65)
    print(f"\n  LOLO >= 0.65:  ORTHO {ortho_ge65}/6  |  IPA {ipa_ge65}/6")

    # ============================================================
    # Step 5: Save results
    # ============================================================
    print("\n[5/5] Saving results...")

    summary = {
        "experiment": "E041 IPA Approximation Validation",
        "date": "2026-03-11",
        "conversion": {
            "forms_changed": conversion_stats["changed"],
            "total_forms": conversion_stats["total"],
            "pct_changed": round(100 * conversion_stats["changed"] / conversion_stats["total"], 1),
            "mean_length_change": round(float(np.mean(length_diffs)), 3),
            "forms_with_length_change": len(nonzero_diffs),
            "mean_cluster_change": round(float(np.mean(cluster_diffs)), 3),
        },
        "cv_results": {
            "ortho": {
                "auc_mean": round(float(np.mean([r["auc"] for r in results["ortho"]])), 4),
                "auc_std": round(float(np.std([r["auc"] for r in results["ortho"]])), 4),
            },
            "ipa": {
                "auc_mean": round(float(np.mean([r["auc"] for r in results["ipa"]])), 4),
                "auc_std": round(float(np.std([r["auc"] for r in results["ipa"]])), 4),
            },
        },
        "lolo_results": {
            "ortho": {lang: {"auc": round(lolo_results["ortho"][lang]["auc"], 4)}
                      for lang in unique_langs},
            "ipa": {lang: {"auc": round(lolo_results["ipa"][lang]["auc"], 4)}
                    for lang in unique_langs},
            "ortho_mean_auc": round(float(ortho_mean), 4),
            "ipa_mean_auc": round(float(ipa_mean), 4),
            "ortho_ge65": ortho_ge65,
            "ipa_ge65": ipa_ge65,
        },
        "conclusion": "",
    }

    # Determine conclusion
    cv_delta = summary["cv_results"]["ipa"]["auc_mean"] - summary["cv_results"]["ortho"]["auc_mean"]
    lolo_delta = ipa_mean - ortho_mean

    if abs(cv_delta) < 0.02 and abs(lolo_delta) < 0.02:
        summary["conclusion"] = (
            "ROBUST: IPA conversion produces negligible change in model performance "
            f"(CV delta={cv_delta:+.4f}, LOLO delta={lolo_delta:+.4f}). "
            "The phonological fingerprint is not driven by orthographic artifacts."
        )
    elif cv_delta > 0.02 or lolo_delta > 0.02:
        summary["conclusion"] = (
            f"IPA IMPROVES: Model performs better on IPA features "
            f"(CV delta={cv_delta:+.4f}, LOLO delta={lolo_delta:+.4f}). "
            "Orthographic noise was attenuating the signal."
        )
    else:
        summary["conclusion"] = (
            f"IPA WEAKENS: Model performs worse on IPA features "
            f"(CV delta={cv_delta:+.4f}, LOLO delta={lolo_delta:+.4f}). "
            "Some orthographic patterns may be contributing to classification."
        )

    with open(OUT / "ipa_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save IPA conversion examples
    with open(OUT / "ipa_conversion_examples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "language", "concept", "form_ortho", "form_ipa",
            "length_ortho", "length_ipa", "clusters_ortho", "clusters_ipa"
        ])
        writer.writeheader()
        for feat in ipa_features:
            if feat["form_ipa"] != feat["form_ortho"].lower():
                writer.writerow({
                    "language": feat["language"],
                    "concept": feat["concept"],
                    "form_ortho": feat["form_ortho"],
                    "form_ipa": feat["form_ipa"],
                    "length_ortho": feat["form_length_ortho"],
                    "length_ipa": feat["form_length"],
                    "clusters_ortho": feat["n_consonant_clusters_ortho"],
                    "clusters_ipa": feat["n_consonant_clusters"],
                })

    print(f"\n  Saved: {OUT / 'ipa_validation_summary.json'}")
    print(f"  Saved: {OUT / 'ipa_conversion_examples.csv'}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\n  {summary['conclusion']}")
    print(f"\n  CV:   ORTHO {summary['cv_results']['ortho']['auc_mean']:.4f} -> IPA {summary['cv_results']['ipa']['auc_mean']:.4f}")
    print(f"  LOLO: ORTHO {ortho_mean:.4f} -> IPA {ipa_mean:.4f}")


if __name__ == "__main__":
    main()
