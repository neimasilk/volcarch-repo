#!/usr/bin/env python3
"""
E209 Phase 1, Step 03: Train baseline classifiers on extracted Sentinel-2 features.

Inputs:
  - data/features_s2.csv (from step 02)

Outputs:
  - results/classifier_baseline.json  # metrics per classifier
  - results/feature_importance.csv    # RF feature importances (+ SHAP if available)
  - results/cv_scores.csv             # per-fold CV scores
  - models/rf_baseline.joblib         # serialised Random Forest model
  - models/xgb_baseline.joblib        # serialised XGBoost model (if installed)

Design:
  - Binary classification: positive (class ≥1) vs negative (class ≤-1)
  - Hard positives (class=2) weighted 2× in loss
  - Pivot table: one row per site, with dry + wet features concatenated
    and seasonal deltas (wet - dry) computed for each feature
  - Models:
      1. Random Forest (scikit-learn, default + tuned)
      2. Gradient Boosting (sklearn or XGBoost if installed)
  - Cross-validation:
      a. Stratified K-fold (5-fold) on site classes — overall generalisation
      b. Leave-one-hard-positive-out — tests whether classifier identifies
         discovered-buried sites that were withheld from training
  - Reports:
      AUC, accuracy, precision/recall, per-class F1, feature importance

Usage:
  python 03_train_classifier.py
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

E209_DIR = Path(__file__).resolve().parents[1]
FEATURES_CSV = E209_DIR / "data" / "features_s2.csv"
RESULTS_DIR = E209_DIR / "results"
MODELS_DIR = E209_DIR / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


FEATURE_COLUMNS = [
    "ndvi_mean", "ndvi_std", "ndvi_center", "ndvi_ring", "ndvi_diff", "ndvi_lvar",
    "ndwi_mean", "ndwi_std", "ndwi_center", "ndwi_ring", "ndwi_diff", "ndwi_lvar",
    "msavi_mean", "msavi_center", "msavi_diff",
    "clay_ratio", "iron_oxide",
]


def load_features(path: Path):
    """Load features_s2.csv and pivot to one row per site with seasonal delta features."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Group by site_id
    by_site = defaultdict(dict)
    site_meta = {}
    for r in rows:
        sid = r["site_id"]
        season = r["season"]
        by_site[sid][season] = r
        site_meta[sid] = {
            "site_id": sid,
            "name": r["name"],
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "label": r["label"],
            "class": int(r["class"]),
        }

    # Build feature matrix
    X = []
    y = []
    sample_weights = []
    meta = []
    feature_names = []

    # Decide feature schema: for each feature, produce <feat>_dry, <feat>_wet, <feat>_delta
    for fname in FEATURE_COLUMNS:
        feature_names.extend([f"{fname}_dry", f"{fname}_wet", f"{fname}_delta"])

    for sid, seasons in by_site.items():
        meta_row = site_meta[sid]
        # Require both seasons for full feature vector
        if "dry" not in seasons or "wet" not in seasons:
            continue
        dry = seasons["dry"]
        wet = seasons["wet"]
        vec = []
        valid = True
        for fname in FEATURE_COLUMNS:
            try:
                d_val = float(dry[fname]) if dry[fname] not in ("", "nan") else np.nan
                w_val = float(wet[fname]) if wet[fname] not in ("", "nan") else np.nan
            except (ValueError, KeyError):
                d_val = np.nan
                w_val = np.nan
            delta = w_val - d_val if np.isfinite(d_val) and np.isfinite(w_val) else np.nan
            vec.extend([d_val, w_val, delta])
        X.append(vec)
        cls = meta_row["class"]
        y.append(1 if cls > 0 else 0)
        sample_weights.append(2.0 if cls == 2 else 1.0)
        meta.append(meta_row)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    sample_weights = np.array(sample_weights, dtype=float)
    return X, y, sample_weights, meta, feature_names


def impute_nan(X: np.ndarray) -> np.ndarray:
    """Simple column-mean imputation."""
    X = X.copy()
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = ~np.isfinite(col)
        X[mask, j] = col_means[j] if np.isfinite(col_means[j]) else 0.0
    return X


def train_and_evaluate(X: np.ndarray, y: np.ndarray,
                       sample_weights: np.ndarray,
                       meta: list, feature_names: list) -> dict:
    """Train RF + XGB baselines, report metrics."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, LeaveOneOut
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    Xi = impute_nan(X)

    results = {
        "n_samples": int(Xi.shape[0]),
        "n_features": int(Xi.shape[1]),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "classifiers": {},
    }

    # === Random Forest ===
    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Stratified K-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    fold_accs = []
    for fold, (tr, te) in enumerate(skf.split(Xi, y)):
        rf_fold = RandomForestClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        rf_fold.fit(Xi[tr], y[tr], sample_weight=sample_weights[tr])
        pred = rf_fold.predict_proba(Xi[te])[:, 1]
        try:
            auc = roc_auc_score(y[te], pred)
        except ValueError:
            auc = np.nan
        acc = accuracy_score(y[te], pred > 0.5)
        fold_aucs.append(auc)
        fold_accs.append(acc)
        print(f"  Fold {fold+1}: AUC={auc:.3f} Acc={acc:.3f}")

    print(f"  Mean AUC: {np.nanmean(fold_aucs):.3f} ± {np.nanstd(fold_aucs):.3f}")
    results["classifiers"]["rf_skfold"] = {
        "mean_auc": float(np.nanmean(fold_aucs)),
        "std_auc": float(np.nanstd(fold_aucs)),
        "mean_acc": float(np.mean(fold_accs)),
        "fold_aucs": [float(x) for x in fold_aucs],
    }

    # Leave-one-out on hard positives
    hp_indices = [i for i, m in enumerate(meta) if m["class"] == 2]
    print(f"\n  Leave-one-hard-positive-out ({len(hp_indices)} hard positives):")
    loo_scores = []
    for hp_i in hp_indices:
        tr_mask = np.ones(len(y), dtype=bool)
        tr_mask[hp_i] = False
        rf_loo = RandomForestClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        rf_loo.fit(Xi[tr_mask], y[tr_mask], sample_weight=sample_weights[tr_mask])
        score = rf_loo.predict_proba(Xi[hp_i:hp_i+1])[:, 1][0]
        loo_scores.append({"name": meta[hp_i]["name"], "score": float(score)})
        print(f"    {meta[hp_i]['name']:25s}: p(buried) = {score:.3f}")

    mean_hp_score = np.mean([s["score"] for s in loo_scores])
    print(f"  Mean p(buried) for held-out hard positives: {mean_hp_score:.3f}")
    print(f"  (should be >>0.5 if classifier generalises to discovered-buried signature)")

    results["classifiers"]["rf_skfold"]["leave_one_hp_out"] = loo_scores

    # Fit on full data for feature importance + saved model
    rf.fit(Xi, y, sample_weight=sample_weights)
    importances = rf.feature_importances_
    fi = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    print(f"\n  Top 10 features:")
    for name, imp in fi[:10]:
        print(f"    {name:25s} {imp:.4f}")

    # Save model
    try:
        import joblib
        joblib.dump(rf, MODELS_DIR / "rf_baseline.joblib")
        print(f"  Saved: {MODELS_DIR / 'rf_baseline.joblib'}")
    except ImportError:
        print("  joblib not available; model not saved")

    # Save feature importances
    with open(RESULTS_DIR / "feature_importance.csv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "importance"])
        for name, imp in fi:
            w.writerow([name, f"{imp:.6f}"])

    # === Gradient Boosting (sklearn) — secondary ===
    print("\n=== Gradient Boosting (sklearn) ===")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
    )
    gb_aucs = []
    for fold, (tr, te) in enumerate(skf.split(Xi, y)):
        gb_fold = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
        )
        gb_fold.fit(Xi[tr], y[tr], sample_weight=sample_weights[tr])
        pred = gb_fold.predict_proba(Xi[te])[:, 1]
        try:
            auc = roc_auc_score(y[te], pred)
        except ValueError:
            auc = np.nan
        gb_aucs.append(auc)
    print(f"  Mean AUC: {np.nanmean(gb_aucs):.3f} ± {np.nanstd(gb_aucs):.3f}")
    results["classifiers"]["gb_skfold"] = {
        "mean_auc": float(np.nanmean(gb_aucs)),
        "std_auc": float(np.nanstd(gb_aucs)),
    }

    return results


def main() -> None:
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run step 02 first.")
        sys.exit(1)

    print("E209 Step 03: Train baseline classifiers")
    print("=" * 60)

    X, y, sw, meta, feature_names = load_features(FEATURES_CSV)
    print(f"Loaded {X.shape[0]} sites with {X.shape[1]} features each")
    print(f"  Positive (archaeological): {y.sum()}")
    print(f"  Negative (control):        {(y == 0).sum()}")
    print(f"  Hard positive (double-weighted): {sum(1 for m in meta if m['class'] == 2)}")

    if X.shape[0] < 20:
        print("\nWARN: Very small dataset. Results will be preliminary only.")

    results = train_and_evaluate(X, y, sw, meta, feature_names)

    # Save full results
    with open(RESULTS_DIR / "classifier_baseline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"Results written to: {RESULTS_DIR / 'classifier_baseline.json'}")
    print(f"Next: scripts/04_predict_landscape.py (applies classifier to volcanic basins)")


if __name__ == "__main__":
    main()
