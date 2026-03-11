#!/usr/bin/env python3
"""
E037 — Prasasti Dating Model (ML on Undated Inscriptions)
==========================================================
Question: Can we predict the approximate date (century) of undated inscriptions
          using linguistic and content features from the DHARMA corpus?

Method:
  1. Train on 166 dated inscriptions using features from E023/E030/E035
  2. Features: word_count, language, keyword counts (indic/pre_indic/ambiguous),
     botanical keywords, specific ritual markers (hyang, manhuri, wuku)
  3. Models: Random Forest, XGBoost (regression → century prediction)
  4. Validation: Leave-one-out cross-validation (LOOCV) + temporal split
  5. Apply to ~102 undated inscriptions for estimated dating

This tests I-005 and feeds P5 (ritual temporal patterns) + P14 (Pararaton dating).

Author: VOLCARCH project (AI-assisted)
Date: 2026-03-10
"""

import sys
import io
import os
import json
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats

warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("E037 — Prasasti Dating Model")
print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════
# 1. LOAD AND MERGE DATA
# ═════════════════════════════════════════════════════════════════════════

print("\n[1] Loading and merging datasets...")

# E030: dated inscriptions with keyword features
dated_csv = os.path.join(REPO, "experiments", "E030_prasasti_temporal_nlp",
                         "results", "dated_inscriptions.csv")
df_dated = pd.read_csv(dated_csv)

# E023: full corpus (268 inscriptions) with classification
full_csv = os.path.join(REPO, "experiments", "E023_ritual_screening",
                        "results", "full_corpus_classification.csv")
df_full = pd.read_csv(full_csv)

# E035: botanical hits
bot_csv = os.path.join(REPO, "experiments", "E035_prasasti_botanical",
                       "results", "botanical_inscriptions.csv")
df_bot = pd.read_csv(bot_csv) if os.path.exists(bot_csv) else pd.DataFrame()

print(f"  Dated inscriptions: {len(df_dated)}")
print(f"  Full corpus: {len(df_full)}")
print(f"  Botanical hits: {len(df_bot)}")

# Merge: use full corpus as base, add dates from E030
df = df_full.copy()

# Create date lookup from E030
date_lookup = {}
for _, row in df_dated.iterrows():
    date_lookup[row['filename']] = row['year_ce']

df['year_ce'] = df['filename'].map(date_lookup)
df['has_date'] = df['year_ce'].notna()
df['century'] = df['year_ce'].apply(lambda y: int(y // 100) + 1 if pd.notna(y) else None)

# Add botanical features from E035
if not df_bot.empty:
    bot_lookup = {}
    for _, row in df_bot.iterrows():
        bot_lookup[row['filename']] = {
            'n_plants': row['n_plants'],
            'plants': row['plants'],
        }

    df['n_botanical'] = df['filename'].map(
        lambda f: bot_lookup.get(f, {}).get('n_plants', 0))
    df['has_padi'] = df['filename'].map(
        lambda f: 'padi' in bot_lookup.get(f, {}).get('plants', ''))
    df['has_waringin'] = df['filename'].map(
        lambda f: 'waringin' in bot_lookup.get(f, {}).get('plants', ''))
    df['has_sirih'] = df['filename'].map(
        lambda f: 'sirih' in bot_lookup.get(f, {}).get('plants', ''))
    df['has_cendana'] = df['filename'].map(
        lambda f: 'cendana' in bot_lookup.get(f, {}).get('plants', ''))
else:
    df['n_botanical'] = 0
    df['has_padi'] = False
    df['has_waringin'] = False
    df['has_sirih'] = False
    df['has_cendana'] = False

# Language encoding
df['is_kawi'] = df['lang'].str.contains('kaw', na=False).astype(int)
df['is_malay'] = df['lang'].str.contains('omy|may', na=False).astype(int)

# Borobudur flag
df['is_borobudur'] = df['filename'].str.contains('Borobudur', na=False).astype(int)

# Feature engineering
df['keyword_density'] = df['total_keywords'] / df['word_count'].clip(lower=1)
df['indic_ratio'] = df['indic'] / (df['indic'] + df['pre_indic']).clip(lower=1)

print(f"\n  Dated: {df['has_date'].sum()}")
print(f"  Undated: {(~df['has_date']).sum()}")
print(f"  Date range: {df['year_ce'].min():.0f} - {df['year_ce'].max():.0f} CE")


# ═════════════════════════════════════════════════════════════════════════
# 2. FEATURE SELECTION
# ═════════════════════════════════════════════════════════════════════════

print("\n[2] Feature selection...")

FEATURES = [
    'word_count',
    'total_keywords',
    'indic',
    'pre_indic',
    'ambiguous',
    'pre_indic_ratio',
    'has_hyang',
    'has_manhuri',
    'has_wuku',
    'is_kawi',
    'is_malay',
    'is_borobudur',
    'keyword_density',
    'indic_ratio',
    'n_botanical',
    'has_padi',
    'has_waringin',
    'has_sirih',
    'has_cendana',
]

# Convert boolean columns
for col in ['has_hyang', 'has_manhuri', 'has_wuku',
            'has_padi', 'has_waringin', 'has_sirih', 'has_cendana']:
    df[col] = df[col].astype(int)

# Prepare training data (exclude Borobudur labels — short, noisy, all 750 CE)
df_train = df[df['has_date'] & (df['is_borobudur'] == 0)].copy()
df_train_all = df[df['has_date']].copy()  # With Borobudur for comparison
df_predict = df[~df['has_date']].copy()

X_train = df_train[FEATURES].fillna(0).values
y_train = df_train['year_ce'].values

X_train_all = df_train_all[FEATURES].fillna(0).values
y_train_all = df_train_all['year_ce'].values

X_predict = df_predict[FEATURES].fillna(0).values

print(f"  Features: {len(FEATURES)}")
print(f"  Training (excl. Borobudur): {len(X_train)}")
print(f"  Training (all): {len(X_train_all)}")
print(f"  To predict: {len(X_predict)}")

# Feature correlation with year
print(f"\n  Feature correlations with year_ce:")
for feat in FEATURES:
    vals = df_train[feat].fillna(0).values
    if np.std(vals) > 0:
        r, p = stats.pearsonr(vals, y_train)
        sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
        print(f"    {feat:<25} r={r:+.3f}  p={p:.4f}{sig}")


# ═════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING — Leave-One-Out Cross-Validation
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[3] Model Training with LOOCV")
print("=" * 70)

models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=3,
        learning_rate=0.05, random_state=42
    ),
}

loo = LeaveOneOut()
results = {}

for name, model in models.items():
    print(f"\n  {name}:")

    # LOOCV predictions
    y_pred_loo = cross_val_predict(model, X_train, y_train, cv=loo)

    mae = mean_absolute_error(y_train, y_pred_loo)
    r2 = r2_score(y_train, y_pred_loo)
    rmse = np.sqrt(np.mean((y_train - y_pred_loo) ** 2))

    # Century accuracy (within 1 century)
    century_true = (y_train // 100) + 1
    century_pred = (y_pred_loo // 100) + 1
    century_exact = np.mean(century_true == century_pred) * 100
    century_within1 = np.mean(np.abs(century_true - century_pred) <= 1) * 100

    print(f"    MAE: {mae:.1f} years")
    print(f"    RMSE: {rmse:.1f} years")
    print(f"    R²: {r2:.3f}")
    print(f"    Century exact: {century_exact:.1f}%")
    print(f"    Century ±1: {century_within1:.1f}%")

    results[name] = {
        'model': model,
        'y_pred_loo': y_pred_loo,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'century_exact': century_exact,
        'century_within1': century_within1,
    }

# Pick best model
best_name = min(results, key=lambda k: results[k]['mae'])
print(f"\n  Best model: {best_name} (MAE={results[best_name]['mae']:.1f} years)")


# ═════════════════════════════════════════════════════════════════════════
# 4. TEMPORAL SPLIT VALIDATION
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[4] Temporal Split Validation")
print("=" * 70)

# Train on C7-C10, predict C11-C14 (tests forward extrapolation)
early = df_train[df_train['year_ce'] <= 1000]
late = df_train[df_train['year_ce'] > 1000]

if len(early) >= 10 and len(late) >= 5:
    X_early = early[FEATURES].fillna(0).values
    y_early = early['year_ce'].values
    X_late = late[FEATURES].fillna(0).values
    y_late = late['year_ce'].values

    for name, info in results.items():
        model_copy = type(info['model'])(**info['model'].get_params())
        model_copy.fit(X_early, y_early)
        y_pred_late = model_copy.predict(X_late)

        mae_late = mean_absolute_error(y_late, y_pred_late)
        r2_late = r2_score(y_late, y_pred_late)

        print(f"\n  {name} (train≤1000, test>1000):")
        print(f"    Train: {len(X_early)}, Test: {len(X_late)}")
        print(f"    MAE: {mae_late:.1f} years")
        print(f"    R²: {r2_late:.3f}")
        results[name]['temporal_mae'] = mae_late
        results[name]['temporal_r2'] = r2_late
else:
    print(f"  Insufficient data for temporal split (early={len(early)}, late={len(late)})")


# ═════════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[5] Feature Importance")
print("=" * 70)

# Fit best model on all training data
best_model = results[best_name]['model']
best_model.fit(X_train, y_train)

importances = best_model.feature_importances_
feat_imp = sorted(zip(FEATURES, importances), key=lambda x: -x[1])

print(f"\n  {best_name} Feature Importance:")
print(f"  {'Feature':<25} {'Importance':>12}")
print("  " + "-" * 40)
for feat, imp in feat_imp:
    bar = "█" * int(imp * 100)
    print(f"  {feat:<25} {imp:>11.4f} {bar}")


# ═════════════════════════════════════════════════════════════════════════
# 6. PREDICT UNDATED INSCRIPTIONS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[6] Predicting Dates for Undated Inscriptions")
print("=" * 70)

if len(X_predict) > 0:
    # Use ensemble: average of both models
    predictions = {}
    for name, info in results.items():
        model = info['model']
        model.fit(X_train, y_train)  # Retrain on full training set
        pred = model.predict(X_predict)
        predictions[name] = pred

    # Ensemble average
    ensemble_pred = np.mean(list(predictions.values()), axis=0)

    # Confidence: use RF individual tree predictions for spread
    rf_model = results['RandomForest']['model']
    rf_model.fit(X_train, y_train)
    tree_preds = np.array([tree.predict(X_predict) for tree in rf_model.estimators_])
    pred_std = tree_preds.std(axis=0)

    # Create output dataframe
    df_predict = df_predict.copy()
    df_predict['predicted_year'] = ensemble_pred.astype(int)
    df_predict['predicted_century'] = (ensemble_pred // 100 + 1).astype(int)
    df_predict['uncertainty_years'] = pred_std.astype(int)
    df_predict['confidence'] = pd.cut(pred_std,
                                       bins=[0, 50, 100, 150, 999],
                                       labels=['HIGH', 'MEDIUM', 'LOW', 'VERY LOW'])

    # Sort by predicted year
    df_out = df_predict[['filename', 'title', 'lang', 'word_count',
                         'predicted_year', 'predicted_century',
                         'uncertainty_years', 'confidence']].sort_values('predicted_year')

    print(f"\n  Predicted dates for {len(df_out)} undated inscriptions:")
    print(f"  {'Title':<45} {'Pred':>6} {'C':>3} {'±':>4} {'Conf'}")
    print("  " + "-" * 70)
    for _, row in df_out.iterrows():
        title = str(row['title'])[:42]
        print(f"  {title:<45} {row['predicted_year']:>6} "
              f"C{row['predicted_century']:>2} {row['uncertainty_years']:>4} "
              f"{row['confidence']}")

    # Summary statistics
    print(f"\n  Prediction summary:")
    print(f"    Range: {df_out['predicted_year'].min()}-{df_out['predicted_year'].max()} CE")
    print(f"    Median: {df_out['predicted_year'].median():.0f} CE")
    print(f"    Mean uncertainty: ±{df_out['uncertainty_years'].mean():.0f} years")

    # Confidence distribution
    print(f"\n  Confidence distribution:")
    for conf, count in df_out['confidence'].value_counts().items():
        print(f"    {conf}: {count} inscriptions")

    # Save predictions
    df_out.to_csv(os.path.join(RESULTS_DIR, 'undated_predictions.csv'), index=False)
    print(f"\n  Saved: undated_predictions.csv")


# ═════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[7] Generating visualizations...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E037 — Prasasti Dating Model\n'
             'ML-Predicted Dates for Undated Inscriptions',
             fontsize=14, fontweight='bold', y=0.98)

# Panel A: LOOCV predicted vs actual
ax1 = axes[0, 0]
best_loo = results[best_name]['y_pred_loo']
ax1.scatter(y_train, best_loo, s=20, alpha=0.6, color='#3498db', edgecolors='gray',
            linewidth=0.3)
# Perfect prediction line
lims = [min(y_train.min(), best_loo.min()) - 20,
        max(y_train.max(), best_loo.max()) + 20]
ax1.plot(lims, lims, 'k--', alpha=0.3, label='Perfect prediction')
# ±50 year band
ax1.fill_between(lims, [l-50 for l in lims], [l+50 for l in lims],
                 alpha=0.1, color='green', label='±50 year band')
ax1.set_xlabel('Actual Year (CE)')
ax1.set_ylabel('Predicted Year (LOOCV)')
ax1.set_title(f'A. LOOCV: {best_name}\n'
              f'MAE={results[best_name]["mae"]:.0f}yr, R²={results[best_name]["r2"]:.3f}',
              fontsize=11)
ax1.legend(fontsize=8)
ax1.set_xlim(lims)
ax1.set_ylim(lims)

# Panel B: Feature importance
ax2 = axes[0, 1]
top_n = 10
top_feats = feat_imp[:top_n]
feat_names = [f[0] for f in top_feats]
feat_vals = [f[1] for f in top_feats]
ax2.barh(range(len(feat_names)), feat_vals, color='#27ae60', edgecolor='white')
ax2.set_yticks(range(len(feat_names)))
ax2.set_yticklabels(feat_names, fontsize=9)
ax2.set_xlabel('Importance')
ax2.set_title(f'B. Top {top_n} Features ({best_name})', fontsize=11)
ax2.invert_yaxis()

# Panel C: Residual distribution
ax3 = axes[1, 0]
residuals = y_train - best_loo
ax3.hist(residuals, bins=20, color='#9b59b6', edgecolor='white', alpha=0.8)
ax3.axvline(0, color='red', linestyle='--', alpha=0.5)
ax3.set_xlabel('Residual (Actual - Predicted, years)')
ax3.set_ylabel('Count')
ax3.set_title(f'C. Residual Distribution\n(median={np.median(residuals):.0f}, '
              f'IQR={np.percentile(residuals, 25):.0f} to {np.percentile(residuals, 75):.0f})',
              fontsize=11)

# Panel D: Predicted dates for undated inscriptions
ax4 = axes[1, 1]
if len(X_predict) > 0:
    # Histogram of predicted centuries
    pred_centuries = df_out['predicted_century'].values
    actual_centuries = (y_train // 100 + 1).astype(int)

    bins_c = np.arange(6.5, 15.5, 1)
    ax4.hist(actual_centuries, bins=bins_c, alpha=0.5, color='#3498db',
             edgecolor='white', label='Dated (actual)', density=True)
    ax4.hist(pred_centuries, bins=bins_c, alpha=0.5, color='#e74c3c',
             edgecolor='white', label='Undated (predicted)', density=True)
    ax4.set_xlabel('Century CE')
    ax4.set_ylabel('Proportion')
    ax4.set_title('D. Century Distribution\n(Dated vs Predicted Undated)', fontsize=11)
    ax4.legend(fontsize=8)
    ax4.set_xticks(range(7, 15))
    ax4.set_xticklabels([f'C{c}' for c in range(7, 15)])
else:
    ax4.text(0.5, 0.5, 'No undated inscriptions', ha='center', va='center')

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(RESULTS_DIR, 'dating_model_4panel.png'), dpi=150,
            bbox_inches='tight')
print("  Saved: dating_model_4panel.png")
plt.close('all')


# ═════════════════════════════════════════════════════════════════════════
# 8. STRUCTURED OUTPUT
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[8] Saving results...")
print("=" * 70)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

summary = {
    "experiment": "E037_prasasti_dating_ml",
    "n_training": len(X_train),
    "n_training_all": len(X_train_all),
    "n_predict": len(X_predict),
    "n_features": len(FEATURES),
    "models": {},
}

for name, info in results.items():
    summary["models"][name] = {
        "mae_years": round(info['mae'], 1),
        "rmse_years": round(info['rmse'], 1),
        "r2": round(info['r2'], 3),
        "century_exact_pct": round(info['century_exact'], 1),
        "century_within1_pct": round(info['century_within1'], 1),
    }
    if 'temporal_mae' in info:
        summary["models"][name]["temporal_mae"] = round(info['temporal_mae'], 1)
        summary["models"][name]["temporal_r2"] = round(info['temporal_r2'], 3)

summary["best_model"] = best_name
summary["feature_importance"] = {f: round(float(i), 4) for f, i in feat_imp}

with open(os.path.join(RESULTS_DIR, 'dating_model_summary.json'), 'w',
          encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

print("  Saved: dating_model_summary.json")


# ═════════════════════════════════════════════════════════════════════════
# 9. HEADLINE
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HEADLINE FINDING")
print("=" * 70)

best = results[best_name]
print(f"""
  MODEL: {best_name}
  Training: {len(X_train)} dated inscriptions (excl. Borobudur labels)
  Features: {len(FEATURES)} (keywords, language, botanical, length)

  LOOCV PERFORMANCE:
  MAE: {best['mae']:.1f} years
  RMSE: {best['rmse']:.1f} years
  R²: {best['r2']:.3f}
  Century exact: {best['century_exact']:.1f}%
  Century ±1: {best['century_within1']:.1f}%
""")

if 'temporal_mae' in best:
    print(f"  TEMPORAL SPLIT (train≤1000, test>1000):")
    print(f"  MAE: {best['temporal_mae']:.1f} years")
    print(f"  R²: {best['temporal_r2']:.3f}")

print(f"\n  PREDICTIONS: {len(X_predict)} undated inscriptions dated")
if len(X_predict) > 0:
    print(f"  Range: {df_out['predicted_year'].min()}-{df_out['predicted_year'].max()} CE")
    print(f"  Mean uncertainty: ±{df_out['uncertainty_years'].mean():.0f} years")

print(f"\n  TOP FEATURES:")
for f, i in feat_imp[:5]:
    print(f"    {f}: {i:.4f}")

print("\n" + "=" * 70)
print("E037 COMPLETE")
print("=" * 70)
