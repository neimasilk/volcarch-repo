"""
E101 — Colonial Burial Depth Multivariate Model
================================================
Models burial depth as f(distance, eruption frequency, site age).
Uses E083 + E070 data (genuinely independent colonial-era observations).

Experiment #102 in the VOLCARCH series.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import json

print("=" * 70)
print("E101 — COLONIAL BURIAL DEPTH MULTIVARIATE MODEL")
print("=" * 70)

# --- 1. Load and merge data ---
print("\n[1/5] Loading data...")

e083 = pd.read_csv("experiments/E083_tephra_archaeological_correlation/results/tephra_archaeological_correlation.csv")
e070 = pd.read_csv("experiments/E070_colonial_literature_mining/results/colonial_site_register_v1.0.csv")

# E083: sites with measured depth
e083_depth = e083[e083['burial_depth_m'].notna() & (e083['burial_depth_m'] > 0)].copy()
print(f"  E083 with depth > 0: {len(e083_depth)}")

# E070: sites with measured depth
e070_depth = e070[e070['burial_depth_m'].notna() & (e070['burial_depth_m'] > 0)].copy()
print(f"  E070 with depth > 0: {len(e070_depth)}")

# Merge into unified dataset
# E083 columns: site_name, burial_depth_m, volcano_name, site_distance_km, site_lat, site_lon
# E070 columns: site_name, burial_depth_m, volcanic_system, volcano_dist_km, lat, lon, built_ce

records = []

for _, row in e083_depth.iterrows():
    if pd.notna(row.get('site_distance_km')):
        records.append({
            'site': row['site_name'],
            'depth_m': row['burial_depth_m'],
            'volcano': row['volcano_name'],
            'dist_km': row['site_distance_km'],
            'lat': row.get('site_lat', np.nan),
            'lon': row.get('site_lon', np.nan),
            'built_ce': np.nan,
            'source': 'E083',
        })

for _, row in e070_depth.iterrows():
    if pd.notna(row.get('volcano_dist_km')):
        records.append({
            'site': row['site_name'],
            'depth_m': row['burial_depth_m'],
            'volcano': row.get('volcanic_system', ''),
            'dist_km': row['volcano_dist_km'],
            'lat': row.get('lat', np.nan),
            'lon': row.get('lon', np.nan),
            'built_ce': row.get('built_ce', np.nan),
            'source': 'E070',
        })

df = pd.DataFrame(records)
# Deduplicate by site name (prefer E083 which has more metadata)
df = df.drop_duplicates(subset='site', keep='first')
print(f"  Merged unique sites with depth: {len(df)}")

# --- 2. Feature engineering ---
print("\n[2/5] Engineering features...")

# Parse built_ce for age calculation
def parse_year(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('~', '').replace('pre-', '').strip()
    try:
        parts = val.split('-')
        return float(parts[0])
    except:
        return np.nan

df['built_year'] = df['built_ce'].apply(parse_year)
df['age_years'] = 2026 - df['built_year']

# Volcano eruption frequency (from GVP, per volcano)
eruption_freq = {
    'Kelud': 30,      # ~30 eruptions in 1000 years (very active)
    'Merapi': 50,     # ~50 eruptions in 1000 years (most active)
    'Arjuno-Welirang': 5,  # relatively quiet
    'Dieng': 10,
    'Ungaran': 3,
    'Bromo': 15,
    'Semeru': 40,
    'Sumatran volcanic arc': 10,
}

df['eruption_freq'] = df['volcano'].map(eruption_freq).fillna(10)

# Log distance (diminishing effect)
df['log_dist'] = np.log1p(df['dist_km'])

# Interaction: freq × inverse distance
df['freq_x_inv_dist'] = df['eruption_freq'] / (df['dist_km'] + 1)

print(f"  Features: dist_km, log_dist, eruption_freq, freq_x_inv_dist, age_years")
print(f"  Sites with all features: {df.dropna(subset=['dist_km']).shape[0]}")
print(f"  Sites with age: {df['age_years'].notna().sum()}")

# --- 3. Univariate correlations ---
print("\n[3/5] Univariate correlations with burial depth...")

for feat in ['dist_km', 'log_dist', 'eruption_freq', 'freq_x_inv_dist', 'age_years']:
    valid = df[['depth_m', feat]].dropna()
    if len(valid) > 5:
        rho, p = stats.spearmanr(valid['depth_m'], valid[feat])
        print(f"  {feat:<20} rho={rho:>7.4f}  p={p:.4f}  n={len(valid)}")

# --- 4. Multivariate model ---
print("\n[4/5] Multivariate regression...")

# Use features available for most sites
features = ['dist_km', 'eruption_freq', 'freq_x_inv_dist']
df_model = df.dropna(subset=['depth_m'] + features)
X = df_model[features].values
y = df_model['depth_m'].values
print(f"  Training samples: {len(df_model)}")

# Linear regression
lr = LinearRegression()
lr.fit(X, y)
y_pred_lr = lr.predict(X)
r2_lr = r2_score(y, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y, y_pred_lr))
print(f"\n  Linear Regression:")
print(f"    R² = {r2_lr:.4f}")
print(f"    RMSE = {rmse_lr:.2f} m")
for feat, coef in zip(features, lr.coef_):
    print(f"    {feat}: {coef:.4f}")
print(f"    intercept: {lr.intercept_:.4f}")

# Gradient Boosting (handles non-linearity)
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42,
                                learning_rate=0.1, min_samples_leaf=3)
gb.fit(X, y)
y_pred_gb = gb.predict(X)
r2_gb = r2_score(y, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y, y_pred_gb))
print(f"\n  Gradient Boosting:")
print(f"    R² = {r2_gb:.4f}")
print(f"    RMSE = {rmse_gb:.2f} m")
for feat, imp in zip(features, gb.feature_importances_):
    print(f"    {feat}: importance={imp:.4f}")

# --- 5. Leave-One-Out Cross-Validation ---
print("\n[5/5] Leave-One-Out Cross-Validation...")

loo = LeaveOneOut()
loo_preds_lr = np.zeros(len(y))
loo_preds_gb = np.zeros(len(y))

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    lr_loo = LinearRegression().fit(X_train, y_train)
    loo_preds_lr[test_idx] = lr_loo.predict(X_test)

    gb_loo = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42,
                                        learning_rate=0.1, min_samples_leaf=3)
    gb_loo.fit(X_train, y_train)
    loo_preds_gb[test_idx] = gb_loo.predict(X_test)

r2_loo_lr = r2_score(y, loo_preds_lr)
rmse_loo_lr = np.sqrt(mean_squared_error(y, loo_preds_lr))
r2_loo_gb = r2_score(y, loo_preds_gb)
rmse_loo_gb = np.sqrt(mean_squared_error(y, loo_preds_gb))

print(f"  LOO Linear:    R²={r2_loo_lr:.4f}, RMSE={rmse_loo_lr:.2f} m")
print(f"  LOO GBoosting: R²={r2_loo_gb:.4f}, RMSE={rmse_loo_gb:.2f} m")

# Prediction vs actual
print(f"\n  Prediction scatter (LOO, top 10 by actual depth):")
df_model = df_model.copy()
df_model['pred_lr'] = loo_preds_lr
df_model['pred_gb'] = loo_preds_gb
top = df_model.nlargest(10, 'depth_m')
for _, row in top.iterrows():
    print(f"    {row['site'][:35]:<35} actual={row['depth_m']:.1f}m  LR={row['pred_lr']:.1f}m  GB={row['pred_gb']:.1f}m")

# --- Save ---
results = {
    'meta': {
        'experiment': 'E101',
        'date': '2026-03-17',
        'n_sites': len(df_model),
        'features': features,
    },
    'univariate': {},
    'linear_regression': {
        'r2': float(r2_lr), 'rmse': float(rmse_lr),
        'r2_loo': float(r2_loo_lr), 'rmse_loo': float(rmse_loo_lr),
        'coefficients': {feat: float(c) for feat, c in zip(features, lr.coef_)},
        'intercept': float(lr.intercept_),
    },
    'gradient_boosting': {
        'r2': float(r2_gb), 'rmse': float(rmse_gb),
        'r2_loo': float(r2_loo_gb), 'rmse_loo': float(rmse_loo_gb),
        'feature_importance': {feat: float(imp) for feat, imp in zip(features, gb.feature_importances_)},
    },
}

# Add univariate
for feat in ['dist_km', 'log_dist', 'eruption_freq', 'freq_x_inv_dist', 'age_years']:
    valid = df[['depth_m', feat]].dropna()
    if len(valid) > 5:
        rho, p = stats.spearmanr(valid['depth_m'], valid[feat])
        results['univariate'][feat] = {'rho': float(rho), 'p': float(p), 'n': len(valid)}

with open("experiments/E101_burial_depth_model/results/e101_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("E101 SUMMARY")
print("=" * 70)
print(f"  Sites modeled: {len(df_model)}")
print(f"  Linear R² (LOO): {r2_loo_lr:.4f}, RMSE: {rmse_loo_lr:.2f} m")
print(f"  GBoosting R² (LOO): {r2_loo_gb:.4f}, RMSE: {rmse_loo_gb:.2f} m")
best = "GBoosting" if r2_loo_gb > r2_loo_lr else "Linear"
print(f"  Best model: {best}")
print(f"  Top feature: {features[np.argmax(gb.feature_importances_)]}")
print("=" * 70)
