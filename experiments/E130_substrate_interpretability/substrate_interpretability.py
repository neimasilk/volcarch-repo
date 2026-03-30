"""
E130: Substrate Detection Interpretability — What Do Pre-Indic Words Tell Us?
Deep dive into E027 ML substrate detection results.

Instead of just asking "can we detect substrate?" (E027: yes, AUC=0.76),
ask "what are the substrate words, and what do they reveal about pre-Indic culture?"

This directly addresses the manifesto's core question: what was Nusantara like BEFORE 400 CE?
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
REPO = Path(__file__).parent.parent.parent

# === LOAD DATA ===

df = pd.read_csv(REPO / "experiments/E027_ml_substrate_detection/data/features_matrix.csv")
print(f"Total lexical forms: {len(df)}")

# Label: 1 = Austronesian cognate, 0 = residual (potential substrate)
# Based on column exploration
print(f"Label distribution: {df['label'].value_counts().to_dict()}")
print(f"Languages: {df['language'].unique()}")

# === TRAIN MODEL WITH INTERPRETABILITY ===

feature_cols = [c for c in df.columns if c not in
                ["form_id", "language", "concept", "form", "label", "semantic_domain",
                 "is_core_vocab", "language_id_encoded", "initial_char",
                 "language_cognacy_coverage"]]

# Only use numeric columns
numeric_mask = df[feature_cols].dtypes.apply(lambda t: np.issubdtype(t, np.number))
feature_cols = [c for c, is_num in zip(feature_cols, numeric_mask) if is_num]
X = df[feature_cols].values
y = df["label"].values

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Feature importances
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nTop 10 features:")
for feat, imp in importances.head(10).items():
    print(f"  {feat}: {imp:.4f}")

# === EXTRACT SUBSTRATE WORDS ===

# Predict probabilities
probs = rf.predict_proba(X)
df["prob_cognate"] = probs[:, 1]  # probability of being Austronesian cognate
df["prob_substrate"] = probs[:, 0]  # probability of being substrate/residual

# High-confidence substrate words
substrate = df[df["prob_substrate"] > 0.8].copy()
cognate = df[df["prob_cognate"] > 0.8].copy()

print(f"\nHigh-confidence substrate words (prob > 0.8): {len(substrate)}")
print(f"High-confidence cognate words (prob > 0.8): {len(cognate)}")

# === ANALYZE SUBSTRATE BY SEMANTIC DOMAIN ===

print(f"\n{'=' * 70}")
print("SUBSTRATE WORDS BY SEMANTIC DOMAIN")
print("=" * 70)

if "semantic_domain" in df.columns:
    domain_analysis = {}
    for domain in df["semantic_domain"].unique():
        domain_mask = df["semantic_domain"] == domain
        domain_sub = substrate[substrate["semantic_domain"] == domain]
        domain_cog = cognate[cognate["semantic_domain"] == domain]
        domain_total = domain_mask.sum()

        domain_analysis[domain] = {
            "total": int(domain_total),
            "substrate_count": len(domain_sub),
            "cognate_count": len(domain_cog),
            "substrate_rate": len(domain_sub) / domain_total if domain_total > 0 else 0,
        }

    # Sort by substrate rate
    sorted_domains = sorted(domain_analysis.items(), key=lambda x: x[1]["substrate_rate"], reverse=True)

    print(f"\n  {'Domain':<20} {'Total':>6} {'Substrate':>10} {'Cognate':>8} {'Sub Rate':>9}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*9}")
    for domain, data in sorted_domains:
        if data["total"] >= 5:  # only show domains with enough data
            print(f"  {domain:<20} {data['total']:>6} {data['substrate_count']:>10} "
                  f"{data['cognate_count']:>8} {data['substrate_rate']:>8.1%}")

# === EXTRACT ACTUAL SUBSTRATE WORD FORMS ===

print(f"\n{'=' * 70}")
print("TOP 50 MOST CONFIDENTLY SUBSTRATE WORDS")
print("=" * 70)

top_substrate = substrate.nlargest(50, "prob_substrate")

print(f"\n  {'Form':<15} {'Language':<12} {'Concept':<20} {'Domain':<12} {'P(sub)':>6}")
print(f"  {'-'*15} {'-'*12} {'-'*20} {'-'*12} {'-'*6}")
for _, row in top_substrate.iterrows():
    print(f"  {str(row['form'])[:15]:<15} {str(row['language'])[:12]:<12} "
          f"{str(row['concept'])[:20]:<20} {str(row.get('semantic_domain', 'N/A'))[:12]:<12} "
          f"{row['prob_substrate']:>5.2f}")

# === LANGUAGE-SPECIFIC ANALYSIS ===

print(f"\n{'=' * 70}")
print("SUBSTRATE RATE BY LANGUAGE")
print("=" * 70)

for lang in df["language"].unique():
    lang_mask = df["language"] == lang
    lang_sub = substrate[substrate["language"] == lang]
    lang_total = lang_mask.sum()
    rate = len(lang_sub) / lang_total if lang_total > 0 else 0
    print(f"  {lang:<20}: {len(lang_sub):>4} substrate / {lang_total:>4} total = {rate:.1%}")

# === PHONOLOGICAL PATTERNS IN SUBSTRATE ===

print(f"\n{'=' * 70}")
print("PHONOLOGICAL PATTERNS: What Makes Substrate Words Different?")
print("=" * 70)

if len(substrate) >= 10 and len(cognate) >= 10:
    for feat in ["form_length", "vowel_ratio", "ends_in_vowel", "has_glottal",
                  "has_nasal_cluster", "has_reduplication", "n_consonant_clusters"]:
        if feat in substrate.columns and feat in cognate.columns:
            sub_mean = substrate[feat].mean()
            cog_mean = cognate[feat].mean()
            diff = sub_mean - cog_mean
            direction = "higher" if diff > 0 else "lower"
            print(f"  {feat:<25}: substrate={sub_mean:.3f}, cognate={cog_mean:.3f} "
                  f"({direction}, delta={abs(diff):.3f})")

# === WHAT PRE-INDIC CULTURE LOOKED LIKE ===

print(f"\n{'=' * 70}")
print("INTERPRETATION: What Substrate Words Reveal About Pre-Indic Culture")
print("=" * 70)

# Group substrate words by semantic domain and list concepts
if len(substrate) > 0:
    for domain in substrate["semantic_domain"].unique():
        domain_words = substrate[substrate["semantic_domain"] == domain]
        if len(domain_words) >= 3:
            concepts = domain_words["concept"].unique()[:10]
            print(f"\n  {domain} ({len(domain_words)} substrate words):")
            for c in concepts:
                forms = domain_words[domain_words["concept"] == c]["form"].values
                print(f"    '{c}': {', '.join(str(f) for f in forms[:3])}")

# === SAVE ===

summary = {
    "experiment": "E130_substrate_interpretability",
    "total_forms": len(df),
    "high_confidence_substrate": len(substrate),
    "high_confidence_cognate": len(cognate),
    "domain_analysis": {k: v for k, v in sorted_domains} if "semantic_domain" in df.columns else {},
    "top_features": dict(importances.head(10)),
}

with open(RESULTS_DIR / "substrate_interpretability.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

# Save substrate word list
substrate_list = substrate[["form", "language", "concept", "semantic_domain", "prob_substrate"]].to_dict("records")
with open(RESULTS_DIR / "substrate_words.json", "w") as f:
    json.dump(substrate_list, f, indent=2, ensure_ascii=False, default=str)

print(f"\n  Saved to {RESULTS_DIR}/")
print(f"  {len(substrate)} substrate words cataloged")
