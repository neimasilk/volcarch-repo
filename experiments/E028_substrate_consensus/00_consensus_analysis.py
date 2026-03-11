"""
E028: Cross-Method Consensus Analysis
========================================
Combines E022 (rule-based subtraction) and E027 (ML substrate detection)
to identify high-confidence substrate candidates via cross-method agreement.

Hypothesis: Forms identified as substrate by BOTH methods are more likely
to be genuine pre-Austronesian remnants than forms flagged by only one method.

Outputs:
  results/consensus_summary.json    — agreement stats, quadrant counts
  results/consensus_substrates.csv  — CS candidates ranked by P(substrate)
  results/cross_language_consensus.csv — concepts in CS for 4+/6 languages
  results/quadrant_comparison.png   — 2x2 visualization
"""
import io
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost not available, falling back to RandomForest")

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

E022_RESIDUALS = ROOT.parent / "E022_linguistic_subtraction" / "results" / "poc_residuals_detail.csv"
E027_RANKING   = ROOT.parent / "E027_ml_substrate_detection" / "results" / "substrate_ranking.csv"
E027_FEATURES  = ROOT.parent / "E027_ml_substrate_detection" / "data" / "features_matrix.csv"

# Same feature setup as E027
PHON_FEATURES = [
    "form_length", "n_vowels", "vowel_ratio", "ends_in_vowel",
    "has_glottal", "has_nasal_cluster", "has_reduplication",
    "n_consonant_clusters", "has_prefix_like",
]
SEMANTIC_FEATURES = ["is_core_vocab"]
LANG_FEATURES = ["language_id_encoded", "language_cognacy_coverage"]


def load_and_prepare():
    """Load features matrix, retrain E027 Model B, get P(substrate) for ALL forms."""
    print("[1/5] Loading data and retraining E027 Model B for full predictions...")

    # Load features matrix (all 1357 forms with label)
    fm = pd.read_csv(E027_FEATURES, encoding="utf-8")
    print(f"  Features matrix: {len(fm)} forms")
    print(f"    label=0 (E022 residual): {(fm['label']==0).sum()}")
    print(f"    label=1 (E022 cognate):  {(fm['label']==1).sum()}")

    # One-hot encode (same as E027 script 02)
    ic_dummies = pd.get_dummies(fm["initial_char"], prefix="init")
    sd_dummies = pd.get_dummies(fm["semantic_domain"], prefix="sem")
    fm_enc = pd.concat([fm, ic_dummies, sd_dummies], axis=1)

    init_cols = [c for c in fm_enc.columns if c.startswith("init_")]
    sem_cols  = [c for c in fm_enc.columns if c.startswith("sem_")]

    model_b_cols = PHON_FEATURES + init_cols + SEMANTIC_FEATURES + sem_cols + LANG_FEATURES

    y = fm_enc["label"].values
    X = fm_enc[model_b_cols].values.astype(float)

    # Train model (same hyperparams as E027)
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

    clf.fit(X, y)
    probs = clf.predict_proba(X)
    # P(substrate) = P(class 0)
    fm["p_substrate"] = probs[:, 0]

    print(f"  Model retrained. P(substrate) assigned to all {len(fm)} forms.")

    # E022 binary label: label=0 means E022 residual (substrate), label=1 means cognate
    # For consensus: e022_substrate = (label == 0)
    fm["e022_substrate"] = (fm["label"] == 0).astype(int)

    # ML binary prediction: P(substrate) >= 0.5
    fm["ml_substrate"] = (fm["p_substrate"] >= 0.5).astype(int)

    return fm, X, model_b_cols


def compute_agreement(fm):
    """Compute inter-method agreement statistics."""
    print("\n[2/5] Computing cross-method agreement...")

    e022 = fm["e022_substrate"].values
    ml   = fm["ml_substrate"].values

    # Cohen's kappa
    kappa = cohen_kappa_score(e022, ml)
    print(f"  Cohen's kappa: {kappa:.4f}")

    # Correlation between E027 P(substrate) and E022 binary label
    p_sub = fm["p_substrate"].values
    r_pearson, p_pearson = pearsonr(p_sub, e022)
    r_spearman, p_spearman = spearmanr(p_sub, e022)
    print(f"  Pearson r(P_substrate, E022):  {r_pearson:.4f}  (p={p_pearson:.2e})")
    print(f"  Spearman rho(P_substrate, E022): {r_spearman:.4f}  (p={p_spearman:.2e})")

    # Quadrant assignment
    fm["quadrant"] = "UNKNOWN"
    fm.loc[(fm["e022_substrate"]==1) & (fm["ml_substrate"]==1), "quadrant"] = "CS"  # Consensus Substrate
    fm.loc[(fm["e022_substrate"]==0) & (fm["ml_substrate"]==0), "quadrant"] = "CA"  # Consensus Austronesian
    fm.loc[(fm["e022_substrate"]==0) & (fm["ml_substrate"]==1), "quadrant"] = "MO"  # ML-only
    fm.loc[(fm["e022_substrate"]==1) & (fm["ml_substrate"]==0), "quadrant"] = "RO"  # Rule-only

    # Quadrant counts
    q_counts = fm["quadrant"].value_counts().to_dict()
    print(f"\n  Quadrant distribution:")
    for q in ["CS", "CA", "MO", "RO"]:
        n = q_counts.get(q, 0)
        pct = 100 * n / len(fm)
        label = {
            "CS": "Consensus Substrate",
            "CA": "Consensus Austronesian",
            "MO": "ML-only substrate",
            "RO": "Rule-only substrate",
        }[q]
        print(f"    {q}: {n:>5} ({pct:5.1f}%) — {label}")

    # Mean P(substrate) per quadrant
    print(f"\n  Mean P(substrate) by quadrant:")
    for q in ["CS", "CA", "MO", "RO"]:
        subset = fm[fm["quadrant"] == q]
        if len(subset) > 0:
            print(f"    {q}: {subset['p_substrate'].mean():.4f} (median {subset['p_substrate'].median():.4f})")

    stats = {
        "n_total": len(fm),
        "cohens_kappa": round(kappa, 4),
        "pearson_r": round(r_pearson, 4),
        "pearson_p": float(f"{p_pearson:.4e}"),
        "spearman_rho": round(r_spearman, 4),
        "spearman_p": float(f"{p_spearman:.4e}"),
        "quadrant_counts": {q: int(q_counts.get(q, 0)) for q in ["CS", "CA", "MO", "RO"]},
        "quadrant_mean_p_substrate": {
            q: round(float(fm[fm["quadrant"]==q]["p_substrate"].mean()), 4)
            for q in ["CS", "CA", "MO", "RO"]
            if len(fm[fm["quadrant"]==q]) > 0
        },
    }
    return fm, stats


def analyze_consensus_substrates(fm):
    """Analyze the Consensus Substrate (CS) quadrant in detail."""
    print("\n[3/5] Analyzing Consensus Substrate (CS) candidates...")

    cs = fm[fm["quadrant"] == "CS"].copy()
    print(f"  {len(cs)} CS candidates total")

    # Count per language
    lang_counts = cs["language"].value_counts()
    print(f"\n  CS per language:")
    for lang, n in lang_counts.items():
        total_lang = len(fm[fm["language"] == lang])
        pct = 100 * n / total_lang
        print(f"    {lang:<16} {n:>4} / {total_lang:<4} ({pct:5.1f}%)")

    # Semantic domain distribution
    domain_counts = cs["semantic_domain"].value_counts()
    print(f"\n  CS by semantic domain:")
    for dom, n in domain_counts.items():
        pct = 100 * n / len(cs)
        print(f"    {dom:<12} {n:>4} ({pct:5.1f}%)")

    # Cross-language consensus: concepts that appear as CS in 4+ languages
    concept_langs = cs.groupby("concept")["language"].nunique().reset_index()
    concept_langs.columns = ["concept", "n_languages_cs"]

    cross_lang_cs = concept_langs[concept_langs["n_languages_cs"] >= 4].copy()
    cross_lang_cs = cross_lang_cs.sort_values("n_languages_cs", ascending=False)

    print(f"\n  Concepts in CS for 4+/6 languages: {len(cross_lang_cs)}")
    if len(cross_lang_cs) > 0:
        for _, row in cross_lang_cs.iterrows():
            concept = row["concept"]
            n = row["n_languages_cs"]
            # Get forms and mean p_substrate for this concept
            concept_cs = cs[cs["concept"] == concept]
            forms = ", ".join(f"{r['language']}:{r['form']}" for _, r in concept_cs.iterrows())
            mean_p = concept_cs["p_substrate"].mean()
            print(f"    {concept:<30} {n} langs  mean_P={mean_p:.3f}  [{forms}]")

    # Build cross-language output
    cross_lang_rows = []
    for _, row in cross_lang_cs.iterrows():
        concept = row["concept"]
        concept_cs = cs[cs["concept"] == concept]
        langs = sorted(concept_cs["language"].unique())
        forms = "; ".join(f"{r['language']}:{r['form']}" for _, r in concept_cs.iterrows())
        mean_p = concept_cs["p_substrate"].mean()
        domain = concept_cs["semantic_domain"].mode().iloc[0] if len(concept_cs) > 0 else "UNKNOWN"
        cross_lang_rows.append({
            "concept": concept,
            "n_languages": row["n_languages_cs"],
            "languages": "|".join(langs),
            "forms": forms,
            "mean_p_substrate": round(mean_p, 4),
            "semantic_domain": domain,
        })

    cross_lang_df = pd.DataFrame(cross_lang_rows)

    # Save CS candidates ranked by p_substrate
    cs_ranked = cs[["language", "concept", "form", "p_substrate", "semantic_domain",
                     "form_length", "vowel_ratio", "ends_in_vowel",
                     "has_glottal", "has_nasal_cluster", "has_reduplication",
                     "n_consonant_clusters", "has_prefix_like"]].copy()
    cs_ranked = cs_ranked.sort_values("p_substrate", ascending=False).reset_index(drop=True)
    cs_ranked.index = cs_ranked.index + 1
    cs_ranked.index.name = "rank"

    return cs_ranked, cross_lang_df, {
        "n_cs": len(cs),
        "per_language": lang_counts.to_dict(),
        "per_domain": domain_counts.to_dict(),
        "n_cross_language_4plus": len(cross_lang_cs),
    }


def analyze_disagreements(fm):
    """Analyze MO (ML-only) and RO (Rule-only) quadrants."""
    print("\n[4/5] Analyzing disagreement quadrants...")

    mo = fm[fm["quadrant"] == "MO"]  # E022=cognate, ML=substrate
    ro = fm[fm["quadrant"] == "RO"]  # E022=residual, ML=Austronesian
    cs = fm[fm["quadrant"] == "CS"]
    ca = fm[fm["quadrant"] == "CA"]

    # Phonological features to compare
    phon_cols = ["form_length", "vowel_ratio", "ends_in_vowel",
                 "has_glottal", "has_nasal_cluster", "has_reduplication",
                 "n_consonant_clusters", "has_prefix_like"]

    # MO vs CA: Are ML-only substrates phonologically different from Consensus Austronesian?
    print(f"\n  ML-only (MO, n={len(mo)}) vs Consensus Austronesian (CA, n={len(ca)}):")
    print(f"  Question: Are MO forms potentially missed substrates?")
    print(f"  {'Feature':<25} {'MO mean':>10} {'CA mean':>10} {'Delta':>10}")
    print(f"  {'-'*55}")

    mo_vs_ca = {}
    for col in phon_cols:
        mo_mean = mo[col].mean() if len(mo) > 0 else 0
        ca_mean = ca[col].mean() if len(ca) > 0 else 0
        delta = mo_mean - ca_mean
        print(f"  {col:<25} {mo_mean:>10.3f} {ca_mean:>10.3f} {delta:>+10.3f}")
        mo_vs_ca[col] = {"MO_mean": round(mo_mean, 4), "CA_mean": round(ca_mean, 4),
                          "delta": round(delta, 4)}

    # RO vs CS: Are Rule-only substrates phonologically different from Consensus Substrate?
    print(f"\n  Rule-only (RO, n={len(ro)}) vs Consensus Substrate (CS, n={len(cs)}):")
    print(f"  Question: Are RO forms false positives in E022?")
    print(f"  {'Feature':<25} {'RO mean':>10} {'CS mean':>10} {'Delta':>10}")
    print(f"  {'-'*55}")

    ro_vs_cs = {}
    for col in phon_cols:
        ro_mean = ro[col].mean() if len(ro) > 0 else 0
        cs_mean = cs[col].mean() if len(cs) > 0 else 0
        delta = ro_mean - cs_mean
        print(f"  {col:<25} {ro_mean:>10.3f} {cs_mean:>10.3f} {delta:>+10.3f}")
        ro_vs_cs[col] = {"RO_mean": round(ro_mean, 4), "CS_mean": round(cs_mean, 4),
                          "delta": round(delta, 4)}

    # MO: per language breakdown
    print(f"\n  MO per language:")
    if len(mo) > 0:
        for lang, n in mo["language"].value_counts().items():
            total = len(fm[fm["language"] == lang])
            print(f"    {lang:<16} {n:>3} / {total} ({100*n/total:.1f}%)")

    # RO: per language breakdown
    print(f"\n  RO per language:")
    if len(ro) > 0:
        for lang, n in ro["language"].value_counts().items():
            total = len(fm[fm["language"] == lang])
            print(f"    {lang:<16} {n:>3} / {total} ({100*n/total:.1f}%)")

    # Diagnostic: top MO candidates (potential missed substrates)
    if len(mo) > 0:
        top_mo = mo.nlargest(10, "p_substrate")[["language", "concept", "form", "p_substrate", "semantic_domain"]]
        print(f"\n  Top 10 ML-only candidates (potential missed substrates):")
        for _, r in top_mo.iterrows():
            print(f"    P={r['p_substrate']:.3f}  {r['language']:<16} {r['concept']:<25} {r['form']}")

    # Diagnostic: top RO candidates (potential E022 false positives)
    if len(ro) > 0:
        bottom_ro = ro.nsmallest(10, "p_substrate")[["language", "concept", "form", "p_substrate", "semantic_domain"]]
        print(f"\n  Bottom 10 Rule-only candidates (lowest ML confidence, likely false positives):")
        for _, r in bottom_ro.iterrows():
            print(f"    P={r['p_substrate']:.3f}  {r['language']:<16} {r['concept']:<25} {r['form']}")

    return {
        "mo_vs_ca_features": mo_vs_ca,
        "ro_vs_cs_features": ro_vs_cs,
        "mo_per_language": mo["language"].value_counts().to_dict() if len(mo) > 0 else {},
        "ro_per_language": ro["language"].value_counts().to_dict() if len(ro) > 0 else {},
    }


def make_visualization(fm):
    """Create 2x2 quadrant visualization."""
    print("\n[5/5] Generating quadrant visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    colors = {"CS": "#2ecc71", "CA": "#3498db", "MO": "#e74c3c", "RO": "#f39c12"}
    quadrant_labels = {
        "CS": "Consensus Substrate\n(E022=residual, ML P>=0.5)",
        "CA": "Consensus Austronesian\n(E022=cognate, ML P<0.5)",
        "MO": "ML-only Substrate\n(E022=cognate, ML P>=0.5)",
        "RO": "Rule-only Substrate\n(E022=residual, ML P<0.5)",
    }

    # --- Panel A: Scatter plot P(substrate) vs E022 label with jitter ---
    ax = axes[0, 0]
    for q in ["CA", "RO", "MO", "CS"]:
        subset = fm[fm["quadrant"] == q]
        jitter_x = subset["e022_substrate"] + np.random.normal(0, 0.05, len(subset))
        ax.scatter(jitter_x, subset["p_substrate"], c=colors[q], alpha=0.4,
                   s=12, label=f"{q} (n={len(subset)})", edgecolors="none")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("E022 Label (0=cognate, 1=residual)", fontsize=10)
    ax.set_ylabel("E027 P(substrate)", fontsize=10)
    ax.set_title("A. Cross-Method Agreement", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="center left")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.05, 1.05)

    # --- Panel B: Quadrant bar chart per language ---
    ax = axes[0, 1]
    languages = sorted(fm["language"].unique())
    x = np.arange(len(languages))
    width = 0.2
    for i, q in enumerate(["CS", "CA", "MO", "RO"]):
        counts = [len(fm[(fm["language"]==lang) & (fm["quadrant"]==q)]) for lang in languages]
        ax.bar(x + i*width, counts, width, label=q, color=colors[q], alpha=0.85)
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(languages, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("B. Quadrant Distribution per Language", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # --- Panel C: P(substrate) distribution per quadrant ---
    ax = axes[1, 0]
    for q in ["CS", "MO", "RO", "CA"]:
        subset = fm[fm["quadrant"] == q]
        if len(subset) > 0:
            ax.hist(subset["p_substrate"], bins=30, alpha=0.5, color=colors[q],
                    label=f"{q} (n={len(subset)})", density=True)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("P(substrate)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("C. P(substrate) Distribution by Quadrant", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # --- Panel D: Phonological feature comparison (CS vs RO vs MO vs CA) ---
    ax = axes[1, 1]
    phon_cols = ["form_length", "vowel_ratio", "has_glottal", "has_nasal_cluster",
                 "has_prefix_like", "has_reduplication"]
    x = np.arange(len(phon_cols))
    width = 0.2

    # Normalize feature means to [0,1] for comparability
    for i, q in enumerate(["CS", "CA", "MO", "RO"]):
        subset = fm[fm["quadrant"] == q]
        means = []
        for col in phon_cols:
            col_min = fm[col].min()
            col_max = fm[col].max()
            if col_max > col_min:
                norm_mean = (subset[col].mean() - col_min) / (col_max - col_min)
            else:
                norm_mean = 0
            means.append(norm_mean)
        ax.bar(x + i*width, means, width, label=q, color=colors[q], alpha=0.85)

    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels([c.replace("has_", "").replace("_", "\n") for c in phon_cols],
                       fontsize=7, rotation=0)
    ax.set_ylabel("Normalized Mean", fontsize=10)
    ax.set_title("D. Phonological Profile by Quadrant", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7)

    plt.suptitle("E028: Cross-Method Substrate Consensus (E022 x E027)\n"
                 "6 Sulawesi languages, 1357 lexical forms",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS / "quadrant_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS / 'quadrant_comparison.png'}")


def main():
    print("=" * 70)
    print("E028: Cross-Method Substrate Consensus Analysis")
    print("  E022 (rule-based) x E027 (ML-based)")
    print("=" * 70)

    # Step 1: Load data and retrain model for full predictions
    fm, X, model_b_cols = load_and_prepare()

    # Step 2: Compute agreement
    fm, agreement_stats = compute_agreement(fm)

    # Step 3: Analyze Consensus Substrate
    cs_ranked, cross_lang_df, cs_stats = analyze_consensus_substrates(fm)

    # Step 4: Analyze disagreements
    disagree_stats = analyze_disagreements(fm)

    # Step 5: Visualization
    make_visualization(fm)

    # ============================================================
    # Save outputs
    # ============================================================
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # 1. consensus_summary.json
    summary = {
        "experiment": "E028_substrate_consensus",
        "description": "Cross-method consensus between E022 (rule-based) and E027 (ML-based) substrate detection",
        "data": {
            "n_total_forms": agreement_stats["n_total"],
            "e022_residuals": int((fm["e022_substrate"] == 1).sum()),
            "e022_cognates": int((fm["e022_substrate"] == 0).sum()),
            "ml_substrates_p05": int((fm["ml_substrate"] == 1).sum()),
            "ml_austronesian_p05": int((fm["ml_substrate"] == 0).sum()),
        },
        "agreement": {
            "cohens_kappa": agreement_stats["cohens_kappa"],
            "pearson_r": agreement_stats["pearson_r"],
            "pearson_p": agreement_stats["pearson_p"],
            "spearman_rho": agreement_stats["spearman_rho"],
            "spearman_p": agreement_stats["spearman_p"],
        },
        "quadrants": agreement_stats["quadrant_counts"],
        "quadrant_mean_p_substrate": agreement_stats["quadrant_mean_p_substrate"],
        "consensus_substrate": cs_stats,
        "disagreement_analysis": {
            "mo_vs_ca_features": disagree_stats["mo_vs_ca_features"],
            "ro_vs_cs_features": disagree_stats["ro_vs_cs_features"],
            "mo_per_language": disagree_stats["mo_per_language"],
            "ro_per_language": disagree_stats["ro_per_language"],
        },
    }

    with open(RESULTS / "consensus_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {RESULTS / 'consensus_summary.json'}")

    # 2. consensus_substrates.csv
    cs_ranked.to_csv(RESULTS / "consensus_substrates.csv", encoding="utf-8")
    print(f"  Saved: {RESULTS / 'consensus_substrates.csv'} ({len(cs_ranked)} rows)")

    # 3. cross_language_consensus.csv
    cross_lang_df.to_csv(RESULTS / "cross_language_consensus.csv", index=False, encoding="utf-8")
    print(f"  Saved: {RESULTS / 'cross_language_consensus.csv'} ({len(cross_lang_df)} rows)")

    # 4. quadrant_comparison.png — already saved in make_visualization()

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    kappa = agreement_stats["cohens_kappa"]
    n_cs = cs_stats["n_cs"]
    n_cross = cs_stats["n_cross_language_4plus"]
    n_total = agreement_stats["n_total"]

    # Interpret kappa
    if kappa >= 0.81:
        kappa_label = "almost perfect"
    elif kappa >= 0.61:
        kappa_label = "substantial"
    elif kappa >= 0.41:
        kappa_label = "moderate"
    elif kappa >= 0.21:
        kappa_label = "fair"
    else:
        kappa_label = "slight/poor"

    print(f"\n  Cohen's kappa = {kappa:.4f} ({kappa_label} agreement)")
    print(f"  Consensus Substrates: {n_cs} / {n_total} forms ({100*n_cs/n_total:.1f}%)")
    print(f"  Cross-language CS (4+/6 langs): {n_cross} concepts")

    cs_pct = 100 * n_cs / (fm["e022_substrate"] == 1).sum()
    print(f"  CS as % of E022 residuals: {cs_pct:.1f}%")

    if kappa >= 0.4 and n_cs >= 100:
        verdict = "SUCCESS"
        verdict_text = (
            f"Moderate-to-substantial agreement (kappa={kappa:.3f}). "
            f"{n_cs} consensus substrates provide a high-confidence core set. "
            f"{n_cross} concepts attested across 4+ languages suggest "
            f"genuine pre-Austronesian remnants."
        )
    elif kappa >= 0.2 and n_cs >= 50:
        verdict = "PARTIAL"
        verdict_text = (
            f"Fair agreement (kappa={kappa:.3f}). "
            f"{n_cs} consensus substrates form a moderate core set. "
            f"Disagreements ({agreement_stats['quadrant_counts'].get('MO',0)} ML-only, "
            f"{agreement_stats['quadrant_counts'].get('RO',0)} rule-only) warrant investigation."
        )
    else:
        verdict = "INCONCLUSIVE"
        verdict_text = (
            f"Low agreement (kappa={kappa:.3f}). Methods may be detecting "
            f"different signals. More data or refined methods needed."
        )

    print(f"\n  Verdict: {verdict}")
    print(f"  {verdict_text}")

    # Save verdict
    verdict_obj = {
        "verdict": verdict,
        "kappa": kappa,
        "kappa_label": kappa_label,
        "n_consensus_substrates": n_cs,
        "n_cross_language_4plus": n_cross,
        "summary": verdict_text,
    }
    with open(RESULTS / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_obj, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {RESULTS / 'verdict.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
