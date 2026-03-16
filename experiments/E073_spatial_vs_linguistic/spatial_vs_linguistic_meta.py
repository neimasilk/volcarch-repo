#!/usr/bin/env python3
"""
E073: Spatial vs Linguistic Evidence Meta-Test
===============================================
Tests the hypothesis: "Volcanic informedness is behavioral/spatial, not lexical."

Compiles all volcanic-awareness test results across two domains:
  SPATIAL: E065 (candi proximity), E066 (candi orientation), ADV-3 (survey control)
  LINGUISTIC: E029 (substrate phonology), E038 (volcanic vocabulary), E067 (toponyms)

Applies Fisher's combined probability test within each domain, then compares.
If spatial domain shows combined significance while linguistic does not,
this supports the "behavioral not lexical" thesis.

Also: effect size comparison using rank-biserial correlation of p-values.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import csv

# ── Evidence compilation ──────────────────────────────────────────────

SPATIAL_EVIDENCE = [
    {
        "experiment": "E065",
        "test_name": "Candi volcanic zone overrepresentation (chi-squared)",
        "p_value": 1e-6,  # conservative bound; actual is << 1e-6
        "effect_size": 17.9,  # observed/expected ratio in Zone A
        "effect_type": "overrepresentation_ratio",
        "n": 142,
        "domain": "spatial",
        "supports_thesis": True,
        "notes": "60 candi in Zone A (<10km) vs 3.4 expected"
    },
    {
        "experiment": "E065",
        "test_name": "Candi western quadrant clustering (Rayleigh)",
        "p_value": 3.4e-8,
        "effect_size": 0.472,  # 67/142 in west vs 0.25 expected
        "effect_type": "proportion_excess",
        "n": 142,
        "domain": "spatial",
        "supports_thesis": True,
        "notes": "Western (upslope) preference for candi placement"
    },
    {
        "experiment": "E066",
        "test_name": "Candi equinox alignment (binomial)",
        "p_value": 4.9e-14,
        "effect_size": 0.85,  # 17/20
        "effect_type": "proportion",
        "n": 20,
        "domain": "spatial",
        "supports_thesis": True,
        "notes": "Astronomical orientation distinct from volcanic proximity"
    },
    {
        "experiment": "E066",
        "test_name": "Candi NOT volcano-facing (McNemar)",
        "p_value": 0.0016,
        "effect_size": 0.35,  # only 7/20 face volcano
        "effect_type": "proportion",
        "n": 20,
        "domain": "spatial",
        "supports_thesis": True,
        "notes": "Equinox dominates orientation; volcanic proximity affects SITING not FACING"
    },
    {
        "experiment": "ADV-3",
        "test_name": "Volcanic proximity after survey control (quasi-Poisson LR)",
        "p_value": 0.0015,
        "effect_size": -0.477,  # beta coefficient
        "effect_type": "regression_beta",
        "n": 703,
        "domain": "spatial",
        "supports_thesis": True,
        "notes": "Fewer sites near volcanoes even after controlling road/BPCB/university distance"
    },
]

LINGUISTIC_EVIDENCE = [
    {
        "experiment": "E029",
        "test_name": "Substrate cross-linguistic cognacy (permutation)",
        "p_value": 0.569,
        "effect_size": 0.769,  # mean distance (higher = less similar)
        "effect_type": "mean_distance",
        "n": 266,
        "domain": "linguistic",
        "supports_thesis": False,
        "notes": "No coherent pre-AN substrate phonology"
    },
    {
        "experiment": "E038",
        "test_name": "Volcanic vocabulary diversity by proximity (t-test)",
        "p_value": 0.68,
        "effect_size": -0.42,  # t-statistic
        "effect_type": "t_statistic",
        "n": 1330,
        "domain": "linguistic",
        "supports_thesis": False,
        "notes": "Volcanic terms are core Swadesh — too stable to drift"
    },
    {
        "experiment": "E067",
        "test_name": "Volcanic toponym proximity correlation (Spearman)",
        "p_value": 0.146,
        "effect_size": 0.140,  # rho
        "effect_type": "correlation_rho",
        "n": 110,
        "domain": "linguistic",
        "supports_thesis": False,
        "notes": "Volcanic morphemes NOT concentrated near volcanoes"
    },
    {
        "experiment": "E067",
        "test_name": "Volcanic toponym close vs far (Mann-Whitney)",
        "p_value": 0.734,
        "effect_size": 0.001,  # 3.9% vs 4.0%
        "effect_type": "proportion_difference",
        "n": 25244,
        "domain": "linguistic",
        "supports_thesis": False,
        "notes": "Near-volcano villages use volcanic names at same rate as far villages"
    },
]


def fishers_combined(p_values):
    """Fisher's method: -2 * sum(ln(p)) ~ chi2(2k)"""
    p_arr = np.array(p_values)
    # Clip to avoid log(0)
    p_arr = np.clip(p_arr, 1e-300, 1.0)
    chi2_stat = -2.0 * np.sum(np.log(p_arr))
    df = 2 * len(p_arr)
    combined_p = 1.0 - stats.chi2.cdf(chi2_stat, df)
    return chi2_stat, df, combined_p


def stouffers_z(p_values, directions):
    """Stouffer's Z-method with direction signs."""
    z_scores = []
    for p, d in zip(p_values, directions):
        p_clipped = np.clip(p, 1e-300, 1.0 - 1e-10)
        z = stats.norm.ppf(1 - p_clipped / 2)  # two-tailed to z
        z_scores.append(z * d)
    combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
    combined_p = 2 * (1 - stats.norm.cdf(abs(combined_z)))
    return combined_z, combined_p


def domain_asymmetry_test(spatial_ps, linguistic_ps):
    """
    Mann-Whitney U test: are spatial p-values systematically lower than linguistic?
    Also compute rank-biserial correlation as effect size.
    """
    u_stat, mw_p = stats.mannwhitneyu(spatial_ps, linguistic_ps, alternative='less')
    n1, n2 = len(spatial_ps), len(linguistic_ps)
    # Rank-biserial correlation
    r = 1 - (2 * u_stat) / (n1 * n2)
    return u_stat, mw_p, r


def main():
    print("=" * 70)
    print("E073: Spatial vs Linguistic Evidence — Meta-Test")
    print("  'Volcanic informedness is behavioral, not lexical'")
    print("=" * 70)

    all_evidence = SPATIAL_EVIDENCE + LINGUISTIC_EVIDENCE

    # ── Display evidence table ────────────────────────────────────────
    print(f"\n{'Domain':<12} {'Exp':<8} {'Test':<50} {'p-value':<12} {'Supports?'}")
    print("-" * 95)
    for e in all_evidence:
        marker = "YES" if e['supports_thesis'] else "NO"
        pstr = f"{e['p_value']:.2e}" if e['p_value'] < 0.01 else f"{e['p_value']:.3f}"
        print(f"{e['domain']:<12} {e['experiment']:<8} {e['test_name'][:50]:<50} {pstr:<12} {marker}")

    # ── Fisher's combined test per domain ─────────────────────────────
    spatial_ps = [e['p_value'] for e in SPATIAL_EVIDENCE]
    linguistic_ps = [e['p_value'] for e in LINGUISTIC_EVIDENCE]

    print("\n" + "=" * 70)
    print("FISHER'S COMBINED PROBABILITY TEST (per domain)")
    print("=" * 70)

    s_chi2, s_df, s_p = fishers_combined(spatial_ps)
    print(f"\nSPATIAL domain ({len(spatial_ps)} tests):")
    print(f"  Chi² = {s_chi2:.2f}, df = {s_df}, combined p = {s_p:.2e}")

    l_chi2, l_df, l_p = fishers_combined(linguistic_ps)
    print(f"\nLINGUISTIC domain ({len(linguistic_ps)} tests):")
    print(f"  Chi² = {l_chi2:.2f}, df = {l_df}, combined p = {l_p:.4f}")

    # ── Stouffer's Z ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STOUFFER'S Z-METHOD (per domain)")
    print("=" * 70)

    spatial_dirs = [1 if e['supports_thesis'] else -1 for e in SPATIAL_EVIDENCE]
    linguistic_dirs = [1 if e['supports_thesis'] else -1 for e in LINGUISTIC_EVIDENCE]

    sz, sp = stouffers_z(spatial_ps, spatial_dirs)
    print(f"\nSPATIAL: Z = {sz:.4f}, combined p = {sp:.2e}")

    lz, lp = stouffers_z(linguistic_ps, linguistic_dirs)
    print(f"LINGUISTIC: Z = {lz:.4f}, combined p = {lp:.4f}")

    # ── Domain asymmetry ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DOMAIN ASYMMETRY TEST")
    print("  H0: Spatial p-values = Linguistic p-values")
    print("  H1: Spatial p-values < Linguistic p-values (one-tailed)")
    print("=" * 70)

    u, mw_p, rbc = domain_asymmetry_test(spatial_ps, linguistic_ps)
    print(f"\n  Mann-Whitney U = {u:.1f}")
    print(f"  p-value (one-tailed) = {mw_p:.6f}")
    print(f"  Rank-biserial correlation = {rbc:.4f}")
    print(f"  Interpretation: {'SIGNIFICANT' if mw_p < 0.05 else 'NOT SIGNIFICANT'} asymmetry")

    # ── Log-odds ratio (vote counting) ────────────────────────────────
    print("\n" + "=" * 70)
    print("VOTE COUNTING (at alpha = 0.05)")
    print("=" * 70)

    spatial_sig = sum(1 for p in spatial_ps if p < 0.05)
    linguistic_sig = sum(1 for p in linguistic_ps if p < 0.05)
    print(f"  Spatial: {spatial_sig}/{len(spatial_ps)} significant")
    print(f"  Linguistic: {linguistic_sig}/{len(linguistic_ps)} significant")

    # Fisher's exact test on 2x2 table
    table = [[spatial_sig, len(spatial_ps) - spatial_sig],
             [linguistic_sig, len(linguistic_ps) - linguistic_sig]]
    fe_or, fe_p = stats.fisher_exact(table, alternative='greater')
    print(f"  Fisher's exact: OR = {fe_or:.1f}, p = {fe_p:.4f}")

    # ── Effect size summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MEDIAN -LOG10(P) BY DOMAIN")
    print("=" * 70)

    spatial_logp = [-np.log10(max(p, 1e-300)) for p in spatial_ps]
    linguistic_logp = [-np.log10(max(p, 1e-300)) for p in linguistic_ps]

    print(f"  Spatial median -log10(p) = {np.median(spatial_logp):.2f}")
    print(f"  Linguistic median -log10(p) = {np.median(linguistic_logp):.2f}")
    print(f"  Ratio = {np.median(spatial_logp) / max(np.median(linguistic_logp), 0.01):.1f}×")

    # ── Synthesis ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    asymmetric = mw_p < 0.05
    spatial_combined_sig = s_p < 0.05
    linguistic_combined_null = l_p > 0.05

    if asymmetric and spatial_combined_sig and linguistic_combined_null:
        verdict = "STRONGLY SUPPORTED"
        detail = ("Volcanic informedness is BEHAVIORAL/SPATIAL, not lexical.\n"
                  "  Builders chose WHERE to build (near volcanoes) and HOW to orient\n"
                  "  (equinox alignment), demonstrating sophisticated spatial knowledge.\n"
                  "  But this knowledge left NO detectable trace in language: no volcanic\n"
                  "  vocabulary enrichment, no toponym concentration, no substrate signal.\n\n"
                  "  This is consistent with EMBODIED knowledge (practice, ritual, oral\n"
                  "  tradition) rather than LEXICALIZED knowledge (vocabulary, naming).\n"
                  "  Implication: archaeological survey (spatial methods) will recover\n"
                  "  evidence of volcanic awareness; linguistic methods will not.")
    elif spatial_combined_sig and not linguistic_combined_null:
        verdict = "PARTIALLY SUPPORTED"
        detail = "Both domains show significance. Asymmetry may be degree, not kind."
    else:
        verdict = "NOT SUPPORTED"
        detail = "No clear domain asymmetry."

    print(f"\n  Thesis: '{verdict}'")
    print(f"\n  {detail}")

    # ── Save results ──────────────────────────────────────────────────
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    results = {
        "experiment": "E073",
        "title": "Spatial vs Linguistic Evidence Meta-Test",
        "thesis": "Volcanic informedness is behavioral/spatial, not lexical",
        "verdict": verdict,
        "spatial_domain": {
            "n_tests": len(spatial_ps),
            "n_significant": spatial_sig,
            "fisher_chi2": round(s_chi2, 2),
            "fisher_df": s_df,
            "fisher_p": float(f"{s_p:.2e}"),
            "stouffer_z": round(sz, 4),
            "stouffer_p": float(f"{sp:.2e}"),
            "median_neg_log10_p": round(np.median(spatial_logp), 2),
            "tests": [{"experiment": e["experiment"], "test": e["test_name"],
                       "p": e["p_value"], "supports": bool(e["supports_thesis"])}
                      for e in SPATIAL_EVIDENCE]
        },
        "linguistic_domain": {
            "n_tests": len(linguistic_ps),
            "n_significant": linguistic_sig,
            "fisher_chi2": round(l_chi2, 2),
            "fisher_df": l_df,
            "fisher_p": round(l_p, 4),
            "stouffer_z": round(lz, 4),
            "stouffer_p": round(lp, 4),
            "median_neg_log10_p": round(np.median(linguistic_logp), 2),
            "tests": [{"experiment": e["experiment"], "test": e["test_name"],
                       "p": e["p_value"], "supports": bool(e["supports_thesis"])}
                      for e in LINGUISTIC_EVIDENCE]
        },
        "asymmetry_test": {
            "mann_whitney_u": round(u, 1),
            "p_value_one_tailed": round(mw_p, 6),
            "rank_biserial_r": round(rbc, 4),
            "significant": bool(asymmetric)
        },
        "vote_counting": {
            "spatial_significant": spatial_sig,
            "spatial_total": len(spatial_ps),
            "linguistic_significant": linguistic_sig,
            "linguistic_total": len(linguistic_ps),
            "fisher_exact_or": round(fe_or, 1) if fe_or != float('inf') else "inf",
            "fisher_exact_p": round(fe_p, 4)
        }
    }

    with open(results_dir / "e073_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_dir / 'e073_results.json'}")

    # CSV for all evidence
    with open(results_dir / "evidence_table.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "domain", "experiment", "test_name", "p_value",
            "effect_size", "effect_type", "n", "supports_thesis", "notes"
        ])
        writer.writeheader()
        for e in all_evidence:
            writer.writerow(e)
    print(f"  Evidence table saved to {results_dir / 'evidence_table.csv'}")


if __name__ == "__main__":
    main()
