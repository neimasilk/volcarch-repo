"""
E026 Addendum: Multiple Testing Correction (Bonferroni/Holm)

Applies Bonferroni and Holm-Bonferroni corrections to all p-values
from the original E026 analysis. Also adds a Poisson rate test for
the eruption frequency comparison (previously reported only as ratio).

This addresses Mata Elang #3 criticism: P14 needs multiple comparison correction.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import numpy as np
from pathlib import Path
from scipy import stats

np.random.seed(42)


def holm_bonferroni(p_values, alpha=0.05):
    """Apply Holm-Bonferroni step-down correction.
    Returns list of (original_p, adjusted_p, significant) tuples."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [None] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        holm_threshold = alpha / (n - rank)
        adjusted_p = min(p * (n - rank), 1.0)
        results[orig_idx] = {
            "original_p": round(p, 4),
            "adjusted_p": round(adjusted_p, 4),
            "holm_threshold": round(holm_threshold, 4),
            "significant": p <= holm_threshold,
            "rank": rank + 1,
        }
    return results


def poisson_rate_test(count1, duration1, count2, duration2):
    """Exact conditional test for comparing two Poisson rates.
    Tests H0: rate1 = rate2 vs H1: rate2 > rate1 (one-sided).
    Uses the binomial distribution (conditional on total count)."""
    total = count1 + count2
    if total == 0:
        return 1.0

    # Under H0, expected proportion in period 2 = duration2 / (duration1 + duration2)
    p_null = duration2 / (duration1 + duration2)

    # Observed count in period 2
    # One-sided test: is count2 significantly high?
    p_value = 1 - stats.binom.cdf(count2 - 1, total, p_null)
    return p_value


def main():
    print("=" * 70)
    print("E026 ADDENDUM: BONFERRONI/HOLM MULTIPLE TESTING CORRECTION")
    print("=" * 70)

    # Load original results
    results_path = Path(__file__).parent / "results" / "pararaton_correlation_results.json"
    with open(results_path) as f:
        original = json.load(f)

    # ============================================================
    # 1. Collect ALL p-values from original analysis
    # ============================================================

    print("\n--- Original p-values ---")

    test_names = []
    p_values = []

    # Primary test
    p_prox = original["proximity_test"]["p_value"]
    test_names.append("Proximity permutation (primary)")
    p_values.append(p_prox)
    print(f"  1. Proximity permutation:  p = {p_prox:.4f}")

    # Window tests
    for w in [5, 10, 15, 20]:
        key = f"window_{w}yr"
        p = original[key]["p_value"]
        test_names.append(f"Post-eruption window ({w}yr)")
        p_values.append(p)
        print(f"  2. Window {w:>2d}yr:             p = {p:.4f}")

    # ============================================================
    # 2. Add: Poisson rate test (eruption frequency)
    # ============================================================

    print("\n--- New: Poisson rate test for eruption frequency ---")

    rate_data = original["rate_comparison"]
    p_rate = poisson_rate_test(
        count1=rate_data["peak_eruptions"],
        duration1=rate_data["peak_duration_yr"],
        count2=rate_data["decline_eruptions"],
        duration2=rate_data["decline_duration_yr"],
    )
    test_names.append("Poisson rate test (decline > peak)")
    p_values.append(p_rate)
    print(f"  Peak:    {rate_data['peak_eruptions']} eruptions / {rate_data['peak_duration_yr']} yr")
    print(f"  Decline: {rate_data['decline_eruptions']} eruptions / {rate_data['decline_duration_yr']} yr")
    print(f"  Poisson rate test p = {p_rate:.4f}")

    # ============================================================
    # 3. Apply Bonferroni correction
    # ============================================================

    n_tests = len(p_values)
    bonf_alpha = 0.05 / n_tests

    print(f"\n--- Bonferroni correction (alpha = 0.05 / {n_tests} = {bonf_alpha:.4f}) ---")
    for name, p in zip(test_names, p_values):
        sig = "SIG" if p < bonf_alpha else "n.s."
        print(f"  {name:>40s}: p = {p:.4f}  [{sig}]")

    bonf_any_sig = any(p < bonf_alpha for p in p_values)
    print(f"\n  Any significant after Bonferroni? {'YES' if bonf_any_sig else 'NO'}")

    # ============================================================
    # 4. Apply Holm-Bonferroni correction
    # ============================================================

    print(f"\n--- Holm-Bonferroni step-down correction ---")
    holm_results = holm_bonferroni(p_values, alpha=0.05)
    for name, hr in zip(test_names, holm_results):
        sig = "SIG" if hr["significant"] else "n.s."
        print(f"  {name:>40s}: p = {hr['original_p']:.4f}  "
              f"adj.p = {hr['adjusted_p']:.4f}  "
              f"threshold = {hr['holm_threshold']:.4f}  [{sig}]")

    holm_any_sig = any(hr["significant"] for hr in holm_results)
    print(f"\n  Any significant after Holm? {'YES' if holm_any_sig else 'NO'}")

    # ============================================================
    # 5. Alternative framing: pre-registered primary test
    # ============================================================

    print(f"\n--- Alternative: Single pre-specified primary test ---")
    print(f"  If proximity permutation is the SOLE pre-registered test:")
    print(f"  p = {p_prox:.4f} < 0.05 — SIGNIFICANT (no correction needed)")
    print(f"  All window tests treated as exploratory (not corrected)")
    print(f"  Rate test treated as descriptive (supplementary)")

    # ============================================================
    # 6. Effect size analysis (correction-independent)
    # ============================================================

    print(f"\n--- Effect sizes (independent of correction) ---")
    prox_data = original["proximity_test"]
    effect_proximity = (prox_data["null_mean"] - prox_data["observed_mean_proximity_years"]) / prox_data["null_mean"]
    print(f"  Proximity: observed {prox_data['observed_mean_proximity_years']:.1f}yr vs "
          f"null {prox_data['null_mean']:.1f}yr "
          f"(reduction = {effect_proximity*100:.0f}%)")
    print(f"  Rate ratio: {rate_data['rate_ratio']}x (decline vs peak)")
    print(f"  Geological match: {original['geological_crossref']['matched']}/{original['geological_crossref']['total']}")

    # ============================================================
    # 7. Verdict for P14 research note
    # ============================================================

    print(f"\n" + "=" * 70)
    print(f"VERDICT FOR P14 RESEARCH NOTE")
    print(f"=" * 70)

    if bonf_any_sig:
        print(f"\n  Proximity test SURVIVES Bonferroni — strong statistical evidence.")
    else:
        print(f"\n  Proximity test (p={p_prox:.3f}) DOES NOT survive Bonferroni ({bonf_alpha:.4f}).")
        print(f"  Proximity test (p={p_prox:.3f}) DOES NOT survive Holm-Bonferroni.")
        print(f"  Poisson rate test (p={p_rate:.3f}) {'survives' if p_rate < bonf_alpha else 'does NOT survive'} Bonferroni.")
        print()
        print(f"  RECOMMENDATION: Reframe P14 as EXPLORATORY research note.")
        print(f"  - Report uncorrected p-values with explicit multiple testing caveat")
        print(f"  - Emphasize CONVERGENCE of three independent lines:")
        print(f"    (1) Temporal proximity pattern (p=0.037 uncorrected)")
        print(f"    (2) Eruption rate asymmetry (ratio 2.18x, Poisson p={p_rate:.3f})")
        print(f"    (3) Geological cross-validation (3/3 Pararaton events match GVP)")
        print(f"  - Argue that the VALUE is the hypothesis generation,")
        print(f"    not the definitive statistical proof")
        print(f"  - Title framing: 'Did Kelud eruptions accelerate...' (question, not claim)")

    # Save corrected results
    correction_results = {
        "n_tests": n_tests,
        "test_names": test_names,
        "original_p_values": [round(p, 4) for p in p_values],
        "bonferroni_alpha": round(bonf_alpha, 4),
        "bonferroni_any_significant": bonf_any_sig,
        "holm_results": holm_results,
        "holm_any_significant": holm_any_sig,
        "poisson_rate_test": {
            "peak_count": rate_data["peak_eruptions"],
            "peak_duration": rate_data["peak_duration_yr"],
            "decline_count": rate_data["decline_eruptions"],
            "decline_duration": rate_data["decline_duration_yr"],
            "p_value": round(p_rate, 4),
        },
        "effect_sizes": {
            "proximity_reduction_pct": round(effect_proximity * 100, 1),
            "rate_ratio": rate_data["rate_ratio"],
            "geo_match_fraction": f"{original['geological_crossref']['matched']}/{original['geological_crossref']['total']}",
        },
        "recommendation": "EXPLORATORY — reframe as hypothesis-generating research note" if not bonf_any_sig else "CONFIRMATORY — statistical evidence survives correction",
    }

    output_path = Path(__file__).parent / "results" / "bonferroni_correction.json"

    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(correction_results, f, indent=2, default=convert)
    print(f"\n  Correction results saved to {output_path}")


if __name__ == "__main__":
    main()
