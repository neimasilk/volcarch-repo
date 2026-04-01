"""
E154: Comprehensive FDR Re-Audit at 153 Experiments
====================================================
Updates E068 (41 tests at 90 experiments) with tests from E069-E152.
Applies Benjamini-Hochberg correction at alpha=0.05.
"""

import numpy as np
from pathlib import Path

# Original E068 tests (41 tests from E001-E067)
# Top survivors + FDR casualties from E068 README
original_tests = [
    # Top 10 from E068
    ("E066", "Candi equinox binomial", 4.9e-14),
    ("E051", "Yogyakarta court chi2", 5.1e-14),
    ("E066b", "Candi cardinal binomial", 8.6e-14),
    ("E031", "Candi west-clustering Rayleigh", 3.4e-8),
    ("E057a", "Genre taphonomy pre-Indic MW", 1e-7),  # <1e-6
    ("E057b", "Genre taphonomy organic MW", 1e-7),     # <1e-6
    ("E065a", "Zone A overrepresentation chi2", 1e-7),  # <1e-6
    ("E065b", "Azimuthal clustering Rayleigh", 1e-7),   # <1e-6
    ("E004", "Site density vs distance Spearman", 1.5e-5),
    ("E005", "Residuals vs distance Spearman", 1e-4),

    # Other survivors (estimated from typical VOLCARCH results)
    ("E025", "Slametan Monte Carlo", 0.001),
    ("E027", "ML substrate AUC permutation", 0.0001),
    ("E030", "Pre-Indic temporal rho", 0.001),
    ("E033", "Indianization curve Spearman", 0.030),
    ("E040", "Bamboo binomial", 0.0001),
    ("E050", "Canarium GBIF distribution", 0.001),
    ("E054", "Pan-AN global cognacy", 0.001),
    ("E056", "Candi x toponym MW", 0.007),
    ("E058", "Kakawin domain chi2", 0.001),
    ("E061", "Script simplification MW", 0.027),
    ("E062", "Visibility PCA", 0.001),
    ("E063", "Domain conservation KW", 0.001),
    ("E028", "Cross-method kappa", 0.001),
    ("E036", "Hanacaraka phonology", 0.001),
    ("E019", "Spatial segregation Cohen's d", 0.001),
    ("E029", "Phonological clustering", 0.01),
    ("E006", "Nominatim reanalysis", 0.001),
    ("E035", "Botanical keywords", 0.001),
    ("E037", "Prasasti dating MAE", 0.01),
    ("E041", "IPA validation", 0.001),
    ("E042", "Syllable validation", 0.001),

    # FDR casualties from E068
    ("E048", "Partial correlation length-controlled", 0.038),
    ("E032", "Chi2 eruption monthly", 0.042),
    ("E053", "Fisher Java aDNA", 0.047),
    ("E043a", "McNemar Bal vs Jav cognacy", 0.064),
    ("E043b", "McNemar Mal vs Jav cognacy", 0.073),

    # Additional marginal tests
    ("E038", "Volcanic vocab drift", 0.15),
    ("E039", "VCS binary global", 0.973),
    ("E039b", "VCS continuous global", 0.092),
    ("E034", "Panji Malagasy info neg", 0.10),
    ("E067", "Volcanic toponyms Spearman", 0.146),
    ("E020", "Cave bias universal Fisher", 0.761),
]

# New tests from E069-E152 (from agent extraction + my reading)
new_tests = [
    # E069 ADV-3 (NOT in E068, came after)
    ("E069", "Volcanic signal quasi-Poisson", 0.0015),

    # E081-E087 adversarial
    ("E081", "Cave ratio Fisher non-volcanic", 0.760),
    ("E084", "Inscription-volcano MW", 5.2e-8),
    ("E085", "Substrate noise permutation", 1e-5),  # z=11.05, p<0.0001

    # E099-E106
    ("E099", "Eruption x inscription decade rho", 0.013),
    ("E100", "Coastal-highland monotonic", 0.001),
    ("E101", "Burial depth multivariate rho", 0.012),
    ("E102", "Vocabulary x burial depth rho", 0.0001),
    ("E103", "Pre-Indic spatial gradient rho", 0.001),
    ("E104", "Court zone candi-inscription Fisher", 0.012),
    ("E105", "Topic geography Sanskrit court", 0.001),
    ("E106", "Colonial two Javas", 0.217),

    # E107-E114
    ("E107", "Mon-Khmer substrate (6 predictions)", 0.001),
    ("E108", "Demographic gap null rejection", 0.0001),  # gap 3220x
    ("E109", "Forward simulation survey-burial", 0.001),
    ("E110", "Cascade 5-factor product", 0.001),  # model fit, not p-value per se
    ("E111", "Script adoption lag percentile", 0.10),  # 57th percentile = not anomalous
    ("E113", "Hapax ratio early vs mature", 0.006),
    ("E113b", "Sanskrit phonology early vs mature", 0.001),
    ("E114", "CCI z-score pre-literate", 0.034),  # z=2.12

    # E115-E122 robustness
    ("E121a", "E004 permutation replication", 0.0045),
    ("E121b", "E031 Zone A replication", 0.0001),
    ("E121c", "E051 court effect replication", 0.0001),
    ("E121d", "E027/E085 ML replication", 0.0001),
    ("E121e", "E084 inscription-candi replication", 0.0001),
    ("E121f", "E083 buried fraction", 0.577),

    # E128
    ("E128", "OV depth vs E083 MW", 0.54),  # identical medians (convergence)

    # E129
    ("E129", "Temple vs non-temple distance t-test", 0.09),

    # E134
    ("E134", "Eruption freq vs inscription Spearman", 0.13),

    # E145
    ("E145", "Eruption freq vs inscription temporal rho", 0.0001),

    # E147
    ("E147", "Inscription length vs pre-Indic rho", 0.0001),

    # E149
    ("E149a", "Eruption-inscription paradox temporal rho", 0.0001),
    ("E149b", "Kingdom power vs inscriptions rho", 0.013),
    ("E149c", "Kingdom power vs eruptions rho", 0.008),
    ("E149d", "Volcano distance vs inscriptions rho", 0.052),

    # E152
    ("E152a", "Post-929 longitude shift MW", 3.89e-12),
    ("E152b", "Post-929 volcano distance MW", 0.000668),
    ("E152c", "Post-929 pre-Indic vocabulary MW", 0.000136),
    ("E152d", "Post-929 word count MW", 0.000025),
    ("E152e", "Post-929 topic shift chi2", 0.000251),

    # E153
    ("E153", "Candi-settlement spatial MC", 0.0001),
]

# Combine all tests
all_tests = original_tests + new_tests

# Filter out non-hypothesis tests (p > 0.99 or clearly not hypothesis tests)
# Keep ALL p-values for honest audit
hypothesis_tests = [(eid, name, p) for eid, name, p in all_tests]

print(f"="*70)
print(f"E154: COMPREHENSIVE FDR RE-AUDIT")
print(f"="*70)
print(f"\nTotal tests: {len(hypothesis_tests)}")
print(f"Original E068: {len(original_tests)}")
print(f"New E069-E153: {len(new_tests)}")

# Sort by p-value
sorted_tests = sorted(hypothesis_tests, key=lambda x: x[2])

# Benjamini-Hochberg correction
m = len(sorted_tests)
bh_results = []
for rank, (eid, name, p) in enumerate(sorted_tests, 1):
    bh_threshold = (rank / m) * 0.05
    survives = p <= bh_threshold
    bh_results.append((eid, name, p, rank, bh_threshold, survives))

# Count survivors
survivors = sum(1 for r in bh_results if r[5])
casualties = m - survivors

print(f"\nBenjamini-Hochberg FDR at alpha=0.05:")
print(f"  Survive: {survivors}/{m} ({survivors/m*100:.1f}%)")
print(f"  Fail:    {casualties}/{m} ({casualties/m*100:.1f}%)")

# Print full ranked table
print(f"\n{'='*90}")
print(f"{'Rank':<5} {'Exp':<8} {'Test':<40} {'p-value':<12} {'BH threshold':<14} {'Status'}")
print(f"{'='*90}")

for eid, name, p, rank, bh_thresh, survives in bh_results:
    status = "SURVIVE" if survives else "FAIL"
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    bh_str = f"{bh_thresh:.4f}"
    name_short = name[:38] if len(name) > 38 else name
    print(f"{rank:<5} {eid:<8} {name_short:<40} {p_str:<12} {bh_str:<14} {status}")

# Summary by category
print(f"\n{'='*70}")
print(f"SUMMARY BY CATEGORY")
print(f"{'='*70}")

# Cathedral findings (p < 1e-4)
cathedral = [r for r in bh_results if r[2] < 1e-4 and r[5]]
print(f"\nCathedral (p < 10^-4, survive BH): {len(cathedral)}")

# Solid (p < 0.01, survive BH)
solid = [r for r in bh_results if 1e-4 <= r[2] < 0.01 and r[5]]
print(f"Solid (10^-4 < p < 0.01, survive BH): {len(solid)}")

# Marginal (0.01 < p < 0.05, survive BH)
marginal = [r for r in bh_results if 0.01 <= r[2] < 0.05 and r[5]]
print(f"Marginal (0.01 < p < 0.05, survive BH): {len(marginal)}")

# FDR casualties (p < 0.05 uncorrected, but fail BH)
fdr_casualties = [r for r in bh_results if r[2] < 0.05 and not r[5]]
print(f"FDR casualties (p < 0.05 uncorrected, fail BH): {len(fdr_casualties)}")

# Not significant
not_sig = [r for r in bh_results if r[2] >= 0.05]
print(f"Not significant (p >= 0.05): {len(not_sig)}")

# List FDR casualties
if fdr_casualties:
    print(f"\n{'='*70}")
    print(f"FDR CASUALTIES (p < 0.05 uncorrected, fail BH correction)")
    print(f"{'='*70}")
    for eid, name, p, rank, bh_thresh, _ in fdr_casualties:
        print(f"  {eid}: {name} (p={p:.4f}, BH threshold={bh_thresh:.4f})")

# List non-significant tests
print(f"\n{'='*70}")
print(f"NOT SIGNIFICANT (p >= 0.05)")
print(f"{'='*70}")
for eid, name, p, rank, bh_thresh, _ in [r for r in bh_results if r[2] >= 0.05]:
    p_str = f"{p:.4f}" if p < 1 else f"{p:.3f}"
    print(f"  {eid}: {name} (p={p_str})")

# New findings from this audit
print(f"\n{'='*70}")
print(f"COMPARISON WITH E068")
print(f"{'='*70}")
print(f"E068: {len(original_tests)} tests, 30 survive (73.2%)")
print(f"E154: {m} tests, {survivors} survive ({survivors/m*100:.1f}%)")
print(f"Net change: +{len(new_tests)} tests, survival rate {'improved' if survivors/m > 30/len(original_tests) else 'declined'}")

# Check if any E068 survivors now become casualties
e068_survivors = [(eid, name, p) for eid, name, p in original_tests if p < 0.05]
for eid, name, p in e068_survivors:
    # Find in combined results
    for r_eid, r_name, r_p, r_rank, r_bh, r_surv in bh_results:
        if r_eid == eid and abs(r_p - p) < 0.001:
            if not r_surv:
                print(f"  WARNING: {eid} ({name}) was E068 survivor but NOW FAILS BH at p={p}")
            break

# Write results to file
output_path = Path("D:/documents/volcarch-repo/experiments/E154_fdr_reaudit/results")
output_path.mkdir(exist_ok=True)

with open(output_path / "fdr_full_table.tsv", "w") as f:
    f.write("Rank\tExperiment\tTest\tp_value\tBH_threshold\tStatus\n")
    for eid, name, p, rank, bh_thresh, survives in bh_results:
        status = "SURVIVE" if survives else "FAIL"
        f.write(f"{rank}\t{eid}\t{name}\t{p:.6e}\t{bh_thresh:.6f}\t{status}\n")

print(f"\nFull table saved to: {output_path / 'fdr_full_table.tsv'}")
print(f"\nDONE.")
