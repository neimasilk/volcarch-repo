"""
E107: ADV-5 Re-examination — Is Iban+Malay Really a Negative Control?
=====================================================================
E087 found that C5 (Iban+Malay) achieves AUC=0.713 on the substrate
detector, nearly matching Sulawesi's 0.727. This was interpreted as
the detector picking up documentation artifacts.

BUT: Iban has well-documented Mon-Khmer (Aslian) substrate influence
(Adelaar 1985, 1992, 2005; Blust 2010). If the detector is picking
up REAL Mon-Khmer substrate in Iban, then C5 is actually a POSITIVE
control, and E027 is STRONGER than originally assessed.

This experiment tests: do C5 "residual" forms look like Mon-Khmer
substrate (short, CVC, no AN prefixes) or like generic documentation
artifacts (similar profile to Sulawesi residuals)?

PREDICTIONS:
  If Mon-Khmer substrate → C5 residuals should be:
    SHORTER (monosyllabic/sesquisyllabic)
    MORE consonant-final (CVC not CVCV)
    FEWER Austronesian prefixes
    MORE consonant clusters
  If documentation artifact → C5 residuals should:
    MATCH Sulawesi residual profile (generic non-mainstream)
"""
import csv
import json
import re
import sys
import io
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

REPO = Path(__file__).parent.parent.parent
ABVD = REPO / "experiments" / "E022_linguistic_subtraction" / "data" / "abvd" / "cldf"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# Language sets
SULAWESI_LANGS = {"27": "Muna", "48": "Bugis", "166": "Makassar",
                  "192": "Wolio", "226": "Toraja-Sadan", "674": "Tolaki"}
C5_LANGS = {"28": "Iban", "233": "Malay"}
C1_LANGS = {"277": "Tagalog", "153": "Cebuano"}

# Known Mon-Khmer loanwords in Malay/Iban (from Adelaar 1995, Blust 2010, Thurgood 1999)
# These are words with established Mon-Khmer etymologies
KNOWN_MK_LOANS = {
    # Animal terms
    "kerbau", "kerbaw",  # water buffalo (< MK *krpau)
    "beruk",              # macaque (< MK)
    "tupai",              # squirrel
    "kutu",               # body louse (debated)
    # Agricultural/plant
    "padi",               # rice (some argue MK origin)
    "tebu",               # sugarcane (< MK *tbuus)
    "kelapa",             # coconut (debated)
    # Material/technology
    "besi",               # iron (< MK *besi)
    "perahu",             # boat (debated AN vs MK)
    "timah",              # tin
    # Body/nature
    "bulan",              # moon (core AN but MK cognate exists)
    "tanah",              # earth/soil
    # Cultural
    "raja",               # king (< Sanskrit, but early route via MK?)
}

VOWELS = set("aeiouəɛɨɔæøüöäåãẽĩõũâêîôûàèìòùáéíóú")
AN_PREFIXES = ("ma-", "me-", "mo-", "pa-", "ka-", "ta-", "na-", "po-",
               "ma", "me", "mo", "pa", "ka", "ta", "na", "po",
               "maŋ", "meŋ", "moŋ", "paŋ", "aŋ", "mak-", "mat-")
NASAL_CLUSTERS = ("ng", "mb", "nd", "nj", "mp", "nk", "ŋg", "ŋk", "nc", "nt", "ŋ")


def clean_form(raw):
    s = raw.strip()
    s = re.sub(r'\[.*?\]\s*', '', s)
    return s.strip(" -,;.")


def phonological_profile(form):
    """Extract Mon-Khmer diagnostic features."""
    fl = form.lower()
    length = len(form)
    n_vowels = sum(1 for c in fl if c in VOWELS)
    vowel_ratio = n_vowels / max(length, 1)
    ends_vowel = 1 if (fl and fl[-1] in VOWELS) else 0

    # Syllable count estimate (vowel sequences)
    syllables = 0
    in_vowel = False
    for c in fl:
        if c in VOWELS:
            if not in_vowel:
                syllables += 1
                in_vowel = True
        else:
            in_vowel = False
    syllables = max(syllables, 1)

    # Prefix detection
    has_prefix = 0
    for p in AN_PREFIXES:
        if fl.startswith(p):
            has_prefix = 1
            break

    # Consonant clusters
    n_cc = 0
    consec = 0
    for c in fl:
        if c not in VOWELS and c.isalpha():
            consec += 1
            if consec == 2:
                n_cc += 1
        else:
            consec = 0

    # Glottal
    has_glottal = 1 if ("ʔ" in form or "'" in form) else 0

    # Mon-Khmer diagnostic: monosyllabic (1 syl) or sesquisyllabic (1.5 syl = schwa+CVC)
    is_monosyllabic = 1 if syllables == 1 else 0
    is_sesquisyllabic = 1 if (syllables == 2 and length <= 5) else 0
    is_mk_shape = 1 if (is_monosyllabic or is_sesquisyllabic) else 0

    return {
        "length": length,
        "n_vowels": n_vowels,
        "vowel_ratio": round(vowel_ratio, 4),
        "ends_vowel": ends_vowel,
        "syllables": syllables,
        "has_an_prefix": has_prefix,
        "n_consonant_clusters": n_cc,
        "has_glottal": has_glottal,
        "is_monosyllabic": is_monosyllabic,
        "is_sesquisyllabic": is_sesquisyllabic,
        "is_mk_shape": is_mk_shape,
    }


def load_lang_data(target_langs):
    """Load ABVD data for specified languages, return forms with labels."""
    params = {}
    with open(ABVD / "parameters.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params[row["ID"]] = row["Name"]

    cognate_info = defaultdict(list)
    with open(ABVD / "cognates.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cognate_info[row["Form_ID"]].append(row["Cognateset_ID"])

    forms = []
    with open(ABVD / "forms.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Language_ID"] not in target_langs:
                continue
            raw = row.get("Value", "") or row.get("Form", "")
            form = clean_form(raw)
            if not form or len(form) < 1:
                continue

            cognacy = row.get("Cognacy", "").strip()
            cog_sets = cognate_info.get(row["ID"], [])
            label = 1 if (cognacy or cog_sets) else 0  # 1=AN, 0=residual

            concept = params.get(row["Parameter_ID"], "?")
            lang_name = target_langs[row["Language_ID"]]

            profile = phonological_profile(form)
            profile["form"] = form
            profile["concept"] = concept
            profile["language"] = lang_name
            profile["label"] = label
            profile["form_id"] = row["ID"]
            forms.append(profile)

    return forms


def compare_profiles(residuals_a, residuals_b, name_a, name_b):
    """Compare phonological profiles of two sets of residual forms."""
    metrics = ["length", "vowel_ratio", "ends_vowel", "syllables",
               "has_an_prefix", "n_consonant_clusters", "has_glottal",
               "is_mk_shape"]

    results = {}
    print(f"\n  {'Feature':<25} {name_a:>12} {name_b:>12} {'p-value':>10} {'Direction':>15}")
    print(f"  {'-'*74}")

    for m in metrics:
        vals_a = [r[m] for r in residuals_a]
        vals_b = [r[m] for r in residuals_b]

        mean_a = np.mean(vals_a)
        mean_b = np.mean(vals_b)

        # Mann-Whitney for ordinal/continuous, Fisher for binary
        if m in ("ends_vowel", "has_an_prefix", "has_glottal", "is_mk_shape"):
            # Binary: use chi2 or Mann-Whitney
            stat, p = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        else:
            stat, p = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')

        if mean_a > mean_b:
            direction = f"{name_a} higher"
        elif mean_a < mean_b:
            direction = f"{name_b} higher"
        else:
            direction = "equal"

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {m:<25} {mean_a:>12.3f} {mean_b:>12.3f} {p:>9.4f}{sig} {direction:>15}")

        results[m] = {
            f"mean_{name_a}": round(float(mean_a), 4),
            f"mean_{name_b}": round(float(mean_b), 4),
            "p_value": round(float(p), 6),
            "direction": direction,
        }

    return results


def main():
    print("=" * 70)
    print("E107: ADV-5 RE-EXAMINATION")
    print("Is Iban+Malay Really a Negative Control?")
    print("=" * 70)

    # Load data for all language sets
    print("\n[1] Loading ABVD data...")
    sulawesi_forms = load_lang_data(SULAWESI_LANGS)
    c5_forms = load_lang_data(C5_LANGS)
    c1_forms = load_lang_data(C1_LANGS)

    # Separate residuals and Austronesian
    sul_residuals = [f for f in sulawesi_forms if f["label"] == 0]
    sul_austronesian = [f for f in sulawesi_forms if f["label"] == 1]
    c5_residuals = [f for f in c5_forms if f["label"] == 0]
    c5_austronesian = [f for f in c5_forms if f["label"] == 1]
    c1_residuals = [f for f in c1_forms if f["label"] == 0]

    # C5 breakdown by language
    iban_residuals = [f for f in c5_residuals if f["language"] == "Iban"]
    malay_residuals = [f for f in c5_residuals if f["language"] == "Malay"]

    print(f"\n  Sulawesi: {len(sulawesi_forms)} forms, {len(sul_residuals)} residual ({100*len(sul_residuals)/len(sulawesi_forms):.1f}%)")
    print(f"  C5 (Iban+Malay): {len(c5_forms)} forms, {len(c5_residuals)} residual ({100*len(c5_residuals)/len(c5_forms):.1f}%)")
    print(f"    Iban: {len(iban_residuals)} residual")
    print(f"    Malay: {len(malay_residuals)} residual")
    print(f"  C1 (Tag+Ceb): {len(c1_forms)} forms, {len(c1_residuals)} residual ({100*len(c1_residuals)/max(len(c1_forms),1):.1f}%)")

    # ================================================================
    # TEST 1: Mon-Khmer shape analysis of C5 residuals
    # ================================================================
    print("\n" + "=" * 70)
    print("[2] TEST 1: MON-KHMER SHAPE ANALYSIS")
    print("If C5 residuals are MK substrate: more monosyllabic/sesquisyllabic")
    print("If documentation artifact: similar to Sulawesi residuals")
    print("=" * 70)

    # MK shape rates
    mk_rate_c5 = np.mean([f["is_mk_shape"] for f in c5_residuals])
    mk_rate_sul = np.mean([f["is_mk_shape"] for f in sul_residuals])
    mk_rate_c5_an = np.mean([f["is_mk_shape"] for f in c5_austronesian])
    mk_rate_sul_an = np.mean([f["is_mk_shape"] for f in sul_austronesian])
    mk_rate_iban = np.mean([f["is_mk_shape"] for f in iban_residuals]) if iban_residuals else 0

    print(f"\n  Mon-Khmer shape rate (monosyllabic OR sesquisyllabic):")
    print(f"    C5 residuals:       {mk_rate_c5:.3f} ({sum(f['is_mk_shape'] for f in c5_residuals)}/{len(c5_residuals)})")
    print(f"    C5 Austronesian:    {mk_rate_c5_an:.3f}")
    print(f"    Iban residuals:     {mk_rate_iban:.3f} ({sum(f['is_mk_shape'] for f in iban_residuals)}/{len(iban_residuals)})")
    print(f"    Sulawesi residuals: {mk_rate_sul:.3f}")
    print(f"    Sulawesi AN:        {mk_rate_sul_an:.3f}")

    # Fisher exact: C5 residuals vs C5 Austronesian for MK shape
    c5_res_mk = sum(f["is_mk_shape"] for f in c5_residuals)
    c5_res_notmk = len(c5_residuals) - c5_res_mk
    c5_an_mk = sum(f["is_mk_shape"] for f in c5_austronesian)
    c5_an_notmk = len(c5_austronesian) - c5_an_mk
    fisher_or, fisher_p = stats.fisher_exact([[c5_res_mk, c5_res_notmk],
                                               [c5_an_mk, c5_an_notmk]])
    print(f"\n  Fisher exact (C5 residual vs C5 AN, MK shape):")
    print(f"    OR = {fisher_or:.3f}, p = {fisher_p:.6f}")

    # ================================================================
    # TEST 2: Full phonological profile comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("[3] TEST 2: PHONOLOGICAL PROFILE COMPARISON")
    print("=" * 70)

    print("\n  --- C5 Residuals vs Sulawesi Residuals ---")
    c5_vs_sul = compare_profiles(c5_residuals, sul_residuals, "C5_res", "Sul_res")

    print("\n  --- C5 Residuals vs C5 Austronesian ---")
    c5_res_vs_an = compare_profiles(c5_residuals, c5_austronesian, "C5_res", "C5_AN")

    print("\n  --- Iban Residuals vs Sulawesi Residuals ---")
    if len(iban_residuals) >= 10:
        iban_vs_sul = compare_profiles(iban_residuals, sul_residuals, "Iban_res", "Sul_res")
    else:
        print(f"  SKIPPED: only {len(iban_residuals)} Iban residuals")
        iban_vs_sul = {}

    # ================================================================
    # TEST 3: Known Mon-Khmer loan overlap
    # ================================================================
    print("\n" + "=" * 70)
    print("[4] TEST 3: KNOWN MON-KHMER LOAN OVERLAP")
    print("=" * 70)

    c5_res_forms = {f["form"].lower() for f in c5_residuals}
    iban_res_forms = {f["form"].lower() for f in iban_residuals}
    mk_overlap_c5 = c5_res_forms & KNOWN_MK_LOANS
    mk_overlap_iban = iban_res_forms & KNOWN_MK_LOANS

    print(f"\n  Known MK loans in database: {len(KNOWN_MK_LOANS)}")
    print(f"  Overlap with C5 residuals: {mk_overlap_c5 if mk_overlap_c5 else 'none'}")
    print(f"  Overlap with Iban residuals: {mk_overlap_iban if mk_overlap_iban else 'none'}")

    # Also check: what concepts are the C5 residuals?
    c5_res_concepts = defaultdict(int)
    for f in c5_residuals:
        c5_res_concepts[f["concept"]] += 1

    # Concepts shared between C5 and Sulawesi residuals
    sul_res_concepts = {f["concept"] for f in sul_residuals}
    c5_res_concept_set = {f["concept"] for f in c5_residuals}
    shared_concepts = c5_res_concept_set & sul_res_concepts
    c5_only_concepts = c5_res_concept_set - sul_res_concepts

    print(f"\n  Concept analysis:")
    print(f"    C5 residual concepts: {len(c5_res_concept_set)}")
    print(f"    Sulawesi residual concepts: {len(sul_res_concepts)}")
    print(f"    Shared (both residual): {len(shared_concepts)} ({100*len(shared_concepts)/max(len(c5_res_concept_set),1):.1f}%)")
    print(f"    C5-only residual concepts: {len(c5_only_concepts)}")

    # ================================================================
    # TEST 4: Syllable distribution comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("[5] TEST 4: SYLLABLE DISTRIBUTION (Mon-Khmer Diagnostic)")
    print("=" * 70)

    for name, forms in [("C5 residuals", c5_residuals),
                        ("Iban residuals", iban_residuals),
                        ("Sulawesi residuals", sul_residuals),
                        ("C1 residuals", c1_residuals)]:
        if not forms:
            continue
        syls = [f["syllables"] for f in forms]
        mono = sum(1 for s in syls if s == 1)
        di = sum(1 for s in syls if s == 2)
        tri = sum(1 for s in syls if s == 3)
        poly = sum(1 for s in syls if s >= 4)
        n = len(syls)
        print(f"\n  {name} (N={n}):")
        print(f"    1-syl: {mono:>4} ({100*mono/n:>5.1f}%)  {'<-- Mon-Khmer' if mono/n > 0.15 else ''}")
        print(f"    2-syl: {di:>4} ({100*di/n:>5.1f}%)  {'<-- Austronesian canonical' if di/n > 0.4 else ''}")
        print(f"    3-syl: {tri:>4} ({100*tri/n:>5.1f}%)")
        print(f"    4+syl: {poly:>4} ({100*poly/n:>5.1f}%)")
        print(f"    Mean: {np.mean(syls):.2f}, Median: {np.median(syls):.1f}")

    # ================================================================
    # TEST 5: Residual-level prediction overlap
    # ================================================================
    print("\n" + "=" * 70)
    print("[6] TEST 5: DO THE SAME CONCEPTS GET FLAGGED?")
    print("    If C5 and Sulawesi flag the same concepts as residual,")
    print("    the signal is documentation bias. If different, it's real.")
    print("=" * 70)

    # For each concept present in both sets, check if residual status matches
    all_concepts = set()
    for f in sulawesi_forms + c5_forms:
        all_concepts.add(f["concept"])

    # For each concept, compute residual rate in each language set
    concept_rates = {}
    for concept in sorted(all_concepts):
        sul_for_concept = [f for f in sulawesi_forms if f["concept"] == concept]
        c5_for_concept = [f for f in c5_forms if f["concept"] == concept]

        if not sul_for_concept or not c5_for_concept:
            continue

        sul_rate = np.mean([f["label"] == 0 for f in sul_for_concept])
        c5_rate = np.mean([f["label"] == 0 for f in c5_for_concept])
        concept_rates[concept] = {"sulawesi": sul_rate, "c5": c5_rate}

    if concept_rates:
        sul_rates = [v["sulawesi"] for v in concept_rates.values()]
        c5_rates = [v["c5"] for v in concept_rates.values()]
        rho, rho_p = stats.spearmanr(sul_rates, c5_rates)

        print(f"\n  Concept-level residual rate correlation:")
        print(f"    N concepts (in both): {len(concept_rates)}")
        print(f"    Spearman rho: {rho:.4f}")
        print(f"    p-value: {rho_p:.6f}")
        print(f"\n  Interpretation:")
        if rho > 0.3 and rho_p < 0.05:
            print(f"    HIGH correlation → same concepts are 'residual' in both")
            print(f"    → suggests DOCUMENTATION ARTIFACT (ABVD gaps shared)")
            concept_verdict = "DOCUMENTATION_ARTIFACT"
        elif rho < 0.1:
            print(f"    LOW correlation → different concepts are 'residual'")
            print(f"    → suggests DIFFERENT MECHANISMS (real substrate in each)")
            concept_verdict = "DIFFERENT_MECHANISMS"
        else:
            print(f"    MODERATE correlation → mixed signal")
            concept_verdict = "MIXED"
    else:
        rho, rho_p = 0, 1
        concept_verdict = "INSUFFICIENT_DATA"

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 70)
    print("[7] VERDICT")
    print("=" * 70)

    # Scoring
    evidence_for_mk = 0
    evidence_for_artifact = 0

    # MK shape rate
    if mk_rate_c5 > mk_rate_sul + 0.05:
        evidence_for_mk += 1
        print(f"\n  [+MK] C5 residuals have MORE MK shapes ({mk_rate_c5:.3f}) than Sulawesi ({mk_rate_sul:.3f})")
    else:
        evidence_for_artifact += 1
        print(f"\n  [+ART] C5 residuals have SIMILAR MK shapes ({mk_rate_c5:.3f}) vs Sulawesi ({mk_rate_sul:.3f})")

    # Syllable length
    c5_mean_syl = np.mean([f["syllables"] for f in c5_residuals])
    sul_mean_syl = np.mean([f["syllables"] for f in sul_residuals])
    if c5_mean_syl < sul_mean_syl - 0.1:
        evidence_for_mk += 1
        print(f"  [+MK] C5 residuals are SHORTER ({c5_mean_syl:.2f} syl) than Sulawesi ({sul_mean_syl:.2f} syl)")
    else:
        evidence_for_artifact += 1
        print(f"  [+ART] C5 residuals are similar length ({c5_mean_syl:.2f} syl) vs Sulawesi ({sul_mean_syl:.2f} syl)")

    # Prefix rate
    c5_prefix = np.mean([f["has_an_prefix"] for f in c5_residuals])
    sul_prefix = np.mean([f["has_an_prefix"] for f in sul_residuals])
    if c5_prefix < sul_prefix - 0.05:
        evidence_for_mk += 1
        print(f"  [+MK] C5 residuals have FEWER AN prefixes ({c5_prefix:.3f}) than Sulawesi ({sul_prefix:.3f})")
    else:
        evidence_for_artifact += 1
        print(f"  [+ART] C5 residuals have similar prefix rate ({c5_prefix:.3f}) vs Sulawesi ({sul_prefix:.3f})")

    # Concept overlap
    if concept_verdict == "DOCUMENTATION_ARTIFACT":
        evidence_for_artifact += 2
        print(f"  [++ART] Same concepts flagged as residual (rho={rho:.3f})")
    elif concept_verdict == "DIFFERENT_MECHANISMS":
        evidence_for_mk += 2
        print(f"  [++MK] Different concepts flagged (rho={rho:.3f}) → different mechanisms")
    else:
        print(f"  [NEUTRAL] Concept overlap is mixed (rho={rho:.3f})")

    # Vowel-final rate
    c5_vf = np.mean([f["ends_vowel"] for f in c5_residuals])
    sul_vf = np.mean([f["ends_vowel"] for f in sul_residuals])
    if c5_vf < sul_vf - 0.05:
        evidence_for_mk += 1
        print(f"  [+MK] C5 residuals end in consonant MORE ({1-c5_vf:.3f}) than Sulawesi ({1-sul_vf:.3f})")
    else:
        evidence_for_artifact += 1
        print(f"  [+ART] C5 residuals have similar vowel-final rate ({c5_vf:.3f}) vs Sulawesi ({sul_vf:.3f})")

    total = evidence_for_mk + evidence_for_artifact
    mk_score = evidence_for_mk / max(total, 1)

    print(f"\n  SCORE: {evidence_for_mk} Mon-Khmer / {evidence_for_artifact} Artifact")
    print(f"  MK probability: {mk_score:.1%}")

    if mk_score >= 0.6:
        verdict = "MON_KHMER_SUBSTRATE"
        implication = (
            "C5 (Iban+Malay) is NOT a clean negative control. "
            "The detector is likely picking up genuine Mon-Khmer substrate in Iban. "
            "E087 ADV-5 should be RECLASSIFIED from GREY ZONE to PARTIAL POSITIVE CONTROL. "
            "E027 substrate detection is STRONGER than previously assessed. "
            "L4 (Cosmological Overwrite) evidence is UPGRADED."
        )
    elif mk_score <= 0.3:
        verdict = "DOCUMENTATION_ARTIFACT"
        implication = (
            "C5 high AUC is driven by ABVD documentation gaps, not real substrate. "
            "E087 ADV-5 stands as originally assessed (GREY ZONE). "
            "E027 results must continue to be reported with caveat. "
            "L4 evidence remains at current level."
        )
    else:
        verdict = "MIXED_SIGNAL"
        implication = (
            "C5 signal contains BOTH documentation artifact AND possible Mon-Khmer substrate. "
            "E087 ADV-5 is partially rehabilitated. "
            "The true negative control performance is likely between C1 (0.568) and C5 (0.713). "
            "Recommend: test with a BETTER negative control (e.g., Formosan language pair)."
        )

    print(f"\n  VERDICT: {verdict}")
    print(f"  IMPLICATION: {implication}")

    # ================================================================
    # Save results
    # ================================================================
    results = {
        "experiment": "E107_adv5_reexamination",
        "date": "2026-03-17",
        "question": "Is C5 (Iban+Malay) really a negative control for substrate detection?",
        "key_insight": "Iban has documented Mon-Khmer (Aslian) substrate (Adelaar 1985, 1992, 2005)",
        "samples": {
            "sulawesi_residuals": len(sul_residuals),
            "c5_residuals": len(c5_residuals),
            "iban_residuals": len(iban_residuals),
            "malay_residuals": len(malay_residuals),
            "c1_residuals": len(c1_residuals),
        },
        "mk_shape_analysis": {
            "c5_residual_mk_rate": round(float(mk_rate_c5), 4),
            "sulawesi_residual_mk_rate": round(float(mk_rate_sul), 4),
            "iban_residual_mk_rate": round(float(mk_rate_iban), 4),
            "fisher_test": {"OR": round(float(fisher_or), 4), "p": round(float(fisher_p), 6)},
        },
        "syllable_analysis": {
            "c5_mean_syllables": round(float(c5_mean_syl), 3),
            "sulawesi_mean_syllables": round(float(sul_mean_syl), 3),
        },
        "concept_overlap": {
            "n_shared_concepts": len(concept_rates) if concept_rates else 0,
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(rho_p), 6),
            "interpretation": concept_verdict,
        },
        "phonological_comparison": c5_vs_sul,
        "scoring": {
            "evidence_for_mk": evidence_for_mk,
            "evidence_for_artifact": evidence_for_artifact,
            "mk_probability": round(float(mk_score), 3),
        },
        "verdict": verdict,
        "implication": implication,
    }

    with open(OUT / "e107_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUT / 'e107_results.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
