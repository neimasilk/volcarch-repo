"""
E022: Linguistic Subtraction POC — Muna Language
=================================================
Test whether vocabulary subtraction reveals a non-trivial residual
(potential pre-Austronesian substrate) in Muna basic vocabulary.

Steps:
1. Load ABVD data for 6 target Sulawesi languages
2. Load cognacy data (PAn cognate detection)
3. Tag Sanskrit loanwords (from de Casparis 1997 common list)
4. Tag Arabic loanwords (from known common Arabic loans in Indonesian)
5. Tag Malay trade vocabulary
6. Count residual per language
7. Cross-language intersection of residuals

Run: python experiments/E022_linguistic_subtraction/poc_muna_subtraction.py
"""
import csv
import re
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).parent.parent.parent
ABVD = REPO / "experiments" / "E022_linguistic_subtraction" / "data" / "abvd" / "cldf"
OUT = REPO / "experiments" / "E022_linguistic_subtraction" / "results"
OUT.mkdir(exist_ok=True)

# 6 target languages with their ABVD IDs
TARGET_LANGS = {
    "27": "Muna",
    "48": "Bugis",       # need to verify ID
    "166": "Makassar",
    "192": "Wolio",
    "226": "Toraja-Sadan",
    "674": "Tolaki",
}

# ============================================================
# Known Sanskrit loanwords common in Austronesian languages
# Source: de Casparis 1997, Zoetmulder 1982, general knowledge
# These are CONCEPTS (ABVD parameter IDs) where Sanskrit borrowing
# is common across Indonesian languages
# ============================================================
SANSKRIT_FORMS = {
    # Common Sanskrit-derived words found across Nusantara
    # Format: lowercase form fragments that indicate Sanskrit origin
    "guru", "raja", "dewa", "agama", "bahasa", "bangsa", "bumi",
    "candra", "desa", "dharma", "jiwa", "kala", "karma", "karya",
    "kota", "loka", "maha", "mantra", "manusia", "marga", "muka",
    "naga", "negara", "pura", "pustaka", "rasa", "rupa", "sakti",
    "satya", "singa", "surya", "swarga", "warna", "yuga",
    "bisa", "basa", "ratu", "putra", "putri", "duta", "kuda",
    "gaja", "singa", "naga", "garuda",
}

# Known Arabic loanwords common in Austronesian languages
ARABIC_FORMS = {
    "akhir", "akal", "alam", "amal", "badan", "dunia", "halal",
    "haram", "hakim", "hewan", "hikayat", "ilmu", "iman", "insaf",
    "jawab", "kabar", "kalbu", "kitab", "kursi", "lahir", "masjid",
    "nafsu", "niat", "roh", "sabun", "salam", "selamat", "sultan",
    "tarikh", "umur", "waktu", "wajib", "zaman",
    "kubur", "doa", "nikah", "rizki", "tobat", "iblis", "malaikat",
}

# Common Malay trade vocabulary (non-Sanskrit, non-Arabic)
# that spread as trade lingua franca
MALAY_TRADE = {
    "pasar", "harga", "jual", "beli", "uang", "dagang",
    "kapal", "layar", "perahu", "sampan",
}


def load_languages():
    """Load language metadata."""
    langs = {}
    with open(ABVD / "languages.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            langs[row["ID"]] = row
    return langs


def load_forms():
    """Load all lexical forms."""
    forms = defaultdict(list)
    with open(ABVD / "forms.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            forms[row["Language_ID"]].append(row)
    return forms


def load_parameters():
    """Load concept/parameter list (Swadesh-210)."""
    params = {}
    with open(ABVD / "parameters.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            params[row["ID"]] = row
    return params


def load_cognates():
    """Load cognate set assignments."""
    cognates = {}
    with open(ABVD / "cognates.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cognates[row["Form_ID"]] = row
    return cognates


def is_likely_loanword(form_value, loan_set):
    """Check if a form matches any known loanword pattern."""
    form_lower = form_value.lower().strip()
    for loan in loan_set:
        if loan in form_lower or form_lower.startswith(loan[:3]):
            return True
    return False


def check_loan_field(form_row):
    """Check if ABVD already marks this as a loan."""
    loan = form_row.get("Loan", "").strip()
    return loan != "" and loan.lower() not in ("false", "0", "")


def analyze_language(lang_id, lang_name, forms, cognates, params):
    """Run subtraction analysis on one language."""
    lang_forms = forms.get(lang_id, [])
    if not lang_forms:
        print(f"  WARNING: No forms found for {lang_name} (ID={lang_id})")
        return None

    results = []
    n_total = 0
    n_has_cognacy = 0
    n_marked_loan = 0
    n_sanskrit = 0
    n_arabic = 0
    n_malay = 0
    n_residual = 0

    for form in lang_forms:
        n_total += 1
        form_id = form["ID"]
        value = form.get("Value", "") or form.get("Form", "")
        param_id = form.get("Parameter_ID", "")
        cognacy = form.get("Cognacy", "")

        # Get concept name
        concept = params.get(param_id, {}).get("Name", param_id)

        # Classification
        tags = []

        # 1. Check if ABVD marks it as loan
        if check_loan_field(form):
            tags.append("ABVD_LOAN")
            n_marked_loan += 1

        # 2. Check cognacy (if has cognate set = likely Austronesian)
        if cognacy and cognacy.strip():
            tags.append("HAS_COGNACY")
            n_has_cognacy += 1

        # 3. Check Sanskrit
        if is_likely_loanword(value, SANSKRIT_FORMS):
            tags.append("SANSKRIT?")
            n_sanskrit += 1

        # 4. Check Arabic
        if is_likely_loanword(value, ARABIC_FORMS):
            tags.append("ARABIC?")
            n_arabic += 1

        # 5. Check Malay trade
        if is_likely_loanword(value, MALAY_TRADE):
            tags.append("MALAY_TRADE?")
            n_malay += 1

        # Residual: no cognacy AND no loan tags
        is_residual = (not cognacy or not cognacy.strip()) and \
                      not check_loan_field(form) and \
                      "SANSKRIT?" not in tags and \
                      "ARABIC?" not in tags and \
                      "MALAY_TRADE?" not in tags

        if is_residual:
            n_residual += 1
            tags.append("RESIDUAL")

        results.append({
            "form_id": form_id,
            "concept": concept,
            "form": value,
            "cognacy": cognacy,
            "tags": "|".join(tags) if tags else "UNTAGGED",
            "is_residual": is_residual,
        })

    # Summary
    summary = {
        "language": lang_name,
        "lang_id": lang_id,
        "n_total": n_total,
        "n_has_cognacy": n_has_cognacy,
        "n_marked_loan": n_marked_loan,
        "n_sanskrit": n_sanskrit,
        "n_arabic": n_arabic,
        "n_malay": n_malay,
        "n_residual": n_residual,
        "pct_residual": round(100 * n_residual / n_total, 1) if n_total > 0 else 0,
    }

    return summary, results


def cross_language_intersection(all_results):
    """Find concepts that are residual across multiple languages."""
    # concept -> set of languages where it is residual
    concept_residuals = defaultdict(set)

    for lang_name, results in all_results.items():
        for r in results:
            if r["is_residual"]:
                concept_residuals[r["concept"]].add(lang_name)

    # Sort by number of languages
    ranked = sorted(concept_residuals.items(), key=lambda x: -len(x[1]))
    return ranked


def main():
    print("=" * 60)
    print("E022: Linguistic Subtraction POC")
    print("6 Sulawesi Languages — ABVD Basic Vocabulary")
    print("=" * 60)

    print("\nLoading ABVD data...")
    langs = load_languages()
    forms = load_forms()
    params = load_parameters()
    cognates = load_cognates()

    print(f"  Languages: {len(langs)}")
    print(f"  Forms: {sum(len(v) for v in forms.values()):,}")
    print(f"  Concepts: {len(params)}")
    print(f"  Cognate entries: {len(cognates):,}")

    # Verify target languages exist
    print("\nTarget languages:")
    for lid, lname in TARGET_LANGS.items():
        found = langs.get(lid)
        n_forms = len(forms.get(lid, []))
        if found:
            print(f"  [{lid}] {found['Name']} — {n_forms} forms")
        else:
            # Try partial match
            print(f"  [{lid}] {lname} — NOT FOUND by ID, searching...")
            for k, v in langs.items():
                if lname.lower() in v["Name"].lower():
                    print(f"       → Found: [{k}] {v['Name']}")

    # Also find Bugis
    bugis_candidates = [(k, v) for k, v in langs.items()
                        if "bugis" in v["Name"].lower() or "bugi" in v["Name"].lower()]
    print(f"\n  Bugis candidates: {[(k, v['Name']) for k, v in bugis_candidates[:5]]}")

    # Run analysis for each target language
    print("\n" + "=" * 60)
    print("SUBTRACTION ANALYSIS")
    print("=" * 60)

    summaries = []
    all_results = {}

    for lid, lname in TARGET_LANGS.items():
        if lid not in forms:
            # Try to find by name match
            for k, v in langs.items():
                if lname.lower().replace("-", "").replace(" ", "") in \
                   v["Name"].lower().replace("-", "").replace(" ", ""):
                    lid = k
                    break

        print(f"\n--- {lname} (ID={lid}) ---")
        result = analyze_language(lid, lname, forms, cognates, params)
        if result:
            summary, details = result
            summaries.append(summary)
            all_results[lname] = details

            print(f"  Total forms: {summary['n_total']}")
            print(f"  Has cognacy code: {summary['n_has_cognacy']} "
                  f"({100*summary['n_has_cognacy']/max(summary['n_total'],1):.0f}%)")
            print(f"  Marked as loan (ABVD): {summary['n_marked_loan']}")
            print(f"  Sanskrit-like: {summary['n_sanskrit']}")
            print(f"  Arabic-like: {summary['n_arabic']}")
            print(f"  Malay trade: {summary['n_malay']}")
            print(f"  ** RESIDUAL: {summary['n_residual']} "
                  f"({summary['pct_residual']}%) **")

    # Cross-language intersection
    print("\n" + "=" * 60)
    print("CROSS-LANGUAGE RESIDUAL INTERSECTION")
    print("=" * 60)

    ranked = cross_language_intersection(all_results)
    tier1 = [(c, ls) for c, ls in ranked if len(ls) >= 5]
    tier2 = [(c, ls) for c, ls in ranked if len(ls) == 4]
    tier3 = [(c, ls) for c, ls in ranked if len(ls) == 3]

    print(f"\nTier 1 (5-6 languages): {len(tier1)} concepts")
    for concept, lang_set in tier1[:20]:
        print(f"  {concept} — in {len(lang_set)} langs: {', '.join(sorted(lang_set))}")

    print(f"\nTier 2 (4 languages): {len(tier2)} concepts")
    for concept, lang_set in tier2[:15]:
        print(f"  {concept} — {', '.join(sorted(lang_set))}")

    print(f"\nTier 3 (3 languages): {len(tier3)} concepts")
    for concept, lang_set in tier3[:10]:
        print(f"  {concept} — {', '.join(sorted(lang_set))}")

    # Save detailed results
    out_csv = OUT / "poc_subtraction_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if summaries:
            writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
    print(f"\nSaved: {out_csv}")

    # Save residual details for all languages
    out_residuals = OUT / "poc_residuals_detail.csv"
    with open(out_residuals, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "concept", "form", "cognacy", "tags"])
        writer.writeheader()
        for lang_name, details in all_results.items():
            for r in details:
                if r["is_residual"]:
                    writer.writerow({
                        "language": lang_name,
                        "concept": r["concept"],
                        "form": r["form"],
                        "cognacy": r["cognacy"],
                        "tags": r["tags"],
                    })
    print(f"Saved: {out_residuals}")

    # Save cross-language intersection
    out_intersection = OUT / "poc_cross_language_intersection.csv"
    with open(out_intersection, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["concept", "n_languages", "languages"])
        writer.writeheader()
        for concept, lang_set in ranked:
            writer.writerow({
                "concept": concept,
                "n_languages": len(lang_set),
                "languages": "|".join(sorted(lang_set)),
            })
    print(f"Saved: {out_intersection}")

    # GO/NO-GO assessment
    print("\n" + "=" * 60)
    print("GO/NO-GO ASSESSMENT")
    print("=" * 60)
    avg_residual = sum(s["pct_residual"] for s in summaries) / len(summaries) if summaries else 0
    print(f"\nAverage residual across {len(summaries)} languages: {avg_residual:.1f}%")
    print(f"Tier 1 concepts (5+ languages): {len(tier1)}")

    if avg_residual > 10 and len(tier1) > 5:
        print("\n>>> VERDICT: GO — Substantial residual detected. Proceed to full pipeline.")
    elif avg_residual > 5:
        print("\n>>> VERDICT: CONDITIONAL GO — Moderate residual. Consider expanding beyond Swadesh-210.")
    else:
        print("\n>>> VERDICT: NEEDS REVIEW — Low residual. Swadesh core vocabulary may be too conservative.")
        print("    Next step: expand to extended vocabulary (van den Berg 1996 dictionary).")


if __name__ == "__main__":
    main()
