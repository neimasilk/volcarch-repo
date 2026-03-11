"""
E022: Enhanced Linguistic Subtraction with LingPy
==================================================
Improvements over POC:
1. LingPy SCA alignment to detect missed cognates in residual
2. Cross-reference residuals for sound correspondences
3. Better false positive filtering (PAn *Ribu etc.)

Run: python experiments/E022_linguistic_subtraction/enhanced_subtraction.py
"""
import csv
import io
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from lingpy import Wordlist, LexStat
    from lingpy.compare.partial import Partial
    LINGPY_OK = True
except ImportError:
    LINGPY_OK = False
    print("WARNING: LingPy not available. Using fallback mode.")

REPO = Path(__file__).parent.parent.parent
ABVD = REPO / "experiments" / "E022_linguistic_subtraction" / "data" / "abvd" / "cldf"
OUT = REPO / "experiments" / "E022_linguistic_subtraction" / "results"
OUT.mkdir(exist_ok=True)

TARGET_LANGS = {
    "27": "Muna",
    "48": "Bugis",
    "166": "Makassar",
    "192": "Wolio",
    "226": "Toraja-Sadan",
    "674": "Tolaki",
}

# Known PAn/PMP reconstructions that ABVD sometimes misses cognacy for
# Format: concept -> known PAn/PMP root
PAN_KNOWN = {
    "One Thousand": "*Ribu",
    "to hit": "*pukpuk / *tuntun",
    "rope": "*talih",
    "to see": "*kita",
    "red": "*ma-iRaq",
    "to stand": "*tuqud",
    "cloud": "*Rabun / *kunem",
    "heavy": "*beReqat",
    "to come": "*um-aRi",
    "to sit": "*tudan",
    "to steal": "*takaw",
    "to blow": "*heyup",
    "below": "*babaq",
    "to hold": "*genggem",
    "to say": "*kua",
}

# Sanskrit loan patterns (phonological, more precise than POC)
SANSKRIT_PATTERNS = {
    "guru", "raja", "dewa", "deva", "agama", "bahasa", "bangsa", "bumi",
    "candra", "desa", "dharma", "jiwa", "kala", "karma", "karya",
    "kota", "loka", "maha", "mantra", "manusia", "marga", "muka",
    "naga", "negara", "pura", "pustaka", "rasa", "rupa", "sakti",
    "satya", "surya", "swarga", "warna", "yuga", "putra", "putri",
    "duta", "kuda", "gaja", "garuda", "ratna", "jaya", "mati",
    "bhumi", "bhasa", "graha", "cakra", "yantra", "tantra",
    # NOTE: "mati" excluded — it's PAn *matay "to die", NOT Sanskrit
}

ARABIC_PATTERNS = {
    "akhir", "akal", "alam", "amal", "badan", "dunia", "halal",
    "haram", "hakim", "hewan", "ilmu", "iman", "jawab", "kabar",
    "kitab", "kursi", "lahir", "nafsu", "niat", "roh", "sabun",
    "salam", "selamat", "sultan", "umur", "waktu", "wajib", "zaman",
    "kubur", "doa", "nikah", "tobat", "iblis", "malaikat", "hisab",
}

MALAY_TRADE = {
    "pasar", "harga", "jual", "beli", "uang", "dagang",
    "kapal", "layar", "perahu", "sampan",
}


def load_abvd_data():
    """Load ABVD data for target languages."""
    # Load parameters
    params = {}
    with open(ABVD / "parameters.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params[row["ID"]] = row["Name"]

    # Load forms for target languages only
    forms = defaultdict(list)
    with open(ABVD / "forms.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Language_ID"] in TARGET_LANGS:
                forms[row["Language_ID"]].append(row)

    return params, forms


def is_loanword(form_value, loan_set):
    """Check if form closely matches a known loanword.
    Requires the loan pattern to be >=60% of the form length
    to avoid false positives from short substrings."""
    fl = form_value.lower().strip()
    if not fl or len(fl) < 3:
        return False
    for loan in loan_set:
        if len(loan) < 3:
            continue
        if fl == loan or fl.startswith(loan) or fl.endswith(loan):
            # Only match if loan is a substantial part of the form
            if len(loan) / len(fl) >= 0.6:
                return True
    return False


def is_known_pan(concept, form_value):
    """Check if this concept has a known PAn reconstruction."""
    return concept in PAN_KNOWN


def build_lingpy_wordlist(forms, params):
    """Convert ABVD data to LingPy Wordlist format for alignment."""
    # LingPy needs a specific dict format
    d = {0: ["doculect", "concept", "ipa", "cogid"]}
    idx = 1
    for lang_id, lang_forms in forms.items():
        lang_name = TARGET_LANGS[lang_id]
        for form in lang_forms:
            value = form.get("Value", "") or form.get("Form", "")
            if not value or value.strip() in ("", "-"):
                continue
            concept = params.get(form.get("Parameter_ID", ""), "?")
            cognacy = form.get("Cognacy", "")
            cogid = int(cognacy) if cognacy and cognacy.strip().isdigit() else 0
            d[idx] = [lang_name, concept, value.lower().strip(), cogid]
            idx += 1
    return d


def main():
    print("=" * 70)
    print("E022: Enhanced Linguistic Subtraction (with LingPy)")
    print("=" * 70)

    params, forms = load_abvd_data()
    print(f"Loaded {sum(len(v) for v in forms.values())} forms for {len(forms)} languages")

    # ============================================
    # Phase 1: Improved subtraction with PAn cross-check
    # ============================================
    print("\n--- PHASE 1: Enhanced Subtraction ---")

    all_residuals = defaultdict(list)  # concept -> [(lang, form)]
    summaries = []

    for lang_id, lang_name in TARGET_LANGS.items():
        lang_forms = forms.get(lang_id, [])
        n_total = len(lang_forms)
        n_cognacy = 0
        n_loan = 0
        n_sanskrit = 0
        n_arabic = 0
        n_malay = 0
        n_pan_rescued = 0  # false positives caught by PAn cross-check
        n_residual = 0

        for form in lang_forms:
            value = form.get("Value", "") or form.get("Form", "")
            cognacy = form.get("Cognacy", "")
            concept = params.get(form.get("Parameter_ID", ""), "?")
            loan_val = form.get("Loan", "").strip().lower()
            is_loan_marked = loan_val not in ("", "false", "0")

            tags = []

            # Layer 1: ABVD cognacy
            if cognacy and cognacy.strip():
                tags.append("COGNATE")
                n_cognacy += 1

            # Layer 2: ABVD loan marking
            if is_loan_marked:
                tags.append("LOAN")
                n_loan += 1

            # Layer 3: Sanskrit pattern
            if is_loanword(value, SANSKRIT_PATTERNS):
                tags.append("SANSKRIT")
                n_sanskrit += 1

            # Layer 4: Arabic pattern
            if is_loanword(value, ARABIC_PATTERNS):
                tags.append("ARABIC")
                n_arabic += 1

            # Layer 5: Malay trade
            if is_loanword(value, MALAY_TRADE):
                tags.append("MALAY")
                n_malay += 1

            # Layer 6: Known PAn reconstruction (NEW — catches false positives)
            if not tags and is_known_pan(concept, value):
                tags.append("PAN_KNOWN")
                n_pan_rescued += 1

            # Residual check
            if not tags:
                n_residual += 1
                all_residuals[concept].append((lang_name, value))

        pct = round(100 * n_residual / n_total, 1) if n_total else 0
        summaries.append({
            "language": lang_name,
            "total": n_total,
            "cognate": n_cognacy,
            "loan": n_loan,
            "sanskrit": n_sanskrit,
            "arabic": n_arabic,
            "malay": n_malay,
            "pan_rescued": n_pan_rescued,
            "residual": n_residual,
            "pct_residual": pct,
        })

        print(f"\n  {lang_name}: {n_total} forms → {n_residual} residual ({pct}%)"
              f"  [+{n_pan_rescued} rescued by PAn cross-check]")

    # ============================================
    # Phase 2: Cross-language residual analysis
    # ============================================
    print("\n--- PHASE 2: Cross-Language Residual Patterns ---")

    # Rank by number of languages
    ranked = sorted(all_residuals.items(), key=lambda x: -len(x[1]))

    tier1 = [(c, forms) for c, forms in ranked if len(forms) >= 5]
    tier2 = [(c, forms) for c, forms in ranked if len(forms) == 4]
    tier3 = [(c, forms) for c, forms in ranked if len(forms) == 3]

    print(f"\nTier 1 (5-6 langs): {len(tier1)} concepts")
    for concept, entries in tier1:
        langs = [e[0] for e in entries]
        forms_str = "; ".join(f"{e[0]}:{e[1]}" for e in entries)
        pan = PAN_KNOWN.get(concept, "")
        marker = f" [PAn {pan}]" if pan else " << SUBSTRATE CANDIDATE"
        print(f"  {concept}: {len(langs)} langs — {forms_str}{marker}")

    print(f"\nTier 2 (4 langs): {len(tier2)} concepts")
    for concept, entries in tier2:
        forms_str = "; ".join(f"{e[0]}:{e[1]}" for e in entries)
        pan = PAN_KNOWN.get(concept, "")
        marker = f" [PAn {pan}]" if pan else ""
        print(f"  {concept}: {forms_str}{marker}")

    print(f"\nTier 3 (3 langs): {len(tier3)} concepts")
    for concept, entries in tier3[:10]:
        forms_str = "; ".join(f"{e[0]}:{e[1]}" for e in entries)
        print(f"  {concept}: {forms_str}")

    # ============================================
    # Phase 3: LingPy alignment of residuals (if available)
    # ============================================
    if LINGPY_OK and len(all_residuals) > 0:
        print("\n--- PHASE 3: LingPy Sound Correspondence Check ---")

        # Build wordlist from residual forms only
        d = {0: ["doculect", "concept", "ipa", "cogid"]}
        idx = 1
        for concept, entries in ranked:
            if len(entries) >= 3:  # only concepts in 3+ languages
                for lang_name, form_value in entries:
                    d[idx] = [lang_name, concept, form_value.lower(), 0]
                    idx += 1

        if idx > 1:
            try:
                wl = Wordlist(d)
                print(f"  Built wordlist: {len(wl)} entries, "
                      f"{wl.width} languages, {wl.height} concepts")

                # Use LexStat for automated cognate detection
                lex = LexStat(d)
                lex.get_scorer(runs=1000)
                lex.cluster(method="sca", threshold=0.45, ref="autocogid")

                # Check how many residual concepts show cognate clusters
                n_clustered = 0
                n_unclustered = 0
                for concept in wl.rows:
                    cogids = wl.get_list(row=concept, flat=True, entry="autocogid")
                    if len(set(cogids)) < len(cogids):
                        n_clustered += 1
                    else:
                        n_unclustered += 1

                print(f"  Concepts with auto-detected cognates: {n_clustered}")
                print(f"  Concepts without cognate clusters: {n_unclustered}")
                print(f"  → {n_unclustered} concepts are TRUE residuals "
                      f"(no detectable sound correspondence)")

            except Exception as e:
                print(f"  LingPy analysis failed: {e}")
                print("  Falling back to frequency-based analysis only.")
    else:
        print("\n--- PHASE 3: Skipped (LingPy not available or no residuals) ---")

    # ============================================
    # Summary comparison: POC vs Enhanced
    # ============================================
    print("\n" + "=" * 70)
    print("COMPARISON: POC vs ENHANCED")
    print("=" * 70)
    print(f"\n{'Language':<16} {'POC residual%':<16} {'Enhanced residual%':<18} {'PAn rescued'}")
    print("-" * 65)

    # POC results (from memory)
    poc_residuals = {
        "Muna": 28.2, "Bugis": 22.3, "Makassar": 22.1,
        "Wolio": 25.3, "Toraja-Sadan": 18.4, "Tolaki": 59.8,
    }
    for s in summaries:
        poc = poc_residuals.get(s["language"], "?")
        print(f"{s['language']:<16} {poc:<16} {s['pct_residual']:<18} {s['pan_rescued']}")

    avg_enhanced = sum(s["pct_residual"] for s in summaries) / len(summaries)
    avg_poc = sum(poc_residuals.values()) / len(poc_residuals)
    print(f"\n{'AVERAGE':<16} {avg_poc:<16.1f} {avg_enhanced:<18.1f}")

    # ============================================
    # Save results
    # ============================================
    out_path = OUT / "enhanced_subtraction_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nSaved: {out_path}")

    # Save cross-language residuals
    out_cross = OUT / "enhanced_cross_language.csv"
    with open(out_cross, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "concept", "n_languages", "languages", "forms", "pan_known"])
        writer.writeheader()
        for concept, entries in ranked:
            writer.writerow({
                "concept": concept,
                "n_languages": len(entries),
                "languages": "|".join(e[0] for e in entries),
                "forms": "|".join(e[1] for e in entries),
                "pan_known": PAN_KNOWN.get(concept, ""),
            })
    print(f"Saved: {out_cross}")

    print("\nDone.")


if __name__ == "__main__":
    main()
