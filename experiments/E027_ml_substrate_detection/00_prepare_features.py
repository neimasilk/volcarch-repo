"""
E027 Script 00: Prepare Feature Matrix
=======================================
Loads ABVD CLDF data for 6 Sulawesi languages, computes 18 features,
assigns labels from E022 residual classification.

Output: data/features_matrix.csv
"""
import csv
import io
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).parent.parent.parent
ABVD = REPO / "experiments" / "E022_linguistic_subtraction" / "data" / "abvd" / "cldf"
E022_RESIDUALS = REPO / "experiments" / "E022_linguistic_subtraction" / "results" / "poc_residuals_detail.csv"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# Same 6 target languages as E022
TARGET_LANGS = {
    "27": "Muna",
    "48": "Bugis",
    "166": "Makassar",
    "192": "Wolio",
    "226": "Toraja-Sadan",
    "674": "Tolaki",
}

# Language cognacy coverage rates (from E022 enhanced results)
LANG_COVERAGE = {
    "Muna": 1.0 - 0.282,
    "Bugis": 1.0 - 0.223,
    "Makassar": 1.0 - 0.221,
    "Wolio": 1.0 - 0.253,
    "Toraja-Sadan": 1.0 - 0.184,
    "Tolaki": 1.0 - 0.598,
}

# Swadesh 100 core vocabulary concepts (standard list)
SWADESH_100 = {
    "hand", "leg/foot", "to walk", "road/path", "to come", "to turn", "to swim",
    "skin", "back", "belly", "bone", "intestines", "liver", "breast", "shoulder",
    "blood", "head", "neck", "hair", "nose", "mouth", "tooth", "tongue",
    "to laugh", "to cry", "to vomit", "to eat", "to drink", "to bite",
    "to see", "to hear", "to sleep", "to lie down", "to sit", "to stand",
    "person/human being", "man/male", "woman/female", "child", "husband", "wife",
    "mother", "father", "house", "name", "to say", "rope", "to sew", "needle",
    "to hunt", "to hit", "to steal", "to kill", "to die, be dead", "to live, be alive",
    "to cut, hack", "stick/wood", "to split", "sharp", "dull, blunt",
    "to work", "to plant", "to choose", "to grow", "to swell",
    "to squeeze", "to hold", "to dig", "to buy", "to open, uncover",
    "to pound, beat", "to throw", "to fall", "to fly",
    "dog", "bird", "egg", "feather", "fish", "louse", "mosquito", "rat",
    "meat/flesh", "fat/grease", "tail", "snake", "worm (earthworm)",
    "tree", "leaf", "root", "flower", "fruit", "grass",
    "earth/soil", "stone", "sand", "water", "to flow", "sea", "salt",
    "lake", "woods/forest", "sky", "moon", "star", "cloud", "fog",
    "rain", "thunder", "lightning", "wind", "to blow",
    "warm", "cold", "dry", "wet", "heavy",
    "fire", "to burn", "smoke", "ashes",
    "black", "white", "red", "yellow", "green",
    "small", "big", "short", "long", "thin", "thick", "narrow", "wide",
    "painful, sick", "shy, ashamed", "old", "new", "good", "bad, evil",
    "correct, true", "night", "day", "year",
    "when?", "to hide", "to climb", "at", "in, inside", "above", "below",
    "this", "that", "near", "far", "where?", "I", "thou", "he/she",
    "we (inclusive)", "you", "they", "what?", "who?", "other", "all",
    "and", "if", "how?", "no, not", "to count",
    "One", "Two", "Three", "Four", "Five",
}

# Semantic domain mapping based on concept content
SEMANTIC_DOMAINS = {
    "BODY": {"hand", "leg/foot", "back", "belly", "bone", "intestines", "liver",
             "breast", "shoulder", "blood", "head", "neck", "hair", "nose",
             "mouth", "tooth", "tongue", "skin", "ear", "eye", "feather",
             "wing", "tail", "fat/grease", "meat/flesh", "egg"},
    "NATURE": {"earth/soil", "stone", "sand", "water", "sea", "salt", "lake",
               "woods/forest", "sky", "moon", "star", "cloud", "fog", "rain",
               "thunder", "lightning", "wind", "fire", "smoke", "ashes",
               "tree", "leaf", "root", "flower", "fruit", "grass",
               "road/path", "dust", "dirty"},
    "ACTION": {"to walk", "to come", "to turn", "to swim", "to breathe",
               "to sniff, smell", "to laugh", "to cry", "to vomit", "to spit",
               "to chew", "to eat", "to cook", "to drink", "to bite",
               "to see", "to hear", "to sleep", "to lie down", "to sit",
               "to stand", "to say", "to sew", "to hunt", "to hit",
               "to steal", "to kill", "to die, be dead", "to live, be alive",
               "to cut, hack", "to split", "to work", "to plant", "to choose",
               "to grow", "to swell", "to squeeze", "to hold", "to dig",
               "to buy", "to open, uncover", "to pound, beat", "to throw",
               "to fall", "to fly", "to flow", "to blow", "to burn",
               "to hide", "to climb", "to count", "to think", "to fear",
               "to yawn", "to dream", "to scratch", "to stab, pierce",
               "to tie up, fasten", "to know, be knowledgeable", "to turn"},
    "QUALITY": {"warm", "cold", "dry", "wet", "heavy", "black", "white",
                "red", "yellow", "green", "small", "big", "short", "long",
                "thin", "thick", "narrow", "wide", "painful, sick",
                "shy, ashamed", "old", "new", "good", "bad, evil",
                "correct, true", "sharp", "dull, blunt", "rotten", "dirty"},
    "NUMBER": {"One", "Two", "Three", "Four", "Five", "Six", "Seven",
               "Eight", "Nine", "Ten", "Twenty", "Fifty", "One Hundred",
               "One Thousand"},
    "GRAMMAR": {"this", "that", "near", "far", "where?", "I", "thou",
                "he/she", "we (inclusive)", "you", "they", "what?", "who?",
                "other", "all", "and", "if", "how?", "no, not", "when?",
                "in, inside", "above", "below", "at"},
}


def classify_domain(concept):
    """Assign semantic domain to a concept."""
    for domain, concepts in SEMANTIC_DOMAINS.items():
        if concept in concepts:
            return domain
    return "OTHER"


def is_core_vocab(concept):
    """Check if concept is in Swadesh-100."""
    return 1 if concept in SWADESH_100 else 0


# ============================================================
# Phonological feature extractors
# ============================================================

VOWELS = set("aeiouəɛɨɔæøüöäåãẽĩõũâêîôûàèìòùáéíóú")
AUSTRONESIAN_PREFIXES = ("ma-", "me-", "mo-", "pa-", "ka-", "ta-", "na-", "po-",
                         "ma", "me", "mo", "pa", "ka", "ta", "na", "po",
                         "maŋ", "meŋ", "moŋ", "paŋ", "aŋ", "mak-", "mat-")
NASAL_CLUSTERS = ("ng", "mb", "nd", "nj", "mp", "nk", "ŋg", "ŋk",
                  "nc", "nt", "ŋ")  # ŋ alone also counts as nasal


def compute_form_length(form):
    """Character count of cleaned form."""
    return len(form)


def count_vowels(form):
    """Count vowels."""
    return sum(1 for c in form.lower() if c in VOWELS)


def vowel_ratio(form):
    """Ratio of vowels to total characters."""
    if len(form) == 0:
        return 0.0
    return round(count_vowels(form) / len(form), 4)


def ends_in_vowel(form):
    """Binary: does form end in a vowel?"""
    if not form:
        return 0
    return 1 if form[-1].lower() in VOWELS else 0


def get_initial_class(form):
    """Classify initial character into canonical classes."""
    if not form:
        return "other"
    c = form[0].lower()
    if c in ('m', 'a', 'b', 't', 'k', 'p', 's'):
        return c
    return "other"


def has_glottal(form):
    """Binary: contains glottal stop marker."""
    return 1 if ("ʔ" in form or "'" in form) else 0


def has_nasal_cluster(form):
    """Binary: contains nasal cluster."""
    fl = form.lower()
    for nc in NASAL_CLUSTERS:
        if nc in fl:
            return 1
    return 0


def has_reduplication(form):
    """Binary: contains reduplication marker or repeated syllable pattern."""
    if "-" in form:
        return 1
    # Check for repeated 2-3 char patterns
    fl = form.lower()
    for plen in (2, 3):
        for i in range(len(fl) - plen * 2 + 1):
            chunk = fl[i:i+plen]
            if chunk == fl[i+plen:i+plen*2]:
                return 1
    return 0


def count_consonant_clusters(form):
    """Count CC+ sequences (consecutive consonants)."""
    fl = form.lower()
    count = 0
    in_cluster = False
    consec = 0
    for c in fl:
        if c not in VOWELS and c.isalpha():
            consec += 1
            if consec == 2 and not in_cluster:
                count += 1
                in_cluster = True
        else:
            consec = 0
            in_cluster = False
    return count


def has_prefix_like(form):
    """Binary: starts with Austronesian-like prefix."""
    fl = form.lower()
    for prefix in AUSTRONESIAN_PREFIXES:
        if fl.startswith(prefix):
            return 1
    return 0


def clean_form(raw):
    """Clean form string: remove brackets, trim whitespace."""
    s = raw.strip()
    # Remove bracketed content like [kayu], [maŋa]
    s = re.sub(r'\[.*?\]\s*', '', s)
    # Remove leading/trailing punctuation
    s = s.strip(" -,;.")
    return s


def main():
    print("=" * 70)
    print("E027 Script 00: Prepare Feature Matrix")
    print("=" * 70)

    # ============================================================
    # Step 1: Load E022 residual labels
    # ============================================================
    print("\n[1/5] Loading E022 residual labels...")
    residual_set = set()
    with open(E022_RESIDUALS, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["language"].strip()
            concept = row["concept"].strip()
            form = row["form"].strip()
            residual_set.add((lang, concept, form))
    print(f"  Loaded {len(residual_set)} residual entries from E022")

    # ============================================================
    # Step 2: Load ABVD data
    # ============================================================
    print("\n[2/5] Loading ABVD CLDF data...")

    # Load parameters (concepts)
    params = {}
    with open(ABVD / "parameters.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params[row["ID"]] = row["Name"]

    # Load cognate set info
    print("  Loading cognate sets...")
    cognate_info = defaultdict(list)  # form_id -> [cognateset_id, ...]
    cognate_set_sizes = defaultdict(int)  # cognateset_id -> count
    with open(ABVD / "cognates.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            form_id = row["Form_ID"]
            cog_id = row["Cognateset_ID"]
            cognate_info[form_id].append(cog_id)
            cognate_set_sizes[cog_id] += 1

    print(f"  {len(cognate_set_sizes)} unique cognate sets")

    # Load forms
    print("  Loading forms for target languages...")
    forms = []
    with open(ABVD / "forms.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Language_ID"] in TARGET_LANGS:
                forms.append(row)
    print(f"  {len(forms)} forms loaded")

    # ============================================================
    # Step 3: Assign labels
    # ============================================================
    print("\n[3/5] Assigning labels...")

    # Map E022 language names to ABVD Language_IDs
    # E022 uses: Muna, Bugis, Makassar, Wolio, Toraja-Sadan, Tolaki
    lang_name_map = {v: k for k, v in TARGET_LANGS.items()}

    n_cognate = 0
    n_residual = 0
    n_ambiguous = 0

    labeled_forms = []
    for row in forms:
        lang_id = row["Language_ID"]
        lang_name = TARGET_LANGS[lang_id]
        form_id = row["ID"]
        concept = params.get(row["Parameter_ID"], "?")
        raw_value = row.get("Value", "") or row.get("Form", "")
        form_value = clean_form(raw_value)

        if not form_value or len(form_value) < 1:
            continue

        # Check if this form is in E022 residuals
        # E022 uses the raw form, try matching
        is_residual = False
        # Try exact match first
        if (lang_name, concept, raw_value.strip()) in residual_set:
            is_residual = True
        elif (lang_name, concept, form_value) in residual_set:
            is_residual = True
        else:
            # Check cognacy field — if it has cognacy, it's Austronesian
            cognacy = row.get("Cognacy", "").strip()
            if cognacy:
                is_residual = False
            else:
                # No cognacy AND not in residual set — might be missing
                # Use cognate_info as backup
                cog_sets = cognate_info.get(form_id, [])
                if cog_sets:
                    is_residual = False
                else:
                    # No cognacy anywhere — treat as residual
                    is_residual = True

        label = 0 if is_residual else 1  # 1 = Austronesian, 0 = Candidate substrate

        if is_residual:
            n_residual += 1
        else:
            n_cognate += 1

        # Get cognate set info for distributional features
        cog_sets = cognate_info.get(form_id, [])
        max_cog_size = max((cognate_set_sizes[cs] for cs in cog_sets), default=0)
        n_cog_sets = len(cog_sets)

        labeled_forms.append({
            "form_id": form_id,
            "language": lang_name,
            "language_id": lang_id,
            "concept": concept,
            "form": form_value,
            "raw_form": raw_value.strip(),
            "label": label,
            "max_cognate_set_size": max_cog_size,
            "n_cognate_sets": n_cog_sets,
        })

    print(f"  Austronesian (label=1): {n_cognate}")
    print(f"  Candidate substrate (label=0): {n_residual}")
    print(f"  Total: {len(labeled_forms)}")

    # ============================================================
    # Step 4: Compute concept-level distributional features
    # ============================================================
    print("\n[4/5] Computing concept-level distributional features...")

    # Concept residual rate: fraction of languages where this concept is residual
    concept_residual = defaultdict(lambda: {"total": 0, "residual": 0})
    for lf in labeled_forms:
        concept_residual[lf["concept"]]["total"] += 1
        if lf["label"] == 0:
            concept_residual[lf["concept"]]["residual"] += 1

    # Cross-language residual count per concept
    concept_cross_lang = defaultdict(set)
    for lf in labeled_forms:
        if lf["label"] == 0:
            concept_cross_lang[lf["concept"]].add(lf["language"])

    # ============================================================
    # Step 5: Build feature matrix
    # ============================================================
    print("\n[5/5] Building feature matrix...")

    # Language label encoding
    lang_encode = {name: i for i, name in enumerate(sorted(TARGET_LANGS.values()))}

    feature_rows = []
    for lf in labeled_forms:
        form = lf["form"]
        concept = lf["concept"]
        lang = lf["language"]

        # Phonological features (10)
        fl = compute_form_length(form)
        nv = count_vowels(form)
        vr = vowel_ratio(form)
        eiv = ends_in_vowel(form)
        ic = get_initial_class(form)
        hg = has_glottal(form)
        hnc = has_nasal_cluster(form)
        hr = has_reduplication(form)
        ncc = count_consonant_clusters(form)
        hpl = has_prefix_like(form)

        # Distributional features (4) — Model A only
        mcs = lf["max_cognate_set_size"]
        ncs = lf["n_cognate_sets"]
        crr = concept_residual[concept]["residual"] / max(concept_residual[concept]["total"], 1)
        cclc = len(concept_cross_lang.get(concept, set()))

        # Semantic features (2)
        sd = classify_domain(concept)
        icv = is_core_vocab(concept)

        # Language control features (2)
        lid = lang_encode[lang]
        lcov = LANG_COVERAGE.get(lang, 0.5)

        feature_rows.append({
            # Identifiers
            "form_id": lf["form_id"],
            "language": lang,
            "concept": concept,
            "form": form,
            "label": lf["label"],
            # Phonological (10)
            "form_length": fl,
            "n_vowels": nv,
            "vowel_ratio": round(vr, 4),
            "ends_in_vowel": eiv,
            "initial_char": ic,
            "has_glottal": hg,
            "has_nasal_cluster": hnc,
            "has_reduplication": hr,
            "n_consonant_clusters": ncc,
            "has_prefix_like": hpl,
            # Distributional (4) — Model A only
            "max_cognate_set_size": mcs,
            "n_cognate_sets": ncs,
            "concept_residual_rate": round(crr, 4),
            "concept_cross_lang_count": cclc,
            # Semantic (2)
            "semantic_domain": sd,
            "is_core_vocab": icv,
            # Language control (2)
            "language_id_encoded": lid,
            "language_cognacy_coverage": round(lcov, 4),
        })

    # Save
    outpath = OUT / "features_matrix.csv"
    fieldnames = list(feature_rows[0].keys())
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    print(f"\n  Saved: {outpath}")
    print(f"  Shape: {len(feature_rows)} rows × {len(fieldnames)} columns")

    # ============================================================
    # Summary statistics
    # ============================================================
    print("\n" + "=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)

    # Label distribution by language
    print(f"\n{'Language':<18} {'Austronesian':>13} {'Substrate':>10} {'Total':>7} {'%Substr':>8}")
    print("-" * 60)
    lang_stats = defaultdict(lambda: {"cognate": 0, "residual": 0})
    for fr in feature_rows:
        if fr["label"] == 1:
            lang_stats[fr["language"]]["cognate"] += 1
        else:
            lang_stats[fr["language"]]["residual"] += 1

    for lang in sorted(lang_stats.keys()):
        s = lang_stats[lang]
        total = s["cognate"] + s["residual"]
        pct = round(100 * s["residual"] / total, 1)
        print(f"{lang:<18} {s['cognate']:>13} {s['residual']:>10} {total:>7} {pct:>7.1f}%")

    tot_cog = sum(s["cognate"] for s in lang_stats.values())
    tot_res = sum(s["residual"] for s in lang_stats.values())
    tot = tot_cog + tot_res
    pct_tot = round(100 * tot_res / tot, 1)
    print(f"{'TOTAL':<18} {tot_cog:>13} {tot_res:>10} {tot:>7} {pct_tot:>7.1f}%")

    # Semantic domain distribution
    print(f"\n{'Domain':<12} {'Count':>6} {'% Substrate':>12}")
    print("-" * 35)
    domain_stats = defaultdict(lambda: {"total": 0, "residual": 0})
    for fr in feature_rows:
        domain_stats[fr["semantic_domain"]]["total"] += 1
        if fr["label"] == 0:
            domain_stats[fr["semantic_domain"]]["residual"] += 1
    for dom in sorted(domain_stats.keys()):
        ds = domain_stats[dom]
        pct = round(100 * ds["residual"] / ds["total"], 1) if ds["total"] else 0
        print(f"{dom:<12} {ds['total']:>6} {pct:>11.1f}%")

    # Feature value ranges
    print("\nFeature value ranges:")
    numeric_feats = ["form_length", "n_vowels", "vowel_ratio", "ends_in_vowel",
                     "has_glottal", "has_nasal_cluster", "has_reduplication",
                     "n_consonant_clusters", "has_prefix_like",
                     "max_cognate_set_size", "n_cognate_sets",
                     "concept_residual_rate", "concept_cross_lang_count",
                     "is_core_vocab", "language_id_encoded", "language_cognacy_coverage"]
    for feat in numeric_feats:
        vals = [fr[feat] for fr in feature_rows]
        print(f"  {feat:<30} min={min(vals):<8} max={max(vals):<8} mean={sum(vals)/len(vals):.3f}")

    print("\nDone. Feature matrix ready for ML training.")


if __name__ == "__main__":
    main()
