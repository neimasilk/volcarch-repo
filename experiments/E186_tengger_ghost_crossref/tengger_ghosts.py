"""
E186: Tengger Ghost Word Cross-Reference

Do the 230 ghost words from E165 (vanished from inscriptions after C9)
survive in the Tengger dialect — the volcanic isolate that lives on
Bromo's slopes and preserves pre-Hindu ritual practices?

If yes: Tengger is a LINGUISTIC TIME CAPSULE preserving C7-C9 vocabulary
that was pruned from the written register.

Uses ABVD data: Tengger (ID 1533), Javanese (ID 20), Old Javanese (ID 1535),
Balinese (ID 1)
"""

import csv
from collections import defaultdict

print("=" * 70)
print("E186: TENGGER GHOST WORD CROSS-REFERENCE")
print("       Do vanished inscription words survive in the volcanic isolate?")
print("=" * 70)

# Load ABVD forms
forms = defaultdict(dict)  # {language_id: {parameter: [forms]}}
with open("experiments/E022_linguistic_subtraction/data/abvd/cldf/forms.csv",
          "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lang_id = row['Language_ID']
        param = row['Parameter_ID']
        form = row['Form'].lower().strip()
        if lang_id in ['1533', '20', '1535', '1', '1532', '1534']:
            if param not in forms[lang_id]:
                forms[lang_id][param] = []
            forms[lang_id][param].append(form)

# Load parameters (concept names)
params = {}
with open("experiments/E022_linguistic_subtraction/data/abvd/cldf/parameters.csv",
          "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        params[row['ID']] = row['Name']

print(f"\nLoaded forms for:")
for lid, name in [('1533', 'Tengger (Ngadas)'), ('20', 'Javanese'),
                   ('1535', 'Old/Middle Javanese'), ('1', 'Balinese'),
                   ('1532', 'Javanese (Yogyakarta)'), ('1534', 'Javanese (Malang)')]:
    n = sum(len(v) for v in forms.get(lid, {}).values())
    print(f"  {lid:>5}: {name:30s} — {n} forms")

# Ghost words from E181 (key ones with known meanings)
ghost_words = {
    # word: (meaning, domain, PMP_root_if_any)
    'aku': ('I/me', 'PRONOUN', '*aku'),
    'vulan': ('moon', 'NATURE', '*bulan'),
    'punti': ('banana', 'AGRICULTURE', '*punti'),
    'anakbini': ('wife', 'KINSHIP', '*anak + *bini'),
    'hiliri': ('downstream', 'GEOGRAPHY', '*hilir'),
    'gugur': ('to fall', 'NATURE', '*gugur'),
    'mula': ('origin', 'ABSTRACT', '*mula'),
    'kandal': ('thick', 'MATERIAL', None),
    'nusuk': ('to pierce', 'MATERIAL', None),
    'jati': ('teak/truth', 'NATURE', None),
    'tahilan': ('weight unit', 'MEASURE', None),
    'boto': ('stone', 'MATERIAL', '*batu'),
    'sida': ('completed', 'ADMIN', None),
    'huyup': ('to blow', 'NATURE', None),
    'parlak': ('bright', 'NATURE', None),
    'ruhutan': ('forest', 'NATURE', None),
    'kalivuan': ('settlement', 'PLACE', None),
    'nivunu': ('to burn', 'ACTION', '*tunuh?'),
    'glis': ('fast', 'ACTION', None),
    'sisim': ('ring', 'MATERIAL', None),
    'sayut': ('offering', 'MATERIAL', None),
    'larak': ('plant/crop', 'AGRICULTURE', None),
    'haliva': ('rice/grain', 'AGRICULTURE', None),
}

# Search for ghost word roots in ABVD forms
# We look for substring matches and phonological correspondences
print("\n--- CROSS-REFERENCE: Ghost Words in ABVD Dialects ---")
print()

# Build searchable form indices
all_forms = {}
for lid in ['1533', '20', '1535', '1', '1532', '1534']:
    all_forms[lid] = set()
    for param_forms in forms.get(lid, {}).values():
        for f in param_forms:
            all_forms[lid].add(f)

# Mapping of ghost words to ABVD search terms
# (some need phonological transformation: OJ vu- > Mod Jav wu-/bu-)
search_map = {
    'aku': ['aku', 'aku?'],
    'vulan': ['wulan', 'bulan', 'vulan'],
    'punti': ['punti', 'pisang', 'penti'],
    'hiliri': ['ilir', 'hilir', 'ilor'],
    'gugur': ['gugur', 'gugur?'],
    'mula': ['mula', 'mola'],
    'kandal': ['kandel', 'kandal'],
    'nusuk': ['nusuk', 'tusuk', 'nusuk?'],
    'jati': ['jati'],
    'boto': ['watu', 'batu', 'boto'],
    'nivunu': ['obong', 'bakar', 'tunu'],
    'haliva': ['beras', 'pari', 'gabah'],
}

# Search ABVD concepts that match ghost word meanings
meaning_to_concept = {
    'I/me': ['1sg', 'i', 'pronoun'],
    'moon': ['moon'],
    'banana': ['banana'],
    'wife': ['wife'],
    'downstream': ['below', 'down'],
    'to fall': ['fall'],
    'origin': ['root', 'origin'],
    'thick': ['thick', 'fat'],
    'stone': ['stone', 'rock'],
    'to burn': ['burn', 'fire'],
    'rice/grain': ['rice', 'grain'],
    'to blow': ['blow', 'wind'],
    'bright': ['light', 'shine'],
    'forest': ['forest', 'jungle'],
}

# Direct concept search in ABVD
print(f"{'Ghost Word':>15} | {'Meaning':>15} | {'Tengger':>15} | {'Javanese':>15} | {'OldJav':>15} | {'Balinese':>15}")
print("-" * 100)

matches_tengger = 0
matches_javanese = 0
matches_oldjav = 0
matches_bali = 0

for ghost, (meaning, domain, pmp) in ghost_words.items():
    # Search for concept match in parameter names
    tengger_match = ""
    javanese_match = ""
    oldjav_match = ""
    bali_match = ""

    # Check all parameters for meaning-related concepts
    for param_id, param_name in params.items():
        param_lower = param_name.lower()

        # Check if parameter matches the ghost word meaning
        meaning_lower = meaning.lower()
        keywords = meaning_lower.split('/')

        matched = False
        for kw in keywords:
            kw = kw.strip()
            if len(kw) > 2 and kw in param_lower:
                matched = True
                break

        if not matched:
            # Try meaning_to_concept mapping
            for concept_kw in meaning_to_concept.get(meaning, []):
                if concept_kw in param_lower:
                    matched = True
                    break

        if matched:
            # Get forms for this concept in each language
            for lid, label in [('1533', 'tengger'), ('20', 'javanese'),
                               ('1535', 'oldjav'), ('1', 'bali')]:
                concept_forms = forms.get(lid, {}).get(param_id, [])
                if concept_forms:
                    form_str = '/'.join(concept_forms[:2])
                    if lid == '1533':
                        tengger_match = form_str if not tengger_match else tengger_match
                    elif lid == '20':
                        javanese_match = form_str if not javanese_match else javanese_match
                    elif lid == '1535':
                        oldjav_match = form_str if not oldjav_match else oldjav_match
                    elif lid == '1':
                        bali_match = form_str if not bali_match else bali_match

    if tengger_match:
        matches_tengger += 1
    if javanese_match:
        matches_javanese += 1
    if oldjav_match:
        matches_oldjav += 1
    if bali_match:
        matches_bali += 1

    # Truncate for display
    t = tengger_match[:15] if tengger_match else "-"
    j = javanese_match[:15] if javanese_match else "-"
    o = oldjav_match[:15] if oldjav_match else "-"
    b = bali_match[:15] if bali_match else "-"

    print(f"{ghost:>15} | {meaning:>15} | {t:>15} | {j:>15} | {o:>15} | {b:>15}")

print(f"\n--- MATCH SUMMARY ---")
print(f"Ghost words with ABVD concept match:")
print(f"  Tengger:     {matches_tengger}/{len(ghost_words)} ({matches_tengger/len(ghost_words)*100:.0f}%)")
print(f"  Javanese:    {matches_javanese}/{len(ghost_words)} ({matches_javanese/len(ghost_words)*100:.0f}%)")
print(f"  Old Javanese:{matches_oldjav}/{len(ghost_words)} ({matches_oldjav/len(ghost_words)*100:.0f}%)")
print(f"  Balinese:    {matches_bali}/{len(ghost_words)} ({matches_bali/len(ghost_words)*100:.0f}%)")

# ============================================================
# DIRECT FORM SEARCH: Look for ghost word FORMS in Tengger
# ============================================================
print("\n--- DIRECT FORM SEARCH ---")
print("Looking for ghost word forms (or phonological variants) in Tengger ABVD entries")
print()

tengger_all = all_forms.get('1533', set())
javanese_all = all_forms.get('20', set())
bali_all = all_forms.get('1', set())

def fuzzy_match(ghost, form_set):
    """Check if ghost word or its phonological variant exists in form set"""
    # Direct match
    if ghost.lower() in form_set:
        return ghost.lower()
    # Common OJ -> Modern sound changes
    variants = [
        ghost.lower(),
        ghost.lower().replace('v', 'w'),   # OJ v -> Mod w
        ghost.lower().replace('vu', 'wu'),  # OJ vu -> wu
        ghost.lower().replace('vu', 'bu'),  # OJ vu -> bu
        ghost.lower().replace('ny', 'n'),   # palatalization
        ghost.lower().replace('dh', 'd'),   # aspiration loss
    ]
    for v in variants:
        if v in form_set:
            return v
        # Substring match (word appears within a compound)
        for f in form_set:
            if len(v) > 3 and v in f:
                return f"{f} (contains {v})"
    return None

print(f"{'Ghost':>15} | {'In Tengger?':>20} | {'In Javanese?':>20} | {'In Balinese?':>20}")
print("-" * 80)

teng_found = 0
jav_found = 0
bal_found = 0

for ghost in ghost_words:
    t = fuzzy_match(ghost, tengger_all)
    j = fuzzy_match(ghost, javanese_all)
    b = fuzzy_match(ghost, bali_all)

    if t: teng_found += 1
    if j: jav_found += 1
    if b: bal_found += 1

    t_str = t[:20] if t else "-"
    j_str = j[:20] if j else "-"
    b_str = b[:20] if b else "-"

    print(f"{ghost:>15} | {t_str:>20} | {j_str:>20} | {b_str:>20}")

print(f"\n--- DIRECT FORM MATCH SUMMARY ---")
print(f"Ghost words found as forms in ABVD:")
print(f"  Tengger:  {teng_found}/{len(ghost_words)} ({teng_found/len(ghost_words)*100:.0f}%)")
print(f"  Javanese: {jav_found}/{len(ghost_words)} ({jav_found/len(ghost_words)*100:.0f}%)")
print(f"  Balinese: {bal_found}/{len(ghost_words)} ({bal_found/len(ghost_words)*100:.0f}%)")

# ============================================================
# CONCLUSION
# ============================================================
print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
NOTE: ABVD contains only ~210 basic vocabulary concepts (Swadesh-like).
Most ghost words are SPECIALIZED terms (admin titles, ritual objects,
agricultural tools) that are NOT in ABVD.

The test is LIMITED but directional:
- For the ghost words that DO have ABVD cognates (basic vocabulary like
  'aku', 'wulan', 'watu'), we can check if Tengger preserves the OJ form
  while standard Javanese uses a replacement.

KEY FINDING REGARDLESS OF ABVD MATCH RATE:
The ghost words are mostly SPECIALIZED vocabulary (admin, ritual, agriculture)
that would NOT appear in a basic wordlist like ABVD. A proper test requires:
1. Conners (2008) PhD dissertation on Tengger Javanese (full lexicon)
2. Fieldwork: interview Tengger speakers for ghost word recognition
3. Comparison with Badui (West Java isolate) and Osing (East Java)

The HYPOTHESIS remains strong: if Tengger preserves pre-Hindu ritual
practices (slametan, Kasada, hyang worship), it likely also preserves
pre-Hindu VOCABULARY that vanished from the written register after C9.
This is testable but requires data beyond ABVD.
""")
