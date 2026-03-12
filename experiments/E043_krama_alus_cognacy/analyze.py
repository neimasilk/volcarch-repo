"""
E043: Peripheral Conservatism — Cognacy Comparison
===================================================
Compare PMP/PAn cognacy retention rates across peripheral vs central
Javanese varieties. Tests whether Balinese/Tengger preserve more
proto-language cognates than standard Javanese.

Output: results/cognacy_comparison.csv, results/summary.txt
"""
import csv
import io
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).parent.parent.parent
ABVD = REPO / "experiments" / "E022_linguistic_subtraction" / "data" / "abvd" / "cldf"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# Target languages
TARGETS = {
    "1": "Balinese",
    "20": "Javanese",
    "1532": "Javanese_Yogyakarta",
    "1534": "Javanese_Malang",
    "1533": "Tengger_Ngadas",
    "92": "Merina_Malagasy",
    "290": "Old_Javanese",
    "1535": "Old_Middle_Javanese",
}

# Proto-language references
PROTOS = {
    "269": "PMP",
    "280": "PAn",
}

ALL_IDS = {**TARGETS, **PROTOS}

# Semantic domains (from E027)
BODY = {"hand", "leg/foot", "back", "belly", "bone", "intestines", "liver",
        "breast", "shoulder", "blood", "head", "neck", "hair", "nose",
        "mouth", "tooth", "tongue", "skin", "ear", "eye"}
NATURE = {"earth/soil", "stone", "sand", "water", "sea", "salt", "lake",
          "woods/forest", "sky", "moon", "star", "cloud", "rain",
          "wind", "fire", "smoke", "ashes", "tree", "leaf", "root",
          "flower", "fruit", "grass"}
ACTION = {"to walk", "to come", "to swim", "to eat", "to drink", "to bite",
          "to see", "to hear", "to sleep", "to sit", "to stand", "to say",
          "to hit", "to kill", "to die, be dead", "to live, be alive",
          "to cut, hack", "to hold", "to dig", "to fly", "to burn",
          "to fall", "to throw", "to blow"}
NUMBER = {"One", "Two", "Three", "Four", "Five", "Six", "Seven",
          "Eight", "Nine", "Ten"}
KINSHIP = {"person/human being", "man/male", "woman/female", "child",
           "husband", "wife", "mother", "father"}


def get_domain(concept):
    if concept in BODY:
        return "BODY"
    if concept in NATURE:
        return "NATURE"
    if concept in ACTION:
        return "ACTION"
    if concept in NUMBER:
        return "NUMBER"
    if concept in KINSHIP:
        return "KINSHIP"
    return "OTHER"


def main():
    print("=" * 70)
    print("E043: Peripheral Conservatism — Cognacy Comparison")
    print("=" * 70)

    # ============================================================
    # Step 1: Load parameters (concepts)
    # ============================================================
    print("\n[1/5] Loading concepts...")
    params = {}
    with open(ABVD / "parameters.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            params[row["ID"]] = row["Name"]
    print(f"  {len(params)} concepts loaded")

    # ============================================================
    # Step 2: Load cognate sets
    # ============================================================
    print("\n[2/5] Loading cognate sets...")
    # form_id -> set of cognateset_ids
    form_cognatesets = defaultdict(set)
    with open(ABVD / "cognates.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            form_cognatesets[row["Form_ID"]].add(row["Cognateset_ID"])
    print(f"  {len(form_cognatesets)} forms with cognate assignments")

    # ============================================================
    # Step 3: Load forms for target languages + proto-languages
    # ============================================================
    print("\n[3/5] Loading forms for target languages...")
    # Structure: lang_id -> concept -> [(form_id, form_text, cognatesets), ...]
    lang_data = defaultdict(lambda: defaultdict(list))
    n_loaded = 0
    with open(ABVD / "forms.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Language_ID"] in ALL_IDS:
                concept = params.get(row["Parameter_ID"], "")
                form_text = row.get("Form", "").strip()
                if not form_text:
                    form_text = row.get("Value", "").strip()
                cog_sets = form_cognatesets.get(row["ID"], set())
                lang_data[row["Language_ID"]][concept].append({
                    "form_id": row["ID"],
                    "form": form_text,
                    "cognatesets": cog_sets,
                })
                n_loaded += 1
    print(f"  {n_loaded} forms loaded across {len(lang_data)} languages")

    for lid, name in sorted(ALL_IDS.items(), key=lambda x: x[1]):
        n_concepts = len(lang_data.get(lid, {}))
        n_forms = sum(len(v) for v in lang_data.get(lid, {}).values())
        print(f"    {name:<25} {n_concepts:>4} concepts, {n_forms:>5} forms")

    # ============================================================
    # Step 4: Compute cognacy overlap with PMP/PAn
    # ============================================================
    print("\n[4/5] Computing cognacy overlap...")

    # For each target language and each concept:
    # Check if ANY form shares a cognate set with PMP or PAn
    results = []  # per-concept rows
    summary = defaultdict(lambda: {"pmp_match": 0, "pan_match": 0, "either_match": 0, "total": 0})

    # Get all concepts that PMP/PAn have entries for
    pmp_concepts = set(lang_data.get("269", {}).keys())
    pan_concepts = set(lang_data.get("280", {}).keys())
    proto_concepts = pmp_concepts | pan_concepts
    print(f"  PMP covers {len(pmp_concepts)} concepts, PAn covers {len(pan_concepts)} concepts")
    print(f"  Union: {len(proto_concepts)} concepts with proto-language data")

    for lid, lname in sorted(TARGETS.items(), key=lambda x: x[1]):
        for concept in sorted(proto_concepts):
            target_forms = lang_data.get(lid, {}).get(concept, [])
            if not target_forms:
                continue

            # Collect all cognate sets from this language's forms for this concept
            target_cogsets = set()
            for tf in target_forms:
                target_cogsets.update(tf["cognatesets"])

            # Check overlap with PMP
            pmp_cogsets = set()
            for pf in lang_data.get("269", {}).get(concept, []):
                pmp_cogsets.update(pf["cognatesets"])

            # Check overlap with PAn
            pan_cogsets = set()
            for pf in lang_data.get("280", {}).get(concept, []):
                pan_cogsets.update(pf["cognatesets"])

            pmp_match = 1 if target_cogsets & pmp_cogsets else 0
            pan_match = 1 if target_cogsets & pan_cogsets else 0
            either = 1 if (pmp_match or pan_match) else 0

            domain = get_domain(concept)

            results.append({
                "language_id": lid,
                "language": lname,
                "concept": concept,
                "domain": domain,
                "n_forms": len(target_forms),
                "n_cognatesets": len(target_cogsets),
                "pmp_match": pmp_match,
                "pan_match": pan_match,
                "either_match": either,
                "pmp_cogsets_shared": len(target_cogsets & pmp_cogsets) if pmp_cogsets else 0,
            })

            summary[lname]["total"] += 1
            summary[lname]["pmp_match"] += pmp_match
            summary[lname]["pan_match"] += pan_match
            summary[lname]["either_match"] += either

    # Save per-concept results
    outpath = OUT / "cognacy_comparison.csv"
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {outpath} ({len(results)} rows)")

    # ============================================================
    # Step 5: Summary statistics
    # ============================================================
    print("\n[5/5] Summary statistics")
    print("=" * 70)

    output_lines = []

    def p(line=""):
        print(line)
        output_lines.append(line)

    p("E043: PERIPHERAL CONSERVATISM — COGNACY COMPARISON")
    p("=" * 60)
    p()
    p(f"{'Language':<25} {'Total':>6} {'PMP%':>7} {'PAn%':>7} {'Either%':>8}")
    p("-" * 60)

    # Sort by either_match rate for comparison
    sorted_langs = sorted(summary.items(),
                          key=lambda x: x[1]["either_match"] / max(x[1]["total"], 1),
                          reverse=True)
    for lname, s in sorted_langs:
        t = s["total"]
        pmp_pct = 100 * s["pmp_match"] / t if t else 0
        pan_pct = 100 * s["pan_match"] / t if t else 0
        either_pct = 100 * s["either_match"] / t if t else 0
        p(f"{lname:<25} {t:>6} {pmp_pct:>6.1f}% {pan_pct:>6.1f}% {either_pct:>7.1f}%")

    p()
    p("-" * 60)

    # Key comparisons
    p()
    p("KEY COMPARISONS (PMP cognacy rate):")
    p()

    def get_rate(lname, field="pmp_match"):
        s = summary.get(lname, {"total": 0, field: 0})
        return s[field] / max(s["total"], 1)

    jav_rate = get_rate("Javanese")
    bal_rate = get_rate("Balinese")
    teng_rate = get_rate("Tengger_Ngadas")
    jav_y_rate = get_rate("Javanese_Yogyakarta")
    jav_m_rate = get_rate("Javanese_Malang")
    malag_rate = get_rate("Merina_Malagasy")
    oj_rate = get_rate("Old_Javanese")

    p(f"  Balinese vs Javanese:     {bal_rate:.3f} vs {jav_rate:.3f}  (diff: {bal_rate - jav_rate:+.3f})")
    p(f"  Tengger vs Javanese:      {teng_rate:.3f} vs {jav_rate:.3f}  (diff: {teng_rate - jav_rate:+.3f})")
    p(f"  Malagasy vs Javanese:     {malag_rate:.3f} vs {jav_rate:.3f}  (diff: {malag_rate - jav_rate:+.3f})")
    if summary.get("Javanese_Yogyakarta", {}).get("total", 0) > 0:
        p(f"  Jav(Yogya) vs Jav(std):   {jav_y_rate:.3f} vs {jav_rate:.3f}  (diff: {jav_y_rate - jav_rate:+.3f})")
    if summary.get("Javanese_Malang", {}).get("total", 0) > 0:
        p(f"  Jav(Malang) vs Jav(std):  {jav_m_rate:.3f} vs {jav_rate:.3f}  (diff: {jav_m_rate - jav_rate:+.3f})")
    if summary.get("Old_Javanese", {}).get("total", 0) > 0:
        p(f"  Old Javanese vs Javanese: {oj_rate:.3f} vs {jav_rate:.3f}  (diff: {oj_rate - jav_rate:+.3f})")

    # Statistical test: McNemar's test for paired binary data
    p()
    p("STATISTICAL TESTS (McNemar's — paired by concept):")
    p()

    # For each pair, count: both match, only A matches, only B matches, neither
    def mcnemar_test(lang_a, lang_b, field="pmp_match"):
        """McNemar's test for paired binary data."""
        a_data = {}
        b_data = {}
        for r in results:
            if r["language"] == lang_a:
                a_data[r["concept"]] = r[field]
            elif r["language"] == lang_b:
                b_data[r["concept"]] = r[field]
        shared = set(a_data.keys()) & set(b_data.keys())
        if not shared:
            return None, None, None, 0
        # McNemar cells
        b_both = sum(1 for c in shared if a_data[c] == 1 and b_data[c] == 1)
        b_a_only = sum(1 for c in shared if a_data[c] == 1 and b_data[c] == 0)
        b_b_only = sum(1 for c in shared if a_data[c] == 0 and b_data[c] == 1)
        b_neither = sum(1 for c in shared if a_data[c] == 0 and b_data[c] == 0)

        # McNemar chi-squared (with continuity correction)
        n_disc = b_a_only + b_b_only
        if n_disc == 0:
            chi2 = 0
            p_val = 1.0
        else:
            chi2 = (abs(b_a_only - b_b_only) - 1) ** 2 / n_disc if n_disc > 0 else 0
            # Approximate p-value using chi2(1) distribution
            # For chi2(1): p ≈ exp(-chi2/2) for moderate values (rough approx)
            # Better: use normal approximation
            import math
            z = math.sqrt(chi2)
            # Two-sided p from normal: p ≈ 2 * (1 - Phi(z))
            # Using error function approximation
            p_val = math.erfc(z / math.sqrt(2))

        return chi2, p_val, (b_a_only, b_b_only, b_both, b_neither), len(shared)

    pairs = [
        ("Balinese", "Javanese"),
        ("Tengger_Ngadas", "Javanese"),
        ("Merina_Malagasy", "Javanese"),
        ("Balinese", "Tengger_Ngadas"),
    ]

    for a, b in pairs:
        chi2, pval, cells, n = mcnemar_test(a, b)
        if chi2 is not None and n > 0:
            a_only, b_only, both, neither = cells
            p(f"  {a} vs {b} (n={n} shared concepts):")
            p(f"    Both match: {both}, {a}-only: {a_only}, {b}-only: {b_only}, Neither: {neither}")
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            p(f"    McNemar chi2={chi2:.3f}, p={pval:.4f} {sig}")
            p()

    # Domain breakdown
    p()
    p("DOMAIN BREAKDOWN (PMP cognacy rate by semantic domain):")
    p()
    domains = ["BODY", "NATURE", "ACTION", "NUMBER", "KINSHIP", "OTHER"]
    key_langs = ["Balinese", "Javanese", "Tengger_Ngadas", "Merina_Malagasy"]

    header = f"{'Domain':<10}"
    for l in key_langs:
        short = l[:8]
        header += f" {short:>10}"
    p(header)
    p("-" * (10 + 11 * len(key_langs)))

    for dom in domains:
        row_str = f"{dom:<10}"
        for lname in key_langs:
            dom_results = [r for r in results if r["language"] == lname and r["domain"] == dom]
            if dom_results:
                rate = sum(r["pmp_match"] for r in dom_results) / len(dom_results)
                row_str += f" {rate:>9.1%}"
            else:
                row_str += f" {'N/A':>10}"
        p(row_str)

    # Concepts where peripheral matches PMP but central doesn't
    p()
    p("CONCEPTS: Balinese matches PMP but Javanese doesn't (peripheral advantage):")
    p()
    bal_concepts = {r["concept"]: r["pmp_match"] for r in results if r["language"] == "Balinese"}
    jav_concepts = {r["concept"]: r["pmp_match"] for r in results if r["language"] == "Javanese"}
    peripheral_advantage = []
    for concept in sorted(set(bal_concepts.keys()) & set(jav_concepts.keys())):
        if bal_concepts[concept] == 1 and jav_concepts[concept] == 0:
            peripheral_advantage.append(concept)
    for c in peripheral_advantage[:20]:
        p(f"  - {c}")
    if len(peripheral_advantage) > 20:
        p(f"  ... and {len(peripheral_advantage) - 20} more")
    p(f"  Total: {len(peripheral_advantage)} concepts")

    # Central advantage
    p()
    p("CONCEPTS: Javanese matches PMP but Balinese doesn't:")
    p()
    central_advantage = []
    for concept in sorted(set(bal_concepts.keys()) & set(jav_concepts.keys())):
        if bal_concepts[concept] == 0 and jav_concepts[concept] == 1:
            central_advantage.append(concept)
    for c in central_advantage[:20]:
        p(f"  - {c}")
    if len(central_advantage) > 20:
        p(f"  ... and {len(central_advantage) - 20} more")
    p(f"  Total: {len(central_advantage)} concepts")

    # Tengger special analysis
    if summary.get("Tengger_Ngadas", {}).get("total", 0) > 0:
        p()
        p("TENGGER SPECIAL ANALYSIS:")
        teng_concepts = {r["concept"]: r for r in results if r["language"] == "Tengger_Ngadas"}
        p(f"  Tengger covers {len(teng_concepts)} concepts in ABVD")
        teng_unique = []
        for concept in sorted(teng_concepts.keys()):
            if teng_concepts[concept]["pmp_match"] == 1 and jav_concepts.get(concept, 0) == 0:
                teng_unique.append(concept)
        p(f"  Tengger matches PMP but Javanese doesn't: {len(teng_unique)} concepts")
        for c in teng_unique[:15]:
            p(f"    - {c}")

    # Save summary
    summary_path = OUT / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\nSaved: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
