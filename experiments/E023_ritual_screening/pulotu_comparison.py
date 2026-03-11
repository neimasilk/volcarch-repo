"""
E023: Cross-Austronesian mortuary belief comparison using Pulotu database.
Compares ancestral spirit beliefs and afterlife practices across
Toraja, Merina (Madagascar), Tanala (Madagascar), and Java-related cultures.

Run: python experiments/E023_ritual_screening/pulotu_comparison.py
"""
import csv
import io
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PULOTU = Path(__file__).parent / "data" / "pulotu" / "cldf"
OUT = Path(__file__).parent / "results"

# Questions relevant to P5 mortuary comparison
KEY_QUESTIONS = {
    "2": "Belief in god(s)",
    "3": "Belief in nature god(s)",
    "4": "Belief in deified ancestor(s)",
    "5": "Belief in ancestral spirits",
    "6": "Belief in nature spirits",
    "7": "Supernatural punishment for impiety",
    "10": "Actions of others after death affect afterlife",
    "11": "One's actions while living affect afterlife",
    "12": "Belief in culture hero(es)",
    "21": "Mana as concept",
    "35": "Costly sacrifices and offerings",
}

# Scale: 0=absent, 1=present minor, 2=present major, 3=principal focus
CODE_LABELS = {
    "0": "Absent",
    "1": "Present (minor)",
    "2": "Present (major focus)",
    "3": "Principal focus",
}


def main():
    # Load all data
    cultures = {}
    with open(PULOTU / "cultures.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cultures[row["ID"]] = row

    questions = {}
    with open(PULOTU / "questions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            questions[row["ID"]] = row

    codes = {}
    with open(PULOTU / "codes.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            codes[row["ID"]] = row

    responses = defaultdict(dict)  # culture -> {q_id: value}
    comments = defaultdict(dict)   # culture -> {q_id: comment}
    with open(PULOTU / "responses.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["Language_ID"]
            qid = row["Parameter_ID"]
            responses[cid][qid] = row.get("Value", "")
            if row.get("Comment"):
                comments[cid][qid] = row["Comment"]

    # Target cultures for P5 comparison
    # Austronesian cultures with strong mortuary traditions
    targets = [
        "southern-toraja", "eastern-toraja",  # Sulawesi (Rambu Solo)
        "merina", "tanala",                   # Madagascar (Famadihana)
    ]

    # Also find any other cultures with strong ancestral practices
    strong_ancestor = []
    for cid, resp in responses.items():
        q5 = resp.get("5", "")
        q10 = resp.get("10", "")
        if q5 == "3" or q10 == "2":
            c = cultures.get(cid, {})
            strong_ancestor.append((cid, c.get("Name", cid)))

    print("=" * 70)
    print("PULOTU CROSS-AUSTRONESIAN MORTUARY COMPARISON")
    print(f"Database: 137 Austronesian cultures, 134 variables")
    print("=" * 70)

    # ============================================
    # 1. Target culture comparison
    # ============================================
    print("\n--- ANCESTRAL BELIEF PROFILE ---")
    print(f"{'Variable':<50}", end="")
    for t in targets:
        name = cultures.get(t, {}).get("Name", t)[:12]
        print(f" {name:>12}", end="")
    print()
    print("-" * (50 + 13 * len(targets)))

    for qid, qname in KEY_QUESTIONS.items():
        print(f"{qname:<50}", end="")
        for t in targets:
            val = responses.get(t, {}).get(qid, "?")
            label = CODE_LABELS.get(val, val)[:12]
            print(f" {label:>12}", end="")
        print()

    # ============================================
    # 2. Key finding: Q10 comparison
    # ============================================
    print("\n--- Q10: Do actions of others AFTER death affect afterlife? ---")
    print("(This is the key slametan question: do post-mortem rituals matter?)")
    print()
    for t in targets:
        val = responses.get(t, {}).get("10", "?")
        name = cultures.get(t, {}).get("Name", t)
        label = CODE_LABELS.get(val, val)
        comment = comments.get(t, {}).get("10", "")
        print(f"  {name:25s}: {label}")
        if comment:
            print(f"    Note: {comment[:100]}")

    # ============================================
    # 3. Cultures where post-death actions are PRINCIPAL determinant
    # ============================================
    print("\n--- Cultures where post-death actions = PRINCIPAL determinant (Q10=2) ---")
    q10_principal = [(cid, cultures.get(cid, {}).get("Name", cid))
                     for cid, resp in responses.items() if resp.get("10") == "2"]
    for cid, name in sorted(q10_principal, key=lambda x: x[1]):
        lat = cultures.get(cid, {}).get("Latitude", "?")
        lon = cultures.get(cid, {}).get("Longitude", "?")
        q5 = CODE_LABELS.get(responses.get(cid, {}).get("5", "?"), "?")[:20]
        print(f"  {name:25s} ({lat:>6s}, {lon:>6s})  Ancestral spirits: {q5}")

    # ============================================
    # 4. Cultures with ancestral spirits as PRINCIPAL focus (Q5=3)
    # ============================================
    print("\n--- Cultures where ancestral spirits = PRINCIPAL focus (Q5=3) ---")
    q5_principal = [(cid, cultures.get(cid, {}).get("Name", cid))
                    for cid, resp in responses.items() if resp.get("5") == "3"]
    for cid, name in sorted(q5_principal, key=lambda x: x[1]):
        lat = cultures.get(cid, {}).get("Latitude", "?")
        lon = cultures.get(cid, {}).get("Longitude", "?")
        q10 = CODE_LABELS.get(responses.get(cid, {}).get("10", "?"), "?")[:20]
        print(f"  {name:25s} ({lat:>6s}, {lon:>6s})  Post-death actions: {q10}")

    # ============================================
    # 5. Cross-pattern: which cultures share the full "package"?
    # ============================================
    print("\n--- FULL MORTUARY PACKAGE: Q4>=2 AND Q5>=2 AND Q10>=1 ---")
    print("(Deified ancestors + ancestral spirits + post-death ritual efficacy)")
    full_package = []
    for cid, resp in responses.items():
        q4 = int(resp.get("4", "0") or "0")
        q5 = int(resp.get("5", "0") or "0")
        q10 = int(resp.get("10", "-1") or "-1")
        if q4 >= 2 and q5 >= 2 and q10 >= 1:
            name = cultures.get(cid, {}).get("Name", cid)
            lat = cultures.get(cid, {}).get("Latitude", "?")
            lon = cultures.get(cid, {}).get("Longitude", "?")
            full_package.append((name, lat, lon, q4, q5, q10))

    print(f"\n  {len(full_package)}/{len(cultures)} cultures have the full package:")
    for name, lat, lon, q4, q5, q10 in sorted(full_package, key=lambda x: x[0]):
        q10_label = CODE_LABELS.get(str(q10), "?")[:15]
        print(f"  {name:25s} ({lat:>6s}, {lon:>6s})  "
              f"Ancestors={q4} Spirits={q5} PostDeath={q10_label}")

    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 70)
    print("P5 IMPLICATIONS")
    print("=" * 70)

    # Check if both Toraja and Merina share the same pattern
    toraja_q10 = max(
        int(responses.get("southern-toraja", {}).get("10", "0") or "0"),
        int(responses.get("eastern-toraja", {}).get("10", "0") or "0"),
    )
    merina_q10 = int(responses.get("merina", {}).get("10", "0") or "0")
    tanala_q10 = int(responses.get("tanala", {}).get("10", "0") or "0")

    print(f"""
Key Question Q10: "Do the actions of others AFTER death affect afterlife?"
  Southern Toraja: {CODE_LABELS.get(responses.get('southern-toraja',{}).get('10','?'), '?')}
  Eastern Toraja:  {CODE_LABELS.get(responses.get('eastern-toraja',{}).get('10','?'), '?')}
  Merina:          {CODE_LABELS.get(responses.get('merina',{}).get('10','?'), '?')}
  Tanala:          {CODE_LABELS.get(responses.get('tanala',{}).get('10','?'), '?')}

This is the FOUNDATIONAL BELIEF behind both:
  - Javanese slametan (3-7-40-100-1000 day post-death rituals)
  - Malagasy famadihana (periodic exhumation and rewrapping)
  - Torajan Rambu Solo and ma'nene (delayed funeral + annual rewrapping)

If Q10 >= 1 across both Toraja AND Madagascar, it confirms:
  → The belief that post-death actions matter is SHARED across geographically
    separated Austronesian cultures
  → This supports a common Austronesian origin for the concept
  → The specific NUMBERS may differ (7-40-100-1000 vs 3-5-7 year cycles)
    but the underlying belief structure is the same

Number of cultures with the full mortuary package: {len(full_package)}/137
""")

    # Save comparison table
    out_path = OUT / "pulotu_mortuary_comparison.csv"
    rows = []
    for t in list(targets) + [cid for cid, _ in q10_principal if cid not in targets]:
        row = {"culture": cultures.get(t, {}).get("Name", t)}
        for qid, qname in KEY_QUESTIONS.items():
            val = responses.get(t, {}).get(qid, "")
            row[qname] = CODE_LABELS.get(val, val)
        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
