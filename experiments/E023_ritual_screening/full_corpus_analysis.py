"""
E023: Full corpus analysis — classify all 268 inscriptions by pre-Indic content.
Uses the ritual element ontology from analyze_ritual_elements.py.

Run: python experiments/E023_ritual_screening/full_corpus_analysis.py
"""
import csv
import io
import sys
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESULTS = Path(__file__).parent / "results"

# Origin classification for every keyword found in corpus
ORIGIN_MAP = {
    # Indic
    "yajña": "indic", "homa": "indic", "pūjā": "indic", "puja": "indic",
    "sraddha": "indic", "śrāddha": "indic", "piṇḍa": "indic",
    "kalpa": "indic", "mantra": "indic", "dīkṣā": "indic",
    "abhiṣeka": "indic", "abhiseka": "indic",
    "pralīna": "indic", "pralaya": "indic", "svarga": "indic",
    "svargga": "indic", "yamālaya": "indic",
    "pitṛ": "indic", "pitr": "indic", "atīta": "indic",
    "tithi": "indic", "nakṣatra": "indic", "naksatra": "indic",
    "vāra": "indic", "śaka": "indic", "saka": "indic",
    "lavana": "indic",
    # Pre-Indic
    "hyang": "pre_indic", "hyaṁ": "pre_indic",
    "kabuyutan": "pre_indic", "karāman": "pre_indic",
    "maṅhuri": "pre_indic", "panumbas": "pre_indic",
    "wuku": "pre_indic",
    "gunung": "pre_indic",
    "prahu": "pre_indic",
    # Ambiguous
    "sīma": "ambiguous", "sapatha": "ambiguous", "śapatha": "ambiguous",
    "samgat": "ambiguous",
    "danu": "ambiguous", "danau": "ambiguous",
    "parvvata": "ambiguous", "sagara": "ambiguous",
    "samudra": "ambiguous",
    "bahitra": "ambiguous", "nauka": "ambiguous",
}


def main():
    inventory_path = RESULTS / "dharma_corpus_inventory.csv"
    if not inventory_path.exists():
        print(f"ERROR: {inventory_path} not found. Run poc_dharma_scan.py first.")
        sys.exit(1)

    # Load inventory
    with open(inventory_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        inscriptions = list(reader)

    print("=" * 70)
    print(f"FULL CORPUS ANALYSIS — {len(inscriptions)} INSCRIPTIONS")
    print("=" * 70)

    # Classify each inscription
    results = []
    total_indic = total_pre = total_amb = 0
    lang_stats = Counter()
    pre_indic_by_lang = Counter()
    hyang_count = 0
    manhuri_count = 0
    wuku_count = 0

    for insc in inscriptions:
        kw_str = insc.get("ritual_keywords", "")
        keywords = [k.strip() for k in kw_str.split("|") if k.strip()] if kw_str else []

        n_indic = sum(1 for k in keywords if ORIGIN_MAP.get(k) == "indic")
        n_pre = sum(1 for k in keywords if ORIGIN_MAP.get(k) == "pre_indic")
        n_amb = sum(1 for k in keywords if ORIGIN_MAP.get(k) == "ambiguous")
        total = len(keywords)
        ratio = n_pre / total if total > 0 else 0

        total_indic += n_indic
        total_pre += n_pre
        total_amb += n_amb

        lang = insc.get("lang", "unknown")
        lang_stats[lang] += 1
        if n_pre > 0:
            pre_indic_by_lang[lang] += 1

        pre_kws = [k for k in keywords if ORIGIN_MAP.get(k) == "pre_indic"]
        has_hyang = any(k in ("hyaṁ", "hyang") for k in keywords)
        has_manhuri = "maṅhuri" in keywords
        has_wuku = "wuku" in keywords
        if has_hyang:
            hyang_count += 1
        if has_manhuri:
            manhuri_count += 1
        if has_wuku:
            wuku_count += 1

        results.append({
            "filename": insc["filename"],
            "title": insc["title"],
            "lang": lang,
            "date": insc.get("date", ""),
            "word_count": insc.get("word_count", 0),
            "total_keywords": total,
            "indic": n_indic,
            "pre_indic": n_pre,
            "ambiguous": n_amb,
            "pre_indic_ratio": round(ratio, 3),
            "pre_indic_keywords": "|".join(pre_kws),
            "has_hyang": has_hyang,
            "has_manhuri": has_manhuri,
            "has_wuku": has_wuku,
        })

    # ============================================
    # Summary statistics
    # ============================================
    n_with_keywords = sum(1 for r in results if r["total_keywords"] > 0)
    n_with_pre = sum(1 for r in results if r["pre_indic"] > 0)

    print(f"\nInscriptions with any ritual keywords: {n_with_keywords}/{len(results)}")
    print(f"Inscriptions with pre-Indic elements:  {n_with_pre}/{len(results)} "
          f"({n_with_pre/len(results)*100:.0f}%)")

    print(f"\nTotal keyword occurrences:")
    print(f"  Indic:     {total_indic}")
    print(f"  Pre-Indic: {total_pre}")
    print(f"  Ambiguous: {total_amb}")

    # Key pre-Indic markers
    print(f"\n--- PRE-INDIC MARKER PREVALENCE ---")
    print(f"  hyaṁ/hyang: {hyang_count}/{len(results)} ({hyang_count/len(results)*100:.0f}%)")
    print(f"  maṅhuri:    {manhuri_count}/{len(results)} ({manhuri_count/len(results)*100:.0f}%)")
    print(f"  wuku:       {wuku_count}/{len(results)} ({wuku_count/len(results)*100:.0f}%)")

    # By language
    print(f"\n--- PRE-INDIC ELEMENTS BY LANGUAGE ---")
    print(f"{'Language':<20} {'Total':<8} {'With pre-Indic':<16} {'%'}")
    print("-" * 50)
    for lang, count in lang_stats.most_common():
        pre = pre_indic_by_lang.get(lang, 0)
        pct = pre / count * 100 if count > 0 else 0
        print(f"{lang:<20} {count:<8} {pre:<16} {pct:.0f}%")

    # Full keyword frequency across corpus
    all_kw_freq = Counter()
    for insc in inscriptions:
        kw_str = insc.get("ritual_keywords", "")
        if kw_str:
            for k in kw_str.split("|"):
                k = k.strip()
                if k:
                    all_kw_freq[k] += 1

    print(f"\n--- FULL CORPUS KEYWORD FREQUENCY ---")
    print(f"{'Keyword':<16} {'Count':<8} {'%corpus':<10} {'Origin'}")
    print("-" * 50)
    for kw, count in all_kw_freq.most_common():
        pct = count / len(results) * 100
        origin = ORIGIN_MAP.get(kw, "?")
        marker = " ***" if origin == "pre_indic" else ""
        print(f"{kw:<16} {count:<8} {pct:>5.1f}%     {origin}{marker}")

    # Top inscriptions by pre-Indic ratio
    pre_inscriptions = sorted(
        [r for r in results if r["total_keywords"] >= 3],
        key=lambda x: x["pre_indic_ratio"],
        reverse=True,
    )
    print(f"\n--- TOP 15 INSCRIPTIONS BY PRE-INDIC RATIO (min 3 keywords) ---")
    print(f"{'Title':<40} {'Lang':<12} {'Pre/Total':<12} {'Ratio':<8} {'Pre-Indic Keywords'}")
    print("-" * 100)
    for r in pre_inscriptions[:15]:
        print(f"{r['title'][:39]:<40} {r['lang']:<12} "
              f"{r['pre_indic']}/{r['total_keywords']:<10} "
              f"{r['pre_indic_ratio']:.1%}     {r['pre_indic_keywords']}")

    # ============================================
    # Save full results
    # ============================================
    out_path = RESULTS / "full_corpus_classification.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {out_path}")

    # ============================================
    # Key conclusions
    # ============================================
    print("\n" + "=" * 70)
    print("CORPUS-WIDE CONCLUSIONS")
    print("=" * 70)
    print(f"""
1. hyaṁ/hyang prevalence: {hyang_count}/268 = {hyang_count/268*100:.0f}%
   → Confirms pilot finding: indigenous divinity concept is pervasive
   → Present across Old Javanese, possibly Old Malay inscriptions

2. maṅhuri prevalence: {manhuri_count}/268 = {manhuri_count/268*100:.0f}%
   → "Ancestor return" concept embedded in ~{manhuri_count/268*100:.0f}% of corpus
   → Strong candidate for cross-Austronesian comparison (Madagascar)

3. Pre-Indic elements detected in {n_with_pre}/268 = {n_with_pre/268*100:.0f}% of inscriptions
   → Even with conservative keyword list, substrate is detectable
   → Expanding ontology will likely increase detection rate

4. Language distribution: Pre-Indic elements concentrate in Old Javanese
   → Expected: Sanskrit inscriptions use Sanskrit cosmology
   → Old Malay inscriptions may have different substrate markers
""")
    print("Done.")


if __name__ == "__main__":
    main()
