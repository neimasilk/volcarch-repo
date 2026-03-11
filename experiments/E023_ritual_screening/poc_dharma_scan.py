"""
E023: Ritual Screening POC — DHARMA Corpus Scan
================================================
Scan DHARMA Nusantara epigraphy corpus:
1. Extract text from all 268 XML inscriptions
2. Identify inscriptions with ritual/cosmological content
3. Build corpus inventory (name, date, language, length, ritual keywords)
4. Extract text for 10 sample inscriptions for pilot AI screening

Run: python experiments/E023_ritual_screening/poc_dharma_scan.py
"""
import re
import csv
import sys
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).parent.parent.parent
DHARMA = REPO / "experiments" / "E023_ritual_screening" / "data" / "dharma" / "xml"
OUT = REPO / "experiments" / "E023_ritual_screening" / "results"
OUT.mkdir(exist_ok=True)

# TEI namespace
NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Ritual/cosmological keywords to search for in inscriptions
# Mix of Sanskrit, Old Javanese, and general terms
RITUAL_KEYWORDS = {
    # Sanskrit-layer ritual terms
    "yajña", "homa", "pūjā", "puja", "sraddha", "śrāddha", "piṇḍa",
    "kalpa", "mantra", "dīkṣā", "abhiṣeka", "abhiseka",
    # OJ ritual/administrative terms
    "sīma", "sapatha", "śapatha",  # oath/curse (boundary ritual)
    "maṅhuri",  # ritual offering
    "panumbas",  # ritual purchase/exchange
    "samgat",  # title (ritual official)
    # Death/afterlife terms
    "pralīna", "pralaya", "svarga", "svargga", "yamālaya",
    "pitṛ", "pitr", "atīta",  # ancestors, deceased
    # Indigenous cosmological terms
    "hyang", "hyaṁ",  # divine/sacred (pre-Indic?)
    "kabuyutan",  # sacred ancestral site
    "karāman",  # village ritual community
    "danu", "danau",  # lake (sacred geography)
    "gunung", "parvvata",  # mountain cosmology
    "sagara", "samudra",  # ocean
    # Calendar/time
    "tithi", "nakṣatra", "naksatra", "vāra", "wuku",
    "śaka", "saka",  # Saka calendar
    # Maritime
    "bahitra", "nauka", "prahu",  # boat/ship
    "samudra", "lavana",  # ocean
}


def extract_text(xml_path):
    """Extract transliterated text from DHARMA XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return "", {}

    metadata = {}

    # Get title
    title_el = root.find(".//tei:titleStmt/tei:title", NS)
    metadata["title"] = title_el.text.strip() if title_el is not None and title_el.text else xml_path.stem

    # Get language
    edition = root.find(".//tei:div[@type='edition']", NS)
    if edition is not None:
        metadata["lang"] = edition.get("{http://www.w3.org/XML/1998/namespace}lang", "unknown")
    else:
        metadata["lang"] = "unknown"

    # Get date if available
    date_el = root.find(".//tei:history//tei:date", NS)
    if date_el is not None:
        metadata["date"] = date_el.text.strip() if date_el.text else date_el.get("when", "")
    else:
        metadata["date"] = ""

    # Extract all text content from edition
    text_parts = []
    if edition is not None:
        for elem in edition.iter():
            if elem.text:
                text_parts.append(elem.text.strip())
            if elem.tail:
                text_parts.append(elem.tail.strip())

    # Also check translation if available
    translation = root.find(".//tei:div[@type='translation']", NS)
    trans_text = ""
    if translation is not None:
        trans_parts = []
        for elem in translation.iter():
            if elem.text:
                trans_parts.append(elem.text.strip())
            if elem.tail:
                trans_parts.append(elem.tail.strip())
        trans_text = " ".join(trans_parts)
        metadata["has_translation"] = True
    else:
        metadata["has_translation"] = False

    # Check commentary
    commentary = root.find(".//tei:div[@type='commentary']", NS)
    metadata["has_commentary"] = commentary is not None

    full_text = " ".join(text_parts)
    metadata["trans_text"] = trans_text

    return full_text, metadata


def find_keywords(text, trans_text=""):
    """Find ritual/cosmological keywords in text."""
    combined = (text + " " + trans_text).lower()
    found = []
    for kw in RITUAL_KEYWORDS:
        if kw.lower() in combined:
            found.append(kw)
    return found


def main():
    print("=" * 60)
    print("E023: DHARMA Corpus Scan — Ritual Content Inventory")
    print("=" * 60)

    xml_files = sorted(DHARMA.glob("*.xml"))
    print(f"\nTotal XML files: {len(xml_files)}")

    inventory = []
    all_keywords = Counter()
    with_ritual = 0
    with_translation = 0
    lang_counts = Counter()

    for xml_path in xml_files:
        text, meta = extract_text(xml_path)
        keywords = find_keywords(text, meta.get("trans_text", ""))

        word_count = len(text.split())
        entry = {
            "filename": xml_path.name,
            "title": meta.get("title", ""),
            "lang": meta.get("lang", ""),
            "date": meta.get("date", ""),
            "word_count": word_count,
            "has_translation": meta.get("has_translation", False),
            "has_commentary": meta.get("has_commentary", False),
            "n_ritual_keywords": len(keywords),
            "ritual_keywords": "|".join(keywords),
        }
        inventory.append(entry)

        if keywords:
            with_ritual += 1
            for kw in keywords:
                all_keywords[kw] += 1

        if meta.get("has_translation"):
            with_translation += 1

        lang_counts[meta.get("lang", "unknown")] += 1

    # Summary
    print(f"\n--- CORPUS SUMMARY ---")
    print(f"Total inscriptions: {len(inventory)}")
    print(f"With ritual/cosmological keywords: {with_ritual} ({100*with_ritual/len(inventory):.0f}%)")
    print(f"With English/translation: {with_translation} ({100*with_translation/len(inventory):.0f}%)")

    print(f"\n--- LANGUAGES ---")
    for lang, count in lang_counts.most_common():
        print(f"  {lang}: {count}")

    print(f"\n--- TOP RITUAL KEYWORDS ---")
    for kw, count in all_keywords.most_common(25):
        print(f"  {kw}: {count} inscriptions")

    # Find best candidates for pilot screening
    # (have translation + ritual keywords + substantial length)
    pilot_candidates = [e for e in inventory
                        if e["has_translation"] and e["n_ritual_keywords"] >= 2 and e["word_count"] > 50]
    pilot_candidates.sort(key=lambda x: -x["n_ritual_keywords"])

    print(f"\n--- PILOT SCREENING CANDIDATES ---")
    print(f"(have translation + 2+ ritual keywords + >50 words)")
    print(f"Found: {len(pilot_candidates)}")
    for entry in pilot_candidates[:15]:
        print(f"  {entry['filename']}: {entry['title'][:60]}")
        print(f"    {entry['word_count']} words, {entry['n_ritual_keywords']} keywords: {entry['ritual_keywords']}")

    # Save inventory
    out_csv = OUT / "dharma_corpus_inventory.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=inventory[0].keys())
        writer.writeheader()
        writer.writerows(inventory)
    print(f"\nSaved: {out_csv}")

    # Extract pilot texts (top 10 candidates with translations)
    pilot_out = OUT / "pilot_inscriptions.txt"
    with open(pilot_out, "w", encoding="utf-8") as f:
        for i, entry in enumerate(pilot_candidates[:10]):
            xml_path = DHARMA / entry["filename"]
            text, meta = extract_text(xml_path)
            f.write(f"{'='*60}\n")
            f.write(f"INSCRIPTION {i+1}: {entry['filename']}\n")
            f.write(f"Title: {entry['title']}\n")
            f.write(f"Language: {entry['lang']}\n")
            f.write(f"Date: {entry['date']}\n")
            f.write(f"Ritual keywords: {entry['ritual_keywords']}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"--- TRANSLITERATION ---\n{text}\n\n")
            if meta.get("trans_text"):
                f.write(f"--- TRANSLATION ---\n{meta['trans_text']}\n\n")
            f.write("\n")
    print(f"Saved: {pilot_out}")

    # GO/NO-GO
    print(f"\n{'='*60}")
    print(f"GO/NO-GO ASSESSMENT")
    print(f"{'='*60}")
    print(f"\nCorpus size: {len(inventory)} inscriptions")
    print(f"With ritual content: {with_ritual} ({100*with_ritual/len(inventory):.0f}%)")
    print(f"With translations: {with_translation} ({100*with_translation/len(inventory):.0f}%)")
    print(f"Pilot candidates (trans + ritual + length): {len(pilot_candidates)}")

    if len(pilot_candidates) >= 10:
        print(f"\n>>> VERDICT: GO — Sufficient corpus for ritual screening pipeline.")
    elif len(pilot_candidates) >= 5:
        print(f"\n>>> VERDICT: CONDITIONAL GO — Limited pilot candidates. Supplement with literary texts.")
    else:
        print(f"\n>>> VERDICT: NEEDS SUPPLEMENT — Too few translated ritual inscriptions. Use secondary literature.")


if __name__ == "__main__":
    main()
