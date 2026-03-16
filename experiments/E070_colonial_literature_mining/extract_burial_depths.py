"""
E070: Extract burial depth references from colonial OV reports.
Searches for Dutch keywords related to burial, depth, and volcanic deposits.
"""
import os
import re
from pathlib import Path

OV_DIR = Path("data/raw/colonial_sources/OV")
RESULTS_DIR = Path("experiments/E070_colonial_literature_mining/results")

# Dutch keywords for burial/depth/volcanic context
DEPTH_PATTERNS = [
    # Direct depth measurements
    r'(\d+[\.,]\d+)\s*[Mm]\.?\s*diep',           # X.X M. diep(te)
    r'diepte\s*van\s*(\d+[\.,]\d+)\s*[Mm]',       # diepte van X.X M
    r'(\d+)\s*(?:voet|vadem)\s*diep',              # X voet/vadem diep
    r'op\s*(?:een|eene)?\s*diepte\s*van\s*(\d+)',  # op een diepte van X
    # Burial language
    r'(?:onder|in)\s*den?\s*grond\s*(?:bedolven|begraven|geraak|gezakt|gevonden)',
    r'geheel\s*(?:in|onder)\s*den?\s*grond',
    r'boven\s*den?\s*grond\s*uitstekend',
    # Volcanic context
    r'(?:vulk|lava|lahar|modder|asch|puim)[\w]*',
    r'(?:eruptie|uitbarsting|krater)',
    r'(?:Kloet|Keloed|Merapi|Smeroe|Semeru|Bromo|Ringgit|Ardjoeno|Welirang)',
]

# Temple/site identification
SITE_PATTERNS = [
    r'[Tt]jandi\s+\w+',           # Tjandi X (= Candi X)
    r'[Cc]andi\s+\w+',
    r'[Bb]aksteenen?\s+\w+',      # brick structures
    r'[Ff]undament\w*',            # foundations
    r'[Oo]ntgraving\w*',          # excavations
    r'[Oo]pgraving\w*',           # excavations
]

def extract_context(text, match_start, match_end, context_chars=300):
    """Extract surrounding context for a match."""
    start = max(0, match_start - context_chars)
    end = min(len(text), match_end + context_chars)
    return text[start:end].replace('\n', ' ').strip()

def search_ov_volume(filepath):
    """Search a single OV volume for burial depth references."""
    year = re.search(r'OV_(\d{4})', filepath.name)
    year = year.group(1) if year else "unknown"

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    results = []

    # Search for depth patterns
    for pattern in DEPTH_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            context = extract_context(text, match.start(), match.end())

            # Check if context mentions a temple or archaeological site
            has_site = any(re.search(sp, context, re.IGNORECASE) for sp in SITE_PATTERNS)

            results.append({
                'year': year,
                'pattern': pattern[:50],
                'match': match.group(),
                'has_site_context': has_site,
                'context': context[:500],
                'position': match.start(),
            })

    return results

def main():
    all_results = []

    for filepath in sorted(OV_DIR.glob("OV_*_fulltext.txt")):
        print(f"Scanning {filepath.name}...")
        results = search_ov_volume(filepath)
        all_results.extend(results)

        # Count by type
        depth_mentions = sum(1 for r in results if any(c.isdigit() for c in r['match']))
        burial_mentions = sum(1 for r in results if 'grond' in r['match'].lower())
        volcanic_mentions = sum(1 for r in results if any(v in r['match'].lower() for v in ['kloet', 'keloed', 'merapi', 'smeroe', 'bromo', 'vulk', 'lava', 'lahar']))
        site_context = sum(1 for r in results if r['has_site_context'])

        print(f"  {len(results)} matches: {depth_mentions} depth, {burial_mentions} burial, {volcanic_mentions} volcanic, {site_context} with site context")

    # Write detailed results
    output_file = RESULTS_DIR / "ov_extraction_raw.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"OV Extraction Results\n")
        f.write(f"Total matches: {len(all_results)}\n")
        f.write(f"Volumes scanned: {len(list(OV_DIR.glob('OV_*_fulltext.txt')))}\n\n")

        # Priority: matches with both depth AND site context
        priority = [r for r in all_results if r['has_site_context'] and any(c.isdigit() for c in r['match'])]
        f.write(f"=== PRIORITY MATCHES (depth + site context): {len(priority)} ===\n\n")
        for r in sorted(priority, key=lambda x: x['year']):
            f.write(f"[OV {r['year']}] Match: {r['match']}\n")
            f.write(f"Context: {r['context']}\n")
            f.write(f"---\n\n")

        # Volcanic context matches
        volcanic = [r for r in all_results if any(v in r['match'].lower() for v in ['kloet', 'keloed', 'merapi', 'smeroe', 'bromo', 'vulk', 'lava', 'lahar', 'eruptie', 'uitbarsting'])]
        f.write(f"\n=== VOLCANIC CONTEXT MATCHES: {len(volcanic)} ===\n\n")
        for r in sorted(volcanic, key=lambda x: x['year']):
            f.write(f"[OV {r['year']}] Match: {r['match']}\n")
            f.write(f"Context: {r['context']}\n")
            f.write(f"---\n\n")

        # All burial mentions
        burial = [r for r in all_results if 'grond' in r['context'].lower() and ('begraven' in r['context'].lower() or 'bedolven' in r['context'].lower() or 'geraak' in r['context'].lower())]
        f.write(f"\n=== BURIAL MENTIONS: {len(burial)} ===\n\n")
        for r in sorted(burial, key=lambda x: x['year']):
            f.write(f"[OV {r['year']}] Match: {r['match']}\n")
            f.write(f"Context: {r['context']}\n")
            f.write(f"---\n\n")

    print(f"\nResults written to {output_file}")
    print(f"Total matches across all volumes: {len(all_results)}")
    print(f"Priority (depth + site): {len(priority)}")
    print(f"Volcanic context: {len(volcanic)}")

if __name__ == "__main__":
    main()
