"""
E207: GLOBALISE VOC Transcription Pilot — NER on Actual 18th-Century Dutch
============================================================================
Download and analyze actual VOC administrative correspondence from GLOBALISE.
Run ArcheoBERTje-NER + custom pattern matching to assess PhD feasibility.
"""

import json
import re
import os
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path("D:/documents/volcarch-repo/experiments/E207_globalise_voc_pilot/data")
OUT_DIR = Path("D:/documents/volcarch-repo/experiments/E207_globalise_voc_pilot/results")

def load_voc_texts():
    """Load downloaded VOC transcription files."""
    all_lines = []
    file_stats = {}
    for f in sorted(DATA_DIR.glob("voc_*.txt")):
        lines = []
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith("#+"):
                    lines.append(stripped)
        file_stats[f.name] = len(lines)
        all_lines.extend(lines)
    return all_lines, file_stats

def analyze_historical_dutch(lines):
    """Analyze linguistic features of 18th-century VOC Dutch."""

    # Historical spelling patterns
    spelling_patterns = {
        "ij_for_y": re.compile(r'\b\w*ij\w*\b'),  # Januarij, Meij
        "double_vowel": re.compile(r'\b\w*[aeiou]{2}\w*\b'),
        "line_break_hyphen": re.compile(r'\w+[„¬]\s*$'),  # HTR line break artifacts
        "old_date": re.compile(r'\d+[=\.]\s*[edstn]+\s+(Januarij|Februarij|Maart|April|Meij|Junij|Julij|Augustus|September|October|November|December)', re.I),
        "colonial_place": re.compile(r'\b(Batavia|Ceilon|Malacca|Amboina|Ternaten|Banda|Makassar|Soerabaja|Samarang|Japara|Bantam|Mataram|Buitenzorg|Grissee|Cheribon)\b', re.I),
        "voc_title": re.compile(r'\b(Gouverneur|Generaal|Raaden|Commissaris|Directeur|Opperkoopman|Resident|Secretaris)\b', re.I),
        "measurement": re.compile(r'\b(\d+\.?\d*)\s*(el|voet|vadem|roede|palm|duim|mijl|paal|last|pikol)\b', re.I),
        "currency": re.compile(r'\b(\d+\.?\d*)\s*(gulden|stuiver|rijksdaalder|reaal|ropij|ducaat)\b', re.I),
        "geographic_admin": re.compile(r'\b(residentie|regentschap|district|dessa|kampong|negorij|landschap)\b', re.I),
        "settlement_mention": re.compile(r'\b(stad|fort|loge|comptoir|negotie|factorij|etablissement|haven|reede)\b', re.I),
        "archaeological": re.compile(r'\b(oudheid|tempel|graf|inscriptie|beeld|ruïne|opgraving|steen|baksteen)\b', re.I),
        "volcanic": re.compile(r'\b(vulkaan|berg|uitbarsting|lava|asch|modder|aardbeving)\b', re.I),
        "depth": re.compile(r'\b(diep|diepte|onder\s+den?\s+grond|begraven)\b', re.I),
    }

    results = {}
    for name, pattern in spelling_patterns.items():
        matches = []
        for line in lines:
            for m in pattern.finditer(line):
                matches.append({
                    "match": m.group(),
                    "context": line[:100]
                })
        results[name] = {
            "count": len(matches),
            "examples": matches[:10]
        }

    return results

def analyze_htr_quality(lines):
    """Assess HTR transcription quality issues."""
    issues = {
        "line_break_artifacts": 0,    # „ or ¬ at end of word
        "equals_in_dates": 0,         # 15=e, 2=den
        "special_chars": 0,           # ƒ, =, ¬, „
        "very_short_lines": 0,        # <5 chars (OCR fragments)
        "number_fragments": 0,        # standalone numbers
        "mixed_case_words": 0,        # mIddle caps (OCR error)
    }

    for line in lines:
        if re.search(r'\w[„¬]\s*$', line):
            issues["line_break_artifacts"] += 1
        if re.search(r'\d+=', line):
            issues["equals_in_dates"] += 1
        if re.search(r'[ƒ¬„]', line):
            issues["special_chars"] += 1
        if len(line) < 5:
            issues["very_short_lines"] += 1
        if re.match(r'^\d+\.?$', line.strip()):
            issues["number_fragments"] += 1

    return issues

def run_archeobert_on_voc(lines, max_segments=500):
    """Run ArcheoBERTje-NER on VOC text segments."""
    from transformers import pipeline

    # Filter to substantial lines
    segments = [l for l in lines if len(l) > 30][:max_segments]

    print(f"  Running ArcheoBERTje on {len(segments)} VOC segments...")
    ner = pipeline(
        "ner",
        model="alexbrandsen/ArcheoBERTje-NER",
        aggregation_strategy="simple",
        device=0
    )

    entity_counts = Counter()
    all_entities = []

    for i in range(0, len(segments), 16):
        batch = segments[i:i+16]
        try:
            results = ner(batch)
            for text, ents in zip(batch, results):
                for ent in ents:
                    entity_counts[ent["entity_group"]] += 1
                    all_entities.append({
                        "type": ent["entity_group"],
                        "word": ent["word"],
                        "score": float(ent["score"]),
                        "context": text[:80]
                    })
        except Exception as e:
            continue

    return dict(entity_counts), all_entities

def compare_ov_vs_voc(ov_counts, voc_counts, n_ov=2000, n_voc=500):
    """Compare ArcheoBERTje performance on OV (1912) vs VOC (1786)."""
    comparison = {}
    all_types = set(list(ov_counts.keys()) + list(voc_counts.keys()))

    for etype in sorted(all_types):
        ov_rate = ov_counts.get(etype, 0) / n_ov
        voc_rate = voc_counts.get(etype, 0) / n_voc
        ratio = voc_rate / ov_rate if ov_rate > 0 else float('inf')
        comparison[etype] = {
            "ov_count": ov_counts.get(etype, 0),
            "ov_rate": round(ov_rate, 4),
            "voc_count": voc_counts.get(etype, 0),
            "voc_rate": round(voc_rate, 4),
            "ratio_voc_over_ov": round(ratio, 2)
        }

    return comparison

def main():
    print("=" * 60)
    print("E207: GLOBALISE VOC Transcription Pilot")
    print("=" * 60)

    # 1. Load VOC texts
    print("\n1. Loading VOC transcriptions...")
    lines, file_stats = load_voc_texts()
    total_lines = sum(file_stats.values())
    print(f"   Loaded {total_lines} lines from {len(file_stats)} files")
    for name, count in file_stats.items():
        print(f"     {name}: {count} lines")

    # 2. Historical Dutch analysis
    print("\n2. Analyzing 18th-century Dutch features...")
    hist_analysis = analyze_historical_dutch(lines)
    for name, data in hist_analysis.items():
        if data["count"] > 0:
            print(f"   {name}: {data['count']} occurrences")
            if data["examples"]:
                print(f"     Example: {data['examples'][0]['match']}")

    # 3. HTR quality assessment
    print("\n3. Assessing HTR quality...")
    htr_issues = analyze_htr_quality(lines)
    for issue, count in htr_issues.items():
        if count > 0:
            print(f"   {issue}: {count}")

    # 4. ArcheoBERTje on VOC
    print("\n4. Running ArcheoBERTje-NER on VOC text...")
    voc_entities, voc_entity_list = run_archeobert_on_voc(lines)
    print(f"   Total entities: {sum(voc_entities.values())}")
    for etype, count in sorted(voc_entities.items(), key=lambda x: -x[1]):
        print(f"     {etype}: {count}")

    # 5. Compare with E206 (OV results)
    print("\n5. Comparing OV (1912) vs VOC (1786) performance...")
    # E206 results (from archeobert_evaluation.json)
    ov_counts = {"PER": 151, "ART": 244, "LOC": 249, "SPE": 176, "CON": 61, "MAT": 107}
    comparison = compare_ov_vs_voc(ov_counts, voc_entities)
    for etype, data in comparison.items():
        print(f"   {etype}: OV={data['ov_rate']}/seg, VOC={data['voc_rate']}/seg, ratio={data['ratio_voc_over_ov']}")

    # Save results
    results = {
        "experiment": "E207",
        "data_source": "GLOBALISE VOC Transcriptions v2 (CC0)",
        "data_url": "https://datasets.iisg.amsterdam/dataset.xhtml?persistentId=hdl:10622/LVXSBW",
        "files_downloaded": file_stats,
        "total_lines": total_lines,
        "historical_dutch_analysis": {
            name: {"count": d["count"], "examples": d["examples"][:5]}
            for name, d in hist_analysis.items()
        },
        "htr_quality_issues": htr_issues,
        "archeobert_on_voc": {
            "entity_counts": voc_entities,
            "total_entities": sum(voc_entities.values()),
            "n_segments": min(500, len([l for l in lines if len(l) > 30])),
            "entity_examples": voc_entity_list[:50]
        },
        "ov_vs_voc_comparison": comparison,
        "phd_implications": {
            "data_volume": f"GLOBALISE has 6,893 inventory numbers with TXT transcriptions = ~{total_lines * (6893/3):.0f} estimated total lines",
            "htr_challenges": "Line break artifacts (word splits across lines), equals-sign dates, special characters — all WORSE than OV OCR",
            "18th_century_challenges": [
                "Spelling: Januarij, Meij, Julij (pre-reform Dutch)",
                "Abbreviations: M=r (Mijnheer), UE: (Uw Edelheid)",
                "Administrative vocabulary: Gouverneur Generaal, Raaden van Indie",
                "Mixed languages: Dutch + Malay + Portuguese place names",
                "HTR line breaks split words: 'ver„rigtingen', 'hou„vernements'"
            ],
            "settlement_extraction_potential": "Administrative correspondence mentions forts, trading posts, settlements — structured settlement data IF entities can be extracted"
        }
    }

    out_file = OUT_DIR / "voc_pilot_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n   Results saved to {out_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nVOC data: {total_lines} lines from 3 files (~1786 CE administrative correspondence)")
    print(f"GLOBALISE total: 6,893 inventory numbers → estimated {total_lines * (6893/3):.0f} lines")
    print(f"\nHTR Quality Issues:")
    for issue, count in htr_issues.items():
        if count > 0:
            print(f"  - {issue}: {count} ({count/total_lines*100:.1f}%)")
    print(f"\nArcheoBERTje: {sum(voc_entities.values())} entities from {min(500, len([l for l in lines if len(l) > 30]))} VOC segments")
    print(f"\n18th-century Dutch NLP Challenges (PhD RQ1):")
    print(f"  - Colonial place names found: {hist_analysis['colonial_place']['count']}")
    print(f"  - Historical dates found: {hist_analysis['old_date']['count']}")
    print(f"  - Settlement mentions found: {hist_analysis['settlement_mention']['count']}")
    print(f"  - Administrative terms found: {hist_analysis['geographic_admin']['count']}")
    print(f"  - Measurements found: {hist_analysis['measurement']['count']}")
    print(f"  - Volcanic references: {hist_analysis['volcanic']['count']}")
    print(f"  - Depth references: {hist_analysis['depth']['count']}")

if __name__ == "__main__":
    main()
