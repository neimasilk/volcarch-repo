"""
GLOBALISE VOC Text Preprocessor
=================================
Cleans HTR-transcribed VOC text for NLP processing:
1. Strip GLOBALISE headers/metadata lines
2. Rejoin hyphenated words broken across lines (HTR line-break artifacts)
3. Normalize special characters
4. Filter noise (very short lines, number-only lines)
5. Paragraph segmentation

Usage:
    python preprocess_voc.py --input data/raw/globalise_voc/ --output data/processed/globalise_voc/
    python preprocess_voc.py --input path/to/file.txt  (single file mode, prints to stdout)
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import Counter


# HTR line-break characters used by GLOBALISE Loghi
BREAK_CHARS = set(["\\u201e", "\\u00ac", "\u201e", "\u00ac"])  # „ and ¬


def strip_metadata(lines):
    """Remove GLOBALISE header comments and XML reference lines."""
    clean = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#+"):
            continue
        clean.append(stripped)
    return clean


def rejoin_broken_words(lines):
    """Rejoin words split across lines by HTR line-break markers."""
    rejoined = []
    carry = ""

    for line in lines:
        if not line:
            if carry:
                rejoined.append(carry)
                carry = ""
            rejoined.append("")
            continue

        # If previous line ended with break char, join
        if carry:
            line = carry + line
            carry = ""

        # Check if this line ends with a break character
        if line and line[-1] in ("\u201e", "\u00ac"):  # „ or ¬
            carry = line[:-1]  # remove break char, carry forward
        elif re.search(r'\w[„¬]\s*$', line):
            # Break char before trailing whitespace
            carry = re.sub(r'[„¬]\s*$', '', line)
        else:
            rejoined.append(line)

    if carry:
        rejoined.append(carry)

    return rejoined


def normalize_chars(text):
    """Normalize special characters from HTR output."""
    replacements = {
        "\u201e": "",     # „ (low double quote used as hyphen)
        "\u00ac": "",     # ¬ (not sign used as hyphen)
        "\u0192": "f",    # ƒ (florin sign → f)
        "=": "",          # = in dates (15=e → 15e)
        "\u2018": "'",    # ' → '
        "\u2019": "'",    # ' → '
        "\u201c": '"',    # " → "
        "\u201d": '"',    # " → "
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def filter_noise(lines, min_length=3):
    """Filter out noise lines (very short, number-only, etc.)."""
    clean = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean.append("")
            continue
        if len(stripped) < min_length:
            continue
        if re.match(r'^\d+\.?\s*$', stripped):
            continue  # standalone page numbers
        clean.append(stripped)
    return clean


def segment_paragraphs(lines, min_paragraph_length=50):
    """Group lines into paragraphs based on blank-line boundaries."""
    paragraphs = []
    current = []

    for line in lines:
        if not line.strip():
            if current:
                para = " ".join(current)
                if len(para) >= min_paragraph_length:
                    paragraphs.append(para)
                current = []
        else:
            current.append(line.strip())

    if current:
        para = " ".join(current)
        if len(para) >= min_paragraph_length:
            paragraphs.append(para)

    return paragraphs


def compute_stats(raw_lines, clean_lines, paragraphs):
    """Compute preprocessing statistics."""
    return {
        "raw_lines": len(raw_lines),
        "after_metadata_strip": len([l for l in raw_lines if not l.strip().startswith("#+")]),
        "after_rejoin": len(clean_lines),
        "after_filter": len([l for l in clean_lines if l.strip()]),
        "paragraphs": len(paragraphs),
        "total_chars": sum(len(p) for p in paragraphs),
        "avg_paragraph_length": sum(len(p) for p in paragraphs) / max(len(paragraphs), 1),
        "total_words": sum(len(p.split()) for p in paragraphs),
    }


def preprocess_file(input_path):
    """Full preprocessing pipeline for a single VOC transcription file."""
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    # Pipeline
    lines = strip_metadata(raw_lines)
    lines = rejoin_broken_words(lines)
    lines = [normalize_chars(l) for l in lines]
    lines = filter_noise(lines)
    paragraphs = segment_paragraphs(lines)
    stats = compute_stats(raw_lines, lines, paragraphs)

    return lines, paragraphs, stats


def preprocess_directory(input_dir, output_dir):
    """Preprocess all VOC files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    total_paragraphs = 0

    txt_files = sorted(input_dir.glob("*.txt"))
    print(f"Preprocessing {len(txt_files)} VOC files...")

    for f in txt_files:
        lines, paragraphs, stats = preprocess_file(f)

        # Save cleaned lines
        out_lines = output_dir / f"clean_{f.name}"
        with open(out_lines, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        # Save paragraphs (one per line, ready for NER)
        out_paras = output_dir / f"paras_{f.name}"
        with open(out_paras, "w", encoding="utf-8") as fh:
            fh.write("\n".join(paragraphs))

        all_stats[f.name] = stats
        total_paragraphs += len(paragraphs)
        print(f"  {f.name}: {stats['raw_lines']} raw -> {stats['paragraphs']} paragraphs ({stats['total_words']} words)")

    # Save aggregate stats
    stats_file = output_dir / "preprocessing_stats.json"
    aggregate = {
        "total_files": len(txt_files),
        "total_paragraphs": total_paragraphs,
        "total_words": sum(s["total_words"] for s in all_stats.values()),
        "per_file": all_stats
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nDone. {total_paragraphs} paragraphs, {aggregate['total_words']} words total.")
    print(f"Stats saved to {stats_file}")
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Preprocess GLOBALISE VOC transcriptions")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", help="Output directory (required for directory mode)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        lines, paragraphs, stats = preprocess_file(input_path)
        print(json.dumps(stats, indent=2))
        print(f"\n--- {len(paragraphs)} paragraphs ---\n")
        for p in paragraphs[:10]:
            print(p[:200])
            print()
    elif input_path.is_dir():
        if not args.output:
            print("Error: --output required for directory mode")
            sys.exit(1)
        preprocess_directory(input_path, args.output)
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
