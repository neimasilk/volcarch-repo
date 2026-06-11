"""
Archaeological Mention Extractor for VOC Colonial Dutch Text
============================================================
Core new component of VOC-ArchNLP v1.0.

Extracts sentences that mention archaeological features from preprocessed
VOC dagregister transcriptions. Supports six mention types:
  MONUMENT  — temples, shrines, statues (tempel, candi, arca, ...)
  GRAVE     — burial contexts (graf, begraven, ...)
  RUIN      — collapsed structures (ruïne, puing, vervallen, ...)
  ARTIFACT  — portable objects (beeld, penning, inscriptie, ...)
  DEPTH     — depth measurements (n voet onder de grond, ...)
  INSCRIPTION — inscribed objects (inscriptie, opschrift, prasasti, ...)

Output format (CSV):
  source_file, sentence_id, sentence_text, mention_types, keywords_found,
  depth_value_m, context_before, context_after

Usage:
    from voc_archnlp.extractor import ArchaeologicalMentionExtractor
    ext = ArchaeologicalMentionExtractor()
    mentions = ext.extract_from_file("path/to/clean_1053.txt")
    ext.extract_from_directory("data/processed/", "results/mentions.csv")

    python extractor.py --input data/processed/ --output results/mentions.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --- Keyword Lexicons ---

MONUMENT_KEYWORDS = [
    r"\bcandi\b", r"\btjandi\b", r"\bkraton\b", r"\bkraton\b",
    r"\btempel\b", r"\bpagode\b", r"\bheiligdom\b", r"\bpura\b",
    r"\bdagob\b", r"\bstupa\b", r"\bwihara\b",
    r"\barca\b", r"\bstandbeeld\b", r"\bgodenbeeld\b",
    r"\blingga\b", r"\byoni\b", r"\bgaṇeśa\b", r"\bganesa\b",
    r"\boutpost\b",  # colonial watch posts = often near monuments
]

GRAVE_KEYWORDS = [
    r"\bgraf\b", r"\bgraven\b", r"\bbegrafenis\b",
    r"\bbegraven\b", r"\bbegrafplaats\b",
    r"\bgrafkuil\b", r"\bgrafkamer\b",
    r"\bkubur\b", r"\bmakam\b",
    r"\bsarcophaag\b",
]

RUIN_KEYWORDS = [
    r"\bruïne\b", r"\bruine\b", r"\bpuinhoop\b", r"\bpuing\b",
    r"\bvervallen\b", r"\binstort\b", r"\binstorting\b",
    r"\boverwoekerd\b", r"\boverdekt\b",
    r"\bresten\b.*\b(gebouw|steen|muur)\b",
    r"\b(gebouw|muur|steen)\b.*\bresten\b",
    r"\boutgravingen\b", r"\bopgravingen\b",
]

ARTIFACT_KEYWORDS = [
    r"\boud(heden|heid|heidkund)\b",
    r"\boudheidkundig\b",
    r"\bantiek\b", r"\bantika\b",
    r"\bpenning\b", r"\bpenningen\b",
    r"\bschat\b", r"\bschatten\b",
    r"\bgoud\b.*\b(oud|antiek|gevonden)\b",
    r"\bkoper\b.*\b(oud|antiek|gevonden)\b",
    r"\bbeelden\b", r"\bbeeldhouwwerk\b",
]

INSCRIPTION_KEYWORDS = [
    r"\binscriptie\b", r"\binscripties\b",
    r"\bopschrift\b", r"\bopschriften\b",
    r"\bprasasti\b",
    r"\bletters\b.*\b(steen|klip|rots)\b",
    r"\bgegraveerd\b",
    r"\bhiëroglyfen\b", r"\bschriftteken\b",
]

DEPTH_PATTERN = re.compile(
    r"(\d+[\.,]?\d*)\s*(voet|voeten|el|ellen|palm|palmen|duim|duimen|meter|meters|m)\s*"
    r"(onder|diep|diepte|beneden|begraven|diepte|verborgen)",
    re.IGNORECASE
)

# Dutch measurement unit → metres
UNIT_TO_METRES = {
    "voet": 0.3048,
    "voeten": 0.3048,
    "el": 0.6858,     # Rijnlandse el
    "ellen": 0.6858,
    "palm": 0.1,
    "palmen": 0.1,
    "duim": 0.0254,
    "duimen": 0.0254,
    "meter": 1.0,
    "meters": 1.0,
    "m": 1.0,
}

MENTION_TYPE_PATTERNS = {
    "MONUMENT": MONUMENT_KEYWORDS,
    "GRAVE": GRAVE_KEYWORDS,
    "RUIN": RUIN_KEYWORDS,
    "ARTIFACT": ARTIFACT_KEYWORDS,
    "INSCRIPTION": INSCRIPTION_KEYWORDS,
}


class ArchaeologicalMentionExtractor:
    """Extract archaeological mention sentences from preprocessed VOC text."""

    def __init__(self, context_window: int = 1):
        """
        Args:
            context_window: number of surrounding sentences to include as context
        """
        self.context_window = context_window
        self._compile_patterns()

    def _compile_patterns(self):
        self.compiled = {}
        for mtype, patterns in MENTION_TYPE_PATTERNS.items():
            self.compiled[mtype] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def _detect_types(self, sentence: str) -> Tuple[List[str], List[str]]:
        """Return (mention_types, keywords_found) for a sentence."""
        types_found = []
        keywords_found = []
        for mtype, patterns in self.compiled.items():
            for pat in patterns:
                m = pat.search(sentence)
                if m:
                    if mtype not in types_found:
                        types_found.append(mtype)
                    kw = m.group(0).strip()
                    if kw not in keywords_found:
                        keywords_found.append(kw)
        return types_found, keywords_found

    def _parse_depth(self, sentence: str) -> Optional[float]:
        """Extract depth value in metres from sentence, or None."""
        m = DEPTH_PATTERN.search(sentence)
        if not m:
            return None
        val_str = m.group(1).replace(",", ".")
        unit = m.group(2).lower()
        try:
            val = float(val_str)
            return round(val * UNIT_TO_METRES.get(unit, 1.0), 3)
        except ValueError:
            return None

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter for Dutch text."""
        # Split on ". " or ".\n" followed by uppercase, or on newlines
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZA-ZÀÁÂÄÆÃÅĀ])', text)
        # Also split on paragraph breaks
        result = []
        for sent in sentences:
            parts = sent.split("\n")
            result.extend(p.strip() for p in parts if len(p.strip()) > 20)
        return result

    def extract_from_text(self, text: str, source_file: str = "") -> List[Dict]:
        """Extract archaeological mentions from a text string.

        Returns list of mention dicts, one per matching sentence.
        """
        sentences = self._split_sentences(text)
        mentions = []

        for i, sent in enumerate(sentences):
            types_found, keywords_found = self._detect_types(sent)
            if not types_found:
                continue

            # Context window
            ctx_before = " | ".join(sentences[max(0, i - self.context_window):i])
            ctx_after = " | ".join(sentences[i + 1:i + 1 + self.context_window])

            depth_m = self._parse_depth(sent)
            if depth_m is None:
                # Check context too
                for ctx in [ctx_before, ctx_after]:
                    depth_m = self._parse_depth(ctx)
                    if depth_m is not None:
                        break

            mentions.append({
                "source_file": source_file,
                "sentence_id": i,
                "sentence_text": sent,
                "mention_types": "|".join(types_found),
                "keywords_found": "|".join(keywords_found),
                "depth_value_m": depth_m if depth_m is not None else "",
                "context_before": ctx_before,
                "context_after": ctx_after,
            })

        return mentions

    def extract_from_file(self, file_path: str) -> List[Dict]:
        """Extract mentions from a single preprocessed file."""
        path = Path(file_path)
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        return self.extract_from_text(text, source_file=path.name)

    def extract_from_directory(
        self,
        input_dir: str,
        output_csv: str,
        output_json: Optional[str] = None,
        glob_pattern: str = "paras_*.txt",
    ) -> Dict:
        """Extract mentions from all preprocessed files in a directory.

        Args:
            input_dir: directory with preprocessed VOC files (paras_*.txt)
            output_csv: path for CSV output
            output_json: optional path for JSON output
            glob_pattern: file glob to match (default: paragraph files)

        Returns dict with summary statistics.
        """
        input_dir = Path(input_dir)
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob(glob_pattern))
        if not files:
            # Fallback: any txt file
            files = sorted(input_dir.glob("*.txt"))

        print(f"Extracting from {len(files)} files in {input_dir} ...")

        all_mentions = []
        type_counts = {}

        for f in files:
            mentions = self.extract_from_file(str(f))
            all_mentions.extend(mentions)
            for m in mentions:
                for t in m["mention_types"].split("|"):
                    type_counts[t] = type_counts.get(t, 0) + 1
            if mentions:
                print(f"  {f.name}: {len(mentions)} mentions")

        # Write CSV
        fieldnames = [
            "source_file", "sentence_id", "sentence_text",
            "mention_types", "keywords_found", "depth_value_m",
            "context_before", "context_after",
        ]
        with open(output_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_mentions)

        # Optional JSON
        if output_json:
            with open(output_json, "w", encoding="utf-8") as fh:
                json.dump(all_mentions, fh, ensure_ascii=False, indent=2)

        summary = {
            "files_processed": len(files),
            "total_mentions": len(all_mentions),
            "mentions_with_depth": sum(1 for m in all_mentions if m["depth_value_m"] != ""),
            "type_distribution": type_counts,
            "output_csv": str(output_csv),
        }

        print(f"\nExtraction complete.")
        print(f"  Total mentions: {summary['total_mentions']}")
        print(f"  With depth: {summary['mentions_with_depth']}")
        print(f"  By type: {type_counts}")
        print(f"  CSV: {output_csv}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract archaeological mentions from preprocessed VOC text"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input file or directory of preprocessed VOC files"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--json", dest="output_json", default=None,
        help="Optional JSON output path"
    )
    parser.add_argument(
        "--context", type=int, default=1,
        help="Context window (sentences around mention). Default: 1"
    )
    parser.add_argument(
        "--glob", default="paras_*.txt",
        help="Glob pattern for input files. Default: paras_*.txt"
    )
    args = parser.parse_args()

    extractor = ArchaeologicalMentionExtractor(context_window=args.context)
    input_path = Path(args.input)

    if input_path.is_file():
        mentions = extractor.extract_from_file(str(input_path))
        print(f"Found {len(mentions)} mentions in {input_path.name}")
        for m in mentions:
            print(f"  [{m['mention_types']}] {m['sentence_text'][:100]}")
        sys.exit(0)

    extractor.extract_from_directory(
        str(input_path),
        args.output,
        output_json=args.output_json,
        glob_pattern=args.glob,
    )


if __name__ == "__main__":
    main()
