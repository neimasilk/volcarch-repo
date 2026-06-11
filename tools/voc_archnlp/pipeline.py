"""
VOC-ArchNLP End-to-End Pipeline Orchestrator
=============================================
Runs the full 4-stage pipeline:
  Stage 1: Preprocess HTR-transcribed VOC text
  Stage 2: Normalize colonial Dutch orthography
  Stage 3: Extract archaeological mention sentences
  Stage 4: Write structured output (CSV + summary JSON)

Usage:
    from voc_archnlp.pipeline import VOCArchPipeline
    pipeline = VOCArchPipeline()
    summary = pipeline.run(
        raw_dir="data/raw/globalise_voc/",
        output_dir="results/",
    )

    python pipeline.py --raw data/raw/globalise_voc/ --output results/
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "globalise_pipeline"))

try:
    from preprocess_voc import preprocess_directory
    from normalize_colonial_dutch import ColonialDutchNormalizer
except ImportError as e:
    raise ImportError(
        "Cannot import pipeline dependencies. "
        "Ensure tools/globalise_pipeline/ is present.\n"
        f"Details: {e}"
    )

from .extractor import ArchaeologicalMentionExtractor


class VOCArchPipeline:
    """Four-stage pipeline: preprocess → normalize → extract → output."""

    def __init__(self, context_window: int = 1, norm_level: str = "full"):
        """
        Args:
            context_window: sentences of context around each mention
            norm_level: normalization depth ('light', 'medium', 'full')
        """
        self.normalizer = ColonialDutchNormalizer()
        self.extractor = ArchaeologicalMentionExtractor(context_window=context_window)
        self.norm_level = norm_level

    def _normalize_directory(self, processed_dir: Path, normalized_dir: Path) -> int:
        """Apply colonial Dutch normalization to all preprocessed paragraph files."""
        normalized_dir.mkdir(parents=True, exist_ok=True)
        paras_files = sorted(processed_dir.glob("paras_*.txt"))
        count = 0

        for f in paras_files:
            with open(f, encoding="utf-8", errors="replace") as fh:
                text = fh.read()

            normalized = self.normalizer.normalize(text, level=self.norm_level)

            out_path = normalized_dir / f"norm_{f.name}"
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(normalized)
            count += 1

        print(f"  Normalized {count} paragraph files → {normalized_dir}")
        return count

    def run(
        self,
        raw_dir: str,
        output_dir: str,
        skip_preprocess: bool = False,
        processed_dir: str = None,
    ) -> dict:
        """Run the full pipeline.

        Args:
            raw_dir: directory with raw GLOBALISE *.txt files
            output_dir: root output directory
            skip_preprocess: if True, assume processed_dir already exists
            processed_dir: path to already-processed files (used when skip_preprocess=True)

        Returns: summary dict with stage statistics.
        """
        raw_dir = Path(raw_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        proc_dir = Path(processed_dir) if processed_dir else output_dir / "processed"
        norm_dir = output_dir / "normalized"
        results_dir = output_dir / "mentions"
        results_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "raw_dir": str(raw_dir),
            "output_dir": str(output_dir),
            "stages": {},
        }

        # Stage 1: Preprocess
        if skip_preprocess:
            print(f"Stage 1: SKIPPED (using existing {proc_dir})")
            raw_files = sorted(raw_dir.glob("*.txt"))
            summary["stages"]["preprocess"] = {
                "status": "skipped",
                "input_files": len(raw_files),
            }
        else:
            print(f"Stage 1: Preprocessing {raw_dir} ...")
            stats = preprocess_directory(str(raw_dir), str(proc_dir))
            summary["stages"]["preprocess"] = {
                "status": "done",
                "input_files": stats["total_files"],
                "total_paragraphs": stats["total_paragraphs"],
                "total_words": stats["total_words"],
            }
            print(
                f"  → {stats['total_files']} files, "
                f"{stats['total_paragraphs']} paragraphs, "
                f"{stats['total_words']} words"
            )

        # Stage 2: Normalize
        print(f"\nStage 2: Normalizing colonial Dutch ({self.norm_level} level) ...")
        n_normalized = self._normalize_directory(proc_dir, norm_dir)
        summary["stages"]["normalize"] = {
            "status": "done",
            "files_normalized": n_normalized,
            "level": self.norm_level,
        }

        # Stage 3: Extract
        print(f"\nStage 3: Extracting archaeological mentions ...")
        out_csv = results_dir / "voc_archaeological_mentions.csv"
        out_json = results_dir / "voc_archaeological_mentions.json"
        ext_summary = self.extractor.extract_from_directory(
            str(norm_dir),
            str(out_csv),
            output_json=str(out_json),
            glob_pattern="norm_paras_*.txt",
        )
        summary["stages"]["extract"] = ext_summary

        # Stage 4: Write master summary
        summary_path = output_dir / "pipeline_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nPipeline complete. Summary: {summary_path}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="VOC-ArchNLP: run full 4-stage archaeological mention extraction pipeline"
    )
    parser.add_argument("--raw", required=True, help="Raw GLOBALISE txt directory")
    parser.add_argument("--output", required=True, help="Output root directory")
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip preprocessing stage (use --processed-dir)"
    )
    parser.add_argument("--processed-dir", help="Path to already-preprocessed files")
    parser.add_argument(
        "--norm-level", default="full", choices=["light", "medium", "full"],
        help="Normalization depth. Default: full"
    )
    parser.add_argument(
        "--context", type=int, default=1,
        help="Context window for extraction. Default: 1"
    )
    args = parser.parse_args()

    pipeline = VOCArchPipeline(
        context_window=args.context,
        norm_level=args.norm_level,
    )
    pipeline.run(
        raw_dir=args.raw,
        output_dir=args.output,
        skip_preprocess=args.skip_preprocess,
        processed_dir=args.processed_dir,
    )


if __name__ == "__main__":
    main()
