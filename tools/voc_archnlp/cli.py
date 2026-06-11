"""
VOC-ArchNLP Unified Command-Line Interface
==========================================
Single entry point for all pipeline operations.

Commands:
  download    Fetch VOC transcriptions from GLOBALISE Dataverse
  preprocess  Clean HTR-transcribed text
  normalize   Map colonial Dutch spelling to modern Dutch
  extract     Extract archaeological mention sentences
  run         Full 4-stage pipeline (preprocess + normalize + extract)

Usage:
  python -m voc_archnlp download --n 500 --output data/raw/globalise_voc/
  python -m voc_archnlp preprocess --input data/raw/ --output data/processed/
  python -m voc_archnlp normalize --input data/processed/ --output data/normalized/
  python -m voc_archnlp extract --input data/normalized/ --output results/mentions.csv
  python -m voc_archnlp run --raw data/raw/ --output results/
"""

import argparse
import sys
from pathlib import Path

# Allow sibling pipeline tools to be found
sys.path.insert(0, str(Path(__file__).parent.parent / "globalise_pipeline"))


def cmd_download(args):
    from download_globalise import get_file_index, download_batch
    index = get_file_index(args.cache_dir)
    download_batch(
        index,
        args.output,
        n=args.n,
        inv_range=getattr(args, "range", None),
    )


def cmd_preprocess(args):
    from preprocess_voc import preprocess_directory, preprocess_file
    import json
    p = Path(args.input)
    if p.is_file():
        _, paragraphs, stats = preprocess_file(str(p))
        print(json.dumps(stats, indent=2))
        print(f"\n{len(paragraphs)} paragraphs:")
        for para in paragraphs[:5]:
            print(f"  {para[:120]}")
    else:
        if not args.output:
            print("Error: --output required for directory mode", file=sys.stderr)
            sys.exit(1)
        preprocess_directory(args.input, args.output)


def cmd_normalize(args):
    from normalize_colonial_dutch import ColonialDutchNormalizer
    import re
    norm = ColonialDutchNormalizer()
    input_path = Path(args.input)
    level = getattr(args, "level", "full")

    if input_path.is_file():
        with open(input_path, encoding="utf-8") as f:
            text = f.read()
        print(norm.normalize(text, level=level))
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(input_path.glob("*.txt")):
        with open(f, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        normalized = norm.normalize(text, level=level)
        out = output_dir / f"norm_{f.name}"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(normalized)
        count += 1
    print(f"Normalized {count} files -> {output_dir}")


def cmd_extract(args):
    from voc_archnlp.extractor import ArchaeologicalMentionExtractor
    ext = ArchaeologicalMentionExtractor(context_window=args.context)
    p = Path(args.input)
    if p.is_file():
        mentions = ext.extract_from_file(str(p))
        print(f"Found {len(mentions)} mentions in {p.name}")
        for m in mentions:
            print(f"  [{m['mention_types']}] {m['sentence_text'][:100]}")
    else:
        ext.extract_from_directory(
            str(p),
            args.output,
            output_json=getattr(args, "json_out", None),
            glob_pattern=args.glob,
        )


def cmd_run(args):
    from voc_archnlp.pipeline import VOCArchPipeline
    pipeline = VOCArchPipeline(
        context_window=args.context,
        norm_level=args.norm_level,
    )
    pipeline.run(
        raw_dir=args.raw,
        output_dir=args.output,
        skip_preprocess=args.skip_preprocess,
        processed_dir=getattr(args, "processed_dir", None),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voc_archnlp",
        description="VOC-ArchNLP: Dutch Colonial Archive Mining for Indonesian Archaeology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m voc_archnlp download --n 500\n"
            "  python -m voc_archnlp run --raw data/raw/ --output results/\n"
        ),
    )
    parser.add_argument("--version", action="version", version="VOC-ArchNLP 1.0.0")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # download
    p_dl = sub.add_parser("download", help="Fetch VOC files from GLOBALISE Dataverse")
    p_dl.add_argument("--n", type=int, default=50, help="Number of files to download")
    p_dl.add_argument("--range", help="Inventory range, e.g. 1053-1200")
    p_dl.add_argument("--output", default="data/raw/globalise_voc/")
    p_dl.add_argument("--cache-dir", default="tools/globalise_pipeline/")
    p_dl.set_defaults(func=cmd_download)

    # preprocess
    p_pre = sub.add_parser("preprocess", help="Clean HTR-transcribed VOC text")
    p_pre.add_argument("--input", required=True)
    p_pre.add_argument("--output")
    p_pre.set_defaults(func=cmd_preprocess)

    # normalize
    p_norm = sub.add_parser("normalize", help="Map colonial Dutch to modern Dutch")
    p_norm.add_argument("--input", required=True)
    p_norm.add_argument("--output")
    p_norm.add_argument("--level", default="full", choices=["light", "medium", "full"])
    p_norm.set_defaults(func=cmd_normalize)

    # extract
    p_ext = sub.add_parser("extract", help="Extract archaeological mention sentences")
    p_ext.add_argument("--input", required=True)
    p_ext.add_argument("--output", default="results/voc_mentions.csv")
    p_ext.add_argument("--json-out", dest="json_out", default=None)
    p_ext.add_argument("--context", type=int, default=1)
    p_ext.add_argument("--glob", default="paras_*.txt")
    p_ext.set_defaults(func=cmd_extract)

    # run (full pipeline)
    p_run = sub.add_parser("run", help="Full pipeline: preprocess → normalize → extract")
    p_run.add_argument("--raw", required=True, help="Raw GLOBALISE directory")
    p_run.add_argument("--output", required=True, help="Output root directory")
    p_run.add_argument("--skip-preprocess", action="store_true")
    p_run.add_argument("--processed-dir")
    p_run.add_argument("--norm-level", default="full", choices=["light", "medium", "full"])
    p_run.add_argument("--context", type=int, default=1)
    p_run.set_defaults(func=cmd_run)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
