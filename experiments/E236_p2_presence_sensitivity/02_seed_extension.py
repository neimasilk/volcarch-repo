"""E236 declared extension (EXTENSION_SEEDS_20261102.md): claim A on ten seeds, cleaned presence set.

Runs the unchanged E217b script through the same site-file wrapper as 01_run_part.py, with N_SEEDS = 10,
writing to results/A_seeds10/ -- results/A_e217b/ (the pre-registered run) is never touched.
    python experiments/E236_p2_presence_sensitivity/02_seed_extension.py
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clean_sites  # noqa: E402

clean_sites.patch()
OUT = HERE / "results" / "A_seeds10"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    assert not (OUT / "e217b_raw_results.csv").exists(), "extension already ran; move the old output first"
    m = load("e236_e217b_seeds10", HERE.parent / "E217_maxent_benchmark" / "02_matched_evaluation.py")
    m.N_SEEDS = 10
    m.RESULTS_DIR = OUT
    m.main()


if __name__ == "__main__":
    main()
