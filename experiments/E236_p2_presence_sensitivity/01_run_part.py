"""E236 part runner: re-runs one real-data analysis of P2 on the cleaned presence set (DESIGN.md section 4).

    python 01_run_part.py A    # E217b  common-background benchmark (5 seeds)
    python 01_run_part.py B    # E218   stage A only: 3 training designs x 4 evaluation backgrounds x 20 seeds
    python 01_run_part.py C    # E218b  hard-negative sweep (5 seeds x 11 values)
    python 01_run_part.py D    # E221   10-seed maps, priority sets, densities (maps kept out of git)
    python 01_run_part.py E    # E235   near - far suitability on the part-D maps

The original scripts are imported unchanged; only their output folders are redirected into results/<part>_*, so
none of the original results files can be overwritten.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import clean_sites  # noqa: E402

clean_sites.patch()
RES = HERE / "results"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def part_a():
    m = load("e236_e217b", EXP / "E217_maxent_benchmark" / "02_matched_evaluation.py")
    m.RESULTS_DIR = RES / "A_e217b"
    m.main()


def part_b():
    m = load("e236_e218", EXP / "E218_evaluation_artefact" / "01_artefact_robustness.py")
    m.RESULTS_DIR = RES / "B_e218a"
    m.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pres, frame_all, frame_buf, designs, prop, n_pa = m.prepare(decimate=m.base.DECIMATE)
    print(f"  presences={len(pres)}  frame={len(frame_all):,}  background target={n_pa}")
    m.stage_a(pres, frame_all, frame_buf, designs, prop, n_pa)        # writes e218_stageA_raw.csv


def part_c():
    m = load("e236_e218b", EXP / "E218_evaluation_artefact" / "02_mechanism_hardfrac.py")
    m.RESULTS_DIR = RES / "C_e218b"
    m.main()


def part_d():
    m = load("e236_e221", EXP / "E221_seed_ensemble_stability" / "01_seed_stability.py")
    m.RESULTS_DIR = RES / "D_e221"
    m.MAPS_DIR = m.RESULTS_DIR / "maps"
    m.main()


def part_e():
    import numpy as np
    m = load("e236_e235", EXP / "E235_elevation_vs_volcanic_control" / "01_elevation_vs_volcanic.py")
    m.RESULTS_DIR = RES / "E_e235"
    m.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    src = str(Path(EXP / "E221_seed_ensemble_stability" / "results").resolve())
    dst = RES / "D_e221"
    original = np.load

    def redirected(path, *a, **k):          # E235 reads the E221 priority sets by a fixed path
        p = Path(path).resolve()
        if str(p.parent) == src and p.name.startswith("e221_priority_sets_"):
            p = dst / p.name
            print(f"  [E236] reading {p}")
        return original(p, *a, **k)

    np.load = redirected
    try:
        m.main()
    finally:
        np.load = original


if __name__ == "__main__":
    part = sys.argv[1].upper()
    t0 = time.time()
    {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e}[part]()
    print(f"\n[E236] part {part} finished in {(time.time() - t0) / 60:.1f} min")
