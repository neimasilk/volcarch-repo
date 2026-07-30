# E216 — Zenodo Deposit Package (SIG G7 reproducibility)

**Status:** SKELETON ONLY. Not yet uploaded. Upload is human-gated (Pak Amien's Zenodo
account) — see `../SUBMISSION_CHECKLIST.md` item 3.

## What goes in this package

Copy (not move) the following into this folder before zipping for upload:

```
zenodo_upload/
├── README.md                              (this file → becomes the deposit description)
├── code/
│   ├── e216_detection_function.py         (S3-S8 forward model, fixed 2026-07-07)
│   ├── e216_sensitivity_sweep.py          (parameter sensitivity sweep, NEW 2026-07-07)
│   └── e216_figure.py                     (figure generation, fixed 2026-07-07)
├── PREREG.md                               (locked pre-registration, timestamped 2026-06-25)
├── OPUS_REVIEW_20260625.md                 (cross-model review that drove the fixes)
├── results/
│   ├── core_coverage_table.csv
│   ├── detection_probability_table.csv
│   ├── missing_core_corner_table.csv       (NEW)
│   ├── sensitivity_network_detection.csv   (NEW)
│   ├── sensitivity_missing_core_corners.csv (NEW)
│   ├── sensitivity_summary.json            (NEW)
│   ├── OUTCOME.json
│   └── missing_core_spec.json
└── PAPER_DRAFT_OUTLINE.md
```

## Mini-G1 blind recompute (do before upload)

Per Submission Integrity Gate G1: before uploading, have someone who did NOT write the
code (ideally the palynologist co-author, or a fresh Claude session with no memory of
this one) re-run `e216_detection_function.py` then `e216_sensitivity_sweep.py` from a
clean checkout and confirm the outputs match `results/OUTCOME.json` and
`results/sensitivity_summary.json` bit-for-bit (or within floating-point tolerance).
Record the result (pass/fail + any discrepancy) in a new file
`MINI_G1_BLIND_RECOMPUTE_<date>.md` in this folder before upload.

## Suggested Zenodo metadata

- **Title:** "E216: A Pre-Registered Palaeoecological Detection-Power Test of the
  Pre-400 CE Population Hypothesis in Volcanic Java (Code + Data)"
- **Upload type:** Software / Dataset (dual — Zenodo allows one type per deposit;
  Software is likely correct since the scientific contribution is the detection
  function + sensitivity sweep, not a novel primary dataset)
- **License:** CC-BY 4.0 (matches project convention, e.g. VOC-ArchNLP)
- **Keywords:** palaeoecology, detection power, REVEALS, pre-registration, Java,
  archaeology, palynology, Michelson-Morley falsification design
- **Related identifiers:** link to E214 (palynology SLR) and E196 (population model)
  experiment folders in the parent GitHub repo, once public

## Sequencing reminder

This package should be finalized and uploaded AFTER the palynologist co-author (G2/G10)
has reviewed the method, not before — their review may change parameter choices,
which would mean re-running the sweep and invalidating an earlier deposit. See
`../SUBMISSION_CHECKLIST.md` for the full sequencing.
