"""
E229 -- cell-by-cell comparison of the written tables with the independent re-derivation.

Why this third script exists: `02_independent_check.py` was written (by a second agent that never read
`01_tables.py`) while the first run had stopped on a mis-ordered anchor (DESIGN amendment A1), so no table
file existed yet and most of its comparisons came out "NOT COMPARABLE". Its *values* are complete, though:
T1 from the raw forms.csv, and the XGBoost cross-validation and leave-one-language-out numbers from the
stored feature matrix of March 2026, with its own loop and metric code. This script only does the diff
between those stored values (`results/independent_check.json`) and the CSVs that `01_tables.py` wrote after
the anchor was corrected. It computes nothing new.

SD convention: the independent script used the sample SD (ddof = 1), the tables the population SD (ddof = 0;
DESIGN amendment A5). The independent SDs are converted with sqrt((n-1)/n) before comparing.

Run:  PYTHONIOENCODING=utf-8 python experiments/E229_p8_revision_tables/03_compare_with_independent.py
"""
import io
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
R = Path(__file__).resolve().parent / "results"
ind = json.loads((R / "independent_check.json").read_text(encoding="utf-8"))
t1i, cvi, loi = ind["t1_independent"], ind["independent_stored_order"]["cv"], ind["independent_stored_order"]["lolo"]
T1 = pd.read_csv(R / "T1_label_by_list.csv").set_index("list_key")
T2 = pd.read_csv(R / "T2_cv_performance.csv")
T3 = pd.read_csv(R / "T3_lolo.csv")

rows = []


def add(q, a, b, tol):
    d = abs(float(a) - float(b))
    rows.append({"quantity": q, "e229": a, "independent": b, "abs_diff": d, "tolerance": tol,
                 "verdict": "AGREE" if d <= tol else "DISAGREE"})


# T1: counts must be exact, percentages are printed to one decimal
for key, ikey in [(k, k) for k in ("Muna", "Bugis", "Makassar", "Wolio", "Toraja-Sadan", "Tolaki")] + [("ALL", "TOTAL")]:
    for c, ic, tol in (("forms", "forms", 0), ("coded_n", "coded", 0), ("candidate_n", "candidate", 0),
                       ("coded_pct", "pct_coded", 0.05), ("candidate_pct", "pct_candidate", 0.05)):
        add(f"T1 {key} {c}", T1.loc[key, c], t1i[ikey][ic], tol)

# T2, XGBoost rows: six metric means, both spreads, the always-coded accuracy
MET = {"auc": "auc", "accuracy": "accuracy", "f1_candidate": "f1_cand", "precision_candidate": "prec_cand",
       "recall_candidate": "rec_cand", "f1_coded": "f1_coded"}
for s in ("FM25", "F17"):
    r = T2[(T2.input_set == s) & (T2.classifier == "XGBoost")].iloc[0]
    for m, im in MET.items():
        add(f"T2 {s} XGBoost {m} mean", r[f"{m}_mean"], cvi[s][im]["mean"], 0.0005)
        add(f"T2 {s} XGBoost {m} SD of 10 seed means (ddof 0)", r[f"{m}_sd_seed_means"],
            cvi[s][im]["sd_seed_means"] * math.sqrt(9 / 10), 0.0005)
        add(f"T2 {s} XGBoost {m} SD of 50 folds (ddof 0)", r[f"{m}_sd_folds"],
            cvi[s][im]["sd_folds"] * math.sqrt(49 / 50), 0.0005)
    add(f"T2 {s} always-coded accuracy", r["accuracy_always_coded"], cvi[s]["always_coded"]["mean"], 0.0005)

# T3: every per-list cell, and the summary row
for s in ("FM25", "F17"):
    aucs = []
    for L in ("Muna", "Bugis", "Makassar", "Wolio", "Toraja-Sadan", "Tolaki"):
        r = T3[(T3.input_set == s) & (T3.list_key == L)].iloc[0]
        i = loi[s][L]
        aucs.append(i["auc"])
        for c, ic, tol in (("n", "n", 0), ("n_candidates", "n_cand", 0), ("auc", "auc", 0.0005),
                           ("accuracy", "accuracy", 0.0005), ("accuracy_majority_answer", "majority_acc", 0.0005),
                           ("accuracy_always_coded", "always_coded", 0.0005), ("f1_candidate", "f1_cand", 0.0005)):
            add(f"T3 {s} {L} {c}", r[c], i[ic], tol)
    sm = T3[(T3.input_set == s) & (T3.list_key == "SUMMARY")].iloc[0]
    mean = sum(aucs) / 6
    sd0 = math.sqrt(sum((a - mean) ** 2 for a in aucs) / 6)
    add(f"T3 {s} mean AUC over lists", sm["auc"], mean, 0.0005)
    add(f"T3 {s} SD of AUC over lists (ddof 0)", sm["auc_sd_over_lists"], sd0, 0.0005)
    add(f"T3 {s} lists with AUC >= 0.65", sm["n_lists_auc_ge_0.65"], sum(a >= 0.65 for a in aucs), 0)

out = pd.DataFrame(rows)
out.to_csv(R / "independent_comparison.csv", index=False, encoding="utf-8")
n_dis = int((out.verdict == "DISAGREE").sum())
print(f"{len(out)} cells compared; {n_dis} disagree; largest difference in a non-count cell: "
      f"{out[out.tolerance > 0].abs_diff.max():.2e}")
if n_dis:
    print(out[out.verdict == "DISAGREE"].to_string(index=False))
    sys.exit(1)
print("Not covered by the independent re-derivation: Random Forest and logistic-regression rows of T2, T4, T5, T6, T7, F1 "
      "(these rest on the anchors of 01_tables.py against E227/E228).")
