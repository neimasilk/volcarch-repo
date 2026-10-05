"""
E228 S7 (DESIGN.md amendment A1) — what each group of inputs contributes.

Run:  python experiments/E228_p8_revision_analyses/02_feature_groups.py
"""
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import p8common as P

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "results"

f = P.load_lists(P.TARGET)
y, langs = f.coded.values, f.language.values
F0 = P.featurize(f.form, f.concept, f.language)

GROUPS = {
    "form_only_17": P.PHON + P.INIT,
    "meaning_only_8": ["is_core_vocab"] + P.SEM,
    "language_only_1": ["language_id_encoded"],
    "form_plus_meaning_25": P.PURE25,
    "form_meaning_identity_26": P.ABL26,
}
res = {}
for name, cols in GROUPS.items():
    auc, sd, _ = P.cv_auc(F0[cols].values, y)
    row = {"n_inputs": len(cols), "cv_auc": round(auc, 4), "cv_sd_seed_means": round(sd, 4)}
    if name != "language_only_1":
        lo, _ = P.lolo_auc(F0[cols].values, y, langs)
        row["lolo_mean_auc"] = round(float(np.mean(list(lo.values()))), 4)
        row["lolo_per_language"] = {k: round(v, 3) for k, v in lo.items()}
        row["lolo_languages_ge_0.65"] = int(sum(v >= 0.65 for v in lo.values()))
    res[name] = row
    print(name, row, flush=True)

fo, mo = res["form_only_17"]["cv_auc"], res["meaning_only_8"]["cv_auc"]
res["_decision"] = {
    "form_only_cv_auc": fo,
    "phonology_alone": "weak signal (< 0.65): downgrade" if fo < 0.65 else "at or above the manuscript's 0.65 line",
    "meaning_vs_form": "meaning-only >= form-only: say so in the revision" if mo >= fo else "form-only > meaning-only",
}
print(res["_decision"])
(OUT / "S7_feature_groups.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
