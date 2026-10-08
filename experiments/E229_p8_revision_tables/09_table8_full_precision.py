"""E229 amendment A11 (2026-10-08): the numbers of Table 8 (glottal conventions) at full precision.

Why: E228 stores the held-out-list mean rounded to four decimals and the per-list values rounded to three. The article
builder took the mean of the three-decimal per-list values, so two cells of Table 8 printed a difference of -0.019 where
the unrounded difference is -0.0197 (the text of the article says 0.020, correctly). A difference at three decimals
needs the unrounded means; this script re-runs section S1 of E228 01_revision_analyses.py for the 25-input model with
the same functions and seeds, checks every value against the stored E228 file at four decimals, and writes the
unrounded values. Nothing else is computed and no E228 file is touched.

Run:  python experiments/E229_p8_revision_tables/09_table8_full_precision.py     (about two minutes)
Output: results/T8_glottal_full_precision.csv
"""
import csv, io, json, re, sys, warnings
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
E228 = HERE.parent / "E228_p8_revision_analyses"
sys.path.insert(0, str(E228))
import p8common as P   # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

f = P.load_lists(P.TARGET)
y = f.coded.values
langs = f.language.values
F0 = P.featurize(f.form, f.concept, f.language)
assert len(f) == 1357 and int(f.candidate.sum()) == 438
MARK = "ʔ'"


def geminate(s):   # as in E228 S1: every mark is deleted and a following consonant letter is doubled
    out = []
    for i, ch in enumerate(s):
        if ch in MARK:
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if nxt and nxt.isalpha() and nxt.lower() not in P.VOWELS:
                out.append(nxt)
        else:
            out.append(ch)
    return "".join(out)


CONV = {
    "V0_as_published": lambda s: s,
    "V1_q": lambda s: re.sub("[ʔ']", "q", s),
    "V2_k": lambda s: re.sub("[ʔ']", "k", s),
    "V3_unwritten": lambda s: re.sub("[ʔ']", "", s),
    "V4_geminate": geminate,
    "V5_all_as_glottal_letter": lambda s: s.replace("'", "ʔ"),
}
res = {}
for name, fn in CONV.items():
    forms_v = f.form.map(fn)
    forms_v = forms_v.where(forms_v != "", f.form)
    Fv = P.featurize(forms_v, f.concept, f.language)
    a, sd, _ = P.cv_auc(Fv[P.PURE25].values, y)
    lo, _ = P.lolo_auc(Fv[P.PURE25].values, y, langs)
    res[name] = {"forms_changed": int((forms_v.values != f.form.values).sum()), "cv": float(a),
                 "lolo": float(np.mean(list(lo.values())))}
    print(name, res[name], flush=True)
c2 = [c for c in P.PURE25 if c != "has_glottal"]
a, sd, _ = P.cv_auc(F0[c2].values, y)
lo, _ = P.lolo_auc(F0[c2].values, y, langs)
res["V6_feature_removed"] = {"forms_changed": 0, "cv": float(a), "lolo": float(np.mean(list(lo.values())))}
print("V6_feature_removed", res["V6_feature_removed"], flush=True)

stored = json.load(open(E228 / "results" / "S1_glottal_conventions.json", encoding="utf-8"))
base = res["V0_as_published"]
rows = []
for k, v in res.items():
    s = stored[k]
    assert round(v["cv"], 4) == s["cv_auc_25"] and round(v["lolo"], 4) == s["lolo_mean_25"] and v["forms_changed"] == s["forms_changed"], \
        f"{k}: re-run differs from the stored E228 values"
    rows.append({"convention": k, "forms_changed": v["forms_changed"], "cv_auc_25": f"{v['cv']:.6f}",
                 "delta_cv_25": f"{v['cv'] - base['cv']:.6f}", "lolo_mean_25": f"{v['lolo']:.6f}",
                 "delta_lolo_25": f"{v['lolo'] - base['lolo']:.6f}"})
with open(HERE / "results" / "T8_glottal_full_precision.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nall seven rows agree with E228 S1_glottal_conventions.json at four decimals")
for r in rows:
    print(r)
