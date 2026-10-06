"""
E229 note N2 (DESIGN amendment A7, POST HOC, descriptive) -- do forms for action meanings look different?

Why: the submitted discussion (tex 504-506) reads the excess of action meanings among candidates as a property of
the vocabulary ("verbs resist replacement"). T6 and T7 suggest another reading: verbs are cited in longer, affixed
forms. This script only tabulates, per list, the share of candidates, the mean length and the share of
prefix-like onsets for action meanings and for all other meanings. It tests nothing.

Run:  PYTHONIOENCODING=utf-8 python experiments/E229_p8_revision_tables/05_action_meaning_by_form.py
"""
import io
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E228_p8_revision_analyses"))
import p8common as P  # noqa: E402

f = P.load_lists(P.TARGET)
X = P.featurize(f.form, f.concept)
X["candidate"] = f.candidate.values
X["list"] = f.language.values
rows = []
for L in list(P.TARGET.values()) + ["ALL"]:
    m = (X.list == L) if L != "ALL" else pd.Series(True, index=X.index)
    for act, name in ((1, "action meanings"), (0, "other meanings")):
        g = X[m & (X.sem_ACTION == act)]
        rows.append({"list": L, "meanings": name, "forms": len(g), "candidate_share": round(g.candidate.mean(), 3),
                     "mean_length": round(g.form_length.mean(), 2), "prefix_like_share": round(g.has_prefix_like.mean(), 3),
                     "begins_with_a_share": round(g.init_a.mean(), 3),
                     "mean_consonant_letter_clusters": round(g.n_consonant_clusters.mean(), 3)})
out = pd.DataFrame(rows)
out.to_csv(HERE / "results" / "N2_action_meaning_by_form.csv", index=False, encoding="utf-8")
print(out.to_string(index=False))
