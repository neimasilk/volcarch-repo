"""
E229 note N1 (DESIGN amendment A6, POST HOC, descriptive) -- which lists have consonant-final forms at all.

Why: in Figure 1 the input "ends in a vowel" points toward *candidate*, while inside the lists vowel-final forms
are less often candidates (T6, OR 0.66). The two can only differ if the input also says something about which
list a form comes from. This script counts, per list, the vowel-final and consonant-final forms and how many of
the latter end in a glottal marker. It tests nothing.

Run:  PYTHONIOENCODING=utf-8 python experiments/E229_p8_revision_tables/04_final_consonant_by_list.py
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
rows = []
for L in list(P.TARGET.values()) + ["ALL"]:
    m = (f.language == L) if L != "ALL" else pd.Series(True, index=f.index)
    cons = m & (X.ends_in_vowel == 0)
    vow = m & (X.ends_in_vowel == 1)
    glot = cons & f.form.str.endswith(("ʔ", "'"))
    rows.append({"list": L, "forms": int(m.sum()), "vowel_final": int(vow.sum()), "consonant_final": int(cons.sum()),
                 "of_which_final_glottal_marker": int(glot.sum()),
                 "candidate_share_vowel_final": round(float(f.candidate[vow].mean()), 3) if vow.sum() else None,
                 "candidate_share_consonant_final": round(float(f.candidate[cons].mean()), 3) if cons.sum() else None})
out = pd.DataFrame(rows)
out.to_csv(HERE / "results" / "N1_final_consonant_by_list.csv", index=False, encoding="utf-8")
print(out.to_string(index=False))
