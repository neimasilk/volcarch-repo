"""E228, correction of 2026-10-12 (POST HOC): how many uncoded Tolaki forms match a CODED Proto-Bungku-Tolaki entry?

01_revision_analyses.py stores "of_which_PBT_entry_is_cognate_coded_in_ABVD": 7 (results/S3_lookalikes.json). That
count is by MEANING: it asks whether any Proto-Bungku-Tolaki (ABVD list 780) entry of the meaning carries a cognate-set
number, not whether the entry that the Tolaki form matches does. The article wrote "seven match an entry that ABVD does
assign to a cognate set", which is a statement about the matched entry. This script counts that.

Same data, normalisation and distance as S3a (p8common.norm, p8common.d_stem; threshold 0.34).
Output: results/S3_pbt_matched_entry_recount.csv (one row per uncoded Tolaki form within 0.34 of a PBT entry).
Result of the run of 2026-10-12: 34 forms; by meaning 7; a nearest entry is coded for 6 ('to hear': Tolaki podea matches
the uncoded entry *podea°, the coded entry *rongoR is at 0.80); for one of the six, 'Eight', an uncoded entry (*hoalu)
is as near as the coded one (*halu), so every nearest entry is coded for 5.
"""
import csv
from pathlib import Path
import pandas as pd
import p8common as P

HERE = Path(__file__).resolve().parent
CLDF = HERE.parent / "E022_linguistic_subtraction" / "data" / "abvd" / "cldf"
fa = pd.read_csv(CLDF / "forms.csv", dtype=str, keep_default_na=False)
tol, pbt = fa[fa.Language_ID == "674"], fa[fa.Language_ID == "780"]
rows = []
for r in tol.itertuples():
    if r.Cognacy.strip() != "":
        continue                                   # uncoded Tolaki forms only
    n = P.norm(r.Form)
    if len(n) < 2:
        continue
    cands = [(P.d_stem(n, P.norm(q.Form)), q.Form, q.Cognacy.strip())
             for q in pbt[pbt.Parameter_ID == r.Parameter_ID].itertuples() if len(P.norm(q.Form)) >= 2]
    if not cands:
        continue
    dmin = min(c[0] for c in cands)
    if dmin > 0.34:
        continue
    best = [c for c in cands if abs(c[0] - dmin) < 1e-9]
    rows.append({"abvd_form_id": r.ID, "meaning": P.PNAME[r.Parameter_ID], "tolaki_form": r.Form, "distance": round(dmin, 3),
                 "nearest_pbt_entries": "; ".join(f"{b[1]} [{b[2] or 'no set'}]" for b in best),
                 "all_pbt_entries_of_the_meaning": "; ".join(f"{c[1]} [{c[2] or 'no set'}] {c[0]:.2f}" for c in cands),
                 "meaning_has_a_coded_pbt_entry": int(any(c[2] for c in cands)),
                 "a_nearest_entry_is_coded": int(any(b[2] for b in best)),
                 "every_nearest_entry_is_coded": int(all(b[2] for b in best))})
out = HERE / "results" / "S3_pbt_matched_entry_recount.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
n = len(rows)
by_meaning = sum(r["meaning_has_a_coded_pbt_entry"] for r in rows)
by_entry = sum(r["a_nearest_entry_is_coded"] for r in rows)
strict = sum(r["every_nearest_entry_is_coded"] for r in rows)
print(f"uncoded Tolaki forms within 0.34 of a PBT entry: {n}; by meaning: {by_meaning}; a nearest entry coded: {by_entry}; every nearest entry coded: {strict}")
assert (n, by_meaning, by_entry, strict) == (34, 7, 6, 5), (n, by_meaning, by_entry, strict)
for r in rows:
    if r["meaning_has_a_coded_pbt_entry"]:
        print("  ", r["meaning"], "|", r["tolaki_form"], "|", r["distance"], "|", r["nearest_pbt_entries"])
