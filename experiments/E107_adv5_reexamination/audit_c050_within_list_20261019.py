"""C050 downstream audit (2026-10-19): does E107's "Mon-Khmer signature" belong to Iban's uncoded forms,
or to Iban as a language?

Post hoc audit check, not a new experiment. E107 scored every criterion by comparing C5 (Iban+Malay)
uncoded forms with *Sulawesi* uncoded forms, so any difference in the general phonotactics of the two
language groups counts as "Mon-Khmer". The comparison that bears on substrate is inside the list:
Iban uncoded vs Iban coded forms, same orthography, same compiler.

Reading rule, written before this script was run:
  - If Iban uncoded forms do NOT differ from Iban coded forms in the direction E107 predicted on at least
    3 of its 4 stated predictions (shorter, more consonant-final, fewer prefix-like strings, more
    consonant clusters), E107's verdict has no basis in its own data: the "signature" is Iban phonotactics.
  - If they do, E107's reading stays open (still unconfirmed: no Mon-Khmer comparandum was used).

Uses E107's own loader and feature function unchanged. Reads ABVD only; writes one JSON next to this file.
E107's stored results are not touched.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# stdout is wrapped as UTF-8 by the E107 module on import (wrapping twice closes the buffer)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adv5_reexamination as e107  # noqa: E402

PREDICTIONS = {  # E107 docstring: direction expected for uncoded forms if Mon-Khmer
    "syllables": "lower",
    "ends_vowel": "lower",          # = more consonant-final
    "has_an_prefix": "lower",
    "n_consonant_clusters": "higher",
}
OTHER = ["length", "vowel_ratio", "is_mk_shape", "has_glottal"]


def compare(a, b):
    out = {}
    for m in list(PREDICTIONS) + OTHER:
        va, vb = [r[m] for r in a], [r[m] for r in b]
        p = stats.mannwhitneyu(va, vb, alternative="two-sided").pvalue
        out[m] = {"uncoded": round(float(np.mean(va)), 4), "coded": round(float(np.mean(vb)), 4),
                  "p_mannwhitney": round(float(p), 6)}
        if m in PREDICTIONS:
            want = PREDICTIONS[m]
            moved = np.mean(va) < np.mean(vb) if want == "lower" else np.mean(va) > np.mean(vb)
            out[m]["e107_direction"] = want
            out[m]["in_e107_direction_and_p_lt_0.05"] = bool(moved and p < 0.05)
    return out


def main():
    c5 = e107.load_lang_data(e107.C5_LANGS)
    sul = e107.load_lang_data(e107.SULAWESI_LANGS)
    res = {"note": "post hoc audit check for ledger C050; see module docstring for the reading rule"}
    for lang in ("Iban", "Malay"):
        forms = [f for f in c5 if f["language"] == lang]
        unc = [f for f in forms if f["label"] == 0]
        cod = [f for f in forms if f["label"] == 1]
        res[lang] = {"n_forms": len(forms), "n_uncoded": len(unc), "n_coded": len(cod)}
        if len(unc) >= 10:
            res[lang]["uncoded_vs_coded"] = compare(unc, cod)
            res[lang]["predictions_met"] = sum(v.get("in_e107_direction_and_p_lt_0.05", False)
                                               for v in res[lang]["uncoded_vs_coded"].values())
    # Iban coded forms against Sulawesi uncoded forms, scored E107's way: if these also "look Mon-Khmer",
    # the criterion measures language, not uncodedness.
    iban_cod = [f for f in c5 if f["language"] == "Iban" and f["label"] == 1]
    sul_unc = [f for f in sul if f["label"] == 0]
    res["Iban_coded_vs_Sulawesi_uncoded"] = {
        m: {"iban_coded": round(float(np.mean([r[m] for r in iban_cod])), 4),
            "sulawesi_uncoded": round(float(np.mean([r[m] for r in sul_unc])), 4)}
        for m in list(PREDICTIONS) + OTHER}
    met = res["Iban"].get("predictions_met")
    res["reading"] = ("E107 verdict has no basis in its own data (signature = Iban phonotactics)"
                      if met is not None and met < 3 else "E107 reading stays open (unconfirmed)")
    (HERE / "results" / "audit_c050_within_list_20261019.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
