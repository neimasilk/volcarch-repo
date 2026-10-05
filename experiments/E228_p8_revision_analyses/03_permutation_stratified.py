"""
E228 S5x (DESIGN.md amendment A2, POST HOC / exploratory) — stratified permutation.

Run:  python experiments/E228_p8_revision_analyses/03_permutation_stratified.py
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import p8common as P

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parent / "results"
SEED = 228

f = P.load_lists(P.TARGET)
F0 = P.featurize(f.form, f.concept)
sub = f[(f.candidate == 1) & (~f.concept.isin(P.NUMERALS))].reset_index(drop=True)
dom = F0.semantic_domain[(f.candidate == 1).values & (~f.concept.isin(P.NUMERALS)).values].reset_index(drop=True)
nf = [P.norm(x) for x in sub.form]
n = len(nf)
D_plain = np.zeros((n, n)); D_stem = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        D_plain[i, j] = D_plain[j, i] = P.ned(nf[i], nf[j])
        D_stem[i, j] = D_stem[j, i] = P.d_stem(nf[i], nf[j]) if len(nf[i]) >= 3 and len(nf[j]) >= 3 else P.ned(nf[i], nf[j])
lang = pd.factorize(sub.language)[0]
conc = pd.factorize(sub.concept)[0]
strata_lang = lang
strata_lang_dom = pd.factorize(sub.language + "|" + dom)[0]
pairs = (lang[:, None] != lang[None, :]) & np.triu(np.ones((n, n), dtype=bool), 1)


def test(D, strata, n_perm=10000):
    same = (conc[:, None] == conc[None, :]) & pairs
    T, N = float(D[same].mean()), int((D[same] <= 0.34).sum())
    rs = np.random.default_rng(SEED)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    Tp, Np = np.empty(n_perm), np.empty(n_perm)
    for k in range(n_perm):
        c2 = conc.copy()
        for g in groups:
            c2[g] = conc[rs.permutation(g)]
        s_ = (c2[:, None] == c2[None, :]) & pairs
        v = D[s_]
        Tp[k], Np[k] = v.mean(), (v <= 0.34).sum()
    return {"n_forms": n, "n_pairs": int(same.sum()), "T_observed": round(T, 4),
            "T_permutation_mean": round(float(Tp.mean()), 4), "T_permutation_sd": round(float(Tp.std()), 4),
            "excess_similarity": round(float(Tp.mean() - T), 4),
            "p_T": round(float(((Tp <= T).sum() + 1) / (n_perm + 1)), 5),
            "lookalike_pairs_observed": N, "lookalike_pairs_expected": round(float(Np.mean()), 2),
            "p_N": round(float(((Np >= N).sum() + 1) / (n_perm + 1)), 5)}


res = {
    "plain_distance__permute_within_language (= S5 b, repeated)": test(D_plain, strata_lang),
    "plain_distance__permute_within_language_x_domain": test(D_plain, strata_lang_dom),
    "stem_tolerant__permute_within_language": test(D_stem, strata_lang),
    "stem_tolerant__permute_within_language_x_domain": test(D_stem, strata_lang_dom),
}
for k, v in res.items():
    print(k, v, flush=True)
res["_status"] = "POST HOC / exploratory (amendment A2)"
(OUT / "S5x_permutation_stratified.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
