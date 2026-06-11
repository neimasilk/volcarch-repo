"""
E090 v6 — TRADITION-CONTROLLED convergence test (addresses DeepSeek G9 W1, 2026-06-08).

The original exp5 compared INTRA-group mean similarity (within- AND cross-tradition
pairs mixed) to a random baseline. DeepSeek correctly noted this measures within-group
TOPICAL COHERENCE, not CROSS-TRADITION convergence: if a group is dominated by one
tradition, high intra-similarity can be within-tradition homogeneity.

Proper test of "different traditions converge on theme G":
  S_cross(G) = mean cosine of CROSS-tradition pairs within G (the two passages are from
               DIFFERENT traditions).
  Null      = mean cosine of random CROSS-tradition pairs drawn from the whole corpus.
  z_cross   = (S_cross - null_mean) / null_std ; p_cross = P(null >= S_cross).
If z_cross > 1.96 -> theme-G passages from different traditions ARE more similar than
random different-tradition passages = genuine cross-tradition convergence (Finding 1 holds).
Also report S_within vs S_cross and tradition dominance. BH-correct across groups.
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(r"D:/documents/volcarch-repo/experiments")
corpus = json.load(open(BASE/"E089_expanded_textual_corpus/results/nusantara_corpus_v5.json", encoding="utf-8"))
emb = np.load(BASE/"E090_transformer_textual_nlp/results/passage_embeddings.npy")
assert emb.shape[0] == len(corpus), f"embedding/corpus mismatch: {emb.shape} vs {len(corpus)}"
N = len(corpus)
trad = [r["tradition"] for r in corpus]
sim = cosine_similarity(emb)
rng = np.random.RandomState(42)

concept_terms = {
    "JAVA": {"java","yavadvipa","iabadiu","ye-po-ti","shepo","zabaj","jawa"},
    "SUMATRA_GOLD": {"chryse","suvarnabhumi","suvarnadvipa","aurea","golden","gold","emas"},
    "CAMPHOR_BARUS": {"camphor","karpura","kafur","kapur","barus","fansur"},
    "SPICE_TRADE": {"clove","nutmeg","cinnamon","pepper","sandalwood","aromatic","spice"},
    "MARITIME_VOYAGE": {"sail","ship","voyage","merchant","sea","maritime","boat","embarked"},
    "VOLCANO": {"volcano","eruption","mountain","fire","ash","lava","crater","smoke","sulfur","tremor","gunung"},
    "BUDDHIST_WORLD": {"buddha","buddhist","monastery","monk","vihara","stupa","dharma","sangha","pilgrimage","bodhi"},
    "METAL_TRADE": {"gold","silver","copper","tin","iron","bronze","metal","forge","mine","ore","smelting"},
}
groups = {n: [] for n in concept_terms}
for i, ref in enumerate(corpus):
    txt = ref["passage_text"].lower() + " " + " ".join(e["text"].lower() for e in ref.get("entities", []))
    for c, terms in concept_terms.items():
        if any(t in txt for t in terms):
            groups[c].append(i)

# Pool of ALL cross-tradition pair similarities in the whole corpus (the null universe)
pool = np.array([sim[a][b] for a in range(N) for b in range(a+1, N) if trad[a] != trad[b]])
print(f"Corpus N={N}, traditions={len(set(trad))}, cross-tradition pair pool={len(pool)}\n")

def cross_z(s_cross, n_pairs, iters=5000):
    draws = rng.choice(pool, size=(iters, n_pairs), replace=True).mean(axis=1)
    return (s_cross - draws.mean())/draws.std(), float(np.mean(draws >= s_cross)), float(draws.mean())

results = {}
print(f"{'group':<16}{'n':>4}{'nTrad':>6}{'domShr':>7}{'S_within':>9}{'S_cross':>9}{'nullCx':>8}{'z_cross':>9}{'p':>8}")
for c, idx in groups.items():
    if len(idx) < 2:
        results[c] = {"n_members": len(idx), "verdict": "SKIP"}; continue
    cross, within = [], []
    for a in range(len(idx)):
        for b in range(a+1, len(idx)):
            i, j = idx[a], idx[b]
            (cross if trad[i] != trad[j] else within).append(sim[i][j])
    tc = Counter(trad[i] for i in idx)
    r = {"n_members": len(idx), "n_traditions": len(tc),
         "dominant_tradition_share": round(max(tc.values())/len(idx), 2),
         "n_cross_pairs": len(cross), "n_within_pairs": len(within),
         "S_within": round(float(np.mean(within)), 4) if within else None,
         "S_cross": round(float(np.mean(cross)), 4) if cross else None}
    if cross:
        z, p, nullm = cross_z(r["S_cross"], len(cross))
        r.update(z_cross_tradition=round(z, 2), p_cross=round(p, 4), null_cross_mean=round(nullm, 4),
                 verdict="CROSS-TRADITION CONVERGENCE" if (z > 1.96 and p < 0.05) else "NOT cross-tradition")
    else:
        r["verdict"] = "NO cross-tradition pairs (single-tradition group)"
    results[c] = r
    sw = f"{r['S_within']:.3f}" if r.get("S_within") is not None else "  -  "
    print(f"{c:<16}{r['n_members']:>4}{r.get('n_traditions',0):>6}{r['dominant_tradition_share']:>7}"
          f"{sw:>9}{r.get('S_cross', 0):>9.3f}{r.get('null_cross_mean', 0):>8.3f}"
          f"{r.get('z_cross_tradition', 0):>9.2f}{r.get('p_cross', 1):>8.4f}")

# BH correction across groups with a p_cross
ps = [(c, results[c]["p_cross"]) for c in results if "p_cross" in results[c]]
m = len(ps)
for rank, (c, p) in enumerate(sorted(ps, key=lambda x: x[1]), 1):
    results[c]["p_cross_BH"] = round(min(1.0, p*m/rank), 4)

conv = [c for c in results if results[c].get("verdict") == "CROSS-TRADITION CONVERGENCE"]
print(f"\nCROSS-TRADITION convergence (z>1.96, p<0.05): {len(conv)}/{m} groups -> {conv}")
print("Compare to original exp5 claim: 8/8 'converge' (intra-group vs random).")
Path(BASE/"E090_transformer_textual_nlp/results/e090_v6_tradition_controlled.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")
print("Saved: results/e090_v6_tradition_controlled.json")
