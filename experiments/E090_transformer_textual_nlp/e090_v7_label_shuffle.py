"""
E090 v7 — LABEL-SHUFFLE convergence test (addresses DeepSeek G9-R1 W1, 2026-06-10).

DeepSeek's R1 re-review still rejected the v6 test as circular: v6 compares
within-group CROSS-tradition similarity to a WHOLE-CORPUS cross-tradition baseline.
DeepSeek's objection: passages are tagged into a concept group *because they share
keywords*, so they are topically similar BY CONSTRUCTION; a positive z vs the
whole-corpus baseline is near-guaranteed and does not isolate "different traditions
converge" from "selected passages share vocabulary."

DeepSeek's prescribed fix (verbatim): "compares the observed within-concept
cross-tradition similarity to the distribution obtained by randomly SHUFFLING
TRADITION LABELS WITHIN EACH CONCEPT GROUP."

This script implements exactly that. The null universe is now INSIDE the group, so
topical coherence is held constant; the ONLY thing varying is which passages are
labelled which tradition. A positive result here means cross-tradition pairs are more
similar than chance GIVEN the same topical pool = genuine convergence, not selection.

Test per group G with members idx, traditions trad[idx]:
  observed  = mean cosine of pairs whose two members have DIFFERENT traditions.
  null draw = permute the tradition labels among members, recompute the same statistic.
  z         = (observed - null_mean)/null_std ; p = P(null >= observed)  [one-sided high]
  Also p_two = 2*min(P(null>=obs), P(null<=obs)) since convergence could in principle
  push cross-similarity EITHER way; we report both and the sign.
Requires >= 2 traditions AND at least one within- and one cross-tradition pairing to be
non-degenerate (else the permutation cannot move anything). BH-correct across groups.
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
trad = np.array([r["tradition"] for r in corpus])
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


def cross_mean(idx_local, labels):
    """mean cosine of pairs with DIFFERENT labels, over the submatrix sim[idx,idx]."""
    vals = []
    n = len(idx_local)
    for a in range(n):
        for b in range(a+1, n):
            if labels[a] != labels[b]:
                vals.append(sim[idx_local[a]][idx_local[b]])
    return float(np.mean(vals)) if vals else None


ITERS = 5000
results = {}
hdr = f"{'group':<16}{'n':>4}{'nTrad':>6}{'domShr':>7}{'obsCross':>9}{'nullMean':>9}{'z':>7}{'p_hi':>8}{'p_two':>8}"
print(f"Corpus N={N}, traditions={len(set(trad))}\n{hdr}")
for c, idx in groups.items():
    if len(idx) < 2:
        results[c] = {"n_members": len(idx), "verdict": "SKIP <2 members"}; continue
    idx = list(idx)
    labels0 = trad[idx]
    tc = Counter(labels0)
    obs = cross_mean(idx, labels0)
    rec = {"n_members": len(idx), "n_traditions": len(tc),
           "dominant_tradition_share": round(max(tc.values())/len(idx), 2),
           "obs_cross": round(obs, 4) if obs is not None else None}
    if obs is None or len(tc) < 2:
        rec["verdict"] = "DEGENERATE (no cross-tradition pairs / single tradition)"
        results[c] = rec
        print(f"{c:<16}{len(idx):>4}{len(tc):>6}{rec['dominant_tradition_share']:>7}{'  -  ':>9}{'  -  ':>9}{'  -  ':>7}{'  -  ':>8}{'  -  ':>8}")
        continue
    draws = np.empty(ITERS)
    lab = labels0.copy()
    for k in range(ITERS):
        rng.shuffle(lab)
        cm = cross_mean(idx, lab)
        draws[k] = cm if cm is not None else np.nan
    draws = draws[~np.isnan(draws)]
    nm, ns = float(draws.mean()), float(draws.std())
    z = (obs - nm)/ns if ns > 0 else 0.0
    p_hi = float(np.mean(draws >= obs))
    p_two = float(2*min(np.mean(draws >= obs), np.mean(draws <= obs)))
    rec.update(null_mean=round(nm, 4), null_std=round(ns, 4), z=round(z, 2),
               p_high=round(p_hi, 4), p_two_sided=round(min(1.0, p_two), 4),
               verdict="CROSS-TRADITION CONVERGENCE" if (z > 1.96 and p_hi < 0.05) else
                       ("CROSS-TRADITION DIVERGENCE" if (z < -1.96 and p_two < 0.05) else "NULL (topical coherence only)"))
    results[c] = rec
    print(f"{c:<16}{len(idx):>4}{len(tc):>6}{rec['dominant_tradition_share']:>7}{obs:>9.3f}{nm:>9.3f}{z:>7.2f}{p_hi:>8.4f}{min(1.0,p_two):>8.4f}")

# BH correction across groups that produced a one-sided p
ps = [(c, results[c]["p_high"]) for c in results if "p_high" in results[c]]
m = len(ps)
for rank, (c, p) in enumerate(sorted(ps, key=lambda x: x[1]), 1):
    results[c]["p_high_BH"] = round(min(1.0, p*m/rank), 4)
    results[c]["BH_significant"] = results[c]["p_high_BH"] < 0.05 and results[c]["z"] > 1.96

conv = [c for c in results if results[c].get("verdict") == "CROSS-TRADITION CONVERGENCE"]
conv_bh = [c for c in results if results[c].get("BH_significant")]
print(f"\nLabel-shuffle CROSS-TRADITION convergence (z>1.96, p<0.05): {len(conv)}/{m} groups -> {conv}")
print(f"After BH correction: {len(conv_bh)}/{m} -> {conv_bh}")
print("v6 (whole-corpus baseline) claimed 8/8; this is the stricter within-group label-shuffle DeepSeek demanded.")
Path(BASE/"E090_transformer_textual_nlp/results/e090_v7_label_shuffle.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")
print("Saved: results/e090_v7_label_shuffle.json")
