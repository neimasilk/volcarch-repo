#!/usr/bin/env python3
"""
E090 V5 Full Run: All Transformer NLP Experiments on 200-Entry Corpus
=====================================================================
Runs EXP 1, 2, 4, and 5 (extended) from the E090 pipeline on the v5 corpus
(200 entries from 12 traditions).

WHAT CHANGED FROM v3 SELECTIVE (e090_selective_v3.py):
------------------------------------------------------
1. CORPUS: v5 has 200 entries from 12 traditions (was 106 from 10 in v3,
   162 from 11 in v4). This is a 4x increase over the original v2 (50).

2. EXP 4 (BERTopic) REACTIVATED: The v3 run skipped BERTopic because
   106 entries was below the ~200 document threshold for stable topic
   modeling. With 200 entries we can now run it. This is the first time
   BERTopic runs on a corpus large enough for meaningful results.

3. EXP 5 EXTENDED — 3 new concept groups:
   - VOLCANO: eruption/mountain/fire/ash vocabulary — directly tests whether
     volcanic terminology creates cross-tradition semantic convergence.
   - BUDDHIST_WORLD: buddha/monastery/vihara/stupa — tests if Buddhist
     religious networks show convergent descriptions across traditions.
   - METAL_TRADE: gold/silver/copper/tin/iron — tests if metallurgical
     trade vocabulary converges across traditions (distinct from the
     existing SUMATRA_GOLD which focuses on gold-land identification).
   Total concept groups: 8 (was 5).

4. NEW: Delta comparison with previous results — loads the most recent
   previous run (e090_all_results.json or e090_v3_selective_results.json)
   and reports changes in key metrics across corpus versions.

EXPERIMENTS RUN:
  EXP 1 — Sentence-BERT semantic similarity (all-MiniLM-L6-v2)
  EXP 2 — UMAP + HDBSCAN cross-lingual clustering
  EXP 4 — BERTopic latent topic discovery (REACTIVATED)
  EXP 5 — Extended semantic convergence (8 concept groups)

EXPERIMENTS STILL SKIPPED:
  EXP 3 — Zero-shot NER: extraction task, not analysis. Low priority.
  EXP 6 — Cross-tradition NLI: conceptually flawed on translations.
           Measures translation register similarity, not historical validity.

Hardware: RTX 4080 (CUDA). Models loaded on GPU.
"""

import sys
import os
import json
import warnings
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# Import experiment functions from the main E090 script
sys.path.insert(0, os.path.dirname(__file__))
from e090_transformer_nlp import (
    exp1_semantic_similarity,
    exp2_clustering,
    exp4_bertopic,
    exp5_semantic_convergence
)

# ============================================================================
# CORPUS PATH — prefer v5, fall back to v4, then v3
# ============================================================================
_BASE = os.path.join(os.path.dirname(__file__), "..", "E089_expanded_textual_corpus", "results")
V5_CORPUS_PATH = os.path.join(_BASE, "nusantara_corpus_v5.json")
V4_CORPUS_PATH = os.path.join(_BASE, "nusantara_corpus_v4.json")
V3_CORPUS_PATH = os.path.join(_BASE, "nusantara_corpus_v3.json")

if os.path.exists(V5_CORPUS_PATH):
    CORPUS_PATH = V5_CORPUS_PATH
    CORPUS_VERSION = "v5"
elif os.path.exists(V4_CORPUS_PATH):
    CORPUS_PATH = V4_CORPUS_PATH
    CORPUS_VERSION = "v4"
elif os.path.exists(V3_CORPUS_PATH):
    CORPUS_PATH = V3_CORPUS_PATH
    CORPUS_VERSION = "v3"
else:
    CORPUS_PATH = None
    CORPUS_VERSION = None


def load_corpus():
    """Load the best available corpus (v5 preferred, v4/v3 fallback)."""
    if CORPUS_PATH is None or not os.path.exists(CORPUS_PATH):
        print("  ERROR: no corpus found (checked v5, v4, v3).")
        print("  Run E089 expansion first.")
        sys.exit(1)
    print(f"  Using {CORPUS_VERSION} corpus: {CORPUS_PATH}")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# EXP 5 EXTENDED: Semantic Convergence with 8 Concept Groups
# ============================================================================
def exp5_extended_convergence(corpus, embeddings, output_dir):
    """
    Extended version of exp5_semantic_convergence with 8 concept groups.

    Original 5 groups (from e090_transformer_nlp.py):
      JAVA, SUMATRA_GOLD, CAMPHOR_BARUS, SPICE_TRADE, MARITIME_VOYAGE

    New groups added for v5:
      VOLCANO — eruption/mountain/fire/ash vocabulary. Directly tests
        whether volcanic terminology creates cross-tradition convergence.
      BUDDHIST_WORLD — buddha/monastery/vihara/stupa. Tests if Buddhist
        network descriptions converge across traditions.
      METAL_TRADE — gold/silver/copper/tin/iron/bronze. Tests if
        metallurgical trade vocabulary converges (distinct from SUMATRA_GOLD
        which focuses on gold-land identification, not metalworking).
    """
    print("\n" + "=" * 70)
    print("EXP 5 EXTENDED: SEMANTIC CONVERGENCE (8 CONCEPT GROUPS)")
    print("=" * 70)

    from sklearn.metrics.pairwise import cosine_similarity

    # --- Define all 8 concept groups ---
    concept_terms = {
        # Original 5
        "JAVA": {"java", "yavadvipa", "iabadiu", "ye-po-ti", "shepo", "zabaj", "jawa"},
        "SUMATRA_GOLD": {"chryse", "suvarnabhumi", "suvarnadvipa", "aurea", "golden", "gold", "emas"},
        "CAMPHOR_BARUS": {"camphor", "karpura", "kafur", "kapur", "barus", "fansur"},
        "SPICE_TRADE": {"clove", "nutmeg", "cinnamon", "pepper", "sandalwood", "aromatic", "spice"},
        "MARITIME_VOYAGE": {"sail", "ship", "voyage", "merchant", "sea", "maritime", "boat", "embarked"},
        # New 3
        "VOLCANO": {"volcano", "eruption", "mountain", "fire", "ash", "lava", "crater",
                     "smoke", "sulfur", "tremor", "gunung"},
        "BUDDHIST_WORLD": {"buddha", "buddhist", "monastery", "monk", "vihara", "stupa",
                           "dharma", "sangha", "pilgrimage", "bodhi"},
        "METAL_TRADE": {"gold", "silver", "copper", "tin", "iron", "bronze", "metal",
                        "forge", "mine", "ore", "smelting"},
    }

    # Assign passages to concept groups
    concept_groups = {name: [] for name in concept_terms}

    for i, ref in enumerate(corpus):
        text_lower = ref["passage_text"].lower()
        entity_texts = " ".join(e["text"].lower() for e in ref.get("entities", []))
        combined = text_lower + " " + entity_texts

        for concept, terms in concept_terms.items():
            if any(t in combined for t in terms):
                concept_groups[concept].append(i)

    print(f"  Concept groups:")
    for concept, indices in concept_groups.items():
        trads = [corpus[i]["tradition"] for i in indices]
        marker = " [NEW]" if concept in ("VOLCANO", "BUDDHIST_WORLD", "METAL_TRADE") else ""
        print(f"    {concept}: {len(indices)} passages from {len(set(trads))} traditions{marker}")

    # Compute similarity matrix once
    sim_matrix = cosine_similarity(embeddings)
    results = {}

    for concept, indices in concept_groups.items():
        if len(indices) < 2:
            print(f"\n    {concept}: SKIPPED (< 2 members)")
            results[concept] = {
                "n_members": len(indices),
                "n_traditions": len(set(corpus[i]["tradition"] for i in indices)) if indices else 0,
                "verdict": "SKIPPED (< 2 members)"
            }
            continue

        # Intra-group mean similarity
        intra_sims = []
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                intra_sims.append(sim_matrix[indices[a]][indices[b]])
        intra_mean = np.mean(intra_sims)

        # Monte Carlo random baseline: 1000 iterations
        n_pairs = len(intra_sims)
        random_means = []
        rng = np.random.RandomState(42)
        for _ in range(1000):
            rand_pairs = []
            for _ in range(n_pairs):
                a, b = rng.choice(len(corpus), 2, replace=False)
                rand_pairs.append(sim_matrix[a][b])
            random_means.append(np.mean(rand_pairs))

        random_mean = np.mean(random_means)
        random_std = np.std(random_means)
        z_score = (intra_mean - random_mean) / random_std if random_std > 0 else 0
        p_value = np.mean([rm >= intra_mean for rm in random_means])

        # Cross-tradition pairs within the group
        cross_trad_sims = []
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                if corpus[indices[a]]["tradition"] != corpus[indices[b]]["tradition"]:
                    cross_trad_sims.append(sim_matrix[indices[a]][indices[b]])
        cross_trad_mean = np.mean(cross_trad_sims) if cross_trad_sims else None

        marker = " [NEW]" if concept in ("VOLCANO", "BUDDHIST_WORLD", "METAL_TRADE") else ""
        print(f"\n    {concept}:{marker}")
        print(f"      Intra-group similarity: {intra_mean:.4f}")
        print(f"      Random baseline:        {random_mean:.4f} +/- {random_std:.4f}")
        print(f"      Z-score: {z_score:.2f}, p-value: {p_value:.4f}")
        if cross_trad_mean is not None:
            print(f"      Cross-tradition within group: {cross_trad_mean:.4f}")
        verdict = "CONVERGES" if z_score > 1.96 else "NO CONVERGENCE"
        print(f"      Verdict: {verdict}")

        results[concept] = {
            "n_members": len(indices),
            "n_traditions": len(set(corpus[i]["tradition"] for i in indices)),
            "intra_group_similarity": float(intra_mean),
            "random_baseline": float(random_mean),
            "random_std": float(random_std),
            "z_score": float(z_score),
            "p_value": float(p_value),
            "cross_tradition_similarity": float(cross_trad_mean) if cross_trad_mean is not None else None,
            "verdict": verdict
        }

    # Summary
    converging = sum(1 for v in results.values() if v.get("verdict") == "CONVERGES")
    total = sum(1 for v in results.values() if v.get("verdict") in ("CONVERGES", "NO CONVERGENCE"))
    print(f"\n  Summary: {converging}/{total} concept groups show semantic convergence (z > 1.96)")

    # Separate summary for original vs new groups
    orig_conv = sum(1 for k, v in results.items()
                    if k not in ("VOLCANO", "BUDDHIST_WORLD", "METAL_TRADE")
                    and v.get("verdict") == "CONVERGES")
    orig_total = sum(1 for k, v in results.items()
                     if k not in ("VOLCANO", "BUDDHIST_WORLD", "METAL_TRADE")
                     and v.get("verdict") in ("CONVERGES", "NO CONVERGENCE"))
    new_conv = sum(1 for k, v in results.items()
                   if k in ("VOLCANO", "BUDDHIST_WORLD", "METAL_TRADE")
                   and v.get("verdict") == "CONVERGES")
    new_total = sum(1 for k, v in results.items()
                    if k in ("VOLCANO", "BUDDHIST_WORLD", "METAL_TRADE")
                    and v.get("verdict") in ("CONVERGES", "NO CONVERGENCE"))
    print(f"    Original groups: {orig_conv}/{orig_total}")
    print(f"    New groups:      {new_conv}/{new_total}")

    return {
        "method": "Embedding-space convergence (Monte Carlo baseline) — extended 8 groups",
        "concept_groups": results,
        "converging_groups": converging,
        "total_groups": total,
        "original_groups_converging": orig_conv,
        "new_groups_converging": new_conv
    }


# ============================================================================
# DELTA COMPARISON: v5 vs previous results
# ============================================================================
def load_previous_results():
    """Load most recent previous results for delta comparison."""
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    # Try v3 selective first (most comparable), then original all-results
    candidates = [
        os.path.join(results_dir, "e090_v3_selective_results.json"),
        os.path.join(results_dir, "e090_all_results.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            label = os.path.basename(path)
            return data, label
    return None, None


def compute_delta(v5_results, prev_results, prev_label):
    """Compare v5 results against previous run and return delta report."""
    print("\n" + "=" * 70)
    print(f"DELTA COMPARISON: v5 vs {prev_label}")
    print("=" * 70)

    delta = {"compared_against": prev_label, "changes": {}}

    # --- EXP 1: within/between ratio ---
    exp1_v5 = v5_results.get("exp1_semantic_similarity", {})
    exp1_prev = prev_results.get("exp1_semantic_similarity", {})

    if exp1_v5 and exp1_prev:
        ratio_v5 = exp1_v5.get("ratio_within_between")
        ratio_prev = exp1_prev.get("ratio_within_between")
        within_v5 = exp1_v5.get("within_tradition_similarity", {}).get("mean")
        within_prev = exp1_prev.get("within_tradition_similarity", {}).get("mean")
        between_v5 = exp1_v5.get("between_tradition_similarity", {}).get("mean")
        between_prev = exp1_prev.get("between_tradition_similarity", {}).get("mean")
        n_v5 = exp1_v5.get("n_passages", "?")
        n_prev = exp1_prev.get("n_passages", "?")

        print(f"\n  EXP 1 — SBERT Semantic Similarity:")
        print(f"    Corpus size: {n_prev} -> {n_v5}")

        if within_v5 is not None and within_prev is not None:
            d_within = within_v5 - within_prev
            print(f"    Within-tradition:  {within_prev:.4f} -> {within_v5:.4f} ({d_within:+.4f})")
        if between_v5 is not None and between_prev is not None:
            d_between = between_v5 - between_prev
            print(f"    Between-tradition: {between_prev:.4f} -> {between_v5:.4f} ({d_between:+.4f})")
        if ratio_v5 is not None and ratio_prev is not None:
            d_ratio = ratio_v5 - ratio_prev
            pct = 100 * d_ratio / ratio_prev if ratio_prev != 0 else 0
            print(f"    Ratio:             {ratio_prev:.3f} -> {ratio_v5:.3f} ({d_ratio:+.3f}, {pct:+.1f}%)")

        delta["changes"]["exp1"] = {
            "n_passages": {"prev": n_prev, "v5": n_v5},
            "ratio_within_between": {"prev": ratio_prev, "v5": ratio_v5},
            "within_mean": {"prev": within_prev, "v5": within_v5},
            "between_mean": {"prev": between_prev, "v5": between_v5},
        }

    # --- EXP 2: cluster count and mixed% ---
    exp2_v5 = v5_results.get("exp2_clustering", {})
    exp2_prev = prev_results.get("exp2_clustering", {})

    if exp2_v5 and exp2_prev:
        clust_v5 = exp2_v5.get("n_clusters", 0)
        clust_prev = exp2_prev.get("n_clusters", 0)
        mixed_v5 = exp2_v5.get("pct_mixed_clusters", 0)
        mixed_prev = exp2_prev.get("pct_mixed_clusters", 0)
        type_v5 = exp2_v5.get("clustering_type", "?")
        type_prev = exp2_prev.get("clustering_type", "?")

        print(f"\n  EXP 2 — UMAP + HDBSCAN Clustering:")
        print(f"    Clusters:     {clust_prev} -> {clust_v5} ({clust_v5 - clust_prev:+d})")
        print(f"    Mixed%:       {mixed_prev:.0f}% -> {mixed_v5:.0f}% ({mixed_v5 - mixed_prev:+.0f}%)")
        print(f"    Type:         {type_prev} -> {type_v5}")

        delta["changes"]["exp2"] = {
            "n_clusters": {"prev": clust_prev, "v5": clust_v5},
            "pct_mixed": {"prev": mixed_prev, "v5": mixed_v5},
            "clustering_type": {"prev": type_prev, "v5": type_v5},
        }

    # --- EXP 5: convergence z-scores per concept group ---
    exp5_v5 = v5_results.get("exp5_extended_convergence", {})
    exp5_prev = prev_results.get("exp5_semantic_convergence", {})

    if exp5_v5 and exp5_prev:
        groups_v5 = exp5_v5.get("concept_groups", {})
        groups_prev = exp5_prev.get("concept_groups", {})

        print(f"\n  EXP 5 — Semantic Convergence (z-scores):")
        print(f"    {'Concept':<20s}  {'Prev z':>8s}  {'V5 z':>8s}  {'Delta':>8s}  Verdict")
        print(f"    {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}")

        exp5_delta = {}
        for concept in groups_v5:
            v5_data = groups_v5[concept]
            prev_data = groups_prev.get(concept, {})

            z_v5 = v5_data.get("z_score")
            z_prev = prev_data.get("z_score")
            verdict_v5 = v5_data.get("verdict", "?")

            if z_v5 is not None and z_prev is not None:
                dz = z_v5 - z_prev
                print(f"    {concept:<20s}  {z_prev:>8.2f}  {z_v5:>8.2f}  {dz:>+8.2f}  {verdict_v5}")
                exp5_delta[concept] = {"prev_z": z_prev, "v5_z": z_v5, "delta_z": dz, "verdict": verdict_v5}
            elif z_v5 is not None:
                print(f"    {concept:<20s}  {'N/A':>8s}  {z_v5:>8.2f}  {'NEW':>8s}  {verdict_v5}")
                exp5_delta[concept] = {"prev_z": None, "v5_z": z_v5, "delta_z": None, "verdict": verdict_v5}
            else:
                print(f"    {concept:<20s}  {'?':>8s}  {'SKIP':>8s}  {'':>8s}  {verdict_v5}")
                exp5_delta[concept] = {"prev_z": None, "v5_z": None, "verdict": verdict_v5}

        conv_prev = exp5_prev.get("converging_groups", "?")
        tot_prev = exp5_prev.get("total_groups", "?")
        conv_v5 = exp5_v5.get("converging_groups", "?")
        tot_v5 = exp5_v5.get("total_groups", "?")
        print(f"\n    Converging: {conv_prev}/{tot_prev} -> {conv_v5}/{tot_v5}")

        delta["changes"]["exp5"] = exp5_delta

    return delta


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("E090 V5 FULL RUN — TRANSFORMER NLP ON 200-ENTRY CORPUS")
    print("=" * 70)
    print()
    print("What's new in v5:")
    print("  - Corpus: 200 entries from 12 traditions (was 106/v3, 162/v4)")
    print("  - EXP 4 (BERTopic): REACTIVATED — 200 entries meets threshold")
    print("  - EXP 5: Extended to 8 concept groups (+VOLCANO, +BUDDHIST, +METAL)")
    print("  - Delta comparison with previous results")
    print()

    # --- Load corpus ---
    corpus = load_corpus()
    print(f"  Loaded corpus: {len(corpus)} entries")
    traditions = set(r["tradition"] for r in corpus)
    print(f"  Traditions: {len(traditions)} — {', '.join(sorted(traditions))}")

    # Tradition breakdown
    trad_counts = Counter(r["tradition"] for r in corpus)
    for trad in sorted(trad_counts.keys()):
        print(f"    {trad}: {trad_counts[trad]}")
    print()

    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        "meta": {
            "script": "e090_v5_full.py",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "corpus": f"nusantara_corpus_{CORPUS_VERSION}.json",
            "corpus_version": CORPUS_VERSION,
            "corpus_size": len(corpus),
            "n_traditions": len(traditions),
            "experiments_run": [1, 2, 4, 5],
            "experiments_skipped": {
                "3": "NER is extraction not analysis — low priority",
                "6": "NLI on translations is conceptually flawed"
            },
            "exp5_note": "Extended to 8 concept groups: +VOLCANO, +BUDDHIST_WORLD, +METAL_TRADE"
        }
    }

    # ------------------------------------------------------------------
    # EXP 1: Sentence-BERT Semantic Similarity
    # ------------------------------------------------------------------
    exp1_result, embeddings = exp1_semantic_similarity(corpus, output_dir)
    all_results["exp1_semantic_similarity"] = exp1_result

    # ------------------------------------------------------------------
    # EXP 2: Cross-lingual Clustering (UMAP + HDBSCAN)
    # ------------------------------------------------------------------
    exp2_result = exp2_clustering(corpus, embeddings, output_dir)
    all_results["exp2_clustering"] = exp2_result

    # ------------------------------------------------------------------
    # EXP 4: BERTopic (REACTIVATED — 200 entries meets threshold)
    # ------------------------------------------------------------------
    print("\n  NOTE: BERTopic REACTIVATED — 200 entries meets 200+ threshold")
    exp4_result = exp4_bertopic(corpus, embeddings, output_dir)
    all_results["exp4_bertopic"] = exp4_result

    # ------------------------------------------------------------------
    # EXP 5 EXTENDED: Semantic Convergence (8 concept groups)
    # ------------------------------------------------------------------
    exp5_result = exp5_extended_convergence(corpus, embeddings, output_dir)
    all_results["exp5_extended_convergence"] = exp5_result

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "e090_v5_full_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved: {results_path}")

    # ------------------------------------------------------------------
    # DELTA COMPARISON
    # ------------------------------------------------------------------
    prev_results, prev_label = load_previous_results()
    delta = None
    if prev_results is not None:
        delta = compute_delta(all_results, prev_results, prev_label)
        # Save delta
        delta_path = os.path.join(output_dir, "e090_v5_delta.json")
        with open(delta_path, "w", encoding="utf-8") as f:
            json.dump(delta, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Saved delta: {delta_path}")
    else:
        print("\n  No previous results found for delta comparison.")
        print("  (Looked for e090_v3_selective_results.json, e090_all_results.json)")

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("E090 V5 FULL RUN — FINAL SUMMARY")
    print("=" * 70)
    print(f"  Corpus: {CORPUS_VERSION}, {len(corpus)} entries, {len(traditions)} traditions")
    print(f"  Experiments run: 4 (EXP 1, 2, 4, 5-extended)")
    print()

    # EXP 1 summary
    ratio = exp1_result.get("ratio_within_between")
    within = exp1_result.get("within_tradition_similarity", {}).get("mean")
    between = exp1_result.get("between_tradition_similarity", {}).get("mean")
    print(f"  EXP 1 — SBERT Semantic Similarity:")
    print(f"    Within-tradition:  {within:.4f}" if within else "    Within-tradition:  N/A")
    print(f"    Between-tradition: {between:.4f}" if between else "    Between-tradition: N/A")
    print(f"    Ratio (within/between): {ratio:.3f}" if ratio else "    Ratio: N/A")
    print()

    # EXP 2 summary
    n_clusters = exp2_result.get("n_clusters", 0)
    pct_mixed = exp2_result.get("pct_mixed_clusters", 0)
    clustering_type = exp2_result.get("clustering_type", "unknown")
    print(f"  EXP 2 — UMAP + HDBSCAN Clustering:")
    print(f"    Clusters: {n_clusters}")
    print(f"    Cross-tradition clusters: {pct_mixed:.0f}%")
    print(f"    Type: {clustering_type}")
    print()

    # EXP 4 summary
    n_topics = exp4_result.get("n_topics", "FAILED")
    exp4_status = exp4_result.get("status", "OK")
    print(f"  EXP 4 — BERTopic (REACTIVATED):")
    if exp4_status == "FAILED":
        print(f"    Status: FAILED — {exp4_result.get('error', 'unknown error')}")
    else:
        print(f"    Topics found: {n_topics}")
        # Print topic words if available
        for topic in exp4_result.get("topics", []):
            tid = topic.get("id", "?")
            count = topic.get("count", 0)
            words = topic.get("words", [])
            word_str = ", ".join(w for w, _ in words[:5]) if words else "(no words)"
            label = "OUTLIER" if tid == -1 else f"Topic {tid}"
            print(f"      {label} ({count} docs): {word_str}")
    print()

    # EXP 5 summary
    converging = exp5_result.get("converging_groups", 0)
    total = exp5_result.get("total_groups", 0)
    orig_conv = exp5_result.get("original_groups_converging", 0)
    new_conv = exp5_result.get("new_groups_converging", 0)
    print(f"  EXP 5 Extended — Semantic Convergence (8 groups):")
    print(f"    Converging: {converging}/{total} (original: {orig_conv}/5, new: {new_conv}/3)")
    if exp5_result.get("concept_groups"):
        for concept, data in exp5_result["concept_groups"].items():
            z = data.get("z_score")
            p = data.get("p_value")
            v = data.get("verdict", "?")
            marker = " [NEW]" if concept in ("VOLCANO", "BUDDHIST_WORLD", "METAL_TRADE") else ""
            if z is not None and p is not None:
                print(f"      {concept:20s}: z={z:.2f}, p={p:.4f} — {v}{marker}")
            else:
                print(f"      {concept:20s}: {v}{marker}")
    print()

    print("=" * 70)
    print("DONE. Results saved to results/e090_v5_full_results.json")
    if delta:
        print(f"Delta comparison saved to results/e090_v5_delta.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
