#!/usr/bin/env python3
"""
E090 Selective V3 Re-run: SBERT + UMAP/HDBSCAN + Semantic Convergence
======================================================================
Runs experiments 1, 2, and 5 from the E090 pipeline on the v3 corpus
(106 entries, ~2x the original).

WHY THIS SELECTIVE RE-RUN EXISTS:
- The v3 corpus (E089) has ~2x more data than v2 (106 vs ~53 entries).
  More data strengthens the embedding-based experiments significantly.
- EXP 1 (SBERT): Benefits directly from larger sample — more pairwise
  comparisons, more robust tradition centroids.
- EXP 2 (UMAP+HDBSCAN): More data points improve cluster stability.
- EXP 5 (Semantic Convergence): Larger concept groups make the Monte
  Carlo convergence test more statistically powerful.

WHY WE SKIP THE OTHERS:
- EXP 3 (Zero-shot NER): Extraction task, not analysis. Results are
  per-passage and don't gain much from corpus expansion. Low priority.
- EXP 4 (BERTopic): Needs ~200+ documents for stable topic modeling.
  106 entries is still below threshold. Will revisit when corpus hits 200.
- EXP 6 (Cross-tradition NLI): Conceptually flawed — NLI tests textual
  entailment between modern translations, not historical consistency.
  The scores reflect translation register similarity, not genuine
  cross-tradition validation. Archived, not re-run.

Hardware: RTX 4080 (CUDA). Models loaded on GPU.
"""

import sys
import os
import json
import warnings
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# Import experiment functions from the main E090 script
sys.path.insert(0, os.path.dirname(__file__))
from e090_transformer_nlp import (
    exp1_semantic_similarity,
    exp2_clustering,
    exp5_semantic_convergence
)

# ============================================================================
# CORPUS PATH — prefer v4, fall back to v3
# ============================================================================
_BASE = os.path.join(os.path.dirname(__file__), "..", "E089_expanded_textual_corpus", "results")
V4_CORPUS_PATH = os.path.join(_BASE, "nusantara_corpus_v4.json")
V3_CORPUS_PATH = os.path.join(_BASE, "nusantara_corpus_v3.json")
# Use v4 (162 entries) if available, otherwise v3 (106 entries)
CORPUS_PATH = V4_CORPUS_PATH if os.path.exists(V4_CORPUS_PATH) else V3_CORPUS_PATH


def load_corpus():
    """Load the best available corpus (v4 preferred, v3 fallback)."""
    if not os.path.exists(CORPUS_PATH):
        print(f"  ERROR: corpus not found at {CORPUS_PATH}")
        print(f"  Run E089 expansion first.")
        sys.exit(1)
    version = "v4" if "v4" in CORPUS_PATH else "v3"
    print(f"  Using {version} corpus: {CORPUS_PATH}")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("E090 SELECTIVE V3 RE-RUN")
    print("v3 corpus (106 entries) — EXP 1, 2, 5 only")
    print("=" * 70)
    print()
    print("Rationale:")
    print("  - EXP 1 (SBERT): more pairwise comparisons with 2x data")
    print("  - EXP 2 (UMAP+HDBSCAN): better cluster stability")
    print("  - EXP 5 (Semantic Convergence): stronger Monte Carlo test")
    print("  - SKIP EXP 3 (NER): extraction, not analysis")
    print("  - SKIP EXP 4 (BERTopic): needs 200+ docs, we have 106")
    print("  - SKIP EXP 6 (NLI): conceptually flawed on translations")
    print()

    corpus = load_corpus()
    print(f"Loaded v3 corpus: {len(corpus)} entries")

    # Count traditions
    traditions = set(r["tradition"] for r in corpus)
    print(f"Traditions: {len(traditions)} — {', '.join(sorted(traditions))}")
    print()

    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        "meta": {
            "script": "e090_selective_v3.py",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "corpus": "nusantara_corpus_v3.json",
            "corpus_size": len(corpus),
            "experiments_run": [1, 2, 5],
            "experiments_skipped": {
                3: "NER is extraction not analysis — low priority",
                4: "BERTopic needs 200+ docs, corpus has 106",
                6: "NLI on translations is conceptually flawed"
            }
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
    # EXP 5: Semantic Convergence in Embedding Space
    # ------------------------------------------------------------------
    exp5_result = exp5_semantic_convergence(corpus, embeddings, output_dir)
    all_results["exp5_semantic_convergence"] = exp5_result

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "e090_v3_selective_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved: {results_path}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("E090 V3 SELECTIVE RE-RUN — SUMMARY")
    print("=" * 70)
    print(f"  Corpus: v3, {len(corpus)} entries, {len(traditions)} traditions")
    print(f"  Experiments run: 3 of 6 (EXP 1, 2, 5)")
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

    # EXP 5 summary
    converging = exp5_result.get("converging_groups", 0)
    total = exp5_result.get("total_groups", 0)
    print(f"  EXP 5 — Semantic Convergence:")
    print(f"    Converging concept groups: {converging}/{total}")
    if exp5_result.get("concept_groups"):
        for concept, data in exp5_result["concept_groups"].items():
            z = data.get("z_score", 0)
            p = data.get("p_value", 1)
            v = data.get("verdict", "?")
            print(f"      {concept:20s}: z={z:.2f}, p={p:.4f} — {v}")
    print()

    print("=" * 70)
    print("DONE. Compare with original v2 results in e090_all_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
