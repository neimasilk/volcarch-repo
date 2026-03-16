#!/usr/bin/env python3
"""
E090: Transformer-based NLP on Ancient Textual Corpus
=====================================================
Applies state-of-the-art NLP techniques to the E089 expanded corpus:

1. Sentence-BERT embeddings → semantic similarity matrix across traditions
2. Cross-lingual clustering → do traditions cluster by CONTENT or by LANGUAGE?
3. Zero-shot NER → entity extraction without training data
4. BERTopic → latent topic discovery
5. Semantic convergence analysis → embedding-space proof of convergence
6. Cross-tradition entailment → can one tradition's claims predict another's?

Hardware: RTX 4080 (CUDA). Models loaded on GPU.

Nature of experiment: EXPLORATORY. Bad results reported as-is. No sugarcoating.
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

# ============================================================================
# LOAD CORPUS
# ============================================================================
CORPUS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "E089_expanded_textual_corpus", "results", "nusantara_corpus_v3.json"
)
# Fallback to v2 if v3 doesn't exist yet
if not os.path.exists(CORPUS_PATH):
    CORPUS_PATH = os.path.join(
        os.path.dirname(__file__),
        "..", "E089_expanded_textual_corpus", "results", "nusantara_corpus_v2.json"
    )

def load_corpus():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# EXPERIMENT 1: Sentence-BERT Embeddings + Semantic Similarity
# ============================================================================
def exp1_semantic_similarity(corpus, output_dir):
    """
    Embed all passage texts using Sentence-BERT (all-MiniLM-L6-v2).
    Compute pairwise cosine similarity. Question: do passages from
    DIFFERENT traditions about the SAME place/commodity cluster together?
    """
    print("\n" + "=" * 70)
    print("EXP 1: SENTENCE-BERT SEMANTIC SIMILARITY")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    # Load model on GPU
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    print(f"  Model loaded: all-MiniLM-L6-v2 (on CUDA)")

    texts = [r["passage_text"] for r in corpus]
    ref_ids = [r["ref_id"] for r in corpus]
    traditions = [r["tradition"] for r in corpus]

    # Encode
    print(f"  Encoding {len(texts)} passages...")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    print(f"  Embedding shape: {embeddings.shape}")

    # Cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Analysis 1: Within-tradition vs between-tradition similarity
    within_sims = []
    between_sims = []
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            if traditions[i] == traditions[j]:
                within_sims.append(sim_matrix[i][j])
            else:
                between_sims.append(sim_matrix[i][j])

    within_mean = np.mean(within_sims) if within_sims else 0
    between_mean = np.mean(between_sims) if between_sims else 0

    print(f"\n  Within-tradition similarity: {within_mean:.4f} (n={len(within_sims)})")
    print(f"  Between-tradition similarity: {between_mean:.4f} (n={len(between_sims)})")
    print(f"  Ratio: {within_mean/between_mean:.3f}")

    # Analysis 2: Find most similar cross-tradition pairs
    cross_pairs = []
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            if traditions[i] != traditions[j]:
                cross_pairs.append({
                    "ref_i": ref_ids[i],
                    "ref_j": ref_ids[j],
                    "trad_i": traditions[i],
                    "trad_j": traditions[j],
                    "similarity": float(sim_matrix[i][j]),
                    "source_i": corpus[i]["source_text"][:50],
                    "source_j": corpus[j]["source_text"][:50]
                })
    cross_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    print(f"\n  Top 10 most similar cross-tradition pairs:")
    for k, p in enumerate(cross_pairs[:10]):
        print(f"    {k+1}. {p['ref_i']} ({p['trad_i']}) ↔ {p['ref_j']} ({p['trad_j']}): {p['similarity']:.4f}")
        print(f"       {p['source_i']} | {p['source_j']}")

    # Analysis 3: Tradition-level centroids
    print(f"\n  Tradition centroid similarities:")
    trad_centroids = {}
    for trad in set(traditions):
        idxs = [i for i, t in enumerate(traditions) if t == trad]
        trad_centroids[trad] = np.mean(embeddings[idxs], axis=0)

    trad_names = sorted(trad_centroids.keys())
    centroid_matrix = cosine_similarity(np.array([trad_centroids[t] for t in trad_names]))

    for i, ti in enumerate(trad_names):
        for j, tj in enumerate(trad_names):
            if i < j:
                print(f"    {ti:20s} ↔ {tj:20s}: {centroid_matrix[i][j]:.4f}")

    # Save
    result = {
        "method": "Sentence-BERT (all-MiniLM-L6-v2)",
        "n_passages": len(texts),
        "embedding_dim": int(embeddings.shape[1]),
        "within_tradition_similarity": {"mean": float(within_mean), "n": len(within_sims)},
        "between_tradition_similarity": {"mean": float(between_mean), "n": len(between_sims)},
        "ratio_within_between": float(within_mean / between_mean) if between_mean > 0 else None,
        "top_cross_tradition_pairs": cross_pairs[:20],
        "tradition_centroid_similarities": {
            f"{ti}-{tj}": float(centroid_matrix[i][j])
            for i, ti in enumerate(trad_names)
            for j, tj in enumerate(trad_names) if i < j
        }
    }

    # Save embeddings for downstream use
    np.save(os.path.join(output_dir, "passage_embeddings.npy"), embeddings)

    return result, embeddings


# ============================================================================
# EXPERIMENT 2: Cross-lingual Clustering (UMAP + HDBSCAN)
# ============================================================================
def exp2_clustering(corpus, embeddings, output_dir):
    """
    Cluster passages in embedding space using UMAP dimensionality reduction
    + HDBSCAN. Question: do clusters form by tradition (language/culture)
    or by content (place/commodity)?
    """
    print("\n" + "=" * 70)
    print("EXP 2: CROSS-LINGUAL CLUSTERING (UMAP + HDBSCAN)")
    print("=" * 70)

    import umap
    import hdbscan

    traditions = [r["tradition"] for r in corpus]
    ref_ids = [r["ref_id"] for r in corpus]

    # UMAP reduction
    print("  Running UMAP (n_components=2)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(10, len(corpus)-1),
                        min_dist=0.1, metric='cosine')
    umap_embeddings = reducer.fit_transform(embeddings)
    print(f"  UMAP shape: {umap_embeddings.shape}")

    # HDBSCAN clustering
    print("  Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean')
    cluster_labels = clusterer.fit_predict(umap_embeddings)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise}/{len(corpus)}")

    # Analysis: what's in each cluster?
    cluster_contents = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        cluster_contents[int(label)].append({
            "ref_id": ref_ids[i],
            "tradition": traditions[i],
            "source": corpus[i]["source_text"][:50],
            "date_ce": corpus[i]["date_ce"]
        })

    print(f"\n  Cluster composition:")
    for cid in sorted(cluster_contents.keys()):
        members = cluster_contents[cid]
        trad_dist = Counter(m["tradition"] for m in members)
        label = "NOISE" if cid == -1 else f"Cluster {cid}"
        print(f"    {label} ({len(members)} members): {dict(trad_dist)}")
        for m in members[:3]:
            print(f"      - {m['ref_id']} ({m['tradition']}, {m['date_ce']}): {m['source']}")

    # Key metric: are clusters tradition-pure or tradition-mixed?
    mixed_clusters = 0
    for cid, members in cluster_contents.items():
        if cid == -1:
            continue
        trads = set(m["tradition"] for m in members)
        if len(trads) > 1:
            mixed_clusters += 1

    pct_mixed = 100 * mixed_clusters / n_clusters if n_clusters > 0 else 0
    print(f"\n  Cross-tradition clusters: {mixed_clusters}/{n_clusters} ({pct_mixed:.0f}%)")
    if pct_mixed > 50:
        print("  → CONTENT-DRIVEN clustering (traditions mix by topic)")
    else:
        print("  → TRADITION-DRIVEN clustering (traditions stay separate)")

    result = {
        "method": "UMAP + HDBSCAN",
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "pct_mixed_clusters": float(pct_mixed),
        "clustering_type": "content-driven" if pct_mixed > 50 else "tradition-driven",
        "cluster_contents": {str(k): v for k, v in cluster_contents.items()},
        "umap_coordinates": [
            {"ref_id": ref_ids[i], "x": float(umap_embeddings[i][0]),
             "y": float(umap_embeddings[i][1]), "cluster": int(cluster_labels[i]),
             "tradition": traditions[i]}
            for i in range(len(corpus))
        ]
    }

    return result


# ============================================================================
# EXPERIMENT 3: Zero-shot NER with Transformer
# ============================================================================
def exp3_zero_shot_ner(corpus, output_dir):
    """
    Use a zero-shot classification pipeline to extract entities from
    passage texts WITHOUT any training data. Compare with hand-annotated
    entities from E089.

    This tests whether modern NLP can automatically identify ancient
    place names, commodities, and actors in translated passages.
    """
    print("\n" + "=" * 70)
    print("EXP 3: ZERO-SHOT NER (Transformer pipeline)")
    print("=" * 70)

    from transformers import pipeline

    # Use zero-shot classification to detect entity types
    classifier = pipeline("zero-shot-classification",
                         model="facebook/bart-large-mnli",
                         device=0)  # GPU
    print("  Model loaded: facebook/bart-large-mnli (on CUDA)")

    candidate_labels = [
        "geographic place or location",
        "trade commodity or product",
        "person or ruler",
        "ship or vessel",
        "kingdom or polity",
        "religious practice or ritual"
    ]

    results = []
    # Process a subset for speed
    sample = corpus[:20]  # first 20

    print(f"  Processing {len(sample)} passages...")
    for i, ref in enumerate(sample):
        text = ref["passage_text"]
        # Truncate long passages
        if len(text) > 500:
            text = text[:500]

        try:
            result = classifier(text, candidate_labels, multi_label=True)
            top_labels = [(l, s) for l, s in zip(result["labels"], result["scores"]) if s > 0.3]

            # Compare with hand annotations
            hand_types = set(e["type"] for e in ref["entities"])
            predicted_types = set()
            for label, score in top_labels:
                if "place" in label or "location" in label:
                    predicted_types.add("PLACE")
                elif "commodity" in label or "product" in label:
                    predicted_types.add("COMMODITY")
                elif "person" in label or "ruler" in label:
                    predicted_types.add("ACTOR")
                elif "ship" in label or "vessel" in label:
                    predicted_types.add("VESSEL")
                elif "kingdom" in label or "polity" in label:
                    predicted_types.add("POLITY")

            overlap = hand_types & predicted_types
            precision = len(overlap) / len(predicted_types) if predicted_types else 0
            recall = len(overlap) / len(hand_types) if hand_types else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                "ref_id": ref["ref_id"],
                "hand_types": list(hand_types),
                "predicted_types": list(predicted_types),
                "top_labels": [(l, round(s, 3)) for l, s in top_labels],
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3)
            })

            if (i + 1) % 5 == 0:
                print(f"    Processed {i+1}/{len(sample)}...")
        except Exception as e:
            print(f"    ERROR on {ref['ref_id']}: {e}")
            results.append({"ref_id": ref["ref_id"], "error": str(e)})

    # Aggregate
    valid = [r for r in results if "f1" in r]
    if valid:
        mean_p = np.mean([r["precision"] for r in valid])
        mean_r = np.mean([r["recall"] for r in valid])
        mean_f1 = np.mean([r["f1"] for r in valid])
        print(f"\n  Zero-shot NER type detection (vs hand annotations):")
        print(f"    Mean precision: {mean_p:.3f}")
        print(f"    Mean recall:    {mean_r:.3f}")
        print(f"    Mean F1:        {mean_f1:.3f}")
    else:
        mean_p = mean_r = mean_f1 = 0
        print(f"\n  No valid results.")

    print(f"\n  Per-passage results:")
    for r in valid[:10]:
        print(f"    {r['ref_id']}: F1={r['f1']:.3f} | hand={r['hand_types']} pred={r['predicted_types']}")

    return {
        "method": "Zero-shot classification (bart-large-mnli)",
        "n_processed": len(sample),
        "n_valid": len(valid),
        "mean_precision": float(mean_p),
        "mean_recall": float(mean_r),
        "mean_f1": float(mean_f1),
        "per_passage": results
    }


# ============================================================================
# EXPERIMENT 4: BERTopic — Latent Topic Discovery
# ============================================================================
def exp4_bertopic(corpus, embeddings, output_dir):
    """
    Use BERTopic to discover latent topics across ALL traditions.
    Question: what topics emerge, and do they align with our manual
    categorization (trade, navigation, kingdom, religion)?
    """
    print("\n" + "=" * 70)
    print("EXP 4: BERTopic LATENT TOPIC DISCOVERY")
    print("=" * 70)

    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    texts = [r["passage_text"] for r in corpus]
    traditions = [r["tradition"] for r in corpus]

    # Custom vectorizer for our domain
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    print("  Fitting BERTopic...")
    try:
        topic_model = BERTopic(
            embedding_model=None,  # we provide embeddings
            vectorizer_model=vectorizer,
            nr_topics="auto",
            min_topic_size=3,
            verbose=False
        )

        topics, probs = topic_model.fit_transform(texts, embeddings)

        topic_info = topic_model.get_topic_info()
        print(f"  Topics found: {len(topic_info) - 1}")  # -1 for outlier topic

        print(f"\n  Topic overview:")
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                label = "OUTLIER"
            else:
                label = f"Topic {tid}"
            count = row["Count"]
            # Get top words
            topic_words = topic_model.get_topic(tid)
            if topic_words:
                words = ", ".join([w for w, _ in topic_words[:5]])
            else:
                words = "(no words)"
            print(f"    {label} ({count} docs): {words}")

        # Which traditions map to which topics?
        print(f"\n  Tradition-topic matrix:")
        trad_topic = defaultdict(lambda: Counter())
        for i, (trad, topic) in enumerate(zip(traditions, topics)):
            trad_topic[trad][topic] += 1

        for trad in sorted(trad_topic.keys()):
            dist = dict(trad_topic[trad])
            print(f"    {trad:20s}: {dist}")

        result = {
            "method": "BERTopic",
            "n_topics": len(topic_info) - 1,
            "topics": [],
            "tradition_topic_matrix": {
                trad: dict(counts) for trad, counts in trad_topic.items()
            },
            "per_document_topics": [
                {"ref_id": corpus[i]["ref_id"], "topic": int(topics[i]),
                 "prob": float(probs[i]) if probs is not None and i < len(probs) else None}
                for i in range(len(corpus))
            ]
        }

        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            topic_words = topic_model.get_topic(tid)
            result["topics"].append({
                "id": int(tid),
                "count": int(row["Count"]),
                "words": [(w, float(s)) for w, s in (topic_words[:10] if topic_words else [])]
            })

        return result

    except Exception as e:
        print(f"  BERTopic FAILED: {e}")
        return {"method": "BERTopic", "status": "FAILED", "error": str(e)}


# ============================================================================
# EXPERIMENT 5: Semantic Convergence in Embedding Space
# ============================================================================
def exp5_semantic_convergence(corpus, embeddings, output_dir):
    """
    Test whether passages that reference the SAME geographic concept
    (e.g., Java, Sumatra, Golden Land) are closer in embedding space
    than random passages. This is the transformer-based equivalent of
    E088's Monte Carlo convergence test.
    """
    print("\n" + "=" * 70)
    print("EXP 5: SEMANTIC CONVERGENCE IN EMBEDDING SPACE")
    print("=" * 70)

    from sklearn.metrics.pairwise import cosine_similarity

    # Define concept groups (passages that reference the same thing)
    concept_groups = {
        "JAVA": [],
        "SUMATRA_GOLD": [],
        "CAMPHOR_BARUS": [],
        "SPICE_TRADE": [],
        "MARITIME_VOYAGE": []
    }

    java_terms = {"java", "yavadvipa", "iabadiu", "ye-po-ti", "shepo", "zabaj", "jawa"}
    gold_terms = {"chryse", "suvarnabhumi", "suvarnadvipa", "aurea", "golden", "gold", "emas"}
    camphor_terms = {"camphor", "karpura", "kafur", "kapur", "barus", "fansur"}
    spice_terms = {"clove", "nutmeg", "cinnamon", "pepper", "sandalwood", "aromatic", "spice"}
    voyage_terms = {"sail", "ship", "voyage", "merchant", "sea", "maritime", "boat", "embarked"}

    for i, ref in enumerate(corpus):
        text_lower = ref["passage_text"].lower()
        entity_texts = " ".join(e["text"].lower() for e in ref.get("entities", []))
        combined = text_lower + " " + entity_texts

        if any(t in combined for t in java_terms):
            concept_groups["JAVA"].append(i)
        if any(t in combined for t in gold_terms):
            concept_groups["SUMATRA_GOLD"].append(i)
        if any(t in combined for t in camphor_terms):
            concept_groups["CAMPHOR_BARUS"].append(i)
        if any(t in combined for t in spice_terms):
            concept_groups["SPICE_TRADE"].append(i)
        if any(t in combined for t in voyage_terms):
            concept_groups["MARITIME_VOYAGE"].append(i)

    print(f"  Concept groups:")
    for concept, indices in concept_groups.items():
        trads = [corpus[i]["tradition"] for i in indices]
        print(f"    {concept}: {len(indices)} passages from {len(set(trads))} traditions")

    # For each concept group, compute intra-group similarity vs random
    sim_matrix = cosine_similarity(embeddings)
    results = {}

    for concept, indices in concept_groups.items():
        if len(indices) < 2:
            print(f"    {concept}: SKIPPED (< 2 members)")
            continue

        # Intra-group mean similarity
        intra_sims = []
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                intra_sims.append(sim_matrix[indices[a]][indices[b]])
        intra_mean = np.mean(intra_sims)

        # Random baseline: sample same number of random pairs 1000 times
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

        # Check if cross-tradition within group
        cross_trad_sims = []
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                if corpus[indices[a]]["tradition"] != corpus[indices[b]]["tradition"]:
                    cross_trad_sims.append(sim_matrix[indices[a]][indices[b]])
        cross_trad_mean = np.mean(cross_trad_sims) if cross_trad_sims else None

        print(f"\n    {concept}:")
        print(f"      Intra-group similarity: {intra_mean:.4f}")
        print(f"      Random baseline:        {random_mean:.4f} ± {random_std:.4f}")
        print(f"      Z-score: {z_score:.2f}, p-value: {p_value:.4f}")
        if cross_trad_mean:
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
            "cross_tradition_similarity": float(cross_trad_mean) if cross_trad_mean else None,
            "verdict": verdict
        }

    # Summary
    converging = sum(1 for v in results.values() if v["verdict"] == "CONVERGES")
    total = len(results)
    print(f"\n  Summary: {converging}/{total} concept groups show semantic convergence (z > 1.96)")

    return {
        "method": "Embedding-space convergence (Monte Carlo baseline)",
        "concept_groups": results,
        "converging_groups": converging,
        "total_groups": total
    }


# ============================================================================
# EXPERIMENT 6: Cross-tradition Entailment
# ============================================================================
def exp6_cross_tradition_entailment(corpus, output_dir):
    """
    Use NLI (Natural Language Inference) to test whether one tradition's
    claims ENTAIL another's. E.g., does the Greek claim about Chryse
    entail the Indian claim about Suvarnabhumi?

    This is a novel application of NLI to historical text analysis.
    """
    print("\n" + "=" * 70)
    print("EXP 6: CROSS-TRADITION ENTAILMENT (NLI)")
    print("=" * 70)

    from transformers import pipeline

    nli = pipeline("zero-shot-classification",
                   model="facebook/bart-large-mnli",
                   device=0)
    print("  Model loaded: bart-large-mnli for NLI")

    # Select key hypothesis-evidence pairs across traditions
    test_pairs = [
        {
            "premise_id": "GRK-002",
            "hypothesis": "There was a wealthy golden land east of India where merchants traded spices and gold.",
            "label": "Periplus → Golden Land hypothesis"
        },
        {
            "premise_id": "IND-P03",
            "hypothesis": "Indian merchants regularly sailed to a golden land called Suvannabhumi to trade.",
            "label": "Sankha Jataka → Suvannabhumi trade"
        },
        {
            "premise_id": "CHN-002",
            "hypothesis": "Southeast Asian peoples had large ships with outriggers capable of carrying hundreds of people.",
            "label": "Wan Chen → SE Asian maritime technology"
        },
        {
            "premise_id": "CHEM-001",
            "hypothesis": "Southeast Asian tree resins were traded to Egypt before 500 BCE.",
            "label": "Saqqara → pre-classical SE Asian trade"
        },
        {
            "premise_id": "ARB-001",
            "hypothesis": "The island kingdom in Southeast Asia produced camphor, cloves, nutmeg, and sandalwood.",
            "label": "Sulayman → Nusantaran commodity exports"
        }
    ]

    # For each pair, test if OTHER traditions' passages entail the hypothesis
    results = []
    ref_lookup = {r["ref_id"]: r for r in corpus}

    for pair in test_pairs:
        premise_ref = ref_lookup.get(pair["premise_id"])
        if not premise_ref:
            continue

        hypothesis = pair["hypothesis"]
        print(f"\n  Testing: {pair['label']}")
        print(f"    Hypothesis: '{hypothesis[:80]}...'")

        # Test against all passages from OTHER traditions
        premise_trad = premise_ref["tradition"]
        entailment_scores = []

        for ref in corpus:
            if ref["tradition"] == premise_trad:
                continue
            if len(ref["passage_text"]) < 30:
                continue

            try:
                text = ref["passage_text"][:500]
                result = nli(text, [hypothesis], multi_label=False)
                # The score for the hypothesis
                score = result["scores"][0]
                entailment_scores.append({
                    "ref_id": ref["ref_id"],
                    "tradition": ref["tradition"],
                    "score": float(score)
                })
            except Exception as e:
                pass

        if entailment_scores:
            entailment_scores.sort(key=lambda x: x["score"], reverse=True)
            mean_score = np.mean([s["score"] for s in entailment_scores])
            max_score = max(s["score"] for s in entailment_scores)

            print(f"    Mean entailment score (other traditions): {mean_score:.3f}")
            print(f"    Max: {entailment_scores[0]['ref_id']} ({entailment_scores[0]['tradition']}): {max_score:.3f}")
            print(f"    Top 3:")
            for s in entailment_scores[:3]:
                print(f"      {s['ref_id']} ({s['tradition']}): {s['score']:.3f}")

            results.append({
                "pair": pair["label"],
                "premise_tradition": premise_trad,
                "hypothesis": hypothesis,
                "mean_score": float(mean_score),
                "max_score": float(max_score),
                "top_matches": entailment_scores[:5],
                "n_tested": len(entailment_scores)
            })

    # Summary
    if results:
        overall_mean = np.mean([r["mean_score"] for r in results])
        print(f"\n  Overall cross-tradition entailment: {overall_mean:.3f}")
        print(f"  (Baseline for unrelated texts is ~0.33 for 3-class NLI)")
        if overall_mean > 0.5:
            print(f"  → STRONG cross-tradition consistency")
        elif overall_mean > 0.4:
            print(f"  → MODERATE cross-tradition consistency")
        else:
            print(f"  → WEAK cross-tradition consistency")

    return {
        "method": "NLI cross-tradition entailment (bart-large-mnli)",
        "pairs_tested": len(results),
        "overall_mean_entailment": float(overall_mean) if results else None,
        "results": results
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("E090: TRANSFORMER-BASED NLP ON ANCIENT TEXTUAL CORPUS")
    print("=" * 70)

    corpus = load_corpus()
    print(f"Loaded corpus: {len(corpus)} references")

    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Exp 1: Sentence-BERT embeddings
    exp1_result, embeddings = exp1_semantic_similarity(corpus, output_dir)
    all_results["exp1_semantic_similarity"] = exp1_result

    # Exp 2: Clustering
    exp2_result = exp2_clustering(corpus, embeddings, output_dir)
    all_results["exp2_clustering"] = exp2_result

    # Exp 3: Zero-shot NER
    exp3_result = exp3_zero_shot_ner(corpus, output_dir)
    all_results["exp3_zero_shot_ner"] = exp3_result

    # Exp 4: BERTopic
    exp4_result = exp4_bertopic(corpus, embeddings, output_dir)
    all_results["exp4_bertopic"] = exp4_result

    # Exp 5: Semantic convergence
    exp5_result = exp5_semantic_convergence(corpus, embeddings, output_dir)
    all_results["exp5_semantic_convergence"] = exp5_result

    # Exp 6: Cross-tradition entailment
    exp6_result = exp6_cross_tradition_entailment(corpus, output_dir)
    all_results["exp6_cross_tradition_entailment"] = exp6_result

    # Save all results
    results_path = os.path.join(output_dir, "e090_all_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved: {results_path}")

    # Summary
    summary = {
        "experiment": "E090",
        "title": "Transformer-based NLP on Ancient Textual Corpus",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "corpus_size": len(corpus),
        "experiments_run": 6,
        "key_results": {
            "exp1_within_vs_between_ratio": exp1_result.get("ratio_within_between"),
            "exp2_clustering_type": exp2_result.get("clustering_type"),
            "exp2_n_clusters": exp2_result.get("n_clusters"),
            "exp3_ner_f1": exp3_result.get("mean_f1"),
            "exp4_n_topics": exp4_result.get("n_topics"),
            "exp5_converging_groups": exp5_result.get("converging_groups"),
            "exp5_total_groups": exp5_result.get("total_groups"),
            "exp6_mean_entailment": exp6_result.get("overall_mean_entailment")
        }
    }
    summary_path = os.path.join(output_dir, "e090_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {summary_path}")

    print("\n" + "=" * 70)
    print("E090 COMPLETE — 6 EXPERIMENTS ON GPU")
    print("=" * 70)


if __name__ == "__main__":
    main()
