#!/usr/bin/env python3
"""
E094 — DHARMA Semantic Search
First application of Sentence-BERT embeddings to Old Javanese epigraphy.

Embeds DHARMA inscription translations with SBERT, clusters with UMAP+HDBSCAN,
runs semantic queries, measures temporal drift, and checks indigenous vs Sanskrit content.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import hdbscan
import numpy as np
import umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DHARMA_DIR = SCRIPT_DIR.parent / "E023_ritual_screening" / "data" / "dharma" / "xml"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ── TEI-XML namespace ─────────────────────────────────────────────────────────
NS = {'tei': 'http://www.tei-c.org/ns/1.0'}


# ── Parsing (reused from E074) ─────────────────────────────────────────────────
def get_text(element):
    """Recursively extract all text content from an XML element."""
    texts = []
    if element.text:
        texts.append(element.text)
    for child in element:
        texts.extend(get_text(child))
        if child.tail:
            texts.append(child.tail)
    return texts


def parse_inscription(xml_path):
    """Parse a DHARMA EpiDoc TEI-XML inscription file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    result = {
        'filename': xml_path.name,
        'title': '',
        'lang': '',
        'date_text': '',
        'date_ce': None,
        'century': None,
        'edition_text': '',
        'translation_text': '',
        'commentary_text': '',
    }

    # Title
    title_el = root.find('.//tei:titleStmt/tei:title', NS)
    if title_el is not None and title_el.text:
        result['title'] = title_el.text.strip()
        ce_match = re.search(r'(\d{3,4})\s*CE', result['title'])
        if ce_match:
            result['date_ce'] = int(ce_match.group(1))
        saka_match = re.search(r'(\d{3,4})\s*Śaka', result['title'])
        if saka_match:
            result['date_ce'] = int(saka_match.group(1)) + 78

    # Language
    edition_div = root.find('.//tei:div[@type="edition"]', NS)
    if edition_div is not None:
        result['lang'] = edition_div.get('{http://www.w3.org/XML/1998/namespace}lang', '')

    # Edition text
    if edition_div is not None:
        result['edition_text'] = ' '.join(get_text(edition_div))

    # Translation text
    trans_div = root.find('.//tei:div[@type="translation"]', NS)
    if trans_div is not None:
        result['translation_text'] = ' '.join(get_text(trans_div))

    # Commentary text
    comm_div = root.find('.//tei:div[@type="commentary"]', NS)
    if comm_div is not None:
        result['commentary_text'] = ' '.join(get_text(comm_div))

    # Century
    if result['date_ce']:
        result['century'] = (result['date_ce'] - 1) // 100 + 1

    return result


# ── Indigenous vs Sanskrit vocabulary lists (from E074 logic) ──────────────────
# Core indigenous Austronesian terms found in Old Javanese inscriptions
INDIGENOUS_TERMS = {
    'sawah', 'huma', 'kebuan', 'tegal', 'sima', 'thani', 'wanua',
    'karaman', 'desa', 'banua', 'ladang', 'rama', 'buyut', 'kabayan',
    'wahuta', 'nayaka', 'pangalasan', 'juru', 'gusti', 'patih',
    'sawahan', 'parlak', 'gaga', 'kasuwakan', 'tambak', 'dawuhan',
    'tiruan', 'tuwiran', 'kayu', 'watu', 'gunung', 'tasik', 'sungai',
    'laut', 'pasir', 'tanjung', 'pulau', 'nusa', 'gili',
    'pande', 'undagi', 'tukang', 'panday', 'perahu', 'sampan',
    'rumah', 'balai', 'pasar', 'mandala', 'kamulan',
}

# Core Sanskrit-derived terms in Old Javanese inscriptions
SANSKRIT_TERMS = {
    'dharma', 'raja', 'deva', 'sri', 'mahārāja', 'maharaja', 'praśasti',
    'prasasti', 'anugraha', 'yajña', 'puja', 'mantra', 'mandala',
    'candi', 'vihara', 'stupa', 'lingga', 'yoni', 'tirtha',
    'brahmana', 'ksatriya', 'vaisya', 'sudra', 'bhumi', 'nagara',
    'grha', 'pura', 'ksetra', 'vana', 'giri', 'parvata',
    'nadi', 'sagara', 'samudra', 'karma', 'moksa', 'nirvana',
    'buddha', 'bodhisattva', 'sangha', 'sutra', 'tantra',
    'rakryan', 'samgat', 'pu', 'dyah', 'sang', 'hyang',
}


def count_vocabulary(text, wordset):
    """Count occurrences of vocabulary words in text (case-insensitive)."""
    text_lower = text.lower()
    count = 0
    for term in wordset:
        count += len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
    return count


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("E094 — DHARMA Semantic Search")
    print("First SBERT application to Old Javanese epigraphy")
    print("=" * 70)

    # ── Step 1: Parse all inscriptions ─────────────────────────────────────
    print("\n[1/7] Parsing DHARMA XML inscriptions...")
    xml_files = sorted(DHARMA_DIR.glob("*.xml"))
    print(f"  Found {len(xml_files)} XML files")

    all_inscriptions = []
    for xf in xml_files:
        parsed = parse_inscription(xf)
        if parsed is not None:
            all_inscriptions.append(parsed)

    print(f"  Successfully parsed: {len(all_inscriptions)}")

    # ── Step 2: Filter to inscriptions with translations ───────────────────
    print("\n[2/7] Filtering to inscriptions with translation text...")
    translated = [
        ins for ins in all_inscriptions
        if ins['translation_text'].strip()
    ]
    print(f"  Inscriptions with translations: {len(translated)}")
    if not translated:
        print("  ERROR: No inscriptions with translation text found. Exiting.")
        return

    # Show century distribution
    century_counts = Counter(ins['century'] for ins in translated if ins['century'])
    print("  Century distribution (translated):")
    for c in sorted(century_counts):
        print(f"    C{c}: {century_counts[c]} inscriptions")
    undated = sum(1 for ins in translated if ins['century'] is None)
    if undated:
        print(f"    Undated: {undated} inscriptions")

    # Language distribution
    lang_counts = Counter(ins['lang'] for ins in translated if ins['lang'])
    print("  Language distribution:")
    for lang, cnt in lang_counts.most_common():
        print(f"    {lang}: {cnt}")

    # ── Step 3: SBERT embedding ────────────────────────────────────────────
    print("\n[3/7] Loading SBERT model (all-MiniLM-L6-v2) on CUDA...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    texts = [ins['translation_text'].strip() for ins in translated]
    print(f"  Embedding {len(texts)} translation texts...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"  Embedding shape: {embeddings.shape}")

    # Save embeddings
    np.save(RESULTS_DIR / "dharma_embeddings.npy", embeddings)
    print(f"  Saved embeddings to results/dharma_embeddings.npy")

    # ── Step 4: UMAP + HDBSCAN clustering ──────────────────────────────────
    print("\n[4/7] UMAP dimensionality reduction + HDBSCAN clustering...")

    # UMAP to 2D for visualization, higher dim for clustering
    umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords_2d = umap_2d.fit_transform(embeddings)

    # UMAP to 5D for better clustering
    n_components_cluster = min(5, len(translated) - 2)
    umap_cluster = umap.UMAP(
        n_components=n_components_cluster, random_state=42,
        n_neighbors=15, min_dist=0.0
    )
    coords_cluster = umap_cluster.fit_transform(embeddings)

    # HDBSCAN
    min_cluster = max(3, len(translated) // 20)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster, min_samples=2)
    labels = clusterer.fit_predict(coords_cluster)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  HDBSCAN found {n_clusters} clusters, {n_noise} noise points")

    # Analyze cluster composition
    cluster_analysis = {}
    for cl in sorted(set(labels)):
        cl_name = f"cluster_{cl}" if cl >= 0 else "noise"
        members = [translated[i] for i in range(len(translated)) if labels[i] == cl]
        cl_centuries = Counter(m['century'] for m in members if m['century'])
        cl_langs = Counter(m['lang'] for m in members if m['lang'])

        # Check indigenous vs Sanskrit for this cluster
        total_indigenous = sum(count_vocabulary(m['edition_text'], INDIGENOUS_TERMS) for m in members)
        total_sanskrit = sum(count_vocabulary(m['edition_text'], SANSKRIT_TERMS) for m in members)

        cluster_analysis[cl_name] = {
            'size': len(members),
            'centuries': {str(k): v for k, v in sorted(cl_centuries.items())},
            'languages': dict(cl_langs.most_common()),
            'indigenous_term_count': total_indigenous,
            'sanskrit_term_count': total_sanskrit,
            'sample_titles': [m['title'][:80] for m in members[:5]],
        }
        print(f"\n  {cl_name} ({len(members)} members):")
        print(f"    Centuries: {dict(sorted(cl_centuries.items()))}")
        print(f"    Languages: {dict(cl_langs.most_common(3))}")
        print(f"    Indigenous terms: {total_indigenous}, Sanskrit terms: {total_sanskrit}")

    # Evaluate: do clusters align with century or content?
    # Compute purity by century
    total_correct = 0
    for cl in set(labels):
        if cl == -1:
            continue
        members = [translated[i] for i in range(len(translated)) if labels[i] == cl]
        cl_centuries = Counter(m['century'] for m in members if m['century'])
        if cl_centuries:
            total_correct += cl_centuries.most_common(1)[0][1]

    dated_clustered = sum(1 for i, ins in enumerate(translated) if labels[i] != -1 and ins['century'])
    century_purity = total_correct / dated_clustered if dated_clustered > 0 else 0
    print(f"\n  Century purity of clusters: {century_purity:.3f}")
    print(f"  (1.0 = clusters perfectly align with centuries, lower = content-based)")

    # Save cluster results
    cluster_output = {
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'century_purity': round(century_purity, 4),
        'clusters': cluster_analysis,
        'umap_2d_coords': [[float(x), float(y)] for x, y in coords_2d],
        'labels': [int(l) for l in labels],
        'filenames': [ins['filename'] for ins in translated],
    }
    with open(RESULTS_DIR / "dharma_clusters.json", 'w', encoding='utf-8') as f:
        json.dump(cluster_output, f, indent=2, ensure_ascii=False)
    print(f"  Saved cluster analysis to results/dharma_clusters.json")

    # ── Step 5: Semantic queries ───────────────────────────────────────────
    print("\n[5/7] Running semantic queries...")

    queries = [
        "village administration and land grants",
        "mountain worship and sacred peaks",
        "water infrastructure irrigation dams",
        "royal genealogy and succession",
        "tax collection and economic regulation",
        "Buddhist monastery and religious donation",
        "volcanic landscape fire mountain",
    ]

    query_embeddings = model.encode(queries, convert_to_numpy=True)
    sim_matrix = cosine_similarity(query_embeddings, embeddings)

    query_results = {}
    for qi, query in enumerate(queries):
        sims = sim_matrix[qi]
        top_indices = np.argsort(sims)[::-1][:10]

        hits = []
        centuries_in_hits = []
        for idx in top_indices:
            ins = translated[idx]
            hit = {
                'rank': len(hits) + 1,
                'filename': ins['filename'],
                'title': ins['title'][:100],
                'similarity': round(float(sims[idx]), 4),
                'century': ins['century'],
                'lang': ins['lang'],
                'translation_snippet': ins['translation_text'][:200].strip(),
            }
            hits.append(hit)
            if ins['century']:
                centuries_in_hits.append(ins['century'])

        cross_century = len(set(centuries_in_hits)) > 1
        century_spread = sorted(set(centuries_in_hits)) if centuries_in_hits else []

        query_results[query] = {
            'top_10': hits,
            'cross_century': cross_century,
            'centuries_represented': century_spread,
            'mean_similarity': round(float(np.mean([h['similarity'] for h in hits])), 4),
        }

        print(f"\n  Query: \"{query}\"")
        print(f"    Mean similarity: {query_results[query]['mean_similarity']:.4f}")
        print(f"    Centuries: {century_spread} ({'cross-century' if cross_century else 'single century'})")
        for h in hits[:3]:
            print(f"    #{h['rank']}: {h['title'][:60]}... (sim={h['similarity']:.3f}, C{h['century']})")

    with open(RESULTS_DIR / "semantic_queries.json", 'w', encoding='utf-8') as f:
        json.dump(query_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved query results to results/semantic_queries.json")

    # ── Step 6: Temporal semantic drift ────────────────────────────────────
    print("\n[6/7] Computing temporal semantic drift (century centroids)...")

    century_embeddings = defaultdict(list)
    for i, ins in enumerate(translated):
        if ins['century']:
            century_embeddings[ins['century']].append(embeddings[i])

    centroids = {}
    for century in sorted(century_embeddings):
        embs = np.array(century_embeddings[century])
        centroids[century] = embs.mean(axis=0)
        print(f"  C{century}: {len(century_embeddings[century])} inscriptions")

    # Compute pairwise centroid distances
    sorted_centuries = sorted(centroids.keys())
    drift_data = {
        'century_sizes': {str(c): len(century_embeddings[c]) for c in sorted_centuries},
        'consecutive_distances': {},
        'all_pairwise_distances': {},
    }

    print("\n  Consecutive century centroid distances (cosine):")
    for i in range(len(sorted_centuries) - 1):
        c1, c2 = sorted_centuries[i], sorted_centuries[i + 1]
        dist = 1.0 - float(cosine_similarity(
            centroids[c1].reshape(1, -1),
            centroids[c2].reshape(1, -1)
        )[0, 0])
        drift_data['consecutive_distances'][f"C{c1}->C{c2}"] = round(dist, 4)
        print(f"    C{c1} -> C{c2}: {dist:.4f}")

    # All pairwise
    for i, c1 in enumerate(sorted_centuries):
        for c2 in sorted_centuries[i + 1:]:
            dist = 1.0 - float(cosine_similarity(
                centroids[c1].reshape(1, -1),
                centroids[c2].reshape(1, -1)
            )[0, 0])
            drift_data['all_pairwise_distances'][f"C{c1}-C{c2}"] = round(dist, 4)

    # Key comparison: pre-929 (C7-C9+early C10) vs post-929 (late C10-C14)
    pre_929 = [embeddings[i] for i, ins in enumerate(translated)
               if ins['date_ce'] and ins['date_ce'] < 929]
    post_929 = [embeddings[i] for i, ins in enumerate(translated)
                if ins['date_ce'] and ins['date_ce'] >= 929]

    if pre_929 and post_929:
        pre_centroid = np.mean(pre_929, axis=0)
        post_centroid = np.mean(post_929, axis=0)
        divide_dist = 1.0 - float(cosine_similarity(
            pre_centroid.reshape(1, -1),
            post_centroid.reshape(1, -1)
        )[0, 0])
        drift_data['pre_post_929_distance'] = round(divide_dist, 4)
        drift_data['pre_929_count'] = len(pre_929)
        drift_data['post_929_count'] = len(post_929)
        print(f"\n  Pre-929 ({len(pre_929)}) vs Post-929 ({len(post_929)}) centroid distance: {divide_dist:.4f}")

    with open(RESULTS_DIR / "temporal_drift.json", 'w', encoding='utf-8') as f:
        json.dump(drift_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved temporal drift data to results/temporal_drift.json")

    # ── Step 7: Indigenous vs Sanskrit per cluster (summary) ───────────────
    print("\n[7/7] Indigenous vs Sanskrit vocabulary summary...")

    vocab_summary = {}
    for cl_name, cl_data in cluster_analysis.items():
        ratio = (cl_data['indigenous_term_count'] /
                 max(cl_data['sanskrit_term_count'], 1))
        vocab_summary[cl_name] = {
            'indigenous': cl_data['indigenous_term_count'],
            'sanskrit': cl_data['sanskrit_term_count'],
            'ratio_indigenous_to_sanskrit': round(ratio, 3),
        }
        print(f"  {cl_name}: indigenous={cl_data['indigenous_term_count']}, "
              f"sanskrit={cl_data['sanskrit_term_count']}, ratio={ratio:.3f}")

    # ── Final results summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    results_summary = {
        'experiment': 'E094_dharma_semantic_search',
        'description': 'First SBERT application to Old Javanese epigraphy',
        'total_xml_parsed': len(all_inscriptions),
        'with_translations': len(translated),
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dim': int(embeddings.shape[1]),
        'clustering': {
            'method': 'UMAP + HDBSCAN',
            'n_clusters': n_clusters,
            'n_noise': int(n_noise),
            'century_purity': round(century_purity, 4),
        },
        'semantic_queries': {
            query: {
                'mean_similarity': qr['mean_similarity'],
                'cross_century': qr['cross_century'],
                'centuries': qr['centuries_represented'],
            }
            for query, qr in query_results.items()
        },
        'temporal_drift': drift_data,
        'vocabulary_analysis': vocab_summary,
        'output_files': [
            'results/dharma_embeddings.npy',
            'results/dharma_clusters.json',
            'results/semantic_queries.json',
            'results/temporal_drift.json',
            'results/e094_results.json',
        ],
    }

    with open(RESULTS_DIR / "e094_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Total inscriptions parsed: {len(all_inscriptions)}")
    print(f"  With translations: {len(translated)}")
    print(f"  Clusters found: {n_clusters} (noise: {n_noise})")
    print(f"  Century purity: {century_purity:.3f}")
    print(f"  Queries run: {len(queries)}")
    print(f"\n  All results saved to {RESULTS_DIR}/")
    print("=" * 70)
    print("E094 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
