"""
E095 — Cross-Lingual XLM-R on Original Old Javanese Inscriptions
=================================================================
First application of multilingual transformer to original-language
Old Javanese epigraphy. Addresses P16 limitation #1.

Compares XLM-R on original text vs SBERT on English translations
for the 112 inscriptions that have both.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

# Optional imports for clustering
try:
    import umap
    import hdbscan
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

print("=" * 70)
print("E095 — CROSS-LINGUAL XLM-R ON ORIGINAL OLD JAVANESE INSCRIPTIONS")
print("=" * 70)

# --- 1. Parse DHARMA inscriptions ---
print("\n[1/7] Parsing DHARMA XML inscriptions...")
xml_dir = Path("experiments/E023_ritual_screening/data/dharma/xml")
ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

inscriptions = []
for f in sorted(xml_dir.glob("*.xml")):
    try:
        tree = ET.parse(f)
        root = tree.getroot()
        body = root.find('.//tei:body', ns)
        if not body:
            continue

        divs = body.findall('.//tei:div', ns)
        orig_text = ''
        trans_text = ''
        lang = ''
        title = ''

        # Get title
        title_el = root.find('.//tei:titleStmt/tei:title', ns)
        if title_el is not None and title_el.text:
            title = title_el.text.strip()

        # Get date
        date_el = root.find('.//tei:origin/tei:origDate', ns)
        date_text = ''
        century = ''
        if date_el is not None:
            when = date_el.get('when', '') or date_el.get('notBefore', '')
            if when:
                date_text = when
                try:
                    year = int(when.split('-')[0])
                    century = f"C{(year - 1) // 100 + 1}" if year > 0 else ''
                except:
                    pass

        for div in divs:
            div_type = div.get('type', '')
            if div_type == 'edition':
                orig_text = ET.tostring(div, encoding='unicode', method='text')
                orig_text = re.sub(r'\s+', ' ', orig_text).strip()
                lang_attr = div.get('{http://www.w3.org/XML/1998/namespace}lang', '')
                if lang_attr:
                    lang = lang_attr
            elif div_type == 'translation':
                trans_text = ET.tostring(div, encoding='unicode', method='text')
                trans_text = re.sub(r'\s+', ' ', trans_text).strip()

        if len(orig_text) > 50:
            inscriptions.append({
                'id': f.stem,
                'title': title,
                'lang': lang,
                'century': century,
                'date': date_text,
                'original': orig_text,
                'translation': trans_text if len(trans_text) > 50 else '',
                'has_both': len(orig_text) > 50 and len(trans_text) > 50,
            })
    except Exception as e:
        pass

print(f"  Inscriptions with original text: {len(inscriptions)}")
paired = [i for i in inscriptions if i['has_both']]
print(f"  With both original + translation: {len(paired)}")
lang_dist = {}
for i in inscriptions:
    lang_dist[i['lang']] = lang_dist.get(i['lang'], 0) + 1
print(f"  Languages: {lang_dist}")

# --- 2. Load XLM-RoBERTa ---
print("\n[2/7] Loading XLM-RoBERTa-base...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

xlmr_name = "xlm-roberta-base"
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_name)
xlmr_model = AutoModel.from_pretrained(xlmr_name).to(device)
xlmr_model.eval()

def encode_xlmr(texts, batch_size=16):
    """Encode texts using XLM-R mean pooling."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = xlmr_tokenizer(batch, padding=True, truncation=True,
                                  max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = xlmr_model(**encoded)
            # Mean pooling over non-padding tokens
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

# --- 3. Encode original texts with XLM-R ---
print("\n[3/7] Encoding original texts with XLM-R...")
orig_texts = [i['original'][:2000] for i in inscriptions]  # Truncate long texts
xlmr_embeddings = encode_xlmr(orig_texts)
print(f"  XLM-R embeddings shape: {xlmr_embeddings.shape}")

# Save embeddings
np.save("experiments/E095_crosslingual_xlmr/results/xlmr_embeddings.npy", xlmr_embeddings)

# --- 4. Load SBERT for comparison (English translations) ---
print("\n[4/7] Loading SBERT for translation comparison...")
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))

# Encode translations for paired inscriptions
paired_indices = [i for i, ins in enumerate(inscriptions) if ins['has_both']]
paired_trans = [inscriptions[i]['translation'][:2000] for i in paired_indices]
sbert_paired = sbert_model.encode(paired_trans, show_progress_bar=False)
xlmr_paired = xlmr_embeddings[paired_indices]

print(f"  Paired inscriptions: {len(paired_indices)}")
print(f"  SBERT translation embeddings: {sbert_paired.shape}")
print(f"  XLM-R original embeddings: {xlmr_paired.shape}")

# --- 5. Compare XLM-R vs SBERT ---
print("\n[5/7] Comparing XLM-R (original) vs SBERT (translation)...")

# Compute similarity matrices
xlmr_sim = cosine_similarity(xlmr_paired)
sbert_sim = cosine_similarity(sbert_paired)

# Extract upper triangle (excluding diagonal)
n = len(paired_indices)
triu_idx = np.triu_indices(n, k=1)
xlmr_flat = xlmr_sim[triu_idx]
sbert_flat = sbert_sim[triu_idx]

# Spearman rank correlation
rho, pval = spearmanr(xlmr_flat, sbert_flat)
print(f"  Spearman rho (XLM-R vs SBERT): {rho:.4f} (p={pval:.2e})")
print(f"  N pairs: {len(xlmr_flat)}")

# Mean similarities
print(f"  XLM-R mean similarity: {xlmr_flat.mean():.4f}")
print(f"  SBERT mean similarity: {sbert_flat.mean():.4f}")

# Per-language analysis
lang_results = {}
for lang_code in ['kaw-Latn', 'san-Latn']:
    lang_mask = [i for i, idx in enumerate(paired_indices) if inscriptions[idx]['lang'] == lang_code]
    if len(lang_mask) > 5:
        lang_xlmr_sim = cosine_similarity(xlmr_paired[lang_mask])
        lang_sbert_sim = cosine_similarity(sbert_paired[lang_mask])
        lang_triu = np.triu_indices(len(lang_mask), k=1)
        lang_rho, lang_p = spearmanr(lang_xlmr_sim[lang_triu], lang_sbert_sim[lang_triu])
        lang_results[lang_code] = {
            'n': len(lang_mask),
            'rho': float(lang_rho),
            'p': float(lang_p),
            'xlmr_mean_sim': float(lang_xlmr_sim[lang_triu].mean()),
            'sbert_mean_sim': float(lang_sbert_sim[lang_triu].mean()),
        }
        print(f"  {lang_code} (n={len(lang_mask)}): rho={lang_rho:.4f}, p={lang_p:.2e}")

# --- 6. UMAP + HDBSCAN clustering on XLM-R ---
print("\n[6/7] Clustering XLM-R embeddings...")

if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_coords = reducer.fit_transform(xlmr_embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=2)
    cluster_labels = clusterer.fit_predict(umap_coords)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = sum(1 for c in cluster_labels if c == -1)
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise: {n_noise}/{len(cluster_labels)}")

    # Check language composition of clusters
    cluster_lang = {}
    for i, cl in enumerate(cluster_labels):
        if cl == -1:
            continue
        cl_key = str(cl)
        if cl_key not in cluster_lang:
            cluster_lang[cl_key] = {}
        lang = inscriptions[i]['lang']
        cluster_lang[cl_key][lang] = cluster_lang[cl_key].get(lang, 0) + 1

    # Cross-language clusters (contain multiple languages)
    cross_lang = sum(1 for cl, langs in cluster_lang.items() if len(langs) > 1)
    print(f"  Cross-language clusters: {cross_lang}/{n_clusters} ({100*cross_lang/max(n_clusters,1):.0f}%)")

    for cl, langs in sorted(cluster_lang.items(), key=lambda x: -sum(x[1].values())):
        print(f"    Cluster {cl} ({sum(langs.values())} members): {langs}")

    # Century purity (for dated inscriptions)
    dated_mask = [i for i, ins in enumerate(inscriptions) if ins['century']]
    if dated_mask:
        century_labels = [inscriptions[i]['century'] for i in dated_mask]
        cluster_at_dated = [cluster_labels[i] for i in dated_mask]
        # Purity = fraction where cluster mates share the same century
        from collections import Counter
        purity_scores = []
        for i, (cl, cent) in enumerate(zip(cluster_at_dated, century_labels)):
            if cl == -1:
                continue
            same_cluster = [(cluster_at_dated[j], century_labels[j]) for j in range(len(dated_mask))
                           if cluster_at_dated[j] == cl and j != i]
            if same_cluster:
                same_cent = sum(1 for _, c in same_cluster if c == cent)
                purity_scores.append(same_cent / len(same_cluster))
        if purity_scores:
            print(f"  Century purity: {np.mean(purity_scores):.3f}")

    cluster_data = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'cross_language_pct': float(100 * cross_lang / max(n_clusters, 1)),
        'cluster_composition': cluster_lang,
        'umap_coords': [{'id': inscriptions[i]['id'], 'x': float(umap_coords[i, 0]),
                         'y': float(umap_coords[i, 1]), 'cluster': int(cluster_labels[i]),
                         'lang': inscriptions[i]['lang']} for i in range(len(inscriptions))],
    }
else:
    print("  SKIP: umap/hdbscan not available")
    cluster_data = {}

# --- 7. Cross-lingual semantic queries ---
print("\n[7/7] Cross-lingual semantic queries (English query → Old Javanese corpus)...")

queries = [
    "village administration and land grants",
    "mountain worship and sacred peaks",
    "water infrastructure irrigation dams",
    "royal genealogy and succession",
    "tax collection and economic regulation",
    "Buddhist monastery and religious donation",
    "volcanic landscape fire mountain",
]

# Encode queries with XLM-R (English)
query_embeddings = encode_xlmr(queries)

# Compute similarity: English queries × Old Javanese corpus
query_results = {}
for qi, query in enumerate(queries):
    sims = cosine_similarity(query_embeddings[qi:qi+1], xlmr_embeddings)[0]
    top10_idx = np.argsort(sims)[-10:][::-1]

    mean_sim = float(sims.mean())
    query_results[query] = {
        'mean_similarity': mean_sim,
        'top_10': [{'id': inscriptions[i]['id'], 'lang': inscriptions[i]['lang'],
                    'similarity': float(sims[i]),
                    'century': inscriptions[i]['century']} for i in top10_idx],
    }
    print(f"  '{query[:45]}': mean={mean_sim:.4f}")

# Compare with E094 (SBERT on translations)
print("\n  Comparison with E094 (SBERT on translations):")
print(f"  {'Query':<45} {'XLM-R':>8} {'SBERT':>8} {'Delta':>8}")
print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")

try:
    with open("experiments/E094_dharma_semantic_search/results/semantic_queries.json", 'r') as f:
        e094_queries = json.load(f)
    for query in queries:
        xlmr_mean = query_results[query]['mean_similarity']
        sbert_mean = e094_queries.get(query, {}).get('mean_similarity', 0)
        delta = xlmr_mean - sbert_mean
        print(f"  {query[:45]:<45} {xlmr_mean:>8.4f} {sbert_mean:>8.4f} {delta:>+8.4f}")
except FileNotFoundError:
    print("  E094 results not found for comparison")

# --- Save results ---
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'meta': {
        'experiment': 'E095',
        'date': '2026-03-17',
        'model': 'xlm-roberta-base',
        'comparison_model': 'all-MiniLM-L6-v2 (SBERT)',
        'n_inscriptions': len(inscriptions),
        'n_paired': len(paired_indices),
        'device': str(device),
    },
    'paired_comparison': {
        'spearman_rho': float(rho),
        'spearman_p': float(pval),
        'xlmr_mean_sim': float(xlmr_flat.mean()),
        'sbert_mean_sim': float(sbert_flat.mean()),
        'n_pairs': int(len(xlmr_flat)),
        'per_language': lang_results,
    },
    'clustering': cluster_data,
    'semantic_queries': query_results,
}

out_path = "experiments/E095_crosslingual_xlmr/results/e095_results.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"  Results saved to {out_path}")

# --- Summary ---
print("\n" + "=" * 70)
print("E095 SUMMARY")
print("=" * 70)
print(f"  Inscriptions encoded (XLM-R): {len(inscriptions)}")
print(f"  Paired comparison (XLM-R vs SBERT): {len(paired_indices)} inscriptions")
print(f"  Spearman rho: {rho:.4f} (p={pval:.2e})")
print(f"  XLM-R mean sim: {xlmr_flat.mean():.4f} | SBERT mean sim: {sbert_flat.mean():.4f}")
if cluster_data:
    print(f"  Clusters: {cluster_data.get('n_clusters', '?')}")
    print(f"  Cross-language: {cluster_data.get('cross_language_pct', '?'):.0f}%")
print(f"  Volcanic query (XLM-R): {query_results.get('volcanic landscape fire mountain', {}).get('mean_similarity', 0):.4f}")
print("=" * 70)
