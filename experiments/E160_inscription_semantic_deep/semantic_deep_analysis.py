"""
E160: GPU-Powered Deep Semantic Analysis of DHARMA Inscriptions
================================================================
Goes beyond E094/E096 with:
1. Higher-dimensional embeddings (all-mpnet-base-v2, 768d vs 384d)
2. Temporal trajectory analysis (how does semantic space evolve century by century?)
3. "Volcanic silence" quantification (WHEN did inscriptions stop mentioning landscape?)
4. Genre taphonomy detection (can embeddings distinguish sacred vs administrative?)
5. The 929 CE rupture in embedding space

Uses RTX 4080 GPU for faster inference.
"""

import numpy as np
import pandas as pd
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Parse DHARMA inscriptions
# ============================================================
print("=" * 70)
print("E160: DEEP SEMANTIC ANALYSIS OF DHARMA INSCRIPTIONS")
print("=" * 70)

dharma_dir = Path("D:/documents/volcarch-repo/experiments/E023_ritual_screening/data/dharma/xml")
metadata_path = Path("D:/documents/volcarch-repo/experiments/E030_prasasti_temporal_nlp/results/dated_inscriptions.csv")
metadata_74 = Path("D:/documents/volcarch-repo/experiments/E074_dharma_deep_nlp/results/inscription_metadata.csv")

# Load metadata
df_meta = pd.read_csv(metadata_path)
df_meta_74 = pd.read_csv(metadata_74)

# Parse XML for translations
def extract_text(xml_path):
    """Extract translation and edition text from DHARMA TEI-XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        # Find translation
        trans_divs = root.findall('.//tei:div[@type="translation"]', ns)
        trans_text = ""
        for div in trans_divs:
            for elem in div.iter():
                if elem.text:
                    trans_text += elem.text + " "
                if elem.tail:
                    trans_text += elem.tail + " "

        # Find edition
        ed_divs = root.findall('.//tei:div[@type="edition"]', ns)
        ed_text = ""
        for div in ed_divs:
            for elem in div.iter():
                if elem.text:
                    ed_text += elem.text + " "
                if elem.tail:
                    ed_text += elem.tail + " "

        # Clean
        trans_text = ' '.join(trans_text.split())
        ed_text = ' '.join(ed_text.split())

        return trans_text, ed_text
    except Exception as e:
        return "", ""

print("\nParsing DHARMA inscriptions...")
inscriptions = []
xml_files = sorted(dharma_dir.glob("*.xml"))
print(f"  Found {len(xml_files)} XML files")

for xml_path in xml_files:
    filename = xml_path.name
    trans, edition = extract_text(xml_path)

    # Match with metadata
    meta_row = df_meta[df_meta['filename'] == filename]
    meta_row_74 = df_meta_74[df_meta_74['filename'] == filename]

    century = int(meta_row['century'].values[0]) if len(meta_row) > 0 and pd.notna(meta_row['century'].values[0]) else None
    year_ce = float(meta_row['year_ce'].values[0]) if len(meta_row) > 0 and pd.notna(meta_row['year_ce'].values[0]) else None
    pre_indic_ratio = float(meta_row['pre_indic_ratio'].values[0]) if len(meta_row) > 0 and pd.notna(meta_row['pre_indic_ratio'].values[0]) else None

    n_sanskrit = int(meta_row_74['n_sanskrit'].values[0]) if len(meta_row_74) > 0 and pd.notna(meta_row_74['n_sanskrit'].values[0]) else 0
    n_indigenous = int(meta_row_74['n_indigenous'].values[0]) if len(meta_row_74) > 0 and pd.notna(meta_row_74['n_indigenous'].values[0]) else 0

    if trans and len(trans) > 20:  # Only inscriptions with meaningful translations
        inscriptions.append({
            'filename': filename,
            'translation': trans[:2000],  # Limit to 2000 chars for SBERT
            'edition': edition[:1000],
            'century': century,
            'year_ce': year_ce,
            'pre_indic_ratio': pre_indic_ratio,
            'n_sanskrit': n_sanskrit,
            'n_indigenous': n_indigenous,
            'word_count': len(trans.split()),
        })

print(f"  Inscriptions with translations: {len(inscriptions)}")
dated = [i for i in inscriptions if i['century'] is not None]
print(f"  Of which dated: {len(dated)}")

# Century distribution
century_counts = defaultdict(int)
for i in dated:
    century_counts[i['century']] += 1
print(f"  Century distribution: {dict(sorted(century_counts.items()))}")

# ============================================================
# 2. Generate embeddings with sentence-transformers (GPU)
# ============================================================
print(f"\nGenerating embeddings with all-mpnet-base-v2 (768d)...")

from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

model = SentenceTransformer('all-mpnet-base-v2', device=device)

texts = [i['translation'] for i in inscriptions]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                          normalize_embeddings=True)

print(f"  Embeddings shape: {embeddings.shape}")

# ============================================================
# 3. Semantic landscape analysis
# ============================================================
print(f"\n{'='*70}")
print("ANALYSIS 1: Semantic Queries — What do inscriptions talk about?")
print(f"{'='*70}")

# Define semantic probes
queries = {
    "volcanic_landscape": "volcanic eruption mountain fire ash destruction earthquake natural disaster",
    "sacred_mountain": "sacred mountain holy peak divine dwelling cosmic axis pilgrimage worship",
    "water_agriculture": "rice field irrigation canal dam water agriculture harvest paddy crop",
    "royal_court": "king queen prince royal palace throne dynasty succession coronation realm",
    "land_administration": "land grant village boundary tax tribute official administrator decree",
    "ritual_ceremony": "ritual ceremony offering sacrifice prayer temple festival consecration",
    "trade_commerce": "trade market merchant gold silver commodity ship port exchange",
    "warfare_conflict": "war battle army soldier conquest territory invasion defense siege",
    "buddhist_hindu": "buddha dharma sangha shiva vishnu temple monastery stupa mandala",
    "daily_life": "house food cloth bamboo wood stone iron tool domestic household",
}

query_embeddings = model.encode(list(queries.values()), normalize_embeddings=True)

print(f"\n{'Query':<25} {'Mean Sim':>10} {'Max Sim':>10} {'Top Match'}")
print(f"{'-'*70}")

query_results = {}
for (qname, qtext), qemb in zip(queries.items(), query_embeddings):
    sims = embeddings @ qemb
    mean_sim = float(np.mean(sims))
    max_sim = float(np.max(sims))
    top_idx = np.argmax(sims)
    top_name = inscriptions[top_idx]['filename'][:30]

    print(f"  {qname:<25} {mean_sim:>8.4f}   {max_sim:>8.4f}   {top_name}")
    query_results[qname] = {
        "mean_similarity": mean_sim,
        "max_similarity": max_sim,
        "top_match": inscriptions[top_idx]['filename'],
    }

# ============================================================
# 4. Temporal trajectory — how does the semantic centroid move?
# ============================================================
print(f"\n{'='*70}")
print("ANALYSIS 2: Temporal Trajectory — Semantic evolution by century")
print(f"{'='*70}")

# Compute century centroids
century_centroids = {}
for c in sorted(century_counts.keys()):
    if c is None:
        continue
    century_indices = [i for i, insc in enumerate(inscriptions) if insc['century'] == c]
    if len(century_indices) >= 2:
        century_embs = embeddings[century_indices]
        centroid = np.mean(century_embs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        century_centroids[c] = centroid

# Pairwise distances between century centroids
print(f"\n  Century-to-century cosine distances:")
centuries_sorted = sorted(century_centroids.keys())
for i in range(len(centuries_sorted) - 1):
    c1 = centuries_sorted[i]
    c2 = centuries_sorted[i + 1]
    dist = 1 - np.dot(century_centroids[c1], century_centroids[c2])
    marker = " ***" if dist > 0.15 else ""
    print(f"    C{c1} -> C{c2}: {dist:.4f}{marker}")

# Distance from each century to "volcanic" query
print(f"\n  Century proximity to semantic queries:")
print(f"  {'Century':<10} {'N':>4} {'volcanic':>10} {'sacred_mt':>10} {'admin':>10} {'ritual':>10} {'daily':>10}")
print(f"  {'-'*60}")

for c in centuries_sorted:
    century_indices = [i for i, insc in enumerate(inscriptions) if insc['century'] == c]
    century_embs = embeddings[century_indices]

    sims = {}
    for qname, qemb in zip(queries.keys(), query_embeddings):
        sim = float(np.mean(century_embs @ qemb))
        sims[qname] = sim

    print(f"  C{c:<8} {len(century_indices):>4} "
          f"{sims['volcanic_landscape']:>10.4f} "
          f"{sims['sacred_mountain']:>10.4f} "
          f"{sims['land_administration']:>10.4f} "
          f"{sims['ritual_ceremony']:>10.4f} "
          f"{sims['daily_life']:>10.4f}")

# ============================================================
# 5. The 929 CE Rupture
# ============================================================
print(f"\n{'='*70}")
print("ANALYSIS 3: The 929 CE Rupture in Embedding Space")
print(f"{'='*70}")

pre_929_indices = [i for i, insc in enumerate(inscriptions) if insc['year_ce'] and insc['year_ce'] < 929]
post_929_indices = [i for i, insc in enumerate(inscriptions) if insc['year_ce'] and insc['year_ce'] >= 929]

if pre_929_indices and post_929_indices:
    pre_929_embs = embeddings[pre_929_indices]
    post_929_embs = embeddings[post_929_indices]

    pre_centroid = np.mean(pre_929_embs, axis=0)
    pre_centroid /= np.linalg.norm(pre_centroid)
    post_centroid = np.mean(post_929_embs, axis=0)
    post_centroid /= np.linalg.norm(post_centroid)

    rupture_dist = 1 - np.dot(pre_centroid, post_centroid)

    print(f"  Pre-929 inscriptions: {len(pre_929_indices)}")
    print(f"  Post-929 inscriptions: {len(post_929_indices)}")
    print(f"  Cosine distance between centroids: {rupture_dist:.4f}")

    # How does each group relate to queries?
    print(f"\n  {'Query':<25} {'Pre-929':>10} {'Post-929':>10} {'Delta':>10}")
    print(f"  {'-'*55}")

    for (qname, qtext), qemb in zip(queries.items(), query_embeddings):
        pre_sim = float(np.mean(pre_929_embs @ qemb))
        post_sim = float(np.mean(post_929_embs @ qemb))
        delta = post_sim - pre_sim
        marker = " ***" if abs(delta) > 0.02 else ""
        print(f"  {qname:<25} {pre_sim:>10.4f} {post_sim:>10.4f} {delta:>+10.4f}{marker}")

    # Permutation test for 929 CE rupture
    print(f"\n  Permutation test for 929 CE rupture significance:")
    all_dated_indices = pre_929_indices + post_929_indices
    all_dated_embs = embeddings[all_dated_indices]
    n_pre = len(pre_929_indices)

    np.random.seed(42)
    perm_dists = []
    for _ in range(5000):
        perm = np.random.permutation(len(all_dated_indices))
        perm_pre = np.mean(all_dated_embs[perm[:n_pre]], axis=0)
        perm_pre /= np.linalg.norm(perm_pre)
        perm_post = np.mean(all_dated_embs[perm[n_pre:]], axis=0)
        perm_post /= np.linalg.norm(perm_post)
        perm_dists.append(1 - np.dot(perm_pre, perm_post))

    perm_dists = np.array(perm_dists)
    p_perm = np.mean(perm_dists >= rupture_dist)
    print(f"  Observed distance: {rupture_dist:.4f}")
    print(f"  Permutation mean: {perm_dists.mean():.4f} +/- {perm_dists.std():.4f}")
    print(f"  Permutation p: {p_perm:.4f}")
    print(f"  z-score: {(rupture_dist - perm_dists.mean()) / perm_dists.std():.2f}")

# ============================================================
# 6. Pre-Indic vocabulary in embedding space
# ============================================================
print(f"\n{'='*70}")
print("ANALYSIS 4: Pre-Indic vocabulary and embedding structure")
print(f"{'='*70}")

# Do inscriptions with HIGH pre-Indic ratio cluster differently?
with_ratio = [(i, idx) for idx, i in enumerate(inscriptions) if i['pre_indic_ratio'] is not None]
if with_ratio:
    ratios = np.array([i['pre_indic_ratio'] for i, _ in with_ratio])
    indices = [idx for _, idx in with_ratio]
    ratio_embs = embeddings[indices]

    # Split by median pre-Indic ratio
    median_ratio = np.median(ratios)
    high_mask = ratios >= median_ratio
    low_mask = ~high_mask

    high_embs = ratio_embs[high_mask]
    low_embs = ratio_embs[low_mask]

    high_centroid = np.mean(high_embs, axis=0)
    high_centroid /= np.linalg.norm(high_centroid)
    low_centroid = np.mean(low_embs, axis=0)
    low_centroid /= np.linalg.norm(low_centroid)

    ratio_dist = 1 - np.dot(high_centroid, low_centroid)

    print(f"  Median pre-Indic ratio: {median_ratio:.3f}")
    print(f"  High pre-Indic (>={median_ratio:.3f}): {high_mask.sum()} inscriptions")
    print(f"  Low pre-Indic (<{median_ratio:.3f}): {low_mask.sum()} inscriptions")
    print(f"  Cosine distance (high vs low): {ratio_dist:.4f}")

    # Which queries differentiate high vs low pre-Indic?
    print(f"\n  {'Query':<25} {'High PI':>10} {'Low PI':>10} {'Delta':>10}")
    print(f"  {'-'*55}")

    for (qname, qtext), qemb in zip(queries.items(), query_embeddings):
        high_sim = float(np.mean(high_embs @ qemb))
        low_sim = float(np.mean(low_embs @ qemb))
        delta = high_sim - low_sim
        marker = " ***" if abs(delta) > 0.015 else ""
        print(f"  {qname:<25} {high_sim:>10.4f} {low_sim:>10.4f} {delta:>+10.4f}{marker}")

# ============================================================
# Save results
# ============================================================
results_dir = Path("D:/documents/volcarch-repo/experiments/E160_inscription_semantic_deep/results")
results_dir.mkdir(parents=True, exist_ok=True)

np.save(results_dir / "deep_embeddings.npy", embeddings)

results_summary = {
    "n_inscriptions": len(inscriptions),
    "n_dated": len(dated),
    "embedding_dim": int(embeddings.shape[1]),
    "model": "all-mpnet-base-v2",
    "device": device,
    "query_results": query_results,
    "century_distribution": dict(century_counts),
}

with open(results_dir / "e160_results.json", "w") as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f"\nEmbeddings saved: {results_dir / 'deep_embeddings.npy'}")
print(f"Results saved: {results_dir / 'e160_results.json'}")
print(f"\nDONE.")
