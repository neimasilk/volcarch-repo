#!/usr/bin/env python3
"""
E029: Phonological Clustering of Consensus Substrate Candidates

Tests whether the 266 consensus substrates (from E028) cluster into
phonologically coherent word families, suggesting a shared pre-Austronesian
substrate layer rather than random lexical gaps.

Key analyses:
1. Pairwise Levenshtein distance matrix for all 266 substrates
2. Hierarchical + DBSCAN clustering
3. Cross-linguistic cognate detection (do substrate forms for the SAME concept
   look similar across languages?)
4. Semantic-phonological correlation (do semantic domains cluster phonologically?)

Author: VOLCARCH project
Date: 2026-03-10
"""

import sys
import io
import os
import csv
import json
import math
import random
from collections import defaultdict
from itertools import combinations

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BASE, "..", ".."))

CONSENSUS_CSV = os.path.join(REPO, "experiments", "E028_substrate_consensus", "results", "consensus_substrates.csv")
FEATURES_CSV = os.path.join(REPO, "experiments", "E027_ml_substrate_detection", "data", "features_matrix.csv")
FORMS_CSV = os.path.join(REPO, "experiments", "E022_linguistic_subtraction", "data", "abvd", "cldf", "forms.csv")
PARAMS_CSV = os.path.join(REPO, "experiments", "E022_linguistic_subtraction", "data", "abvd", "cldf", "parameters.csv")
LANGUAGES_CSV = os.path.join(REPO, "experiments", "E022_linguistic_subtraction", "data", "abvd", "cldf", "languages.csv")

RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Levenshtein Edit Distance (manual DP implementation) ─────────────────────
def levenshtein(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # insertion, deletion, substitution
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def normalized_levenshtein(s1, s2):
    """Normalized Levenshtein distance: 0 = identical, 1 = maximally different."""
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    return levenshtein(s1, s2) / max(len(s1), len(s2))


# ─── 1. Load Data ────────────────────────────────────────────────────────────
print("=" * 70)
print("E029: PHONOLOGICAL CLUSTERING OF CONSENSUS SUBSTRATES")
print("=" * 70)

# Load consensus substrates
print("\n[1] Loading consensus substrates...")
substrates = []
with open(CONSENSUS_CSV, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        substrates.append({
            'rank': int(row['rank']),
            'language': row['language'],
            'concept': row['concept'],
            'form': row['form'],
            'p_substrate': float(row['p_substrate']),
            'semantic_domain': row['semantic_domain'],
        })

print(f"  Loaded {len(substrates)} consensus substrates")
lang_counts = defaultdict(int)
for s in substrates:
    lang_counts[s['language']] += 1
for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
    print(f"    {lang}: {count}")

# Load concept names (parameter mapping)
print("\n  Loading concept parameters...")
concept_names = {}
with open(PARAMS_CSV, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        concept_names[row['ID']] = row['Name']

# Load original forms from ABVD for cross-linguistic comparison
print("  Loading ABVD forms for cross-linguistic analysis...")
# Map our language names to ABVD Language_IDs
LANG_ID_MAP = {
    'Muna': '27',
    'Bugis': '48',  # Buginese (Soppeng Dialect)
    'Makassar': '166',
    'Wolio': '192',
    'Toraja-Sadan': '226',  # Tae' (S.Toraja)
    'Tolaki': '674',
}
LANG_ID_REVERSE = {v: k for k, v in LANG_ID_MAP.items()}

abvd_forms = defaultdict(dict)  # {concept_id: {language_name: [forms]}}
with open(FORMS_CSV, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        lang_id = row['Language_ID']
        if lang_id in LANG_ID_REVERSE:
            lang_name = LANG_ID_REVERSE[lang_id]
            param_id = row['Parameter_ID']
            concept = concept_names.get(param_id, param_id)
            if lang_name not in abvd_forms[concept]:
                abvd_forms[concept][lang_name] = []
            abvd_forms[concept][lang_name].append(row['Value'])

print(f"  Loaded forms for {len(abvd_forms)} concepts across {len(LANG_ID_MAP)} languages")


# ─── 2. Phonological Distance Matrix ─────────────────────────────────────────
print("\n[2] Computing pairwise phonological distance matrix...")
n = len(substrates)
forms = [s['form'].lower().strip() for s in substrates]

# Compute full distance matrix
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        d = normalized_levenshtein(forms[i], forms[j])
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

print(f"  Distance matrix shape: {dist_matrix.shape}")
print(f"  Mean distance: {np.mean(dist_matrix[np.triu_indices(n, k=1)]):.4f}")
print(f"  Median distance: {np.median(dist_matrix[np.triu_indices(n, k=1)]):.4f}")
print(f"  Std distance: {np.std(dist_matrix[np.triu_indices(n, k=1)]):.4f}")
print(f"  Min non-zero distance: {np.min(dist_matrix[np.triu_indices(n, k=1)]):.4f}")
print(f"  Max distance: {np.max(dist_matrix[np.triu_indices(n, k=1)]):.4f}")


# ─── 3. Clustering ───────────────────────────────────────────────────────────
print("\n[3] Hierarchical & DBSCAN clustering...")

# 3a. Hierarchical agglomerative clustering (Ward's method)
# Ward's method requires condensed distance matrix
condensed = squareform(dist_matrix)
Z = linkage(condensed, method='ward')

# Find optimal k via silhouette score
print("\n  Silhouette scores for different k:")
silhouette_results = {}
best_k = 5
best_sil = -1
for k in range(5, 31):
    labels = fcluster(Z, t=k, criterion='maxclust')
    # Need at least 2 clusters with >1 member for silhouette
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        continue
    sil = silhouette_score(dist_matrix, labels, metric='precomputed')
    silhouette_results[k] = sil
    if sil > best_sil:
        best_sil = sil
        best_k = k

for k in sorted(silhouette_results.keys()):
    marker = " <-- BEST" if k == best_k else ""
    print(f"    k={k:2d}: silhouette={silhouette_results[k]:.4f}{marker}")

print(f"\n  Optimal k={best_k} (silhouette={best_sil:.4f})")

# Assign clusters with optimal k
ward_labels = fcluster(Z, t=best_k, criterion='maxclust')
for s, label in zip(substrates, ward_labels):
    s['ward_cluster'] = int(label)

# Summarize Ward clusters
print(f"\n  Ward cluster sizes (k={best_k}):")
cluster_sizes = defaultdict(list)
for s in substrates:
    cluster_sizes[s['ward_cluster']].append(s)
for cid in sorted(cluster_sizes.keys()):
    members = cluster_sizes[cid]
    langs_in = set(m['language'] for m in members)
    domains = defaultdict(int)
    for m in members:
        domains[m['semantic_domain']] += 1
    top_domain = max(domains.items(), key=lambda x: x[1])
    print(f"    Cluster {cid:2d}: {len(members):3d} members, "
          f"{len(langs_in)} langs, top domain: {top_domain[0]} ({top_domain[1]})")

# 3b. DBSCAN clustering
print("\n  DBSCAN clustering:")
dbscan_results = {}
for eps in [0.3, 0.35, 0.4, 0.45, 0.5]:
    for min_samples in [3, 4, 5]:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        db_labels = db.fit_predict(dist_matrix)
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = list(db_labels).count(-1)
        key = f"eps={eps},min={min_samples}"
        dbscan_results[key] = {
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_pct': n_noise / n * 100,
            'labels': db_labels.tolist(),
        }
        if n_clusters >= 2 and n_clusters <= 50:
            non_noise = [i for i, l in enumerate(db_labels) if l != -1]
            if len(set(db_labels[non_noise])) >= 2:
                sil = silhouette_score(dist_matrix[np.ix_(non_noise, non_noise)],
                                       db_labels[non_noise], metric='precomputed')
                dbscan_results[key]['silhouette'] = sil
            else:
                dbscan_results[key]['silhouette'] = None
        else:
            dbscan_results[key]['silhouette'] = None
        print(f"    {key}: {n_clusters} clusters, {n_noise} noise ({n_noise/n*100:.1f}%), "
              f"sil={dbscan_results[key].get('silhouette', 'N/A')}")

# Pick best DBSCAN configuration
best_dbscan = None
best_dbscan_sil = -1
for key, result in dbscan_results.items():
    if result['silhouette'] is not None and result['silhouette'] > best_dbscan_sil:
        best_dbscan_sil = result['silhouette']
        best_dbscan = key

if best_dbscan:
    print(f"\n  Best DBSCAN: {best_dbscan} (silhouette={best_dbscan_sil:.4f})")
    db_labels_best = dbscan_results[best_dbscan]['labels']
    for s, label in zip(substrates, db_labels_best):
        s['dbscan_cluster'] = int(label)
else:
    print("\n  No valid DBSCAN configuration found")
    for s in substrates:
        s['dbscan_cluster'] = -1


# ─── 4. Cross-Linguistic Cognate Detection ──────────────────────────────────
print("\n[4] Cross-linguistic cognate detection...")

# Find concepts that are consensus substrates in 3+ languages
concept_langs = defaultdict(dict)  # {concept: {language: form}}
for s in substrates:
    concept_langs[s['concept']][s['language']] = s['form']

multi_lang_concepts = {c: langs for c, langs in concept_langs.items() if len(langs) >= 3}
print(f"  {len(multi_lang_concepts)} concepts appear as CS in 3+ languages")

cognate_results = []
for concept, lang_forms in sorted(multi_lang_concepts.items(), key=lambda x: -len(x[1])):
    langs = sorted(lang_forms.keys())
    forms_list = [lang_forms[l].lower().strip() for l in langs]

    # Pairwise distances between substrate forms for this concept
    pairwise_dists = []
    for i in range(len(forms_list)):
        for j in range(i + 1, len(forms_list)):
            d = normalized_levenshtein(forms_list[i], forms_list[j])
            pairwise_dists.append(d)

    mean_dist = np.mean(pairwise_dists) if pairwise_dists else 1.0

    cognate_results.append({
        'concept': concept,
        'n_languages': len(langs),
        'languages': langs,
        'forms': {l: lang_forms[l] for l in langs},
        'mean_cross_ling_dist': mean_dist,
        'pairwise_distances': pairwise_dists,
    })

# Null distribution: for the same set of languages, pick random concepts and compute distances
print("  Computing null distribution (1000 random concept sets)...")
random.seed(42)
null_dists = []

# Get all available forms from ABVD for our 6 languages
all_concepts_with_forms = []
for concept, lang_dict in abvd_forms.items():
    available_langs = [l for l in LANG_ID_MAP.keys() if l in lang_dict and len(lang_dict[l]) > 0]
    if len(available_langs) >= 3:
        all_concepts_with_forms.append((concept, available_langs, lang_dict))

for _ in range(1000):
    if not all_concepts_with_forms:
        break
    # Pick a random concept with 3+ languages
    concept, avail_langs, lang_dict = random.choice(all_concepts_with_forms)
    # Pick 3 random languages from those available
    if len(avail_langs) < 3:
        continue
    selected = random.sample(avail_langs, min(3, len(avail_langs)))
    rand_forms = [lang_dict[l][0].lower().strip() for l in selected]

    pairwise = []
    for i in range(len(rand_forms)):
        for j in range(i + 1, len(rand_forms)):
            d = normalized_levenshtein(rand_forms[i], rand_forms[j])
            pairwise.append(d)
    if pairwise:
        null_dists.append(np.mean(pairwise))

null_mean = np.mean(null_dists) if null_dists else 0
null_std = np.std(null_dists) if null_dists else 0

substrate_mean_dists = [r['mean_cross_ling_dist'] for r in cognate_results]
substrate_overall_mean = np.mean(substrate_mean_dists) if substrate_mean_dists else 0

print(f"\n  Cross-linguistic distance results:")
print(f"    Substrate forms (same concept, diff languages): mean={substrate_overall_mean:.4f}")
print(f"    Null distribution (random concepts, diff languages): mean={null_mean:.4f} +/- {null_std:.4f}")

# Statistical comparison
if null_dists and substrate_mean_dists:
    # How many null samples have mean distance <= substrate mean?
    p_value = sum(1 for d in null_dists if d <= substrate_overall_mean) / len(null_dists)
    print(f"    p-value (substrate <= null): {p_value:.4f}")
    if substrate_overall_mean < null_mean:
        effect = (null_mean - substrate_overall_mean) / null_std if null_std > 0 else 0
        print(f"    Effect size (Cohen's d): {effect:.4f}")
        print(f"    => Substrate forms are MORE similar across languages than random!")
    else:
        effect = (substrate_overall_mean - null_mean) / null_std if null_std > 0 else 0
        print(f"    Effect size (Cohen's d): {-effect:.4f}")
        print(f"    => Substrate forms are NOT more similar than random across languages")

# Print individual concept results
print("\n  Per-concept cross-linguistic distances (sorted by distance):")
for r in sorted(cognate_results, key=lambda x: x['mean_cross_ling_dist']):
    forms_str = ", ".join(f"{l}:{r['forms'][l]}" for l in r['languages'])
    z_score = (r['mean_cross_ling_dist'] - null_mean) / null_std if null_std > 0 else 0
    flag = " ***COGNATE?" if r['mean_cross_ling_dist'] < null_mean - 1.0 * null_std else ""
    print(f"    {r['concept']:30s} ({r['n_languages']}L) dist={r['mean_cross_ling_dist']:.3f} "
          f"z={z_score:+.2f}{flag}")
    print(f"      Forms: {forms_str}")


# ─── 5. Semantic-Phonological Correlation ────────────────────────────────────
print("\n[5] Semantic-phonological correlation test...")

# Group substrates by semantic domain
domain_groups = defaultdict(list)
for i, s in enumerate(substrates):
    domain_groups[s['semantic_domain']].append(i)

domains = sorted(domain_groups.keys())
print(f"  Semantic domains: {domains}")
for d in domains:
    print(f"    {d}: {len(domain_groups[d])} substrates")

# Compute within-domain mean distance
within_dists = []
for domain, indices in domain_groups.items():
    if len(indices) < 2:
        continue
    for i, j in combinations(indices, 2):
        within_dists.append(dist_matrix[i, j])

within_mean = np.mean(within_dists) if within_dists else 0

# Compute between-domain mean distance
between_dists = []
for d1, d2 in combinations(domains, 2):
    for i in domain_groups[d1]:
        for j in domain_groups[d2]:
            between_dists.append(dist_matrix[i, j])

between_mean = np.mean(between_dists) if between_dists else 0

print(f"\n  Within-domain mean distance:  {within_mean:.4f} (n={len(within_dists)})")
print(f"  Between-domain mean distance: {between_mean:.4f} (n={len(between_dists)})")
print(f"  Difference: {between_mean - within_mean:.4f}")

# Permutation test: shuffle domain labels 1000 times
print("\n  Permutation test (1000 permutations)...")
random.seed(42)
observed_diff = between_mean - within_mean

perm_diffs = []
domain_labels = [s['semantic_domain'] for s in substrates]
for _ in range(1000):
    shuffled = domain_labels.copy()
    random.shuffle(shuffled)

    perm_domain_groups = defaultdict(list)
    for i, d in enumerate(shuffled):
        perm_domain_groups[d].append(i)

    perm_within = []
    for domain, indices in perm_domain_groups.items():
        if len(indices) < 2:
            continue
        for i_idx, j_idx in combinations(indices, 2):
            perm_within.append(dist_matrix[i_idx, j_idx])

    perm_between = []
    perm_domains = sorted(perm_domain_groups.keys())
    for d1, d2 in combinations(perm_domains, 2):
        for i_idx in perm_domain_groups[d1]:
            for j_idx in perm_domain_groups[d2]:
                perm_between.append(dist_matrix[i_idx, j_idx])

    perm_within_mean = np.mean(perm_within) if perm_within else 0
    perm_between_mean = np.mean(perm_between) if perm_between else 0
    perm_diffs.append(perm_between_mean - perm_within_mean)

p_perm = sum(1 for d in perm_diffs if d >= observed_diff) / len(perm_diffs)
print(f"  Observed between-within difference: {observed_diff:.4f}")
print(f"  Permutation p-value: {p_perm:.4f}")
if p_perm < 0.05:
    print(f"  => SIGNIFICANT: substrates in same semantic domain are phonologically more similar")
else:
    print(f"  => Not significant: no semantic-phonological clustering detected")


# ─── 6. Outputs ──────────────────────────────────────────────────────────────
print("\n[6] Generating outputs...")

# 6a. clustering_summary.json
summary = {
    'n_substrates': len(substrates),
    'languages': dict(lang_counts),
    'distance_matrix_stats': {
        'mean': float(np.mean(dist_matrix[np.triu_indices(n, k=1)])),
        'median': float(np.median(dist_matrix[np.triu_indices(n, k=1)])),
        'std': float(np.std(dist_matrix[np.triu_indices(n, k=1)])),
        'min': float(np.min(dist_matrix[np.triu_indices(n, k=1)])),
        'max': float(np.max(dist_matrix[np.triu_indices(n, k=1)])),
    },
    'ward_clustering': {
        'optimal_k': best_k,
        'silhouette_score': float(best_sil),
        'silhouette_scores_all': {str(k): float(v) for k, v in silhouette_results.items()},
        'cluster_sizes': {str(cid): len(members) for cid, members in cluster_sizes.items()},
    },
    'dbscan_clustering': {
        'best_config': best_dbscan,
        'best_silhouette': float(best_dbscan_sil) if best_dbscan else None,
        'all_configs': {k: {
            'eps': v['eps'],
            'min_samples': v['min_samples'],
            'n_clusters': v['n_clusters'],
            'n_noise': v['n_noise'],
            'noise_pct': v['noise_pct'],
            'silhouette': float(v['silhouette']) if v['silhouette'] is not None else None,
        } for k, v in dbscan_results.items()},
    },
    'cross_linguistic_cognates': {
        'n_concepts_3plus_languages': len(multi_lang_concepts),
        'substrate_mean_cross_ling_dist': float(substrate_overall_mean),
        'null_mean_cross_ling_dist': float(null_mean),
        'null_std_cross_ling_dist': float(null_std),
        'p_value': float(p_value) if null_dists and substrate_mean_dists else None,
    },
    'semantic_phonological_correlation': {
        'within_domain_mean_dist': float(within_mean),
        'between_domain_mean_dist': float(between_mean),
        'observed_difference': float(observed_diff),
        'permutation_p_value': float(p_perm),
        'significant': p_perm < 0.05,
    },
}

with open(os.path.join(RESULTS_DIR, 'clustering_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("  Saved clustering_summary.json")

# 6b. cross_linguistic_cognates.csv
with open(os.path.join(RESULTS_DIR, 'cross_linguistic_cognates.csv'), 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['concept', 'n_languages', 'languages', 'forms', 'mean_cross_ling_dist',
                      'z_score', 'potential_cognate'])
    for r in sorted(cognate_results, key=lambda x: x['mean_cross_ling_dist']):
        z = (r['mean_cross_ling_dist'] - null_mean) / null_std if null_std > 0 else 0
        is_cognate = r['mean_cross_ling_dist'] < null_mean - 1.0 * null_std
        forms_str = "; ".join(f"{l}:{r['forms'][l]}" for l in r['languages'])
        writer.writerow([
            r['concept'],
            r['n_languages'],
            "|".join(r['languages']),
            forms_str,
            f"{r['mean_cross_ling_dist']:.4f}",
            f"{z:+.4f}",
            is_cognate,
        ])
print("  Saved cross_linguistic_cognates.csv")

# 6c. Dendrogram (top 50 most connected substrates)
print("  Generating dendrogram...")
fig, ax = plt.subplots(figsize=(16, 10))

# Create labels
labels = [f"{s['form']} ({s['language'][:3]})" for s in substrates]

# Full dendrogram but with truncation for readability
dendrogram(Z,
           labels=labels,
           leaf_rotation=90,
           leaf_font_size=6,
           truncate_mode='lastp',
           p=50,
           show_leaf_counts=True,
           ax=ax)

ax.set_title(f'Hierarchical Clustering of 266 Consensus Substrates\n'
             f'(Ward\'s method, truncated to 50 leaves, optimal k={best_k}, '
             f'silhouette={best_sil:.3f})', fontsize=13)
ax.set_ylabel('Ward distance', fontsize=11)
ax.set_xlabel('Substrate forms (language abbreviation)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'dendrogram.png'), dpi=150)
plt.close()
print("  Saved dendrogram.png")

# 6d. Distance matrix heatmap with cluster annotations
print("  Generating cluster heatmap...")
fig, ax = plt.subplots(figsize=(14, 12))

# Sort by Ward cluster for visual coherence
sort_idx = np.argsort(ward_labels)
sorted_matrix = dist_matrix[np.ix_(sort_idx, sort_idx)]
sorted_labels_ward = ward_labels[sort_idx]

im = ax.imshow(sorted_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar(im, ax=ax, label='Normalized Levenshtein Distance', shrink=0.8)

# Draw cluster boundaries
cluster_boundaries = []
prev_label = sorted_labels_ward[0]
for i in range(1, len(sorted_labels_ward)):
    if sorted_labels_ward[i] != prev_label:
        cluster_boundaries.append(i - 0.5)
        prev_label = sorted_labels_ward[i]

for boundary in cluster_boundaries:
    ax.axhline(y=boundary, color='red', linewidth=0.5, alpha=0.7)
    ax.axvline(x=boundary, color='red', linewidth=0.5, alpha=0.7)

ax.set_title(f'Phonological Distance Matrix (266 Consensus Substrates)\n'
             f'Sorted by Ward clusters (k={best_k}), red lines = cluster boundaries',
             fontsize=13)
ax.set_xlabel('Substrate index (sorted by cluster)', fontsize=11)
ax.set_ylabel('Substrate index (sorted by cluster)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cluster_heatmap.png'), dpi=150)
plt.close()
print("  Saved cluster_heatmap.png")

# 6e. Cross-linguistic distance histogram
print("  Generating cross-linguistic distance histogram...")
fig, ax = plt.subplots(figsize=(10, 7))

# Null distribution histogram
ax.hist(null_dists, bins=40, alpha=0.6, color='gray', label='Null (random concepts)', density=True)

# Substrate distances as vertical lines
for r in cognate_results:
    color = 'red' if r['mean_cross_ling_dist'] < null_mean - 1.0 * null_std else 'blue'
    ax.axvline(x=r['mean_cross_ling_dist'], color=color, alpha=0.5, linewidth=1.5)

# Add mean lines
ax.axvline(x=null_mean, color='black', linestyle='--', linewidth=2, label=f'Null mean ({null_mean:.3f})')
ax.axvline(x=substrate_overall_mean, color='red', linestyle='--', linewidth=2,
           label=f'Substrate mean ({substrate_overall_mean:.3f})')

# Shade the "potential cognate" region
threshold = null_mean - 1.0 * null_std
ax.axvspan(0, threshold, alpha=0.1, color='red', label=f'Potential cognate zone (<{threshold:.3f})')

ax.set_xlabel('Mean cross-linguistic normalized Levenshtein distance', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Cross-Linguistic Cognate Test:\nSubstrate Forms vs Random Concept Forms', fontsize=13)
ax.legend(fontsize=9, loc='upper left')

# Add text annotation
n_potential = sum(1 for r in cognate_results
                  if r['mean_cross_ling_dist'] < null_mean - 1.0 * null_std)
ax.text(0.97, 0.95,
        f"Substrate concepts (3+L): {len(cognate_results)}\n"
        f"Potential cognates: {n_potential}\n"
        f"p-value: {p_value:.4f}" if substrate_mean_dists else "",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cross_ling_distance_histogram.png'), dpi=150)
plt.close()
print("  Saved cross_ling_distance_histogram.png")


# ─── Final Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  266 consensus substrates analyzed across 6 Sulawesi languages")
print(f"\n  CLUSTERING:")
print(f"    Ward's (optimal k={best_k}): silhouette={best_sil:.4f}")
if best_dbscan:
    print(f"    DBSCAN (best={best_dbscan}): silhouette={best_dbscan_sil:.4f}")
print(f"\n  CROSS-LINGUISTIC COGNATES:")
print(f"    {len(multi_lang_concepts)} concepts in 3+ languages")
print(f"    Substrate cross-ling distance: {substrate_overall_mean:.4f}")
print(f"    Null (random) cross-ling distance: {null_mean:.4f}")
if substrate_overall_mean < null_mean:
    print(f"    => Substrate forms are {((null_mean - substrate_overall_mean) / null_mean * 100):.1f}% "
          f"more similar across languages than random")
else:
    print(f"    => Substrate forms are NOT more similar than random")
n_potential_cognates = sum(1 for r in cognate_results
                           if r['mean_cross_ling_dist'] < null_mean - 1.0 * null_std)
print(f"    Potential cognate concepts (z < -1.0): {n_potential_cognates}")
print(f"\n  SEMANTIC-PHONOLOGICAL CORRELATION:")
print(f"    Within-domain: {within_mean:.4f}, Between-domain: {between_mean:.4f}")
print(f"    Permutation p-value: {p_perm:.4f} ({'significant' if p_perm < 0.05 else 'not significant'})")

print(f"\n  All results saved to: {RESULTS_DIR}")
print("=" * 70)
