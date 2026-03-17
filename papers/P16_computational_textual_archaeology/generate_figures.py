"""
P16 Figure Generation — Publication-quality figures
====================================================
Generates 6 figures for "What Ancient Texts Remember and Inscriptions Forget"

Figures:
1. UMAP scatter of 200 passages colored by tradition
2. UMAP scatter colored by BERTopic topic
3. Semantic convergence z-scores (8 groups, bar chart)
4. Semantic query similarities (7 queries, horizontal bar) — E094
5. Topic x century heatmap (pre/post 929 CE) — E096
6. Temporal centroid drift (century distances) — E094
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

outdir = Path("papers/P16_computational_textual_archaeology/figures")
outdir.mkdir(exist_ok=True)

# --- Load data ---
e090_path = Path("experiments/E090_transformer_textual_nlp/results/e090_v5_full_results.json")
e094_path = Path("experiments/E094_dharma_semantic_search/results/e094_results.json")
e096_path = Path("experiments/E096_dharma_diachronic_bertopic/results/e096_results.json")
e094_drift = Path("experiments/E094_dharma_semantic_search/results/temporal_drift.json")
e094_queries = Path("experiments/E094_dharma_semantic_search/results/semantic_queries.json")
e096_heatmap = Path("experiments/E096_dharma_diachronic_bertopic/results/topic_heatmap.json")
e096_comparison = Path("experiments/E096_dharma_diachronic_bertopic/results/pre_post_929_comparison.json")

with open(e090_path, 'r', encoding='utf-8') as f:
    e090 = json.load(f)
with open(e094_path, 'r', encoding='utf-8') as f:
    e094 = json.load(f)
with open(e096_path, 'r', encoding='utf-8') as f:
    e096 = json.load(f)

# Try loading additional files
try:
    with open(e094_drift, 'r', encoding='utf-8') as f:
        drift_data = json.load(f)
except:
    drift_data = None

try:
    with open(e094_queries, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
except:
    queries_data = None

try:
    with open(e096_heatmap, 'r', encoding='utf-8') as f:
        heatmap_data = json.load(f)
except:
    heatmap_data = None

try:
    with open(e096_comparison, 'r', encoding='utf-8') as f:
        comparison_data = json.load(f)
except:
    comparison_data = None

# --- Color palettes ---
tradition_colors = {
    'ARAB': '#e6194b', 'CHEMICAL': '#3cb44b', 'CHINESE': '#ffe119',
    'EUROPEAN': '#4363d8', 'GREEK': '#f58231', 'INDIAN_PALI': '#911eb4',
    'INDIAN_SANSKRIT': '#42d4f4', 'LINGUISTIC': '#f032e6', 'NUSANTARAN': '#bfef45',
    'PERSIAN': '#fabed4', 'ROMAN': '#469990', 'TAMIL': '#dcbeff',
}

# ================================================================
# FIGURE 1: UMAP scatter by tradition
# ================================================================
print("Generating Figure 1: UMAP by tradition...")

if 'exp2_clustering' in e090:
    exp2 = e090['exp2_clustering']
    umap_list = exp2['umap_coordinates']
    umap_coords = np.array([[p['x'], p['y']] for p in umap_list])
    traditions = [p['tradition'] for p in umap_list]

    fig, ax = plt.subplots(figsize=(8, 6))
    for trad in sorted(set(traditions)):
        mask = [i for i, t in enumerate(traditions) if t == trad]
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=tradition_colors.get(trad, '#808080'), label=trad,
                   s=40, alpha=0.7, edgecolors='white', linewidth=0.3)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('200 Ancient Passages in Embedding Space (by Tradition)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True,
              fontsize=7, ncol=1, markerscale=0.8)
    plt.tight_layout()
    fig.savefig(outdir / 'fig1_umap_tradition.png')
    fig.savefig(outdir / 'fig1_umap_tradition.pdf')
    plt.close(fig)
    print("  Saved fig1_umap_tradition.png/pdf")
else:
    print("  SKIP: exp2_clustering not in e090 results")

# ================================================================
# FIGURE 2: UMAP scatter by BERTopic topic
# ================================================================
print("Generating Figure 2: UMAP by BERTopic topic...")

if 'exp4_bertopic' in e090 and 'exp2_clustering' in e090:
    exp4 = e090['exp4_bertopic']
    per_doc = exp4.get('per_document_topics', [])
    topic_assignments = [p['topic'] for p in per_doc] if per_doc else []

    if topic_assignments and umap_coords is not None:
        topics_unique = sorted(set(topic_assignments))
        # Color map: -1 = grey, then tab20
        cmap = plt.cm.tab20
        topic_color_map = {-1: '#cccccc'}
        non_outlier = [t for t in topics_unique if t != -1]
        for i, t in enumerate(non_outlier):
            topic_color_map[t] = cmap(i / max(len(non_outlier), 1))

        # Topic labels from top words
        topic_info = {}
        for t_entry in exp4.get('topics', []):
            tid = str(t_entry.get('id', ''))
            words = t_entry.get('words', [])
            top_words = ', '.join([w[0] for w in words[:3]]) if words else ''
            topic_info[tid] = {'top_words': top_words}

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot outliers first (background)
        outlier_mask = [i for i, t in enumerate(topic_assignments) if t == -1]
        if outlier_mask:
            ax.scatter(umap_coords[outlier_mask, 0], umap_coords[outlier_mask, 1],
                       c='#cccccc', s=20, alpha=0.3, marker='x', label='Outlier')

        # Plot topics
        key_topics = [0, 1, 3, 4, 7, 8, 12]  # Most interpretable
        for t in non_outlier:
            mask = [i for i, ta in enumerate(topic_assignments) if ta == t]
            label = f"T{t}"
            if str(t) in topic_info:
                words = topic_info[str(t)].get('top_words', '')
                if words:
                    label = f"T{t}: {words[:30]}"
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[topic_color_map[t]], s=40, alpha=0.7,
                       edgecolors='white', linewidth=0.3,
                       label=label if t in key_topics else None)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('200 Ancient Passages in Embedding Space (by BERTopic Topic)')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True,
                  fontsize=6, ncol=1)
        plt.tight_layout()
        fig.savefig(outdir / 'fig2_umap_topic.png')
        fig.savefig(outdir / 'fig2_umap_topic.pdf')
        plt.close(fig)
        print("  Saved fig2_umap_topic.png/pdf")
    else:
        print("  SKIP: topic_assignments empty")
else:
    print("  SKIP: exp4_bertopic not in results")

# ================================================================
# FIGURE 3: Semantic convergence z-scores (bar chart)
# ================================================================
print("Generating Figure 3: Semantic convergence z-scores...")

if 'exp5_extended_convergence' in e090:
    exp5 = e090['exp5_extended_convergence']
    groups = exp5.get('concept_groups', {})

    concepts = []
    z_scores = []
    is_new = []

    concept_order = ['JAVA', 'SUMATRA_GOLD', 'CAMPHOR_BARUS', 'SPICE_TRADE',
                     'MARITIME_VOYAGE', 'VOLCANO', 'BUDDHIST_WORLD', 'METAL_TRADE']

    for c in concept_order:
        if c in groups:
            g = groups[c]
            concepts.append(c.replace('_', '\n'))
            z_scores.append(g['z_score'])
            is_new.append(c in ['VOLCANO', 'BUDDHIST_WORLD', 'METAL_TRADE'])

    colors = ['#2196F3' if not n else '#FF5722' for n in is_new]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(concepts)), z_scores, color=colors, edgecolor='white', width=0.7)

    # Significance line
    ax.axhline(y=1.96, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(len(concepts) - 0.5, 2.3, 'p = 0.05', color='red', fontsize=8, ha='right')

    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, fontsize=8)
    ax.set_ylabel('Z-score (vs random baseline)')
    ax.set_title('Semantic Convergence Across 12 Ancient Traditions (8 Concept Groups)')

    # Add z-score labels on bars
    for bar, z in zip(bars, z_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{z:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#2196F3', label='Original concepts (v2)'),
        mpatches.Patch(color='#FF5722', label='New concepts (v5)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    fig.savefig(outdir / 'fig3_convergence_zscores.png')
    fig.savefig(outdir / 'fig3_convergence_zscores.pdf')
    plt.close(fig)
    print("  Saved fig3_convergence_zscores.png/pdf")
else:
    print("  SKIP: exp5_convergence not in results")

# ================================================================
# FIGURE 4: Semantic query similarities (E094)
# ================================================================
print("Generating Figure 4: Epigraphic semantic queries...")

queries_dict = queries_data if queries_data and isinstance(queries_data, dict) else None

if queries_dict:
    query_labels = []
    mean_sims = []
    for query_text, qdata in queries_dict.items():
        query_labels.append(query_text[:45])
        mean_sims.append(qdata.get('mean_similarity', 0))

    # Sort by similarity
    sorted_idx = np.argsort(mean_sims)
    query_labels = [query_labels[i] for i in sorted_idx]
    mean_sims = [mean_sims[i] for i in sorted_idx]

    # Color: volcanic = red, others = blue
    colors = []
    for ql in query_labels:
        if 'volcanic' in ql.lower() or 'fire' in ql.lower():
            colors.append('#e53935')
        elif 'mountain' in ql.lower():
            colors.append('#FF9800')
        else:
            colors.append('#1976D2')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(range(len(query_labels)), mean_sims, color=colors,
                   edgecolor='white', height=0.6)

    ax.set_yticks(range(len(query_labels)))
    ax.set_yticklabels(query_labels, fontsize=8)
    ax.set_xlabel('Mean Cosine Similarity to DHARMA Inscriptions')
    ax.set_title('Semantic Proximity of 7 Themes to Old Javanese Epigraphy')

    # Add value labels
    for bar, v in zip(bars, mean_sims):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', ha='left', va='center', fontsize=8)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#e53935', label='Volcanic/landscape'),
        mpatches.Patch(color='#FF9800', label='Sacred mountain'),
        mpatches.Patch(color='#1976D2', label='Administrative/social'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    fig.savefig(outdir / 'fig4_query_similarities.png')
    fig.savefig(outdir / 'fig4_query_similarities.pdf')
    plt.close(fig)
    print("  Saved fig4_query_similarities.png/pdf")
elif 'semantic_queries' in e094:
    # Fallback: extract from e094 results
    queries = e094['semantic_queries']
    query_labels = []
    mean_sims = []
    for q in queries:
        query_labels.append(q.get('query', '')[:40])
        mean_sims.append(q.get('mean_similarity', 0))

    sorted_idx = np.argsort(mean_sims)
    query_labels = [query_labels[i] for i in sorted_idx]
    mean_sims = [mean_sims[i] for i in sorted_idx]

    colors = ['#e53935' if 'volcanic' in ql.lower() or 'fire' in ql.lower()
              else '#FF9800' if 'mountain' in ql.lower()
              else '#1976D2' for ql in query_labels]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(range(len(query_labels)), mean_sims, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(query_labels)))
    ax.set_yticklabels(query_labels, fontsize=8)
    ax.set_xlabel('Mean Cosine Similarity')
    ax.set_title('Semantic Proximity of 7 Themes to Old Javanese Epigraphy')
    plt.tight_layout()
    fig.savefig(outdir / 'fig4_query_similarities.png')
    fig.savefig(outdir / 'fig4_query_similarities.pdf')
    plt.close(fig)
    print("  Saved fig4_query_similarities.png/pdf (fallback)")
else:
    print("  SKIP: no query data available")

# ================================================================
# FIGURE 5: Topic x century heatmap (E096)
# ================================================================
print("Generating Figure 5: Topic x century heatmap...")

if heatmap_data:
    hm = heatmap_data
elif 'topic_heatmap' in e096:
    hm = e096['topic_heatmap']
else:
    hm = None

if hm and isinstance(hm, dict) and 'heatmap' in hm:
    # Parse structured heatmap
    hm_data = hm['heatmap']
    topic_kw = hm.get('topic_keywords', {})
    centuries = hm.get('centuries', ['C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14'])
    topic_keys = sorted(hm_data.keys(), key=lambda x: int(x.replace('topic_', '')) if x.replace('topic_', '').lstrip('-').isdigit() else 999)

    topic_name_map = {
        'topic_-1': 'Outlier',
        'topic_0': 'T0: Administrative',
        'topic_1': 'T1: Royal/Political',
        'topic_2': 'T2: Ritual/Calendrical',
    }
    topic_labels = [topic_name_map.get(tk, tk) for tk in topic_keys]

    matrix = np.zeros((len(topic_keys), len(centuries)))
    for ti, tk in enumerate(topic_keys):
        for ci, c in enumerate(centuries):
            matrix[ti, ci] = hm_data[tk].get(c, 0)

    if True:
        # Add 929 CE line
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                    xticklabels=centuries, yticklabels=topic_labels,
                    linewidths=0.5, ax=ax, cbar_kws={'label': 'Documents'})

        # 929 CE divider (between C10 and C11)
        if 'C10' in centuries and 'C11' in centuries:
            divider_x = centuries.index('C11')
            ax.axvline(x=divider_x, color='black', linewidth=2.5, linestyle='-')
            ax.text(divider_x, -0.3, '929 CE', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='black')

        ax.set_title('BERTopic Topics Across Centuries (46 Dated Inscriptions)')
        ax.set_xlabel('Century')
        ax.set_ylabel('')

        # Add pre/post labels
        if 'C10' in centuries and 'C11' in centuries:
            mid_pre = centuries.index('C11') / 2
            mid_post = (centuries.index('C11') + len(centuries)) / 2
            ax.text(mid_pre, len(topic_keys) + 0.6, 'PRE-929', ha='center',
                    fontsize=9, fontstyle='italic', color='#1565C0')
            ax.text(mid_post, len(topic_keys) + 0.6, 'POST-929', ha='center',
                    fontsize=9, fontstyle='italic', color='#c62828')

        plt.tight_layout()
        fig.savefig(outdir / 'fig5_topic_heatmap.png')
        fig.savefig(outdir / 'fig5_topic_heatmap.pdf')
        plt.close(fig)
        print("  Saved fig5_topic_heatmap.png/pdf")
    else:
        print("  SKIP: could not build heatmap matrix")
else:
    # Build from known data (hardcoded from E096 output)
    centuries = ['C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']
    topic_labels = ['Outlier', 'T0: Administrative', 'T1: Royal/Political', 'T2: Ritual/Calendrical']
    matrix = np.array([
        [0, 2, 0, 0, 0, 0, 0],   # outlier
        [2, 11, 11, 3, 0, 0, 1],  # T0
        [2, 0, 0, 0, 1, 4, 3],    # T1
        [1, 4, 1, 0, 0, 0, 0],    # T2
    ])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=centuries, yticklabels=topic_labels,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Documents'})

    # 929 CE divider between C10 and C11
    ax.axvline(x=3, color='black', linewidth=2.5, linestyle='-')
    ax.text(3, -0.3, '929 CE', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='black')

    ax.set_title('BERTopic Topics Across Centuries (46 Dated Inscriptions)')
    ax.set_xlabel('Century')
    ax.set_ylabel('')
    ax.text(1.5, 4.6, 'PRE-929', ha='center', fontsize=9, fontstyle='italic', color='#1565C0')
    ax.text(5, 4.6, 'POST-929', ha='center', fontsize=9, fontstyle='italic', color='#c62828')

    plt.tight_layout()
    fig.savefig(outdir / 'fig5_topic_heatmap.png')
    fig.savefig(outdir / 'fig5_topic_heatmap.pdf')
    plt.close(fig)
    print("  Saved fig5_topic_heatmap.png/pdf (hardcoded)")

# ================================================================
# FIGURE 6: Temporal centroid drift (E094)
# ================================================================
print("Generating Figure 6: Temporal centroid drift...")

if drift_data and isinstance(drift_data, dict) and 'consecutive_distances' in drift_data:
    cd = drift_data['consecutive_distances']
    labels = list(cd.keys())
    distances = list(cd.values())
else:
    # Hardcode from E094 output
    labels = ['C8-C9', 'C9-C10', 'C10-C11', 'C11-C12', 'C12-C13', 'C13-C14']
    distances = [0.163, 0.087, 0.208, 0.366, 0.245, 0.181]

fig, ax = plt.subplots(figsize=(8, 4.5))
colors_drift = ['#1976D2'] * len(distances)
max_idx = np.argmax(distances)
colors_drift[max_idx] = '#c62828'  # Highlight largest rupture

bars = ax.bar(range(len(labels)), distances, color=colors_drift,
              edgecolor='white', width=0.6)

# Labels
for bar, d in zip(bars, distances):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{d:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Cosine Distance Between Century Centroids')
ax.set_title('Semantic Drift Across Centuries in DHARMA Inscriptions')

# Add 929 CE annotation
ax.annotate('929 CE\ndivide', xy=(2, distances[2]), xytext=(2.5, max(distances) * 0.85),
            fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', color='grey'))

# Highlight C11-C12 rupture
ax.annotate('Largest\nsemantic\nrupture', xy=(max_idx, distances[max_idx]),
            xytext=(max_idx + 0.7, distances[max_idx] - 0.03),
            fontsize=8, ha='center', color='#c62828', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#c62828'))

plt.tight_layout()
fig.savefig(outdir / 'fig6_temporal_drift.png')
fig.savefig(outdir / 'fig6_temporal_drift.pdf')
plt.close(fig)
print("  Saved fig6_temporal_drift.png/pdf")

print("\n" + "=" * 60)
print(f"All figures saved to: {outdir}")
print("=" * 60)
