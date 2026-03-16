#!/usr/bin/env python3
"""
E096 — DHARMA Diachronic BERTopic
First application of BERTopic to any epigraphic corpus.

Models topic emergence and disappearance across centuries of Old Javanese inscriptions,
with focus on the 929 CE Mataram collapse as a potential discontinuity.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from scipy.stats import fisher_exact, chi2_contingency
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

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


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("E096 — DHARMA Diachronic BERTopic")
    print("First BERTopic application to any epigraphic corpus")
    print("=" * 70)

    # ── Step 1: Parse and filter ───────────────────────────────────────────
    print("\n[1/8] Parsing DHARMA XML inscriptions...")
    xml_files = sorted(DHARMA_DIR.glob("*.xml"))
    print(f"  Found {len(xml_files)} XML files")

    all_inscriptions = []
    for xf in xml_files:
        parsed = parse_inscription(xf)
        if parsed is not None:
            all_inscriptions.append(parsed)

    print(f"  Successfully parsed: {len(all_inscriptions)}")

    # ── Step 2: Filter to DATED inscriptions with translations ─────────────
    print("\n[2/8] Filtering to DATED inscriptions with translations...")
    dated = [
        ins for ins in all_inscriptions
        if ins['date_ce'] is not None and ins['translation_text'].strip()
    ]
    print(f"  Dated inscriptions with translations: {len(dated)}")

    if len(dated) < 10:
        print("  ERROR: Too few dated inscriptions for topic modeling. Exiting.")
        return

    # Century distribution
    century_counts = Counter(ins['century'] for ins in dated)
    print("  Century distribution:")
    for c in sorted(century_counts):
        print(f"    C{c}: {century_counts[c]} inscriptions")

    # Pre/post 929 split
    pre_929 = [ins for ins in dated if ins['date_ce'] < 929]
    post_929 = [ins for ins in dated if ins['date_ce'] >= 929]
    print(f"\n  Pre-929 CE: {len(pre_929)} inscriptions")
    print(f"  Post-929 CE: {len(post_929)} inscriptions")

    # ── Step 3: SBERT embedding ────────────────────────────────────────────
    print("\n[3/8] Loading SBERT model (all-MiniLM-L6-v2) on CUDA...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    texts = [ins['translation_text'].strip() for ins in dated]
    print(f"  Embedding {len(texts)} translation texts...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"  Embedding shape: {embeddings.shape}")

    # ── Step 4: BERTopic ───────────────────────────────────────────────────
    print("\n[4/8] Running BERTopic...")

    # Configure sub-models for small corpus
    # Adaptive parameters based on corpus size
    n_neighbors = min(15, len(dated) - 1)
    min_cluster_size = max(3, len(dated) // 15)

    umap_model = UMAP(
        n_components=5,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words='english',
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics='auto',
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(texts, embeddings)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info['Topic'] != -1])
    n_outlier = int((np.array(topics) == -1).sum())

    print(f"\n  BERTopic found {n_topics} topics, {n_outlier} outlier documents")
    print("\n  Topic overview:")
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            label = "Outlier"
        else:
            # Get top words for this topic
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                label = ", ".join([w for w, _ in topic_words[:5]])
            else:
                label = "(empty)"
        print(f"    Topic {topic_id} ({row['Count']} docs): {label}")

    # ── Step 5: Topic x Century heatmap ────────────────────────────────────
    print("\n[5/8] Building topic x century heatmap...")

    unique_topics = sorted(set(topics))
    unique_centuries = sorted(set(ins['century'] for ins in dated))

    heatmap = {}
    for topic_id in unique_topics:
        topic_label = f"topic_{topic_id}"
        heatmap[topic_label] = {}
        for century in unique_centuries:
            count = sum(
                1 for i, ins in enumerate(dated)
                if topics[i] == topic_id and ins['century'] == century
            )
            heatmap[topic_label][f"C{century}"] = count

    # Add topic keywords to heatmap
    topic_keywords = {}
    for topic_id in unique_topics:
        if topic_id == -1:
            topic_keywords["topic_-1"] = ["outlier"]
        else:
            words = topic_model.get_topic(topic_id)
            if words:
                topic_keywords[f"topic_{topic_id}"] = [w for w, _ in words[:10]]
            else:
                topic_keywords[f"topic_{topic_id}"] = []

    heatmap_output = {
        'heatmap': heatmap,
        'topic_keywords': topic_keywords,
        'centuries': [f"C{c}" for c in unique_centuries],
        'n_topics': n_topics,
        'century_sizes': {f"C{c}": century_counts[c] for c in unique_centuries},
        'caveat': 'C7(~2), C8(~5), C12(~1) too sparse for standalone analysis',
    }

    print("\n  Heatmap (topic x century counts):")
    header = "  " + "Topic".ljust(12) + "".join(f"C{c}".rjust(5) for c in unique_centuries)
    print(header)
    print("  " + "-" * len(header))
    for topic_id in unique_topics:
        tl = f"topic_{topic_id}"
        row = f"  T{topic_id}".ljust(14)
        for century in unique_centuries:
            row += str(heatmap[tl][f"C{century}"]).rjust(5)
        print(row)

    with open(RESULTS_DIR / "topic_heatmap.json", 'w', encoding='utf-8') as f:
        json.dump(heatmap_output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved heatmap to results/topic_heatmap.json")

    # ── Step 6: Pre-929 vs Post-929 comparison ─────────────────────────────
    print("\n[6/8] Pre-929 vs Post-929 CE topic comparison...")

    pre_929_topics = [topics[i] for i, ins in enumerate(dated) if ins['date_ce'] < 929]
    post_929_topics = [topics[i] for i, ins in enumerate(dated) if ins['date_ce'] >= 929]

    pre_929_set = set(pre_929_topics)
    post_929_set = set(post_929_topics)

    only_pre = pre_929_set - post_929_set
    only_post = post_929_set - pre_929_set
    persistent = pre_929_set & post_929_set

    def topic_label(tid):
        if tid == -1:
            return "outlier"
        words = topic_model.get_topic(tid)
        if words:
            return ", ".join([w for w, _ in words[:5]])
        return "(empty)"

    print(f"\n  Topics ONLY pre-929 ({len(only_pre)}):")
    for t in sorted(only_pre):
        count = pre_929_topics.count(t)
        print(f"    Topic {t} ({count} docs): {topic_label(t)}")

    print(f"\n  Topics ONLY post-929 ({len(only_post)}):")
    for t in sorted(only_post):
        count = post_929_topics.count(t)
        print(f"    Topic {t} ({count} docs): {topic_label(t)}")

    print(f"\n  Topics PERSISTENT across 929 divide ({len(persistent)}):")
    for t in sorted(persistent):
        pre_c = pre_929_topics.count(t)
        post_c = post_929_topics.count(t)
        print(f"    Topic {t} (pre={pre_c}, post={post_c}): {topic_label(t)}")

    comparison = {
        'pre_929_count': len(pre_929_topics),
        'post_929_count': len(post_929_topics),
        'only_pre_929': {
            f"topic_{t}": {
                'count': pre_929_topics.count(t),
                'keywords': topic_label(t),
            }
            for t in sorted(only_pre)
        },
        'only_post_929': {
            f"topic_{t}": {
                'count': post_929_topics.count(t),
                'keywords': topic_label(t),
            }
            for t in sorted(only_post)
        },
        'persistent': {
            f"topic_{t}": {
                'pre_929_count': pre_929_topics.count(t),
                'post_929_count': post_929_topics.count(t),
                'keywords': topic_label(t),
            }
            for t in sorted(persistent)
        },
    }

    # ── Step 7: Statistical tests ──────────────────────────────────────────
    print("\n[7/8] Statistical testing (pre/post 929 topic distributions)...")

    # Build contingency table: each non-outlier topic vs pre/post
    real_topics = sorted([t for t in set(topics) if t != -1])

    if len(real_topics) >= 2:
        # Chi-square test on topic distribution
        contingency = []
        for t in real_topics:
            pre_c = sum(1 for i, ins in enumerate(dated) if topics[i] == t and ins['date_ce'] < 929)
            post_c = sum(1 for i, ins in enumerate(dated) if topics[i] == t and ins['date_ce'] >= 929)
            contingency.append([pre_c, post_c])

        contingency = np.array(contingency)

        # Only run chi-square if we have enough expected counts
        row_totals = contingency.sum(axis=1)
        col_totals = contingency.sum(axis=0)
        grand_total = contingency.sum()
        expected = np.outer(row_totals, col_totals) / grand_total

        if np.all(expected >= 1) and grand_total > 0:
            chi2, p_chi2, dof, _ = chi2_contingency(contingency)
            print(f"  Chi-square test: chi2={chi2:.3f}, p={p_chi2:.4f}, dof={dof}")
            comparison['chi_square'] = {
                'statistic': round(float(chi2), 4),
                'p_value': round(float(p_chi2), 6),
                'dof': int(dof),
                'interpretation': 'significant' if p_chi2 < 0.05 else 'not significant',
            }
        else:
            print("  Chi-square: expected counts too low, skipping")
            comparison['chi_square'] = {'skipped': 'expected counts < 1'}

        # Fisher exact tests for each topic individually (2x2)
        fisher_results = {}
        for t in real_topics:
            pre_in = sum(1 for i, ins in enumerate(dated) if topics[i] == t and ins['date_ce'] < 929)
            pre_out = sum(1 for i, ins in enumerate(dated) if topics[i] != t and ins['date_ce'] < 929)
            post_in = sum(1 for i, ins in enumerate(dated) if topics[i] == t and ins['date_ce'] >= 929)
            post_out = sum(1 for i, ins in enumerate(dated) if topics[i] != t and ins['date_ce'] >= 929)

            table = [[pre_in, pre_out], [post_in, post_out]]
            odds, p_fisher = fisher_exact(table)
            fisher_results[f"topic_{t}"] = {
                'contingency': table,
                'odds_ratio': round(float(odds), 4) if np.isfinite(odds) else 'inf',
                'p_value': round(float(p_fisher), 6),
                'keywords': topic_label(t),
                'interpretation': 'significant' if p_fisher < 0.05 else 'not significant',
            }
            sig_marker = " *" if p_fisher < 0.05 else ""
            print(f"  Fisher exact Topic {t}: OR={odds:.2f}, p={p_fisher:.4f}{sig_marker}")

        comparison['fisher_exact_per_topic'] = fisher_results
    else:
        print("  Fewer than 2 topics found — skipping statistical tests")
        comparison['chi_square'] = {'skipped': 'fewer than 2 topics'}
        comparison['fisher_exact_per_topic'] = {}

    # ── Step 7b: Focused C9-C10 vs C11-C14 analysis ───────────────────────
    print("\n  Focused analysis: C9-C10 (dense) vs C11-C14 (sparse)...")

    c9_c10_topics = [topics[i] for i, ins in enumerate(dated) if ins['century'] in (9, 10)]
    c11_c14_topics = [topics[i] for i, ins in enumerate(dated) if ins['century'] in (11, 12, 13, 14)]

    print(f"  C9-C10: {len(c9_c10_topics)} inscriptions")
    print(f"  C11-C14: {len(c11_c14_topics)} inscriptions")

    c9c10_dist = Counter(c9_c10_topics)
    c11c14_dist = Counter(c11_c14_topics)

    comparison['focused_c9c10_vs_c11c14'] = {
        'c9_c10_count': len(c9_c10_topics),
        'c11_c14_count': len(c11_c14_topics),
        'c9_c10_topic_distribution': {f"topic_{k}": v for k, v in c9c10_dist.most_common()},
        'c11_c14_topic_distribution': {f"topic_{k}": v for k, v in c11c14_dist.most_common()},
        'caveat': 'C12 has ~1 inscription; C13-C14 very sparse. Interpret with caution.',
    }

    print("  C9-C10 topics:", dict(c9c10_dist.most_common()))
    print("  C11-C14 topics:", dict(c11c14_dist.most_common()))

    # ── Step 8: Save results ───────────────────────────────────────────────
    print("\n[8/8] Saving results...")

    with open(RESULTS_DIR / "pre_post_929_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"  Saved comparison to results/pre_post_929_comparison.json")

    # Full results summary
    results_summary = {
        'experiment': 'E096_dharma_diachronic_bertopic',
        'description': 'First BERTopic application to any epigraphic corpus',
        'total_xml_parsed': len(all_inscriptions),
        'dated_with_translations': len(dated),
        'embedding_model': 'all-MiniLM-L6-v2',
        'n_topics': n_topics,
        'n_outlier_docs': n_outlier,
        'century_distribution': {f"C{c}": century_counts[c] for c in sorted(century_counts)},
        'pre_929_count': len(pre_929),
        'post_929_count': len(post_929),
        'topics_only_pre_929': len(only_pre),
        'topics_only_post_929': len(only_post),
        'topics_persistent': len(persistent),
        'chi_square': comparison.get('chi_square', {}),
        'topic_keywords': topic_keywords,
        'caveats': [
            'C7(~2), C8(~5), C12(~1) too sparse for standalone analysis',
            'Focus on C9-C10 (densest) vs C11-C14 for meaningful comparison',
            'SBERT trained on modern English; semantic fidelity depends on translation quality',
            'BERTopic with small corpus (<100 docs) may produce unstable topics',
        ],
        'output_files': [
            'results/topic_heatmap.json',
            'results/pre_post_929_comparison.json',
            'results/e096_results.json',
        ],
    }

    with open(RESULTS_DIR / "e096_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total inscriptions parsed: {len(all_inscriptions)}")
    print(f"  Dated with translations: {len(dated)}")
    print(f"  BERTopic topics: {n_topics} (+ {n_outlier} outlier docs)")
    print(f"  Pre-929 inscriptions: {len(pre_929)}")
    print(f"  Post-929 inscriptions: {len(post_929)}")
    print(f"  Topics only pre-929: {len(only_pre)}")
    print(f"  Topics only post-929: {len(only_post)}")
    print(f"  Topics persistent: {len(persistent)}")
    if 'chi_square' in comparison and 'statistic' in comparison['chi_square']:
        print(f"  Chi-square p-value: {comparison['chi_square']['p_value']}")
    print(f"\n  All results saved to {RESULTS_DIR}/")
    print("=" * 70)
    print("E096 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
