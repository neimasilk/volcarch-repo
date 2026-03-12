"""
P9 Figure Generation — Peripheral Conservatism Framework
Generates 4 publication-quality figures for JSEAS submission.

AI Disclosure: Figures generated with matplotlib by Claude Code (Anthropic)
under direction of the human author. All data, hypotheses, and interpretations
are by the human author.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import io

# Windows UTF-8 fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ============================================================
# FIGURE 1: Cognacy Gradient
# ============================================================
def fig1_cognacy_gradient():
    """Horizontal bar chart of PMP cognacy rates by variety, color-coded by PCF type."""

    varieties = [
        'Old Javanese',
        'Merina Malagasy',
        'Balinese',
        'Javanese (std)',
        'Javanese (Yogya)',
        'Tengger (Ngadas)',
        'Javanese (Malang)',
    ]
    pmp_rates = [55.3, 40.8, 40.3, 33.0, 28.4, 27.7, 26.0]
    pan_rates = [37.4, 28.6, 26.7, 23.3, 20.0, 19.8, 17.7]

    # PCF type colors
    colors_pmp = [
        '#4a86c8',   # Old Javanese — baseline (blue)
        '#e07b39',   # Malagasy — Type C (orange)
        '#4aaf4a',   # Balinese — Type A large (green)
        '#d44040',   # Javanese std — centre (red)
        '#d44040',   # Yogya — sub-centre (red)
        '#a0a0a0',   # Tengger — small isolate (grey)
        '#d44040',   # Malang — sub-centre (red)
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    y_pos = np.arange(len(varieties))

    # PMP bars (main)
    bars = ax.barh(y_pos, pmp_rates, height=0.55, color=colors_pmp, edgecolor='white', linewidth=0.5)

    # PAn bars (overlaid, lighter)
    ax.barh(y_pos, pan_rates, height=0.55, color=[c + '60' for c in colors_pmp],
            edgecolor='none', alpha=0.4)

    # Add value labels
    for i, (pmp, pan) in enumerate(zip(pmp_rates, pan_rates)):
        ax.text(pmp + 0.8, i, f'{pmp}%', va='center', ha='left', fontsize=9, fontweight='bold')

    # Reference line at Javanese std
    ax.axvline(x=33.0, color='#d44040', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(33.5, -0.7, 'Centre\nbaseline', fontsize=7, color='#d44040', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(varieties, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('PMP Cognacy Rate (%)')
    ax.set_xlim(0, 65)
    ax.set_title('Figure 2. PMP cognacy gradient across Austronesian varieties', fontsize=11, fontweight='bold', loc='left')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#4a86c8', label='Historical baseline'),
        mpatches.Patch(facecolor='#e07b39', label='Type C (temporal)'),
        mpatches.Patch(facecolor='#4aaf4a', label='Type A (geographic, large)'),
        mpatches.Patch(facecolor='#d44040', label='Centre / sub-centre'),
        mpatches.Patch(facecolor='#a0a0a0', label='Type A (small isolate)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=8)

    # Annotation
    ax.annotate('22.3% erosion\n(~1000 years)', xy=(44, 0.3), xytext=(50, 2),
                fontsize=8, ha='center', color='#333',
                arrowprops=dict(arrowstyle='->', color='#666', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#ccc'))

    plt.tight_layout()
    plt.savefig('fig1_cognacy_gradient.png')
    plt.savefig('fig1_cognacy_gradient.pdf')
    plt.close()
    print("  Fig 1: Cognacy gradient — DONE")


# ============================================================
# FIGURE 2: Indianization Wave
# ============================================================
def fig2_indianization_wave():
    """Line chart showing the rise and fall of Sanskrit vocabulary ratio over time."""

    centuries = [7, 9, 10, 11, 13, 14]
    indic_ratio = [0.556, 0.807, 0.791, 0.703, 0.569, 0.876]
    ci_lower = [0.000, 0.721, 0.727, 0.517, 0.369, 0.716]
    ci_upper = [1.000, 0.882, 0.843, 0.864, 0.742, 1.000]
    n_inscriptions = [3, 28, 42, 10, 10, 5]

    # Pre-Indic diversity
    pre_indic_terms = [0, 1, 5, 5, 3, 2]  # approximate unique pre-Indic terms per century

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.08})

    # Top panel: Indic ratio
    yerr_lower = [r - l for r, l in zip(indic_ratio, ci_lower)]
    yerr_upper = [u - r for r, u in zip(indic_ratio, ci_upper)]

    ax1.fill_between(centuries, ci_lower, ci_upper, alpha=0.15, color='#c44e52')
    ax1.plot(centuries, indic_ratio, 'o-', color='#c44e52', linewidth=2, markersize=8, zorder=5)
    ax1.errorbar(centuries, indic_ratio, yerr=[yerr_lower, yerr_upper],
                 fmt='none', ecolor='#c44e52', capsize=4, alpha=0.6)

    # Annotate sample sizes
    for c, r, n in zip(centuries, indic_ratio, n_inscriptions):
        ax1.annotate(f'n={n}', (c, r), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=7, color='#888')

    # Key annotations
    ax1.annotate('Peak: Medang era\n(0.807)', xy=(9, 0.807), xytext=(10.5, 0.88),
                fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color='#666'),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff8f0', edgecolor='#ddd'))

    ax1.annotate('Trough: Singhasari\n(0.569)', xy=(13, 0.569), xytext=(11.5, 0.48),
                fontsize=8, ha='right',
                arrowprops=dict(arrowstyle='->', color='#666'),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff8f0', edgecolor='#ddd'))

    # Trend line (rho = -0.211)
    ax1.text(7.5, 0.42, r'Spearman $\rho$ = $-$0.211, p = 0.030', fontsize=8,
            style='italic', color='#666')

    ax1.set_ylabel('Indic Vocabulary Ratio')
    ax1.set_ylim(0.3, 1.05)
    ax1.set_title('Figure 4. The Indianization wave: Sanskrit vocabulary ratio in Old Javanese inscriptions',
                  fontsize=10, fontweight='bold', loc='left')
    ax1.axhline(y=0.5, color='#ddd', linestyle=':', linewidth=0.8)

    # Bottom panel: Pre-Indic diversity
    ax2.bar(centuries, pre_indic_terms, width=0.6, color='#4a86c8', alpha=0.7, edgecolor='white')
    ax2.set_ylabel('Pre-Indic\nterm types', fontsize=9)
    ax2.set_xlabel('Century CE')
    ax2.set_ylim(0, 6.5)
    ax2.set_xticks(centuries)
    ax2.set_xticklabels([f'C{c}' for c in centuries])

    # Label
    ax2.text(12, 5.5, 'Substrate\ndiversifies', fontsize=8, ha='center', color='#4a86c8',
            style='italic')

    plt.tight_layout()
    plt.savefig('fig2_indianization_wave.png')
    plt.savefig('fig2_indianization_wave.pdf')
    plt.close()
    print("  Fig 2: Indianization wave — DONE")


# ============================================================
# FIGURE 3: Botanical Substitution Chain (4-layer diagram)
# ============================================================
def fig3_botanical_layers():
    """Timeline/diagram showing the 4-layer botanical palimpsest at Javanese cemeteries."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')

    # Title
    ax.text(5, 5.2, 'Figure 5. The four-layer botanical palimpsest at Nusantaran burial sites',
            ha='center', fontsize=11, fontweight='bold')

    # Layer boxes (bottom to top = oldest to newest)
    layers = [
        {
            'y': 0.0, 'color': '#8B6914', 'alpha': 0.25,
            'label': 'Layer 1: Pre-Hindu (pre-400 CE)',
            'desc': 'Exposed burial + native aromatic tree',
            'species': 'Canarium spp. (pan-Austronesian)\nTaru Menyan (Trunyan-specific)',
            'evidence': 'Trunyan (Bali), Madagascar (Canarium resin)',
            'arrow_label': 'Survives at\nperiphery'
        },
        {
            'y': 1.2, 'color': '#B8860B', 'alpha': 0.25,
            'label': 'Layer 2: Indianization (400-1200 CE)',
            'desc': 'Inhumation + aromatic as grave marker',
            'species': 'Cendana (sandalwood) — 11/268 prasasti\nNative aromatic retained (species unknown)',
            'evidence': 'DHARMA inscriptions (100% ritual co-occurrence)',
            'arrow_label': ''
        },
        {
            'y': 2.4, 'color': '#2E8B57', 'alpha': 0.25,
            'label': 'Layer 3: Islamization (1200-1560 CE)',
            'desc': 'Inhumation reinforced + aromatic tradition continues',
            'species': 'Unknown native aromatic\n(gap: identity erased by Layer 4)',
            'evidence': 'No direct evidence — the replacement was complete',
            'arrow_label': ''
        },
        {
            'y': 3.6, 'color': '#CD3700', 'alpha': 0.25,
            'label': 'Layer 4: Portuguese contact (1560s+)',
            'desc': 'Plumeria (kamboja) introduced from New World',
            'species': 'Plumeria spp. — Mexico/Caribbean origin\nIntroduced via Manila-Acapulco galleon',
            'evidence': 'ABSENT from Madagascar (pre-dates introduction)',
            'arrow_label': 'Visible at\ncentre today'
        },
    ]

    for layer in layers:
        y = layer['y']
        # Background rectangle
        rect = mpatches.FancyBboxPatch((0.1, y), 9.3, 1.0,
                                        boxstyle="round,pad=0.05",
                                        facecolor=layer['color'],
                                        alpha=layer['alpha'],
                                        edgecolor=layer['color'],
                                        linewidth=1.5)
        ax.add_patch(rect)

        # Layer label (bold)
        ax.text(0.3, y + 0.78, layer['label'], fontsize=9, fontweight='bold',
                color=layer['color'], va='top')

        # Description
        ax.text(0.3, y + 0.45, layer['desc'], fontsize=8, color='#333', va='top')

        # Species (right column)
        ax.text(5.2, y + 0.78, layer['species'], fontsize=7.5, color='#555',
                va='top', style='italic')

    # Diagnostic prediction box
    box_text = ('Diagnostic prediction:\n'
                'Plumeria ABSENT from Madagascar [confirmed]\n'
                'Canarium PRESENT in Madagascar [confirmed]\n'
                '=> 4-layer model validated')

    props = dict(boxstyle='round,pad=0.4', facecolor='#f0f8ff', edgecolor='#4a86c8', linewidth=1.5)
    ax.text(5, -0.35, box_text, fontsize=8, ha='center', va='top', bbox=props,
            fontweight='normal')

    # Time arrow on left
    ax.annotate('', xy=(-0.3, 4.6), xytext=(-0.3, 0.0),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    ax.text(-0.4, 2.3, 'Time →', rotation=90, fontsize=8, color='#666',
            ha='center', va='center')

    plt.tight_layout()
    plt.savefig('fig3_botanical_layers.png')
    plt.savefig('fig3_botanical_layers.pdf')
    plt.close()
    print("  Fig 3: Botanical layers — DONE")


# ============================================================
# FIGURE 4: PCF Convergence Diagram
# ============================================================
def fig4_pcf_convergence():
    """Conceptual diagram showing the PCF: 3 periphery types × 3 evidence channels → convergence.
    Simplified layout for clear rendering at journal page width."""

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Title
    ax.text(6, 8.7, 'Figure 1. The Peripheral Conservatism Framework',
            ha='center', fontsize=13, fontweight='bold')

    # Column headers
    ax.text(2, 7.9, 'OVERWRITING\nEVENTS', ha='center', fontsize=10,
            fontweight='bold', color='#8B0000')
    ax.text(6, 7.9, 'PERIPHERAL\nARCHIVES', ha='center', fontsize=10,
            fontweight='bold', color='#006400')
    ax.text(10, 7.9, 'RECOVERY\nCHANNELS', ha='center', fontsize=10,
            fontweight='bold', color='#00008B')

    # ---- LEFT COLUMN: Three overwriting events ----
    events = [
        ('Indianization\n400–1500 CE', 6.8),
        ('Islamization\n1200–1700 CE', 5.8),
        ('Colonialism\n1600–1945 CE', 4.8),
    ]
    for label, y in events:
        rect = mpatches.FancyBboxPatch((0.3, y - 0.4), 3.4, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#CD5C5C', alpha=0.15,
                                        edgecolor='#CD5C5C', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(2, y, label, ha='center', va='center', fontsize=10, color='#8B0000')

    # Arrow + label
    ax.annotate('', xy=(2, 3.9), xytext=(2, 4.3),
                arrowprops=dict(arrowstyle='->', color='#8B0000', lw=2))
    ax.text(2, 3.6, 'Erased at centre', ha='center', fontsize=9,
            fontweight='bold', color='#8B0000')

    # ---- MIDDLE COLUMN: Three periphery types ----
    peripheries = [
        ('Type A: Geographic\nBali Aga, Toraja, Muna', 6.8),
        ('Type B: Political\nOsing, Banyumas', 5.8),
        ('Type C: Temporal\nMadagascar (~1200 CE)', 4.8),
    ]
    for label, y in peripheries:
        rect = mpatches.FancyBboxPatch((4.1, y - 0.4), 3.8, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#228B22', alpha=0.12,
                                        edgecolor='#228B22', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(6, y, label, ha='center', va='center', fontsize=10, color='#006400')

    ax.annotate('', xy=(6, 3.9), xytext=(6, 4.3),
                arrowprops=dict(arrowstyle='->', color='#006400', lw=2))
    ax.text(6, 3.6, 'Preserved at periphery', ha='center', fontsize=9,
            fontweight='bold', color='#006400')

    # ---- RIGHT COLUMN: Three recovery channels ----
    channels = [
        ('Lexical\nABVD cognacy rates', 6.8),
        ('Ritual / Mortuary\nTrunyan, famadihana', 5.8),
        ('Botanical\nCanarium distribution', 4.8),
    ]
    for label, y in channels:
        rect = mpatches.FancyBboxPatch((8.1, y - 0.4), 3.8, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#4169E1', alpha=0.12,
                                        edgecolor='#4169E1', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(10, y, label, ha='center', va='center', fontsize=10, color='#00008B')

    ax.annotate('', xy=(10, 3.9), xytext=(10, 4.3),
                arrowprops=dict(arrowstyle='->', color='#00008B', lw=2))
    ax.text(10, 3.6, 'Computational detection', ha='center', fontsize=9,
            fontweight='bold', color='#00008B')

    # ---- BOTTOM: Convergence box ----
    convergence_box = mpatches.FancyBboxPatch((1.0, 1.5), 10.0, 1.8,
                                               boxstyle="round,pad=0.15",
                                               facecolor='#FFD700', alpha=0.2,
                                               edgecolor='#B8860B', linewidth=2.5)
    ax.add_patch(convergence_box)

    ax.text(6, 3.0, 'CONVERGENCE', ha='center', fontsize=13,
            fontweight='bold', color='#8B6914')

    ax.text(6, 2.55, 'Pre-Hindu Nusantaran organic material culture', ha='center',
            fontsize=11, color='#333')

    ax.text(6, 2.05, 'Balinese 40% > Javanese 33% PMP cognacy  |  '
            'Indianization = wave, not permanent  |  '
            'Canarium = pan-Austronesian aromatic',
            ha='center', fontsize=8.5, color='#555')

    # Arrows from all three columns to convergence
    for x in [2, 6, 10]:
        ax.annotate('', xy=(x, 3.3), xytext=(x, 3.55),
                    arrowprops=dict(arrowstyle='->', color='#B8860B', lw=2))

    # Scale paradox note
    ax.text(6, 0.9, 'Scale paradox: conservation requires critical mass (~3M+ speakers).\n'
            'Small isolates (Tengger, ~100K) drift rather than conserve.',
            ha='center', fontsize=9, color='#888', style='italic')

    # AI disclosure
    ax.text(6, 0.2, 'Generated with AI assistance (Claude, Anthropic). '
            'All hypotheses, data, and interpretations by human author.',
            ha='center', fontsize=7.5, color='#aaa', style='italic')

    plt.savefig('fig4_pcf_convergence.png', bbox_inches='tight', pad_inches=0.2)
    plt.savefig('fig4_pcf_convergence.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print("  Fig 4: PCF convergence — DONE")


# ============================================================
# FIGURE 5: Organic vs Lithic Materials (E040)
# ============================================================
def fig5_organic_civilization():
    """Stacked bar chart showing organic vs lithic material mentions by century."""

    centuries = ['C8', 'C9', 'C10', 'C11', 'C13']
    n_total = [55, 28, 49, 11, 10]
    organic_pct = [13, 68, 82, 91, 90]
    lithic_pct = [5, 29, 39, 45, 40]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [1.2, 1]})

    # Left panel: temporal trend
    x = np.arange(len(centuries))
    width = 0.35

    bars1 = ax1.bar(x - width/2, organic_pct, width, label='Organic materials',
                    color='#4aaf4a', alpha=0.8, edgecolor='white')
    bars2 = ax1.bar(x + width/2, lithic_pct, width, label='Lithic materials',
                    color='#a0a0a0', alpha=0.8, edgecolor='white')

    # Value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{int(bar.get_height())}%', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{int(bar.get_height())}%', ha='center', va='bottom', fontsize=7)

    # Genre annotation
    ax1.annotate('Sanskrit genre\n(short labels)', xy=(0, 13), xytext=(0.8, 30),
                fontsize=7, ha='center', color='#666',
                arrowprops=dict(arrowstyle='->', color='#999'),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff', edgecolor='#ddd'))

    ax1.annotate('Old Javanese sīma genre\n(detailed land grants)', xy=(2, 82), xytext=(2.8, 55),
                fontsize=7, ha='center', color='#666',
                arrowprops=dict(arrowstyle='->', color='#999'),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff', edgecolor='#ddd'))

    ax1.set_xlabel('Century CE')
    ax1.set_ylabel('% of inscriptions mentioning material')
    ax1.set_xticks(x)
    ax1.set_xticklabels(centuries)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title('(a) Temporal pattern', fontsize=10, fontweight='bold')

    # Right panel: overall summary (pie-like stacked bar)
    categories = ['Organic\nonly', 'Both', 'Lithic\nonly', 'Neither']
    values = [103, 67, 6, 92]
    colors = ['#4aaf4a', '#FFD700', '#a0a0a0', '#ddd']

    bars = ax2.barh(categories, values, color=colors, edgecolor='white', height=0.6)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2.,
                f'{val}', ha='left', va='center', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Number of inscriptions (n=268)')
    ax2.set_title('(b) Material class distribution', fontsize=10, fontweight='bold')
    ax2.set_xlim(0, 130)

    # Highlight the 103:6 ratio
    ax2.text(80, 2.8, 'Organic-only : Lithic-only\n= 103 : 6 (17:1)',
            fontsize=8, ha='center', color='#333', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0fff0', edgecolor='#4aaf4a'))

    fig.suptitle('Figure 6. The organic material world: material culture in 268 Old Javanese inscriptions',
                 fontsize=10, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('fig5_organic_civilization.png')
    plt.savefig('fig5_organic_civilization.pdf')
    plt.close()
    print("  Fig 5: Organic civilization — DONE")


# ============================================================
# FIGURE 6: Semantic Domain Heatmap
# ============================================================
def fig6_domain_heatmap():
    """Heatmap showing PMP cognacy rates by semantic domain × language variety."""

    domains = ['NATURE', 'NUMBER', 'KINSHIP', 'BODY', 'ACTION', 'OTHER']
    varieties = ['Balinese', 'Javanese\n(std)', 'Tengger', 'Malagasy']

    data = np.array([
        [72.7, 45.5, 36.4, 54.5],  # NATURE
        [50.0, 50.0, 100.0, 50.0], # NUMBER
        [25.0, 12.5, 12.5, 50.0],  # KINSHIP
        [30.0, 30.0, 33.3, 50.0],  # BODY
        [20.8, 29.2, 25.0, 45.8],  # ACTION
        [40.2, 32.0, 23.1, 34.4],  # OTHER
    ])

    fig, ax = plt.subplots(figsize=(5.5, 4))

    im = ax.imshow(data, cmap='YlGn', aspect='auto', vmin=10, vmax=100)

    ax.set_xticks(np.arange(len(varieties)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels(varieties, fontsize=9)
    ax.set_yticklabels(domains, fontsize=9)

    # Add text annotations
    for i in range(len(domains)):
        for j in range(len(varieties)):
            val = data[i, j]
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                   fontsize=9, color=color, fontweight='bold' if val > 50 else 'normal')

    # Highlight peripheral advantage cells
    # Nature row: Balinese >> Javanese
    rect = mpatches.Rectangle((-0.45, -0.45), 0.9, 0.9,
                               linewidth=2, edgecolor='#FF4500', facecolor='none',
                               linestyle='--')
    ax.add_patch(rect)
    ax.text(-0.5, -0.65, '+27.2%', fontsize=7, color='#FF4500', fontweight='bold', ha='center')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='PMP Cognacy Rate (%)')

    ax.set_title('Figure 3. PMP cognacy by semantic domain', fontsize=10,
                fontweight='bold', loc='left', pad=15)

    plt.tight_layout()
    plt.savefig('fig6_domain_heatmap.png')
    plt.savefig('fig6_domain_heatmap.pdf')
    plt.close()
    print("  Fig 6: Domain heatmap — DONE")


# ============================================================
# RUN ALL
# ============================================================
if __name__ == '__main__':
    print("Generating P9 figures...")
    fig1_cognacy_gradient()
    fig2_indianization_wave()
    fig3_botanical_layers()
    fig4_pcf_convergence()
    fig5_organic_civilization()
    fig6_domain_heatmap()
    print("\nAll 6 figures generated (PNG + PDF).")
    print("AI Disclosure: Figures generated with matplotlib via Claude Code (Anthropic).")
    print("All data, hypotheses, and interpretations by human author Mukhlis Amien.")
