"""
Generate all 7 figures for P18: What Words Remember
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = {
    'indigenous': '#2E86AB',
    'sanskrit': '#E8575A',
    'neutral': '#888888',
    'highlight': '#F6AE2D',
    'nusantara': '#2E86AB',
    'comparanda': '#888888',
}


def fig1_domain_gradient():
    """Figure 1: Domain penetration gradient (native vs Sanskrit)"""
    domains = [
        ("Agriculture", 91, 9),
        ("Technology/Craft", 82, 18),
        ("Nature/Environment", 76, 24),
        ("Body/Medicine", 68, 32),
        ("Kinship/Social", 62, 38),
        ("Trade/Economy", 55, 45),
        ("Governance/Law", 49, 51),
        ("War/Conflict", 45, 55),
        ("Religion/Ritual", 14, 86),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [d[0] for d in domains]
    native = [d[1] for d in domains]
    sanskrit = [d[2] for d in domains]
    y = range(len(domains))

    ax.barh(y, native, color=COLORS['indigenous'], label='Indigenous Austronesian', height=0.7)
    ax.barh(y, [-s for s in sanskrit], color=COLORS['sanskrit'], label='Sanskrit', height=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Percentage of vocabulary')
    ax.set_xlim(-100, 100)
    ax.set_xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(['80%', '60%', '40%', '20%', '0', '20%', '40%', '60%', '80%', '100%'])
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title('Domain Penetration Gradient: Indigenous vs Sanskrit Vocabulary')

    # Add zone labels
    ax.annotate('INDIGENOUS\nFORTRESS', xy=(70, 7.5), fontsize=8, color=COLORS['indigenous'],
                ha='center', fontweight='bold', alpha=0.7)
    ax.annotate('SANSKRIT\nDOMINATED', xy=(-65, 0.5), fontsize=8, color=COLORS['sanskrit'],
                ha='center', fontweight='bold', alpha=0.7)

    plt.tight_layout()
    fig.savefig(OUT / "fig1_domain_gradient.png")
    fig.savefig(OUT / "fig1_domain_gradient.pdf")
    plt.close()
    print("  Fig 1: Domain gradient saved")


def fig2_cultural_radar():
    """Figure 2: Cultural profile radar chart (9 domains)"""
    domains = [
        "Social/\nGovernance", "Knowledge/\nCognition", "Craft/\nTechnology",
        "Spatial/\nNavigation", "Hunting/\nGathering", "Body/\nMedicine",
        "Agriculture", "Fishing/\nMaritime", "Ritual/\nCosmology"
    ]
    values = [18.3, 14.9, 13.7, 13.7, 8.7, 8.3, 8.3, 7.5, 6.6]

    # Normalize to 0-1
    max_val = max(values)
    normalized = [v / max_val for v in values]

    angles = np.linspace(0, 2 * np.pi, len(domains), endpoint=False).tolist()
    normalized += normalized[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, normalized, color=COLORS['indigenous'], alpha=0.25)
    ax.plot(angles, normalized, color=COLORS['indigenous'], linewidth=2)
    ax.scatter(angles[:-1], normalized[:-1], color=COLORS['indigenous'], s=50, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(domains, size=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=7)
    ax.set_title('Cultural Profile of Pre-Hindu Nusantara\n(from 438 substrate vocabulary items)',
                 pad=20, fontsize=12)

    # Add percentage labels
    for angle, val, raw in zip(angles[:-1], normalized[:-1], values):
        ax.annotate(f'{raw}%', xy=(angle, val + 0.08),
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   color=COLORS['indigenous'])

    plt.tight_layout()
    fig.savefig(OUT / "fig2_cultural_radar.png")
    fig.savefig(OUT / "fig2_cultural_radar.pdf")
    plt.close()
    print("  Fig 2: Cultural radar saved")


def fig3_cascade_funnel():
    """Figure 3: Visibility cascade waterfall"""
    factors = [
        ("Total expected\nsettlements", 9659, None),
        ("After volcanic\nburial (×0.58)", 9659 * 0.58, "Volcanic Burial"),
        ("After organic\ndecay (×0.20)", 9659 * 0.58 * 0.20, "Organic Decay"),
        ("After survey\ncoverage (×0.025)", 9659 * 0.58 * 0.20 * 0.025, "Survey Deficit"),
        ("After recognition\nbias (×0.40)", 9659 * 0.58 * 0.20 * 0.025 * 0.40, "Recognition"),
        ("After publication\nbarrier (×0.50)", 9659 * 0.58 * 0.20 * 0.025 * 0.40 * 0.50, "Publication"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [f[0] for f in factors]
    values = [f[1] for f in factors]

    colors = ['#333333', '#E8575A', '#F6AE2D', '#E8575A', '#F6AE2D', '#F6AE2D']

    bars = ax.bar(range(len(factors)), values, color=colors, width=0.6, edgecolor='white', linewidth=0.5)

    ax.set_yscale('log')
    ax.set_ylim(0.5, 20000)
    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Number of sites', fontsize=11)
    ax.set_title('Multiplicative Visibility Cascade: From Expected to Observed', fontsize=12)

    # Add value labels
    for i, (label, val, _) in enumerate(factors):
        if val >= 1:
            ax.text(i, val * 1.3, f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(i, val * 1.5, f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add observed line
    ax.axhline(3, color=COLORS['indigenous'], linewidth=2, linestyle='--', alpha=0.8)
    ax.annotate('Observed: 0-3 sites', xy=(4.5, 3), fontsize=9,
               color=COLORS['indigenous'], fontweight='bold',
               ha='center', va='bottom')

    # Add leverage annotations
    ax.annotate('40× leverage\n(#1 priority)', xy=(3, values[3]),
               xytext=(3.5, 200), fontsize=8, color='red',
               arrowprops=dict(arrowstyle='->', color='red'),
               ha='center')

    plt.tight_layout()
    fig.savefig(OUT / "fig3_cascade_funnel.png")
    fig.savefig(OUT / "fig3_cascade_funnel.pdf")
    plt.close()
    print("  Fig 3: Cascade funnel saved")


def fig4_diffusion_timeline():
    """Figure 4: Global script diffusion with Java highlighted"""
    events = [
        ("Kharosthi", -250, 10, False),
        ("Tamil Brahmi", -200, 60, False),
        ("Greek (from Phoenician)", -800, 250, False),
        ("Latin (from Greek)", -700, 100, False),
        ("Sinhala", 100, 360, False),
        ("Champa", 200, 460, False),
        ("Funan/Cambodia", 250, 510, False),
        ("JAVA (Kutai)", 400, 660, True),
        ("Myanmar", 500, 760, False),
        ("Thailand", 550, 810, False),
        ("Arabic → Swahili", 1000, 400, False),
        ("Arabic → Malay (Jawi)", 1300, 700, False),
        ("Sriwijaya", 683, 943, False),
        ("Chinese → Japanese", 400, 1600, False),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by lag
    events.sort(key=lambda x: x[2])

    y_pos = range(len(events))
    colors = [COLORS['nusantara'] if e[3] else COLORS['comparanda'] for e in events]
    sizes = [300 if e[3] else 100 for e in events]

    for i, (name, date, lag, is_java) in enumerate(events):
        color = COLORS['nusantara'] if is_java else COLORS['comparanda']
        lw = 2.5 if is_java else 1
        ax.barh(i, lag, color=color, height=0.6, alpha=0.7 if not is_java else 1.0,
               edgecolor='black' if is_java else 'none', linewidth=lw)
        ax.text(lag + 20, i, f'{lag} yr', va='center', fontsize=8,
               fontweight='bold' if is_java else 'normal')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([e[0] for e in events], fontsize=9)
    ax.set_xlabel('Adoption lag (years from source script)', fontsize=11)
    ax.set_title('Global Script Diffusion Timelines\nJava (660 yr) = 57th percentile', fontsize=12)

    # Add mean line
    lags = [e[2] for e in events]
    ax.axvline(np.mean(lags), color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.annotate(f'Mean: {np.mean(lags):.0f} yr', xy=(np.mean(lags) + 20, len(events) - 1),
               fontsize=9, color='red')

    plt.tight_layout()
    fig.savefig(OUT / "fig4_diffusion_timeline.png")
    fig.savefig(OUT / "fig4_diffusion_timeline.pdf")
    plt.close()
    print("  Fig 4: Diffusion timeline saved")


def fig5_writing_vocabulary():
    """Figure 5: Writing vocabulary classification table as figure"""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis('off')

    data = [
        ['tulis', 'to write', 'Austronesian', 'PMP ~4000 BP', ''],
        ['surat', 'letter, message', 'Austronesian', 'PAN ~5000 BP', ''],
        ['ukir', 'to carve', 'Austronesian', 'PMP', ''],
        ['lontar', 'palm manuscript', 'Austronesian', 'Javanese', ''],
        ['aksara', 'letter, script', 'Sanskrit', '~1600 BP', ''],
        ['pustaka', 'book', 'Sanskrit', '~1600 BP', ''],
        ['kitab', 'book, scripture', 'Arabic', '~700 BP', ''],
    ]

    colors = []
    for row in data:
        if row[2] == 'Austronesian':
            colors.append([COLORS['indigenous'] + '30'] * 5)
        elif row[2] == 'Sanskrit':
            colors.append([COLORS['sanskrit'] + '30'] * 5)
        else:
            colors.append(['#F6AE2D30'] * 5)

    table = ax.table(
        cellText=data,
        colLabels=['Word', 'Meaning', 'Origin', 'Reconstructed Level', ''],
        cellColours=colors,
        colWidths=[0.12, 0.22, 0.15, 0.2, 0.02],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Header styling
    for j in range(5):
        table[0, j].set_facecolor('#333333')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Writing-Related Vocabulary by Origin\nIndigenous words for PROCESS, Sanskrit words for PRODUCTS',
                fontsize=11, pad=10)

    # Legend
    indigenous_patch = mpatches.Patch(color=COLORS['indigenous'], alpha=0.3, label='Indigenous Austronesian')
    sanskrit_patch = mpatches.Patch(color=COLORS['sanskrit'], alpha=0.3, label='Sanskrit borrowing')
    arabic_patch = mpatches.Patch(color='#F6AE2D', alpha=0.3, label='Arabic borrowing')
    ax.legend(handles=[indigenous_patch, sanskrit_patch, arabic_patch],
             loc='lower center', ncol=3, fontsize=8, frameon=False)

    plt.tight_layout()
    fig.savefig(OUT / "fig5_writing_vocabulary.png")
    fig.savefig(OUT / "fig5_writing_vocabulary.pdf")
    plt.close()
    print("  Fig 5: Writing vocabulary saved")


def fig7_comparative_chart():
    """Figure 7: Comparative complexity chart (10 societies)"""
    societies = [
        ("Nusantara\npre-Hindu", 23, True),
        ("Polynesian\nChiefdoms", 21, False),
        ("Great\nZimbabwe", 20, False),
        ("W. African\nIron Age", 16, False),
        ("Cahokia", 15, False),
        ("Norte Chico\n(Caral)", 14, False),
        ("Megalithic\nEurope", 14, False),
        ("Hopewell", 12, False),
        ("Jomon\nJapan", 11, False),
        ("Poverty\nPoint", 9, False),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    names = [s[0] for s in societies]
    values = [s[1] for s in societies]
    colors = [COLORS['nusantara'] if s[2] else COLORS['comparanda'] for s in societies]
    alphas = [1.0 if s[2] else 0.6 for s in societies]

    bars = ax.bar(range(len(societies)), values, color=colors, width=0.7,
                  edgecolor=['black' if s[2] else 'none' for s in societies],
                  linewidth=[2 if s[2] else 0 for s in societies])

    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    ax.set_xticks(range(len(societies)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Civilization Complexity Index (CCI, max 35)', fontsize=10)
    ax.set_title('Comparative Complexity: Pre-Literate Societies\nNusantara pre-Hindu ranks #1 (CCI=23, z=2.12)',
                fontsize=12)
    ax.set_ylim(0, 30)

    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=10, fontweight='bold')

    # Add mean line
    comparanda_mean = np.mean([s[1] for s in societies if not s[2]])
    ax.axhline(comparanda_mean, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.annotate(f'Comparanda mean: {comparanda_mean:.1f}',
               xy=(8, comparanda_mean + 0.5), fontsize=9, color='red')

    # Note
    ax.annotate('Lowest score: Architecture (1/5)\n= taphonomic prediction',
               xy=(0, 18), fontsize=8, color=COLORS['nusantara'],
               style='italic', ha='center')

    plt.tight_layout()
    fig.savefig(OUT / "fig7_comparative_chart.png")
    fig.savefig(OUT / "fig7_comparative_chart.pdf")
    plt.close()
    print("  Fig 7: Comparative chart saved")


if __name__ == "__main__":
    print("Generating P18 figures...")
    fig1_domain_gradient()
    fig2_cultural_radar()
    fig3_cascade_funnel()
    fig4_diffusion_timeline()
    fig5_writing_vocabulary()
    fig7_comparative_chart()
    print(f"\nAll figures saved to {OUT}")
    print("Note: Fig 6 (West Java map) requires geographic data — create manually or with GIS")
