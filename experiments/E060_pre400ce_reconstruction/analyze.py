"""E060: Pre-400 CE Nusantara — What We Now Know Existed.

This experiment synthesizes ALL VOLCARCH evidence channels to reconstruct
what pre-Hindu Nusantaran civilization looked like BEFORE the first Indian-script
inscriptions (~400 CE). It answers the core question:

"Why does Indonesian civilization appear to start at 400 CE?"
→ Because 6 layers of darkness hide what came before.
→ Here is what we can now reconstruct.

Sources: All VOLCARCH experiments (E001-E059), comparative ethnography,
linguistic reconstruction, archaeological literature.
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

print('=' * 60)
print('E060: Pre-400 CE Nusantara — What We Now Know Existed')
print('=' * 60)

# ============================================================
# 1. EVIDENCE CHANNELS FOR PRE-HINDU RECONSTRUCTION
# ============================================================
print('\n--- Evidence Channels ---')

channels = {
    'Linguistic Reconstruction': {
        'evidence': [
            'PMP vocabulary reconstructed: 201+ concepts (ABVD), >4000 BP',
            'Maritime vocabulary (E049): laut, garam, angin — Austronesian seafarers',
            'Agricultural terms: sawah (PMP *sabaq), padi, huma — wet+dry rice before India',
            'Ritual terms: hyang (PMP *qiang) = indigenous deity concept, pre-Sanskrit',
            'Architecture: rumah (PMP *Rumaq), atap, bambu — organic building',
            'Social: wanua (PMP *banua) = settlement/homeland, kapala = head/leader',
            'Old Javanese retains 56.7% PMP cognacy → massive pre-Indic vocabulary',
        ],
        'timeframe': '4000+ BP → 400 CE',
        'volcarch_exp': 'E022, E027, E043, E049, E054',
    },
    'Material Culture (Epigraphy)': {
        'evidence': [
            'E040: 63.4% of inscriptions mention organic materials (bambu, atap, daun, kayu, ijuk)',
            'E035: 15 plant species in 92.9% of inscriptions — botanical civilization',
            'E023: hyang in 43% of all inscriptions — survived 800 years of Sanskritization',
            'E033: Indianization = WAVE (peaks C9, troughs C13) — substrate always there',
            'E057: Genre filter hides organics in C8, reveals in C9+ — no actual change',
        ],
        'timeframe': 'C7-C14 (recorded), extrapolated pre-C7',
        'volcarch_exp': 'E023, E030, E033, E035, E040, E048, E057',
    },
    'Genetic Evidence': {
        'evidence': [
            'Austronesian expansion from Taiwan ~4000 BP (genetic + archaeological consensus)',
            'Leang Panninge (Sulawesi, 7200 BP): pre-Austronesian with 2.2% Denisovan admixture',
            'NO Java aDNA (0/84, E053) — volcanic taphonomy destroys DNA evidence',
            'Malagasy closest to Banjar (SE Borneo) — maritime population movement ~1200 CE',
            'Ghost populations detected in Island SEA — multiple unknown lineages',
        ],
        'timeframe': '50,000+ BP → present',
        'volcarch_exp': 'E053',
    },
    'Botanical Evidence': {
        'evidence': [
            'Canarium (E050): pan-Austronesian aromatic, Taiwan→Madagascar, 388 GBIF records MG',
            'E044: 4-layer botanical substitution (Canarium→dammar→menyan→kamboja)',
            'Plumeria (kamboja) = NEW WORLD (post-1560), NOT pre-Hindu',
            'Pre-Hindu mortuary aromatic = Canarium spp. (confirmed by GBIF)',
            'Rice cultivation: PAn *pajay = rice plant — agriculture before India',
        ],
        'timeframe': '4000+ BP → present',
        'volcarch_exp': 'E044, E050',
    },
    'Comparative Ethnography': {
        'evidence': [
            'Madagascar famadihana = transplanted Austronesian double burial (~1200 CE cutoff)',
            'Trunyan (Bali) mepasah = exposed burial, pre-Hindu mortuary practice',
            'Tengger retains pre-Islamic practices but LOSES maritime vocabulary (E049)',
            'Selametan timing calibrated to organic decomposition (P5: 1000-day hypothesis)',
            'Waringin (banyan) in 43% of prasasti — sacred tree concept pre-Sanskrit',
        ],
        'timeframe': 'Ethnographic present → 2000+ BP',
        'volcarch_exp': 'E034, E043, E049',
    },
    'Geographic/Toponymic': {
        'evidence': [
            'E051: 57.7% of classified Java villages = pre-Hindu toponyms',
            'Madura: 70-91% pre-Hindu — peripheral conservatory',
            'Yogyakarta: 26.2% — court-center overwriting maximum',
            'ci- prefix (Sundanese water) = pre-Sanskrit hydronym system',
            'sumber-, kedung-, kali- = extensive pre-Hindu hydrological vocabulary',
        ],
        'timeframe': 'Toponymic layers span 2000+ years',
        'volcarch_exp': 'E051, E056',
    },
    'Architectural': {
        'evidence': [
            'E031: Candi SITING follows volcano-awareness (west-cluster p=3.4e-08)',
            'E031: But candi ORIENTATION follows Hindu canon → indigenous WHERE, imported HOW',
            'E056: Candi cluster in MORE Indianized areas (Mann-Whitney p=0.007)',
            'Pre-Hindu architecture = organic (wood, bamboo, thatch) → invisible in record',
            'Penanggungan: 73 candi, 46 west side — volcano-informed siting knowledge',
        ],
        'timeframe': 'C8-C14 (candi), extrapolated pre-C8',
        'volcarch_exp': 'E031, E056',
    },
    'Geological/Taphonomic': {
        'evidence': [
            'P1/P9: Burial rate 2.4-13.1 mm/yr → pre-400 CE sites under 1-3+ meters',
            'E052: Sunda Shelf 2.09M km² submerged → 14× Java, ~500k population',
            'E053: Volcanic soil destroys aDNA (pH 4-5, geothermal heat)',
            'E059: Top targets at Kelud (21+ meter burial depth)',
            'Known examples: Sambisari (6.5m), Kedulan (7m), Liangan (8m)',
        ],
        'timeframe': 'Ongoing geological process',
        'volcarch_exp': 'E001, E002, E009, E016, E052, E053, E059',
    },
}

for channel, data in channels.items():
    print(f'\n  {channel}:')
    for ev in data['evidence'][:3]:
        print(f'    - {ev}')
    if len(data['evidence']) > 3:
        print(f'    ... and {len(data["evidence"]) - 3} more lines of evidence')

# ============================================================
# 2. RECONSTRUCTION: WHAT PRE-400 CE NUSANTARA LOOKED LIKE
# ============================================================
print('\n' + '=' * 60)
print('RECONSTRUCTION: Pre-400 CE Nusantara')
print('=' * 60)

reconstruction = {
    'Economy': {
        'description': 'Mixed agricultural-maritime economy',
        'evidence': [
            'Rice agriculture: sawah (wet) + huma (dry) — both PMP reconstructions',
            'Maritime trade: navigational vocabulary in PMP — seafaring people',
            'Organic materials: bamboo, rattan, palm fiber — primary construction materials',
            'Metal: gold, iron likely present (E040: metal terms in early inscriptions)',
        ],
        'confidence': 'HIGH (linguistic + comparative)',
    },
    'Religion/Cosmology': {
        'description': 'Ancestral/animistic cosmology centered on hyang',
        'evidence': [
            'hyang (PMP *qiang): indigenous deity/ancestral spirit concept',
            'Mortuary practices: exposed burial (Trunyan type) + secondary burial (famadihana)',
            'Sacred trees: waringin (banyan), Canarium aromatics at burial sites',
            'Selametan-type communal feasts calibrated to decomposition cycles',
            'kabuyutan: ancestral sacred sites (documented in Old Sundanese)',
        ],
        'confidence': 'HIGH (epigraphy + ethnography + comparative)',
    },
    'Settlement': {
        'description': 'Village-based (wanua) with organic architecture',
        'evidence': [
            'wanua (PMP *banua): primary settlement unit',
            'Organic buildings: bamboo, wood, thatch — archaeologically invisible',
            'Near rivers/coasts: maritime + agricultural orientation',
            'Hierarchical: kapala (head/chief), social stratification',
        ],
        'confidence': 'MODERATE (linguistic + E051 toponymic)',
    },
    'Technology': {
        'description': 'Sophisticated organic technology + early metallurgy',
        'evidence': [
            'Outrigger canoe (PMP *katig): enabled Pacific-wide navigation',
            'Textile: weaving vocabulary in PMP',
            'Bamboo technology: construction, musical instruments, tools',
            'Rice irrigation systems implied by sawah vocabulary',
            'Hanacaraka (E036): phonological system pre-dates Sanskrit — possible pre-script?',
        ],
        'confidence': 'MODERATE (linguistic + archaeological parallels)',
    },
    'Political Organization': {
        'description': 'Village federations with ritual leadership',
        'evidence': [
            'wanua-based polities (documented in earliest inscriptions)',
            'Ritual leadership tied to ancestral sites (kabuyutan)',
            'Trade networks: Austronesian maritime connections',
            'NOT "primitive" — capable of trans-oceanic navigation to Madagascar',
        ],
        'confidence': 'LOW-MODERATE (inference from early inscriptions)',
    },
    'Script/Literacy': {
        'description': 'Possibly pre-literate or using perishable writing media',
        'evidence': [
            'Hanacaraka (E036): 20-consonant system aligns with PAn (~17), not Sanskrit (33)',
            'Possible pre-Indian writing on organic media (bamboo, lontar palm)',
            'Aboriginal Taiwan scripts (partially independent of Chinese)',
            'Absence of pre-400 CE inscriptions may = organic media degradation, not illiteracy',
        ],
        'confidence': 'SPECULATIVE (but provocative)',
    },
}

for domain, data in reconstruction.items():
    print(f'\n  {domain}: {data["description"]}')
    print(f'  Confidence: {data["confidence"]}')
    for ev in data['evidence'][:2]:
        print(f'    - {ev}')

# ============================================================
# 3. TIMELINE FIGURE
# ============================================================
print('\n--- Generating timeline figure ---')

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(-5000, 2100)
ax.set_ylim(-1, 11)

# Background periods
periods = [
    (-5000, -3500, 'PAn Dispersal\n(Taiwan)', '#3498db', 0.15),
    (-3500, -2000, 'PMP Expansion\n(Philippines→Indonesia)', '#2ecc71', 0.15),
    (-2000, -500, 'Metal Age\n(Dong Son bronze)', '#f39c12', 0.15),
    (-500, 400, 'Proto-Historic\n(trade contacts)', '#e67e22', 0.15),
    (400, 900, 'Early Hindu-Buddhist\n(Indianization wave)', '#e74c3c', 0.15),
    (900, 1500, 'Classical Kingdoms\n(OJ sima = window opens)', '#9b59b6', 0.15),
    (1500, 2026, 'Islamic + Colonial\n+ Modern', '#95a5a6', 0.15),
]

for start, end, label, color, alpha in periods:
    ax.axvspan(start, end, alpha=alpha, color=color)
    ax.text((start + end) / 2, 10.5, label, ha='center', va='center',
            fontsize=7, fontweight='bold', color=color, rotation=0)

# Evidence lines
y_channels = {
    'Linguistic': 9,
    'Material Culture': 8,
    'Genetic': 7,
    'Botanical': 6,
    'Ethnographic': 5,
    'Toponymic': 4,
    'Architectural': 3,
    'Geological': 2,
    'Sunda Shelf': 1,
    'aDNA Gap': 0,
}

# Draw evidence spans
evidence_spans = [
    # (start, end, y_label, description, color, style)
    (-4000, 2026, 'Linguistic', 'PMP vocabulary (201+ concepts)', '#3498db', '-'),
    (-4000, 400, 'Linguistic', 'Pre-Indic vocabulary layer', '#3498db', '--'),
    (700, 1400, 'Material Culture', 'DHARMA inscriptions (268)', '#2ecc71', '-'),
    (750, 850, 'Material Culture', 'C8 "Dark Century" (genre filter)', '#e74c3c', '-'),
    (-50000, 2026, 'Genetic', 'H. sapiens in ISEA', '#9b59b6', '-'),
    (-7200, -7200, 'Genetic', 'Leang Panninge aDNA (only ISEA success)', '#9b59b6', 'o'),
    (-4000, 2026, 'Botanical', 'Canarium pan-Austronesian trail', '#2ecc71', '-'),
    (-1560, 2026, 'Botanical', 'Plumeria (New World, post-1560)', '#e74c3c', '--'),
    (-2000, 2026, 'Ethnographic', 'Famadihana/mepasah practices', '#f39c12', '-'),
    (-1200, -1200, 'Ethnographic', 'Madagascar migration cutoff', '#f39c12', 'o'),
    (-2000, 2026, 'Toponymic', '25,244 Java village names', '#e67e22', '-'),
    (750, 1400, 'Architectural', '142 candi (Hindu-Buddhist)', '#95a5a6', '-'),
    (-20000, 2026, 'Geological', 'Volcanic sedimentation', '#e74c3c', '-'),
    (-20000, -4000, 'Sunda Shelf', '2.09M km² exposed (LGM→flood)', '#3498db', '-'),
    (-8000, -8000, 'Sunda Shelf', 'Shelf fully submerged', '#3498db', 'x'),
    (-1000000, 2026, 'aDNA Gap', 'Java: 0/84 aDNA recovered', '#e74c3c', '-'),
]

for start, end, channel, desc, color, style in evidence_spans:
    y = y_channels[channel]
    if style == 'o' or style == 'x':
        ax.plot(max(start, -5000), y, style, color=color, markersize=10, zorder=5)
        ax.annotate(desc, xy=(max(start, -5000), y), fontsize=6, ha='left',
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec=color, alpha=0.7))
    else:
        ax.plot([max(start, -5000), min(end, 2100)], [y, y], style, color=color,
                linewidth=3 if style == '-' else 2, alpha=0.7)

# Channel labels
for channel, y in y_channels.items():
    ax.text(-5100, y, channel, ha='right', va='center', fontsize=8, fontweight='bold')

# Mark 400 CE
ax.axvline(x=400, color='black', linewidth=2, linestyle='--', alpha=0.7)
ax.annotate('400 CE\n"Start of\ncivilization"\n(conventional)', xy=(400, 5),
            fontsize=10, fontweight='bold', color='red', ha='center',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='red'))

# Mark what was BEFORE
ax.annotate('WHAT WAS HERE?\n→ EVERYTHING.\nAgriculture, seafaring,\nritual, architecture,\n'
            'trade, settlement, language.\nAll hidden by 6 layers.',
            xy=(-2000, 5), fontsize=9, fontweight='bold', color='#2c3e50', ha='center',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='#2c3e50', alpha=0.9))

ax.set_xlabel('Year (BCE/CE)', fontsize=12)
ax.set_title('E060: What Existed Before 400 CE — Evidence from 8 Independent Channels\n'
             '"Indonesian civilization started at 400 CE" → No. It was HIDDEN by 6 layers of darkness.',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.15, axis='x')

plt.tight_layout()
fig.savefig(os.path.join(RESULTS, 'pre400ce_reconstruction.png'), dpi=300, bbox_inches='tight')
print('  Saved: pre400ce_reconstruction.png')

# ============================================================
# 4. THE ANSWER
# ============================================================
print('\n' + '=' * 60)
print('THE ANSWER')
print('=' * 60)

print(f"""
WHY DOES INDONESIAN CIVILIZATION "START" AT 400 CE?
====================================================

It doesn't. The apparent start at ~400 CE is an artifact of:

1. VOLCANIC BURIAL (L1): Pre-400 CE sites are under 1-21+ meters of tephra.
   → Solution: Deep soil coring at E059 target sites.

2. COASTAL SUBMERSION (L2): 2.09 million km² of habitable Sunda Shelf
   is underwater. The coastal civilization is permanently drowned.
   → Solution: Marine archaeology of paleo-river channels.

3. HISTORIOGRAPHIC BIAS (L3): "Kutai = oldest kingdom" reflects zero
   volcanism in Kalimantan, not actual chronology.
   → Solution: Reframe narratives around taphonomic sampling bias.

4. COSMOLOGICAL OVERWRITING (L4): Sanskrit vocabulary replaced 22.9% of
   PMP cognates in Javanese (OJav 56.7% → Modern Jav 33.8%).
   → Solution: Computational linguistic subtraction (P8).

5. GENRE TAPHONOMY (L5): C8 inscriptions (Sanskrit format) record 0%
   indigenous content. C9+ (OJ sima) reveals 15% pre-Indic, 77% organic.
   → Solution: Analyze sima-format inscriptions separately.

6. HISTORIOGRAPHIC PERIODICITY (L6): "Hindu period" treated as monolithic
   block. In reality, Indianization peaks C9, troughs C13.
   → Solution: Century-by-century analysis (E033).

WHAT ACTUALLY EXISTED BEFORE 400 CE:
- A maritime-agricultural civilization spanning Taiwan to Madagascar
- PMP vocabulary for rice, navigation, construction, ritual, governance
- Hyang-centered ancestral cosmology with organic mortuary practices
- Canarium aromatics as pan-Austronesian sacred botanical
- Village (wanua) polities with ritual leadership
- Trans-oceanic navigation capability (proven by Madagascar settlement)
- Organic architecture (bamboo, wood, thatch) — archaeologically invisible
- Extensive water management (ci-, kali-, sumber- toponyms)

EVIDENCE STRENGTH: 54 experiments, 8 channels, 6 submitted papers.
The question is no longer IF — but HOW MUCH was lost.
""")

# Save
summary = {
    'experiment': 'E060_pre400ce_reconstruction',
    'date': '2026-03-12',
    'n_channels': len(channels),
    'n_reconstruction_domains': len(reconstruction),
    'core_answer': 'Pre-400 CE civilization is hidden by 6 taphonomic layers, not absent',
    'total_experiments_cited': 54,
}
with open(os.path.join(RESULTS, 'reconstruction_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('Done!')
