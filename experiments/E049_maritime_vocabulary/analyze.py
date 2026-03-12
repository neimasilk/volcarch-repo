"""E049: Maritime & Nature Vocabulary Conservation Across Austronesian Peripheries.

Tests whether maritime/nature vocabulary is MORE conserved in peripheral
communities than general vocabulary — i.e., does the sea preserve memory
better than the court?

Hypothesis: Maritime and nature vocabulary shows higher PMP cognacy in
peripheral communities (Balinese, Malagasy, Muna) than in centre (Javanese).
This would indicate that the "organic civilization" was also a "maritime
civilization" whose vocabulary survived overwriting.

Uses same ABVD data as E043 but with semantic domain analysis extended
to maritime concepts.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BASE, '..', '..'))
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

ABVD_DIR = os.path.join(REPO, 'experiments/E022_linguistic_subtraction/data/abvd/cldf')

print('=' * 60)
print('E049: Maritime & Nature Vocabulary Conservation')
print('=' * 60)

# ============================================================
# 1. LOAD ABVD DATA
# ============================================================
forms = pd.read_csv(os.path.join(ABVD_DIR, 'forms.csv'))
params = pd.read_csv(os.path.join(ABVD_DIR, 'parameters.csv'))
langs = pd.read_csv(os.path.join(ABVD_DIR, 'languages.csv'))
cognates = pd.read_csv(os.path.join(ABVD_DIR, 'cognates.csv'))

print(f'Loaded: {len(forms)} forms, {len(params)} concepts, {len(langs)} languages')

# ============================================================
# 2. DEFINE SEMANTIC DOMAINS
# ============================================================
# Expanded domain classification for 210 ABVD concepts
maritime_concepts = [
    '9_toswim', '65_rope', '111_fish', '121_sand', '122_water',
    '124_sea', '125_salt', '126_lake', '133_rain', '136_wind',
]

nature_concepts = [
    '79_stickwood', '114_leaf', '115_root', '116_flower', '117_fruit',
    '118_grass', '119_earthsoil', '120_stone', '127_woodsforest',
    '128_sky', '129_moon', '130_star', '131_cloud', '143_fire',
]

body_concepts = [
    '1_hand', '4_legfoot', '12_skin', '14_belly', '15_bone',
    '16_intestines', '17_liver', '18_breast', '22_head', '23_neck',
    '24_hair', '25_nose', '26_tosniffsmell', '27_mouth', '28_tooth',
    '29_tongue', '30_tolarghto', '31_toeat', '33_ear', '34_eye',
    '38_blood', '39_thechin',
]

action_concepts = [
    '5_towalk', '7_tocome', '8_toturn', '9_toswim', '47_tocook',
    '61_tohunt', '63_tokill', '70_todig', '71_tobuy', '83_tosew',
    '84_tocount', '87_tosqueeze', '93_tosay', '94_tosing', '96_toplay',
    '97_tofloat', '98_toflow', '170_tosuck', '171_toblow',
]

kinship_concepts = [
    '100_father', '101_mother', '102_olderbroolder', '103_youngerbrother',
    '104_oldersisterolder', '105_youngersister', '106_wife', '107_husband',
    '108_child', '109_name',
]

numeral_concepts = [
    '196_one', '197_two', '198_three', '199_four', '200_five',
    '201_six', '202_seven', '203_eight', '204_nine', '205_ten',
    '206_twenty', '207_fifty', '208_onehundred', '209_onethousand',
]

# Verify concepts exist
all_param_ids = set(params['ID'].tolist())
for domain_name, domain_ids in [('Maritime', maritime_concepts), ('Nature', nature_concepts),
                                  ('Body', body_concepts), ('Action', action_concepts),
                                  ('Kinship', kinship_concepts), ('Numeral', numeral_concepts)]:
    valid = [c for c in domain_ids if c in all_param_ids]
    missing = [c for c in domain_ids if c not in all_param_ids]
    if missing:
        print(f'  WARNING: {domain_name} missing concepts: {missing[:5]}')
    print(f'  {domain_name}: {len(valid)}/{len(domain_ids)} concepts found')

# ============================================================
# 3. DEFINE TARGET LANGUAGES
# ============================================================
target_langs = {
    'PMP': 269,       # Proto-Malayo-Polynesian
    'Old Javanese': 290,
    'Javanese': 20,
    'Balinese': 1,
    'Malagasy': 92,    # Merina
    'Muna': None,      # Need to find
    'Tengger': 1533,
}

# Find Muna
for _, row in langs.iterrows():
    if 'muna' in str(row.get('Name', '')).lower():
        target_langs['Muna'] = row['ID']
        print(f'\n  Found Muna: ID={row["ID"]}, Name={row["Name"]}')
        break

# Map language ID to forms
lang_id_to_name = {}
for name, lid in target_langs.items():
    if lid is not None:
        lang_id_to_name[lid] = name

print(f'\n  Target languages: {list(target_langs.keys())}')

# ============================================================
# 4. BUILD COGNACY MATRIX
# ============================================================
print('\n--- Building cognacy matrix ---')

# For each language and concept, check if PMP cognate exists
# Method: a form is PMP-cognate if it shares a cognate set with PMP form

def get_cognate_sets(lang_id):
    """Get cognate set IDs for all forms of a language."""
    lang_forms = forms[forms['Language_ID'] == lang_id]
    lang_cognates = cognates[cognates['Form_ID'].isin(lang_forms['ID'])]
    # Map concept -> set of cognate set IDs
    concept_cognates = {}
    for _, row in lang_cognates.iterrows():
        form_row = lang_forms[lang_forms['ID'] == row['Form_ID']]
        if not form_row.empty:
            concept_id = form_row.iloc[0]['Parameter_ID']
            if concept_id not in concept_cognates:
                concept_cognates[concept_id] = set()
            concept_cognates[concept_id].add(row['Cognateset_ID'])
    return concept_cognates

# Get PMP cognate sets
print('  Computing PMP cognate sets...')
pmp_cognates = get_cognate_sets(269)
print(f'  PMP: {len(pmp_cognates)} concepts with cognate sets')

# Get cognate sets for each target language
lang_cognates = {}
for name, lid in target_langs.items():
    if lid is not None and name != 'PMP':
        print(f'  Computing {name} cognate sets...')
        lang_cognates[name] = get_cognate_sets(lid)
        print(f'    {name}: {len(lang_cognates[name])} concepts with cognate sets')

# ============================================================
# 5. COMPUTE DOMAIN-SPECIFIC COGNACY RATES
# ============================================================
print('\n--- Computing domain-specific PMP cognacy rates ---')

domains = {
    'Maritime': maritime_concepts,
    'Nature': nature_concepts,
    'Body': body_concepts,
    'Action': action_concepts,
    'Kinship': kinship_concepts,
    'Numeral': numeral_concepts,
    'ALL': list(all_param_ids),
}

results = []

for lang_name, lcog in lang_cognates.items():
    for domain_name, domain_ids in domains.items():
        if domain_name == 'ALL':
            test_concepts = [c for c in all_param_ids if c in pmp_cognates and c in lcog]
        else:
            test_concepts = [c for c in domain_ids if c in all_param_ids and c in pmp_cognates]

        n_testable = len(test_concepts)
        n_cognate = 0

        for concept in test_concepts:
            if concept in lcog:
                # Check if any cognate set overlaps with PMP
                pmp_sets = pmp_cognates.get(concept, set())
                lang_sets = lcog.get(concept, set())
                if pmp_sets & lang_sets:  # intersection
                    n_cognate += 1

        rate = n_cognate / n_testable * 100 if n_testable > 0 else 0

        results.append({
            'language': lang_name,
            'domain': domain_name,
            'n_testable': n_testable,
            'n_cognate': n_cognate,
            'cognacy_pct': rate,
        })

df_results = pd.DataFrame(results)

# Print summary table
print('\n--- PMP Cognacy Rates by Domain ---')
pivot = df_results.pivot(index='domain', columns='language', values='cognacy_pct')
# Reorder columns
col_order = ['Old Javanese', 'Balinese', 'Javanese', 'Tengger', 'Malagasy', 'Muna']
col_order = [c for c in col_order if c in pivot.columns]
pivot = pivot[col_order]
print(pivot.round(1).to_string())

# ============================================================
# 6. PERIPHERAL ADVANTAGE BY DOMAIN
# ============================================================
print('\n--- Peripheral Advantage (Balinese - Javanese) by Domain ---')

advantages = []
for domain_name in domains.keys():
    bal = df_results[(df_results['language'] == 'Balinese') & (df_results['domain'] == domain_name)]
    jav = df_results[(df_results['language'] == 'Javanese') & (df_results['domain'] == domain_name)]
    if not bal.empty and not jav.empty:
        diff = bal.iloc[0]['cognacy_pct'] - jav.iloc[0]['cognacy_pct']
        advantages.append({
            'domain': domain_name,
            'balinese_pct': bal.iloc[0]['cognacy_pct'],
            'javanese_pct': jav.iloc[0]['cognacy_pct'],
            'advantage': diff,
            'n_concepts': bal.iloc[0]['n_testable'],
        })
        sig = '***' if abs(diff) > 20 else '**' if abs(diff) > 10 else '*' if abs(diff) > 5 else ''
        print(f'  {domain_name:12s}: Bal={bal.iloc[0]["cognacy_pct"]:.1f}% - Jav={jav.iloc[0]["cognacy_pct"]:.1f}% = {diff:+.1f}% {sig}')

df_adv = pd.DataFrame(advantages)

# ============================================================
# 7. MARITIME-SPECIFIC DEEP DIVE
# ============================================================
print('\n--- Maritime Vocabulary Deep Dive ---')
print('  Concept-by-concept PMP cognacy:')
print(f'  {"Concept":20s} {"PMP":>5s} {"OJav":>5s} {"Bal":>5s} {"Jav":>5s} {"Mlg":>5s} {"Tgr":>5s}')
print('  ' + '-' * 65)

for concept_id in maritime_concepts:
    if concept_id not in all_param_ids:
        continue
    concept_name = params[params['ID'] == concept_id].iloc[0]['Name'] if concept_id in params['ID'].values else concept_id

    row_data = [f'  {concept_name:20s}']

    # PMP has form?
    has_pmp = concept_id in pmp_cognates
    row_data.append('Y' if has_pmp else '-')

    for lang_name in ['Old Javanese', 'Balinese', 'Javanese', 'Malagasy', 'Tengger']:
        if lang_name in lang_cognates and concept_id in lang_cognates[lang_name]:
            if has_pmp:
                pmp_sets = pmp_cognates.get(concept_id, set())
                lang_sets = lang_cognates[lang_name].get(concept_id, set())
                is_cognate = bool(pmp_sets & lang_sets)
                row_data.append('  PMP' if is_cognate else '  new')
            else:
                row_data.append('  ?  ')
        else:
            row_data.append('  -  ')

    print(''.join(f'{x:>6s}' for x in row_data))

# ============================================================
# 8. STATISTICAL TEST: Maritime vs General cognacy
# ============================================================
print('\n--- Statistical Tests ---')

for lang_name in ['Balinese', 'Javanese', 'Malagasy', 'Tengger']:
    maritime_rate = df_results[(df_results['language'] == lang_name) & (df_results['domain'] == 'Maritime')]
    general_rate = df_results[(df_results['language'] == lang_name) & (df_results['domain'] == 'ALL')]
    if not maritime_rate.empty and not general_rate.empty:
        m_pct = maritime_rate.iloc[0]['cognacy_pct']
        g_pct = general_rate.iloc[0]['cognacy_pct']
        diff = m_pct - g_pct
        print(f'  {lang_name:12s}: Maritime={m_pct:.1f}% vs General={g_pct:.1f}% → {diff:+.1f}%')

# ============================================================
# 9. VISUALIZATION
# ============================================================
print('\n--- Generating figures ---')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('E049: Semantic Domain Cognacy — Peripheral Conservatism Pattern',
             fontsize=13, fontweight='bold')

# Plot 1: Domain × Language heatmap
ax = axes[0]
domain_order = ['Maritime', 'Nature', 'Body', 'Action', 'Kinship', 'Numeral']
lang_order = ['Old Javanese', 'Balinese', 'Javanese', 'Tengger', 'Malagasy']
lang_order = [l for l in lang_order if l in pivot.columns]

plot_data = pivot.reindex(domain_order)[lang_order]
im = ax.imshow(plot_data.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(lang_order)))
ax.set_yticks(range(len(domain_order)))
ax.set_xticklabels(lang_order, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(domain_order, fontsize=9)

for i in range(len(domain_order)):
    for j in range(len(lang_order)):
        val = plot_data.values[i, j]
        if not np.isnan(val):
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=9)

plt.colorbar(im, ax=ax, label='PMP Cognacy %')
ax.set_title('PMP Cognacy by Domain × Language')

# Plot 2: Peripheral advantage by domain
ax = axes[1]
if len(df_adv) > 0:
    domain_names = df_adv['domain'].tolist()
    advantages = df_adv['advantage'].tolist()
    colors = ['darkblue' if a > 0 else 'red' for a in advantages]
    bars = ax.barh(range(len(domain_names)), advantages, color=colors, alpha=0.7)
    ax.set_yticks(range(len(domain_names)))
    ax.set_yticklabels(domain_names, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Peripheral Advantage (Balinese − Javanese, %)')
    ax.set_title('Peripheral Conservation by Domain')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (v, n) in enumerate(zip(advantages, df_adv['n_concepts'].tolist())):
        ax.text(v + (1 if v >= 0 else -1), i, f'{v:+.1f}% (n={n})',
                va='center', ha='left' if v >= 0 else 'right', fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS, 'domain_cognacy.png'), dpi=300, bbox_inches='tight')
print(f'  Saved: domain_cognacy.png')

# Save results
df_results.to_csv(os.path.join(RESULTS, 'domain_cognacy_rates.csv'), index=False)
pivot.to_csv(os.path.join(RESULTS, 'domain_cognacy_pivot.csv'))
print(f'  Saved: domain_cognacy_rates.csv, domain_cognacy_pivot.csv')

# ============================================================
# 10. SYNTHESIS
# ============================================================
print('\n' + '=' * 60)
print('SYNTHESIS')
print('=' * 60)

# Find which domains show strongest peripheral advantage
if len(df_adv) > 0:
    df_adv_sorted = df_adv.sort_values('advantage', ascending=False)
    print('\n  Domains ranked by peripheral advantage (Balinese - Javanese):')
    for _, row in df_adv_sorted.iterrows():
        print(f'    {row["domain"]:12s}: {row["advantage"]:+.1f}% (n={row["n_concepts"]})')

    strongest = df_adv_sorted.iloc[0]
    print(f'\n  Strongest peripheral advantage: {strongest["domain"]} ({strongest["advantage"]:+.1f}%)')

    # Is maritime in top 3?
    maritime_rank = list(df_adv_sorted['domain']).index('Maritime') + 1 if 'Maritime' in df_adv_sorted['domain'].values else None
    if maritime_rank:
        print(f'  Maritime domain rank: #{maritime_rank} of {len(df_adv_sorted)}')
        if maritime_rank <= 3:
            print(f'  → Maritime vocabulary IS among the most conserved domains in peripheries')
        else:
            print(f'  → Maritime vocabulary is NOT preferentially conserved')

print('\nDone!')
