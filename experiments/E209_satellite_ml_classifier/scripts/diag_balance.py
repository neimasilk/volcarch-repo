import csv
from collections import Counter, defaultdict

rows = list(csv.DictReader(open('experiments/E209_satellite_ml_classifier/data/features_s2.csv', encoding='utf-8')))
print(f'Total: {len(rows)}')
lbl = Counter(r['label'] for r in rows)
sea = Counter(r['season'] for r in rows)
print(f'Labels: {dict(lbl)}')
print(f'Seasons: {dict(sea)}')

s = defaultdict(set)
for r in rows:
    s[r['site_id']].add(r['season'])
both = [sid for sid, seasons in s.items() if 'dry' in seasons and 'wet' in seasons]
print(f'Sites with both dry+wet: {len(both)}')
lbl_both = Counter()
for r in rows:
    if r['site_id'] in both:
        lbl_both[r['label']] += 1
print(f'  Labels (sites counted per row, so ×2): {dict(lbl_both)}')
print(f'  Sites (deduped): {len(both)}')
for lab in sorted(lbl_both.keys()):
    cnt = sum(1 for sid in both if any(r['site_id']==sid and r['label']==lab for r in rows))
    print(f'    {lab}: {cnt}')
