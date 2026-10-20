# Compare every DOI-bearing entry of a .bib against Crossref (authors, year, volume, title).
# Usage: python tools/check_bib_crossref.py refs.bib "key1,key2,..." out.json   (keys optional: cited keys only)
# Written 2026-10-20 for the P2 round-2 revision (ledger C076). Flags are leads, not verdicts: read each one.
import re, sys, json, time, urllib.request, unicodedata
bib = open(sys.argv[1], encoding='utf-8').read()
cited = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else None
entries = re.split(r'\n@', '\n' + bib)
def field(e, f):
    m = re.search(r'\b' + f + r'\s*=\s*[{"](.*?)[}"]\s*,?\s*\n', e, re.S)
    return re.sub(r'\s+', ' ', m.group(1)).strip() if m else ''
def norm(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9 ]', '', s)
out = []
for e in entries:
    m = re.match(r'(\w+)\{([^,]+),', e.strip())
    if not m: continue
    key = m.group(2).strip()
    if cited is not None and key not in cited: continue
    doi = field(e, 'doi')
    rec = {'key': key, 'doi': doi, 'bib_year': field(e, 'year'), 'bib_vol': field(e, 'volume'),
           'bib_authors': field(e, 'author'), 'bib_title': field(e, 'title')}
    if not doi:
        rec['status'] = 'NO_DOI'; out.append(rec); continue
    try:
        req = urllib.request.Request('https://api.crossref.org/works/' + doi,
                                     headers={'User-Agent': 'volcarch-bibcheck/1.0 (mailto:amien@ubhinus.ac.id)'})
        msg = json.load(urllib.request.urlopen(req, timeout=30))['message']
    except Exception as ex:
        rec['status'] = 'DOI_FAIL ' + str(ex)[:60]; out.append(rec); continue
    cr_auth = [a.get('family', a.get('name', '')) for a in msg.get('author', [])]
    issued = (msg.get('published-print') or msg.get('published-online') or msg.get('issued'))['date-parts'][0][0]
    cr_title = (msg.get('title') or [''])[0]
    bib_fams = [norm(a.split(',')[0]) for a in rec['bib_authors'].split(' and ') if a.strip()]
    cr_fams = [norm(a) for a in cr_auth]
    probs = []
    if bib_fams and cr_fams and bib_fams[0] != cr_fams[0]: probs.append(f'FIRST_AUTHOR bib={bib_fams[0]} cr={cr_fams[0]}')
    if set(bib_fams) - set(cr_fams) and cr_fams: probs.append('AUTHORS_NOT_IN_CR ' + ','.join(sorted(set(bib_fams) - set(cr_fams))))
    if len(bib_fams) != len(cr_fams) and cr_fams and not any('others' in b for b in bib_fams): probs.append(f'N_AUTHORS bib={len(bib_fams)} cr={len(cr_fams)}')
    if rec['bib_year'] and str(issued) != rec['bib_year']: probs.append(f'YEAR bib={rec["bib_year"]} cr={issued}')
    if rec['bib_vol'] and msg.get('volume') and rec['bib_vol'] != msg.get('volume'): probs.append(f'VOL bib={rec["bib_vol"]} cr={msg.get("volume")}')
    nt_b, nt_c = norm(rec['bib_title']), norm(cr_title)
    if nt_b and nt_c and nt_b[:40] != nt_c[:40]: probs.append('TITLE cr=' + cr_title[:90])
    rec['cr_authors'] = cr_auth; rec['cr_year'] = issued; rec['cr_title'] = cr_title; rec['cr_vol'] = msg.get('volume')
    rec['status'] = 'OK' if not probs else 'CHECK: ' + ' | '.join(probs)
    out.append(rec); time.sleep(0.3)
for r in out:
    print(f"{r['key']:28s} {r['status']}")
json.dump(out, open(sys.argv[3] if len(sys.argv) > 3 else 'bibcheck.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
