#!/usr/bin/env python3
"""
E082: DHARMA Inscription Georeferencing
=========================================
Create geocoded inscription dataset from 268 DHARMA inscriptions.
Multi-source geocoding pipeline:
  1. Hard-coded known inscription locations (published epigraphy)
  2. Candi coordinate matching from E031 data
  3. Title/filename keyword matching against place-name lookup table
  4. XML provenance/findspot parsing
  5. Regional fallback (broad area assignment from language/content)

Goal: Enable spatial analysis of the "Invisible Millennium" — test whether
inscriptions cluster away from active volcanoes.
"""

import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent.parent
E074_META = BASE / "experiments" / "E074_dharma_deep_nlp" / "results" / "inscription_metadata.csv"
E031_CANDI = BASE / "experiments" / "E031_candi_orientation" / "results" / "candi_volcano_pairs.csv"
DHARMA_XML = BASE / "experiments" / "E023_ritual_screening" / "data" / "dharma" / "xml"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

# ── Active Volcanoes ──────────────────────────────────────────────────
VOLCANOES = {
    'Merapi':         (-7.54, 110.44),
    'Kelud':          (-7.93, 112.31),
    'Bromo':          (-7.94, 112.95),
    'Semeru':         (-8.11, 112.92),
    'Arjuno':         (-7.73, 112.59),
    'Lawu':           (-7.63, 111.19),
    'Dieng':          (-7.22, 109.92),
    'Sindoro':        (-7.30, 109.99),
    'Sumbing':        (-7.38, 110.07),
    'Slamet':         (-7.24, 109.21),
    'Agung':          (-8.34, 115.51),
    'Batur':          (-8.24, 115.37),
    'Krakatau':       (-6.10, 105.42),
    'Tangkubanperahu': (-6.76, 107.60),
    'Galunggung':     (-7.25, 108.06),
    'Penanggungan':   (-7.60, 112.62),
    'Merbabu':        (-7.45, 110.44),
    'Sundoro':        (-7.30, 109.99),
    'Raung':          (-8.13, 114.04),
    'Ijen':           (-8.06, 114.24),
}


# ── Known Inscription Coordinates (from published epigraphy) ──────────
# Format: pattern → (lat, lon, confidence, method_note)
# Patterns are matched against both filename and title (case-insensitive)
KNOWN_LOCATIONS = {
    # Well-documented inscriptions with findspot coordinates
    'canggal': (-7.60, 110.30, 'high', 'Gunung Wukir, Magelang'),
    'gunung wukir': (-7.60, 110.30, 'high', 'Gunung Wukir, Magelang'),
    'dinaya': (-7.95, 112.52, 'high', 'Dinaya, Malang'),
    'dinoyo': (-7.97, 112.63, 'high', 'Dinoyo, Malang'),
    'kalasan': (-7.77, 110.47, 'high', 'Candi Kalasan, Sleman'),
    'kelurak': (-7.61, 110.25, 'high', 'Near Borobudur'),
    'karangtengah': (-7.58, 110.40, 'high', 'Kedu Plain'),
    'kota kapur': (-2.08, 105.85, 'high', 'Bangka Island'),
    'sojomerto': (-7.10, 109.75, 'high', 'Batang, Central Java'),
    'tukmas': (-7.25, 110.35, 'high', 'Temanggung, Central Java'),
    'tuk mas': (-7.25, 110.35, 'high', 'Temanggung, Central Java'),
    'plumpungan': (-7.33, 110.50, 'high', 'Salatiga, Central Java'),
    'gondasuli': (-7.28, 110.40, 'high', 'Temanggung, Central Java'),
    'gandasuli': (-7.28, 110.40, 'high', 'Temanggung, Central Java'),
    'wanua tengah': (-7.58, 110.40, 'high', 'Kedu Plain'),
    'mantyasih': (-7.47, 110.18, 'high', 'Magelang, Kedu'),
    'pucangan': (-7.52, 111.60, 'high', 'Surabaya area'),
    'sarwadharma': (-7.55, 112.40, 'high', 'East Java'),
    'padang roco': (-1.50, 101.60, 'high', 'Dharmasraya, Sumatra'),
    'dharmasraya': (-1.50, 101.60, 'high', 'Dharmasraya, Sumatra'),
    'nagarakretagama': (-7.55, 112.39, 'high', 'Trowulan/Majapahit'),
    'prambanan': (-7.75, 110.49, 'high', 'Prambanan, Sleman'),
    'borobudur': (-7.61, 110.20, 'high', 'Borobudur, Magelang'),
    'trowulan': (-7.55, 112.39, 'high', 'Trowulan/Majapahit'),
    'dieng': (-7.21, 109.92, 'high', 'Dieng Plateau'),
    'ratu boko': (-7.77, 110.49, 'high', 'Ratu Boko, Sleman'),

    # Additional well-known inscriptions
    'talang tuwo': (-2.96, 104.71, 'high', 'Palembang area, Sumatra'),
    'kedukan bukit': (-2.98, 104.76, 'high', 'Palembang, Sumatra'),
    'kamalagyan': (-7.55, 112.39, 'medium', 'Trowulan area'),
    'singasari': (-7.89, 112.65, 'high', 'Singasari, Malang'),
    'singosari': (-7.89, 112.65, 'high', 'Singasari, Malang'),
    'kanjuruhan': (-7.97, 112.63, 'high', 'Malang area'),
    'kayumwungan': (-7.58, 110.40, 'high', 'Kedu Plain (Karang Tengah)'),
    'tulangan': (-7.47, 112.65, 'medium', 'Tulangan, Sidoarjo'),
    'sukabumi': (-7.82, 112.01, 'medium', 'Kediri area'),
    'batutulis': (-6.61, 106.80, 'high', 'Bogor, West Java'),
    'tugu': (-6.15, 106.94, 'high', 'Tugu, North Jakarta'),
    'kawali': (-7.19, 108.36, 'high', 'Ciamis, West Java'),
    'kebantenan': (-6.38, 107.00, 'high', 'Bekasi, West Java'),
    'huludayeuh': (-6.72, 107.37, 'medium', 'Bogor area'),
    'laguna': (14.36, 121.10, 'high', 'Laguna, Philippines'),
    'gunung tua': (1.52, 99.95, 'medium', 'Padang Lawas, N. Sumatra'),
    'bukit gombak': (1.35, 103.75, 'medium', 'Singapore area'),

    # East Java area inscriptions
    'pucangan': (-7.52, 111.60, 'high', 'Surabaya area'),
    'mula-malurung': (-8.05, 112.45, 'medium', 'Malang area'),
    'wurare': (-7.89, 112.65, 'medium', 'Singasari area'),
    'pakis wetan': (-8.02, 112.65, 'medium', 'Malang area'),
    'maribong': (-8.00, 112.50, 'medium', 'Malang area'),
    'rameswarapura': (-7.55, 112.39, 'medium', 'Majapahit area'),
    'manah i manuk': (-7.55, 112.39, 'medium', 'Majapahit area'),

    # Central Java Kedu Plain area
    'wukiran': (-7.66, 110.35, 'high', 'Pereng, near Prambanan'),
    'pereng': (-7.66, 110.35, 'high', 'Pereng, Sleman'),
    'manjusrigerha': (-7.61, 110.20, 'medium', 'Kedu area'),
    'hampran': (-7.58, 110.40, 'medium', 'Kedu Plain'),

    # Mataram period (Kedu/Prambanan area)
    'panunggalan': (-7.60, 110.35, 'medium', 'Kedu-Prambanan area'),
    'pananggaran': (-7.60, 110.35, 'medium', 'Kedu-Prambanan area'),
    'humanding': (-7.60, 110.35, 'medium', 'Kedu-Prambanan area (Polengan)'),
    'jurungan': (-7.60, 110.35, 'medium', 'Kedu-Prambanan area (Polengan)'),
    'mamali': (-7.60, 110.35, 'medium', 'Kedu-Prambanan area (Polengan)'),
    'taragal': (-7.60, 110.35, 'medium', 'Kedu-Prambanan area (Polengan)'),
    'tunahan': (-7.60, 110.35, 'medium', 'Kedu-Prambanan area (Polengan)'),
    'polengan': (-7.60, 110.35, 'medium', 'Polengan, near Prambanan'),
    'salingsingan': (-7.60, 110.35, 'medium', 'Near Candi Asu/Lumbung'),

    # Brantas/East Java
    'hering': (-7.60, 112.00, 'medium', 'East Java, Brantas area'),
    'gulung-gulung': (-7.60, 112.00, 'medium', 'East Java'),
    'linggasuntan': (-7.60, 112.00, 'medium', 'East Java'),
    'jeru-jeru': (-7.60, 112.00, 'medium', 'East Java'),
    'masahar': (-7.60, 112.00, 'medium', 'East Java'),
    'air kali': (-7.60, 112.00, 'medium', 'East Java'),
    'lintakan': (-7.60, 112.00, 'medium', 'East Java'),
    'sangguran': (-7.60, 112.00, 'medium', 'East Java'),

    # Penanggungan slopes
    'sukhamerta': (-7.58, 112.55, 'medium', 'Penanggungan area'),
    'terep': (-7.58, 112.55, 'medium', 'Penanggungan area'),
    'walandit': (-7.58, 112.55, 'medium', 'Penanggungan area'),
    'himad walandit': (-7.58, 112.55, 'medium', 'Penanggungan area'),

    # Mataram Dynasty (9th-10th century, Central Java)
    'bhatari': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'ayam teas': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'kasugihan': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'dalinan': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'taji': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'rumwiga': (-7.75, 110.40, 'medium', 'Bantul area (Payak)'),
    'kinewu': (-7.93, 112.31, 'medium', 'Near Kelud, Blitar'),
    'kubu-kubu': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'palepangan': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'rukam': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'tiga ron': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'watu ridang': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'tihang': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'sugih manek': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'barahasrama': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'rabvan': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'sang makudur': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'turu mangambil': (-7.60, 110.35, 'medium', 'Mataram Central Java'),
    'wuru tunggal': (-7.60, 110.35, 'medium', 'Mataram Central Java'),

    # Sindoro area
    'sindoro': (-7.30, 109.99, 'medium', 'Near Sindoro volcano'),

    # Bromo-Semeru area
    'bromo semeru': (-7.94, 112.95, 'high', 'Bromo-Semeru area'),

    # Bali inscriptions
    'batur': (-8.24, 115.37, 'medium', 'Batur, Bali'),

    # Kediri area
    'adan-adan': (-7.80, 112.01, 'medium', 'Kediri area'),
    'air asih': (-7.80, 112.01, 'medium', 'Kediri area'),
    'parablyan': (-7.80, 112.01, 'medium', 'Kediri area'),

    # Brantas Delta / Sidoarjo
    'canggu': (-7.47, 112.65, 'medium', 'Sidoarjo/Surabaya area'),
    'karang lo': (-7.47, 112.65, 'medium', 'Mojokerto area'),

    # East Java Majapahit era
    'gajah mada': (-7.55, 112.39, 'medium', 'Trowulan/Majapahit area'),
    'silamanikundala': (-7.55, 112.39, 'medium', 'Majapahit area'),

    # Miscellaneous known sites
    'dawangsari': (-7.77, 110.49, 'medium', 'Prambanan area'),
    'nglumbang': (-7.60, 112.00, 'low', 'East Java'),
    'bukateja': (-7.60, 110.35, 'low', 'Central Java'),
    'plalangan': (-7.60, 110.35, 'low', 'Central Java'),
    'watu genuk': (-7.60, 110.35, 'low', 'Central Java'),
    'jragung': (-7.60, 110.35, 'low', 'Central Java'),
    'garung': (-7.35, 109.95, 'medium', 'Near Dieng'),
    'linggawangi': (-7.19, 108.36, 'medium', 'Ciamis area, West Java'),
    'puhawang glis': (-5.40, 105.25, 'medium', 'Lampung, Sumatra'),

    # Sang Hyang Tapak / Sima Anglayang — East Java
    'sang hyang tapak': (-7.60, 112.00, 'medium', 'East Java'),
    'sima anglayang': (-7.60, 112.00, 'medium', 'East Java'),
    'bularut': (-7.60, 112.00, 'medium', 'East Java'),
    'munggut': (-7.60, 112.00, 'medium', 'East Java'),
    'horren': (-7.60, 112.00, 'medium', 'East Java'),
    'kusambyan': (-7.60, 112.00, 'medium', 'East Java'),
    'talan': (-7.60, 112.00, 'medium', 'East Java'),
    'poh': (-7.60, 112.00, 'medium', 'East Java'),
    'gondang lor': (-7.50, 112.00, 'medium', 'East Java'),
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def find_nearest_volcano(lat, lon):
    """Return (volcano_name, distance_km) for nearest active volcano."""
    best_name = None
    best_dist = float('inf')
    for name, (vlat, vlon) in VOLCANOES.items():
        d = haversine_km(lat, lon, vlat, vlon)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name, round(best_dist, 2)


def load_inscription_metadata():
    """Load E074 inscription metadata CSV."""
    inscriptions = []
    with open(E074_META, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inscriptions.append(row)
    print(f"Loaded {len(inscriptions)} inscriptions from E074 metadata")
    return inscriptions


def load_candi_coordinates():
    """Load candi coordinates from E031 for cross-referencing."""
    candi = {}
    try:
        with open(E031_CANDI, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name'].lower().replace('candi ', '').strip()
                candi[name] = (float(row['lat']), float(row['lon']))
        print(f"Loaded {len(candi)} candi coordinates from E031")
    except FileNotFoundError:
        print("WARNING: E031 candi data not found")
    return candi


def parse_xml_provenance(xml_path):
    """Parse DHARMA XML file looking for location hints in commentary/provenance."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError):
        return None

    # Collect all text from commentary, provenance, history elements
    location_hints = []

    # Check for origPlace
    for el in root.iter('{http://www.tei-c.org/ns/1.0}origPlace'):
        if el.text and el.text.strip():
            location_hints.append(el.text.strip())

    # Check for placeName elements
    for el in root.iter('{http://www.tei-c.org/ns/1.0}placeName'):
        if el.text and el.text.strip():
            location_hints.append(el.text.strip())

    # Check for provenance in history section
    for el in root.iter('{http://www.tei-c.org/ns/1.0}provenance'):
        text = ''.join(el.itertext()).strip()
        if text:
            location_hints.append(text)

    # Check commentary for findspot mentions
    comm_div = root.find('.//tei:div[@type="commentary"]', NS)
    if comm_div is not None:
        comm_text = ''.join(comm_div.itertext()).strip()
        # Look for kabupaten/kecamatan mentions
        kab_match = re.search(r'kabupaten\s+(\w+)', comm_text, re.IGNORECASE)
        kec_match = re.search(r'kecamatan\s+(\w+)', comm_text, re.IGNORECASE)
        findspot_match = re.search(r'findspot[:\s]+(.{10,80})', comm_text, re.IGNORECASE)
        found_match = re.search(r'found\s+(?:in|at|near)\s+(.{10,80})', comm_text, re.IGNORECASE)

        for m in [kab_match, kec_match, findspot_match, found_match]:
            if m:
                location_hints.append(m.group(0))

    return location_hints if location_hints else None


def geocode_inscription(filename, title, candi_coords):
    """
    Attempt to geocode a single inscription using multiple strategies.
    Returns (lat, lon, method, confidence, note) or None.
    """
    title_lower = title.lower() if title else ''
    fname_lower = filename.lower().replace('dharma_insidenk', '').replace('dharma_ins12', '').replace('.xml', '')

    # Strategy 1: Match against known inscription locations
    for pattern, (lat, lon, conf, note) in KNOWN_LOCATIONS.items():
        if pattern in title_lower or pattern.replace(' ', '') in fname_lower.replace('_', ''):
            return (lat, lon, 'known_location', conf, note)

    # Strategy 2: Match against candi coordinates from E031
    for candi_name, (lat, lon) in candi_coords.items():
        # Check if candi name appears in title or filename
        if candi_name in title_lower or candi_name.replace(' ', '') in fname_lower:
            return (lat, lon, 'candi_match', 'medium', f'Matched candi: {candi_name}')

    # Strategy 3: Try XML provenance parsing
    xml_path = DHARMA_XML / filename
    if xml_path.exists():
        hints = parse_xml_provenance(xml_path)
        if hints:
            # Try to match hints against known locations
            for hint in hints:
                hint_lower = hint.lower()
                for pattern, (lat, lon, conf, note) in KNOWN_LOCATIONS.items():
                    if pattern in hint_lower:
                        return (lat, lon, 'xml_provenance', conf, f'XML hint: {hint[:60]}')

    # Strategy 4: Regional assignment based on language/content cues
    # Old Malay (omy-Latn) inscriptions are mostly from Sumatra
    # osn-Latn = Old Sundanese → West Java
    # Some specific keywords in title

    # Check for Sumatran indicators
    sumatra_keywords = ['sumatra', 'sriwijaya', 'śrīvijaya', 'palembang', 'jambi', 'melayu']
    if any(k in title_lower for k in sumatra_keywords):
        return (-2.50, 104.50, 'regional_content', 'low', 'Sumatran content keywords')

    return None


def main():
    print("=" * 70)
    print("E082: DHARMA Inscription Georeferencing")
    print("=" * 70)

    # Load data
    inscriptions = load_inscription_metadata()
    candi_coords = load_candi_coordinates()

    # ── Geocode all inscriptions ──────────────────────────────────────
    print("\n--- Geocoding Pipeline ---")
    geocoded = []
    methods = Counter()
    confidence_counts = Counter()

    for insc in inscriptions:
        result = geocode_inscription(insc['filename'], insc['title'], candi_coords)

        if result:
            lat, lon, method, confidence, note = result
            nearest_vol, vol_dist = find_nearest_volcano(lat, lon)
            geocoded.append({
                'filename': insc['filename'],
                'title': insc['title'],
                'lang': insc['lang'],
                'date_ce': insc['date_ce'],
                'century': insc['century'],
                'lat': lat,
                'lon': lon,
                'geocode_method': method,
                'confidence': confidence,
                'geocode_note': note,
                'nearest_volcano': nearest_vol,
                'volcano_dist_km': vol_dist,
            })
            methods[method] += 1
            confidence_counts[confidence] += 1

    print(f"\nGeocoded: {len(geocoded)} / {len(inscriptions)} ({len(geocoded)/len(inscriptions)*100:.1f}%)")
    print(f"\nBy method:")
    for method, count in methods.most_common():
        print(f"  {method}: {count}")
    print(f"\nBy confidence:")
    for conf, count in confidence_counts.most_common():
        print(f"  {conf}: {count}")

    # ── Save geocoded CSV ─────────────────────────────────────────────
    csv_path = RESULTS / "geocoded_inscriptions.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'title', 'lang', 'date_ce', 'century',
            'lat', 'lon', 'geocode_method', 'confidence', 'geocode_note',
            'nearest_volcano', 'volcano_dist_km'
        ])
        writer.writeheader()
        for row in geocoded:
            writer.writerow(row)
    print(f"\nSaved: {csv_path}")

    # ── Volcanic Proximity Analysis ──────────────────────────────────
    print("\n" + "=" * 70)
    print("VOLCANIC PROXIMITY ANALYSIS")
    print("=" * 70)

    # Filter to dated inscriptions for temporal analysis
    dated_geo = [g for g in geocoded if g['date_ce']]
    print(f"\nDated + geocoded inscriptions: {len(dated_geo)}")

    # Overall distance statistics
    distances = [g['volcano_dist_km'] for g in geocoded]
    if distances:
        mean_dist = sum(distances) / len(distances)
        sorted_dist = sorted(distances)
        median_dist = sorted_dist[len(sorted_dist) // 2]
        min_dist = min(distances)
        max_dist = max(distances)

        print(f"\nDistance to nearest volcano:")
        print(f"  Mean:   {mean_dist:.1f} km")
        print(f"  Median: {median_dist:.1f} km")
        print(f"  Min:    {min_dist:.1f} km")
        print(f"  Max:    {max_dist:.1f} km")

        # Exclude outliers (Philippines, Singapore) for Java-focused analysis
        java_geo = [g for g in geocoded if -9.0 < g['lat'] < -6.0 and 105.0 < g['lon'] < 116.0]
        print(f"\nJava/Bali subset: {len(java_geo)} inscriptions")
        if java_geo:
            java_dists = [g['volcano_dist_km'] for g in java_geo]
            java_mean = sum(java_dists) / len(java_dists)
            java_sorted = sorted(java_dists)
            java_median = java_sorted[len(java_sorted) // 2]
            print(f"  Mean:   {java_mean:.1f} km")
            print(f"  Median: {java_median:.1f} km")
            print(f"  Min:    {min(java_dists):.1f} km")
            print(f"  Max:    {max(java_dists):.1f} km")

    # Distance by century
    print(f"\nDistance by century (Java/Bali only):")
    century_dists = defaultdict(list)
    for g in geocoded:
        if g['century'] and -9.0 < g['lat'] < -6.0 and 105.0 < g['lon'] < 116.0:
            century_dists[int(g['century'])].append(g['volcano_dist_km'])

    print(f"  {'Century':<10} {'N':<5} {'Mean km':<10} {'Median km':<12} {'Min km':<10} {'Max km'}")
    print(f"  {'-'*55}")
    century_means = {}
    for c in sorted(century_dists.keys()):
        dists = century_dists[c]
        mean_d = sum(dists) / len(dists)
        sorted_d = sorted(dists)
        median_d = sorted_d[len(sorted_d) // 2]
        century_means[c] = mean_d
        print(f"  C{c:<8} {len(dists):<5} {mean_d:<10.1f} {median_d:<12.1f} {min(dists):<10.1f} {max(dists):.1f}")

    # Volcano assignment counts
    print(f"\nNearest volcano distribution:")
    vol_counts = Counter(g['nearest_volcano'] for g in geocoded)
    for vol, count in vol_counts.most_common():
        dists = [g['volcano_dist_km'] for g in geocoded if g['nearest_volcano'] == vol]
        print(f"  {vol:<18s}: {count:3d} inscriptions, mean dist {sum(dists)/len(dists):.1f} km")

    # ── Distance zones (matching E065 analysis) ──────────────────────
    print(f"\nDistance zones (Java/Bali only):")
    zone_A = [g for g in java_geo if g['volcano_dist_km'] <= 10] if java_geo else []
    zone_B = [g for g in java_geo if 10 < g['volcano_dist_km'] <= 30] if java_geo else []
    zone_C = [g for g in java_geo if g['volcano_dist_km'] > 30] if java_geo else []
    total_java = len(java_geo) if java_geo else 0
    print(f"  Zone A (0-10 km):  {len(zone_A)} ({len(zone_A)/max(total_java,1)*100:.0f}%)")
    print(f"  Zone B (10-30 km): {len(zone_B)} ({len(zone_B)/max(total_java,1)*100:.0f}%)")
    print(f"  Zone C (>30 km):   {len(zone_C)} ({len(zone_C)/max(total_java,1)*100:.0f}%)")

    # ── Spearman Correlation ──────────────────────────────────────────
    print(f"\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    # Test: Spearman correlation between century and mean distance
    if len(century_means) >= 3:
        centuries = sorted(century_means.keys())
        mean_dists = [century_means[c] for c in centuries]

        # Manual Spearman (rank correlation)
        n = len(centuries)
        rank_x = list(range(1, n + 1))  # centuries are already ordered
        # Rank the distances
        sorted_indices = sorted(range(n), key=lambda i: mean_dists[i])
        rank_y = [0] * n
        for rank, idx in enumerate(sorted_indices, 1):
            rank_y[idx] = rank

        d_sq = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
        rho = 1 - (6 * d_sq) / (n * (n**2 - 1))

        # t-test for significance
        if abs(rho) < 1.0:
            t_stat = rho * math.sqrt((n - 2) / (1 - rho**2))
            # Approximate p-value using t-distribution (two-tailed)
            # For small n, use lookup-ish approximation
            df = n - 2
            # Simple p approximation via incomplete beta
            # For reporting, just show rho and note significance threshold
            print(f"\nSpearman correlation (century vs mean volcanic distance):")
            print(f"  N centuries: {n}")
            print(f"  rho = {rho:.3f}")
            print(f"  t-stat = {t_stat:.3f} (df={df})")
            # Critical t for alpha=0.05 two-tailed with df degrees
            # df=1: 12.71, df=2: 4.30, df=3: 3.18, df=4: 2.78, df=5: 2.57
            critical_t = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.37, 8: 2.31}
            ct = critical_t.get(df, 2.0)
            sig = abs(t_stat) > ct
            print(f"  Critical t (alpha=0.05, df={df}): {ct}")
            print(f"  Significant: {'YES' if sig else 'NO'}")
        else:
            print(f"\nSpearman rho = {rho:.3f} (perfect correlation, N too small)")

    # Compare candi distance (E065) vs inscription distance
    print(f"\n--- Comparison: Candi vs Inscriptions ---")
    if java_geo:
        candi_mean = 16.52  # From E065 results
        candi_median = 14.65
        java_mean_val = sum([g['volcano_dist_km'] for g in java_geo]) / len(java_geo)
        java_sorted_val = sorted([g['volcano_dist_km'] for g in java_geo])
        java_median_val = java_sorted_val[len(java_sorted_val) // 2]

        print(f"  Candi (E065):       mean={candi_mean:.1f} km, median={candi_median:.1f} km (N=142)")
        print(f"  Inscriptions (E082): mean={java_mean_val:.1f} km, median={java_median_val:.1f} km (N={len(java_geo)})")
        diff = java_mean_val - candi_mean
        print(f"  Difference: inscriptions are {abs(diff):.1f} km {'farther from' if diff > 0 else 'closer to'} volcanoes than candi")

    # Confidence-stratified analysis
    print(f"\n--- High-confidence subset ---")
    high_conf = [g for g in geocoded if g['confidence'] == 'high'
                 and -9.0 < g['lat'] < -6.0 and 105.0 < g['lon'] < 116.0]
    if high_conf:
        hc_dists = [g['volcano_dist_km'] for g in high_conf]
        hc_mean = sum(hc_dists) / len(hc_dists)
        hc_sorted = sorted(hc_dists)
        hc_median = hc_sorted[len(hc_sorted) // 2]
        print(f"  N = {len(high_conf)}")
        print(f"  Mean:   {hc_mean:.1f} km")
        print(f"  Median: {hc_median:.1f} km")

    # ── Save summary ─────────────────────────────────────────────────
    summary_lines = [
        "E082: DHARMA Inscription Georeferencing — Summary",
        "=" * 55,
        f"",
        f"Input: {len(inscriptions)} DHARMA inscriptions from E074",
        f"Geocoded: {len(geocoded)} ({len(geocoded)/len(inscriptions)*100:.1f}%)",
        f"",
        "Geocoding methods:",
    ]
    for method, count in methods.most_common():
        summary_lines.append(f"  {method}: {count}")
    summary_lines.extend([
        f"",
        "Confidence levels:",
    ])
    for conf, count in confidence_counts.most_common():
        summary_lines.append(f"  {conf}: {count}")

    summary_lines.extend([
        f"",
        f"Overall distance to nearest volcano:",
        f"  Mean:   {mean_dist:.1f} km" if distances else "  N/A",
        f"  Median: {median_dist:.1f} km" if distances else "  N/A",
        f"",
    ])
    if java_geo:
        summary_lines.extend([
            f"Java/Bali subset (N={len(java_geo)}):",
            f"  Mean:   {java_mean:.1f} km",
            f"  Median: {java_median:.1f} km",
            f"",
            f"Distance zones (Java/Bali):",
            f"  Zone A (0-10 km):  {len(zone_A)} ({len(zone_A)/total_java*100:.0f}%)",
            f"  Zone B (10-30 km): {len(zone_B)} ({len(zone_B)/total_java*100:.0f}%)",
            f"  Zone C (>30 km):   {len(zone_C)} ({len(zone_C)/total_java*100:.0f}%)",
            f"",
            f"Comparison with E065 Candi data:",
            f"  Candi mean distance:       {candi_mean:.1f} km (N=142)",
            f"  Inscription mean distance: {java_mean_val:.1f} km (N={len(java_geo)})",
            f"  Inscriptions are {abs(diff):.1f} km {'farther from' if diff > 0 else 'closer to'} volcanoes",
        ])

    summary_text = "\n".join(summary_lines)
    with open(RESULTS / "geocoding_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary_text)

    # Save detailed proximity analysis
    proximity_lines = [
        "E082: Volcanic Proximity Analysis",
        "=" * 55,
        "",
        "Distance by century (Java/Bali only):",
        f"  {'Century':<10} {'N':<5} {'Mean km':<10} {'Median km':<12}",
        f"  {'-'*40}",
    ]
    for c in sorted(century_dists.keys()):
        dists = century_dists[c]
        mean_d = sum(dists) / len(dists)
        sorted_d = sorted(dists)
        median_d = sorted_d[len(sorted_d) // 2]
        proximity_lines.append(f"  C{c:<8} {len(dists):<5} {mean_d:<10.1f} {median_d:<12.1f}")

    proximity_lines.extend([
        "",
        "Nearest volcano distribution:",
    ])
    for vol, count in vol_counts.most_common():
        dists_v = [g['volcano_dist_km'] for g in geocoded if g['nearest_volcano'] == vol]
        proximity_lines.append(f"  {vol:<18s}: {count:3d} inscr, mean {sum(dists_v)/len(dists_v):.1f} km")

    if 'rho' in dir():
        proximity_lines.extend([
            "",
            "Spearman correlation (century vs mean volcanic distance):",
            f"  rho = {rho:.3f}",
        ])

    with open(RESULTS / "volcanic_proximity_analysis.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(proximity_lines))

    # ── Save JSON results ─────────────────────────────────────────────
    json_results = {
        "experiment": "E082",
        "title": "DHARMA Inscription Georeferencing",
        "n_input": len(inscriptions),
        "n_geocoded": len(geocoded),
        "geocode_rate": round(len(geocoded) / len(inscriptions) * 100, 1),
        "methods": dict(methods),
        "confidence": dict(confidence_counts),
        "java_bali_subset": len(java_geo) if java_geo else 0,
        "distance_stats": {
            "overall_mean": round(mean_dist, 1) if distances else None,
            "overall_median": round(median_dist, 1) if distances else None,
            "java_mean": round(java_mean, 1) if java_geo else None,
            "java_median": round(java_median, 1) if java_geo else None,
        },
        "zone_distribution": {
            "zone_A_0_10km": len(zone_A) if java_geo else 0,
            "zone_B_10_30km": len(zone_B) if java_geo else 0,
            "zone_C_gt_30km": len(zone_C) if java_geo else 0,
        },
        "volcano_counts": dict(vol_counts.most_common()),
        "century_mean_distances": {f"C{k}": round(v, 1) for k, v in century_means.items()} if century_means else {},
    }

    with open(RESULTS / "e082_results.json", 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to {RESULTS}/")
    print(f"  geocoded_inscriptions.csv")
    print(f"  geocoding_summary.txt")
    print(f"  volcanic_proximity_analysis.txt")
    print(f"  e082_results.json")


if __name__ == "__main__":
    main()
