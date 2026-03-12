#!/usr/bin/env python3
"""
E051: Java Toponymic Substrate Analysis
========================================
Classifies Java village names into linguistic layers (PRE_HINDU, SANSKRIT,
ARABIC, MIXED, UNKNOWN) and analyzes geographic distribution by province
and kabupaten.

Data source: cahyadsn/wilayah (Kepmendagri 2025) via GitHub
"""

import sys
import io
import os
import re
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

# Windows UTF-8 fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Configuration ---
BASE_DIR = Path(__file__).parent
SQL_FILE = BASE_DIR / "data_wilayah.sql"
SQL_L12_FILE = BASE_DIR / "data_wilayah_level12.sql"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Java province codes
JAVA_PROVINCES = {
    "31": "DKI Jakarta",
    "32": "Jawa Barat",
    "33": "Jawa Tengah",
    "34": "DI Yogyakarta",
    "35": "Jawa Timur",
    "36": "Banten",
}

# =====================================================================
# MORPHEME DICTIONARIES
# =====================================================================

# Sanskrit / Indic layer indicators
SANSKRIT_MORPHEMES = {
    # Suffixes (most common in toponyms)
    "suffixes": [
        "pura", "nagara", "negara", "kerta", "karta", "graha",
        "wana", "sari", "asri", "dharma", "darma", "tirta", "tirto",
        "dewa", "dewi", "bumi", "loka", "adi", "maha",
        "raja", "rejo", "ratna", "jaya", "mukti", "praja",
        "harjo", "mulyo", "rahayu", "agung",
    ],
    # Prefixes
    "prefixes": [
        "candi", "kraton", "keraton", "prambanan",
        "sri", "maha", "adi",
    ],
    # Standalone / root morphemes in names
    "roots": [
        "kerta", "karto", "rejo", "mulyo", "harjo", "rahayu",
        "agung", "sari", "jaya", "mukti", "praja", "pura",
        "wana", "tirta", "tirto", "dewa", "dewi", "dharma",
        "darma", "ratna", "graha", "asri", "nagara", "negara",
        "loka", "bumi", "indah",
    ],
}

# Arabic / Islamic layer indicators
ARABIC_MORPHEMES = {
    "suffixes": [
        "abad", "aman", "barokah", "falah", "hidayah", "hikmah",
        "ikhlas", "iman", "islam", "karim", "makmur", "rahmat",
        "salam", "sejahtera",
    ],
    "prefixes": [
        "masjid", "mesjid", "surau", "al ",
    ],
    "roots": [
        "barokah", "falah", "hidayah", "hikmah", "ikhlas",
        "iman", "islam", "karim", "makmur", "rahmat", "salam",
        "sejahtera", "mulia", "aman",
    ],
}

# Pre-Hindu Austronesian layer indicators
# These are PMP / PAN / Old Javanese / Sundanese morphemes
PREHIDU_MORPHEMES = {
    "suffixes": [],  # -an handled specially
    "prefixes": [
        "ranu", "danu", "dano",            # lake (PMP *danaw)
        "watu", "batu",                      # stone (PMP *batu)
        "gunung", "gnung",                   # mountain
        "tegal", "tegalan",                  # dry field
        "sawah",                             # wet rice field
        "lebak",                             # lowland/swamp
        "lembah",                            # valley
        "ci", "ci-",                         # Sundanese water prefix
        "banyu", "banjar",                   # water
        "kampung", "kampong", "dukuh",       # settlement
        "rimba", "alas",                     # forest
        "pasir",                             # sand/hill (Sundanese)
        "rawa", "rowo",                      # swamp
        "sumber", "sumur",                   # spring/well
        "kali",                              # river
        "kedung",                            # deep pool
        "glagah",                            # tall grass
        "wringin", "waringin",              # banyan tree
        "jati",                              # teak
        "pring",                             # bamboo
        "wono", "wana",                      # forest (note: also Sanskrit)
        "tlogo", "tlaga", "telaga",         # lake
        "sendang",                           # spring
        "gede", "gedhe",                     # big
        "cilik", "lik",                      # small
    ],
    "roots": [
        "ranu", "danu", "dano",
        "watu", "batu",
        "gunung",
        "tegal",
        "sawah",
        "lebak",
        "lembah",
        "banyu", "banjar",
        "kampung", "dukuh",
        "rimba", "alas",
        "pasir",
        "rawa", "rowo",
        "sumber", "sumur",
        "kali",
        "kedung",
        "glagah",
        "wringin", "waringin",
        "jati",
        "pring",
        "wono",
        "tlogo", "tlaga", "telaga",
        "sendang",
        "gede", "gedhe",
        "cilik",
        "karang",           # coral/settlement
        "desa",             # village (but also Sanskrit)
        "tanjung",          # cape (PMP *tanjung)
        "natar", "nater",   # plain
        "kulon",            # west (PJv)
        "wetan",            # east (PJv)
        "lor",              # north (PJv)
        "kidul",            # south (PJv)
        "tengah",           # middle
        "girang",           # upstream (Sundanese)
        "hilir",            # downstream
        "bojong",           # river bend (Sundanese)
        "leuwi",            # deep river pool (Sundanese)
        "babakan",          # new settlement (Sundanese)
        "cibogo",           # buffalo water (Sundanese)
    ],
}

# Sundanese ci- prefix pattern (water-related toponyms)
CI_PATTERN = re.compile(r"^ci[a-z]", re.IGNORECASE)

# Ka-...-an circumfix pattern (Austronesian nominalizer)
KA_AN_PATTERN = re.compile(r"^ka.+an$", re.IGNORECASE)

# Javanese directional suffixes
JAVANESE_DIRECTIONS = ["kulon", "wetan", "lor", "kidul", "tengah"]


# =====================================================================
# DATA LOADING
# =====================================================================

def parse_sql_inserts(filepath):
    """Parse INSERT statements from the wilayah SQL file.
    Returns list of (kode, nama) tuples.
    """
    records = []
    # Pattern to match VALUES entries: ('kode','nama')
    # The SQL uses: ('kode','nama'),  or ('kode','nama');
    val_pattern = re.compile(r"\('([^']+)','([^']+)'\)")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if "INSERT INTO" in line.upper() or line.strip().startswith("("):
                matches = val_pattern.findall(line)
                for kode, nama in matches:
                    records.append((kode, nama))

    print(f"  Parsed {len(records):,} records from {filepath.name}")
    return records


def parse_level12_sql(filepath):
    """Parse level 1-2 SQL for province/kabupaten coordinates."""
    coords = {}
    # Match: ('kode','nama','ibukota', lat, lng, ...)
    val_pattern = re.compile(
        r"\('([^']+)','([^']+)','([^']*)',\s*([-\d.]+),\s*([-\d.]+)"
    )
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            matches = val_pattern.findall(line)
            for kode, nama, ibukota, lat, lng in matches:
                coords[kode] = {
                    "nama": nama,
                    "ibukota": ibukota,
                    "lat": float(lat),
                    "lng": float(lng),
                }
    print(f"  Parsed {len(coords):,} level-1/2 records with coordinates")
    return coords


def filter_java_villages(records):
    """Filter to only Java village-level records (4 segments in kode)."""
    java = []
    for kode, nama in records:
        parts = kode.split(".")
        province_code = parts[0]
        if province_code in JAVA_PROVINCES:
            # Village level has 4 segments: prov.kab.kec.desa
            if len(parts) == 4:
                java.append({
                    "kode": kode,
                    "nama": nama,
                    "province_code": province_code,
                    "province": JAVA_PROVINCES[province_code],
                    "kab_code": f"{parts[0]}.{parts[1]}",
                    "kec_code": f"{parts[0]}.{parts[1]}.{parts[2]}",
                })
    print(f"  Filtered to {len(java):,} Java village-level records")
    return java


def get_kabupaten_names(records):
    """Get kabupaten-level names from full records."""
    kab_names = {}
    for kode, nama in records:
        parts = kode.split(".")
        if len(parts) == 2:
            kab_names[kode] = nama
    return kab_names


# =====================================================================
# CLASSIFICATION ENGINE
# =====================================================================

def classify_village_name(nama):
    """
    Classify a village name into linguistic layers.

    Returns dict with:
        layer: PRE_HINDU | SANSKRIT | ARABIC | MIXED | UNKNOWN
        markers: list of matched morphemes
        details: dict of per-layer matches
    """
    name_lower = nama.lower().strip()
    # Remove common administrative prefixes
    for prefix in ["desa ", "kelurahan ", "kel. ", "ds. "]:
        if name_lower.startswith(prefix):
            name_lower = name_lower[len(prefix):]

    # Split into tokens for multi-word names
    tokens = name_lower.split()

    sanskrit_hits = []
    arabic_hits = []
    prehidu_hits = []

    # --- Check Sanskrit morphemes ---
    for morph in SANSKRIT_MORPHEMES["suffixes"]:
        if name_lower.endswith(morph) and len(name_lower) > len(morph):
            sanskrit_hits.append(f"-{morph}")
    for morph in SANSKRIT_MORPHEMES["prefixes"]:
        if name_lower.startswith(morph):
            sanskrit_hits.append(f"{morph}-")
    for morph in SANSKRIT_MORPHEMES["roots"]:
        for token in tokens:
            if token == morph or (len(token) > 3 and morph in token and morph not in [h.strip("-") for h in sanskrit_hits]):
                sanskrit_hits.append(f"[{morph}]")
                break

    # --- Check Arabic morphemes ---
    for morph in ARABIC_MORPHEMES["suffixes"]:
        if name_lower.endswith(morph) and len(name_lower) > len(morph):
            arabic_hits.append(f"-{morph}")
    for morph in ARABIC_MORPHEMES["prefixes"]:
        if name_lower.startswith(morph):
            arabic_hits.append(f"{morph}-")
    for morph in ARABIC_MORPHEMES["roots"]:
        for token in tokens:
            if token == morph:
                arabic_hits.append(f"[{morph}]")
                break

    # --- Check Pre-Hindu morphemes ---
    for morph in PREHIDU_MORPHEMES["prefixes"]:
        if name_lower.startswith(morph):
            prehidu_hits.append(f"{morph}-")
    for morph in PREHIDU_MORPHEMES["roots"]:
        for token in tokens:
            if token == morph or token.startswith(morph):
                if morph not in [h.strip("-") for h in prehidu_hits]:
                    prehidu_hits.append(f"[{morph}]")
                break

    # Sundanese ci- prefix (very strong Pre-Hindu indicator)
    if CI_PATTERN.match(name_lower) and "ci-" not in [h.strip("-") for h in prehidu_hits]:
        # Exclude false positives
        ci_exclusions = ["cirebon"]  # city name, already mixed
        if not any(name_lower.startswith(ex) for ex in ci_exclusions):
            prehidu_hits.append("ci- (Sunda)")

    # Ka-...-an circumfix
    for token in tokens:
        if KA_AN_PATTERN.match(token) and len(token) > 5:
            prehidu_hits.append(f"ka-...-an ({token})")
            break

    # Javanese directional terms
    for direction in JAVANESE_DIRECTIONS:
        if direction in tokens:
            prehidu_hits.append(f"[dir:{direction}]")

    # De-duplicate
    sanskrit_hits = list(dict.fromkeys(sanskrit_hits))
    arabic_hits = list(dict.fromkeys(arabic_hits))
    prehidu_hits = list(dict.fromkeys(prehidu_hits))

    # Resolve "wana/wono" ambiguity: if BOTH Sanskrit and Pre-Hindu claim it,
    # give to Sanskrit (as it's used as ornamental suffix in court names)
    # unless other Pre-Hindu markers are present
    wana_conflict = any("wana" in h or "wono" in h for h in sanskrit_hits) and \
                    any("wana" in h or "wono" in h for h in prehidu_hits)
    if wana_conflict:
        if len(prehidu_hits) > 1:
            # Other Pre-Hindu markers → keep in Pre-Hindu
            sanskrit_hits = [h for h in sanskrit_hits if "wana" not in h and "wono" not in h]
        else:
            # Only wana → give to Sanskrit (ornamental suffix)
            prehidu_hits = [h for h in prehidu_hits if "wana" not in h and "wono" not in h]

    # --- Determine layer ---
    n_skt = len(sanskrit_hits)
    n_arb = len(arabic_hits)
    n_pre = len(prehidu_hits)
    total = n_skt + n_arb + n_pre

    if total == 0:
        layer = "UNKNOWN"
    elif n_skt > 0 and n_arb == 0 and n_pre == 0:
        layer = "SANSKRIT"
    elif n_arb > 0 and n_skt == 0 and n_pre == 0:
        layer = "ARABIC"
    elif n_pre > 0 and n_skt == 0 and n_arb == 0:
        layer = "PRE_HINDU"
    else:
        layer = "MIXED"

    return {
        "layer": layer,
        "sanskrit_hits": sanskrit_hits,
        "arabic_hits": arabic_hits,
        "prehidu_hits": prehidu_hits,
        "markers": sanskrit_hits + arabic_hits + prehidu_hits,
    }


# =====================================================================
# ANALYSIS FUNCTIONS
# =====================================================================

def analyze_by_province(villages):
    """Aggregate layer counts by province."""
    prov_counts = defaultdict(lambda: Counter())
    for v in villages:
        prov_counts[v["province"]][v["layer"]] += 1
    return dict(prov_counts)


def analyze_by_kabupaten(villages, kab_names):
    """Aggregate layer counts by kabupaten."""
    kab_counts = defaultdict(lambda: Counter())
    for v in villages:
        kab_code = v["kab_code"]
        kab_name = kab_names.get(kab_code, kab_code)
        kab_counts[(kab_code, kab_name)][v["layer"]] += 1
    return dict(kab_counts)


def compute_prehidu_ratio(counter):
    """Compute Pre-Hindu / (Pre-Hindu + Sanskrit) ratio."""
    pre = counter.get("PRE_HINDU", 0)
    skt = counter.get("SANSKRIT", 0)
    total = pre + skt
    if total == 0:
        return None
    return pre / total


def compute_layer_percentages(counter):
    """Compute percentage for each layer."""
    total = sum(counter.values())
    if total == 0:
        return {}
    return {layer: count / total * 100 for layer, count in counter.items()}


# =====================================================================
# VOLCANO DATA (major Java volcanoes)
# =====================================================================

JAVA_VOLCANOES = {
    "Krakatau": (-6.102, 105.423),
    "Anak Krakatau": (-6.102, 105.423),
    "Salak": (-6.720, 106.730),
    "Gede": (-6.780, 106.980),
    "Tangkuban Parahu": (-6.770, 107.600),
    "Papandayan": (-7.320, 107.730),
    "Galunggung": (-7.250, 108.058),
    "Ceremai": (-6.892, 108.400),
    "Slamet": (-7.242, 109.208),
    "Dieng": (-7.200, 109.920),
    "Sundoro": (-7.300, 109.992),
    "Sumbing": (-7.384, 110.072),
    "Merapi": (-7.541, 110.446),
    "Merbabu": (-7.455, 110.440),
    "Lawu": (-7.625, 111.192),
    "Wilis": (-7.808, 111.758),
    "Kelud": (-7.930, 112.308),
    "Arjuno-Welirang": (-7.725, 112.580),
    "Penanggungan": (-7.616, 112.630),
    "Bromo/Tengger": (-7.942, 112.950),
    "Semeru": (-8.108, 112.922),
    "Lamongan": (-7.979, 113.342),
    "Raung": (-8.125, 114.042),
    "Ijen": (-8.058, 114.242),
}


def min_volcano_distance(lat, lng):
    """Compute minimum distance to any Java volcano (km, Haversine)."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371  # Earth radius km
    min_d = float("inf")
    lat1 = radians(lat)
    lng1 = radians(lng)
    for vname, (vlat, vlng) in JAVA_VOLCANOES.items():
        lat2 = radians(vlat)
        lng2 = radians(vlng)
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlng/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        d = R * c
        if d < min_d:
            min_d = d
    return min_d


# =====================================================================
# VISUALIZATION
# =====================================================================

def plot_province_distribution(prov_counts, results_dir):
    """Bar chart of layer distribution by province."""
    layers = ["PRE_HINDU", "SANSKRIT", "ARABIC", "MIXED", "UNKNOWN"]
    colors = {
        "PRE_HINDU": "#2ca02c",   # green
        "SANSKRIT": "#d62728",    # red
        "ARABIC": "#1f77b4",      # blue
        "MIXED": "#ff7f0e",       # orange
        "UNKNOWN": "#7f7f7f",     # grey
    }

    provinces = sorted(prov_counts.keys())
    x = np.arange(len(provinces))
    width = 0.15

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Absolute counts
    for i, layer in enumerate(layers):
        vals = [prov_counts[p].get(layer, 0) for p in provinces]
        ax1.bar(x + i*width, vals, width, label=layer.replace("_", " "),
                color=colors[layer], alpha=0.85)
    ax1.set_xlabel("Province")
    ax1.set_ylabel("Village Count")
    ax1.set_title("Toponymic Layer Distribution by Province (Absolute)")
    ax1.set_xticks(x + 2*width)
    ax1.set_xticklabels([p.replace("Jawa ", "J.") for p in provinces],
                        rotation=30, ha="right", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Percentage stacked
    bottoms = np.zeros(len(provinces))
    for layer in layers:
        vals = []
        for p in provinces:
            total = sum(prov_counts[p].values())
            vals.append(prov_counts[p].get(layer, 0) / total * 100 if total else 0)
        ax2.bar(x, vals, bottom=bottoms, label=layer.replace("_", " "),
                color=colors[layer], alpha=0.85)
        bottoms += np.array(vals)
    ax2.set_xlabel("Province")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Toponymic Layer Distribution by Province (Percentage)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace("Jawa ", "J.") for p in provinces],
                        rotation=30, ha="right", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    outpath = results_dir / "fig1_province_distribution.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"  Saved {outpath.name}")


def plot_prehidu_ratio_by_kabupaten(kab_data, results_dir, top_n=30):
    """Horizontal bar chart: Pre-Hindu ratio by kabupaten (top and bottom)."""
    # Compute ratio for each
    ratios = []
    for (kab_code, kab_name), counter in kab_data.items():
        ratio = compute_prehidu_ratio(counter)
        total = sum(counter.values())
        if ratio is not None and total >= 20:  # min 20 villages
            ratios.append({
                "kab_code": kab_code,
                "kab_name": kab_name,
                "ratio": ratio,
                "total": total,
                "pre": counter.get("PRE_HINDU", 0),
                "skt": counter.get("SANSKRIT", 0),
            })

    ratios.sort(key=lambda x: x["ratio"], reverse=True)

    # Top and bottom
    top = ratios[:top_n]
    bottom = ratios[-top_n:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Top (highest Pre-Hindu)
    names = [r["kab_name"][:25] for r in top]
    vals = [r["ratio"] * 100 for r in top]
    bars = ax1.barh(range(len(top)), vals, color="#2ca02c", alpha=0.8)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(names, fontsize=7)
    ax1.set_xlabel("Pre-Hindu Ratio (%)")
    ax1.set_title(f"Top {top_n} Kabupaten: Highest Pre-Hindu Ratio")
    ax1.invert_yaxis()
    ax1.axvline(50, color="gray", linestyle="--", alpha=0.5)
    for bar, r in zip(bars, top):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{r['ratio']:.1%} (n={r['total']})", fontsize=6, va="center")

    # Bottom (lowest Pre-Hindu = highest Sanskrit)
    names = [r["kab_name"][:25] for r in bottom]
    vals = [r["ratio"] * 100 for r in bottom]
    bars = ax2.barh(range(len(bottom)), vals, color="#d62728", alpha=0.8)
    ax2.set_yticks(range(len(bottom)))
    ax2.set_yticklabels(names, fontsize=7)
    ax2.set_xlabel("Pre-Hindu Ratio (%)")
    ax2.set_title(f"Bottom {top_n} Kabupaten: Lowest Pre-Hindu Ratio")
    ax2.invert_yaxis()
    ax2.axvline(50, color="gray", linestyle="--", alpha=0.5)
    for bar, r in zip(bars, bottom):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{r['ratio']:.1%} (n={r['total']})", fontsize=6, va="center")

    plt.tight_layout()
    outpath = results_dir / "fig2_prehidu_ratio_kabupaten.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"  Saved {outpath.name}")

    return ratios


def plot_morpheme_frequency(villages, results_dir, top_n=30):
    """Bar chart of most frequent morpheme hits."""
    all_markers = Counter()
    for v in villages:
        for m in v.get("markers", []):
            all_markers[m] += 1

    top = all_markers.most_common(top_n)
    if not top:
        print("  No morpheme markers found, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    names = [m[0] for m in top]
    counts = [m[1] for m in top]

    # Color by layer
    bar_colors = []
    for name in names:
        if any(s in name for s in ["ci-", "ka-", "dir:", "watu", "batu", "gunung",
               "tegal", "sawah", "lebak", "rawa", "rowo", "sumber", "sumur",
               "kali", "kedung", "pasir", "jati", "sendang", "tlogo", "tlaga",
               "telaga", "gede", "karang", "tanjung", "kulon", "wetan", "lor",
               "kidul", "tengah", "bojong", "babakan", "leuwi", "girang",
               "hilir", "kampung", "dukuh", "rimba", "alas", "banyu", "banjar",
               "ranu", "danu", "pring", "glagah", "wringin", "waringin",
               "cilik", "desa"]):
            bar_colors.append("#2ca02c")
        elif any(s in name for s in ["sari", "rejo", "mulyo", "harjo", "jaya",
                "pura", "kerta", "karto", "agung", "rahayu", "mukti",
                "wana", "tirta", "tirto", "dewa", "dewi", "dharma", "darma",
                "ratna", "graha", "asri", "indah", "maha", "raja", "adi",
                "sri", "praja", "nagara", "negara", "loka", "bumi", "candi"]):
            bar_colors.append("#d62728")
        elif any(s in name for s in ["abad", "aman", "barokah", "falah",
                "hidayah", "hikmah", "ikhlas", "iman", "islam", "karim",
                "makmur", "rahmat", "salam", "sejahtera", "masjid", "mesjid",
                "surau", "mulia"]):
            bar_colors.append("#1f77b4")
        else:
            bar_colors.append("#7f7f7f")

    ax.barh(range(len(top)), counts, color=bar_colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Frequency (villages)")
    ax.set_title(f"Top {top_n} Toponymic Morphemes in Java Village Names")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="Pre-Hindu/Austronesian"),
        Patch(facecolor="#d62728", label="Sanskrit/Indic"),
        Patch(facecolor="#1f77b4", label="Arabic/Islamic"),
        Patch(facecolor="#7f7f7f", label="Other/Ambiguous"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    outpath = results_dir / "fig3_morpheme_frequency.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"  Saved {outpath.name}")


def plot_volcanic_distance_analysis(kab_ratios, coords, results_dir):
    """Scatter plot: kabupaten Pre-Hindu ratio vs distance to nearest volcano."""
    points = []
    for r in kab_ratios:
        kab_code = r["kab_code"]
        if kab_code in coords:
            lat = coords[kab_code]["lat"]
            lng = coords[kab_code]["lng"]
            dist = min_volcano_distance(lat, lng)
            points.append({
                "kab_name": r["kab_name"],
                "ratio": r["ratio"],
                "dist_km": dist,
                "total": r["total"],
                "lat": lat,
                "lng": lng,
            })

    if not points:
        print("  No kabupaten with coordinates found, skipping volcanic distance plot.")
        return None

    print(f"  {len(points)} kabupaten with coordinates for volcanic distance analysis")

    dists = np.array([p["dist_km"] for p in points])
    ratios = np.array([p["ratio"] for p in points])
    sizes = np.array([p["total"] for p in points])

    # Pearson and Spearman correlations
    from scipy import stats
    r_pearson, p_pearson = stats.pearsonr(dists, ratios)
    r_spearman, p_spearman = stats.spearmanr(dists, ratios)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(dists, ratios * 100, s=sizes / 5, alpha=0.5,
                         c=ratios, cmap="RdYlGn", edgecolors="gray", linewidths=0.3)
    ax.set_xlabel("Distance to Nearest Volcano (km)")
    ax.set_ylabel("Pre-Hindu Toponymic Ratio (%)")
    ax.set_title("Pre-Hindu Name Survival vs Volcanic Proximity\n"
                 f"Pearson r={r_pearson:.3f} (p={p_pearson:.4f}), "
                 f"Spearman rho={r_spearman:.3f} (p={p_spearman:.4f})")
    ax.grid(alpha=0.3)

    # Trend line
    z = np.polyfit(dists, ratios * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(dists.min(), dists.max(), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.5, label=f"Linear fit: y={z[0]:.2f}x+{z[1]:.1f}")
    ax.legend()

    plt.colorbar(scatter, ax=ax, label="Pre-Hindu Ratio")
    plt.tight_layout()
    outpath = results_dir / "fig4_volcanic_distance.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"  Saved {outpath.name}")

    return {
        "n": len(points),
        "r_pearson": r_pearson,
        "p_pearson": p_pearson,
        "r_spearman": r_spearman,
        "p_spearman": p_spearman,
        "points": points,
    }


def plot_east_west_gradient(prov_counts, results_dir):
    """Show east-west gradient: Banten (west) to Jawa Timur (east)."""
    # Order provinces west to east
    order = ["Banten", "DKI Jakarta", "Jawa Barat", "Jawa Tengah",
             "DI Yogyakarta", "Jawa Timur"]
    layers = ["PRE_HINDU", "SANSKRIT", "ARABIC", "MIXED", "UNKNOWN"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(order))

    # Pre-Hindu ratio (pre / (pre + sanskrit))
    ratios = []
    for prov in order:
        if prov in prov_counts:
            ratio = compute_prehidu_ratio(prov_counts[prov])
            ratios.append(ratio * 100 if ratio is not None else 0)
        else:
            ratios.append(0)

    bars = ax.bar(x, ratios, color="#2ca02c", alpha=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Pre-Hindu Ratio (%) = Pre-Hindu / (Pre-Hindu + Sanskrit)")
    ax.set_title("East-West Gradient: Pre-Hindu Toponymic Ratio Across Java")
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="50% line")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=9)

    ax.legend()
    plt.tight_layout()
    outpath = results_dir / "fig5_east_west_gradient.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"  Saved {outpath.name}")


def plot_layer_map(kab_ratios, coords, results_dir):
    """Simple scatter map of Java with kabupaten colored by Pre-Hindu ratio."""
    points = []
    for r in kab_ratios:
        kab_code = r["kab_code"]
        if kab_code in coords:
            points.append({
                "lat": coords[kab_code]["lat"],
                "lng": coords[kab_code]["lng"],
                "ratio": r["ratio"],
                "name": r["kab_name"],
                "total": r["total"],
            })

    if not points:
        print("  No coordinate data for map, skipping.")
        return

    fig, ax = plt.subplots(figsize=(16, 5))
    lats = [p["lat"] for p in points]
    lngs = [p["lng"] for p in points]
    ratios = [p["ratio"] * 100 for p in points]
    sizes = [p["total"] / 3 for p in points]

    scatter = ax.scatter(lngs, lats, c=ratios, cmap="RdYlGn", s=sizes,
                         alpha=0.7, edgecolors="gray", linewidths=0.3,
                         vmin=0, vmax=100)

    # Plot volcanoes
    for vname, (vlat, vlng) in JAVA_VOLCANOES.items():
        ax.plot(vlng, vlat, "^", color="red", markersize=6, alpha=0.6)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Java Toponymic Landscape: Pre-Hindu Ratio by Kabupaten\n"
                 "(Green = more Pre-Hindu, Red = more Sanskrit; triangles = volcanoes)")
    ax.set_xlim(104.5, 115)
    ax.set_ylim(-8.8, -5.8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    plt.colorbar(scatter, ax=ax, label="Pre-Hindu Ratio (%)", shrink=0.8)

    plt.tight_layout()
    outpath = results_dir / "fig6_toponymic_map.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"  Saved {outpath.name}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("E051: Java Toponymic Substrate Analysis")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1] Loading village data...")
    records = parse_sql_inserts(SQL_FILE)
    kab_names = get_kabupaten_names(records)
    villages = filter_java_villages(records)

    # Step 2: Load coordinates
    print("\n[2] Loading kabupaten coordinates...")
    coords = parse_level12_sql(SQL_L12_FILE)

    # Step 3: Classify village names
    print("\n[3] Classifying village names...")
    layer_counter = Counter()
    for v in villages:
        result = classify_village_name(v["nama"])
        v.update(result)
        layer_counter[v["layer"]] += 1

    print("\n  === OVERALL CLASSIFICATION ===")
    total = len(villages)
    for layer in ["PRE_HINDU", "SANSKRIT", "ARABIC", "MIXED", "UNKNOWN"]:
        count = layer_counter.get(layer, 0)
        print(f"  {layer:12s}: {count:6,} ({count/total*100:5.1f}%)")
    print(f"  {'TOTAL':12s}: {total:6,}")

    # Step 4: Province analysis
    print("\n[4] Province-level analysis...")
    prov_counts = analyze_by_province(villages)
    print(f"\n  {'Province':<18s} {'PRE_HINDU':>10s} {'SANSKRIT':>10s} {'ARABIC':>8s} {'MIXED':>8s} {'UNKNOWN':>8s} {'P-H Ratio':>10s}")
    print("  " + "-" * 80)
    for prov in sorted(prov_counts.keys()):
        c = prov_counts[prov]
        t = sum(c.values())
        ratio = compute_prehidu_ratio(c)
        ratio_str = f"{ratio:.1%}" if ratio is not None else "N/A"
        print(f"  {prov:<18s} {c.get('PRE_HINDU',0):>10,} {c.get('SANSKRIT',0):>10,} "
              f"{c.get('ARABIC',0):>8,} {c.get('MIXED',0):>8,} {c.get('UNKNOWN',0):>8,} "
              f"{ratio_str:>10s}")

    # Step 5: Kabupaten analysis
    print("\n[5] Kabupaten-level analysis...")
    kab_data = analyze_by_kabupaten(villages, kab_names)

    # Step 6: Generate plots
    print("\n[6] Generating visualizations...")
    plot_province_distribution(prov_counts, RESULTS_DIR)
    kab_ratios = plot_prehidu_ratio_by_kabupaten(kab_data, RESULTS_DIR)
    plot_morpheme_frequency(villages, RESULTS_DIR)
    plot_east_west_gradient(prov_counts, RESULTS_DIR)
    plot_layer_map(kab_ratios, coords, RESULTS_DIR)

    # Step 7: Volcanic distance analysis
    print("\n[7] Volcanic distance correlation analysis...")
    volc_result = plot_volcanic_distance_analysis(kab_ratios, coords, RESULTS_DIR)

    # Step 8: Export detailed results
    print("\n[8] Exporting results...")

    # Export village classification
    csv_path = RESULTS_DIR / "village_classifications.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kode", "nama", "province", "layer", "markers"])
        for v in villages:
            writer.writerow([
                v["kode"], v["nama"], v["province"],
                v["layer"], "; ".join(v.get("markers", []))
            ])
    print(f"  Saved {csv_path.name} ({len(villages):,} rows)")

    # Export kabupaten summary
    kab_csv_path = RESULTS_DIR / "kabupaten_summary.csv"
    with open(kab_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kab_code", "kab_name", "province", "total",
                         "pre_hindu", "sanskrit", "arabic", "mixed", "unknown",
                         "prehidu_ratio", "lat", "lng", "dist_volcano_km"])
        for r in kab_ratios:
            kab_code = r["kab_code"]
            prov_code = kab_code.split(".")[0]
            lat = coords.get(kab_code, {}).get("lat", "")
            lng = coords.get(kab_code, {}).get("lng", "")
            dist = ""
            if lat and lng:
                dist = f"{min_volcano_distance(lat, lng):.1f}"
            counter = kab_data.get((kab_code, r["kab_name"]), Counter())
            writer.writerow([
                kab_code, r["kab_name"], JAVA_PROVINCES.get(prov_code, ""),
                r["total"], r["pre"], r["skt"],
                counter.get("ARABIC", 0), counter.get("MIXED", 0),
                counter.get("UNKNOWN", 0), f"{r['ratio']:.4f}",
                lat, lng, dist
            ])
    print(f"  Saved {kab_csv_path.name} ({len(kab_ratios)} rows)")

    # Export top examples per layer
    examples_path = RESULTS_DIR / "layer_examples.txt"
    with open(examples_path, "w", encoding="utf-8") as f:
        for layer in ["PRE_HINDU", "SANSKRIT", "ARABIC", "MIXED"]:
            f.write(f"\n{'='*60}\n")
            f.write(f"  LAYER: {layer}\n")
            f.write(f"{'='*60}\n")
            examples = [v for v in villages if v["layer"] == layer][:50]
            for v in examples:
                f.write(f"  {v['nama']:<35s} [{v['province']:<15s}] "
                       f"markers: {', '.join(v.get('markers', []))}\n")
    print(f"  Saved {examples_path.name}")

    # Step 9: Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Pre-Hindu ratio across Java
    pre_total = layer_counter.get("PRE_HINDU", 0)
    skt_total = layer_counter.get("SANSKRIT", 0)
    overall_ratio = pre_total / (pre_total + skt_total) if (pre_total + skt_total) > 0 else 0
    print(f"\n  Overall Pre-Hindu ratio: {overall_ratio:.1%}")
    print(f"  Pre-Hindu: {pre_total:,} | Sanskrit: {skt_total:,} | Arabic: {layer_counter.get('ARABIC',0):,}")
    print(f"  Mixed: {layer_counter.get('MIXED',0):,} | Unknown: {layer_counter.get('UNKNOWN',0):,}")

    # East-west gradient
    print("\n  East-West Pre-Hindu Ratio:")
    for prov in ["Banten", "DKI Jakarta", "Jawa Barat", "Jawa Tengah",
                 "DI Yogyakarta", "Jawa Timur"]:
        if prov in prov_counts:
            ratio = compute_prehidu_ratio(prov_counts[prov])
            ratio_str = f"{ratio:.1%}" if ratio is not None else "N/A"
            print(f"    {prov:<18s}: {ratio_str}")

    # Volcanic distance correlation
    if volc_result:
        print(f"\n  Volcanic Distance Correlation:")
        print(f"    N kabupaten: {volc_result['n']}")
        print(f"    Pearson r:   {volc_result['r_pearson']:.4f} (p={volc_result['p_pearson']:.4f})")
        print(f"    Spearman rho: {volc_result['r_spearman']:.4f} (p={volc_result['p_spearman']:.4f})")

        # Interpretation
        if volc_result['p_spearman'] < 0.05:
            if volc_result['r_spearman'] > 0:
                print(f"    -> SIGNIFICANT: Pre-Hindu names MORE common FARTHER from volcanoes")
                print(f"       (Consistent with court-center Sanskrit hypothesis)")
            else:
                print(f"    -> SIGNIFICANT: Pre-Hindu names MORE common CLOSER to volcanoes")
                print(f"       (Consistent with highland-preservation hypothesis)")
        else:
            print(f"    -> NOT SIGNIFICANT at alpha=0.05")

    # Top kabupaten
    if kab_ratios:
        print(f"\n  Top 5 Kabupaten by Pre-Hindu Ratio:")
        for r in kab_ratios[:5]:
            print(f"    {r['kab_name']:<30s}: {r['ratio']:.1%} (n={r['total']})")
        print(f"\n  Bottom 5 Kabupaten by Pre-Hindu Ratio:")
        for r in kab_ratios[-5:]:
            print(f"    {r['kab_name']:<30s}: {r['ratio']:.1%} (n={r['total']})")

    print("\n" + "=" * 70)
    print("Analysis complete. See results/ for detailed output.")
    print("=" * 70)


if __name__ == "__main__":
    main()
