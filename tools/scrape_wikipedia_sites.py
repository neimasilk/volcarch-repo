"""
tools/scrape_wikipedia_sites.py

Scrape archaeological sites in East Java from Wikipedia.

Targets:
  - https://en.wikipedia.org/wiki/List_of_Hindu_temples_in_Indonesia
  - https://id.wikipedia.org/wiki/Daftar_candi_di_Jawa_Timur
  - https://en.wikipedia.org/wiki/Category:Archaeological_sites_in_East_Java

Usage:
    python tools/scrape_wikipedia_sites.py

Output:
    data/processed/east_java_sites_wiki.csv

Notes:
  - Wikipedia coordinates come from {{coord}} templates, exposed via Wikidata/wikitable.
  - We use the Wikipedia REST API (action=parse) to get structured tables.
  - Sites without coordinates are included with lat/lon=None and accuracy_level='no_coords'.
    These still contribute to the type/period catalog even without spatial analysis.
  - Data is under CC-BY-SA 4.0 — cite appropriately.
"""

import csv
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "east_java_sites_wiki.csv"

HEADERS = {
    "User-Agent": "VOLCARCH-research/0.1 (academic; Mukhlis Amien; contact volcarch-research)",
}

# Wikipedia pages to scrape — Indonesian and English
WIKI_PAGES = [
    {
        "lang": "id",
        "title": "Daftar_candi_di_Indonesia",
        "url": "https://id.wikipedia.org/wiki/Daftar_candi_di_Indonesia",
        "note": "Indonesian Wikipedia — list of all candis in Indonesia (filter East Java)",
    },
    {
        "lang": "en",
        "title": "List_of_Hindu_temples_in_Indonesia",
        "url": "https://en.wikipedia.org/wiki/List_of_Hindu_temples_in_Indonesia",
        "note": "English Wikipedia — partial, but has name catalog for many sites",
    },
    {
        "lang": "en",
        "title": "Candi_(Indonesian_archaeology)",
        "url": "https://en.wikipedia.org/wiki/Candi_(Indonesian_archaeology)",
        "note": "Overview article with links to individual candis",
    },
]

# Separate individual Wikipedia articles for major sites to get precise coords via Wikidata
MAJOR_SITES = [
    {"name": "Candi Singosari", "wikidata": "Q1083994", "type": "candi", "period": "Singosari (~1300 CE)"},
    {"name": "Candi Jago", "wikidata": "Q1058993", "type": "candi", "period": "Singosari (~1268 CE)"},
    {"name": "Candi Kidal", "wikidata": "Q1058994", "type": "candi", "period": "Singosari (~1248 CE)"},
    {"name": "Candi Jawi", "wikidata": "Q1083981", "type": "candi", "period": "Singosari (~1300 CE)"},
    {"name": "Candi Panataran", "wikidata": "Q654870", "type": "candi", "period": "Majapahit (~1197-1454 CE)"},
    {"name": "Candi Penataran", "wikidata": "Q654870", "type": "candi", "period": "Majapahit"},
    {"name": "Candi Sawentar", "wikidata": "Q16962428", "type": "candi", "period": "Majapahit"},
    {"name": "Candi Tikus", "wikidata": "Q3658688", "type": "candi", "period": "Majapahit (~14c CE)"},
    {"name": "Candi Bajang Ratu", "wikidata": "Q3658695", "type": "candi", "period": "Majapahit (~1400 CE)"},
    {"name": "Candi Brahu", "wikidata": "Q3658680", "type": "candi", "period": "Majapahit"},
    {"name": "Candi Wringinlawang", "wikidata": "Q11665399", "type": "candi", "period": "Majapahit"},
    {"name": "Candi Rimbi", "wikidata": "Q6029003", "type": "candi", "period": "Majapahit"},
    {"name": "Candi Surowono", "wikidata": "Q16950060", "type": "candi", "period": "Majapahit"},
    {"name": "Candi Gunung Gangsir", "wikidata": "Q5619406", "type": "candi", "period": "Medang (~10c CE)"},
    {"name": "Candi Songgoriti", "wikidata": "Q3879124", "type": "candi", "period": "Medang (~10c CE)"},
    {"name": "Prasasti Dinoyo", "wikidata": "Q3894534", "type": "prasasti", "period": "Kanjuruhan (~760 CE)"},
    {"name": "Situs Kanjuruhan", "wikidata": "Q4103578", "type": "settlement", "period": "Kanjuruhan (~760 CE)"},
    {"name": "Dwarapala Singosari", "wikidata": "Q17018700", "type": "arca", "period": "Singosari (~1268 CE)"},
    {"name": "Candi Lebak Jabung", "wikidata": "Q12503680", "type": "candi", "period": "Majapahit"},
    {"name": "Candi Kedaton", "wikidata": "Q12455540", "type": "candi", "period": "Majapahit"},
]

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIPEDIA_PARSE_API = "https://{lang}.wikipedia.org/w/api.php"


def fetch_wikidata_sparql_sites() -> list[dict]:
    """
    Use Wikidata SPARQL to get all East Java archaeological/temple sites with coordinates.

    Query strategy:
    - Items with P131 (located in admin entity) = Jawa Timur (Q3701765)
      OR coordinates within East Java bounding box
    - That are instances of: candi (Q1244719), Hindu temple (Q44539),
      Buddhist temple, or archaeological site
    - With P625 (coordinate location)
    """
    # Bounding box for East Java: lat -9 to -6.5, lon 111 to 115
    sparql_query = """
SELECT DISTINCT ?item ?itemLabel ?coord ?typeLabel ?inceptionLabel WHERE {
  ?item wdt:P625 ?coord .
  ?item wdt:P31 ?type .
  VALUES ?type {
    wd:Q1244719    # candi (Indonesian temple)
    wd:Q44539      # Hindu temple
    wd:Q856724     # Buddhist temple
    wd:M9571466    # candi in Indonesia
    wd:Q839954     # archaeological site
    wd:Q570116     # tourist attraction (catch-all for minor candis)
  }
  # Filter by coordinates within East Java bounding box
  FILTER(
    geof:latitude(?coord) >= -9.0 &&
    geof:latitude(?coord) <= -6.5 &&
    geof:longitude(?coord) >= 111.0 &&
    geof:longitude(?coord) <= 115.0
  )
  OPTIONAL { ?item wdt:P571 ?inception }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "id,en" }
}
ORDER BY ?itemLabel
"""
    headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Accept": "application/sparql-results+json",
    }
    try:
        resp = requests.get(
            WIKIDATA_SPARQL,
            params={"query": sparql_query, "format": "json"},
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])
        print(f"    SPARQL returned {len(bindings)} results")

        sites = []
        seen_qids = set()
        for b in bindings:
            qid = b.get("item", {}).get("value", "").split("/")[-1]
            if qid in seen_qids:
                continue
            seen_qids.add(qid)

            coord_str = b.get("coord", {}).get("value", "")
            # Wikidata SPARQL coords come as "Point(lon lat)"
            lat, lon = None, None
            if coord_str.startswith("Point("):
                try:
                    inner = coord_str[6:-1]  # strip "Point(" and ")"
                    parts = inner.split()
                    lon, lat = float(parts[0]), float(parts[1])
                except (ValueError, IndexError):
                    pass

            name = b.get("itemLabel", {}).get("value", qid)
            site_type = b.get("typeLabel", {}).get("value", "unknown")
            period = b.get("inceptionLabel", {}).get("value", "unknown")

            sites.append({
                "name": name,
                "type": site_type.lower().replace(" ", "_"),
                "period": period,
                "lat": lat,
                "lon": lon,
                "source": f"Wikidata SPARQL ({qid})",
                "osm_id": None,
                "accuracy_level": "wikidata_p625" if lat else "no_coords",
                "discovery_year": None,
                "notes": "",
                "wikipedia": f"https://www.wikidata.org/wiki/{qid}",
                "wikidata": qid,
            })

        return sites

    except requests.RequestException as e:
        print(f"    SPARQL query failed: {e}")
        return []


def scrape_wikipedia_table(page: dict) -> list[dict]:
    """
    Fetch a Wikipedia page and extract site rows from wikitables.
    Returns a list of site dicts.
    """
    sites = []
    url = page["url"]

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Could not fetch {url}: {e}")
        return sites

    soup = BeautifulSoup(resp.text, "lxml")

    # Strategy 1: find wikitables with site names
    tables = soup.find_all("table", class_=re.compile(r"wikitable"))
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        # Try to identify header columns
        header_row = rows[0]
        headers = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            row_data = [c.get_text(strip=True) for c in cells]
            if not row_data or not row_data[0]:
                continue

            # Try to extract a name (first non-empty cell)
            name = row_data[0] if row_data else "unknown"
            if len(name) < 2:  # skip separator rows
                continue

            # Try to find coordinates in the row (geo microformat or decimal in cells)
            lat, lon = extract_coords_from_row(row, row_data, headers)

            # Try to infer period/type from cell content
            period = "unknown"
            site_type = "candi"  # default for this source
            for i, h in enumerate(headers):
                if "period" in h or "era" in h or "abad" in h or "tahun" in h or "masa" in h:
                    if i < len(row_data):
                        period = row_data[i]
                if "type" in h or "jenis" in h or "tipe" in h:
                    if i < len(row_data):
                        site_type = row_data[i].lower()

            sites.append({
                "name": name,
                "type": site_type,
                "period": period,
                "lat": lat,
                "lon": lon,
                "source": f"Wikipedia ({page['lang']}): {page['title']}",
                "osm_id": None,
                "accuracy_level": "wikipedia_table" if lat else "no_coords",
                "discovery_year": None,
                "notes": f"Scraped from {url}",
                "wikipedia": url,
                "wikidata": "",
            })

    # Strategy 2: extract from bullet lists if no tables found
    if not sites:
        sites.extend(scrape_wikipedia_lists(soup, url, page))

    return sites


def extract_coords_from_row(row, row_data: list, headers: list) -> tuple:
    """Try multiple strategies to extract coords from a table row."""
    # Geo microformat: <span class="geo">lat; lon</span>
    geo = row.find("span", class_="geo")
    if geo:
        try:
            parts = geo.get_text().split(";")
            return float(parts[0].strip()), float(parts[1].strip())
        except (ValueError, IndexError):
            pass

    # Decimal in cells matching lat/lon patterns
    for cell_text in row_data:
        match = re.search(r'(-?\d{1,3}\.\d{3,})\s*[,;]\s*(-?\d{1,3}\.\d{3,})', cell_text)
        if match:
            try:
                lat, lon = float(match.group(1)), float(match.group(2))
                # Sanity check for East Java
                if -9 < lat < -6 and 111 < lon < 115:
                    return lat, lon
            except ValueError:
                pass

    return None, None


def scrape_wikipedia_lists(soup: BeautifulSoup, url: str, page: dict) -> list[dict]:
    """Extract sites from bullet list items when no wikitable is found."""
    sites = []
    for li in soup.find_all("li"):
        link = li.find("a")
        if not link:
            continue
        name = link.get_text(strip=True)
        if not name or len(name) < 3:
            continue
        # Filter: must look like an archaeological site name
        if not any(kw in name.lower() for kw in ["candi", "prasasti", "arca", "pura", "situs", "temple"]):
            continue
        sites.append({
            "name": name,
            "type": "unknown",
            "period": "unknown",
            "lat": None,
            "lon": None,
            "source": f"Wikipedia ({page['lang']}): {page['title']}",
            "osm_id": None,
            "accuracy_level": "no_coords",
            "discovery_year": None,
            "notes": f"List item from {url}",
            "wikipedia": url,
            "wikidata": "",
        })
    return sites


def fetch_major_sites_via_wikidata() -> list[dict]:
    """
    Fetch East Java archaeological sites via Wikidata SPARQL.
    This is more reliable than direct QID lookups as it queries by geography.
    """
    print("  Querying Wikidata SPARQL for East Java sites with coordinates...")
    sites = fetch_wikidata_sparql_sites()
    with_coords = sum(1 for s in sites if s["lat"] is not None)
    print(f"  Wikidata SPARQL: {len(sites)} sites, {with_coords} with coordinates")
    for s in sites[:5]:
        if s["lat"]:
            print(f"    {s['name']}: {s['lat']:.4f}, {s['lon']:.4f} ({s['type']})")
    if len(sites) > 5:
        print(f"    ... and {len(sites) - 5} more")
    return sites


FIELDNAMES = ["name", "type", "period", "lat", "lon", "source", "osm_id",
              "accuracy_level", "discovery_year", "notes", "wikipedia", "wikidata"]


def main():
    all_sites = []

    # 1. Major sites via Wikidata (best coordinate quality)
    print("=" * 60)
    print("Step 1: Fetching major sites via Wikidata...")
    print("=" * 60)
    major = fetch_major_sites_via_wikidata()
    all_sites.extend(major)
    print(f"  Major sites collected: {len(major)}")

    # 2. Wikipedia table scraping
    print("\n" + "=" * 60)
    print("Step 2: Scraping Wikipedia tables...")
    print("=" * 60)
    for page in WIKI_PAGES:
        print(f"\n  Scraping: {page['url']}")
        scraped = scrape_wikipedia_table(page)
        all_sites.extend(scraped)
        print(f"  Sites found: {len(scraped)}")
        time.sleep(1)

    # 3. Write output
    print(f"\nTotal sites collected: {len(all_sites)}")
    with_coords = [s for s in all_sites if s["lat"] is not None]
    print(f"With coordinates: {len(with_coords)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_sites)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
