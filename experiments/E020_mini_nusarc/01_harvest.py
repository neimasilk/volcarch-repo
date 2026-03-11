"""
E020: Mini-NusaRC Phase 1 — Automated Paper Harvest.

Queries Semantic Scholar and OpenAlex APIs for papers containing
radiocarbon dates from Nusantaran archaeological sites.

Run from repo root:
    python experiments/E020_mini_nusarc/01_harvest.py
"""

import json
import time
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote_plus, urlencode
from urllib.error import HTTPError, URLError

RESULTS_DIR = Path(__file__).parent / "data"
RESULTS_DIR.mkdir(exist_ok=True)

# Rate limit: 1 request per second for Semantic Scholar
RATE_LIMIT_S = 1.2

# Search queries — designed to find papers with C14 dates in Nusantara
QUERIES = [
    # Core radiocarbon + region queries
    "radiocarbon dating Indonesia archaeology",
    "radiocarbon dating Sulawesi cave",
    "radiocarbon dating Java archaeological",
    "radiocarbon dating Borneo Kalimantan archaeology",
    "radiocarbon dating Philippines cave archaeology",
    "radiocarbon dating Timor archaeology Pleistocene",
    "radiocarbon dating Maluku archaeology",
    "radiocarbon Madagascar human colonization",
    # Specific site queries
    "Niah Cave radiocarbon chronology",
    "Liang Bua chronology Flores",
    "Song Terus Java Pleistocene",
    "Maros Pangkep cave art dating Sulawesi",
    "Leang Timpuseng dating",
    # Broader Pleistocene queries
    "Pleistocene human Southeast Asia cave",
    "Late Pleistocene Island Southeast Asia",
    "Homo sapiens arrival Southeast Asia dating",
    "Austronesian expansion radiocarbon",
    "Neolithic Southeast Asia chronology",
    # Sumatra specific
    "Lida Ajer Sumatra dating",
    "Padang Highlands cave archaeology",
]


def query_semantic_scholar(query, limit=50):
    """Query Semantic Scholar API for papers."""
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,authors,year,externalIds,isOpenAccess,openAccessPdf,abstract,citationCount"
    params = urlencode({
        "query": query,
        "limit": limit,
        "fields": fields,
    })
    url = f"{base}?{params}"

    req = Request(url)
    req.add_header("User-Agent", "NusaRC-Harvester/1.0 (academic research)")

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data.get("data", [])
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        print(f"    Error querying Semantic Scholar: {e}")
        return []


def query_openalex(query, limit=50):
    """Query OpenAlex API for papers."""
    base = "https://api.openalex.org/works"
    params = urlencode({
        "search": query,
        "per_page": limit,
        "select": "id,doi,title,publication_year,open_access,authorships,cited_by_count,abstract_inverted_index",
    })
    url = f"{base}?{params}"

    req = Request(url)
    req.add_header("User-Agent", "mailto:nusarc@volcarch.org")

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data.get("results", [])
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        print(f"    Error querying OpenAlex: {e}")
        return []


def extract_doi(paper, source):
    """Extract DOI from paper metadata."""
    if source == "semantic_scholar":
        ext = paper.get("externalIds", {})
        if ext and "DOI" in ext:
            return ext["DOI"]
    elif source == "openalex":
        doi = paper.get("doi")
        if doi:
            return doi.replace("https://doi.org/", "")
    return None


def normalize_paper(paper, source):
    """Normalize paper metadata from different sources."""
    if source == "semantic_scholar":
        doi = extract_doi(paper, source)
        is_oa = paper.get("isOpenAccess", False)
        oa_url = None
        if paper.get("openAccessPdf"):
            oa_url = paper["openAccessPdf"].get("url")
        authors = []
        for a in (paper.get("authors") or []):
            if a.get("name"):
                authors.append(a["name"])
        return {
            "doi": doi,
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "authors": "; ".join(authors[:5]),
            "is_open_access": is_oa,
            "oa_url": oa_url,
            "citations": paper.get("citationCount", 0),
            "abstract": (paper.get("abstract") or "")[:500],
            "source": "semantic_scholar",
        }
    elif source == "openalex":
        doi = extract_doi(paper, source)
        oa = paper.get("open_access", {})
        is_oa = oa.get("is_oa", False)
        oa_url = oa.get("oa_url")
        authors = []
        for a in (paper.get("authorships") or []):
            name = a.get("author", {}).get("display_name")
            if name:
                authors.append(name)
        # Reconstruct abstract from inverted index
        abstract = ""
        inv = paper.get("abstract_inverted_index")
        if inv:
            words = {}
            for word, positions in inv.items():
                for pos in positions:
                    words[pos] = word
            abstract = " ".join(words[k] for k in sorted(words.keys()))[:500]
        return {
            "doi": doi,
            "title": (paper.get("title") or ""),
            "year": paper.get("publication_year"),
            "authors": "; ".join(authors[:5]),
            "is_open_access": is_oa,
            "oa_url": oa_url,
            "citations": paper.get("cited_by_count", 0),
            "abstract": abstract,
            "source": "openalex",
        }
    return None


def is_relevant(paper):
    """Basic relevance filter based on title and abstract."""
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()

    # Must mention a region
    regions = ["indonesia", "java", "jawa", "sulawesi", "borneo", "kalimantan",
               "sumatra", "sumatera", "philippines", "filipina", "timor", "flores",
               "maluku", "moluccas", "madagascar", "nusantara", "southeast asia",
               "island southeast", "wallacea", "sunda", "sahul", "niah", "tabon",
               "palawan", "maros", "liang bua"]
    has_region = any(r in text for r in regions)

    # Must mention dating or archaeology
    dating = ["radiocarbon", "c14", "c-14", "ams dat", "luminescence", "osl",
              "u-series", "uranium", "chronolog", "archaeolog", "pleistocene",
              "holocene", "neolithic", "paleolithic", "cave site", "rock art",
              "excavat", "stratigraphy", "cal bp", "years bp"]
    has_dating = any(d in text for d in dating)

    return has_region and has_dating


def main():
    print("=" * 60)
    print("E020 Phase 1: Automated Paper Harvest")
    print("=" * 60)

    all_papers = {}  # DOI -> paper, for dedup
    no_doi_papers = []

    # --- Semantic Scholar ---
    print(f"\n[1/2] Querying Semantic Scholar ({len(QUERIES)} queries)...")
    for i, query in enumerate(QUERIES):
        print(f"  [{i+1}/{len(QUERIES)}] '{query}'")
        papers = query_semantic_scholar(query, limit=30)
        for p in papers:
            norm = normalize_paper(p, "semantic_scholar")
            if norm and is_relevant(norm):
                doi = norm["doi"]
                if doi:
                    if doi not in all_papers or norm["citations"] > all_papers[doi].get("citations", 0):
                        all_papers[doi] = norm
                else:
                    no_doi_papers.append(norm)
        print(f"    Found {len(papers)} results, {len(all_papers)} unique relevant so far")
        time.sleep(RATE_LIMIT_S)

    # --- OpenAlex ---
    print(f"\n[2/2] Querying OpenAlex ({len(QUERIES)} queries)...")
    for i, query in enumerate(QUERIES):
        print(f"  [{i+1}/{len(QUERIES)}] '{query}'")
        papers = query_openalex(query, limit=30)
        for p in papers:
            norm = normalize_paper(p, "openalex")
            if norm and is_relevant(norm):
                doi = norm["doi"]
                if doi:
                    if doi not in all_papers:
                        all_papers[doi] = norm
                    elif norm.get("oa_url") and not all_papers[doi].get("oa_url"):
                        # Prefer version with OA URL
                        all_papers[doi]["oa_url"] = norm["oa_url"]
                        all_papers[doi]["is_open_access"] = True
                else:
                    no_doi_papers.append(norm)
        print(f"    Found {len(papers)} results, {len(all_papers)} unique relevant so far")
        time.sleep(0.5)  # OpenAlex is more generous

    # --- Summary ---
    papers_list = sorted(all_papers.values(), key=lambda p: p.get("citations", 0), reverse=True)

    oa_papers = [p for p in papers_list if p.get("is_open_access")]
    paywalled = [p for p in papers_list if not p.get("is_open_access")]

    print(f"\n{'=' * 60}")
    print(f"HARVEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total unique papers (with DOI): {len(papers_list)}")
    print(f"  Open access: {len(oa_papers)} ({100*len(oa_papers)/max(len(papers_list),1):.0f}%)")
    print(f"  Paywalled: {len(paywalled)} ({100*len(paywalled)/max(len(papers_list),1):.0f}%)")
    print(f"  Without DOI: {len(no_doi_papers)}")

    print(f"\n  Top 20 open-access papers by citations:")
    for i, p in enumerate(oa_papers[:20]):
        print(f"    {i+1}. [{p.get('citations',0)} cit] {p['title'][:80]}")
        print(f"       DOI: {p['doi']} | Year: {p.get('year')}")
        if p.get("oa_url"):
            print(f"       URL: {p['oa_url']}")

    # --- Save results ---
    output = {
        "harvest_date": "2026-03-05",
        "n_queries": len(QUERIES),
        "n_total": len(papers_list),
        "n_open_access": len(oa_papers),
        "n_paywalled": len(paywalled),
        "papers": papers_list,
        "no_doi_papers": no_doi_papers[:50],  # cap at 50
    }
    out_path = RESULTS_DIR / "harvest_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    # Save OA papers as simple CSV for quick review
    csv_path = RESULTS_DIR / "harvest_open_access.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("doi,year,citations,title,oa_url\n")
        for p in oa_papers:
            title = p["title"].replace('"', "'")
            url = p.get("oa_url", "")
            f.write(f'"{p["doi"]}",{p.get("year","")},{p.get("citations",0)},"{title}","{url}"\n')
    print(f"  Saved: {csv_path}")

    print(f"\nPhase 1 COMPLETE. Next: Phase 2 (extract C14 dates from top OA papers).")


if __name__ == "__main__":
    main()
