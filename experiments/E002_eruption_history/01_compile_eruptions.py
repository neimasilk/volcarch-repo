"""
E002 / Step 1: Compile eruption history for East Java volcanoes affecting Malang basin.

Volcanoes covered:
    - Kelud       (GVP: 263280) — most impactful on Malang basin
    - Semeru      (GVP: 263300) — frequent, continuous low-level
    - Arjuno-Welirang (GVP: 263260)
    - Bromo       (GVP: 263310)

Data source: Global Volcanism Program (Smithsonian Institution)
    URL: https://volcano.si.edu/database/search_eruption_excel.cfm
    The GVP provides eruption data as downloadable Excel/CSV.
    We use their publicly accessible eruption search results.

Output: data/processed/eruption_history.csv

Run from repo root:
    python experiments/E002_eruption_history/01_compile_eruptions.py

NOTE: GVP data requires citation:
    Global Volcanism Program, [year]. [Database entry reference.]
    In: Venzke, E (ed.), 2013-. Volcanoes of the World, v. 5.x.x.
    Smithsonian Institution. https://doi.org/10.5479/si.GVP.VOTW5-2023.5.2
"""

import csv
import time
import requests
import pandas as pd
from pathlib import Path
from io import StringIO

REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "eruption_history.csv"
RAW_DIR = REPO_ROOT / "data" / "raw" / "gvp"

# GVP volcano IDs and names
VOLCANOES = {
    "Kelud": "263280",
    "Semeru": "263300",
    "Arjuno-Welirang": "263260",
    "Bromo": "263310",
}

# GVP eruption search API endpoint (returns CSV)
GVP_ERUPTION_URL = "https://volcano.si.edu/database/search_eruption_excel.cfm"


def fetch_gvp_eruptions(volcano_id: str, volcano_name: str) -> pd.DataFrame | None:
    """
    Fetch eruption records from GVP for a single volcano.

    GVP search form posts to the above URL with volc_num as the volcano number.
    Returns a DataFrame or None if the fetch fails.

    Note: GVP does not have a formal JSON API. We use the CSV export form.
    If this breaks, fall back to manual download from:
        https://volcano.si.edu/volcano.cfm?vn=<volcano_id>#tabEruptions
    """
    params = {
        "volc_num": volcano_id,
        "eruption_startdate": "0000",  # all dates
        "eruption_stopdate": "2026",
        "eruption_category": "Confirmed",
        "submit": "Search+GVP+Database",
    }

    headers = {
        "User-Agent": "VOLCARCH-research/0.1 (academic; contact: volcarch-research)",
        "Referer": "https://volcano.si.edu/",
    }

    try:
        print(f"  Fetching GVP data for {volcano_name} (ID: {volcano_id})...")
        resp = requests.get(GVP_ERUPTION_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()

        # GVP returns an HTML page with embedded table, not raw CSV via GET.
        # Check if we got an Excel/CSV redirect or embedded data.
        content_type = resp.headers.get("Content-Type", "")

        if "excel" in content_type or "spreadsheet" in content_type:
            # Got binary Excel — save raw and parse
            raw_path = RAW_DIR / f"gvp_{volcano_id}.xlsx"
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(resp.content)
            df = pd.read_excel(raw_path, header=0)
            print(f"    Got Excel: {len(df)} rows")
            return df

        elif "csv" in content_type or "text/plain" in content_type:
            df = pd.read_csv(StringIO(resp.text))
            print(f"    Got CSV: {len(df)} rows")
            return df

        else:
            # GVP likely returned HTML — direct CSV download not available via GET
            print(f"    WARNING: GVP returned HTML (not CSV/Excel) for {volcano_name}.")
            print(f"    Content-Type: {content_type}")
            print(f"    Manual download required. See instructions in fallback CSV template.")
            return None

    except requests.RequestException as e:
        print(f"    ERROR fetching {volcano_name}: {e}")
        return None


def load_manual_eruption_data() -> pd.DataFrame:
    """
    Returns a DataFrame with manually-compiled eruption data as fallback.

    This seed data is compiled from:
    - Global Volcanism Program (https://volcano.si.edu)
    - Wikipedia eruption histories
    - Published papers (see data/sources.md for full citations)

    Focused on eruptions with documented or estimated ashfall at Malang distance.
    INCOMPLETE — supplement with full GVP download.
    """
    records = [
        # Kelud eruptions with documented Malang impact
        # Source: GVP, Bonadonna & Houghton 2005, Thouret et al. 1998
        {"volcano": "Kelud", "gvp_id": "263280", "year": 1919, "vei": 4,
         "start_date": "1919-05-19", "end_date": "1919-05-19",
         "ashfall_malang_cm_est": 10.0,
         "ashfall_malang_cm_source": "Thouret et al. 1998 — isopach estimate at ~40km",
         "source": "GVP + Thouret1998", "notes": "VEI 4, major event, ~5000 deaths"},

        {"volcano": "Kelud", "gvp_id": "263280", "year": 1951, "vei": 3,
         "start_date": "1951-08-31", "end_date": "1951-08-31",
         "ashfall_malang_cm_est": 2.0,
         "ashfall_malang_cm_source": "Estimated from VEI 3 at ~40km, literature range 1-5cm",
         "source": "GVP", "notes": "VEI 3"},

        {"volcano": "Kelud", "gvp_id": "263280", "year": 1966, "vei": 4,
         "start_date": "1966-04-26", "end_date": "1966-04-26",
         "ashfall_malang_cm_est": 8.0,
         "ashfall_malang_cm_source": "Estimated from VEI 4, similar to 1919 event",
         "source": "GVP", "notes": "VEI 4"},

        {"volcano": "Kelud", "gvp_id": "263280", "year": 1990, "vei": 4,
         "start_date": "1990-02-10", "end_date": "1990-02-10",
         "ashfall_malang_cm_est": 5.0,
         "ashfall_malang_cm_source": "Published isopach, Malang ~3-8cm (Rodolfo et al. 1993)",
         "source": "GVP + Rodolfo1993", "notes": "VEI 4, well-documented"},

        {"volcano": "Kelud", "gvp_id": "263280", "year": 2014, "vei": 4,
         "start_date": "2014-02-13", "end_date": "2014-02-13",
         "ashfall_malang_cm_est": 3.0,
         "ashfall_malang_cm_source": "Documented ashfall ~2-5cm in Malang (PVMBG reports)",
         "source": "GVP + PVMBG", "notes": "VEI 4, most recent major eruption"},

        # Semeru — near-continuous; select notable larger events
        {"volcano": "Semeru", "gvp_id": "263300", "year": 2021, "vei": 3,
         "start_date": "2021-12-04", "end_date": "2021-12-04",
         "ashfall_malang_cm_est": 0.2,
         "ashfall_malang_cm_source": "Estimated — Semeru is farther from Malang city (~75km)",
         "source": "GVP", "notes": "Notable recent dome collapse + pyroclastic flows"},

        # Bromo
        {"volcano": "Bromo", "gvp_id": "263310", "year": 2016, "vei": 2,
         "start_date": "2016-01-01", "end_date": "2016-12-31",
         "ashfall_malang_cm_est": 0.1,
         "ashfall_malang_cm_source": "Estimated — Bromo VEI 2, ~80km from Malang",
         "source": "GVP", "notes": "Prolonged activity period"},

        {"volcano": "Bromo", "gvp_id": "263310", "year": 2010, "vei": 3,
         "start_date": "2010-11-08", "end_date": "2011-01-15",
         "ashfall_malang_cm_est": 0.5,
         "ashfall_malang_cm_source": "Estimated from VEI 3 at ~80km distance",
         "source": "GVP", "notes": "VEI 3, larger than typical Bromo activity"},
    ]

    return pd.DataFrame(records)


def normalize_gvp_dataframe(df: pd.DataFrame, volcano_name: str, gvp_id: str) -> pd.DataFrame:
    """
    Normalize a raw GVP DataFrame to our standard schema.
    GVP column names vary; this maps known variants.
    """
    col_map = {
        "Volcano Number": "gvp_id",
        "Volcano Name": "volcano",
        "Eruption Number": "eruption_number",
        "Eruption Category": "category",
        "Start Year": "year",
        "Start Month": "start_month",
        "Start Day": "start_day",
        "End Year": "end_year",
        "End Month": "end_month",
        "End Day": "end_day",
        "VEI": "vei",
        "Evidence Type": "evidence_type",
    }

    df = df.rename(columns=col_map)

    # Build start_date
    if "year" in df.columns:
        df["start_date"] = df["year"].astype(str)
    else:
        df["start_date"] = "unknown"

    # Add our fields
    df["volcano"] = volcano_name
    df["gvp_id"] = gvp_id
    df["ashfall_malang_cm_est"] = None
    df["ashfall_malang_cm_source"] = "not estimated — raw GVP import"
    df["source"] = "GVP Smithsonian"
    df["notes"] = ""

    # Select only our schema columns
    schema = ["volcano", "gvp_id", "year", "vei", "start_date", "end_date",
              "ashfall_malang_cm_est", "ashfall_malang_cm_source", "source", "notes"]
    for col in schema:
        if col not in df.columns:
            df[col] = None

    return df[schema]


def main():
    print("E002: Compiling eruption history for East Java volcanoes")
    print("=" * 60)

    all_frames = []
    api_success = False

    for name, vid in VOLCANOES.items():
        df = fetch_gvp_eruptions(vid, name)
        time.sleep(2)  # be polite to GVP servers

        if df is not None:
            normalized = normalize_gvp_dataframe(df, name, vid)
            all_frames.append(normalized)
            api_success = True
        else:
            print(f"  Falling through to manual data for {name}.")

    if not api_success:
        print("\nGVP API not accessible via automated download.")
        print("Using manually-compiled seed dataset (partial).")
        print("ACTION REQUIRED: Download full GVP data manually from:")
        print("  https://volcano.si.edu/database/search_eruption_excel.cfm")
        print(f"  Save to: {RAW_DIR}/")
        print()

        manual_df = load_manual_eruption_data()
        all_frames.append(manual_df)
        print(f"Manual seed data: {len(manual_df)} key eruption records loaded.")

    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)

        # Filter: only confirmed eruptions (for GVP data)
        if "category" in result.columns:
            result = result[result["category"].isin(["Confirmed", None, ""])]

        # Sort by volcano and year
        result = result.sort_values(["volcano", "year"], na_position="last")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(OUTPUT_PATH, index=False)

        print(f"\nSaved {len(result)} eruption records to {OUTPUT_PATH}")
        print(f"\nRecord count by volcano:\n{result['volcano'].value_counts().to_string()}")

        # VEI distribution
        if "vei" in result.columns:
            print(f"\nVEI distribution:\n{result['vei'].value_counts().sort_index().to_string()}")

        # Flag eruptions with Malang ashfall estimates
        with_ashfall = result[result["ashfall_malang_cm_est"].notna()]
        print(f"\nEruptions with Malang ashfall estimates: {len(with_ashfall)}")
        if len(with_ashfall) > 0:
            total_cm = with_ashfall["ashfall_malang_cm_est"].sum()
            print(f"Total estimated ashfall (documented events only): {total_cm:.1f} cm")

    else:
        print("ERROR: No eruption data collected.")


if __name__ == "__main__":
    main()
