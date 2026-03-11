"""
E039b: VCS Distance-Based Test
===============================
Refines E039 by using CONTINUOUS distance to nearest active volcano
instead of binary island-type classification.

Uses GVP Holocene volcano locations + Pulotu culture coordinates.
Tests: Does proximity to active volcanoes correlate with ritual complexity?
"""
import io
import json
import sys
import warnings
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

PULOTU = Path("D:/documents/volcarch-repo/experiments/E023_ritual_screening/data/pulotu/cldf")
GVP = Path("D:/documents/volcarch-repo/data/raw/gvp/GVP_Eruption_Search_Result.xlsx")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def haversine(lon1, lat1, lon2, lat2):
    """Haversine distance in km."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))


def load_gvp_volcanoes():
    """Load unique volcano locations from GVP database."""
    try:
        gvp = pd.read_excel(GVP)
        print(f"GVP eruptions loaded: {len(gvp)}")

        # Get unique volcanoes with lat/lon
        vol_cols = [c for c in gvp.columns if 'lat' in c.lower()]
        lon_cols = [c for c in gvp.columns if 'lon' in c.lower()]
        name_cols = [c for c in gvp.columns if 'volcano' in c.lower() and 'name' in c.lower()]
        num_cols = [c for c in gvp.columns if 'volcano' in c.lower() and 'num' in c.lower()]
        vei_cols = [c for c in gvp.columns if 'vei' in c.lower()]

        print(f"  Lat columns: {vol_cols}")
        print(f"  Lon columns: {lon_cols}")
        print(f"  Name columns: {name_cols}")
        print(f"  VEI columns: {vei_cols}")

        lat_col = vol_cols[0] if vol_cols else None
        lon_col = lon_cols[0] if lon_cols else None
        name_col = name_cols[0] if name_cols else None
        num_col = num_cols[0] if num_cols else None
        vei_col = vei_cols[0] if vei_cols else None

        if not lat_col or not lon_col:
            print("ERROR: Cannot find lat/lon columns in GVP")
            return pd.DataFrame()

        # Unique volcanoes with eruption count and max VEI
        volcanoes = gvp.groupby(num_col if num_col else name_col).agg({
            name_col: 'first',
            lat_col: 'first',
            lon_col: 'first',
            vei_col: ['max', 'count'] if vei_col else [],
        }).reset_index()

        # Flatten multi-level columns
        volcanoes.columns = ['_'.join(str(c) for c in col).strip('_') for col in volcanoes.columns]

        # Standardize column names
        rename = {}
        for c in volcanoes.columns:
            if 'lat' in c.lower() and 'first' in c.lower():
                rename[c] = 'latitude'
            elif 'lon' in c.lower() and 'first' in c.lower():
                rename[c] = 'longitude'
            elif 'name' in c.lower() and 'first' in c.lower():
                rename[c] = 'volcano_name'
            elif 'max' in c.lower() and 'vei' in c.lower():
                rename[c] = 'max_vei'
            elif 'count' in c.lower():
                rename[c] = 'n_eruptions'

        volcanoes = volcanoes.rename(columns=rename)
        volcanoes = volcanoes.dropna(subset=['latitude', 'longitude'])

        print(f"  Unique volcanoes: {len(volcanoes)}")
        return volcanoes

    except Exception as e:
        print(f"ERROR loading GVP: {e}")
        # Fallback: use known major volcanoes
        print("  Using fallback: 45 known Indonesian + Pacific volcanoes")
        return pd.DataFrame()


def load_gvp_simple():
    """Load unique volcano locations from GVP Excel (sheet='Eruption List', header=1)."""
    try:
        gvp = pd.read_excel(GVP, sheet_name='Eruption List', header=1)
        print(f"GVP eruptions loaded: {len(gvp)}")

        volcanoes = gvp.drop_duplicates(subset=['Volcano Name'])[
            ['Volcano Name', 'Latitude', 'Longitude']
        ].copy()
        volcanoes.columns = ['volcano_name', 'latitude', 'longitude']
        volcanoes = volcanoes.dropna()

        # Count eruptions per volcano
        eruption_counts = gvp['Volcano Name'].value_counts().to_dict()
        volcanoes['n_eruptions'] = volcanoes['volcano_name'].map(eruption_counts)

        print(f"Unique volcanoes: {len(volcanoes)}")
        print(f"Top 10 by eruption count:")
        for _, row in volcanoes.nlargest(10, 'n_eruptions').iterrows():
            print(f"  {row['volcano_name']}: {int(row['n_eruptions'])} eruptions "
                  f"({row['latitude']:.1f}, {row['longitude']:.1f})")

        return volcanoes

    except Exception as e:
        print(f"GVP load error: {e}")
        return pd.DataFrame()


def compute_nearest_volcano(cultures_df, volcanoes_df):
    """For each culture, find nearest active volcano and distance."""
    results = []
    for _, culture in cultures_df.iterrows():
        c_lat, c_lon = culture['Latitude'], culture['Longitude']
        if pd.isna(c_lat) or pd.isna(c_lon):
            results.append({'nearest_volcano': None, 'distance_km': np.nan,
                            'nearest_eruptions': np.nan})
            continue

        min_dist = float('inf')
        nearest = None
        nearest_eruptions = 0
        for _, vol in volcanoes_df.iterrows():
            d = haversine(c_lon, c_lat, vol['longitude'], vol['latitude'])
            if d < min_dist:
                min_dist = d
                nearest = vol['volcano_name']
                nearest_eruptions = vol.get('n_eruptions', 0)

        results.append({
            'nearest_volcano': nearest,
            'distance_km': round(min_dist, 1),
            'nearest_eruptions': nearest_eruptions
        })

    return pd.DataFrame(results)


def load_pulotu_pivot():
    """Load and pivot Pulotu data."""
    cultures = pd.read_csv(PULOTU / "cultures.csv", encoding="utf-8")
    responses = pd.read_csv(PULOTU / "responses.csv", encoding="utf-8")

    merged = responses.merge(cultures[["ID", "Name", "Latitude", "Longitude"]],
                             left_on="Language_ID", right_on="ID", how="left",
                             suffixes=("_resp", "_cult"))
    merged["code_value"] = merged["Code_ID"].apply(
        lambda x: int(str(x).split("-")[-1]) if pd.notna(x) and "-" in str(x) else np.nan
    )
    def safe_float(val):
        try: return float(val)
        except: return np.nan

    merged["numeric_value"] = merged["code_value"].where(
        merged["code_value"].notna(), merged["Value"].apply(safe_float))

    pivot = merged.pivot_table(
        index=["Language_ID", "Name", "Latitude", "Longitude"],
        columns="Parameter_ID", values="numeric_value", aggfunc="first"
    ).reset_index()
    pivot.columns = [str(c) for c in pivot.columns]

    # Compute ritual indices
    ritual_vars = ["2", "3", "4", "5", "6", "7", "10", "11", "12", "21", "35"]
    available = [v for v in ritual_vars if v in pivot.columns]
    pivot["ritual_complexity"] = pivot[available].mean(axis=1)

    mort_vars = ["4", "5", "10"]
    available_mort = [v for v in mort_vars if v in pivot.columns]
    pivot["mortuary_index"] = pivot[available_mort].mean(axis=1)

    return pivot


def main():
    print("=" * 70)
    print("E039b: VCS Distance-Based Test")
    print("=" * 70)

    # Load data
    cultures = load_pulotu_pivot()
    print(f"\nPulotu cultures: {len(cultures)}")

    volcanoes = load_gvp_simple()
    if len(volcanoes) == 0:
        print("ABORT: No volcano data available")
        return

    # Compute nearest volcano for each culture
    print(f"\nComputing nearest volcano for {len(cultures)} cultures...")
    nearest = compute_nearest_volcano(cultures, volcanoes)
    cultures = pd.concat([cultures.reset_index(drop=True), nearest], axis=1)

    # Summary
    print(f"\nDistance to nearest active volcano:")
    print(f"  Mean:   {cultures['distance_km'].mean():.0f} km")
    print(f"  Median: {cultures['distance_km'].median():.0f} km")
    print(f"  Min:    {cultures['distance_km'].min():.0f} km ({cultures.loc[cultures['distance_km'].idxmin(), 'Name']})")
    print(f"  Max:    {cultures['distance_km'].max():.0f} km ({cultures.loc[cultures['distance_km'].idxmax(), 'Name']})")

    # Top 10 closest to volcanoes
    print(f"\nTop 10 closest to active volcanoes:")
    closest = cultures.nsmallest(10, 'distance_km')
    for _, row in closest.iterrows():
        print(f"  {row['Name']:<25} {row['distance_km']:>6.0f} km  "
              f"ritual={row['ritual_complexity']:.3f}  "
              f"volcano={row['nearest_volcano']}")

    # Top 10 farthest
    print(f"\nTop 10 farthest from active volcanoes:")
    farthest = cultures.nlargest(10, 'distance_km')
    for _, row in farthest.iterrows():
        print(f"  {row['Name']:<25} {row['distance_km']:>6.0f} km  "
              f"ritual={row['ritual_complexity']:.3f}")

    # === CORRELATION TESTS ===
    valid = cultures.dropna(subset=['distance_km', 'ritual_complexity'])

    print(f"\n{'='*70}")
    print("CORRELATION TESTS (n={})".format(len(valid)))
    print(f"{'='*70}")

    # Test 1: Distance vs ritual complexity (Spearman)
    rho, p = stats.spearmanr(valid['distance_km'], valid['ritual_complexity'])
    print(f"\n--- Test 1: Distance vs Ritual Complexity ---")
    print(f"  Spearman rho = {rho:.3f}, p = {p:.4f}")
    print(f"  VCS predicts: NEGATIVE (closer = more complex)")
    print(f"  Result: {'SUPPORTS VCS' if rho < 0 and p < 0.05 else 'DOES NOT SUPPORT VCS'}")

    # Test 2: Distance vs mortuary index
    valid_mort = cultures.dropna(subset=['distance_km', 'mortuary_index'])
    rho_m, p_m = stats.spearmanr(valid_mort['distance_km'], valid_mort['mortuary_index'])
    print(f"\n--- Test 2: Distance vs Mortuary Index ---")
    print(f"  Spearman rho = {rho_m:.3f}, p = {p_m:.4f}")

    # Test 3: Log-distance (volcanic effects decay exponentially)
    valid['log_distance'] = np.log10(valid['distance_km'] + 1)
    rho_log, p_log = stats.spearmanr(valid['log_distance'], valid['ritual_complexity'])
    print(f"\n--- Test 3: Log-Distance vs Ritual Complexity ---")
    print(f"  Spearman rho = {rho_log:.3f}, p = {p_log:.4f}")

    # Test 4: Nearest eruption count vs ritual complexity
    valid_erupt = cultures.dropna(subset=['nearest_eruptions', 'ritual_complexity'])
    if valid_erupt['nearest_eruptions'].nunique() > 2:
        rho_e, p_e = stats.spearmanr(valid_erupt['nearest_eruptions'], valid_erupt['ritual_complexity'])
        print(f"\n--- Test 4: Nearest Volcano Eruption Count vs Ritual Complexity ---")
        print(f"  Spearman rho = {rho_e:.3f}, p = {p_e:.4f}")
        print(f"  VCS predicts: POSITIVE (more eruptions = more complex)")

    # Test 5: Quartile analysis
    print(f"\n--- Test 5: Distance Quartiles ---")
    valid['dist_quartile'] = pd.qcut(valid['distance_km'], 4, labels=['Q1_closest', 'Q2', 'Q3', 'Q4_farthest'])
    for q, group in valid.groupby('dist_quartile'):
        print(f"  {q}: n={len(group)}, ritual={group['ritual_complexity'].mean():.3f} "
              f"± {group['ritual_complexity'].std():.3f}, "
              f"dist={group['distance_km'].min():.0f}-{group['distance_km'].max():.0f} km")

    # Kruskal-Wallis across quartiles
    groups = [g['ritual_complexity'].values for _, g in valid.groupby('dist_quartile')]
    h_stat, p_kw = stats.kruskal(*groups)
    print(f"  Kruskal-Wallis: H={h_stat:.2f}, p={p_kw:.4f}")

    # Test 6: Indonesian subset only
    print(f"\n--- Test 6: Indonesian/Melanesian Subset ---")
    indo_mel = valid[(valid['Latitude'].between(-15, 10)) & (valid['Longitude'].between(95, 180))]
    if len(indo_mel) > 10:
        rho_im, p_im = stats.spearmanr(indo_mel['distance_km'], indo_mel['ritual_complexity'])
        print(f"  n={len(indo_mel)}")
        print(f"  Spearman rho = {rho_im:.3f}, p = {p_im:.4f}")
        print(f"  Mean distance: {indo_mel['distance_km'].mean():.0f} km")
        print(f"  Mean ritual: {indo_mel['ritual_complexity'].mean():.3f}")

    # === VERDICT ===
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    if rho < -0.15 and p < 0.05:
        verdict = "SUCCESS"
        explanation = f"Proximity to active volcanoes correlates with ritual complexity (rho={rho:.3f}, p={p:.4f})"
    elif rho < 0 and p < 0.10:
        verdict = "SUGGESTIVE"
        explanation = f"Weak trend: closer to volcano = more ritual (rho={rho:.3f}, p={p:.4f})"
    else:
        verdict = "NOT SIGNIFICANT"
        explanation = f"No correlation between volcano proximity and ritual complexity (rho={rho:.3f}, p={p:.4f})"

    print(f"\n  >>> {verdict}")
    print(f"  >>> {explanation}")

    # Save
    results = {
        "test1_distance_ritual": {"rho": round(rho, 4), "p": round(p, 4)},
        "test2_distance_mortuary": {"rho": round(rho_m, 4), "p": round(p_m, 4)},
        "test3_logdist_ritual": {"rho": round(rho_log, 4), "p": round(p_log, 4)},
        "verdict": verdict,
        "n_cultures": len(valid),
        "n_volcanoes": len(volcanoes),
    }
    with open(OUT / "vcs_distance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT / 'vcs_distance_results.json'}")

    # Save enriched culture data
    export_cols = ["Name", "Latitude", "Longitude", "ritual_complexity", "mortuary_index",
                   "distance_km", "nearest_volcano", "nearest_eruptions"]
    export_cols = [c for c in export_cols if c in cultures.columns]
    cultures[export_cols].to_csv(OUT / "culture_volcano_distances.csv", index=False)
    print(f"Saved: {OUT / 'culture_volcano_distances.csv'}")


if __name__ == "__main__":
    main()
