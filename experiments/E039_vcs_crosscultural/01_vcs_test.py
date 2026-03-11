"""
E039: Volcanic Cultural Selection — Cross-Cultural Test
========================================================
Hypothesis: Austronesian cultures on volcanic islands show higher
ritual complexity than those on non-volcanic islands.

Data: Pulotu (137 cultures × 86 variables) + island type classification.
Method: Compare ritual complexity index between volcanic vs non-volcanic cultures.

Tests:
  1. Island type (Q32: volcanic high island vs others) → ritual complexity
  2. Mortuary ritual elaboration (Q4+Q5+Q10) → volcanic vs non-volcanic
  3. Communal ritual investment (Q35: costly sacrifices) → volcanic vs non-volcanic
  4. Pre-Austronesian contact (Q22) as potential confound
"""
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

PULOTU = Path("D:/documents/volcarch-repo/experiments/E023_ritual_screening/data/pulotu/cldf")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def load_pulotu():
    """Load and merge Pulotu CLDF tables."""
    cultures = pd.read_csv(PULOTU / "cultures.csv", encoding="utf-8")
    questions = pd.read_csv(PULOTU / "questions.csv", encoding="utf-8")
    codes = pd.read_csv(PULOTU / "codes.csv", encoding="utf-8")
    responses = pd.read_csv(PULOTU / "responses.csv", encoding="utf-8")

    print(f"Cultures: {len(cultures)}")
    print(f"Questions: {len(questions)}")
    print(f"Responses: {len(responses)}")

    return cultures, questions, codes, responses


def pivot_responses(cultures, responses):
    """Create culture × question matrix with numeric codes."""
    # responses columns: ID, Language_ID, Parameter_ID, Value, Code_ID, Comment, Source
    # cultures columns: ID, Name, Macroarea, Latitude, Longitude, ...
    # Language_ID in responses = ID in cultures

    merged = responses.merge(cultures[["ID", "Name", "Latitude", "Longitude"]],
                             left_on="Language_ID", right_on="ID", how="left",
                             suffixes=("_resp", "_cult"))

    # Code_ID format: "QNN-N" where last N is the code value
    merged["code_value"] = merged["Code_ID"].apply(
        lambda x: int(str(x).split("-")[-1]) if pd.notna(x) and "-" in str(x) else np.nan
    )

    # For coded questions (Q2-Q86), use code_value
    # For numeric questions (lat/lon etc.), try to parse Value as float
    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    merged["numeric_value"] = merged["code_value"].where(
        merged["code_value"].notna(),
        merged["Value"].apply(safe_float)
    )

    pivot = merged.pivot_table(
        index=["Language_ID", "Name", "Latitude", "Longitude"],
        columns="Parameter_ID",
        values="numeric_value",
        aggfunc="first"
    ).reset_index()

    # Rename Language_ID to Culture_ID for clarity
    pivot = pivot.rename(columns={"Language_ID": "Culture_ID"})
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot


def classify_volcanic(df):
    """Classify cultures as volcanic or non-volcanic using Q32 (island type)."""
    # Column "32" = island type: 1=raised atoll/limestone, 2=volcanic high island,
    # 3=continental island, 4=mainland/continent
    col = "32"
    if col in df.columns:
        df["is_volcanic"] = (df[col] == 2).astype(int)
        df["island_type"] = df[col].map({
            0: "atoll", 1: "raised_atoll", 2: "volcanic_high",
            3: "continental", 4: "mainland"
        })
    else:
        print("WARNING: Column 32 (island type) not found.")
        df["is_volcanic"] = np.nan

    return df


def compute_ritual_indices(df):
    """Compute ritual complexity indices from Pulotu variables."""
    # Broad ritual complexity: sum of supernatural belief + ritual practice variables
    ritual_vars = ["2", "3", "4", "5", "6", "7", "10", "11", "12", "21", "35"]
    available = [v for v in ritual_vars if v in df.columns]
    df["ritual_complexity"] = df[available].mean(axis=1)  # mean to handle missing

    # Mortuary-specific: ancestor belief + post-death action efficacy
    mort_vars = ["4", "5", "10"]
    available_mort = [v for v in mort_vars if v in df.columns]
    df["mortuary_index"] = df[available_mort].mean(axis=1)

    # Communal investment: costly sacrifices
    if "35" in df.columns:
        df["sacrifice_level"] = df["35"]

    # Supernatural punishment (social enforcement)
    if "7" in df.columns:
        df["punishment_level"] = df["7"]

    # Pre-Austronesian contact (confound)
    if "22" in df.columns:
        df["pre_austronesian"] = df["22"]

    return df, available, available_mort


def run_tests(df):
    """Run statistical tests comparing volcanic vs non-volcanic cultures."""
    results = {}

    volcanic = df[df["is_volcanic"] == 1].dropna(subset=["ritual_complexity"])
    non_volcanic = df[df["is_volcanic"] == 0].dropna(subset=["ritual_complexity"])

    print(f"\n{'='*70}")
    print(f"VOLCANIC CULTURES: n={len(volcanic)}")
    print(f"NON-VOLCANIC CULTURES: n={len(non_volcanic)}")
    print(f"{'='*70}")

    # Show island type distribution
    print(f"\nIsland type distribution:")
    for itype, count in df["island_type"].value_counts().items():
        print(f"  {itype}: {count}")

    # Test 1: Broad ritual complexity
    print(f"\n--- Test 1: Broad Ritual Complexity ---")
    v_mean = volcanic["ritual_complexity"].mean()
    nv_mean = non_volcanic["ritual_complexity"].mean()
    t_stat, p_val = stats.mannwhitneyu(
        volcanic["ritual_complexity"].dropna(),
        non_volcanic["ritual_complexity"].dropna(),
        alternative="greater"
    )
    d = (v_mean - nv_mean) / np.sqrt(
        (volcanic["ritual_complexity"].std()**2 + non_volcanic["ritual_complexity"].std()**2) / 2
    )
    print(f"  Volcanic mean:     {v_mean:.3f} (n={len(volcanic)})")
    print(f"  Non-volcanic mean: {nv_mean:.3f} (n={len(non_volcanic)})")
    print(f"  Mann-Whitney U p:  {p_val:.4f} (one-tailed: volcanic > non-volcanic)")
    print(f"  Cohen's d:         {d:.3f}")
    results["ritual_complexity"] = {
        "volcanic_mean": round(v_mean, 3), "nonvolcanic_mean": round(nv_mean, 3),
        "p_value": round(p_val, 4), "cohens_d": round(d, 3),
        "n_volcanic": len(volcanic), "n_nonvolcanic": len(non_volcanic)
    }

    # Test 2: Mortuary-specific
    print(f"\n--- Test 2: Mortuary Ritual Index (Q4+Q5+Q10) ---")
    v_mort = volcanic["mortuary_index"].dropna()
    nv_mort = non_volcanic["mortuary_index"].dropna()
    if len(v_mort) > 2 and len(nv_mort) > 2:
        v_mm = v_mort.mean()
        nv_mm = nv_mort.mean()
        _, p_mort = stats.mannwhitneyu(v_mort, nv_mort, alternative="greater")
        d_mort = (v_mm - nv_mm) / np.sqrt((v_mort.std()**2 + nv_mort.std()**2) / 2)
        print(f"  Volcanic mean:     {v_mm:.3f} (n={len(v_mort)})")
        print(f"  Non-volcanic mean: {nv_mm:.3f} (n={len(nv_mort)})")
        print(f"  Mann-Whitney U p:  {p_mort:.4f}")
        print(f"  Cohen's d:         {d_mort:.3f}")
        results["mortuary_index"] = {
            "volcanic_mean": round(v_mm, 3), "nonvolcanic_mean": round(nv_mm, 3),
            "p_value": round(p_mort, 4), "cohens_d": round(d_mort, 3)
        }

    # Test 3: Costly sacrifices (Q35)
    print(f"\n--- Test 3: Costly Sacrifices (Q35) ---")
    if "sacrifice_level" in df.columns:
        v_sac = volcanic["sacrifice_level"].dropna()
        nv_sac = non_volcanic["sacrifice_level"].dropna()
        if len(v_sac) > 2 and len(nv_sac) > 2:
            _, p_sac = stats.mannwhitneyu(v_sac, nv_sac, alternative="greater")
            print(f"  Volcanic mean:     {v_sac.mean():.3f} (n={len(v_sac)})")
            print(f"  Non-volcanic mean: {nv_sac.mean():.3f} (n={len(nv_sac)})")
            print(f"  Mann-Whitney U p:  {p_sac:.4f}")
            results["costly_sacrifices"] = {
                "volcanic_mean": round(v_sac.mean(), 3),
                "nonvolcanic_mean": round(nv_sac.mean(), 3),
                "p_value": round(p_sac, 4)
            }

    # Test 4: Supernatural punishment (Q7) — social enforcement mechanism
    print(f"\n--- Test 4: Supernatural Punishment (Q7) ---")
    if "punishment_level" in df.columns:
        v_pun = volcanic["punishment_level"].dropna()
        nv_pun = non_volcanic["punishment_level"].dropna()
        if len(v_pun) > 2 and len(nv_pun) > 2:
            _, p_pun = stats.mannwhitneyu(v_pun, nv_pun, alternative="greater")
            print(f"  Volcanic mean:     {v_pun.mean():.3f} (n={len(v_pun)})")
            print(f"  Non-volcanic mean: {nv_pun.mean():.3f} (n={len(nv_pun)})")
            print(f"  Mann-Whitney U p:  {p_pun:.4f}")
            results["punishment"] = {
                "volcanic_mean": round(v_pun.mean(), 3),
                "nonvolcanic_mean": round(nv_pun.mean(), 3),
                "p_value": round(p_pun, 4)
            }

    # Confound check: Pre-Austronesian contact
    print(f"\n--- Confound Check: Pre-Austronesian Contact (Q22) ---")
    if "pre_austronesian" in df.columns:
        v_pa = volcanic["pre_austronesian"].dropna()
        nv_pa = non_volcanic["pre_austronesian"].dropna()
        if len(v_pa) > 2 and len(nv_pa) > 2:
            _, p_pa = stats.mannwhitneyu(v_pa, nv_pa, alternative="two-sided")
            print(f"  Volcanic mean:     {v_pa.mean():.3f}")
            print(f"  Non-volcanic mean: {nv_pa.mean():.3f}")
            print(f"  Mann-Whitney U p:  {p_pa:.4f} (two-sided)")
            results["confound_pre_austronesian"] = {
                "volcanic_mean": round(v_pa.mean(), 3),
                "nonvolcanic_mean": round(nv_pa.mean(), 3),
                "p_value": round(p_pa, 4)
            }

    # Per-variable breakdown for volcanic vs non-volcanic
    print(f"\n--- Per-Variable Breakdown ---")
    print(f"  {'Variable':<8} {'Volcanic':<12} {'Non-Volcanic':<14} {'Delta':<8} {'p':<8}")
    for var in ["2", "3", "4", "5", "6", "7", "10", "11", "12", "21", "35"]:
        if var in df.columns:
            v_val = volcanic[var].dropna()
            nv_val = non_volcanic[var].dropna()
            if len(v_val) > 2 and len(nv_val) > 2:
                _, p = stats.mannwhitneyu(v_val, nv_val, alternative="greater")
                delta = v_val.mean() - nv_val.mean()
                sig = "*" if p < 0.05 else ""
                print(f"  {var:<8} {v_val.mean():<12.3f} {nv_val.mean():<14.3f} {delta:<+8.3f} {p:<8.4f} {sig}")

    return results


def malagasy_analysis(df):
    """Specific analysis of Malagasy cultures as VCS control group."""
    print(f"\n{'='*70}")
    print(f"MALAGASY CONTROL GROUP ANALYSIS")
    print(f"{'='*70}")

    # Find Malagasy cultures
    malagasy = df[df["Name"].str.contains("Merina|Tanala|Malagasy|Sakalava|Betsileo|Bara",
                                           case=False, na=False)]
    if len(malagasy) == 0:
        # Try broader search
        malagasy = df[(df["Latitude"].between(-25, -12)) & (df["Longitude"].between(43, 51))]

    if len(malagasy) == 0:
        print("  No Malagasy cultures found in dataset.")
        return {}

    print(f"  Found {len(malagasy)} Malagasy culture(s):")
    for _, row in malagasy.iterrows():
        print(f"    {row['Name']} (lat={row['Latitude']}, lon={row['Longitude']})")
        print(f"      Ritual complexity: {row.get('ritual_complexity', 'N/A')}")
        print(f"      Mortuary index:    {row.get('mortuary_index', 'N/A')}")
        print(f"      Island type:       {row.get('island_type', 'N/A')}")

    # Compare Malagasy with volcanic cultures
    volcanic = df[df["is_volcanic"] == 1]
    if len(volcanic) > 0 and len(malagasy) > 0:
        print(f"\n  Malagasy vs Volcanic Austronesian cultures:")
        for var in ["ritual_complexity", "mortuary_index"]:
            if var in df.columns:
                v_mean = volcanic[var].mean()
                m_mean = malagasy[var].mean()
                print(f"    {var}: Malagasy={m_mean:.3f}, Volcanic={v_mean:.3f}, delta={m_mean-v_mean:+.3f}")

    # Find Indonesian cultures for comparison
    indonesian = df[df["Name"].str.contains("Toraja|Java|Bali|Batak|Nias|Minangkabau|Bugis",
                                             case=False, na=False)]
    if len(indonesian) > 0:
        print(f"\n  Indonesian cultures in dataset:")
        for _, row in indonesian.iterrows():
            print(f"    {row['Name']}: ritual={row.get('ritual_complexity', 'N/A'):.3f}, "
                  f"mortuary={row.get('mortuary_index', 'N/A'):.3f}, "
                  f"type={row.get('island_type', 'N/A')}")

    return {"n_malagasy": len(malagasy)}


def main():
    print("=" * 70)
    print("E039: Volcanic Cultural Selection — Cross-Cultural Test")
    print("=" * 70)

    cultures, questions, codes, responses = load_pulotu()

    # Build culture × question matrix
    df = pivot_responses(cultures, responses)
    print(f"\nPivot table: {df.shape}")

    # Classify volcanic vs non-volcanic
    df = classify_volcanic(df)

    # Compute ritual indices
    df, ritual_vars, mort_vars = compute_ritual_indices(df)

    # Summary stats
    print(f"\nRitual variables used: {ritual_vars}")
    print(f"Mortuary variables used: {mort_vars}")
    print(f"Volcanic cultures (Q32=2): {(df['is_volcanic']==1).sum()}")
    print(f"Non-volcanic cultures: {(df['is_volcanic']==0).sum()}")
    print(f"Missing Q32: {df['is_volcanic'].isna().sum()}")

    # Run main tests
    results = run_tests(df)

    # Malagasy control
    malagasy_results = malagasy_analysis(df)
    results["malagasy"] = malagasy_results

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    rc = results.get("ritual_complexity", {})
    p_rc = rc.get("p_value", 1.0)
    d_rc = rc.get("cohens_d", 0)

    if p_rc < 0.05 and d_rc > 0.3:
        verdict = "SUCCESS"
        explanation = f"Volcanic cultures show significantly higher ritual complexity (p={p_rc}, d={d_rc})"
    elif p_rc < 0.10:
        verdict = "SUGGESTIVE"
        explanation = f"Trend toward higher ritual complexity in volcanic cultures (p={p_rc}, d={d_rc})"
    else:
        verdict = "NOT SIGNIFICANT"
        explanation = f"No significant difference in ritual complexity (p={p_rc}, d={d_rc})"

    print(f"\n  >>> {verdict}")
    print(f"  >>> {explanation}")
    results["verdict"] = verdict
    results["explanation"] = explanation

    # Save
    with open(OUT / "vcs_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT / 'vcs_test_results.json'}")

    # Save per-culture data for inspection
    export_cols = ["Culture_ID", "Name", "Latitude", "Longitude", "island_type",
                   "is_volcanic", "ritual_complexity", "mortuary_index"]
    export_cols = [c for c in export_cols if c in df.columns]
    df[export_cols].to_csv(OUT / "culture_ritual_scores.csv", index=False)
    print(f"Saved: {OUT / 'culture_ritual_scores.csv'}")


if __name__ == "__main__":
    main()
