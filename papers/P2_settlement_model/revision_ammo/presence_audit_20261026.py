# Ledger C081 (2026-10-26). Usage: python presence_audit_20261026.py OUTDIR  -> OUTDIR/p2_presence_audit.csv + counts on stdout.
"""Which of P2's 378 presences are what? Reproduces E217's presence filter (identical to E013's clip) and classifies
the retained rows. Read-only on the repo."""
import importlib.util, re, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(r"D:\documents\volcarch-repo")
spec = importlib.util.spec_from_file_location("e217", REPO / "experiments/E217_maxent_benchmark/01_maxent_benchmark.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

sites = m.load_sites()
xy = np.column_stack([sites.geometry.x, sites.geometry.y])
feat = pd.DataFrame({c: m.sample_at_points(m.DEM_DIR / m.RASTER_FILES[c], xy) for c in m.FULL_COLS})
s = sites.reset_index(drop=True).copy()
for c in m.FULL_COLS:
    s[c] = feat[c].values
s = s.dropna(subset=m.FULL_COLS)
s = s[s["elevation"] > 0].reset_index(drop=True)
ll = s.to_crs("EPSG:4326")
s["lon"], s["lat"] = ll.geometry.x.round(5), ll.geometry.y.round(5)
print("presences reproduced:", len(s))

MODERN = re.compile(
    r"\b(tugu|monumen|monument|patung|museum|makam pahlawan|taman makam|memorial|stasiun|gedung|"
    r"jembatan|alun|lapangan|kantor|sekolah|masjid|gereja|klenteng|vihara|benteng|pabrik|"
    r"gelanggang|stadion|taman)\b", re.I)
ELSEWHERE = re.compile(r"borobudur|prambanan|besakih|mendut|pawon|plaosan|sewu|kalasan|sambisari|ratu boko|"
                       r"dieng|gedong songo|tanah lot|uluwatu|ulun danu|tirta empul|gilimanuk|bali barat|muara takus|"
                       r"padang lawas|jiwa|batujaya|cangkuang", re.I)
name = s["name"].fillna("")
s["t_monument"] = s["type"].eq("monument")
s["n_modern"] = name.str.contains(MODERN)
s["n_elsewhere"] = name.str.contains(ELSEWHERE)
dup = s.groupby(["lon", "lat"])["name"].transform("size")
s["coord_dup"] = dup > 1
s["any_flag"] = s.t_monument | s.n_modern | s.n_elsewhere

print("type counts:", s["type"].value_counts().to_dict())
print("source counts:", s["source"].str.slice(0, 30).value_counts().to_dict())
print("type == monument:", int(s.t_monument.sum()))
print("modern-name regex (C029 list):", int(s.n_modern.sum()))
print("monument OR modern name:", int((s.t_monument | s.n_modern).sum()))
print("named temples of other regions (mis-geocoded or outside East Java):", int(s.n_elsewhere.sum()))
print("rows sharing an exact coordinate with another row:", int(s.coord_dup.sum()), "| distinct coordinates:", s.groupby(["lon", "lat"]).ngroups)
print("ANY flag (monument / modern name / elsewhere):", int(s.any_flag.sum()), "| left:", int((~s.any_flag).sum()))
clean = s[~s.any_flag].drop_duplicates(["lon", "lat"])
print("left after flags AND one row per coordinate:", len(clean))
print("\nexamples, monument or modern name:")
print(s[s.t_monument | s.n_modern][["name", "type", "lon", "lat"]].head(25).to_string())
print("\nexamples, elsewhere:")
print(s[s.n_elsewhere][["name", "type", "lon", "lat"]].to_string())
print("\nunflagged type=unknown sample:")
print(s[(~s.any_flag) & s["type"].eq("unknown")][["name", "source", "lon", "lat"]].sample(20, random_state=1).to_string())
s.drop(columns="geometry").to_csv(Path(sys.argv[1]) / "p2_presence_audit.csv", index=False)

# Monument-typed rows include a few real temples (e.g. Candi Jawi is tagged monument in OSM): split by name.
anc = name.str.contains("candi|situs|arca|prasasti|petirtaan|gapura|punden|yoni|lingga|reruntuhan|pertapaan|goa|gua|"
                        "sumur|kolam segaran|makam troloyo|trowulan|watu|batu", case=False)
flag = s.t_monument | s.n_modern
print("\nmonument/modern-name rows:", int(flag.sum()), "| name looks ancient:", int((flag & anc).sum()),
      "| name does not (modern):", int((flag & ~anc).sum()))
