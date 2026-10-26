"""E236: the cleaned presence set (rule frozen in DESIGN.md section 3) and a patch that makes every E217-based
script read it instead of the canonical file.

Why a patch of geopandas.read_file and not of `load_sites`: E217b, E218, E218b, E221 and E235 each load their own
copy of the E217 base module (importlib, under several names), so there is no single function to replace. All of
them read the site file through `gpd.read_file(SITES_PATH)`, so intercepting that one call reaches every copy and
leaves the original scripts and their results untouched.
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd

REPO = Path(__file__).resolve().parents[2]
SITES = (REPO / "data" / "processed" / "east_java_sites.geojson").resolve()

# R2: monument-typed or modern-named rows, unless the name is that of an ancient feature (C029 list + C081).
MODERN = re.compile(
    r"\b(tugu|monumen|monument|patung|museum|makam pahlawan|taman makam|memorial|stasiun|gedung|"
    r"jembatan|alun|lapangan|kantor|sekolah|masjid|gereja|klenteng|vihara|benteng|pabrik|"
    r"gelanggang|stadion|taman)\b", re.I)
ANCIENT = re.compile(r"candi|situs|arca|prasasti|petirtaan|gapura|punden|yoni|lingga|reruntuhan|pertapaan|goa|gua|"
                     r"sumur|kolam segaran|makam troloyo|trowulan|watu|batu", re.I)
# R3: temples of other regions, tourist objects, modern institutions.
ELSEWHERE = re.compile(r"borobudur|prambanan|besakih|mendut|pawon|plaosan|sewu|kalasan|sambisari|ratu boko|"
                       r"dieng|gedong songo|tanah lot|uluwatu|ulun danu|tirta empul|gilimanuk|bali barat|muara takus|"
                       r"padang lawas|jiwa|batujaya|cangkuang", re.I)
OTHER = re.compile(r"^pura\b|ponpes|pabriek", re.I)


def clean(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    name = gdf["name"].fillna("")
    r1 = gdf["accuracy_level"].eq("nominatim")                                   # coordinates geocoded from a name
    r2 = (gdf["type"].eq("monument") | name.str.contains(MODERN)) & ~name.str.contains(ANCIENT)
    r3 = name.str.contains(ELSEWHERE) | gdf["type"].eq("objek_wisata") | name.str.contains(OTHER)
    keep = gdf[~(r1 | r2 | r3)].copy()
    ll = keep.geometry.to_crs("EPSG:4326")
    key = list(zip(ll.x.round(5), ll.y.round(5)))
    keep = keep[~keep.assign(_k=key)["_k"].duplicated().to_numpy()]            # R4: one row per coordinate
    return keep


_original_read_file = gpd.read_file


def _read_file(path, *args, **kwargs):
    gdf = _original_read_file(path, *args, **kwargs)
    try:
        same = Path(path).resolve() == SITES
    except TypeError:
        same = False
    return clean(gdf) if same else gdf


def patch() -> None:
    gpd.read_file = _read_file


if __name__ == "__main__":
    full = _original_read_file(SITES)
    c = clean(full)
    print(f"canonical rows {len(full)} -> cleaned rows {len(c)} (before the study-area clip and raster sampling)")
