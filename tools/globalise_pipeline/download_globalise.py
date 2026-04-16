"""
GLOBALISE VOC Corpus Downloader
================================
Downloads TXT transcriptions from GLOBALISE Dataverse (CC0 license).
Uses Dataverse API to list and download files by inventory number.

Usage:
    python download_globalise.py --n 50 --output data/raw/globalise_voc/
    python download_globalise.py --range 1053-1100 --output data/raw/globalise_voc/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system("pip install requests -q")
    import requests

DATAVERSE_API = "https://datasets.iisg.amsterdam/api"
DATASET_PID = "hdl:10622/LVXSBW"
INDEX_CACHE = "globalise_file_index.json"


def get_file_index(cache_dir):
    """Fetch or load cached file index from Dataverse API."""
    cache_path = Path(cache_dir) / INDEX_CACHE

    if cache_path.exists():
        with open(cache_path) as f:
            index = json.load(f)
        print(f"Loaded cached index: {len(index)} files")
        return index

    print("Fetching file index from GLOBALISE Dataverse (this may take a minute)...")
    url = f"{DATAVERSE_API}/datasets/:persistentId?persistentId={DATASET_PID}"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    files = data["data"]["latestVersion"]["files"]

    # Build index: filename -> file_id (TXT files only)
    index = {}
    for f in files:
        df = f["dataFile"]
        name = df.get("filename", "")
        if name.endswith(".txt"):
            inv_num = name.replace(".txt", "")
            index[inv_num] = {
                "file_id": df["id"],
                "filename": name,
                "size": df.get("filesize", 0)
            }

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Indexed {len(index)} TXT files. Cached to {cache_path}")
    return index


def download_file(file_id, output_path, retries=3):
    """Download a single file from Dataverse."""
    url = f"{DATAVERSE_API}/access/datafile/{file_id}"

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAILED after {retries} attempts: {e}")
                return False
    return False


def download_batch(index, output_dir, n=None, inv_range=None):
    """Download a batch of VOC transcription files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select files to download
    inv_numbers = sorted(index.keys(), key=lambda x: int(x) if x.isdigit() else 0)

    if inv_range:
        start, end = inv_range.split("-")
        inv_numbers = [i for i in inv_numbers if int(start) <= int(i) <= int(end)]

    if n:
        inv_numbers = inv_numbers[:n]

    print(f"\nDownloading {len(inv_numbers)} VOC transcription files...")

    downloaded = 0
    skipped = 0
    failed = 0
    total_size = 0

    for i, inv in enumerate(inv_numbers):
        info = index[inv]
        out_path = output_dir / info["filename"]

        # Skip if already downloaded
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue

        success = download_file(info["file_id"], out_path)
        if success:
            downloaded += 1
            total_size += out_path.stat().st_size
        else:
            failed += 1

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(inv_numbers)} (downloaded={downloaded}, skipped={skipped}, failed={failed})")

        # Rate limiting: be nice to the server
        time.sleep(0.5)

    print(f"\nDone. Downloaded={downloaded}, Skipped={skipped}, Failed={failed}")
    print(f"Total new data: {total_size / 1024 / 1024:.1f} MB")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download GLOBALISE VOC transcriptions")
    parser.add_argument("--n", type=int, default=50, help="Number of files to download")
    parser.add_argument("--range", type=str, help="Inventory number range (e.g., 1053-1100)")
    parser.add_argument("--output", type=str, default="data/raw/globalise_voc/", help="Output directory")
    parser.add_argument("--cache-dir", type=str, default="tools/globalise_pipeline/", help="Cache directory for file index")
    args = parser.parse_args()

    index = get_file_index(args.cache_dir)
    download_batch(index, args.output, n=args.n, inv_range=args.range)


if __name__ == "__main__":
    main()
