#!/usr/bin/env python3
"""
Download SRTM30_PLUS bathymetry data at full resolution for Sunda Shelf.
SRTM30_PLUS: ~1km (30 arc-second) resolution, based on satellite altimetry + ship soundings.
Source: Becker et al. (2009), ERDDAP hosted by NOAA CoastWatch.
"""
import sys, io, os, requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUTDIR = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(OUTDIR, 'data_srtm30plus_full.nc')

if os.path.exists(OUTFILE) and os.path.getsize(OUTFILE) > 1_000_000:
    print(f"Data already exists: {OUTFILE} ({os.path.getsize(OUTFILE):,} bytes)")
    print("Skipping download.")
    sys.exit(0)

# Full resolution (stride=1) for Sunda Shelf region
# 95E to 120E, 10S to 10N
# At 30 arc-sec: 25*120=3000 lon points, 20*120=2400 lat points
# ~28.8 MB estimated

query_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/srtm30plus.nc?z[(-10):1:(10)][(95):1:(120)]"
print(f"Downloading SRTM30_PLUS full resolution...")
print(f"URL: {query_url[:80]}...")

try:
    r = requests.get(query_url, timeout=300, stream=True)
    ct = r.headers.get('Content-Type', '?')
    cl = r.headers.get('Content-Length', '?')
    print(f"Status: {r.status_code}, Type: {ct}, Expected size: {cl}")

    if r.status_code == 200:
        total = 0
        with open(OUTFILE, 'wb') as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)
                if total % 5_000_000 < 65536:
                    print(f"  {total:,} bytes downloaded...")
        print(f"Download complete: {total:,} bytes")

        # Verify
        import xarray as xr
        ds = xr.open_dataset(OUTFILE, engine='scipy')
        print(f"Variables: {list(ds.data_vars)}")
        print(f"Dimensions: {dict(ds.dims)}")
        z = ds['z'].values
        print(f"Shape: {z.shape}")
        print(f"Depth range: {z.min():.0f}m to {z.max():.0f}m")
        print(f"Latitude: {ds.latitude.values[0]:.2f} to {ds.latitude.values[-1]:.2f}")
        print(f"Longitude: {ds.longitude.values[0]:.2f} to {ds.longitude.values[-1]:.2f}")

        # Count shelf cells
        shelf_cells = ((z <= 0) & (z >= -120)).sum()
        total_cells = z.size
        print(f"Shelf cells (0 to -120m): {shelf_cells:,} / {total_cells:,} ({100*shelf_cells/total_cells:.1f}%)")
        ds.close()
        print("SUCCESS!")
    else:
        print(f"Failed: {r.text[:300]}")
except Exception as e:
    print(f"Error: {e}")
