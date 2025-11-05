# built_vrt.py
"""
Find ArcticDEM STRIP tiles that overlap a Sentinel-2 tile and
write a URL list + a small fetch script to download, extract, and build a VRT.

# Automation done with the help of LLM (ChatGPT 5)

Usage:
  python built_vrt.py \
    --index "/path/to/ArcticDEM_Strip_Index.shp" \
    --out-dir "/path/to/data/raw/ArcticDEM" \
    --sentinel-bounds 399960 7490220 509760 7600020 \
    --sentinel-crs EPSG:32622 \
    --buffer-m 1000 \
    --resolution 10m  # or 2m

Requires: geopandas, shapely, GDAL in PATH (for gdalbuildvrt)
"""

import argparse
import os
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


# ---- small helpers (keep it simple) -----------------------------------------

def pick_url_column(df: gpd.GeoDataFrame) -> str:
    """Try a few common column names. If unsure, guess the first that looks like URLs."""
    candidates = ["fileurl", "url", "downloadurl", "dem_url", "FILEURL", "FileURL", "file_url", "href"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first column with at least one value starting with http
    for c in df.columns:
        s = df[c].astype(str)
        if s.str.startswith("http").any():
            return c
    raise KeyError("Couldn't find a URL column in the index file.")


def rewrite_res(url: str, target: str) -> str:
    """Swap 2m <-> 10m tokens in the path/filename. If nothing matches, return as-is."""
    if target not in ("2m", "10m"):
        return url
    other = "10m" if target == "2m" else "2m"
    return url.replace(f"/{other}/", f"/{target}/").replace(f"_{other}_", f"_{target}_")


def write_fetch_script(out_dir: Path, urls_txt: Path, tiles_dir: Path, vrt_path: Path) -> Path:
    """Very basic downloader/extractor + VRT builder."""
    script = f"""#!/bin/bash
set -euo pipefail

URLS="{urls_txt}"
TILES="{tiles_dir}"
VRT="{vrt_path}"

mkdir -p "$TILES"
cd "$TILES"

echo "Downloading ArcticDEM archives..."
while IFS= read -r url; do
  [[ -z "$url" ]] && continue
  fname="$(basename "$url")"
  echo "-> $fname"
  curl -C - -fL --progress-bar -o "$fname" "$url"
  tar -xzvf "$fname"
  rm -f "$fname"
done < "$URLS"

echo "Building VRT from *_dem.tif files..."
gdalbuildvrt -overwrite "$VRT" "$TILES"/*_dem.tif

echo "Done. VRT: $VRT"
"""
    sh_path = out_dir / "fetch-dem.sh"
    sh_path.write_text(script)
    os.chmod(sh_path, 0o755)
    return sh_path


# ---- main -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="ArcticDEM STRIP index (shp/gpkg)")
    p.add_argument("--out-dir", required=True, help="Output dir, e.g. .../data/raw/ArcticDEM")
    p.add_argument("--sentinel-bounds", nargs=4, type=float, metavar=("MINX","MINY","MAXX","MAXY"),
                   required=True, help="Sentinel AOI bounds in the given CRS")
    p.add_argument("--sentinel-crs", default="EPSG:32622", help="CRS of the bounds (default EPSG:32622)")
    p.add_argument("--buffer-m", type=float, default=1000.0, help="Buffer around AOI (meters)")
    p.add_argument("--resolution", choices=["2m","10m"], default="10m", help="DEM resolution to request")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    tiles_dir = out_dir / "tiles"
    vrt_path  = out_dir / "arcticdem_mosaic.vrt"

    # Build AOI polygon (+ buffer)
    minx, miny, maxx, maxy = args.sentinel_bounds
    aoi = box(minx, miny, maxx, maxy).buffer(args.buffer_m)

    # Read index and project to Sentinel CRS if needed
    idx = gpd.read_file(args.index)
    if str(idx.crs) != args.sentinel_crs:
        idx = idx.to_crs(args.sentinel_crs)

    # Select overlapping DEM strips
    sel = idx[idx.intersects(aoi)].copy()
    print(f"Found {len(sel)} overlapping strips.")
    if sel.empty:
        raise SystemExit("No overlap. Check CRS/bounds or increase --buffer-m.")

    url_col = pick_url_column(sel)
    urls = sel[url_col].astype(str).dropna().drop_duplicates().tolist()
    urls = [rewrite_res(u, args.resolution) for u in urls]

    # Write URL list and fetch script
    out_dir.mkdir(parents=True, exist_ok=True)
    urls_txt = out_dir / "dem_urls.txt"
    urls_txt.write_text("\n".join(urls) + "\n")

    fetch_sh = write_fetch_script(out_dir=out_dir, urls_txt=urls_txt,
                                  tiles_dir=tiles_dir, vrt_path=vrt_path)

    print(f"Wrote: {urls_txt}")
    print(f"Fetch script: {fetch_sh}")
    print(f"Run:\n  bash {fetch_sh}")


if __name__ == "__main__":
    main()