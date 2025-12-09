"""
File: build_vrt.py
Purpose: Identify overlapping ArcticDEM STRIP tiles, generate download lists,
         create a fetch script, and assemble DEM tiles into a GDAL VRT mosaic.
         Forms the DEM foundation for alignment with Sentinel-2 imagery.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box

# -------------- helpers --------------

def detect_url_column(df: gpd.GeoDataFrame) -> str:
    for c in ["fileurl", "url", "downloadurl", "dem_url", "FILEURL", "FileURL", "file_url", "href"]:
        if c in df.columns:
            return c
    # last resort: any column that starts with http for some rows
    for c in df.columns:
        try:
            if df[c].astype(str).str.startswith("http").any():
                return c
        except Exception:
            pass
    raise KeyError("No URL column found in index. Inspect columns and update detect_url_column().")

def detect_id_column(df: gpd.GeoDataFrame) -> str:
    for c in ["tile", "strip", "name", "dem_id", "strip_id", "ProductID"]:
        if c in df.columns:
            return c
    return df.columns[0]

def detect_date_column(df: gpd.GeoDataFrame) -> str | None:
    """
    Try to find an acquisition date column for secondary sort.
    Returns column name or None.
    """
    candidates = [
        "acqdate", "acq_date", "acquisition_da", "acquisition", "date",
        "ACQDATE", "Date", "DATE"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: any column that looks datetime-like for at least some rows
    for c in df.columns:
        if "date" in c.lower() or "acq" in c.lower():
            return c
    return None

def rewrite_resolution(url: str, target_res: str) -> str:
    """
    Switch between 2m and 10m URL variants by editing path and filename tokens.
    """
    if target_res not in ("2m", "10m"):
        return url
    other = "10m" if target_res == "2m" else "2m"
    return url.replace(f"/{other}/", f"/{target_res}/").replace(f"_{other}_", f"_{target_res}_")

def write_fetch_script(out_dir: Path, urls_txt: Path, tiles_dir: Path, vrt_path: Path):
    """
    Write a robust downloader:
      - bypasses curl aliases (uses `command curl`)
      - disables remote-header-name (no hidden -J)
      - resumes partial downloads (-C -)
      - retries 10m → 2m if 10m missing (404)
      - extracts .tar.gz, deletes archives
      - builds VRT from *_dem.tif
    """
    urls_txt = urls_txt.resolve()
    tiles_dir = tiles_dir.resolve()
    vrt_path = vrt_path.resolve()
    script = f"""#!/bin/bash
set -euo pipefail

URLS="{urls_txt}"
TILES="{tiles_dir}"
VRT="{vrt_path}"

mkdir -p "$TILES"
cd "$TILES"

echo "==> Downloading ArcticDEM archives from $URLS"
while IFS= read -r url; do
  [[ -z "$url" ]] && continue

  fname="$(basename "$url")"
  echo "↓ $fname"

  # try as-is (likely 10m); resume, follow redirects, fail on HTTP errors
  if ! command curl --no-remote-header-name -C - -fL --progress-bar -o "$fname" "$url"; then
    # if 10m fails (e.g., 404), try 2m variant
    alt="${{url/\\/10m\\//\\/2m/}}"
    alt="${{alt/_10m_/_2m_}}"
    alt_fname="$(basename "$alt")"
    echo "   10m failed. Trying 2m: $alt_fname"
    if ! command curl --no-remote-header-name -C - -fL --progress-bar -o "$alt_fname" "$alt"; then
      echo "✗ failed both 10m and 2m for: $url"
      continue
    fi
    fname="$alt_fname"
  fi

  echo "📦 Extracting $fname"
  tar -xzvf "$fname"
  rm -f "$fname"
done < "$URLS"

echo "==> Keeping only DEM GeoTIFFs (optional cleanup of extras)"
# Comment out the next line if you want to keep QA/mask files too
find "$TILES" -type f -name "*.tif" ! -name "*_dem.tif" -delete

echo "==> Building VRT"
gdalbuildvrt -overwrite "$VRT" "$TILES"/*_dem.tif

echo "✅ Done. VRT at: $VRT"
"""
    sh_path = out_dir / "fetch-dem.sh"
    sh_path.write_text(script)
    os.chmod(sh_path, 0o755)
    return sh_path

# -------------- main --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to ArcticDEM STRIP index (shp/gpkg)")
    ap.add_argument("--out-dir", required=True, help="Base output dir (e.g., .../data/raw/ArcticDEM)")
    ap.add_argument(
        "--sentinel-bounds",
        nargs=4,
        type=float,
        metavar=("MINX","MINY","MAXX","MAXY"),
        help="Sentinel AOI bounds in the given CRS (e.g., EPSG:32622)"
    )
    ap.add_argument("--sentinel-crs", default="EPSG:32622", help="CRS of sentinel-bounds")
    ap.add_argument("--buffer-m", type=float, default=1000.0, help="Buffer (meters) around AOI")
    ap.add_argument("--resolution", choices=["2m","10m"], default="10m", help="DEM resolution to download")
    ap.add_argument(
        "--max-strips",
        type=int,
        default=10,
        help="Maximum number of best-overlap strips to use (after ranking)."
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tiles_dir = out_dir / "tiles"
    vrt_path  = out_dir / "arcticdem_mosaic.vrt"

    # AOI geom
    if not args.sentinel_bounds:
        raise SystemExit("Error: --sentinel-bounds required (minx miny maxx maxy).")
    minx, miny, maxx, maxy = args.sentinel_bounds
    aoi = box(minx, miny, maxx, maxy).buffer(args.buffer_m)

    # Load index
    idx = gpd.read_file(args.index)
    if str(idx.crs) != args.sentinel_crs:
        idx = idx.to_crs(args.sentinel_crs)

    # Select overlapping strips
    sel = idx[idx.intersects(aoi)].copy()
    print(f"Found {len(sel)} overlapping DEM strips.")
    if sel.empty:
        raise SystemExit("No overlapping strips found. Increase buffer or verify CRS/bounds.")

    # ---------- NEW: rank by intersection area (+ date if available) ----------

    # Intersection area with AOI
    sel["intersect_area"] = sel.geometry.intersection(aoi).area

    # Drop anything with zero / NaN intersect area, just in case
    sel = sel[sel["intersect_area"] > 0]
    if sel.empty:
        raise SystemExit("Overlapping strips had zero intersection area. Check bounds/CRS.")

    url_col = detect_url_column(sel)
    id_col  = detect_id_column(sel)
    date_col = detect_date_column(sel)

    if date_col is not None:
        # try to coerce to datetime, ignore errors
        try:
            sel[date_col] = gpd.pd.to_datetime(sel[date_col], errors="coerce")
        except Exception:
            # if conversion fails, just ignore date
            date_col = None

    # Sort: 1) biggest intersection area, 2) newest acquisition date (if available)
    sort_cols = ["intersect_area"]
    ascending = [False]
    if date_col is not None:
        sort_cols.append(date_col)
        ascending.append(False)

    sel = sel.sort_values(by=sort_cols, ascending=ascending)

    # Keep only top-K strips
    max_k = max(1, args.max_strips)
    sel_best = sel.head(max_k).copy()

    print(f"Selected top {len(sel_best)} strips (max-strips={max_k}) after ranking.")
    # Optional: show quick summary
    cols_to_show = [id_col, "intersect_area"]
    if date_col is not None:
        cols_to_show.append(date_col)
    print(sel_best[cols_to_show])

    # -------------------------------------------------------------------------

    # Raw URLs from best strips
    raw_urls = sel_best[url_col].astype(str).dropna().drop_duplicates().tolist()

    # Build canonical 2m and 10m lists by rewriting tokens
    urls_2m  = [rewrite_resolution(u, "2m") for u in raw_urls]
    urls_10m = [rewrite_resolution(u, "10m") for u in raw_urls]

    # Write lists + CSV (in current working directory)
    here = out_dir
    (here / "dem_urls_raw.txt").write_text("\n".join(raw_urls) + "\n")
    (here / "dem_urls_2m.txt").write_text("\n".join(urls_2m) + "\n")
    (here / "dem_urls_10m.txt").write_text("\n".join(urls_10m) + "\n")
    sel_best[[id_col, url_col]].dropna().drop_duplicates().to_csv(here / "dem_urls.csv", index=False)

    # Choose active list by requested resolution, mirror into out_dir as dem_urls.txt
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen_src  = here / ("dem_urls_10m.txt" if args.resolution == "10m" else "dem_urls_2m.txt")
    chosen_copy = out_dir / "dem_urls.txt"
    chosen_copy.write_text(chosen_src.read_text())

    print(f"Wrote:\n  dem_urls.csv\n  dem_urls_raw.txt\n  dem_urls_2m.txt\n  dem_urls_10m.txt")
    print(f"Chosen resolution: {args.resolution} → {chosen_copy}")

    # Write fetch script (with fixed curl flags)
    fetch_sh = write_fetch_script(out_dir=out_dir, urls_txt=chosen_copy,
                                  tiles_dir=tiles_dir, vrt_path=vrt_path)
    print(f"Fetch script ready: {fetch_sh}")
    print("Run:\n  bash", fetch_sh)

if __name__ == "__main__":
    main()
