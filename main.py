"""
File: main.py
Purpose: Run the geospatial preprocessing pipeline for a Sentinel-2 .SAFE scene:
         compute NDWI, align ArcticDEM via VRT, apply an elevation filter, and
         export supraglacial lake masks as rasters and vector polygons
         (single-tile or batch mode via CLI).
"""

import sys
print("PYTHON USED:", sys.executable)

import os
import re
import glob
import argparse

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

from Sentinel import load_bands
from CRS import crs_utils
from DEM import dem_utils
from Geometry import geometry_utils

# helpers to modularize
def safe_id(safe_path: str):
    """Return (YYYY-MM-DD, 'T22WDA', basename) from a SAFE folder name."""
    name = os.path.basename(safe_path.rstrip("/"))
    m_date = re.search(r"_(\d{8})T", name)
    date_str = ""
    if m_date:
        ymd = m_date.group(1)
        date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    m_tile = re.search(r"_(T\d{2}[A-Z]{3})_", name)
    tile = m_tile.group(1) if m_tile else "TXXXX"
    return date_str, tile, name

def polygonize_mask_to_vectors(mask_path: str, out_path: str,
                               min_area_m2: float,
                               safe_name: str,
                               ndwi_thr: float,
                               elev_min: float):
    """Binary mask (1=lake) -> vector polygons with area filter in meters."""
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    feats = []
    for geom, val in shapes(arr, transform=transform):
        if int(val) != 1:
            continue
        feats.append(shape(geom).buffer(0))  # clean tiny topo errors

    if not feats:
        print("⚠️ No polygons found after thresholding.")
        return None

    gdf = gpd.GeoDataFrame(geometry=feats, crs=crs)
    gdf_m = gdf.to_crs(32622)  # Jakobshavn UTM zone → meters
    gdf_m["area_m2"] = gdf_m.area
    gdf_m = gdf_m[gdf_m["area_m2"] >= float(min_area_m2)]
    if gdf_m.empty:
        print("⚠️ All polygons filtered out by min area.")
        return None

    # metadata
    acq_date = ""
    m = re.search(r"_(\d{8})T", safe_name)
    if m:
        ymd = m.group(1); acq_date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    gdf_m["safe_name"] = safe_name
    gdf_m["date"] = acq_date
    gdf_m["ndwi_thr"] = ndwi_thr
    gdf_m["elev_min_m"] = elev_min

    driver = "GPKG" if out_path.lower().endswith(".gpkg") else "ESRI Shapefile"
    gdf_m.to_file(out_path, driver=driver)
    return out_path

def process_safe(safe_path: str, dem_path: str, out_root: str,
                 ndwi_thresh: float, elev_min: float, min_area_m2: float,
                 vector_ext: str = ".gpkg"):
    os.makedirs(out_root, exist_ok=True)
    date_str, tile, safe_name = safe_id(safe_path)
    tag = f"{date_str}_{tile}" if date_str else os.path.basename(safe_path).replace(".SAFE", "")

    # PART 01 Sentinel SAFE (Reference Grid)
    sentinel_crs, sentinel_epsg = crs_utils.get_safe_crs(safe_path)
    profile = load_bands.load_profile_from_safe(safe_path, "B03_10m")
    sentinel_bounds = array_bounds(profile["height"], profile["width"], profile["transform"])

    print("📌 Sentinel EPSG:", sentinel_epsg)
    print("🧭 Sentinel bounds:", sentinel_bounds)

    # PART 02 DEM (A Mosaic VRT - using gdalbuildvrt)
    with rasterio.open(dem_path) as dem_src:
        dem_crs = dem_src.crs
        dem_bounds = dem_src.bounds

    dem_bounds_in_s2 = geometry_utils.transform_bounds_to_match_crs(
        bounds=dem_bounds, src_crs=dem_crs, dst_crs=sentinel_crs)

    if not geometry_utils.bounds_overlap(dem_bounds_in_s2, sentinel_bounds):
        print(f"❌ DEM does not cover {tag}. Skipping.")
        return
    print(f"✅ DEM-Sentinel overlap ratio: "
          f"{geometry_utils.overlap_ratio(dem_bounds_in_s2, sentinel_bounds):.3f}")

    # PART 03 Align DEM to Sentinel grid (writes GeoTIFFs)

    print("📁 Checking cached DEM alignments…")

    s2_b03_path = crs_utils.find_band_path(safe_path, "B03_10m")

    # reducing recomputation over same target grid across temporal scales
    aligned_dem_path = os.path.join(out_root, "dem_to_sentinel.tif")
    clipped_dem_path = os.path.join(out_root, "dem_to_sentinel_clipped.tif")

    need_align = not os.path.exists(aligned_dem_path)
    need_clip = not os.path.exists(clipped_dem_path)

    if need_align or need_clip:
        print("🔄 Computing DEM alignment (first time)…")
        aligned_dem_path, clipped_dem_path = dem_utils.align_dem_to_sentinel(
            dem_path=dem_path,
            sentinel_band_path=s2_b03_path,
            out_aligned_path=aligned_dem_path,
            out_clipped_path=clipped_dem_path,
            dst_nodata=-9999
        )
    else:
        print("🗺️ Reusing existing DEM alignment:", aligned_dem_path)
        print("✂️  Reusing existing DEM clip:", clipped_dem_path)

    # PART 04 NDWI + Save Raw Mask
    ndwi, ndwi_profile = load_bands.load_ndwi_from_safe(safe_path)
    ndwi_mask = (ndwi > ndwi_thresh).astype(np.uint8)
    ndwi_mask_path = os.path.join(out_root, f"{tag}_ndwi_{ndwi_thresh:.2f}.tif")
    mask_profile = ndwi_profile.copy()
    mask_profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="deflate")
    with rasterio.open(ndwi_mask_path, "w", **mask_profile) as dst:
        dst.write(ndwi_mask, 1)
    print("💾 Raw NDWI mask saved:", ndwi_mask_path)

    # PART 05 DEM filter (Ensuring Same Grid)
    print("🔎 reading clipped DEM:", clipped_dem_path)
    with rasterio.open(clipped_dem_path) as dem_ds:
        dem_arr = dem_ds.read(1)  # DEM values
        dem_nodata = dem_ds.nodata
        dem_transform = dem_ds.transform
        dem_crs = dem_ds.crs
        dem_h, dem_w = dem_ds.height, dem_ds.width
        # dem_extent only needed if plotting:
        # dem_extent = [dem_ds.bounds.left, dem_ds.bounds.right,
        #               dem_ds.bounds.bottom, dem_ds.bounds.top]

    # now work with arrays (dataset is closed)
    dem_float = dem_arr.astype("float32")
    if dem_nodata is not None:
        dem_masked = np.where(dem_arr == dem_nodata, np.nan, dem_float)
    else:
        dem_masked = dem_float

    # Ensure NDWI mask grid matches DEM grid (size/transform/CRS)
    with rasterio.open(ndwi_mask_path) as msk_src:
        same_grid = (
                msk_src.width == dem_w and msk_src.height == dem_h and
                msk_src.crs == dem_crs and msk_src.transform == dem_transform
        )
        if not same_grid:
            reproj = np.zeros((dem_h, dem_w), dtype=np.uint8)
            reproject(
                source=rasterio.band(msk_src, 1),
                destination=reproj,
                src_transform=msk_src.transform,
                src_crs=msk_src.crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.nearest,
                dst_nodata=0,
            )
            ndwi_mask = reproj
        else:
            ndwi_mask = msk_src.read(1)

    # Save the supraglacial lake mask
    lake_mask = (ndwi_mask == 1) & (dem_float > float(elev_min))

    lake_mask_path = os.path.join(out_root, f"{tag}_lake_ndwi{ndwi_thresh:.2f}_dem{int(elev_min)}.tif")
    lake_profile = {
        "driver": "GTiff",
        "height": dem_h, "width": dem_w, "count": 1,
        "dtype": rasterio.uint8, "crs": dem_crs, "transform": dem_transform,
        "nodata": 0, "compress": "deflate"
    }
    with rasterio.open(lake_mask_path, "w", **lake_profile) as dst:
        dst.write(lake_mask.astype(np.uint8), 1)
    print("💾 Supraglacial lake mask saved:", lake_mask_path)

    # PART 06 Polygonize into a vector
    vec_ext = ".gpkg" if vector_ext.lower() == ".gpkg" else ".shp"
    vec_path = os.path.join(out_root, f"{tag}_lakes{vec_ext}")
    out_vec = polygonize_mask_to_vectors(
        mask_path=lake_mask_path,
        out_path=vec_path,
        min_area_m2=min_area_m2,
        safe_name=safe_name,
        ndwi_thr=ndwi_thresh,
        elev_min=elev_min
    )
    if out_vec:
        print(f"✅ Vector lakes written to: {out_vec}")
    else:
        print("⚠️ No vector output created.")

# CLI
def main():
    ap = argparse.ArgumentParser(description="NDWI + DEM filter -> vector lakes (single or batch)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("process", help="Process a single .SAFE")
    p1.add_argument("--safe", required=True, help="Path to a Sentinel-2 .SAFE directory")
    p1.add_argument("--dem", required=True, help="Path to DEM (GeoTIFF or VRT)")
    p1.add_argument("--out", required=True, help="Output directory")
    p1.add_argument("--ndwi", type=float, default=0.25, help="NDWI threshold")
    p1.add_argument("--emin", type=float, default=0.0, help="Minimum elevation (m)")
    p1.add_argument("--min-area-m2", type=float, default=1000.0, help="Min polygon area in m^2")
    p1.add_argument("--ext", choices=["gpkg","shp"], default="gpkg", help="Vector format")

    p2 = sub.add_parser("batch", help="Process all .SAFE under a folder (recursive)")
    p2.add_argument("--safe-root", required=True, help="Root folder containing .SAFE directories")
    p2.add_argument("--glob", default="**/*.SAFE", help="Glob to find SAFE dirs (default recursive)")
    p2.add_argument("--dem", required=True, help="Path to DEM (GeoTIFF or VRT)")
    p2.add_argument("--out", required=True, help="Output directory")
    p2.add_argument("--ndwi", type=float, default=0.25)
    p2.add_argument("--emin", type=float, default=0.0)
    p2.add_argument("--min-area-m2", type=float, default=1000.0)
    p2.add_argument("--ext", choices=["gpkg","shp"], default="gpkg")

    args = ap.parse_args()

    if args.cmd == "process":
        process_safe(
            safe_path=args.safe, dem_path=args.dem, out_root=args.out,
            ndwi_thresh=args.ndwi, elev_min=args.emin, min_area_m2=args.min_area_m2,
            vector_ext="." + args.ext
        )

    elif args.cmd == "batch":
        paths = [p for p in glob.glob(os.path.join(args.safe_root, args.glob), recursive=True)
                 if p.endswith(".SAFE") and os.path.isdir(p)]
        if not paths:
            print("No .SAFE folders found. Check --safe-root or --glob.")
            return
        for i, s in enumerate(sorted(paths), 1):
            print(f"\n[{i}/{len(paths)}] {os.path.basename(s)}")
            process_safe(
                safe_path=s, dem_path=args.dem, out_root=args.out,
                ndwi_thresh=args.ndwi, elev_min=args.emin, min_area_m2=args.min_area_m2,
                vector_ext="." + args.ext
            )

if __name__ == "__main__":
    main()
