"""
File name: dem_utils.py
Purpose: Load DEMs, reproject them to match Sentinel-2 grids, and optionally
         clip to raster bounds. Provides alignment utilities for glacier-
         scale DEM–Sentinel fusion.
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds

def load_dem(dem_path):
    """
    Load DEM and its profile.
    """
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_profile = src.profile
    return dem_data, dem_profile

def reproject_dem_to_match_profile(dem_path, target_profile, out_path=None, resampling=Resampling.bilinear,
                                   src_nodata=None, dst_nodata=-9999):
    """
    Reprojects the DEM to match the CRS, transform, width, height of a reference profile.
    If out_path is provided, writes GeoTIFF and returns (array, profile).
    Otherwise, returns (array, profile) in memory.

    Note:
    - We pass src_nodata and dst_nodata explicitly so resampling respects empties.
    - target_profile must contain: crs, transform, width, height
    """
    with rasterio.open(dem_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        if src_nodata is None:
            src_nodata = src.nodata

        dst_crs = target_profile["crs"]
        dst_transform = target_profile["transform"]
        dst_width = target_profile["width"]
        dst_height = target_profile["height"]

        dst_data = np.full((dst_height, dst_width), dst_nodata, dtype=np.float32)

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
            resampling=resampling
        )

        out_profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "nodata": dst_nodata,
            "compress": "deflate",
            "predictor": 2
        }

        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(dst_data, 1)

    return dst_data, out_profile

def clip_to_ref_bounds(src_path, ref_path, out_path):
    """
    Clip src (already in ref CRS/grid) to the ref raster's bounds, writing a GeoTIFF.
    Useful when src was reprojected to ref CRS but not exactly the same extent.
    """
    with rasterio.open(ref_path) as ref:
        rb = ref.bounds
        rcrs = ref.crs

    with rasterio.open(src_path) as src:
        assert str(src.crs) == str(rcrs), "Clip requires src already in reference CRS."
        win = from_bounds(*rb, transform=src.transform).round_offsets().round_lengths()
        data = src.read(window=win)
        out_transform = src.window_transform(win)
        profile = src.profile.copy()
        profile.update({"height": data.shape[1], "width": data.shape[2], "transform": out_transform})

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

def align_dem_to_sentinel(dem_path, sentinel_band_path, out_aligned_path, out_clipped_path=None,
                          resampling=Resampling.bilinear, src_nodata=None, dst_nodata=-9999):
    """
    High-level convenience:
      1) Read Sentinel band profile (CRS/grid).
      2) Reproject DEM onto that exact grid (write to out_aligned_path).
      3) Optionally clip to the exact Sentinel footprint (write to out_clipped_path).
    Returns paths written.
    """
    with rasterio.open(sentinel_band_path) as s2:
        ref_profile = {
            "crs": s2.crs,
            "transform": s2.transform,
            "width": s2.width,
            "height": s2.height
        }

    _, aligned_profile = reproject_dem_to_match_profile(
        dem_path, ref_profile, out_path=out_aligned_path,
        resampling=resampling, src_nodata=src_nodata, dst_nodata=dst_nodata
    )

    if out_clipped_path:
        clip_to_ref_bounds(out_aligned_path, sentinel_band_path, out_clipped_path)
        return out_aligned_path, out_clipped_path

    return out_aligned_path, None
