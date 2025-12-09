"""
File name: crs_utils.py
Purpose: Extract CRS and EPSG information from Sentinel-2 .SAFE directories
         and ArcticDEM GeoTIFFs. Used to ensure consistent spatial reference
         before reprojection and alignment.
"""

import os
import rasterio

def find_band_path(safe_path, band_name="B03_10m"):
    for root, dirs, files in os.walk(safe_path):
        for file in files:
            if band_name in file and file.endswith(".jp2"):
                return os.path.join(root, file)
    raise FileNotFoundError(f"{band_name} not found in {safe_path}")

def get_safe_crs(safe_path, band_name="B03_10m"):
    """
    Extract CRS and EPSG code from a Sentinel-2 SAFE folder.
    """
    band_path = find_band_path(safe_path, band_name)
    with rasterio.open(band_path) as src:
        crs = src.crs
    return crs, crs.to_epsg()

def get_dem_crs(dem_path):
    """
    Extract CRS and EPSG code from a DEM file.
    """
    with rasterio.open(dem_path) as src:
        crs = src.crs
    return crs, crs.to_epsg()
