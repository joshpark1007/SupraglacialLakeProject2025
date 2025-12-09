"""
File name: load_bands.py
Purpose: Locate and load Sentinel-2 spectral bands (B03, B08) from .SAFE folders
         and compute NDWI arrays for supraglacial lake detection workflows.
"""

import rasterio
import numpy as np
from CRS.crs_utils import find_band_path

def load_profile_from_safe(safe_folder, band_name="B03_10m"):
    band_path = find_band_path(safe_folder, band_name)
    with rasterio.open(band_path) as src:
        return src.profile

def load_band(band_path):
    with rasterio.open(band_path) as src:
        band = src.read(1).astype(np.float32)
        profile = src.profile
    return band, profile

def compute_ndwi(b3, b8):
    return (b3 - b8) / (b3 + b8 + 1e-10)

def load_ndwi_from_safe(safe_folder):
    b3_path = find_band_path(safe_folder, "B03_10m")
    b8_path = find_band_path(safe_folder, "B08_10m")
    b3, _ = load_band(b3_path)
    b8, profile = load_band(b8_path)
    ndwi = compute_ndwi(b3, b8)
    return ndwi, profile
