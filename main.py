"""
Preliminary Visualization Code for Jakobshavn Supraglacial Lake detection
Target SAFE file: T22WDA 20240803T192030 | reference: safe_folder_name
> Next Step: utilizing sys.argv in order to fix the hardcoding issues with runner file

Output: side by side plot to see preliminary map and check if the code is working
"""

import os
import numpy as np
import rasterio
from rasterio.transform import array_bounds

from Sentinel import load_bands
from CRS import crs_utils
from DEM import dem_utils
from Geometry import geometry_utils

# PART 01 Sentinel SAFE (Reference Grid)
safe_folder = "/Users/josh/Projects/GLDetection/data/raw/SAFE"
safe_folder_name = "S2B_MSIL2A_20240803T151809_N0511_R068_T22WDA_20240803T192030.SAFE"
safe_folder_path = os.path.join(safe_folder, safe_folder_name)

sentinel_crs, sentinel_epsg = crs_utils.get_safe_crs(safe_folder_path)
profile = load_bands.load_profile_from_safe(safe_folder_path, "B03_10m")
sentinel_bounds = array_bounds(profile["height"], profile["width"], profile["transform"])

print("📌 Sentinel EPSG:", sentinel_epsg)
print("🧭 Sentinel bounds:", sentinel_bounds)

# PART 02 DEM (A Mosaic VRT - using gdalbuildvrt)
dem_path = "/Users/josh/Projects/GLDetection/data/raw/ArcticDEM/SETSM_s2s041_WV02_20210823_10300100C40D0900_10300100C47B7700_2m_lsf_seg1/SETSM_s2s041_WV02_20210823_10300100C40D0900_10300100C47B7700_2m_lsf_seg1_dem.tif"
with rasterio.open(dem_path) as dem_src:
    dem_crs = dem_src.crs
    dem_bounds = dem_src.bounds

dem_bounds_in_s2 = geometry_utils.transform_bounds_to_match_crs(
    bounds=dem_bounds, src_crs=dem_crs, dst_crs=sentinel_crs
)

if not geometry_utils.bounds_overlap(dem_bounds_in_s2, sentinel_bounds):
    raise ValueError("❌ DEM mosaic does not cover Sentinel footprint. Download nearby tiles and rebuild VRT.")

print(f"✅ DEM-Sentinel overlap ratio: {geometry_utils.overlap_ratio(dem_bounds_in_s2, sentinel_bounds):.3f}")

# PART 03 Align DEM to Sentinel grid (writes GeoTIFFs)
out_dir = "/Users/josh/Projects/GLDetection/outputs"
os.makedirs(out_dir, exist_ok=True)
s2_b03_path = crs_utils.find_band_path(safe_folder_path, "B03_10m")

aligned_dem_path, clipped_dem_path = dem_utils.align_dem_to_sentinel(
    dem_path=dem_path,
    sentinel_band_path=s2_b03_path,
    out_aligned_path=os.path.join(out_dir, "dem_to_sentinel.tif"),
    out_clipped_path=os.path.join(out_dir, "dem_to_sentinel_clipped.tif"),
    dst_nodata=-9999
)
print("🗺️ DEM aligned:", aligned_dem_path)
print("✂️  DEM clipped:", clipped_dem_path)

# PART 04 NDWI (for quick overlay sanity)
ndwi, ndwi_profile = load_bands.load_ndwi_from_safe(safe_folder_path)

"""
Building Mosaic
gdalbuildvrt -overwrite /Users/josh/Projects/GLDetection/data/raw/ArcticDEM/arcticdem_mosaic.vrt /Users/josh/Projects/GLDetection/data/raw/ArcticDEM/*/*.tif
"""

# PART 05 Producing Hillshade Plot
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from rasterio.warp import reproject, Resampling

# 5.1 Threshold NDWI → mask
NDWI_THRESH = 0.25   # tweakable
ELEV_MIN    = 0.0  # meters a.s.l., tweakable (based on statistical distribution)

# --- initial NDWI mask ---
ndwi_mask = (ndwi > NDWI_THRESH).astype(np.uint8)

# 5.2 Save the raw NDWI mask
mask_profile = ndwi_profile.copy()
mask_profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="deflate")
ndwi_mask_path = os.path.join(out_dir, "ndwi_mask_raw.tif")
with rasterio.open(ndwi_mask_path, "w", **mask_profile) as dst:
    dst.write(ndwi_mask, 1)
print("💾 Raw NDWI mask saved:", ndwi_mask_path)

# 5.3 Load the (clipped) DEM
with rasterio.open(clipped_dem_path) as dem_src:
    dem = dem_src.read(1)    # DEM values
    dem_nodata = dem_src.nodata
    dem_extent = [dem_src.bounds.left, dem_src.bounds.right,
                  dem_src.bounds.bottom, dem_src.bounds.top]
    dem_transform = dem_src.transform
    dem_crs = dem_src.crs

if dem_nodata is not None:
    dem_masked = np.where(dem == dem_nodata, np.nan, dem)
else:
    dem_masked = dem.astype(float)

# 5.4 Ensure NDWI mask grid matches DEM grid (size/transform/CRS)
with rasterio.open(ndwi_mask_path) as msk_src:
    if (msk_src.width, msk_src.height, msk_src.crs, msk_src.transform) != \
       (dem.shape[1], dem.shape[0], dem_crs, dem_transform):
        reprojected = np.zeros(dem.shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(msk_src, 1),
            destination=reprojected,
            src_transform=msk_src.transform,
            src_crs=msk_src.crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.nearest,
            dst_nodata=0,
        )
        ndwi_mask = reprojected
    else:
        ndwi_mask = msk_src.read(1)

# --- NEW STEP: Apply DEM elevation filter ---
lake_mask = (ndwi_mask == 1) & (dem_masked > ELEV_MIN)

# Save the supraglacial lake mask
lake_profile = ndwi_profile.copy()
lake_profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="deflate")
lake_mask_path = os.path.join(out_dir, f"lake_mask_ndwi{NDWI_THRESH}_dem{int(ELEV_MIN)}.tif")
with rasterio.open(lake_mask_path, "w", **lake_profile) as dst:
    dst.write(lake_mask.astype(np.uint8), 1)
print("💾 Supraglacial lake mask saved:", lake_mask_path)

# === Diagnostics before plotting ===
# A: Look at elevation distribution of NDWI>thr pixels
water = (ndwi > NDWI_THRESH) & np.isfinite(dem_masked)
elev_on_water = dem_masked[water]

print("[NDWI>thr] elevations (m):")
print(" min =", np.nanmin(elev_on_water))
print(" p5, p10, p25, p50, p75, p90 =",
      np.nanpercentile(elev_on_water, [5,10,25,50,75,90]))

# B: Sweep different DEM floors to see effect
for test_floor in [50, 75, 100, 125, 150, 175, 200]:
    lm = (ndwi > NDWI_THRESH) & (dem_masked > test_floor)
    px = int(lm.sum())
    km2 = px * 100.0 / 1e6   # Sentinel-2 10 m → 100 m² per pixel
    print(f"ELEV_MIN={test_floor:>3} m -> {px:,} px ({km2:.2f} km²)")

# 5.5 Side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Panel 1: Raw NDWI ---
im1 = axes[0].imshow(ndwi, cmap="RdYlBu", vmin=-1, vmax=1, extent=sentinel_bounds)
axes[0].set_title("NDWI (Normalized Difference Water Index)")
axes[0].set_xlabel("Easting (m)"); axes[0].set_ylabel("Northing (m)")
cbar = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
cbar.set_label("NDWI value")

# --- Panel 2: Supraglacial lakes (NDWI + DEM filter) ---
ls = LightSource(azdeg=315, altdeg=45)
hs_rgb = ls.shade(np.where(np.isfinite(dem_masked), dem_masked, np.nan),
                  cmap=plt.cm.gray, blend_mode="overlay")

axes[1].imshow(hs_rgb, extent=dem_extent)

# crisp magenta overlay (binary)
magenta_cmap = ListedColormap([(1, 0, 1, 0.0),  # transparent for 0
                               (1, 0, 1, 0.9)]) # magenta for 1
axes[1].imshow(lake_mask.astype(np.uint8), cmap=magenta_cmap, extent=dem_extent)

axes[1].set_title(f"Supraglacial Lakes (NDWI>{NDWI_THRESH}, DEM>{int(ELEV_MIN)} m)")
axes[1].set_xlabel("Easting (m)"); axes[1].set_ylabel("Northing (m)")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ndwi_panels_supraglacial.png"), dpi=300)
plt.show()