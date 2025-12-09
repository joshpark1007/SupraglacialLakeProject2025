# How To Run on Local Machine
Joshua (Chang Hyeon) Park | University of Chicago

This guide walks through running the full pipeline on a small example: from raw Sentinel-2 Level-2A data and ArcticDEM strips
to U-Net lake segmentation outputs.

## Environment Setup:


## Data Directory Layout:

![Data Architecture](images/supraglacial_dataarchitecture.png)

## Run:

Main
"""
August 3rd, 2024 Tile T22WDA (Model Set-up)
python main.py process \
  --safe "data/raw/SAFE/S2B_MSIL2A_20240803T151809_N0511_R068_T22WDA_20240803T192030.SAFE" \
  --dem  "data/raw/ArcticDEM/arcticdem_mosaic.vrt" \
  --out  "data/derived/lakes" \
  --ndwi 0.25 \
  --emin 0 \
  --min-area-m2 1000 \
  --ext gpkg    
"""

Build VRT
Usage:

 python3 VRT/build_vrt.py \
  --index "data/indexes/ArcticDEM_Strip_Index_s2s041_shp/ArcticDEM_Strip_Index_s2s041.shp" \
  --out-dir "data/raw/ArcticDEM" \
  --sentinel-bounds 399960 7490220 509760 7600020 \
  --sentinel-crs EPSG:32622 \
  --buffer-m 1000 \
  --resolution 2m \
  --max-strips 5
 
make_tiles.py
Expected Script:
python make_tiles.py \
  --in-dir data/derived/lakes \
  --out-dir data/tiles \
  --tile-size 256 \
  --stride 128

What the script does:
- Searches for NDWI rasters named:
      <timestamp>_<tile>_ndwi_0.25.tif
- Finds the corresponding lake mask raster:
      <timestamp>_<tile>_lake_ndwi0.25_dem0.tif
- Ensures both rasters share identical grid shape + transform
- Slides a window across both rasters:
      window size = tile-size (default 256)
      step size   = stride (default 128)
- Saves each tile as a compressed .npz file:
      images/  → NDWI tiles (shape: 1 × H × W)
      masks/   → binary lake-mask tiles (shape: H × W)

Purpose: finding files like 2024-08-03_T22WDA_ndwi_0.25.tif, match with 2024-08-03_T22WDA_lake_ndwi0.25_dem0.tif
and cut them into aligned tiles for image segmentation modeling

Output:
Creates a dataset directory containing two subfolders:
    data/derived/tiles/images/
    data/derived/tiles/masks/
