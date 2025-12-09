# Supraglacial Lake Detection Project | Fall 2025
Joshua (Chang Hyeon) Park | University of Chicago
## Jakobshavn Supraglacial Lake Detection

As anthropogenic climate change accelerates surface melt across the polar regions, the retreat of marine-terminating glaciers has become a major contributor to global sea-level instability. On the Greenland Ice Sheet (GrIS), the rapid formation and drainage of supraglacial lakes play a critical role in hydrofracture and ice-flow acceleration.

This project applies computer vision techniques to Sentinel-2 imagery and trains a neural network to automatically detect and delineate supraglacial meltwater ponds.

## Project Description:
The Supraglacial Lake Detection Project focuses on automating the identification of melt ponds on Jakobshavn Glacier using a combination of satellite imagery, geospatial preprocessing, and a U-Net segmentation model. The goal is to build a fully reproducible workflow that transforms raw Sentinel-2 Level-2A imagery and ArcticDEM elevation data into spatially aligned datasets suitable for machine learning.

![Project Goals](images/supraglcial_projectgoals.jpg)

The project has two major components:

(1) Geospatial Data Preparation  
This includes locating and loading Sentinel-2 .SAFE directories, extracting CRS metadata, identifying overlapping ArcticDEM strips, building a VRT mosaic, reprojecting the DEM to match the Sentinel geometry, clipping rasters to a shared footprint, computing NDWI, and generating initial binary meltwater masks.

(2) Machine Learning Pipeline  
Using NDWI and corresponding masks, the project splits large rasters into 256×256 image tiles, constructs PyTorch datasets, trains a U-Net segmentation model, and visualizes predictions. This allows the network to learn melt pond characteristics beyond simple thresholding.

![Pipeline Diagram](images/supraglacial_pipeline.jpg)

### Data Set Preparation ###
#### Method Overview:
1. **CRS Extraction**  
Read EPSG and CRS metadata from Sentinel-2 `.SAFE` folders and ArcticDEM `GeoTIFF`.
2. **Geometric 'preflight' check**  
Confirm overlapping bounds between datasets before reprojection.  
3. **DEM Reprojection and Alignment**
Match DEM resolution, grid, and CRS to Sentinel-2 imagery. This process also involves selecting a target tile from a user friendly interface in
the Copernicus dataset, extracting CRS from target tiles and compiling a list of urls to download target DEM areas.
4. **Clipping**  
Trim DEMs to exact Sentinel footprint for spatial consistency.
5. **Integration with Shapefiles**  
Use aligned rasters for creating and overlaying shapefiles of melt ponds, catchments, or training polygons. This process will
use GIS/ArcGISPro for tracing.
6. **NDWI Computation & Mask Generation**  
Compute the Normalized Difference Water Index (NDWI) from Sentinel-2 bands (B03, B08) to identify meltwater bodies.  
Apply thresholds to generate preliminary binary masks separating water and ice.

### Machine Learning ###
#### Method Overview:
7. **Dataset Prepartion for ML**
- Convert large rasters (NDWI + mask) into overlapping tiles (e.g., 256×256 with stride 128).  
- Split into train / val / test sets.
8. **CNN/U-Net Segmentation**
- Train a convolutional U-Net on tiles to segment meltwater features.  
- The network learns spatial patterns beyond simple thresholding
9. **Evaluation and Visualization**
- Metrics: IoU, precision, recall (per tile and aggregated).  
- Visualize probability maps and thresholded predictions side-by-side with NDWI and ground truth.

## Python Tools and Libraries ##
`rasterio` - core library for reading, writing, and transforming geospatial raster data (used for CRS extraction, reprojection, and masking)  
`GDAL` - underlying geospatial engine that powers raster operations; used here via command-line tools like `gdalbuildvrt` for mosaicking DEM tiles  
`geopandas` - for reading shapefiles and vector indices  
`numpy` - array math for raster operations, resampling, and mask creation  
`matplotlib` - for quick visualization of raster (mostly for debugging and analysis)  

## File Descriptions (bookkeeping) ##
`crs_utils.py`: 
Extracts coordinate reference system (CRS) and EPSG metadata from Sentinel-2 .SAFE folders and ArcticDEM GeoTIFFs.
Used to ensure all datasets share the same spatial reference before reprojection or alignment.

`dem_utils.py`: 
Handles Digital Elevation Model (DEM) loading, reprojection, and clipping.
Reprojects ArcticDEM tiles to match the CRS, grid, and resolution of Sentinel-2 imagery, producing spatially aligned elevation data.

`load_bands.py`: 
Loads Sentinel-2 spectral bands (e.g., B03, B08) from .SAFE directories, computes the Normalized Difference Water Index (NDWI),
and prepares raster data for meltwater mapping and machine learning input. Includes utilities to extract band profiles, read arrays, and generate NDWI masks.

`build_vrt.py`: # this step was automated with the help of existing LLM  
Finds ArcticDEM STRIP tiles that overlap a Sentinel-2 area of interest (AOI).
Generates URL lists and a small fetch script (fetch-dem.sh) to download and extract the tiles, then uses GDAL’s gdalbuildvrt to assemble them into a Virtual Raster Tile (VRT) — a lightweight mosaic referencing all DEM tiles without physically merging them.
This forms the base DEM mosaic for later reprojection and NDWI alignment.
