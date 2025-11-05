# Fall25-Python-FinalProject
Joshua (Chang Hyeon) Park
## Jakobshavn Supraglacial Lake Detection

With anthropogenic climate change accelerating polar melt, the retreat of marine-terminating glaciers poses a major
threat to global sea-level instability. One of the biggest challenges in the Greenland Ice Sheet (GrIS) is the increasing amount of
supraglacial lake formation that leads to mass drainages and hydrofracturing. This project aims to apply computer vision
techniques to analyze satellite imagery, train a neural network to identify and track changes to the lake formation over time.

## Project Description:
A geospatial workflow for aligning Sentinel-2 imagery (European Space Agency) and Arctic Digital Elevation Model (DEM) to
accurately sort out and segment glacier surface melt against ice mass. The goal is to prepare spatially consistent datasets for melt pond detection
by building a pipeline that aligns and reprojects two different maps based on coordinate reference systems (CRS).

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
Convert raster masks and corresponding Sentinel-2 imagery into training patches, ensuring spatial alignment.
Split the dataset into training (70%), validation (15%), and test (15%) subsets.  
8. **CNN/U-Net Segmentation**
Train a convolutional neural network (U-Net architecture) to automatically segment meltwater features.
The model learns spatial patterns in NDWI imagery and DEM context to detect lakes with higher precision and reproducibility 
beyond target Sentinel tile and timefrmae.
9. **Evaluation and Visualization**
Assess segmentation performance (IoU, precision, recall) and visualize detected melt ponds across time to analyze spatial-temporal evolution.

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
