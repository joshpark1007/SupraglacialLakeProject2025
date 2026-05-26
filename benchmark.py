import os

import numpy as np
import torch
import rasterio
import geopandas as gpd

from rasterio.features import rasterize
from Sentinel import load_bands
from unet import UNetSmall

# Debug Helpers
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import box

## For Benchmark, internal process to convert SAFE to NDWI tif
def save_raw_ndwi_from_safe(safe_path, out_tif):
    ndwi, profile = load_bands.load_ndwi_from_safe(safe_path)

    profile = profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=None,
        compress="deflate",
    )

    os.makedirs(os.path.dirname(out_tif), exist_ok=True)

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(ndwi.astype("float32"), 1)

    return out_tif


def load_model(weights_path, device):
    model = UNetSmall(in_channels=1, out_channels=1).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def rasterize_lakes(geojson_paths, reference_tif):
    with rasterio.open(reference_tif) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    geoms = []
    for path in geojson_paths:
        gdf = gpd.read_file(path)

        # GeoJSON coordinates appear to be lon/lat, so assume EPSG:4326 if missing.
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        gdf = gdf.to_crs(crs)
        geoms.extend(gdf.geometry[gdf.geometry.notnull()])

    mask = rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask


def predict_full_raster(model, ndwi_tif, device, tile_size=256, stride=128):
    with rasterio.open(ndwi_tif) as src:
        ndwi = src.read(1).astype("float32")

    ndwi = np.clip(ndwi, -1.0, 1.0)

    h, w = ndwi.shape
    prob_sum = np.zeros((h, w), dtype="float32")
    count = np.zeros((h, w), dtype="float32")

    ys = range(0, h - tile_size + 1, stride)
    xs = range(0, w - tile_size + 1, stride)

    with torch.no_grad():
        for y in tqdm(ys, desc="Predicting rows"):
            for x in xs:
                tile = ndwi[y:y + tile_size, x:x + tile_size]
                tensor = torch.from_numpy(tile[None, None, :, :]).float().to(device)

                logits = model(tensor)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

                prob_sum[y:y + tile_size, x:x + tile_size] += prob
                count[y:y + tile_size, x:x + tile_size] += 1
    prob_map = np.zeros_like(prob_sum)
    valid = count > 0
    prob_map[valid] = prob_sum[valid] / count[valid]
    return prob_map

def metrics(pred_mask, true_mask):
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    tn = np.logical_and(~pred, ~true).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "accuracy": accuracy,
    }

def main():
    # Apple Silicon/MPS if available (for faster computing)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    weights = "unet_lakes.pth"
    safe_path = (
        "benchmarkdata/"
        "S2B_MSIL2A_20180805T152039_N0500_R068_T22WEA_20230624T025517.SAFE"
    )
    ndwi_tif = "benchmarkdata/derived/2018_T22WEA_raw_ndwi.tif"

    surface_geojsons = [
        "benchmarkdata/lakes/surface/surface_CW2018.geojson",
        "benchmarkdata/lakes/surface/surface_SW2018.geojson",
    ]

    # computing
    ndwi_tif = save_raw_ndwi_from_safe(safe_path, ndwi_tif)

    # Temporary Debug Code
    debug_surface_overlap(ndwi_tif, year="2018")
    return

    model = load_model(weights, device)

    # rasterizing
    true_surface = rasterize_lakes(surface_geojsons, ndwi_tif)

    print("Truth pixels:", true_surface.sum())
    if true_surface.sum() == 0:
        print("No published lake polygons overlap this Sentinel tile.")
        return

    # Fast Pass Stride
    prob_map = predict_full_raster(model, ndwi_tif, device, tile_size=256, stride=256)

    for threshold in [0.1, 0.3, 0.5, 0.7]:
        pred_mask = prob_map >= threshold
        print("threshold:", threshold)
        print("Predicted pixels:", pred_mask.sum())
        print(metrics(pred_mask, true_surface))

### Debug Helper
"""
Primary issue I've been observing with using T22WDA Sentinel Tile, used for training the model is that
it showed 0 True pixel overlaps when tested against the target polygon dataset from Dunmire et al., 2021.
This is because the categorization of different datasets are geo-referenced and different in the experimental setup.

To troubleshoot overlap, I am iteratively looking for the overlapping sentinel tiles.

File: surface_CW2019.geojson
Original CRS: EPSG:4326
Original bounds: [-51.50309426  68.24387504 -46.78020172  71.96354124]
Projected bounds: [ 482432.24887416 7570274.23517506  670890.26255108 7984908.39338459]
Intersecting polygons: 99

File: surface_SW2019.geojson
Original CRS: EPSG:4326
Original bounds: [-50.25640092  61.38052479 -44.48736773  68.27413008]
Projected bounds: [ 534047.76036915 6809887.73292072  846600.01622653 7577640.1444553 ]
Intersecting polygons: 259

Tested on T22WEA

"""
def debug_surface_overlap(ndwi_tif, lakes_dir="benchmarkdata/lakes/surface", year="2018"):
    with rasterio.open(ndwi_tif) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        raster_poly = box(*raster_bounds)

    print("Raster CRS:", raster_crs)
    print("Raster bounds:", raster_bounds)

    for path in sorted(Path(lakes_dir).glob(f"surface_*{year}.geojson")):
        gdf = gpd.read_file(path)

        print("\nFile:", path.name)
        print("Original CRS:", gdf.crs)
        print("Original bounds:", gdf.total_bounds)

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        gdf = gdf.to_crs(raster_crs)
        intersects = gdf.intersects(raster_poly)

        print("Projected bounds:", gdf.total_bounds)
        print("Intersecting polygons:", int(intersects.sum()))

if __name__ == "__main__":
    main()
