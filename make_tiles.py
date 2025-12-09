"""
File: make_tiles.py
Purpose: Convert aligned NDWI and supraglacial lake mask rasters into
         overlapping tiles for U-Net training. Searches for matching NDWI /
         lake-mask pairs, slides a window across them, and saves each pair
         as compressed .npz tiles under data/tiles/images and data/tiles/masks.
"""

import os
import argparse
import glob

import numpy as np
import rasterio

def find_pairs(in_dir, ndwi_suffix="_ndwi_0.25.tif",
               lake_suffix="_lake_ndwi0.25_dem0.tif"):
    """Find (ndwi_path, lake_path) pairs based on naming scheme."""
    ndwi_paths = sorted(
        glob.glob(os.path.join(in_dir, f"*{ndwi_suffix}"))
    )
    pairs = []
    for ndwi_path in ndwi_paths:
        base = os.path.basename(ndwi_path).replace(ndwi_suffix, "")
        lake_name = f"{base}{lake_suffix}"
        lake_path = os.path.join(in_dir, lake_name)
        if os.path.exists(lake_path):
            pairs.append((ndwi_path, lake_path))
        else:
            print(f"⚠️ No lake mask for {ndwi_path}, expected {lake_path}")
    print(f"Found {len(pairs)} NDWI/lake pairs.")
    return pairs

def tile_pair(ndwi_path, lake_path, out_img_dir, out_mask_dir,
              tile_size=256, stride=128,
              keep_empty=False, max_empty_frac=0.2):
    """Cut a single NDWI+mask pair into tiles and save as .npz."""
    with rasterio.open(ndwi_path) as src_img, rasterio.open(lake_path) as src_mask:
        if (src_img.width != src_mask.width or
            src_img.height != src_mask.height or
            src_img.transform != src_mask.transform):
            print(f"⚠️ Shape/transform mismatch, skipping:\n  {ndwi_path}\n  {lake_path}")
            return 0

        h, w = src_img.height, src_img.width
        ndwi = src_img.read(1)   # (H, W)
        mask = src_mask.read(1)  # (H, W)

    # normalize NDWI to something reasonable (-1..1)
    ndwi_clipped = np.clip(ndwi, -1.0, 1.0).astype(np.float32)

    basename = os.path.basename(ndwi_path).replace("_ndwi_0.25.tif", "")
    count = 0
    empty_tiles = 0

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            ndwi_tile = ndwi_clipped[y:y+tile_size, x:x+tile_size]
            mask_tile = mask[y:y+tile_size, x:x+tile_size]

            if ndwi_tile.shape != (tile_size, tile_size):
                continue

            # Decide whether to keep empty tiles
            if mask_tile.sum() == 0:
                empty_tiles += 1
                if not keep_empty:
                    continue

            tile_id = f"{basename}_y{y}_x{x}"
            img_out = os.path.join(out_img_dir, f"{tile_id}.npz")
            mask_out = os.path.join(out_mask_dir, f"{tile_id}.npz")

            # add channel dimension for image: (1, H, W)
            np.savez_compressed(img_out, ndwi=ndwi_tile[None, ...])
            np.savez_compressed(mask_out, mask=mask_tile.astype(np.uint8))

            count += 1

    print(f"🧩 Tiled {basename}: {count} tiles (empty tiles seen: {empty_tiles})")
    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory with NDWI + lake rasters")
    ap.add_argument("--out-dir", required=True, help="Output root for tiles")
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--keep-empty", action="store_true",
                    help="Keep tiles with no lake pixels (mask sum == 0)")
    args = ap.parse_args()

    in_dir = args.in_dir
    out_root = args.out_dir
    out_img_dir = os.path.join(out_root, "images")
    out_mask_dir = os.path.join(out_root, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    pairs = find_pairs(in_dir)
    total_tiles = 0
    for ndwi_path, lake_path in pairs:
        total_tiles += tile_pair(
            ndwi_path, lake_path,
            out_img_dir, out_mask_dir,
            tile_size=args.tile_size,
            stride=args.stride,
            keep_empty=args.keep_empty
        )

    print(f"\n✅ Done. Total tiles written: {total_tiles}")
    print(f"   Images: {out_img_dir}")
    print(f"   Masks:  {out_mask_dir}")

if __name__ == "__main__":
    main()
