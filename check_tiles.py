"""
check_tiles.py for sanity check code
"""

from pathlib import Path
import numpy as np

IMAGES_DIR = Path("data/tiles/images")
MASKS_DIR  = Path("data/tiles/masks")

def load_first_array(npz_path):
    data = np.load(npz_path)
    # Use the first array stored in the npz file
    key = list(data.files)[0]
    return data[key]

def main():
    image_files = sorted(IMAGES_DIR.glob("*.npz"))
    mask_files  = sorted(MASKS_DIR.glob("*.npz"))

    print(f"Found {len(image_files)} image tiles")
    print(f"Found {len(mask_files)} mask tiles")

    if len(image_files) == 0:
        print("⚠️ No image tiles found. Check your paths or extensions.")
        return
    if len(mask_files) == 0:
        print("⚠️ No mask tiles found. Check your paths or extensions.")
        return

    # Pair by stem (filename without extension)
    image_stems = {f.stem for f in image_files}
    mask_stems  = {f.stem for f in mask_files}

    missing_masks  = image_stems - mask_stems
    missing_images = mask_stems - image_stems

    if missing_masks:
        print("⚠️ These image stems have no matching mask:")
        for s in list(missing_masks)[:10]:
            print("   ", s)
    if missing_images:
        print("⚠️ These mask stems have no matching image:")
        for s in list(missing_images)[:10]:
            print("⚠️ These mask stems have no matching image:")
            print("   ", s)

    if not missing_masks and not missing_images:
        print("✅ Every image has a matching mask (by stem).")

    # Inspect one sample pair
    sample_img = image_files[0]
    sample_mask = MASKS_DIR / (sample_img.stem + ".npz")

    print("\nInspecting sample pair:")
    print("  Image:", sample_img)
    print("  Mask: ", sample_mask)

    img = load_first_array(sample_img)
    mask = load_first_array(sample_mask)

    print("  Image shape:", img.shape)
    print("  Mask shape: ", mask.shape)
    print("  Image dtype:", img.dtype)
    print("  Mask dtype: ", mask.dtype)

    if img.shape[-2:] == mask.shape[-2:]:
        print("✅ Spatial dimensions match (H, W).")
    else:
        print("⚠️ Spatial dimension mismatch between image and mask!")

    # quick peek at mask values to ensure it's reasonable (0/1 or small ints)
    unique_mask_vals = np.unique(mask)
    print("  Unique mask values (first 10):", unique_mask_vals[:10])

if __name__ == "__main__":
    main()
