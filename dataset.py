"""
Workflow: npz -> NumPy array -> torch tensor -> U-Net -> logits -> loss -> gradients
"""

from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LakeTileDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform # for later augmentation

        # extraction
        image_files = sorted(self.images_dir.glob("*.npz"))
        mask_files = sorted(self.masks_dir.glob("*.npz"))

        # Pair by stem
        image_stems = {f.stem for f in image_files}
        mask_stems = {f.stem for f in mask_files}
        common_stems = sorted(image_stems & mask_stems)

        if not common_stems:
            raise RuntimeError("No matching image/mask stems found!")

        self.stems = common_stems

    def __len__(self) -> int:
        return len(self.stems)

    def _load_npz_first_array(self, path: Path) -> np.ndarray:
        data = np.load(path)
        key = list(data.files)[0]
        return data[key]

    # Heart of the dataset file
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        img_path = self.images_dir / f"{stem}.npz"
        mask_path = self.masks_dir / f"{stem}.npz"

        img_np = self._load_npz_first_array(img_path)   # (1, H, W)
        mask_np = self._load_npz_first_array(mask_path) # (H, W)

        # Ensure shapes (1, H, W)
        if mask_np.ndim == 2:
            mask_np = mask_np[None, ...]  # (1, H, W)

        # convert to tensors
        img = torch.from_numpy(img_np).float()          # (1, H, W)
        mask = torch.from_numpy(mask_np).float()        # (1, H, W)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask
