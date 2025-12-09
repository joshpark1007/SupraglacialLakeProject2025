"""
File: train_sanity.py
Purpose: Lightweight sanity check for the tiling and model setup. Loads a
         batch from LakeTileDataset, runs a forward pass through UNetSmall,
         computes an initial loss, and performs one optimizer step to verify
         that training can proceed without errors.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LakeTileDataset
from unet import UNetSmall


def main():

    print("Loading dataset...")
    dataset = LakeTileDataset("data/tiles/images", "data/tiles/masks")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("Dataset size:", len(dataset))

    # Get one batch
    imgs, masks = next(iter(loader))
    print("Batch shapes:", imgs.shape, masks.shape)

    # Create model
    model = UNetSmall(in_channels=1, out_channels=1)
    print("Model created!")

    # Forward pass
    logits = model(imgs)  # output shape (B,1,256,256)
    print("Logits shape:", logits.shape)

    # Loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, masks)
    print("Initial loss:", loss.item())

    # One training step (optional)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("One training step completed!")


if __name__ == "__main__":
    main()
