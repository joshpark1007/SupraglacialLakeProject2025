"""
File: train.py
Purpose: Train a small U-Net segmentation model on NDWI + lake-mask tiles.
         Splits the dataset into train/validation sets, runs a few epochs of
         BCEWithLogitsLoss optimization, reports losses, and saves weights
         to unet_lakes.pth.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import LakeTileDataset
from unet import UNetSmall


def main():
    # 1) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) Dataset & split (optional: train/val)
    full_dataset = LakeTileDataset("data/tiles/images", "data/tiles/masks")
    print("Total tiles:", len(full_dataset))

    # simple 90/10 split
    val_size = max(1, len(full_dataset) // 10)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print("Train tiles:", len(train_dataset))
    print("Val tiles:  ", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 3) Model, loss, optimizer
    model = UNetSmall(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 3  # start small just to see it move

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)          # (B,1,256,256)
            masks = masks.to(device)        # (B,1,256,256)

            optimizer.zero_grad()
            logits = model(imgs)            # (B,1,256,256)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        avg_train_loss = running_loss / len(train_dataset)

        # quick val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)

        avg_val_loss = val_loss / len(val_dataset)

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"- train loss: {avg_train_loss:.4f} "
            f"- val loss: {avg_val_loss:.4f}"
        )

    # 4) Save model weights
    torch.save(model.state_dict(), "unet_lakes.pth")
    print("✅ Training complete, model saved to unet_lakes.pth")


if __name__ == "__main__":
    main()
