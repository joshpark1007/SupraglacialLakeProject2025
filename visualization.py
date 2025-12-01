# visualize.py
# preliminary code
import torch
import matplotlib.pyplot as plt

from dataset import LakeTileDataset
from unet import UNetSmall


def visualize_samples(indices=None, threshold=0.5, num_samples=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load dataset
    dataset = LakeTileDataset("data/tiles/images", "data/tiles/masks")
    print("Dataset size:", len(dataset))

    # 2) Load trained model
    model = UNetSmall(in_channels=1, out_channels=1).to(device)
    state_dict = torch.load("unet_lakes.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded model from unet_lakes.pth")

    # 3) Choose some indices to visualize
    if indices is None:
        # just take evenly spaced samples if not provided
        step = max(1, len(dataset) // num_samples)
        indices = [i * step for i in range(num_samples)]

    print("Visualizing indices:", indices)

    for idx in indices:
        img, mask = dataset[idx]      # img: (1,H,W), mask: (1,H,W)
        img_batch = img.unsqueeze(0).to(device)  # (1,1,H,W)

        with torch.no_grad():
            logits = model(img_batch)          # (1,1,H,W)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H,W)

        ndwi = img[0].numpy()                  # (H,W)
        true_mask = mask[0].numpy()            # (H,W)
        pred_binary = (probs > threshold).astype(float)

        # 4) Plot
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        fig.suptitle(f"Tile index {idx}", fontsize=14)

        axes[0].imshow(ndwi, cmap="gray")
        axes[0].set_title("NDWI tile")
        axes[0].axis("off")

        axes[1].imshow(true_mask, cmap="gray")
        axes[1].set_title("True mask")
        axes[1].axis("off")

        im2 = axes[2].imshow(probs, cmap="viridis", vmin=0, vmax=1)
        axes[2].set_title("Predicted prob")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(ndwi, cmap="gray")
        axes[3].imshow(pred_binary, alpha=0.4)
        axes[3].set_title(f"Pred > {threshold}")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # change indices if you want specific tiles
    visualize_samples(indices=[200, 600, 1200, 2500])
