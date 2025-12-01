import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    (Conv2d -> ReLU -> Conv2d -> ReLU)
    Keeps spatial size, changes channels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)   # 256 -> 128

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)   # 128 -> 64

        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)   # 64 -> 32

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 32 -> 64
        self.dec3 = DoubleConv(256, 128)  # 128 (up) + 128 (skip)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 64 -> 128
        self.dec2 = DoubleConv(128, 64)   # 64 (up) + 64 (skip)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # 128 -> 256
        self.dec1 = DoubleConv(64, 32)    # 32 (up) + 32 (skip)

        # Final 1×1 conv → 1 channel (lake vs not-lake)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.down1(x)
        x2 = self.pool1(x1)

        x3 = self.down2(x2)
        x4 = self.pool2(x3)

        x5 = self.down3(x4)
        x6 = self.pool3(x5)

        # Bottleneck
        x_bottleneck = self.bottleneck(x6)

        # Decoder with skip connections
        x = self.up3(x_bottleneck)
        x = torch.cat([x, x5], dim=1)   # concat along channel axis
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        logits = self.final_conv(x)     # (B, 1, H, W)
        return logits
