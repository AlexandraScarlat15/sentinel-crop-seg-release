# src/training/model_unet_4levels.py
from __future__ import annotations

import torch
import torch.nn as nn


def _valid_groups(num_groups: int, channels: int) -> int:
    """Return a GroupNorm group count that divides channels."""
    g = min(num_groups, channels)
    while g > 1 and (channels % g) != 0:
        g -= 1
    return max(g, 1)


class DoubleConvGN(nn.Module):
    """(Conv -> GroupNorm -> ReLU) * 2, optional Dropout2d after the block."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        g = _valid_groups(num_groups, out_ch)

        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        ]

        # Spatial dropout can help regularize segmentation models.
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout2d(p=float(dropout)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet4LevelsGN(nn.Module):
    """
    U-Net with 4 encoder levels (down to 16x16 for 256x256 input).

    Input:  (B, in_channels, 256, 256)
    Output: (B, num_classes, 256, 256)

    Encoder resolutions:
      256 -> 128 -> 64 -> 32 -> 16 (bottleneck at 16)

    Uses GroupNorm (stable for small batch sizes) and optional Dropout in bottleneck/decoder.
    """
    def __init__(
        self,
        *,
        in_channels: int = 10,
        num_classes: int = 11,
        base: int = 32,
        num_groups: int = 8,
        dropout_bottleneck: float = 0.1,
        dropout_decoder: float = 0.0,
        up_mode: str = "bilinear",  # "bilinear" or "transpose"
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = DoubleConvGN(in_channels, base, num_groups=num_groups, dropout=0.0)
        self.enc2 = DoubleConvGN(base, base * 2, num_groups=num_groups, dropout=0.0)
        self.enc3 = DoubleConvGN(base * 2, base * 4, num_groups=num_groups, dropout=0.0)
        self.enc4 = DoubleConvGN(base * 4, base * 8, num_groups=num_groups, dropout=0.0)

        # Bottleneck
        self.bottleneck = DoubleConvGN(
            base * 8,
            base * 16,
            num_groups=num_groups,
            dropout=dropout_bottleneck,
        )

        # Upsampling choice
        if up_mode not in ("bilinear", "transpose"):
            raise ValueError(f"Invalid up_mode: {up_mode}. Use 'bilinear' or 'transpose'.")

        self.up_mode = up_mode
        if up_mode == "bilinear":
            self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            # ConvTranspose2d reduces reliance on interpolation; both are fine.
            self.up4 = nn.ConvTranspose2d(base * 16, base * 16, kernel_size=2, stride=2)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 8, kernel_size=2, stride=2)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 4, kernel_size=2, stride=2)
            self.up1 = nn.ConvTranspose2d(base * 2, base * 2, kernel_size=2, stride=2)

        # Decoder
        self.dec4 = DoubleConvGN(base * 16 + base * 8, base * 8, num_groups=num_groups, dropout=dropout_decoder)
        self.dec3 = DoubleConvGN(base * 8 + base * 4, base * 4, num_groups=num_groups, dropout=dropout_decoder)
        self.dec2 = DoubleConvGN(base * 4 + base * 2, base * 2, num_groups=num_groups, dropout=dropout_decoder)
        self.dec1 = DoubleConvGN(base * 2 + base, base, num_groups=num_groups, dropout=dropout_decoder)

        # Head
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)        # (B, base, 256, 256)
        p1 = self.pool(x1)       # (B, base, 128, 128)

        x2 = self.enc2(p1)       # (B, 2b, 128, 128)
        p2 = self.pool(x2)       # (B, 2b, 64, 64)

        x3 = self.enc3(p2)       # (B, 4b, 64, 64)
        p3 = self.pool(x3)       # (B, 4b, 32, 32)

        x4 = self.enc4(p3)       # (B, 8b, 32, 32)
        p4 = self.pool(x4)       # (B, 8b, 16, 16)

        # Bottleneck
        b = self.bottleneck(p4)  # (B, 16b, 16, 16)

        # Decoder
        u4 = self.up4(b)                 # (B, 16b, 32, 32)
        u4 = torch.cat([u4, x4], dim=1)  # (B, 16b+8b, 32, 32)
        d4 = self.dec4(u4)               # (B, 8b, 32, 32)

        u3 = self.up3(d4)                # (B, 8b, 64, 64)
        u3 = torch.cat([u3, x3], dim=1)  # (B, 8b+4b, 64, 64)
        d3 = self.dec3(u3)               # (B, 4b, 64, 64)

        u2 = self.up2(d3)                # (B, 4b, 128, 128)
        u2 = torch.cat([u2, x2], dim=1)  # (B, 4b+2b, 128, 128)
        d2 = self.dec2(u2)               # (B, 2b, 128, 128)

        u1 = self.up1(d2)                # (B, 2b, 256, 256)
        u1 = torch.cat([u1, x1], dim=1)  # (B, 2b+b, 256, 256)
        d1 = self.dec1(u1)               # (B, b, 256, 256)

        logits = self.head(d1)           # (B, num_classes, 256, 256)
        return logits


if __name__ == "__main__":
    model = UNet4LevelsGN(in_channels=10, num_classes=11, base=32, dropout_bottleneck=0.1)
    x = torch.randn(2, 10, 256, 256)
    y = model(x)
    print(y.shape)  # torch.Size([2, 11, 256, 256])