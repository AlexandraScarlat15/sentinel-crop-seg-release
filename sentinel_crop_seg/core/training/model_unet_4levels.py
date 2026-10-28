"""Four-level U-Net with GroupNorm for multichannel crop segmentation."""
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
    """Apply two convolution/GroupNorm/ReLU layers with optional dropout.

    Args:
        in_ch: Number of input feature channels.
        out_ch: Number of output feature channels.
        num_groups: Preferred number of GroupNorm groups.
        dropout: Spatial dropout probability after the second activation.
    """
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

        if dropout and dropout > 0.0:
            layers.append(nn.Dropout2d(p=float(dropout)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return block features for the input tensor."""
        return self.net(x)


class UNet4LevelsGN(nn.Module):
    """Segment multichannel patches with four downsampling and decoding levels.

    GroupNorm supports small training batches. Input height and width must be
    compatible with four successive pooling operations; logits retain the
    input spatial dimensions and have `num_classes` channels.

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of output class logits per pixel.
        base: Feature channels in the first encoder level.
        num_groups: Preferred GroupNorm groups per convolutional block.
        dropout_bottleneck: Spatial dropout probability in the bottleneck.
        dropout_decoder: Spatial dropout probability in decoder blocks.
        up_mode: Upsampling method, `bilinear` or `transpose`.

    Raises:
        ValueError: If `up_mode` is unsupported.
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
        up_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConvGN(in_channels, base, num_groups=num_groups, dropout=0.0)
        self.enc2 = DoubleConvGN(base, base * 2, num_groups=num_groups, dropout=0.0)
        self.enc3 = DoubleConvGN(base * 2, base * 4, num_groups=num_groups, dropout=0.0)
        self.enc4 = DoubleConvGN(base * 4, base * 8, num_groups=num_groups, dropout=0.0)

        self.bottleneck = DoubleConvGN(
            base * 8,
            base * 16,
            num_groups=num_groups,
            dropout=dropout_bottleneck,
        )

        if up_mode not in ("bilinear", "transpose"):
            raise ValueError(f"Invalid up_mode: {up_mode}. Use 'bilinear' or 'transpose'.")

        self.up_mode = up_mode
        if up_mode == "bilinear":
            self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.up4 = nn.ConvTranspose2d(base * 16, base * 16, kernel_size=2, stride=2)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 8, kernel_size=2, stride=2)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 4, kernel_size=2, stride=2)
            self.up1 = nn.ConvTranspose2d(base * 2, base * 2, kernel_size=2, stride=2)

        self.dec4 = DoubleConvGN(base * 16 + base * 8, base * 8, num_groups=num_groups, dropout=dropout_decoder)
        self.dec3 = DoubleConvGN(base * 8 + base * 4, base * 4, num_groups=num_groups, dropout=dropout_decoder)
        self.dec2 = DoubleConvGN(base * 4 + base * 2, base * 2, num_groups=num_groups, dropout=dropout_decoder)
        self.dec1 = DoubleConvGN(base * 2 + base, base, num_groups=num_groups, dropout=dropout_decoder)

        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-class logits with the same spatial extent as `x`."""
        x1 = self.enc1(x)
        p1 = self.pool(x1)

        x2 = self.enc2(p1)
        p2 = self.pool(x2)

        x3 = self.enc3(p2)
        p3 = self.pool(x3)

        x4 = self.enc4(p3)
        p4 = self.pool(x4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        u4 = torch.cat([u4, x4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3, x3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(u1)

        logits = self.head(d1)
        return logits


if __name__ == "__main__":
    model = UNet4LevelsGN(in_channels=10, num_classes=11, base=32, dropout_bottleneck=0.1)
    x = torch.randn(2, 10, 256, 256)
    y = model(x)
    print(y.shape)