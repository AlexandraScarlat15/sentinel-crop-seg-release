import torch

from sentinel_crop_seg.core.training.model_unet_4levels import UNet4LevelsGN


def test_unet_forward_output_shape():
    model = UNet4LevelsGN(
        in_channels=10,
        num_classes=11,
        base=8,
        dropout_bottleneck=0.0,
        dropout_decoder=0.0,
        up_mode="bilinear",
    )

    model.eval()

    x = torch.randn(2, 10, 64, 64)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 11, 64, 64)
    assert torch.isfinite(y).all()
