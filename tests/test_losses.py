import torch
import pytest

from src.training.losses import CrossEntropyDiceLoss, dice_loss_multiclass


def test_cross_entropy_dice_loss_returns_finite_scalar_and_backpropagates():
    torch.manual_seed(42)

    logits = torch.randn(2, 11, 8, 8, requires_grad=True)
    target = torch.randint(0, 11, (2, 8, 8), dtype=torch.long)

    criterion = CrossEntropyDiceLoss(
        num_classes=11,
        dice_weight=0.4,
        ignore_index=None,
        class_weights=[0.25, 1.0, 1.5, 0.75, 1.0, 3.0, 2.5, 1.5, 2.0, 3.0, 1.0],
        dice_ignore_background=True,
    )

    loss = criterion(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_cross_entropy_dice_loss_supports_ignore_index():
    torch.manual_seed(42)

    logits = torch.randn(1, 11, 8, 8, requires_grad=True)
    target = torch.randint(0, 11, (1, 8, 8), dtype=torch.long)
    target[:, 0, 0] = 255

    criterion = CrossEntropyDiceLoss(
        num_classes=11,
        dice_weight=0.4,
        ignore_index=255,
        dice_ignore_background=True,
    )

    loss = criterion(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dice_loss_rejects_invalid_logits_shape():
    logits = torch.randn(11, 8, 8)
    target = torch.randint(0, 11, (1, 8, 8), dtype=torch.long)

    with pytest.raises(ValueError, match="logits must be"):
        dice_loss_multiclass(logits, target, num_classes=11)
