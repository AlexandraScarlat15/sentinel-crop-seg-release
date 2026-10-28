from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_multiclass(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    ignore_index: int | None = None,
    dice_ignore_background: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"logits must be (B,C,H,W), got {tuple(logits.shape)}")

    if target.ndim != 3:
        raise ValueError(f"target must be (B,H,W), got {tuple(target.shape)}")

    probs = torch.softmax(logits, dim=1)
    _, c, _, _ = probs.shape

    if c != num_classes:
        raise ValueError(f"num_classes={num_classes} but logits has C={c}")

    tgt = target

    if ignore_index is not None:
        valid = tgt != ignore_index
        tgt = torch.where(valid, tgt, torch.zeros_like(tgt))
    else:
        valid = None

    one_hot = (
        F.one_hot(tgt.to(torch.int64), num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .to(probs.dtype)
    )

    if valid is not None:
        valid_f = valid.unsqueeze(1).to(probs.dtype)
        probs = probs * valid_f
        one_hot = one_hot * valid_f

    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    denom = torch.sum(probs, dims) + torch.sum(one_hot, dims)

    dice = (2.0 * intersection + eps) / (denom + eps)

    if dice_ignore_background:
        if num_classes <= 1:
            raise ValueError("Cannot ignore background when num_classes <= 1.")

        dice = dice[1:]

    return 1.0 - dice.mean()


class CrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        dice_weight: float = 0.5,
        ignore_index: int | None = None,
        class_weights: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        dice_ignore_background: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.ignore_index = ignore_index
        self.dice_ignore_background = bool(dice_ignore_background)

        if not (0.0 <= self.dice_weight <= 1.0):
            raise ValueError(f"dice_weight must be in [0, 1], got {self.dice_weight}")

        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)

            if weights.ndim != 1:
                raise ValueError(f"class_weights must be 1D, got shape {tuple(weights.shape)}")

            if weights.numel() != self.num_classes:
                raise ValueError(
                    f"class_weights must have {self.num_classes} values, got {weights.numel()}"
                )

            self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights

        if weight is not None:
            weight = weight.to(device=logits.device, dtype=logits.dtype)

        if self.ignore_index is None:
            ce = F.cross_entropy(logits, target, weight=weight)
        elif bool(torch.any(target != int(self.ignore_index))):
            ce = F.cross_entropy(
                logits,
                target,
                weight=weight,
                ignore_index=int(self.ignore_index),
            )
        else:
            # A batch in which every pixel is ignored makes the mean cross
            # entropy 0/0. Returning NaN would propagate into the weights and
            # destroy the run, so such a batch contributes no gradient instead.
            # Unreachable when ignore_index is None, so the no-ignore baseline
            # is bit-for-bit unaffected.
            ce = (logits.sum() * 0.0).to(logits.dtype)

        d = dice_loss_multiclass(
            logits,
            target,
            self.num_classes,
            ignore_index=self.ignore_index,
            dice_ignore_background=self.dice_ignore_background,
        )

        return ((1.0 - self.dice_weight) * ce) + (self.dice_weight * d)