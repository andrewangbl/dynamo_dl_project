"""ResNet-18 encoder adapted for 11-channel physical-field inputs.

Differences vs ``models/encoder/resnet.py``:
  * ``conv1`` expects ``in_channels=11`` (configurable) rather than 3.
  * NO ImageNet normalization (fixed ImageNet mean/std is meaningless here).
  * A learned-stats BatchNorm over the raw 11 input channels (``affine=False``)
    replaces ImageNet z-score: it tracks per-channel running mean/var so fields
    with vastly different native scales (e.g. near-constant ``concentration``
    vs strain-rate tensor components) do not drown each other out. Without this
    we observed representation collapse in DynaMo SSL on active_matter.
  * ``pretrained`` is FORCED to False -- project spec requires training from
    scratch.

Input/output contract matches the original ``resnet18`` wrapper so DynaMo's
pipeline is unchanged:
  * Accepts arbitrary leading dims (``... C H W``); any ``dims > 4`` are
    flattened to batch and restored at the end.
  * Returns ``(..., output_dim=512)`` features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class ResNet18Physics(nn.Module):
    def __init__(
        self,
        in_channels: int = 11,
        output_dim: int = 512,
        unit_norm: bool = False,
        input_norm: bool = True,
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)

        # Swap conv1 to accept `in_channels` physical-field channels.
        # Kernel/stride/padding match the original (7x7, stride=2, pad=3).
        if in_channels != 3:
            old = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=(old.bias is not None),
            )

        # Non-affine BN over raw input channels: learns per-channel running
        # z-score from the data itself, so we don't hard-code any stats.
        self.input_norm = (
            nn.BatchNorm2d(in_channels, affine=False) if input_norm else nn.Identity()
        )
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # drop FC
        self.flatten = nn.Flatten()
        self.in_channels = int(in_channels)
        self.output_dim = int(output_dim)
        self.unit_norm = bool(unit_norm)
        assert self.output_dim == 512, (
            "resnet18 produces 512-d features; override output_dim only if you "
            "also add a projection head."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = x.ndim
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            x = x.reshape(-1, *orig_shape[-3:])

        x = self.input_norm(x)
        out = self.resnet(x)
        out = self.flatten(out)
        if self.unit_norm:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)

        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], -1)
        return out


def resnet18_physics(
    in_channels: int = 11,
    output_dim: int = 512,
    unit_norm: bool = False,
    input_norm: bool = True,
) -> ResNet18Physics:
    """Factory matching the naming convention of models.encoder.resnet."""
    return ResNet18Physics(
        in_channels=in_channels,
        output_dim=output_dim,
        unit_norm=unit_norm,
        input_norm=input_norm,
    )
