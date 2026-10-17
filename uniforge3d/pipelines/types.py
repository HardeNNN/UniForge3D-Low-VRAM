"""In-memory stage boundaries; none of these values are serialized."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseStructure:
    coords32: torch.Tensor
    coords64: torch.Tensor


@dataclass(frozen=True)
class NormalizedShapeLatent:
    """Normalized flow output, before VAE denormalization."""

    coords: torch.Tensor
    feats: torch.Tensor
