"""Shared frozen DINOv3 encoder for Stage1 and structured-latent conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DINOv3ViTModel


class DinoV3Encoder(nn.Module):
    """Run the exact DINOv3 feature path used by the released checkpoints.

    Four-dimensional inputs preserve the model output dtype used by the shape
    conditioners. Five-dimensional multiview inputs are restored to
    ``[B, V, N, C]`` and converted to float32 as required by Stage1.
    """

    def __init__(self, model_name: str, **download_kwargs):
        super().__init__()
        self.model = DINOv3ViTModel.from_pretrained(
            model_name, dtype=torch.float32, **download_kwargs,
        ).eval()
        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        multiview = images.ndim == 5
        if multiview:
            batch, views, channels, height, width = images.shape
            flat = images.reshape(batch * views, channels, height, width)
        elif images.ndim == 4:
            batch = views = None
            flat = images
            height, width = flat.shape[-2:]
        else:
            raise ValueError(f"Expected DINO images [B,3,H,W] or [B,V,3,H,W], got {tuple(images.shape)}")

        if flat.shape[1] != 3 or height != width:
            raise ValueError(f"Expected square RGB DINO input, got {tuple(images.shape)}")

        mean = self._mean.to(device=flat.device, dtype=flat.dtype)
        std = self._std.to(device=flat.device, dtype=flat.dtype)
        value = ((flat - mean) / std).to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden = self.model.embeddings(value, bool_masked_pos=None)
        position = self.model.rope_embeddings(value)
        for layer in self.model.layer:
            hidden = layer(hidden, position_embeddings=position)
        hidden = F.layer_norm(hidden, hidden.shape[-1:])

        if multiview:
            return hidden.reshape(batch, views, hidden.shape[1], hidden.shape[2]).float()
        return hidden
