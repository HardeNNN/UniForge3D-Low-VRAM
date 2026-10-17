"""Sparse shape latent sampling and mesh decoding."""

from __future__ import annotations

from typing import Any

import torch

from ..models.vaes.flexi_dual_grid import FlexiDualGridVaeDecoder
from ..modules import sparse as sp
from .samplers import FlowEulerSampler, model_predictor


@torch.no_grad()
def sample_shape_latent(
    model: torch.nn.Module,
    latent_coords: torch.Tensor,
    condition: dict[str, Any],
    sampling: dict[str, Any],
    *,
    description: str = "Shape sampling",
    dtype: torch.dtype = torch.bfloat16,
) -> sp.SparseTensor:
    """Sample latent features at fixed coordinates using an encoded condition."""
    # Keep noise on the CPU RNG stream used by the original inference path.
    features = torch.randn(len(latent_coords), int(model.in_channels), device="cpu")
    features = features.to(device=latent_coords.device, dtype=dtype)
    noise = sp.SparseTensor(feats=features, coords=latent_coords)
    condition["cond"] = model.prepare_condition(noise, condition["cond"])
    condition["neg_cond"] = model.prepare_condition(noise, condition["neg_cond"])

    sampler = FlowEulerSampler(**sampling["args"])
    predict = model_predictor(model, condition["cond"], condition["neg_cond"])

    with torch.autocast(device_type=latent_coords.device.type, dtype=dtype):
        return sampler.sample(
            noise,
            predict,
            **sampling["params"],
            verbose=True,
            tqdm_desc=description,
        ).samples


@torch.no_grad()
def decode_shape_latent(
    decoder: FlexiDualGridVaeDecoder,
    normalized_latent,
    normalization: dict[str, Any] | None,
):
    """Denormalize and decode meshes; the pipeline places the decoder."""
    # Decoders keep fp32 input and output layers, so features enter in fp32 whether or
    # not a normalization is present to promote them.
    value = normalized_latent.replace(normalized_latent.feats.float())
    if normalization is not None:
        mean = torch.tensor(normalization["mean"], dtype=torch.float32, device="cpu")
        std = torch.tensor(normalization["std"], dtype=torch.float32, device="cpu")
        mean = mean.to(value.device).view(1, -1)
        std = std.to(value.device).view(1, -1)
        value = value.replace(value.feats * std + mean)
    return decoder(value)
