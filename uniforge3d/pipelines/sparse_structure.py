"""Stage1 model adaptation and occupancy decoding."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..models.vaes.sparse_structure import SparseStructureDecoder
from .samplers import FlowEulerSampler


def decoded_coordinates(occupancy_logits: torch.Tensor):
    """Threshold decoder logits and return occupied coords32 and coords64."""
    decoded64 = occupancy_logits > 0
    if decoded64.shape[-1] != 64:
        if decoded64.shape[-1] < 64 or decoded64.shape[-1] % 64:
            raise RuntimeError(f"Cannot reduce Stage1 decoder output {tuple(decoded64.shape)} to 64^3")
        ratio = decoded64.shape[-1] // 64
        decoded64 = F.max_pool3d(decoded64.float(), kernel_size=ratio, stride=ratio) > 0.5
    coords64 = torch.argwhere(decoded64)[:, [0, 2, 3, 4]].int()
    decoded32 = F.max_pool3d(decoded64.float(), kernel_size=2, stride=2) > 0.5
    coords32 = torch.argwhere(decoded32)[:, [0, 2, 3, 4]].int()
    if len(coords32) == 0 or len(coords64) == 0:
        raise RuntimeError("Stage1 decoded an empty sparse structure")
    return coords32, coords64


@torch.no_grad()
def sample_sparse_structure(
    model: torch.nn.Module,
    noise: torch.Tensor,
    vggt_images: torch.Tensor,
    dino_features: torch.Tensor,
    *,
    steps: int,
    guidance_strength: float,
    guidance_interval: tuple[float, float],
    guidance_rescale: float,
    rescale_t: float,
    sigma_min: float,
) -> torch.Tensor:
    """Sample Stage1 with explicit conditional and unconditional predictions."""
    def predict(sample: torch.Tensor, time: float, conditioned: bool) -> torch.Tensor:
        timestep = torch.full(
            (sample.shape[0],),
            time,
            device=sample.device,
            dtype=torch.float32,
        )
        if conditioned:
            return model(sample, timestep, vggt_images, dino_features)
        return model.forward_unconditional(sample, timestep)

    sampler = FlowEulerSampler(sigma_min=float(sigma_min))
    return sampler.sample(
        noise,
        predict,
        steps=int(steps),
        guidance_strength=float(guidance_strength),
        guidance_interval=tuple(float(value) for value in guidance_interval),
        guidance_rescale=float(guidance_rescale),
        guidance_as_difference=True,
        rescale_t=float(rescale_t),
        tqdm_desc="Sparse structure sampling",
    ).samples


@torch.no_grad()
def decode_sparse_structure(
    decoder: SparseStructureDecoder,
    normalized_latent: torch.Tensor,
    normalization: dict[str, Any] | None,
    *,
    batch_size: int = 1,
) -> torch.Tensor:
    """Denormalize and decode logits; the pipeline places the decoder."""
    # A decoder keeps its input and output layers in fp32 and casts only its torso to
    # self.dtype, so its input has to arrive in fp32 whatever the sampler ran in. The
    # latent is fp16 when the flow model ran under fp16 autocast and this decoder has
    # no normalization to promote it back.
    value = normalized_latent.float()
    if normalization is not None:
        mean = torch.tensor(normalization["mean"], dtype=torch.float32, device="cpu")
        std = torch.tensor(normalization["std"], dtype=torch.float32, device="cpu")
        mean = mean.to(value.device).view(1, -1, 1, 1, 1)
        std = std.to(value.device).view(1, -1, 1, 1, 1)
        value = value * std + mean
    decoded = [
        decoder(value[start : start + batch_size])
        for start in range(0, len(value), batch_size)
    ]
    return torch.cat(decoded, dim=0)
