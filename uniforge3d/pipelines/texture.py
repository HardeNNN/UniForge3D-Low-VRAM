"""PBR sampling, decoding and baking; model placement belongs to the pipeline."""

from __future__ import annotations

from typing import Any

import torch
import trimesh

from ..modules.sparse import SparseTensor
from ..utils.preprocessing import InputBatch
from ..utils.texture_baking import to_glb
from .samplers import FlowEulerSampler, model_predictor
from .types import NormalizedShapeLatent


def denormalize_shape_latent(
    config: dict[str, Any],
    normalized_latent: NormalizedShapeLatent,
    device: torch.device,
) -> SparseTensor:
    """Convert normalized Shape1024 flow output to the shape VAE's latent space."""
    coords, feats = normalized_latent.coords, normalized_latent.feats
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected Shape1024 coords [N,4], got {tuple(coords.shape)}")
    if feats.ndim != 2 or feats.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Shape1024 feature/coordinate mismatch: feats={tuple(feats.shape)}, "
            f"coords={tuple(coords.shape)}"
        )
    spatial = coords[:, 1:]
    if spatial.numel() == 0 or int(spatial.min()) < 0 or int(spatial.max()) >= 64:
        raise ValueError("Shape1024 coordinates are outside [0, 64)")
    if not torch.isfinite(feats).all():
        raise ValueError("Shape1024 features contain NaN or Inf")
    normalization = config["shape_slat_normalization"]
    mean = torch.tensor(normalization["mean"], dtype=torch.float32)[None]
    std = torch.tensor(normalization["std"], dtype=torch.float32)[None]
    # Shape sampling returns normalized flow output; decoding consumes VAE latents.
    feats = feats.float() * std + mean
    return SparseTensor(feats=feats, coords=coords.int()).to(device)


@torch.no_grad()
def sample_texture(
    model: torch.nn.Module,
    config: dict[str, Any],
    conditioner,
    shape_vae_latent: SparseTensor,
    batch: InputBatch,
    dino_features: torch.Tensor,
    view_assignment: dict[str, torch.Tensor],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> SparseTensor:
    shape_norm = config["shape_slat_normalization"]
    mean = torch.tensor(shape_norm["mean"], device=device)[None]
    std = torch.tensor(shape_norm["std"], device=device)[None]
    # Preserve the original VAE -> normalized flow conversion and its rounding.
    shape_condition = shape_vae_latent.replace((shape_vae_latent.feats - mean) / std)
    coords = shape_condition.coords[:, 1:].to(dtype=torch.int32)
    if coords.numel() == 0 or int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise RuntimeError("Shape1024 latent does not contain valid coords64")

    condition = conditioner.encode(
        model.adapter,
        dino_features=dino_features.to(device),
        vggt_images=batch.vggt_images.to(device),
        structure_conditioned_coords=shape_condition.coords,
        input_masks=batch.input_masks.to(device),
        view_assignment=view_assignment,
    )
    noise_channels = model.in_channels - shape_vae_latent.feats.shape[1]
    noise = shape_condition.replace(
        torch.randn(len(coords), noise_channels, device="cpu").to(coords.device, dtype)
    )
    condition["cond"] = model.prepare_condition(noise, condition["cond"])
    condition["neg_cond"] = model.prepare_condition(noise, condition["neg_cond"])
    sampling = config["sampler"]
    sampler = FlowEulerSampler(**sampling["args"])
    predict = model_predictor(
        model,
        condition["cond"],
        condition["neg_cond"],
        concat_cond=shape_condition,
    )
    with torch.autocast(device_type=device.type, dtype=dtype):
        texture_slat = sampler.sample(
            noise,
            predict,
            **sampling["params"],
            verbose=True,
            tqdm_desc="PBR texture sampling",
        ).samples
    texture_norm = config["texture_slat_normalization"]
    mean = torch.tensor(texture_norm["mean"], device=device)[None]
    std = torch.tensor(texture_norm["std"], device=device)[None]
    return texture_slat.replace(texture_slat.feats * std + mean)


@torch.no_grad()
def decode_shape_and_texture(
    shape_decoder: torch.nn.Module,
    texture_decoder: torch.nn.Module,
    shape_vae_latent: SparseTensor,
    texture_vae_latent: SparseTensor,
):
    """Decode with shared subdivisions and fill holes in the reference mesh."""
    # Decoders keep fp32 input and output layers; cast both latents at their boundary.
    shape_vae_latent = shape_vae_latent.replace(shape_vae_latent.feats.float())
    texture_vae_latent = texture_vae_latent.replace(texture_vae_latent.feats.float())
    meshes, subdivisions = shape_decoder(shape_vae_latent, return_subs=True)
    voxels = texture_decoder(texture_vae_latent, guide_subs=subdivisions) * 0.5 + 0.5
    meshes[0].fill_holes()
    return meshes[0], voxels[0]


@torch.no_grad()
def remesh_and_bake_texture(
    texture_config: dict[str, Any],
    mesh_config: dict[str, Any],
    mesh,
    pbr_voxel: SparseTensor,
) -> trimesh.Trimesh:
    """Use the raw decoded surface and the official example's export settings."""
    mesh.simplify(int(mesh_config["pre_simplify_target"]))
    return to_glb(
        mesh.vertices,
        mesh.faces,
        attr_volume=pbr_voxel.feats,
        coords=pbr_voxel.coords[:, 1:],
        attr_layout={
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        },
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=int(mesh_config["grid_size"]),
        texture_size=int(texture_config["texture_size"]),
        decimation_target=int(texture_config.get("decimation_target", mesh_config["decimation_target"])),
        remesh=True,
        remesh_band=float(mesh_config["remesh_band"]),
        remesh_project=float(mesh_config["remesh_project"]),
        verbose=True,
    )
