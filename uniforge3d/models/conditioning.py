"""Fuse cached DINO features with structure-conditioned VGGT features and select views."""

from __future__ import annotations

from typing import Any

import torch

from .vggt_routed_features import FrozenRoutedVGGTFeatureExtractor
from .view_router import GPUVoxelViewRouter


class StructuredLatentConditioner:
    """Share one frozen VGGT and the 32/64-grid routers across all stages."""

    def __init__(
        self,
        *,
        vggt_feature_extractor: dict[str, Any],
        voxel_routers: dict[int, dict[str, Any]],
    ):
        extractor_args = dict(vggt_feature_extractor)
        extractor_args["device"] = torch.device("cpu")
        self.frozen_vggt = FrozenRoutedVGGTFeatureExtractor(**extractor_args)
        self.routers = {
            int(grid): GPUVoxelViewRouter(**dict(router_config))
            for grid, router_config in voxel_routers.items()
        }

    def to(self, device: str | torch.device) -> "StructuredLatentConditioner":
        device = torch.device(device)
        frozen = self.frozen_vggt
        for name in (
            "aggregator",
            "shape_encoder",
            "backbone_shape_adapters",
            "point_head",
            "dpt_shape_adapter_32",
        ):
            module = getattr(frozen, name, None)
            if module is not None:
                module.to(device)
        frozen.device = device
        return self

    @torch.no_grad()
    def encode(
        self,
        adapter: torch.nn.Module,
        dino_features: torch.Tensor,
        vggt_images: torch.Tensor,
        structure_conditioned_coords: torch.Tensor,
        input_masks: torch.Tensor,
        *,
        latent_coords: torch.Tensor | None = None,
        router_grid: int | None = None,
        view_assignment: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Build CFG conditions for one case from [N,4] (batch,x,y,z) coords64.

        A view assignment can be reused from Shape1024; packed attention layouts
        are built separately by the denoiser for its exact sparse-row order.
        """
        packed = structure_conditioned_coords[:, 1:].to(
            device=dino_features.device, dtype=torch.int32
        ).contiguous()
        offsets = torch.tensor([0, len(packed)], device=packed.device, dtype=torch.long)
        dino_bank = dino_features.contiguous()
        cached, patch_start, points, confidence = self.frozen_vggt(
            vggt_images, packed, offsets, return_points=view_assignment is None
        )

        with torch.autocast(
            device_type=dino_features.device.type, dtype=torch.bfloat16,
        ):
            vggt_delta = adapter(cached, patch_start)
        patch_count = int(vggt_delta.shape[2])
        special_tokens = int(dino_bank.shape[2]) - patch_count
        if special_tokens != 5:
            raise RuntimeError(
                f"Expected five DINO special tokens, got dino={dino_bank.shape[2]} vggt={patch_count}"
            )
        dino_patch = dino_bank[:, :, special_tokens:]
        fused_bank = dino_bank.clone()
        fused_bank[:, :, special_tokens:] = dino_patch + vggt_delta.to(
            device=dino_patch.device, dtype=dino_patch.dtype
        )

        if view_assignment is None:
            router = self.routers[router_grid]
            target_coords = (
                latent_coords
                if latent_coords is not None
                else structure_conditioned_coords
            ).to(device=packed.device, dtype=torch.long)
            view_assignment = router(
                points,
                confidence,
                input_masks,
                target_coords=target_coords,
            )
        else:
            view_assignment = {name: value.to(fused_bank.device) for name, value in view_assignment.items()}
        positive = {"full_bank": fused_bank, **view_assignment}
        negative = {
            "full_bank": torch.zeros_like(fused_bank),
            "selected_view": torch.zeros_like(view_assignment["selected_view"]),
        }
        return {"cond": positive, "neg_cond": negative}
