# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from uniforge3d.models.vggt.heads.camera_head import CameraHead
from uniforge3d.models.vggt.heads.dpt_head import DPTHead
from uniforge3d.models.vggt.models.aggregator import Aggregator


class VGGT(nn.Module, PyTorchModelHubMixin):
    """Inference-only VGGT backbone with the camera and point heads we use."""

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
    ):
        super().__init__()
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
            if enable_point
            else None
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        aggregated_tokens, patch_start = self.aggregator(images)
        predictions: dict[str, torch.Tensor] = {}
        with torch.autocast(device_type=images.device.type, enabled=False):
            if self.camera_head is not None:
                poses = self.camera_head(aggregated_tokens)
                predictions["pose_enc"] = poses[-1]
                predictions["pose_enc_list"] = poses
            if self.point_head is not None:
                points, confidence = self.point_head(
                    aggregated_tokens,
                    images=images,
                    patch_start_idx=patch_start,
                )
                predictions["world_points"] = points
                predictions["world_points_conf"] = confidence
        predictions["images"] = images
        return predictions
